# train_driver_all.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import os
import copy
import gc

# --- CONFIGURATION ---
current_file_path = Path(os.path.abspath(__file__))
current_dir = current_file_path.parent
CSV_FILE = current_dir.parent / "Map_Layouts" / "master_dataset.csv"
OUTPUT_DIR = current_dir.parent / "Map_Layouts" / "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WHEELBASE = 2.875
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. DEFINING THE ABLATION STUDY ---
# This list dictates exactly how the script will loop, filter data, and build the network.
EXPERIMENTS = [
    {"name": "M-Proposed-rec",       "data": "recovery_full", "wp": "wp_dyn", "arch": "dual",     "wp_dim": 32},
    {"name": "M-Proposed-rec_pris",  "data": "all",           "wp": "wp_dyn", "arch": "dual",     "wp_dim": 32},
    {"name": "M-Baseline-rec",       "data": "recovery_full", "wp": "wp_dyn", "arch": "baseline", "wp_dim": 32},
    {"name": "M-Pristine-pristine",  "data": "pristine_only", "wp": "wp_dyn", "arch": "dual",     "wp_dim": 32}, # Failure Model
    {"name": "M-PrimitivesOnly-rec", "data": "recovery_prim", "wp": "wp_dyn", "arch": "dual",     "wp_dim": 32},
    {"name": "M-NgramsOnly-rec",     "data": "recovery_ngrm", "wp": "wp_dyn", "arch": "dual",     "wp_dim": 32},
    {"name": "M-Symmetric-rec",      "data": "recovery_full", "wp": "wp_dyn", "arch": "dual",     "wp_dim": 64},
    {"name": "M-FH-5-rec",           "data": "recovery_full", "wp": "wp_5m",  "arch": "dual",     "wp_dim": 32},
    {"name": "M-FH-10-rec",          "data": "recovery_full", "wp": "wp_10m", "arch": "dual",     "wp_dim": 32},
    {"name": "M-FH-20-rec",          "data": "recovery_full", "wp": "wp_20m", "arch": "dual",     "wp_dim": 32},
    {"name": "M-FH-30-rec",          "data": "recovery_full", "wp": "wp_30m", "arch": "dual",     "wp_dim": 32}
]
# The 5 exact kinematic states
STATE_COLS = [
    'cte_input', 'heading_error_input', 'yaw_rate_input', 
    'future_path_curvature_input', 'future_heading_error_input'
]
# --- 2. UTILITIES ---
class EarlyStopping:
    def __init__(self, patience=10, path=""):
        self.patience = patience; self.path = path
        self.counter = 0; self.best_loss = None; self.early_stop = False; self.best_wts = None
    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - 0e-6:
            self.best_loss = val_loss
            self.best_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), self.path)
            self.counter = 0
            print(f'   -> Val loss decreased ({self.best_loss:.5f}). Saved.')
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.targets[idx]

def create_sequences_from_df(df, feature_cols, seq_len):
    all_seqs, all_targs = [], []
    for ep_id, group in df.groupby('episode_id'):
        data = group[feature_cols].values
        targets = group['steer_cmd'].values
        num_samples = len(data) - seq_len
        if num_samples <= 0: continue
        for i in range(num_samples):
            all_seqs.append(data[i : i+seq_len])
            all_targs.append(targets[i + seq_len])
    return np.array(all_seqs), np.array(all_targs)

# --- 3. ARCHITECTURES ---
class LSTM1DCNNDriver(nn.Module):
    """The Proposed Dual-Stream Spatial-Temporal Architecture"""
    def __init__(self, state_dim, num_waypoints, wp_bottleneck):
        super(LSTM1DCNNDriver, self).__init__()
        self.state_dim = state_dim
        self.num_waypoints = num_waypoints
        
        # Stream 1: Kinematics (64 features)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh())
        
        # Stream 2: Waypoints (32 or 64 bottleneck)
        self.wp_encoder = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1), nn.Tanh(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1), nn.Tanh(),
            nn.Flatten(),
            nn.Linear(16 * num_waypoints, wp_bottleneck), nn.Tanh()
        )
        
        # Fusion & Temporal
        self.lstm = nn.GRU(64 + wp_bottleneck, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        b, s, _ = x.size()
        state_x = x[:, :, :self.state_dim].contiguous().view(b * s, self.state_dim)
        wp_x = x[:, :, self.state_dim:].contiguous().view(b * s, self.num_waypoints, 2).permute(0, 2, 1)
        
        state_f = self.state_encoder(state_x)
        wp_f = self.wp_encoder(wp_x)
        
        combined = torch.cat((state_f, wp_f), dim=1).view(b, s, -1)
        out, _ = self.lstm(combined)
        return self.tanh(self.fc(out[:, -1, :]))

class BaselineLSTMDriver(nn.Module):
    """The Simple Baseline Architecture (No CNN, No Dual-Stream)"""
    def __init__(self, input_dim):
        super(BaselineLSTMDriver, self).__init__()
        # Immediately flat-maps all 25 inputs to 64 features
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh())
        self.lstm = nn.GRU(64, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        b, s, f = x.size()
        encoded = self.encoder(x.contiguous().view(b * s, f)).view(b, s, -1)
        out, _ = self.lstm(encoded)
        return self.tanh(self.fc(out[:, -1, :]))

# --- 4. EXPERIMENT ENGINE ---
def run_experiment(config, df_master, df_recovery, df_pristine_sampled):
    name = config['name']
    print(f"\n" + "="*50)
    print(f" STARTING EXPERIMENT: {name}")
    print("="*50)
    
    # 1. Generate specific waypoint column names for this run
    wp_cols = [f"{config['wp']}_{i}_{coord}" for i in range(10) for coord in ['x', 'y']]
    feature_cols = STATE_COLS + wp_cols
    
    # 2. Filter and Sample Data based on ablation criteria
    df = df_master.copy() # Start with the full 50/50 dataset

    if config['data'] == "pristine_only":
        # The FAILURE MODEL: Uses only 0-offset data
        df = df[df['dataset_group'] == 'pristine']
        print(f" [!] Training on 100% PRISTINE data (N={len(df)})")
        
    elif config['data'] == "recovery_full":
        # THE PROPOSED METHOD: 100% Recovery, Full Curriculum
        df = df[df['dataset_group'] == 'recovery']
        print(f" [!] Training on 100% RECOVERY data (Full Curriculum, N={len(df)})")
        
    elif config['data'] == "recovery_prim":
        # ABLATION: Primitives Only (No Chains)
        df = df[(df['dataset_group'] == 'recovery') & (~df['maneuver'].str.contains('-'))]
        print(f" [!] Training on 100% RECOVERY data (Primitives ONLY, N={len(df)})")
        
    elif config['data'] == "recovery_ngrm":
        # ABLATION: N-Grams Only (No Isolated Maneuvers)
        df = df[(df['dataset_group'] == 'recovery') & (df['maneuver'].str.contains('-'))]
        print(f" [!] Training on 100% RECOVERY data (N-Grams ONLY, N={len(df)})")
        
    elif config['data'] == "all":
        # Combine to create the final 80/20 training set
        df = pd.concat([df_recovery, df_pristine_sampled])
        print(f"   -> Dataset: 'All'. Subsampled to 80/20 ratio.")
        print(f"   -> Using {len(df_recovery)} recovery rows and {len(df_pristine_sampled)} pristine rows.")
                    
    # ... The rest of the function (Split & Scale, Model Init, Training Loop) remains identical ...
    # 3. Split & Scale
    unique_eps = df['episode_id'].unique()
    train_eps, val_eps = train_test_split(unique_eps, test_size=0.2, random_state=42)
    train_df = df[df['episode_id'].isin(train_eps)].copy()
    val_df = df[df['episode_id'].isin(val_eps)].copy()
    
    #scaler = RobustScaler()
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    
    # Save Scaler specifically for this experiment
    #np.savez(OUTPUT_DIR / f"scaler_{name}.npz", center=scaler.center_, scale=scaler.scale_, feature_names=feature_cols)
    np.savez(OUTPUT_DIR/ f"scaler_{name}.npz", mean=scaler.mean_, scale=scaler.scale_, feature_names=feature_cols)
    
    # 4. Create Sequences
    X_train, y_train = create_sequences_from_df(train_df, feature_cols, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences_from_df(val_df, feature_cols, SEQUENCE_LENGTH)
    
    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Initialize Model
    if config['arch'] == "dual":
        model = LSTM1DCNNDriver(state_dim=5, num_waypoints=10, wp_bottleneck=config['wp_dim']).to(device)
    else:
        model = BaselineLSTMDriver(input_dim=len(feature_cols)).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    stopper = EarlyStopping(patience=PATIENCE, path=OUTPUT_DIR / f"{name}.pth")
    
    def weighted_mse(pred, target):
        mse = nn.MSELoss()
        return mse(pred, target)
        return torch.mean((pred - target)**2)

    # 6. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = weighted_mse(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, targets in val_loader:
                outputs = model(seqs.to(device))
                val_loss += weighted_mse(outputs, targets.to(device)).item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Ep {epoch+1:03d} | Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f}")
        
        stopper(avg_val, model)
        if stopper.early_stop: 
            print("   -> Early Stopping Triggered.")
            break
            
    stopper.best_wts = copy.deepcopy(model.state_dict())
    torch.save(model.state_dict(), stopper.path)    
    # Clean up memory before next loop
    del model, train_loader, val_loader, X_train, y_train, X_val, y_val, train_df, val_df, df
    torch.cuda.empty_cache()
    gc.collect()
    print(f"--- FINISHED {name} ---")

def write_pt_csv(config):
    # ==============================================================================
    # EXPORT LOGIC
    # ==============================================================================
    name = config['name']
    print(f"\n--- EXPORTING FOR {name} ---")

    MODEL_SAVE_PATH = OUTPUT_DIR / f"{name}.pth"
    SCALER_SAVE_PATH = OUTPUT_DIR / f"scaler_{name}.npz"
    TORCHSCRIPT_MODEL_PATH = OUTPUT_DIR / f"{name}.pt"
    SCALER_EXPORT_PATH = OUTPUT_DIR / f"scaler_{name}.csv"

    if not MODEL_SAVE_PATH.is_file():
        print(f"Skipping export for {name}: Model file not found at {MODEL_SAVE_PATH}")
        return

    # --- Model Loading ---
    device = torch.device("cpu")
    
    # Determine feature columns and input dim
    wp_cols = [f"{config['wp']}_{i}_{coord}" for i in range(10) for coord in ['x', 'y']]
    feature_cols = STATE_COLS + wp_cols
    INPUT_DIM = len(feature_cols)

    print(f"Loading model: {name}")
    if config['arch'] == "dual":
        model = LSTM1DCNNDriver(state_dim=5, num_waypoints=10, wp_bottleneck=config['wp_dim']).to(device)
    else: # baseline
        model = BaselineLSTMDriver(input_dim=INPUT_DIM).to(device)
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    
    # --- Exporting the Model to TorchScript ---
    print("Exporting model to TorchScript...")
    dummy_input = torch.randn(1, SEQUENCE_LENGTH, INPUT_DIM, device=device)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
    print(f"TorchScript model saved to: {TORCHSCRIPT_MODEL_PATH}")
    
    # --- Exporting the Scaler Parameters ---
    print("Exporting scaler parameters...")
    scaler_params = np.load(SCALER_SAVE_PATH)
    center = scaler_params['center']
    scale = scaler_params['scale']
    
    scaler_data_to_save = np.vstack([center, scale])
    np.savetxt(SCALER_EXPORT_PATH, scaler_data_to_save, delimiter=',')
    print(f"Scaler parameters saved to: {SCALER_EXPORT_PATH}")
    print(f"--- EXPORT COMPLETE FOR {name} ---\n")


if __name__ == "__main__":
    print("Loading Master Dataset into Memory...")
    df_master = pd.read_csv(CSV_FILE)
    df = df_master.copy() # Start with the full 50/50 dataset
    # --- THIS IS THE NEW LOGIC FOR THE 80/20 SPLIT ---
    df_recovery = df[df['dataset_group'] == 'recovery']
    df_pristine = df[df['dataset_group'] == 'pristine']
    # We want the final mix to be 80% recovery, 20% pristine.
    # Calculate how many pristine samples make up 20% of the final dataset.
    num_pristine_to_keep = len(df_recovery) // 4  # (e.g., 663 recovery / 4 = ~165 pristine)
    
    # Randomly sample the pristine data
    df_pristine_sampled = df_pristine.sample(n=num_pristine_to_keep, random_state=42)
    print("loaded")
    print(f"Total Rows Loaded: {len(df_master)}")
    
    for config in EXPERIMENTS:
        run_experiment(config, df_master, df_recovery, df_pristine_sampled)
        #write_pt_csv(config)
        
    print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
