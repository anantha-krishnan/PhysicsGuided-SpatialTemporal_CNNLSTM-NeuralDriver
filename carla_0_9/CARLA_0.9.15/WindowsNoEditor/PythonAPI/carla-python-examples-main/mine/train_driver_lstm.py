# train_driver_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import os
import copy
# --- CONFIGURATION ---
current_file_path = Path(os.path.abspath(__file__))
current_dir = current_file_path.parent
CSV_FILE = current_dir.parent / "Map_Layouts" / "lane_change_dataset.csv"
MODEL_SAVE_PATH = current_dir.parent / "Map_Layouts" / "lstm_driver.pth"
SCALER_SAVE_PATH = current_dir.parent / "Map_Layouts" / "scaler_lstm.pkl"
# Feature and Target Columns
feature_cols = ['speed_input', 'speed_error_input', 'cte_input', 'heading_error_input', 'future_cte_input', 'yaw_rate_input','lat_accel_input']
target_cols = ['steer_cmd', 'throttle_cmd', 'brake_cmd']
input_dim = len(feature_cols)
output_dim = 2 # Steer, Longitudinal (Throttle-Brake)

BATCH_SIZE = 64
EPOCHS = 100            # Set high, Early Stopping will cut it short
PATIENCE = 10           # Stop if no improvement for 10 epochs
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 30    # 1 second history
HIDDEN_SIZE = 64
NUM_LAYERS = 2

WHEELBASE = 2.875
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. UTILITY: EARLY STOPPING CLASS ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path=MODEL_SAVE_PATH):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        self.best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), self.path)
        print(f'Validation loss decreased ({self.best_loss:.6f}).  Saving model ...')

    def restore_best_weights(self, model):
        '''Restores the best weights calculated during training'''
        if self.best_model_wts:
            model.load_state_dict(self.best_model_wts)
            print("Restored best model weights.")

# --- 2. DATASET & PRE-PROCESSING ---
class SequenceDataset(Dataset):
    def __init__(self, sequences, targets, raw_inputs):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.raw_inputs = torch.tensor(raw_inputs, dtype=torch.float32)

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.targets[idx], self.raw_inputs[idx]

def create_sequences_from_df(df, seq_len):
    """
    Creates sequences ensuring NO mixing between episodes.
    """
    all_sequences = []
    all_targets = []
    all_raws = []
    
    
    # Group by episode to respect physics continuity
    episode_groups = df.groupby('episode_id')

    for ep_id, group in episode_groups:
        data = group[feature_cols].values
        targets = group[target_cols].values
        
        # Process Targets: [Steer, Throttle-Brake]
        steer = targets[:, 0]
        long_cmd = targets[:, 1] - targets[:, 2]
        processed_targets = np.column_stack((steer, long_cmd))
        
        num_samples = len(data) - seq_len
        if num_samples <= 0: continue

        for i in range(num_samples):
            # Input: t to t+10
            seq = data[i : i+seq_len]
            # Target: Action at t+10
            target = processed_targets[i + seq_len]
            # Raw: Used for physics loss at t+10
            raw = data[i + seq_len] 
            
            all_sequences.append(seq)
            all_targets.append(target)
            all_raws.append(raw)

    return np.array(all_sequences), np.array(all_targets), np.array(all_raws)

# --- 3. MODEL ARCHITECTURE ---
class LSTMDriver(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMDriver, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :] 
        return self.tanh(self.fc(last_out))

# --- 4. PHYSICS LOSS ---
class KinematicLSTMLoss(nn.Module):
    def __init__(self, scaler):
        super(KinematicLSTMLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.L = WHEELBASE
        
        
        self.scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32).to(device)
        self.scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32).to(device)

    def forward(self, predictions, targets, inputs_scaled):
        
        # --- 1. CLONING LOSS (Imitation) ---
        cloning_loss = self.mse(predictions, targets)
        
        # --- 2. PHYSICS PREPARATION ---
        # Unscale inputs to get real Speed (m/s)
        real_inputs = (inputs_scaled * self.scaler_scale) + self.scaler_mean
        speed_ms = real_inputs[:, 0] # Index 0 is Speed
        
        # Get Predictions
        pred_steer = predictions[:, 0]     # -1 to 1
        pred_long = predictions[:, 1]      # -1 (Brake) to 1 (Throttle)
        
        # --- 3. LATERAL PHYSICS (Steering Limit) ---
        MAX_STEER_RAD = 1.22
        steer_rad = pred_steer * MAX_STEER_RAD
        
        # Theoretical Lat Accel = v^2 / L * tan(delta)
        pred_a_lat = (speed_ms**2 / self.L) * torch.tan(steer_rad)
        
        # --- 4. LONGITUDINAL PHYSICS (Approximate) ---
        # Map output (-1 to 1) to roughly G-force
        # Braking is strong (~1.0g), Acceleration is weaker (~0.5g for average car)
        # We approximate: Long Accel ~= pred_long * 9.8
        pred_a_long = pred_long * 9.8
        
        # --- 5. FRICTION CIRCLE LOSS (The "Combined" Constraint) ---
        # Total Gs = sqrt(a_lat^2 + a_long^2)
        total_accel = torch.sqrt(pred_a_lat**2 + pred_a_long**2)
        
        # Limit: 9.0 m/s^2 (approx 0.9g)
        FRICTION_LIMIT = 9.0 
        
        # ReLU: Only penalize if we exceed the limit
        friction_violation = torch.relu(total_accel - FRICTION_LIMIT)
        
        physics_loss = torch.mean(friction_violation**2)
        
        # Combine: 
        # Strong penalty (0.5) because violating physics causes crashes
        return cloning_loss + (0.5 * physics_loss)

# --- 5. TRAINING LOOP ---
def train():
    print("Loading Data...")
    df = pd.read_csv(CSV_FILE)
    
    # A. SPLIT BY EPISODE (Crucial for Validating Generalization)
    unique_episodes = df['episode_id'].unique()
    train_eps, val_eps = train_test_split(unique_episodes, test_size=0.2, random_state=42)
    
    train_df = df[df['episode_id'].isin(train_eps)].copy()
    val_df = df[df['episode_id'].isin(val_eps)].copy()
    
    print(f"Train Episodes: {len(train_eps)} | Val Episodes: {len(val_eps)}")

    # B. FIT SCALER ON TRAINING DATA ONLY (Prevent Data Leakage)
    scaler = StandardScaler()
    
    # Fit on Train, Transform both
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    # C. CREATE SEQUENCES
    print("Generating Sequences...")
    X_train, y_train, raw_train = create_sequences_from_df(train_df, SEQUENCE_LENGTH)
    X_val, y_val, raw_val = create_sequences_from_df(val_df, SEQUENCE_LENGTH)
    
    # D. DATALOADERS
    train_ds = SequenceDataset(X_train, y_train, raw_train)
    val_ds = SequenceDataset(X_val, y_val, raw_val)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # E. MODEL SETUP
    model = LSTMDriver(input_dim=input_dim, hidden_dim=HIDDEN_SIZE, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = KinematicLSTMLoss(scaler)
    
    # Initialize Early Stopping
    early_stopper = EarlyStopping(patience=PATIENCE, path=MODEL_SAVE_PATH)
    
    print(f"Starting Training on {len(X_train)} samples...")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        for seqs, targets, raw_inputs in train_loader:
            seqs, targets, raw_inputs = seqs.to(device), targets.to(device), raw_inputs.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = loss_fn(outputs, targets, raw_inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATE ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, targets, raw_inputs in val_loader:
                seqs, targets, raw_inputs = seqs.to(device), targets.to(device), raw_inputs.to(device)
                outputs = model(seqs)
                loss = loss_fn(outputs, targets, raw_inputs)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.5f} | Val Loss={avg_val_loss:.5f}")
        
        # --- EARLY STOPPING CHECK ---
        early_stopper(avg_val_loss, model)
        
        if early_stopper.early_stop:
            print("Early Stopping triggered. Training halted.")
            break
            
    # Restore best model before exiting
    early_stopper.restore_best_weights(model)
    print("Model Training Complete.")

def get_model_architecture():
    from torchviz import make_dot
    from torchinfo import summary
    import os
    
    # 1. Re-initialize model
    model = LSTMDriver(input_dim=input_dim, hidden_dim=HIDDEN_SIZE, output_dim=output_dim).to(device)
    dummy_input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, input_dim).to(device)

    print("\n" + "="*30)
    print("     GENERATING HIGH-RES CHART")
    print("="*30)

    # 2. Run Forward Pass
    output = model(dummy_input)

    # 3. Create Graph
    # show_attrs=True: Shows the dimensions inside the boxes (Crucial)
    # show_saved=True: Shows memory cells
    dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    
    # --- SETTINGS FOR QUALITY ---
    dot.attr(rankdir='LR')       # Left-to-Right orientation
    dot.attr(dpi='300')          # 300 DPI (Print Quality)
    dot.attr(size='20,20')       # Max size in inches (prevents tiny cramping)
    dot.attr(overlap='false')    # Prevent boxes from covering each other
    dot.attr(fontsize='12')      # Readable font size
    
    # --- EXPORT 1: PDF (Best for Zooming) ---
    dot.format = 'pdf'
    pdf_path = current_dir.parent / "Map_Layouts" / "architecture_quality"
    try:
        dot.render(pdf_path)
        print(f"[SUCCESS] Saved Vector PDF to {pdf_path}.pdf")
    except Exception as e:
        print(f"[ERROR] Could not save PDF. Is Graphviz installed? {e}")

    # --- EXPORT 2: HIGH-RES PNG (Backup) ---
    dot.format = 'png'
    png_path = current_dir.parent / "Map_Layouts" / "architecture_quality"
    try:
        dot.render(png_path)
        print(f"[SUCCESS] Saved High-Res PNG to {png_path}.png")
    except Exception as e:
        print(f"[ERROR] Could not save PNG: {e}")

    # Check if files exist and are not empty
    if os.path.exists(f"{pdf_path}.pdf") and os.path.getsize(f"{pdf_path}.pdf") < 100:
        print("\n!!! WARNING: The generated file is empty (0kb).")
        print("This usually means 'Graphviz' is not installed on your system.")
        print("Please install it: https://graphviz.org/download/")
    
if __name__ == "__main__":
    get_model_architecture()
    train()
