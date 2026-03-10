# train_driver_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import copy
# --- CONFIGURATION ---
current_file_path = Path(os.path.abspath(__file__))
current_dir = current_file_path.parent
CSV_FILE = current_dir.parent / "Map_Layouts" / "lane_change_dataset.csv"
MODEL_SAVE_PATH = current_dir.parent / "Map_Layouts" / "lstm_driver.pth"
SCALER_SAVE_PATH = current_dir.parent / "Map_Layouts" / "scaler_lstm.npz"
#feature_cols = ['speed_input', 'speed_error_input', 'cte_input', 'heading_error_input', 'future_cte_input', 'yaw_rate_input','lat_accel_input','future_path_curvature_input']
feature_cols = ['cte_input', 'heading_error_input', 'yaw_rate_input','future_path_curvature_input', 'future_heading_error_input',
                'wp_0_x','wp_0_y','wp_1_x','wp_1_y','wp_2_x','wp_2_y','wp_3_x','wp_3_y','wp_4_x','wp_4_y','wp_5_x','wp_5_y','wp_6_x','wp_6_y','wp_7_x','wp_7_y','wp_8_x','wp_8_y','wp_9_x','wp_9_y']
target_cols = ['steer_cmd']
input_dim = len(feature_cols)
output_dim = len(target_cols) # Steer, Longitudinal (Throttle-Brake)

BATCH_SIZE = 64
EPOCHS = 100            # Set high, Early Stopping will cut it short
PATIENCE = 10           # Stop if no improvement for 10 epochs
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 30    # 1 second history
HIDDEN_SIZE = 64
NUM_LAYERS = 2
MIN_IMPROVEMENT_DELTA = 1e-6 

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
        # Ensure targets are (Batch, 1)
        if self.targets.ndim == 1:
            self.targets = self.targets.unsqueeze(1)
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
        #long_cmd = targets[:, 1] - targets[:, 2]
        long_cmd = 0
        processed_targets = steer.reshape(-1, 1)
        
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
        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :] 
        return self.tanh(self.fc(last_out))
class LSTM1DCNNDriver(nn.Module):
    def __init__(self, state_dim, num_waypoints, hidden_dim, output_dim, num_layers=2):
        super(LSTM1DCNNDriver, self).__init__()
        self.state_dim = state_dim
        self.num_waypoints = num_waypoints
        # --- 1. STATE ENCODER (Current Kinematics) ---
        # Input: [CTE, Heading_Err, Yaw_Rate, Future_Curv, Future_Heading_Err]
        # We expand 5 inputs -> 64 features using Tanh to preserve -1 to 1 symmetry.
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )
        # --- 2. WAYPOINT ENCODER (1D Spatial CNN Future Path) ---
        # Input: 10 waypoints (x,y) = 20 features
        # Goal: Extract geometry (Curves, S-Turns) but compress to 16 features
        self.wp_encoder = nn.Sequential(
            # Layer 1: Curvature Detector (Window of 3 points)
            # Input: (Batch*Seq, 2, 10) -> Output: (Batch*Seq, 8, 10)
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1), # 10 -> 10, 2->8
            nn.Tanh(),
            # Layer 2: S-Turn Detector (Window of 3 points)
            # Input: (Batch*Seq, 8, 10) -> Output: (Batch*Seq, 16, 10)
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1), # 10 -> 10, 8->16
            nn.Tanh(),
            # effectively captures local geometric patterns in the waypoints while maintaining the sequence length
            # it now has 5 point window curvature similar to akima spline
            # Layer 3: Compression (The "1/3rd Rule")
            nn.Flatten(), # (Batch*Seq, 16*10) = (Batch*Seq, 160)
            nn.Linear(16 * num_waypoints, 32), # 160 -> 32
            nn.Tanh()
        )
        # --- 3. FUSION & DECODER ---
        # Combine State (64) + Waypoint (32) = 96 features
        # LSTM Decoder: 96 -> Hidden -> 32
        # Combined Dimension = 64 (State) + 32 (Map) = 96
        combined_dim = 64 + 32
        self.lstm = nn.GRU(combined_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # x shape: (Batch, Sequence_Length, Total_Features)
        batch_size, seq_len, total_features = x.size()
        # --- 1. SPLIT STATE & WAYPOINTS ---
        state_x  = x[:, :, :self.state_dim] # (Batch, Seq, State_Dim)
        wp_x  = x[:, :, self.state_dim:] # (Batch, Seq, Waypoint_Dim)
        # --- B. The "Batch*Seq" Folding ---
        # Merge Batch and Seq so CNN/Linear layers treat every timestep as an independent sample
        state_x = state_x.contiguous().view(batch_size * seq_len, self.state_dim)
        # Prepare Waypoints for Conv1d (Needs Channels in middle)
        # 1. Reshape to (Batch*Seq, 10 points, 2 coords)
        wp_x = wp_x.contiguous().view(batch_size * seq_len, self.num_waypoints, 2)
        # 2. Permute to (Batch*Seq, 2 coords, 10 points)
        wp_x = wp_x.permute(0, 2, 1)
        
        # --- C. Extract Features ---
        state_features = self.state_encoder(state_x)    # Shape: (Batch*Seq, 64)
        wp_features = self.wp_encoder(wp_x)             # Shape: (Batch*Seq, 16)
        
        # --- D. Feature Fusion ---
        # Concatenate side-by-side: [State (64) | Map (16)]
        combined_features = torch.cat((state_features, wp_features), dim=1) # Shape: (Batch*Seq, 80)
        
        # --- E. Restore Sequence Dimension ---
        # Unfold back to (Batch, Seq, 80) so GRU sees the time history
        combined_features = combined_features.view(batch_size, seq_len, -1)
        
        # --- F. Temporal Processing ---
        out, _ = self.lstm(combined_features)
        
        # Take the output of the LAST time step
        last_out = out[:, -1, :] 
        
        # Generate Steering Command
        return self.tanh(self.fc(last_out))


# --- 4. PHYSICS LOSS ---
class KinematicLSTMLoss(nn.Module):
    def __init__(self, scaler):
        super(KinematicLSTMLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.L = WHEELBASE
        
        
        #self.scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32).to(device)
        self.scaler_center = torch.tensor(scaler.center_, dtype=torch.float32).to(device)
        self.scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32).to(device)
        

    def forward(self, predictions, targets, inputs_scaled):
        
        # --- 1. CLONING LOSS (Imitation) ---
        #cloning_loss = self.mse(predictions, targets)
        base_loss = (predictions-targets)**2
        weights=1+(5*torch.abs(targets))
        weighted_loss = torch.mean(base_loss * weights)
        # --- 2. PHYSICS PREPARATION ---
        # Unscale inputs to get real Speed (m/s)
        #real_inputs = (inputs_scaled * self.scaler_scale) + self.scaler_mean
        #speed_ms = real_inputs[:, 0] # Index 0 is Speed
        
        # Get Predictions
        #pred_steer = predictions[:, 0]     # -1 to 1
        #pred_long = predictions[:, 1]      # -1 (Brake) to 1 (Throttle)
        
        # --- 3. LATERAL PHYSICS (Steering Limit) ---
        #MAX_STEER_RAD = 1.22
        #steer_rad = pred_steer * MAX_STEER_RAD
        
        # Theoretical Lat Accel = v^2 / L * tan(delta)
        #pred_a_lat = (speed_ms**2 / self.L) * torch.tan(steer_rad)
        
        # --- 4. LONGITUDINAL PHYSICS (Approximate) ---
        # Map output (-1 to 1) to roughly G-force
        # Braking is strong (~1.0g), Acceleration is weaker (~0.5g for average car)
        # We approximate: Long Accel ~= pred_long * 9.8
        #pred_a_long = pred_long * 9.8
        
        # --- 5. FRICTION CIRCLE LOSS (The "Combined" Constraint) ---
        # Total Gs = sqrt(a_lat^2 + a_long^2)
        #total_accel = torch.sqrt(pred_a_lat**2 + pred_a_long**2)
        
        # Limit: 9.0 m/s^2 (approx 0.9g)
        #FRICTION_LIMIT = 9.0 
        
        # ReLU: Only penalize if we exceed the limit
        #friction_violation = torch.relu(total_accel - FRICTION_LIMIT)
        
        #physics_loss = torch.mean(friction_violation**2)
        
        # Combine: 
        # Strong penalty (0.5) because violating physics causes crashes
        #return cloning_loss + (0.0 * physics_loss)
        return weighted_loss

# --- 5. TRAINING LOOP ---
def train():
    print("Loading Data...")
    df = pd.read_csv(CSV_FILE)
    # drop columns we don't need
    df = df[feature_cols + target_cols + ['episode_id']]
    # A. SPLIT BY EPISODE (Crucial for Validating Generalization)
    unique_episodes = df['episode_id'].unique()
    train_eps, val_eps = train_test_split(unique_episodes, test_size=0.2, random_state=42)
    
    train_df = df[df['episode_id'].isin(train_eps)].copy()
    val_df = df[df['episode_id'].isin(val_eps)].copy()
    
    print(f"Train Episodes: {len(train_eps)} | Val Episodes: {len(val_eps)}")

    # B. FIT SCALER ON TRAINING DATA ONLY (Prevent Data Leakage)
    scaler = RobustScaler()
    
    # Fit on Train, Transform both
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    feature_names = train_df[feature_cols].columns.to_list()

    #np.savez(SCALER_SAVE_PATH, mean=scaler.mean_, scale=scaler.scale_, feature_names=feature_names)
    np.savez(SCALER_SAVE_PATH, center=scaler.center_, scale=scaler.scale_, feature_names=feature_names)
    
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
    #model = LSTMDriver(input_dim=input_dim, hidden_dim=HIDDEN_SIZE, output_dim=output_dim).to(device)
    model = LSTM1DCNNDriver(
        state_dim=5,         # 5
        num_waypoints=10, # 10
        hidden_dim=HIDDEN_SIZE,      # 64
        output_dim=output_dim        # 1
    ).to(device)
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
    #model = LSTMDriver(input_dim=input_dim, hidden_dim=HIDDEN_SIZE, output_dim=output_dim).to(device)
    model = LSTM1DCNNDriver(
        state_dim=5,         # 5
        num_waypoints=10, # 10
        hidden_dim=HIDDEN_SIZE,      # 64
        output_dim=output_dim        # 1
    ).to(device)
    dummy_input = torch.randn(1, 2, input_dim).to(device)

    print("\n" + "="*30)
    print("     GENERATING HIGH-RES CHART")
    print("="*30)

    # 2. Run Forward Pass
    output = model(dummy_input)

    # 3. Create Graph
    # show_attrs=True: Shows the dimensions inside the boxes (Crucial)
    # show_saved=True: Shows memory cells
    dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
    
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
def get_model_architecture_simple():
    from torchview import draw_graph
    import os
    
    save_dir = current_dir.parent / "Map_Layouts"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Re-initialize model
    model = LSTM1DCNNDriver(
        state_dim=5,         
        num_waypoints=10, 
        hidden_dim=HIDDEN_SIZE,      
        output_dim=output_dim        
    ).to(device)

    print("\n" + "="*30)
    print("     GENERATING CLEAN ARCHITECTURE CHART")
    print("="*30)

    # 2. Draw Graph using torchview
    # We use a small batch/seq size just to trace the layers
    model_graph = draw_graph(
        model, 
        input_size=(1, 2, input_dim), # (Batch, Seq, Features)
        device=device,
        graph_name="LSTM1DCNNDriver",
        expand_nested=True,        # Expands nn.Sequential blocks
        save_graph=True,           # Will save to file
        directory=str(save_dir),   # Save location
        filename="architecture_clean" # Output name
    )
    
    print(f"[SUCCESS] Saved clean architecture chart to {save_dir}/architecture_clean.png")
def get_model_architecture_onnx():
    import os
    
    save_dir = current_dir.parent / "Map_Layouts"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Re-initialize model
    model = LSTM1DCNNDriver(
        state_dim=5,         
        num_waypoints=10, 
        hidden_dim=HIDDEN_SIZE,      
        output_dim=output_dim        
    ).to(device)

    dummy_input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, input_dim).to(device)
    onnx_path = save_dir / "LSTM1DCNNDriver.onnx"

    print("\n" + "="*30)
    print("     EXPORTING ONNX FOR NETRON")
    print("="*30)

    # Export to ONNX format
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=11,          # Standard ONNX opset
        do_constant_folding=True,  # Optimizes graph for viewing
        input_names=['input_KS_Path'], 
        output_names=['steer_command']
    )
    
    print(f"[SUCCESS] Saved ONNX model to {onnx_path}")
    print(">>> Next Step: Drag and drop this file into https://netron.app to see the beautiful architecture diagram! <<<")
if __name__ == "__main__":
    get_model_architecture_onnx()
    #train()
