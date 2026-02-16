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

    feature_cols = ['speed_input', 'cte_input', 'heading_error_input', 'future_cte_input']
    target_cols = ['steer_cmd', 'throttle_cmd', 'brake_cmd']
    
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
        # Convert scaler params to tensors for GPU usage
        self.scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32).to(device)
        self.scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32).to(device)

    def forward(self, predictions, targets, inputs_scaled):
        # 1. Cloning Loss
        loss = self.mse(predictions, targets)
        
        # 2. Physics Check: Unscale the Speed
        # Speed is index 0 in the inputs
        real_inputs = (inputs_scaled * self.scaler_scale) + self.scaler_mean
        speed_ms = real_inputs[:, 0]
        
        # 3. Lat Accel Check
        pred_steer = predictions[:, 0]
        steer_rad = pred_steer * 1.22 # Max steer rad
        
        lat_accel = (speed_ms**2 / self.L) * torch.tan(steer_rad)
        
        # Penalize if > 0.8g (approx 8.0 m/s^2)
        violation = torch.relu(torch.abs(lat_accel) - 8.0)
        physics_penalty = torch.mean(violation**2)
        
        return loss + (0.1 * physics_penalty)

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
    feature_cols = ['speed_input', 'cte_input', 'heading_error_input', 'future_cte_input']
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
    model = LSTMDriver(input_dim=4, hidden_dim=HIDDEN_SIZE, output_dim=2).to(device)
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

if __name__ == "__main__":
    train()