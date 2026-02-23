# neural_driver.py
import carla
import torch
import torch.nn as nn
import numpy as np
import math
from collections import deque
from pathlib import Path
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utility_fncs_train_inference import ControllerUtils
# --- CONFIGURATION ---
current_file_path = Path(os.path.abspath(__file__))
current_dir = current_file_path.parent
MODEL_SAVE_PATH = current_dir.parent / "Map_Layouts" / "lstm_driver.pth"
SCALER_SAVE_PATH = current_dir.parent / "Map_Layouts" / "scaler_lstm.npz"
XODR_DATA = current_dir.parent / "Map_Layouts" / "flattesttrack.xodr"
MODEL_PATH = MODEL_SAVE_PATH
SCALER_PATH = SCALER_SAVE_PATH
SEQUENCE_LENGTH = 30    # Must match training
HIDDEN_SIZE = 64
INPUT_DIM = 8 # speed, speed_error, cte, heading, future_cte, yaw_rate, lat_accel, future_path_curvature
OUTPUT_DIM = 2 # steer, long
last_closest_idx = 0
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. MODEL DEFINITION (MUST MATCH TRAINING SCRIPT) ---
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

# --- 2. THE NEURAL CONTROLLER CLASS ---
class NeuralController:
    def __init__(self):
        print(f"Loading Model from {MODEL_PATH}...")
        self.model = LSTMDriver(INPUT_DIM, HIDDEN_SIZE, OUTPUT_DIM).to(device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        self.model.eval() # Set to Inference Mode
        
        print(f"Loading Scaler from {SCALER_PATH}...")
        # Create a new, empty scaler object
        self.scaler = StandardScaler() 
        # Load the saved parameters
        scaler_params = np.load(SCALER_SAVE_PATH)
        # Manually set the mean and scale of the new object
        self.scaler.mean_ = scaler_params['mean']
        self.scaler.scale_ = scaler_params['scale']
        self.scaler.feature_names_in_ = scaler_params['feature_names']
        print("Scaler reconstructed successfully.")
        
        # The LSTM Memory Buffer (Rolling Window)
        # We initialize it with zeros, but will fill it quickly
        self.history_buffer = deque(maxlen=SEQUENCE_LENGTH)        
    
    def process(self, speed_ms, speed_error, cte, heading_error, future_cte, yaw_rate, lat_accel, future_path_curvature):
        
        # --- STEP 1: CLIP THE RAW INPUTS FIRST ---
        # This ensures the scaler only sees values within the expected range.
        #cte_clipped = np.clip(cte, -3.0, 3.0)
        #future_cte_clipped = np.clip(future_cte, -3.0, 3.0)
        #heading_error_clipped = np.clip(heading_error, -1.5, 1.5) # ~85 degrees
        #speed_error_clipped = np.clip(speed_error, -10.0, 10.0)
        #speed_error_clipped = speed_error  # We can choose to not clip speed error if we want the model to react strongly to large errors. Depends on training data distribution.
        # --- STEP 2: CREATE THE DATAFRAME WITH THE CLIPPED VALUES ---
        raw_input_df = pd.DataFrame(
            [[speed_ms, speed_error, cte, heading_error, future_cte, yaw_rate, lat_accel, future_path_curvature]], 
            columns=['speed_input', 'speed_error_input', 'cte_input', 'heading_error_input', 'future_cte_input', 'yaw_rate_input', 'lat_accel_input','future_path_curvature_input']
        )
        
        # --- STEP 3: SCALE THE (NOW SAFE) INPUT ---
        scaled_input = self.scaler.transform(raw_input_df)

        # --- THE REST OF THE FUNCTION REMAINS THE SAME ---
        if len(self.history_buffer) == 0:
            for _ in range(SEQUENCE_LENGTH):
                self.history_buffer.append(scaled_input[0])
        else:
            self.history_buffer.append(scaled_input[0])
        
        history_np = np.array(self.history_buffer)    
        input_tensor = torch.tensor(history_np, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = self.model(input_tensor).cpu().numpy()[0]
            
        net_steer = output[0]
        net_long = output[1]
        
        steer_cmd = np.clip(net_steer, -1.0, 1.0)
        
        if net_long > 0:
            throttle = np.clip(net_long, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(abs(net_long), 0.0, 1.0)
            
        return steer_cmd, throttle, brake

# --- 3. HELPER: GHOST PATH GENERATOR (SAME AS ORCHESTRATOR) ---
def generate_ghost_path(speed_kph, lc_length, start_y=-1.75, target_y=3.25, run_out=50.0, run_up=30.0):
    # Generates a path from Lane -1 (y=-1.75) to Lane -2 (y=-5.25)
    speed_ms = speed_kph / 3.6
    points = []
    total_dist = run_up + lc_length + run_out
    start_x_offset = 10.0 
    resolution = 0.5
    
    n_points = int(total_dist / resolution)
    
    for i in range(n_points):
        x_dist = i * resolution
        global_x = start_x_offset + x_dist
        current_y = start_y
        
        if x_dist < run_up:
            current_y = start_y
        elif x_dist < (run_up + lc_length):
            p = (x_dist - run_up) / lc_length
            factor = (1 - math.cos(p * math.pi)) / 2.0
            current_y = start_y + (target_y - start_y) * factor
        else:
            current_y = target_y
        points.append([global_x, current_y, 0, speed_ms])
    return np.array(points)

# --- 4. MAIN EXECUTION ---
def main():
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.generate_opendrive_world(XODR_DATA.read_text())
    # Sync Mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.033  # ~30 FPS
    world.apply_settings(settings)
    
    bp = world.get_blueprint_library().filter('model3')[0]
    
    # Initialize Neural Brain
    brain = NeuralController()
    try:
        # TEST SCENARIO: High Speed Lane Change
        # Speed: 50 km/h, Length: 40m
        cruise_speed_kph = 60
        lc_length = 60
        start_y = -1.75
        target_y = -5.25 
        run_up = 30.0  
        run_out = 50.0 
        print(f"Generating Scenario: {cruise_speed_kph} km/h, {lc_length}m Lane Change")
        ghost_path_speed = generate_ghost_path(speed_kph=cruise_speed_kph, lc_length=lc_length, start_y=start_y, target_y=target_y, run_up=run_up, run_out=run_out)
        error_calc = ControllerUtils(data=ghost_path_speed, lookahead_dist=25.0)
        # ghost_path=ghost_path_speed[:, :2]  # Extract only (x, y) for error calculations
        print(f"Generated Ghost Path with {len(ghost_path_speed)} points.")
        # Spawn
        start_pose = carla.Transform(carla.Location(x=10.0, y=-1.75, z=0.5), carla.Rotation(yaw=0.0))
        vehicle = world.spawn_actor(bp, start_pose)
        
        # Spectator
        spectator = world.get_spectator()
        v_trans = vehicle.get_transform()
        fwd_vec = v_trans.get_forward_vector()
        cam_loc = v_trans.location - carla.Location(x=2.0) + carla.Location(z=5.0)
        
        # Rotate camera to look slightly down (-15 degrees)
        cam_rot = v_trans.rotation
        cam_rot.pitch = -15.0
        
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))
        print("Engaging Neural Driver...")
        
        while True:
            # 1. Physics Step
            world.tick()
            
            # 2. Get State & Errors
            # speed, speed_error, cte, he, fut_cte, progress_idx = get_relative_errors(vehicle, ghost_path_speed)
            cte, he, fut_cte, speed, speed_error, progress_idx, future_path_curvature = error_calc.calculate_relative_errors(vehicle)
            # CARLA angular_velocity is in Degrees/s. Convert to Radians/s for training consistency.
            ang_vel_deg = vehicle.get_angular_velocity()
            yaw_rate = math.radians(ang_vel_deg.z)
            # B. Lateral Acceleration (IMU)
            # We use the Centripetal formula: a_lat = velocity * yaw_rate
            # This is cleaner than transforming the noisy IMU vector
            lat_accel = speed * yaw_rate
            # Check if finished
            if progress_idx >= len(ghost_path_speed) - 5:
                print("Track Complete!")
                break
                
            # 3. AI Inference
            steer, throttle, brake = brain.process(speed, speed_error, cte, he, fut_cte, yaw_rate, lat_accel, future_path_curvature)
            if cruise_speed_kph/3.6 > 1.0 and speed < 0.5 and throttle < 0.1:
                print(">>> WATCHDOG TRIGGERED: Kicking car forward")
                throttle = 0.3 # Force a gentle launch
                brake = 0.0
            # 4. Apply Control
            vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake)))
            
            # 5. Camera Follow
            #v_trans = vehicle.get_transform()
            #fwd_vec = v_trans.get_forward_vector()
            #cam_loc = v_trans.location - (fwd_vec * 10.0) + carla.Location(z=5.0)
            #
            ## Rotate camera to look slightly down (-15 degrees)
            #cam_rot = v_trans.rotation
            #cam_rot.pitch = -15.0
            #
            #spectator.set_transform(carla.Transform(cam_loc, cam_rot))
            
            # Debug Print
            print(f"progress_idx: {progress_idx} | Err: {cte:.2f}m | Fut: {fut_cte:.2f}m | Steer: {steer:.2f} | Thr: {throttle:.2f}")

    finally:
        vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Simulation Ended.")

if __name__ == "__main__":
    main()