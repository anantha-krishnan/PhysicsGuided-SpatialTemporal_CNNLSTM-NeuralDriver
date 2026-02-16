import carla
import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import time
from collections import deque
from pathlib import Path
import os
import pandas as pd
# --- CONFIGURATION ---
current_file_path = Path(os.path.abspath(__file__))
current_dir = current_file_path.parent
MODEL_SAVE_PATH = current_dir.parent / "Map_Layouts" / "lstm_driver.pth"
SCALER_SAVE_PATH = current_dir.parent / "Map_Layouts" / "scaler_lstm.pkl"
XODR_DATA = current_dir.parent / "Map_Layouts" / "flattesttrack.xodr"
MODEL_PATH = MODEL_SAVE_PATH
SCALER_PATH = SCALER_SAVE_PATH
SEQUENCE_LENGTH = 30    # Must match training
HIDDEN_SIZE = 64
INPUT_DIM = 4  # speed, cte, heading, future_cte
OUTPUT_DIM = 2 # steer, long

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
        self.scaler = joblib.load(SCALER_PATH)
        
        # The LSTM Memory Buffer (Rolling Window)
        # We initialize it with zeros, but will fill it quickly
        self.history_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
    def process(self, speed_ms, cte, heading_error, future_cte):
        # 1. Prepare Input Vector
        # MUST Match training order: [speed, cte, heading, future]
        raw_input_df = pd.DataFrame(
            [[speed_ms, cte, heading_error, future_cte]], 
            columns=['speed_input', 'cte_input', 'heading_error_input', 'future_cte_input']
        )
        
        # 2. Scale Input (Normalize)
        scaled_input = self.scaler.transform(raw_input_df) # Returns shape (1, 4)
        
        # 3. Update History Buffer
        # If buffer is empty (first frame), fill it with copies of current state
        if len(self.history_buffer) == 0:
            for _ in range(SEQUENCE_LENGTH):
                self.history_buffer.append(scaled_input[0])
        else:
            self.history_buffer.append(scaled_input[0])
            
        # 4. Convert to Tensor
        # Shape: (1, 10, 4) -> (Batch, Seq, Features)
        input_tensor = torch.tensor([list(self.history_buffer)], dtype=torch.float32).to(device)
        
        # 5. Inference
        with torch.no_grad():
            output = self.model(input_tensor).cpu().numpy()[0]
            
        # 6. Parse Output
        # Network Output: [Steer (-1 to 1), Long (-1 to 1)]
        net_steer = output[0]
        net_long = output[1]
        
        # Convert Net Steer to Carla Steer
        # During training we divided by 1.22 (70 deg). We multiply back.
        # But wait! CARLA ApplyControl expects -1.0 to 1.0. 
        # Our training label was also -1.0 to 1.0. 
        # So we can use output directly, just clipped for safety.
        steer_cmd = np.clip(net_steer, -1.0, 1.0)
        
        # Convert Long to Throttle/Brake
        if net_long > 0:
            throttle = np.clip(net_long, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(abs(net_long), 0.0, 1.0)
            
        return steer_cmd, throttle, brake

# --- 3. HELPER: GHOST PATH GENERATOR (SAME AS ORCHESTRATOR) ---
def generate_ghost_path(speed_kph, lc_length):
    # Generates a path from Lane -1 (y=-1.75) to Lane -2 (y=-5.25)
    speed_ms = speed_kph / 3.6
    points = []
    start_y = -1.75
    target_y = -5.25 
    run_up = 20.0  
    run_out = 50.0 
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
        points.append([global_x, current_y])
    return np.array(points)

def get_relative_errors(vehicle, path_points):
    v_trans = vehicle.get_transform()
    v_loc = v_trans.location
    
    dists = np.linalg.norm(path_points - np.array([v_loc.x, v_loc.y]), axis=1)
    min_idx = np.argmin(dists)
    closest_pt = path_points[min_idx]
    
    # CTE
    cte = v_loc.y - closest_pt[1]
    
    # Heading Error
    if min_idx + 1 < len(path_points):
        dx = path_points[min_idx+1][0] - path_points[min_idx][0]
        dy = path_points[min_idx+1][1] - path_points[min_idx][1]
    else:
        dx = path_points[min_idx][0] - path_points[min_idx-1][0]
        dy = path_points[min_idx][1] - path_points[min_idx-1][1]
    
    path_yaw = math.atan2(dy, dx)
    vehicle_yaw = math.radians(v_trans.rotation.yaw)
    he = vehicle_yaw - path_yaw
    while he > math.pi: he -= 2*math.pi
    while he < -math.pi: he += 2*math.pi
    
    # Future CTE
    vel = vehicle.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2)
    lookahead = max(5.0, speed * 1.0)
    
    look_idx = min_idx
    dist_accum = 0
    while look_idx < len(path_points)-1 and dist_accum < lookahead:
        dist_accum += 0.5
        look_idx += 1
    
    future_cte = v_loc.y - path_points[look_idx][1]
    
    return speed, cte, he, future_cte, min_idx

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
        # Speed: 40 km/h, Length: 40m
        print("Generating Scenario: 40 km/h, 40m Lane Change")
        ghost_path = generate_ghost_path(speed_kph=40, lc_length=40)
        print(f"Generated Ghost Path with {len(ghost_path)} points.")
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
            speed, cte, he, fut_cte, progress_idx = get_relative_errors(vehicle, ghost_path)
            
            # Check if finished
            if progress_idx >= len(ghost_path) - 5:
                print("Track Complete!")
                break
                
            # 3. AI Inference
            steer, throttle, brake = brain.process(speed, cte, he, fut_cte)
            
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