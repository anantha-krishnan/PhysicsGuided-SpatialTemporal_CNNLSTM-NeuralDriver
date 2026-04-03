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
import random
from sklearn.preprocessing import StandardScaler, RobustScaler
from utility_fncs_train_inference import ControllerUtils
from minimap import PygameVisualizer
from controller import PathFollower
from path_generators import (
    generate_straight_path,
    generate_90_degree_turn_path,
    generate_hairpin_turn_path,
    generate_s_curve_path,
    generate_lane_change_path,
    generate_chicane_path
)
from train_driver_all import LSTM1DCNNDriver
#from train_driver_lstm import LSTM1DCNNDriver, feature_cols
import queue  
import cv2    
import pygame 
# --- CONFIGURATION ---
current_file_path = Path(os.path.abspath(__file__))
current_dir = current_file_path.parent
MODEL_SAVE_PATH = current_dir.parent / "Map_Layouts" / "M-Proposed-rec_nowe_stdsca.pth" #"lstm_driver_lstm_cnn.pth" #"M-Proposed-rec_noweightedloss.pth"
SCALER_SAVE_PATH = current_dir.parent / "Map_Layouts" / "scaler_M-Proposed-rec_nowe_stdsca.npz" #"scaler_lstm_lstm_cnn.npz" #"scaler_M-Proposed-rec_noweightedloss.npz"
XODR_DATA = current_dir.parent / "Map_Layouts" / "flattesttrack.xodr"
csv_path = current_dir.parent / "Map_Layouts" / "new_waypoints_Processed_town10HD_2.txt"
MODEL_PATH = MODEL_SAVE_PATH
SCALER_PATH = SCALER_SAVE_PATH
SEQUENCE_LENGTH = 30    # Must match training
HIDDEN_SIZE = 64
INPUT_DIM = 5 # cte, heading_error, yaw_rate, future_path_curvature, future_heading_error, future_waypoints (10 waypoints with x and y)
OUTPUT_DIM = 1 # steer
last_closest_idx = 0
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RECORD_VIDEO = False
OUTPUT_VIDEO_NAME = current_dir.parent / "Map_Layouts" /"town10HD__lane_follower_test3.mp4"
LANE_FOLLOWER = False
MAIN_WIDTH = 2560#2560#1920#1280
MAIN_HEIGHT = 1440#1440#1080#720

# Size: 30% of screen width
HUD_SCALE = 0.30 

# POSITION (0.0 to 1.0)
# (0.5, 0.5) is Center. (0.85, 0.85) is Bottom-Right. (0.15, 0.15) is Top-Left.
HUD_CENTER_X = 0.85 
HUD_CENTER_Y = 0.80 

# --- 2. THE NEURAL CONTROLLER CLASS ---
class NeuralController:
    def __init__(self):
        print(f"Loading Model from {MODEL_PATH}...")
        #self.model = LSTMDriver(INPUT_DIM, HIDDEN_SIZE, OUTPUT_DIM).to(device)
        self.model = LSTM1DCNNDriver(5,10,32).to(device)
        #self.model = LSTM1DCNNDriver(5,10, HIDDEN_SIZE, OUTPUT_DIM).to(device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        self.model.eval() # Set to Inference Mode
        
        print(f"Loading Scaler from {SCALER_PATH}...")
        # Create a new, empty scaler object
        self.scaler = StandardScaler() 
        #self.scaler = RobustScaler()
        # Load the saved parameters
        scaler_params = np.load(SCALER_SAVE_PATH)
        # Manually set the mean and scale of the new object
        self.scaler.mean_ = scaler_params['mean']
        #self.scaler.center_ = scaler_params['center']
        self.scaler.scale_ = scaler_params['scale']
        self.scaler.feature_names_in_ = scaler_params['feature_names']
        print("Scaler reconstructed successfully.")
        
        # The LSTM Memory Buffer (Rolling Window)
        # We initialize it with zeros, but will fill it quickly
        self.history_buffer = deque(maxlen=SEQUENCE_LENGTH)
    def physics_steering_filter(self, raw_nn_steer_norm, speed_ms):
        """
        Acts as the "Secondary Network". 
        Takes the raw -1 to 1 NN steer and clamps it dynamically based on velocity.
        """
        WHEELBASE = 2.875 # Meters (Standard CARLA sedan)
        MAX_G_FORCE = 0.6 # The maximum lateral G's before the car spins out
        GRAVITY = 9.81
        MAX_STEER_ANGLE_RAD = 1.22 # ~70 degrees max steering lock
        # 1. Convert normalized NN steer (-1 to 1) to actual Radians
        raw_steer_rad = raw_nn_steer_norm * MAX_STEER_ANGLE_RAD
        
        # 2. If moving very slow, no physics limit is needed
        if speed_ms < 20.0: # If we're below 72 km/h, allow full steering range. This lets the car maneuver in tight spaces and parking scenarios without being overly constrained.
            return raw_nn_steer_norm
            
        # 3. Calculate the absolute MAXIMUM steering angle allowed at this exact speed
        # derived from: a_y_max = (v^2 * tan(delta)) / L
        # We solve for delta: delta_max = atan((a_y_max * L) / v^2)
        max_accel_ms2 = MAX_G_FORCE * GRAVITY
        max_safe_steer_rad = math.atan((max_accel_ms2 * WHEELBASE) / (speed_ms**2))
        
        # 4. Clamp the Neural Network's command
        # If the NN asks for 0.5 rads, but physics says max is 0.1 rads, clamp it to 0.1
        safe_steer_rad = max(-max_safe_steer_rad, min(raw_steer_rad, max_safe_steer_rad))
        
        # 5. Convert back to -1 to 1 for CARLA control
        safe_steer_norm = safe_steer_rad / MAX_STEER_ANGLE_RAD
        
        return safe_steer_norm
    def process(self, speed_ms, speed_error, cte, heading_error, future_cte, yaw_rate, lat_accel, future_path_curvature, fut_yaw, fut_waypoints):
        
        # --- STEP 1: CLIP THE RAW INPUTS FIRST ---
        # This ensures the scaler only sees values within the expected range.
        #cte_clipped = np.clip(cte, -3.0, 3.0)
        #future_cte_clipped = np.clip(future_cte, -3.0, 3.0)
        #heading_error_clipped = np.clip(heading_error, -1.5, 1.5) # ~85 degrees
        #speed_error_clipped = np.clip(speed_error, -10.0, 10.0)
        #speed_error_clipped = speed_error  # We can choose to not clip speed error if we want the model to react strongly to large errors. Depends on training data distribution.
        # --- STEP 2: CREATE THE DATAFRAME WITH THE CLIPPED VALUES ---
        raw_input_df = pd.DataFrame(
            [[cte, heading_error, yaw_rate, future_path_curvature, fut_yaw]+fut_waypoints], 
            #columns=['speed_input', 'speed_error_input', 'cte_input', 'heading_error_input', 'future_cte_input', 'yaw_rate_input', 'lat_accel_input','future_path_curvature_input']
            #columns=['cte_input', 'heading_error_input', 'yaw_rate_input','future_path_curvature_input', 'future_heading_error_input']
            columns = ['cte_input', 'heading_error_input', 'yaw_rate_input','future_path_curvature_input', 'future_heading_error_input',
                'wp_dyn_0_x','wp_dyn_0_y','wp_dyn_1_x','wp_dyn_1_y','wp_dyn_2_x','wp_dyn_2_y','wp_dyn_3_x','wp_dyn_3_y','wp_dyn_4_x','wp_dyn_4_y','wp_dyn_5_x','wp_dyn_5_y','wp_dyn_6_x','wp_dyn_6_y','wp_dyn_7_x','wp_dyn_7_y','wp_dyn_8_x','wp_dyn_8_y','wp_dyn_9_x','wp_dyn_9_y']
            #columns = feature_cols
        )
        #raw_input_df = pd.DataFrame(
        #    [[cte, heading_error, yaw_rate, future_path_curvature, fut_yaw]], 
        #    #columns=['speed_input', 'speed_error_input', 'cte_input', 'heading_error_input', 'future_cte_input', 'yaw_rate_input', 'lat_accel_input','future_path_curvature_input']
        #    #columns=['cte_input', 'heading_error_input', 'yaw_rate_input','future_path_curvature_input', 'future_heading_error_input']
        #    columns = ['cte_input', 'heading_error_input', 'yaw_rate_input','future_path_curvature_input', 'future_heading_error_input',
        #        ]
        #)
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
        #net_long = output[1]
        
        steer_cmd = np.clip(net_steer, -1.0, 1.0)
        #steer_cmd = self.physics_steering_filter(steer_cmd, speed_ms)
        #if net_long > 0:
        #    throttle = np.clip(net_long, 0.0, 1.0)
        #    brake = 0.0
        #else:
        #    throttle = 0.0
        #    brake = np.clip(abs(net_long), 0.0, 1.0)
        
        
        return steer_cmd, 0.35, 0.0 # steer, throttle, brake

import carla
import numpy as np
import pandas as pd
import random
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------
# Utility: Closest waypoint
# ---------------------------------------------------------------------
def get_closest_waypoint(carla_map, location):
    return carla_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )


# ---------------------------------------------------------------------
# Turn classification
# ---------------------------------------------------------------------
def classify_turn(prev_wp, next_wp, threshold_deg=10.0):
    yaw1 = prev_wp.transform.rotation.yaw
    yaw2 = next_wp.transform.rotation.yaw

    dyaw = (yaw2 - yaw1 + 180) % 360 - 180  # [-180, 180]

    if dyaw > threshold_deg:
        return "left"
    elif dyaw < -threshold_deg:
        return "right"
    else:
        return "straight"


# ---------------------------------------------------------------------
# Curvature estimation (deg / meter)
# ---------------------------------------------------------------------
def estimate_curvature(wp_prev, wp_curr):
    yaw1 = wp_prev.transform.rotation.yaw
    yaw2 = wp_curr.transform.rotation.yaw

    dyaw = (yaw2 - yaw1 + 180) % 360 - 180
    dist = wp_prev.transform.location.distance(
        wp_curr.transform.location
    )

    if dist < 1e-3:
        return 0.0

    return abs(dyaw) / dist


# ---------------------------------------------------------------------
# Speed adjustment based on curvature
# ---------------------------------------------------------------------
def speed_from_curvature(
    curvature,
    cruise_speed,
    min_speed=4.0,
    k=0.08
):
    """
    curvature : deg/m
    k         : sensitivity factor
    """
    speed = cruise_speed / (1.0 + k * curvature)
    return max(min_speed, speed)


def choose_next_waypoint(current_wp, next_wps, turn_history):
    # No choice → just follow
    if len(next_wps) == 1:
        return next_wps[0], turn_history

    # Classify all options
    classified = []
    for wp in next_wps:
        turn = classify_turn(current_wp, wp)
        classified.append((wp, turn))

    # If no previous turn, pick any non-straight first
    if not turn_history:
        non_straight = [(wp, t) for wp, t in classified if t != "straight"]
        if non_straight:
            wp, turn = random.choice(non_straight)
        else:
            wp, turn = random.choice(classified)

        #turn_history.append(turn)
        return wp, turn

    # Force opposite of last turn
    last_turn = turn_history

    if last_turn == "left":
        desired = "right"
    else:
        desired = "left"

    if desired:
        options = [wp for wp, t in classified if t == desired]
        if options:
            wp = random.choice(options)
            #turn_history.append(desired)
            return wp, desired

    # Fallback: any valid turn option
    options = [wp for wp, t in classified if t != "straight"]
    wp, turn = random.choice(options) if options else random.choice(classified)
    #turn_history.append(turn)
    return wp, turn

# ---------------------------------------------------------------------
# Main path generation
# ---------------------------------------------------------------------
def extract_and_follow_lane(
    carla_map,
    start_waypoint,
    lane_choice="center",
    sampling_distance=0.5,
    path_length=2000,
    speed_ms=13.9
):
    """
    Returns Nx4 array: [x, y, z, speed]
    """

    current_wp = start_waypoint

    # --- lane offset selection ---
    if lane_choice == "left":
        wp = start_waypoint.get_left_lane()
        if wp and wp.lane_type == carla.LaneType.Driving:
            current_wp = wp
    elif lane_choice == "right":
        wp = start_waypoint.get_right_lane()
        if wp and wp.lane_type == carla.LaneType.Driving:
            current_wp = wp

    waypoints = [current_wp]
    turn_history = None

    # --- trace forward ---
    for _ in range(int(path_length / sampling_distance)):
        next_wps = current_wp.next(sampling_distance)
        if not next_wps:
            break

        current_wp,turn_history = choose_next_waypoint(
            current_wp,
            next_wps,
            turn_history
        )
        waypoints.append(current_wp)

    # -----------------------------------------------------------------
    # Save lane geometry (center / left / right) for analysis
    # -----------------------------------------------------------------
    rows = []
    for wp in waypoints:
        c = wp.transform.location

        left_wp = wp.get_left_lane()
        right_wp = wp.get_right_lane()

        l = (
            left_wp.transform.location
            if left_wp and left_wp.lane_type == carla.LaneType.Driving
            else c
        )
        r = (
            right_wp.transform.location
            if right_wp and right_wp.lane_type == carla.LaneType.Driving
            else c
        )

        rows.append([
            c.x, c.y, c.z,
            l.x, l.y, l.z,
            r.x, r.y, r.z,
        ])

    df = pd.DataFrame(rows, columns=[
        "center_x", "center_y", "center_z",
        "left_x", "left_y", "left_z",
        "right_x", "right_y", "right_z",
    ])

    current_dir = Path(__file__).resolve().parent
    out_path = current_dir.parent / "Map_Layouts" / f"traced_lane_data_{lane_choice}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved lane data to {out_path}")

    # -----------------------------------------------------------------
    # Build controller path with curvature-based speed
    # -----------------------------------------------------------------
    path_xyz = []
    path_speed = []

    for i, wp in enumerate(waypoints):
        loc = wp.transform.location
        path_xyz.append([loc.x, loc.y, loc.z])

        if i == 0:
            speed = speed_ms
        else:
            curvature = estimate_curvature(waypoints[i - 1], wp)
            speed = speed_from_curvature(curvature, speed_ms)

        path_speed.append(speed)

    path_xyz = np.asarray(path_xyz)
    path_speed = np.asarray(path_speed).reshape(-1, 1)

    return np.hstack([path_xyz, path_speed])

# --- 4. MAIN EXECUTION ---
def main():
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    #world = client.generate_opendrive_world(XODR_DATA.read_text())
    world = client.load_world('Town10HD') #Town10HD/Town04
    carla_map = world.get_map()
    if LANE_FOLLOWER:
        # --- Path and Spawn Configuration ---
        # 1. Define where you want to start in the world
        start_location = carla.Location(x=100, y=-35, z=0.5) # Example: A point on a main road in Town10HD
        
        # 2. Choose which lane to follow ('center', 'left', 'right')
        LANE_TO_FOLLOW = 'left'
        
        # 3. Set cruise speed
        cruise_speed_kph = 50.0
        
        # 4. Find the nearest waypoint and generate the path
        start_wp = get_closest_waypoint(carla_map, start_location)
        ghost_path_speed = extract_and_follow_lane(
            carla_map, 
            start_wp, 
            lane_choice=LANE_TO_FOLLOW, 
            speed_ms=(cruise_speed_kph / 3.6)
        )
    else:
        ghost_path_speed = np.loadtxt(csv_path, delimiter=',', skiprows=1) # Assuming the CSV has a header row    
        ghost_path_speed[:,3] = ghost_path_speed[:,3] / 1.3 # 
    # Sync Mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.033  # ~30 FPS
    world.apply_settings(settings)
    
    bp = world.get_blueprint_library().filter('model3')[0]
    # Initialize Neural Brain
    brain = NeuralController()
    
    dist_accum = 0.0
    last_pos = None
    log_ds = 0.1  # meters

    try:
        controller = PathFollower(direct_data=ghost_path_speed)
        
        PYGAME_W, PYGAME_H = 800, 400
        visualizer = PygameVisualizer(window_size=(PYGAME_W, PYGAME_H))
        ghost_path_speed[:,3] = ghost_path_speed[:,3] * 3.6 # Convert from m/s to km/h for visualization and control logic
        visualizer.set_path(ghost_path_speed)
        ghost_path_speed[:,3] = ghost_path_speed[:,3] / 3.6 # Convert from km/h back to m/s for control logic

        error_calc = ControllerUtils(data=ghost_path_speed, lookahead_speed=True)
        print(f"Generated Ghost Path with {len(ghost_path_speed)} points.")
        
        # Spawn at the start of the path
        start_transform = get_closest_waypoint(carla_map, carla.Location(x=ghost_path_speed[0][0], y=ghost_path_speed[0][1], z=ghost_path_speed[0][2])).transform
        start_transform.location.z += 0.5 # Elevate spawn point slightly
        vehicle = world.spawn_actor(bp, start_transform)
        # --- RECORDING SETUP ---
        camera_sensor = None
        image_queue = queue.Queue()
        video_writer = None
        if RECORD_VIDEO:
            # Main Camera
            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(MAIN_WIDTH))
            camera_bp.set_attribute('image_size_y', str(MAIN_HEIGHT))
            camera_bp.set_attribute('fov', '90')
            camera_bp.set_attribute('bloom_intensity', '0.1')  # Default is ~0.67. Set to 0.0 for zero glow.
            cam_transform = carla.Transform(carla.Location(x=-8, z=4), carla.Rotation(pitch=-15))
            camera_sensor = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
            camera_sensor.listen(image_queue.put)
            
            # Video Writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video_writer = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, 30.0, (MAIN_WIDTH, MAIN_HEIGHT))
            print(f">>> Recording started. HUD Center: ({HUD_CENTER_X}, {HUD_CENTER_Y})")
        for _ in range(20): world.tick()
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
        
        previous_steer = 0.0
        while True:
            # 1. Physics Step
            world.tick()
            
            # 2. Get State & Errors
            # speed, speed_error, cte, he, fut_cte, progress_idx = get_relative_errors(vehicle, ghost_path_speed)
            cte, he, fut_cte, speed, speed_error, progress_idx, future_path_curvature, fut_yaw = error_calc.calculate_relative_errors(vehicle)
            fut_waypoints = error_calc.get_local_waypoints_dynamic(vehicle)
            # === DRAW 3D FUTURE PATH IN CARLA WORLD ===
            # 1. Get current vehicle global position & rotation
            v_trans = vehicle.get_transform()
            v_x = v_trans.location.x
            v_y = v_trans.location.y
            v_z = v_trans.location.z
            yaw_rad = math.radians(v_trans.rotation.yaw)
            
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)
            
            # Start the line from the car's current location (slightly elevated so it doesn't clip into the road)
            path_points_3d = [carla.Location(x=v_x, y=v_y, z=v_z + 0.5)]
            
            # 2. Convert Local Waypoints back to Global Map Coordinates
            for i in range(0, len(fut_waypoints), 2):
                lx = fut_waypoints[i]     # Forward
                ly = fut_waypoints[i+1]   # Left/Right
                
                # Inverse rotation to get global X and Y
                gx = v_x + (lx * cos_yaw) - (ly * sin_yaw)
                gy = v_y + (lx * sin_yaw) + (ly * cos_yaw)
                
                # Add to our 3D points list
                path_points_3d.append(carla.Location(x=gx, y=gy, z=v_z + 0.5))
                
            # 3. Tell CARLA to draw red lines connecting these points
            for i in range(len(path_points_3d) - 1):
                world.debug.draw_line(
                    path_points_3d[i], 
                    path_points_3d[i+1], 
                    thickness=0.12,               # Thickness of the line in meters
                    color=carla.Color(150, 0, 0), # Red color
                    life_time=0.05                # Stays visible just long enough for this frame (30fps = 0.033s)
                )
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
            #print(f"index: {progress_idx} | cte: {cte:.2f} | he: {he:.2f} | fut_cte: {fut_cte:.2f} | fut_yaw: {fut_yaw:.2f}")    
            # 3. AI Inference
            steer, throttle, brake = brain.process(speed, speed_error, cte, he, fut_cte, yaw_rate, lat_accel, future_path_curvature, fut_yaw, fut_waypoints)
            # EMA Filter (Alpha determines how fast the steering can change)
            # alpha = 1.0 means no filtering. alpha = 0.1 means very heavy smoothing.
            # Try 0.2 to 0.4 for high-speed stability.
            #alpha = 0.1  
            #smoothed_steer = (alpha * steer) + ((1.0 - alpha) * previous_steer)

            # Update for next frame
            #speed_damping = 1.0
            #if speed > 20.0: # If we're going above 72 km/h, start damping the steering to prevent over-correction at high speeds. This is a common technique in racing games and real cars.
            #    speed_damping = 20.0 / speed
            #steer = smoothed_steer * speed_damping
            #previous_steer = steer
            # Long Control
            sim_time = world.get_snapshot().timestamp.elapsed_seconds
            throttle, brake = controller.get_long_vel(speed, ghost_path_speed[progress_idx][3], sim_time)
            
            if ghost_path_speed[progress_idx][3]/3.6 > 1.0 and speed < 0.5 and throttle < 0.1:
                # add random colour each time to the print statement for visibility
                color_code = f"\033[9{random.randint(1, 6)}m"
                print(f"{color_code}>>> Car Stuck: Kicking car forward\033[0m")
                throttle = 0.3 
                brake = 0.0
            # 4. Apply Control
            vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake)))
            
            # 5. Camera Follow
            v_trans = vehicle.get_transform()
            fwd_vec = v_trans.get_forward_vector()
            cam_loc = v_trans.location - (fwd_vec * 10.0) + carla.Location(z=5.0)
            
            # Rotate camera to look slightly down (-15 degrees)
            cam_rot = v_trans.rotation
            cam_rot.pitch = -15.0
            
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))
            
            pos = v_trans.location            
            if last_pos is not None:
                ds = pos.distance(last_pos)
                dist_accum += ds

            if dist_accum >= log_ds:
                visualizer.render(vehicle, ghost_path_speed[progress_idx][3], speed, steer=steer, cte=cte, he=he, fut_yaw=fut_yaw, future_path_curvature=future_path_curvature, yaw_rate=yaw_rate)
                dist_accum = 0.0

            last_pos = pos

            # Debug Print
            # --- VIDEO PROCESSING ---
            if RECORD_VIDEO:
                # 1. Get Main Frame
                image = image_queue.get()
                main_frame = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                main_frame = np.reshape(main_frame, (MAIN_HEIGHT, MAIN_WIDTH, 4))
                
                # IMPORTANT: Add .copy() here to make the array writable
                main_frame = main_frame[:, :, :3].copy() 
                
                # 2. Get Pygame Frame
                pg_surface = pygame.display.get_surface()
                pg_img = pygame.surfarray.array3d(pg_surface)
                pg_img = np.transpose(pg_img, (1, 0, 2))
                pg_img = cv2.cvtColor(pg_img, cv2.COLOR_RGB2BGR)
                
                # 3. Resize HUD
                hud_w = int(MAIN_WIDTH * HUD_SCALE)
                aspect_ratio = PYGAME_H / PYGAME_W
                hud_h = int(hud_w * aspect_ratio)
                hud_resized = cv2.resize(pg_img, (hud_w, hud_h))
                
                # 4. Calculate Coordinates based on Center Params
                center_x_px = int(MAIN_WIDTH * HUD_CENTER_X)
                center_y_px = int(MAIN_HEIGHT * HUD_CENTER_Y)
                
                x_start = center_x_px - (hud_w // 2)
                y_start = center_y_px - (hud_h // 2)
                x_end = x_start + hud_w
                y_end = y_start + hud_h
                
                # 5. Boundary Checks (Clamp to screen edges to prevent crash)
                # If calculations put it off-screen, push it back in
                if x_start < 0: 
                    x_start = 0
                    x_end = hud_w
                if y_start < 0:
                    y_start = 0
                    y_end = hud_h
                if x_end > MAIN_WIDTH:
                    x_end = MAIN_WIDTH
                    x_start = MAIN_WIDTH - hud_w
                if y_end > MAIN_HEIGHT:
                    y_end = MAIN_HEIGHT
                    y_start = MAIN_HEIGHT - hud_h
                
                # 6. Apply Overlay
                # Draw white border
                cv2.rectangle(main_frame, (x_start-2, y_start-2), (x_end+2, y_end+2), (255, 255, 255), 2)
                main_frame[y_start:y_end, x_start:x_end] = hud_resized
                
                video_writer.write(main_frame)
            #print(f"progress_idx: {progress_idx} | Err: {cte:.2f}m | Fut: {fut_cte:.2f}m | Steer: {steer:.2f} | Thr: {throttle:.2f}")

    finally:
        if RECORD_VIDEO:
            if video_writer: video_writer.release()
            if camera_sensor: camera_sensor.destroy()
            print(f"Saved video to {OUTPUT_VIDEO_NAME}")
        del controller
        vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Simulation Ended.")

# --- Plotting Function (No changes needed here) ---
def plot_distribution_comparison(df_full, df_pristine, column_name, plot_info):
    """
    Generates and saves a comparison histogram for a given feature.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import kurtosis
    import numpy as np

    # --- Configuration ---
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'
    
    print(f"Plotting distribution for: {column_name}")
    data_full = df_full[column_name].dropna()
    data_pristine = df_pristine[column_name].dropna()
    
    metrics_full = {'Mean': data_full.mean(), 'Std Dev': data_full.std(), 'Kurtosis': kurtosis(data_full)}
    metrics_pristine = {'Mean': data_pristine.mean(), 'Std Dev': data_pristine.std(), 'Kurtosis': kurtosis(data_pristine)}

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    #sns.histplot(data_pristine, bins=100, kde=True, ax=ax, color='skyblue', label='Pristine Dataset (No Recovery)', stat='density')
    sns.histplot(data_full, bins=100, kde=True, ax=ax, color='coral', label='Full Dataset (with Recovery)', stat='density')
    
    ax.set_title(f'Distribution Comparison for {plot_info["label"]}', fontsize=14, weight='bold')
    ax.set_xlabel(plot_info["label"], fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlim(plot_info['range'])

    #text_pristine = (f"Pristine:\n  $\\mu = {metrics_pristine['Mean']:.3f}$\n  $\\sigma = {metrics_pristine['Std Dev']:.3f}$\n  Kurtosis = ${metrics_pristine['Kurtosis']:.2f}$")
    text_full = (f"Full Dataset:\n  $\\mu = {metrics_full['Mean']:.3f}$\n  $\\sigma = {metrics_full['Std Dev']:.3f}$\n  Kurtosis = ${metrics_full['Kurtosis']:.2f}$")

    #ax.text(0.02, 0.95, text_pristine, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))
    ax.text(0.02, 0.95, text_full, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='seashell', alpha=0.8))

    output_filename = current_dir.parent / "Map_Layouts" / f'distribution_{column_name}.pdf'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Saved plot to {output_filename}")
    plt.close()
def get_training_datasets(PRISTINE_DATASET_PATH, RECOVERY_DATASET_PATH):
    from tqdm import tqdm # For a nice progress bar


    # Heuristic thresholds to define a "pristine" start
    # We'll check the average error over the first 10 frames (~0.33s)
    NUM_FRAMES_TO_CHECK = 10
    CTE_THRESHOLD = 0.1  # Max average of 10 cm off-center
    HEADING_THRESHOLD = 0.05 # Max average of ~3 degrees heading error

    # --- Main Script ---
    print(f"Loading full dataset from {FULL_DATASET_PATH}...")
    df = pd.read_csv(FULL_DATASET_PATH)
    print(f"Loaded {len(df)} rows and {df['episode_id'].nunique()} unique episodes.")

    episode_groups = df.groupby('episode_id')
    pristine_episodes = []
    recovery_episodes = []

    print("Analyzing episodes to classify them as 'Pristine' or 'Recovery'...")
    for ep_id, group in tqdm(episode_groups):
        if len(group) < NUM_FRAMES_TO_CHECK:
            continue # Skip very short/incomplete episodes

        first_n_frames = group.head(NUM_FRAMES_TO_CHECK)
        
        # Calculate the average absolute error for the initial segment
        avg_abs_cte = first_n_frames['cte_input'].abs().mean()
        avg_abs_heading = first_n_frames['heading_error_input'].abs().mean()
        
        # Apply the heuristic
        if avg_abs_cte < CTE_THRESHOLD and avg_abs_heading < HEADING_THRESHOLD:
            pristine_episodes.append(ep_id)
        else:
            recovery_episodes.append(ep_id)

    print("\n--- Classification Results ---")
    print(f"Found {len(pristine_episodes)} pristine episodes.")
    print(f"Found {len(recovery_episodes)} recovery episodes.")

    # --- Filter and Save D_Pristine ---
    df_pristine = df[df['episode_id'].isin(pristine_episodes)]
    df_pristine.to_csv(PRISTINE_DATASET_PATH, index=False)
    print(f"\nSaved {len(df_pristine)} rows to {PRISTINE_DATASET_PATH}")

    # --- Filter and Save D_Recovery (for analysis) ---
    df_recovery = df[df['episode_id'].isin(recovery_episodes)]
    df_recovery.to_csv(RECOVERY_DATASET_PATH, index=False)
    print(f"Saved {len(df_recovery)} rows to {RECOVERY_DATASET_PATH}")
def get_primitive_dataset(FULL_DATASET_PATH, PRIMITIVES_PATH, NGRAMS_PATH):

    # Define your primitive maneuver names exactly as they appear in the CSV
    PRIMITIVE_MANEUVERS = [
        'lane_change_left', 'lane_change_right', 'straight',
        'turn_left', 'turn_right', 'hairpin_left', 
        's_curve_left', 's_curve_right', 'hairpin_right',
        'chicane_left', 'chicane_right'
    ]

    # --- Main Script ---
    print(f"Loading full dataset from {FULL_DATASET_PATH}...")
    df = pd.read_csv(FULL_DATASET_PATH)

    # --- Create D_PrimitivesOnly ---
    # Select rows WHERE the 'maneuver' column IS IN our list
    df_primitives = df[df['maneuver'].isin(PRIMITIVE_MANEUVERS)]
    df_primitives.to_csv(PRIMITIVES_PATH, index=False)
    print(f"Saved {len(df_primitives)} primitive rows to {PRIMITIVES_PATH}")

    # --- Create D_NgramsOnly ---
    # Select rows WHERE the 'maneuver' column IS NOT IN our list (using ~)
    df_ngrams = df[~df['maneuver'].isin(PRIMITIVE_MANEUVERS)]
    df_ngrams.to_csv(NGRAMS_PATH, index=False)
    print(f"Saved {len(df_ngrams)} N-gram rows to {NGRAMS_PATH}")
def prepare_datasets():
    print("Loading datasets...")
    # --- Configuration ---
    FULL_DATASET_PATH = current_dir.parent / 'Map_Layouts' / 'lane_change_dataset_transition.csv'
    PRIMITIVES_PATH = current_dir.parent /'Map_Layouts' / 'D_PrimitivesOnly.csv'
    NGRAMS_PATH = current_dir.parent / 'Map_Layouts' / 'D_NgramsOnly.csv'
    PRISTINE_DATASET_PATH = current_dir.parent / 'Map_Layouts' / 'D_Pristine.csv'
    RECOVERY_DATASET_PATH = current_dir.parent / 'Map_Layouts' / 'D_Recovery.csv' # Bonus: let's save the recovery data too!
    #get_training_datasets(PRISTINE_DATASET_PATH, RECOVERY_DATASET_PATH)
    #get_primitive_dataset(FULL_DATASET_PATH, PRIMITIVES_PATH, NGRAMS_PATH)
    df_full = pd.read_csv(FULL_DATASET_PATH)
    df_pristine = pd.read_csv(PRISTINE_DATASET_PATH)
    FEATURES_TO_PLOT = {
        'cte_input': {'label': 'Cross-Track Error (m)', 'range': (-2.5, 2.5)},
        'heading_error_input': {'label': 'Heading Error (rad)', 'range': (-0.5, 0.5)},
        'yaw_rate_input': {'label': 'Yaw Rate (rad/s)', 'range': (-1.5, 1.5)},
        'future_path_curvature_input': {'label': 'Future Path Curvature (1/m)', 'range': (-0.05, 0.05)},
        'future_heading_error_input': {'label': 'Future Heading Error (rad)', 'range': (-0.5, 0.5)},
        'steer_cmd': {'label': 'Oracle Steering Command', 'range': (-1, 1)}
    }
    for feature, info in FEATURES_TO_PLOT.items():
        plot_distribution_comparison(df_full, df_pristine, feature, info)
        
    print("\nAll plots generated successfully.")    
def plot_errors():
    import pandas as pd
    import matplotlib.pyplot as plt

    #df = pd.read_csv("telemetry_log_lstm_cnn_final.csv")
    df_lstm = pd.read_csv("telemetry_log.csv")

    plt.figure(figsize=(6, 3))
    # use light grey and dark grey for the two lines to make it more visually appealing and easier to distinguish
    #plt.plot(df.index, df.cte , '0.7', label='cte LSTM-CNN')
    #plt.plot(df.index, df_lstm.cte, '0.3', label='cte LSTM Only')
    plt.plot(df_lstm.index, df_lstm.fut_yaw, '0.5', label='steer LSTM Only')
    plt.xlabel("Time (samples)")
    plt.ylabel("steer -1 to +1")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig("steer_plot_comparison_with_ms.pdf", dpi=600)
# --- Main Execution ---
if __name__ == "__main__":
    main()
    #plot_errors()
