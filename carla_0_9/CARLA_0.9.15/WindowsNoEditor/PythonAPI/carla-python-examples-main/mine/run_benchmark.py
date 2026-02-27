import carla
import pandas as pd
import numpy as np
import math
import os
import time
from pathlib import Path
from datetime import datetime

# Import your system
from neural_driver import NeuralController
from utility_fncs_train_inference import ControllerUtils
from minimap import PygameVisualizer
from path_generators import (
    generate_straight_path,
    generate_90_degree_turn_path,
    generate_lane_change_path,
    generate_chicane_path,
    generate_s_curve_path,
    generate_hairpin_turn_path
)

# --- CONFIG ---
current_file_path = Path(os.path.abspath(__file__))
current_dir = current_file_path.parent
MODEL_SAVE_PATH = current_dir.parent / "Map_Layouts" 
RESULTS_DIR = MODEL_SAVE_PATH / "benchmark_results"
XODR_DATA = MODEL_SAVE_PATH / "flattesttrack.xodr"
RESULTS_DIR.mkdir(exist_ok=True)
TEST_SPEED_KPH = 50.0

def run_benchmark():
    # 1. Setup CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.generate_opendrive_world(XODR_DATA.read_text())
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.033
    world.apply_settings(settings)
    
    bp = world.get_blueprint_library().filter('model3')[0]

    # 2. Define Playlist
    scenarios = [
        ("01_Straight_Line", generate_straight_path(TEST_SPEED_KPH, length=150)),
        ("02_Lane_Change_Left", generate_lane_change_path(TEST_SPEED_KPH, 40, -1.75, -5.25)),
        ("03_Turn_90_Left", generate_90_degree_turn_path(TEST_SPEED_KPH, 20.0, 'left')),
        ("04_Turn_90_Right", generate_90_degree_turn_path(TEST_SPEED_KPH, 20.0, 'right')),
        ("05_Chicane_Left", generate_chicane_path(TEST_SPEED_KPH, 3.0, 40.0, 'left')),
        ("06_S_Curve", generate_s_curve_path(TEST_SPEED_KPH, 25.0, 'left')),
        ("07_Hairpin_Right", generate_hairpin_turn_path(TEST_SPEED_KPH, 12.0, 'right')), # Tighter test
    ]

    print(f"=== Starting Benchmark of {len(scenarios)} Scenarios ===")
    visualizer = PygameVisualizer(window_size=(1000, 500))

    for name, path in scenarios:
        print(f">> Running: {name}")
        visualizer.set_path(path)
        brain = NeuralController()
        
        error_calc = ControllerUtils(data=path, lookahead_dist=25.0)
        # Spawn
        start_x, start_y = path[0][0], path[0][1]
        # Calculate heading from first few points
        dx = path[1][0] - start_x
        dy = path[1][1] - start_y
        yaw = math.degrees(math.atan2(dy, dx))
        
        start_pose = carla.Transform(carla.Location(x=start_x, y=start_y, z=0.5), carla.Rotation(yaw=yaw))
        vehicle = world.try_spawn_actor(bp, start_pose)
        
        if not vehicle:
            print("   Spawn Failed! Skipping.")
            continue
        # Spectator
        spectator = world.get_spectator()
        v_trans = vehicle.get_transform()
        fwd_vec = v_trans.get_forward_vector()
        cam_loc = v_trans.location - carla.Location(x=2.0) + carla.Location(z=5.0)
        
        # Rotate camera to look slightly down (-15 degrees)
        cam_rot = v_trans.rotation
        cam_rot.pitch = -15.0
        
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))    
        # Settle
        for _ in range(20): world.tick()
        
        # Run Loop
        logs = []
        done = False
        steps = 0
        max_steps = 1500 # Timeout
        try:
            while not done:
                world.tick()
                
                # Get Data
                cte, he, fut_cte, speed, speed_error, progress_idx, future_path_curvature = error_calc.calculate_relative_errors(vehicle)
                ang_vel_deg = vehicle.get_angular_velocity()
                yaw_rate = math.radians(ang_vel_deg.z)
                lat_accel = speed * yaw_rate
                visualizer.render(vehicle, path[progress_idx][3], speed)
                
                # Inference
                steer, throttle, brake = brain.process(speed, speed_error, cte, he, fut_cte, yaw_rate, lat_accel, future_path_curvature)
                
                # Control (Steer Only)
                vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake)))
                
                # Logging
                v_trans = vehicle.get_transform()
                logs.append({
                    'step': steps,
                    'ref_x': path[progress_idx][0], 'ref_y': path[progress_idx][1],
                    'act_x': v_trans.location.x, 'act_y': v_trans.location.y,
                    'act_yaw': v_trans.rotation.yaw,
                    'cte': cte,
                    'heading_error': he,
                    'curvature_input': future_path_curvature,
                    'steer_cmd': steer,
                    'speed': speed
                })
                
                steps += 1
                if progress_idx >= len(path) - 5:
                    print("Track Complete!")
                    done = True

        finally:
            vehicle.destroy()
            del error_calc
            del brain
            
        # Save CSV
        df = pd.DataFrame(logs)
        timestamp = datetime.now().strftime("%H%M")
        filename = RESULTS_DIR / f"{name}.csv"
        df.to_csv(filename, index=False)
        print(f"   Saved {len(df)} frames to {filename}")

    # Cleanup
    settings.synchronous_mode = False
    world.apply_settings(settings)
    print("=== Benchmark Complete ===")

if __name__ == "__main__":
    run_benchmark()