from pathlib import Path
import carla
import random
import numpy as np
import math
import csv
import time
import os

# Import your class (assuming it's in a file named 'my_controller.py')
# If it's in the same script, just ensure the class definition is above.
from controller import PathFollower 

# --- CONFIGURATION ---
TOTAL_EPISODES = 500
current_file_path = Path(os.path.abspath(__file__))
current_dir = current_file_path.parent
xodr_file_path = current_dir.parent / "Map_Layouts" / "flattesttrack.xodr"  # Adjust as needed
OUTPUT_FILE = current_dir.parent / "Map_Layouts" / "lane_change_dataset.csv"
MAP_NAME = "FlatTrack" # Best for flat dynamics

class FlatTrackOrchestrator:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        
        print("Loading Custom OpenDRIVE Map...")
        self.world = self.client.generate_opendrive_world(xodr_file_path.read_text())
        
        # SYNC MODE (Physics locked to 30Hz)
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.0333 
        self.world.apply_settings(self.settings)
        
        self.bp_lib = self.world.get_blueprint_library()
        self.vehicle = None

    def generate_ghost_path(self, speed_kph, lc_length):
        """
        Generates a trajectory from Lane -1 (y=-1.75) to Lane -2 (y=-5.25)
        """
        speed_ms = speed_kph / 3.6
        points = []
        
        # Geometry Params
        start_y = -1.75
        target_y = -5.25 # Moving to outer lane
        
        run_up = 30.0  # Drive straight before LC
        run_out = 50.0 # Drive straight after LC
        total_dist = run_up + lc_length + run_out
        
        # Starting X (We assume car spawns around x=10)
        start_x_offset = 10.0 
        
        resolution = 0.5 # Waypoint density
        n_points = int(total_dist / resolution)
        
        for i in range(n_points):
            # Local X relative to start of maneuver
            x_dist = i * resolution
            global_x = start_x_offset + x_dist
            
            current_y = start_y
            
            # 1. Run Up (Straight)
            if x_dist < run_up:
                current_y = start_y
                
            # 2. Lane Change (Sigmoid/Cosine)
            elif x_dist < (run_up + lc_length):
                # Normalized progress 0.0 to 1.0
                p = (x_dist - run_up) / lc_length
                # Cosine interpolation: y = A + (B-A) * (1 - cos(p*pi))/2
                # Note: We want 0 to 1 curve
                factor = (1 - math.cos(p * math.pi)) / 2.0
                current_y = start_y + (target_y - start_y) * factor
                
            # 3. Run Out (Straight)
            else:
                current_y = target_y
            
            # Z is slightly elevated to avoid Z-fighting with road
            points.append([global_x, current_y, 0.0, speed_ms])
            
        return np.array(points)

    def calculate_relative_errors(self, vehicle, path_points):
        """
        Calculates Inputs for the AI:
        1. CTE (Lateral Error)
        2. Heading Error
        3. Future Curvature (Lookahead Error)
        """
        v_trans = vehicle.get_transform()
        v_loc = v_trans.location
        
        # Find closest point
        # Since our path is X-aligned and sorted, we can optimize, 
        # but brute force is safe for short horizons.
        dists = np.linalg.norm(path_points[:, :2] - np.array([v_loc.x, v_loc.y]), axis=1)
        min_idx = np.argmin(dists)
        
        closest_pt = path_points[min_idx]
        
        # 1. Lateral Error (CTE)
        # For this X-aligned map, CTE is simply (y_car - y_path)
        # Note: We need to respect sign relative to path direction.
        # Path is heading East (+X). Y is Left.
        # If car Y > path Y, car is to the left (Positive CTE).
        cte = v_loc.y - closest_pt[1]
        
        # 2. Heading Error
        # Calculate path tangent
        if min_idx + 1 < len(path_points):
            dx = path_points[min_idx+1][0] - path_points[min_idx][0]
            dy = path_points[min_idx+1][1] - path_points[min_idx][1]
        else:
            dx = path_points[min_idx][0] - path_points[min_idx-1][0]
            dy = path_points[min_idx][1] - path_points[min_idx-1][1]
            
        path_yaw = math.atan2(dy, dx)
        vehicle_yaw = math.radians(v_trans.rotation.yaw)
        
        he = vehicle_yaw - path_yaw
        # Normalize to -pi, pi
        while he > math.pi: he -= 2*math.pi
        while he < -math.pi: he += 2*math.pi

        # 3. Lookahead/Future Error (The "Scalability" Feature)
        # What is the CTE 1.0 seconds ahead?
        # Current Speed
        vel = vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2)
        lookahead_dist = max(5.0, speed * 1.0) # Look 1s ahead
        
        # Find index approx lookahead_dist away
        look_idx = min_idx
        dist_accum = 0
        while look_idx < len(path_points)-1 and dist_accum < lookahead_dist:
            dist_accum += 0.5 # resolution
            look_idx += 1
            
        future_pt = path_points[look_idx]
        # Future CTE approximation (simple Y diff for this straight track)
        future_cte = v_loc.y - future_pt[1]

        return cte, he, future_cte

    def run(self):
        print("Starting Data Generation on Flat Track...")
        
        # Expanded Header to include Future CTE
        header = [
            "episode_id", "speed_input", "cte_input", "heading_error_input", "future_cte_input", 
            "steer_cmd", "throttle_cmd", "brake_cmd"
        ]
        
        if not os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'w', newline='') as f:
                csv.writer(f).writerow(header)
                print(f"Created new dataset file: {OUTPUT_FILE}")
        spectator = self.world.get_spectator()
        cam_loc = carla.Location(x=80, z=120.0) # 30 meters high
        cam_rot = carla.Rotation(pitch=-90.0) # Look straight down
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))
        
        for episode in range(TOTAL_EPISODES):
            # 1. Randomize
            target_speed_kph = random.uniform(40, 110) 
            lc_length = random.uniform(30, 80)
            
            print(f"Ep {episode}: Spd={target_speed_kph:.1f}, Len={lc_length:.1f}m")

            # 2. Generate Ghost Path (Lane -1 to -2)
            ghost_path = self.generate_ghost_path(target_speed_kph, lc_length)
            
            # 3. Setup Controller
            controller = PathFollower(direct_data=ghost_path)
            
            # 4. Spawn Vehicle (Explicitly at Start of Lane -1)
            bp = self.bp_lib.filter('model3')[0]
            # Spawn at x=10, y=-1.75 (Lane -1 Center), z=0.5
            start_transform = carla.Transform(
                carla.Location(x=10.0, y=-1.75, z=0.5),
                carla.Rotation(yaw=0.0)
            )
            
            self.vehicle = self.world.try_spawn_actor(bp, start_transform)
            if not self.vehicle:
                print("Spawn failed, retrying...")
                continue
            # Settle
            for _ in range(10): self.world.tick()
            
            episode_data = []

            try:
                while True:
                    # Current Physics State
                    v_trans = self.vehicle.get_transform()
                    vel = self.vehicle.get_velocity()
                    speed_ms = math.sqrt(vel.x**2 + vel.y**2)
                    
                    # End Condition: Speed too high or Path Finished
                    if controller.last_closest_idx >= len(ghost_path) - 10:
                        break
                    
                    # --- GET INPUTS (Features) ---
                    cte, he, fut_cte = self.calculate_relative_errors(self.vehicle, ghost_path)
                    
                    # --- GET OUTPUTS (Labels from Teacher) ---
                    steer_rad = controller.get_pure_pursuit_steering(
                        v_trans.location.x, v_trans.location.y, 
                        math.radians(v_trans.rotation.yaw), speed_ms
                    )
                    
                    # Normalize Steer [-1, 1] (Assuming max steer 70deg)
                    steer_cmd = np.clip(steer_rad / 1.22, -1.0, 1.0) # 1.22 rad ~= 70 deg
                    
                    # Long Control
                    sim_time = self.world.get_snapshot().timestamp.elapsed_seconds
                    thr, brk = controller.get_long_vel(speed_ms, target_speed_kph/3.6, sim_time)
                    
                    # --- APPLY CONTROL ---
                    self.vehicle.apply_control(carla.VehicleControl(
                        throttle=float(thr), steer=float(steer_cmd), brake=float(brk)
                    ))
                    
                    # --- SAVE ROW ---
                    row = [episode, speed_ms, cte, he, fut_cte, steer_cmd, thr, brk]
                    episode_data.append(row)
                    
                    self.world.tick()
            
            except Exception as e:
                print(f"Error: {e}")
                
            finally:
                # Save and Destroy
                if len(episode_data) > 0:
                    with open(OUTPUT_FILE, 'a', newline='') as f:
                        csv.writer(f).writerows(episode_data)
                
                self.vehicle.destroy()
        
        # Cleanup
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)
        print("Done.")

if __name__ == '__main__':
    try:
        orch = FlatTrackOrchestrator()
        orch.run()
    except KeyboardInterrupt:
        print("Cancelled by user")