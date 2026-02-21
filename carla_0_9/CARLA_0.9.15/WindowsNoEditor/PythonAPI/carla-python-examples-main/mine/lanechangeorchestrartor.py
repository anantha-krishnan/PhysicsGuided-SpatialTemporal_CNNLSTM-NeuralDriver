# lanechangeorchestrator.py
from pathlib import Path
import carla
import random
import numpy as np
import math
import csv
import time
import os

from controller import PathFollower 

# --- CONFIGURATION ---
TOTAL_EPISODES = 350
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

    def generate_ghost_path(self, speed_kph, lc_length, start_y=-1.75, target_y=-5.25, run_out=50.0):
        """
        Generates a Sigmoid path with variable start, end, and run-out length.
        """
        speed_ms = speed_kph / 3.6
        points = []
        
        # Geometry Params
        run_up = 30.0  # Time to stabilize speed before maneuver
        total_dist = run_up + lc_length + run_out
        
        # Starting X (We assume car spawns around x=10)
        start_x_offset = 10.0 
        
        resolution = 0.5 # Waypoint density
        n_points = int(total_dist / resolution)
        
        for i in range(n_points):
            # Local X relative to start of maneuver
            x_dist = i * resolution
            global_x = start_x_offset + x_dist
            
            # Logic
            if x_dist < run_up:
                # Phase 1: Run Up (Straight)
                current_y = start_y
                
            elif x_dist < (run_up + lc_length):
                # Phase 2: Lane Change (Sigmoid Interpolation)
                # p goes from 0.0 to 1.0
                p = (x_dist - run_up) / lc_length
                
                # Cosine S-Curve: (1 - cos(p*pi)) / 2 gives smooth 0->1 curve
                factor = (1 - math.cos(p * math.pi)) / 2.0
                current_y = start_y + (target_y - start_y) * factor
                
            else:
                # Phase 3: Run Out (Cruising)
                current_y = target_y
            
            # Z=0, Speed included for the controller
            points.append([global_x, current_y, 0.0, speed_ms])
            
        return np.array(points)
    def get_cte(self, vehicle, closest_pt_on_path, next_pt_on_path):
        """
        Calculates the Cross-Track Error (CTE) for the given vehicle and path.
        CTE is the perpendicular distance of the vehicle location to the tangent line of the path at the closest point. 
        Tangent line is obtained by looking at the next point in the path. 
        CTE sign is obtained based on the cross product of the path tangent and the vector from the closest point to the vehicle.
        """
        v_trans = vehicle.get_transform()
        v_loc = v_trans.location
        # Path tangent vector
        path_tangent = np.array([next_pt_on_path[0] - closest_pt_on_path[0],
                                 next_pt_on_path[1] - closest_pt_on_path[1]])
        path_tangent_norm = np.linalg.norm(path_tangent)
        if path_tangent_norm == 0:
            return 0.0  # Avoid division by zero, treat as zero error if path points are the same
        path_tangent_unit = path_tangent / path_tangent_norm
        # CTE: Vector from closest path point to vehicle
        vec_to_vehicle = np.array([v_loc.x - closest_pt_on_path[0],
                                   v_loc.y - closest_pt_on_path[1]])
        # CTE mag
        cte_mag = np.linalg.norm(vec_to_vehicle)
        # CTE sign: Use cross product to determine if vehicle is left or right of path
        cross_prod = path_tangent_unit[0] * vec_to_vehicle[1] - path_tangent_unit[1] * vec_to_vehicle[0]
        cte_sign = 1.0 if cross_prod > 0 else -1.0
        return cte_sign * cte_mag

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
        #cte = v_loc.y - closest_pt[1]
        cte = self.get_cte(vehicle, closest_pt, path_points[min_idx+1] if min_idx+1 < len(path_points) else closest_pt)
        
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
        #future_cte = v_loc.y - future_pt[1]
        future_cte = self.get_cte(vehicle, future_pt, path_points[look_idx+1] if look_idx+1 < len(path_points) else future_pt)

        return cte, he, future_cte

    def run(self):
        print("Starting Data Generation on Flat Track...")
        
        # Expanded Header to include Future CTE
        header = [
            "episode_id", "speed_input", 'speed_error_input', "cte_input", "heading_error_input", "future_cte_input", 
            "yaw_rate_input", "lat_accel_input",
            "steer_cmd", "throttle_cmd", "brake_cmd"
        ]
        
        # if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(header)
            print(f"Created new dataset file: {OUTPUT_FILE}")
        spectator = self.world.get_spectator()
        cam_loc = carla.Location(x=80, z=120.0) # 30 meters high
        cam_rot = carla.Rotation(pitch=-90.0) # Look straight down
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))
        
        for episode in range(TOTAL_EPISODES):
            # 1. Randomize
            cruise_speed_kph = random.uniform(40, 110) 
            lc_length = random.uniform(30, 80)
            start_y = -1.75  # Lane -1
            # Pick a maneuver type
            maneuver_roll = random.random() 
            if maneuver_roll < 0.4:
                target_y = random.uniform(-5.25, -3.75)  # Lane -2 (Left Lane Change)
            elif maneuver_roll < 0.8:
                target_y = random.uniform(2.25, 3.25)   # Lane 0 (Right Lane Change)
            else:
                offset = random.uniform(-0.5, 0.5)  # Lane -1 (Some speed variation)
                target_y = start_y + offset
                lc_length = random.uniform(40,60) # Longer run for straight to get more stable data
            # cruising data
            run_out_dist = random.uniform(50.0, 200.0)

            print(f"Ep {episode}: Spd={cruise_speed_kph:.1f}, Len={lc_length:.1f}m")

            # 2. Generate Ghost Path (Lane -1 to -2)
            ghost_path = self.generate_ghost_path(cruise_speed_kph, lc_length,start_y, target_y, run_out=run_out_dist)
            
            # 3. Setup Controller
            controller = PathFollower(direct_data=ghost_path)
            
            # 4. Spawn Vehicle (Explicitly at Start of Lane -1)
            bp = self.bp_lib.filter('model3')[0]
            # Spawn at x=10, y=-1.75 (Lane -1 Center), z=0.5
            
            # Introduce random spawn offsets
            y_offset = random.uniform(-2.5, 2.5)  # Spawn up to 2.5m off center
            yaw_offset = random.uniform(-30, 30) # Spawn with up to 30 degrees of heading error
            start_transform = carla.Transform(
                carla.Location(x=10.0, y=-1.75+y_offset, z=0.5),
                carla.Rotation(yaw=yaw_offset)
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
                    # Yaw Rate (rad/s)
                    ang_vel = self.vehicle.get_angular_velocity()
                    yaw_rate = math.radians(ang_vel.z) 
                    
                    # Lateral Acceleration (m/s^2)
                    # We can use the formula: a_lat = v * yaw_rate
                    # Or get it from IMU (acc.y in local frame). Calculation is cleaner for now.
                    lat_accel = speed_ms * yaw_rate 

                    cornering_limit_ms = controller.get_curvature_based_speed(
                        v_trans.location, 
                        max_lat_accel=5.0 # Limit lateral Gs to ~0.5g
                    )
                    speed_error = cruise_speed_kph/3.6 - speed_ms

                    # The actual target is the minimum of "Cruise Setpoint" and "Physics Limit"
                    final_target_ms = min(cruise_speed_kph/3.6, cornering_limit_ms)

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
                    thr, brk = controller.get_long_vel(speed_ms, final_target_ms, sim_time)
                    
                    # --- APPLY CONTROL ---
                    self.vehicle.apply_control(carla.VehicleControl(
                        throttle=float(thr), steer=float(steer_cmd), brake=float(brk)
                    ))
                    
                    # --- SAVE ROW ---
                    row = [episode, 
                           speed_ms, speed_error, cte, he, fut_cte, 
                           yaw_rate, lat_accel,
                           steer_cmd, thr, brk]
                    episode_data.append(row)
                    #print(f"Ep {episode} | Target Spd: {final_target_ms:.1f} m/s | Speed: {speed_ms:.1f} m/s | CTE: {cte:.2f} m | HE: {math.degrees(he):.1f} deg | Fut CTE: {fut_cte:.2f} m | Yaw Rate: {math.degrees(yaw_rate):.1f} deg/s | Lat Accel: {lat_accel:.2f} m/s²")
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

# CarlaUE4.exe /Game/Maps/RaceTrack -windowed -carla-server -benchmark -fps=30        