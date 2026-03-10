# lanechangeorchestrator.py
from pathlib import Path
import carla
import random
import numpy as np
import math
import csv
import time
import os
import itertools
import matplotlib.pyplot as plt
from controller import PathFollower 
from utility_fncs_train_inference import ControllerUtils
from minimap import PygameVisualizer

from path_generators import (
    generate_straight_path,
    generate_90_degree_turn_path,
    generate_hairpin_turn_path,
    generate_s_curve_path,
    generate_lane_change_path,
    generate_chicane_path,
    generate_specific_chain
)
# --- CONFIGURATION ---
TOTAL_EPISODES = 500
current_file_path = Path(os.path.abspath(__file__))
current_dir = current_file_path.parent
xodr_file_path = current_dir.parent / "Map_Layouts" / "flattesttrack.xodr"  # Adjust as needed
OUTPUT_FILE = current_dir.parent / "Map_Layouts" / "lane_change_dataset.csv"
OUTPUT_FILE_TRANSITION = current_dir.parent / "Map_Layouts" / "lane_change_dataset_transition.csv"
MAP_NAME = "FlatTrack" # Best for flat dynamics
MIN_STEPS = 50  # at least 1.5 seconds of driving per episode to ensure we get enough data, especially for longer maneuvers

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
        self.visualizer = None
    
    def set_spectator(self, target_transform,spectator):        
        dist_behind = 15.0  # Distance behind the car
        height = 120.0       # Height above the car (50 is usually too high for a driver view)
        look_down_angle = -90.0 # Degrees (Negative looks down)
        # cam_loc = target_transform.location - target_transform.get_forward_vector() * dist_behind
        cam_loc = target_transform.location
        cam_loc.z += height
        yaw = 0#target_transform.rotation.yaw
        cam_rot = carla.Rotation(pitch=look_down_angle, yaw=yaw, roll=0.0)
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))
    def set_spectator_pitch_up(self, target_transform, spectator, distance=120.0, pitch=-15.0):
        """
        Static camera placement.
        Args:
            target_transform: The car's spawn location.
            spectator: The CARLA spectator actor.
            distance: Line-of-sight distance from car to camera (meters).
            pitch: Look-down angle in degrees (e.g., -15).
        """
        target_loc = target_transform.location
        
        # 1. Calculate Offsets based on Pitch and Distance
        # We use absolute pitch to calculate positive height and back distance
        pitch_rad = math.radians(abs(pitch))
        
        # "Back" distance along global X-axis
        back_offset = distance * math.cos(pitch_rad)
        # "Up" height along global Z-axis
        height_offset = distance * math.sin(pitch_rad)
        
        # 2. Set Camera Location
        # We assume the track moves in positive X. We subtract X to move behind.
        cam_loc = carla.Location(
            x=target_loc.x - back_offset, 
            y=target_loc.y,               # Keep camera inline with the car's lane
            z=target_loc.z + height_offset
        )
        
        # 3. Set Camera Rotation
        # Yaw=0 ensures we look straight down the track, regardless of car rotation
        cam_rot = carla.Rotation(pitch=pitch, yaw=0.0, roll=0.0)
        
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))
    def run_isolated_maneuvers(self):
        print("Starting Data Generation on Flat Track...")
        # Expanded Header to include Future CTE
        header = [
            "episode_id", "maneuver", "speed_input", 'speed_error_input', "cte_input",
            "heading_error_input", "future_cte_input", 
            "yaw_rate_input", "lat_accel_input", "future_path_curvature_input","future_heading_error_input",
            "steer_cmd", "throttle_cmd", "brake_cmd"
        ]
        
        # if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(header)
            print(f"Created new dataset file: {OUTPUT_FILE}")
        spectator = self.world.get_spectator()
        # spectator CONFIGURATION
        maneuvers_per_type = 30 
        full_episode_plan = []
        maneuver_list = [
                'lane_change_left', 'lane_change_right', 'straight',
                'turn_left', 'turn_right', 'hairpin_left', 
                's_curve_left', 's_curve_right', 'hairpin_right',
                'chicane_left', 'chicane_right'
            ]
        for maneuver in maneuver_list:
            full_episode_plan.extend([maneuver] * maneuvers_per_type)
        
        random.shuffle(full_episode_plan) # Shuffle the plan so they run in random order
        self.visualizer = PygameVisualizer(window_size=(1000, 500))
        for episode, maneuver in enumerate(full_episode_plan):
            cruise_speed_kph = random.uniform(40, 90) 
            lc_length = random.uniform(30, 80)
            start_y = -1.75  # Lane -1
            run_out_dist = random.uniform(50.0, 200.0)
            print(f"Ep {episode}: Generating a '{maneuver}' at {cruise_speed_kph:.1f} kph")
            # Pick a maneuver type
            # 2. Generate Ghost Path based on chosen maneuver
            if maneuver == 'lane_change_left':
                start_y = -1.75
                target_y = random.uniform(-10.25, -3.75)  # Lane -2 (Left Lane Change)
                ghost_path = generate_lane_change_path(cruise_speed_kph, lc_length, start_y, target_y, run_out=run_out_dist)
            
            elif maneuver == 'lane_change_right':
                start_y = -1.75
                target_y = random.uniform(2.25, 10.25)  # Lane 2 (Right Lane Change)
                ghost_path = generate_lane_change_path(cruise_speed_kph, lc_length, start_y, target_y, run_out=run_out_dist)

            elif maneuver == 'straight':
                ghost_path = generate_straight_path(cruise_speed_kph, length=300.0)

            elif maneuver == 'turn_left':
                turn_radius = random.uniform(15, 30) # City to rural road turn radius
                ghost_path = generate_90_degree_turn_path(cruise_speed_kph, turn_radius, 'left', run_out=run_out_dist)
            
            elif maneuver == 'turn_right':
                turn_radius = random.uniform(15, 30)
                ghost_path = generate_90_degree_turn_path(cruise_speed_kph, turn_radius, 'right', run_out=run_out_dist)
            elif maneuver == 'hairpin_left':
                turn_radius = random.uniform(10, 15) # Tighter radius for hairpins
                slow_speed = random.uniform(30, 50) # Force slower speed for safety
                ghost_path = generate_hairpin_turn_path(slow_speed, turn_radius, 'left', run_out=run_out_dist)
            elif maneuver == 'hairpin_right':
                turn_radius = random.uniform(10, 15) # Tighter radius for hairpins
                slow_speed = random.uniform(30, 50) # Force slower speed for safety
                ghost_path = generate_hairpin_turn_path(slow_speed, turn_radius, 'right', run_out=run_out_dist)
            elif maneuver == 's_curve_left':
                turn_radius = random.uniform(20, 40)
                ghost_path = generate_s_curve_path(cruise_speed_kph, turn_radius, 'left', run_out=run_out_dist) # or random direction
            elif maneuver == 's_curve_right':
                turn_radius = random.uniform(20, 40)
                ghost_path = generate_s_curve_path(cruise_speed_kph, turn_radius, 'right', run_out=run_out_dist) # or random direction
            elif maneuver == 'chicane_left':
                # A Swerve starting to the Left
                width = random.uniform(2.0, 4.0)   # Swerve 2-4 meters sideways
                length = random.uniform(30.0, 50.0) # Complete the swerve in 30-50 meters
                ghost_path = generate_chicane_path(cruise_speed_kph, width, length, 'left', run_out=run_out_dist)
            elif maneuver == 'chicane_right':
                # A Swerve starting to the Right
                width = random.uniform(2.0, 4.0)
                length = random.uniform(30.0, 50.0)
                ghost_path = generate_chicane_path(cruise_speed_kph, width, length, 'right', run_out=run_out_dist)
            utils = ControllerUtils(data=ghost_path, lookahead_dist=25.0) # For error calculations and lookahead features
            # 3. Setup Controller
            controller = PathFollower(direct_data=ghost_path)
            
            bp = self.bp_lib.filter('model3')[0]
            # Spawn at x=10, y=-1.75 (Lane -1 Center), z=0.5
            
            # Introduce random spawn offsets
            y_offset = random.uniform(-2.5, 2.5)  # Spawn up to 2.5m off center
            yaw_offset = random.uniform(-30, 30) # Spawn with up to 30 degrees of heading error
            ideal_start_x = ghost_path[0][0] # This will now be 1000.0
            ideal_start_y = ghost_path[0][1] # This will now be -1.75
            start_transform = carla.Transform(
                carla.Location(x=ideal_start_x, y=ideal_start_y+y_offset, z=2.0),
                carla.Rotation(yaw=yaw_offset)
            )
            # 4. Spawn Vehicle 
            self.vehicle = self.world.try_spawn_actor(bp, start_transform)
            if not self.vehicle:
                print("Spawn failed, retrying...")
                continue
            for _ in range(20): 
                self.world.tick()
            v_loc = self.vehicle.get_transform().location
            start_dist = np.linalg.norm(ghost_path[0][:2] - np.array([v_loc.x, v_loc.y]))
            end_dist = np.linalg.norm(ghost_path[-1][:2] - np.array([v_loc.x, v_loc.y]))
            print(f"   Spawn Checks: Dist to Start={start_dist:.2f}m, Dist to End={end_dist:.2f}m")
            print(f"   Initial Spawn Location: x={v_loc.x:.2f}, y={v_loc.y:.2f}, z={v_loc.z:.2f}")
            print(f"   Initial ghost start point: x={ghost_path[0][0]:.2f}, y={ghost_path[0][1]:.2f}")
            start_dist = np.linalg.norm(ghost_path[0][:2] - np.array([v_loc.x, v_loc.y]))
            if start_dist > 10.0:
                print(f"   >>> ERROR: Car is {start_dist:.1f}m away from path start! Skipping...")
                self.vehicle.destroy()
                break
            # Settle
            for _ in range(10): self.world.tick()
            self.set_spectator_pitch_up(carla.Transform(carla.Location(x=ideal_start_x, y=ideal_start_y, z=0.5)), spectator, distance=100.0, pitch=-25.0)
            
            episode_data = []
            self.visualizer.set_path(ghost_path)
            try:
                while True:
                    # Current Physics State
                    v_trans = self.vehicle.get_transform()
                    remaining_points = len(ghost_path) - controller.last_closest_idx
                    if len(episode_data) > 20 and speed_ms < 0.1:
                        print("   -> End Reason: Car Stuck (Speed ~ 0)")
                        break
                    if v_trans.location.z < -1.0:
                        print("   -> End Reason: Car fell through map")
                        break
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
                        max_lat_accel=100.0 # Limit lateral Gs to ~0.5g
                    )
                    speed_error = cruise_speed_kph/3.6 - speed_ms

                    # The actual target is the minimum of "Cruise Setpoint" and "Physics Limit"
                    # final_target_ms = min(cruise_speed_kph/3.6, cornering_limit_ms)
                    final_target_ms = cruise_speed_kph/3.6 # For data diversity, we can keep the cruise speed as the target and let the controller learn to slow down for corners based on the path curvature features.

                    # End Condition: Speed too high or Path Finished
                    if controller.last_closest_idx >= len(ghost_path) - 7:
                        print(f"   -> End Reason: Path Finished (Points left: {len(ghost_path) - controller.last_closest_idx}), Ep {episode}: {maneuver}")
                        # stop the iteration and save the data, but we don't want to break here because we want to ensure the finally block runs to save the episode data. So we can use a flag or just let it naturally exit the loop on the next iteration when it checks the condition again. For simplicity, we'll just let it exit naturally.
                        break

                        
                    # --- GET INPUTS (Features) ---
                    cte, he, fut_cte, sp, sp_err, last_closest_idx, future_path_curvature, fut_yaw = utils.calculate_relative_errors(self.vehicle)
                    
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
                    # keep the spectator above the vehicle, slightly behind and looking down
                    #self.set_spectator(v_trans, spectator)
                    # --- SAVE ROW ---
                    row = [episode, maneuver, 
                           speed_ms, speed_error, cte, he, fut_cte, 
                           yaw_rate, lat_accel,future_path_curvature,fut_yaw,
                           steer_cmd, thr, brk]
                    episode_data.append(row)
                    self.visualizer.render(self.vehicle, final_target_ms, speed_ms)

                    #print(f"Ep {episode} | Target Spd: {final_target_ms:.1f} m/s | Speed: {speed_ms:.1f} m/s | CTE: {cte:.2f} m | HE: {math.degrees(he):.1f} deg | Fut CTE: {fut_cte:.2f} m | Yaw Rate: {math.degrees(yaw_rate):.1f} deg/s | Lat Accel: {lat_accel:.2f} m/s²")
                    self.world.tick()
            
            except Exception as e:
                print(f"Error: {e}")
                break
            finally:
                # Save and Destroy
                if len(episode_data) > MIN_STEPS and len(episode_data) > 0:
                    with open(OUTPUT_FILE, 'a', newline='') as f:
                        csv.writer(f).writerows(episode_data)
                        print(f"")
                        if remaining_points < 10:
                            print(f"   -> End Reason: Path Finished (Points left: {remaining_points}), Ep {episode}: Saved {len(episode_data)} rows.")
                    
                else:
                    print(f"Ep {episode}, {maneuver}: DISCARDED (Too short: {len(episode_data)} rows). Check logs!")
                    break
                self.vehicle.destroy()
                # clean up memory
                del self.vehicle
                del controller
                del utils
                del episode_data
                del ghost_path

        
        # Cleanup
        self.visualizer.destroy()
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)
        print("Done.")

    def run_transitions(self):
        # --- DEFINE PRIMITIVES ---
        primitives_for_2gram = [
            'turn_left', 'turn_right',
            'hairpin_left', 'hairpin_right',
            'lane_change_left', 'lane_change_right',
            'chicane_left', 'chicane_right',
            #'s_curve_left', 's_curve_right' 
        ]
        # all primitives including isolated ones (for self-loops in the transition matrix)
        primitives = primitives_for_2gram + ['straight', 's_curve_left', 's_curve_right'] 
        print("Starting Data Generation on Flat Track...")
        
        header = [
            "episode_id", "maneuver", "speed_input", 'speed_error_input', "cte_input",
            "heading_error_input", "future_cte_input", 
            "yaw_rate_input", "lat_accel_input", "future_path_curvature_input","future_heading_error_input",
            "steer_cmd", "throttle_cmd", "brake_cmd"
        ]
        NUM_WAYPOINTS = 10
        for i in range(NUM_WAYPOINTS):
            header.extend([f"wp_{i}_x", f"wp_{i}_y"])
        with open(OUTPUT_FILE_TRANSITION, 'w', newline='') as f:
            csv.writer(f).writerow(header)
            print(f"Created new dataset file: {OUTPUT_FILE_TRANSITION}")
        # --- GENERATE THE COMPLETE TRANSITION MATRIX ---
        # This creates every possible pair: (Straight->Turn), (Turn->Straight), (Turn->Turn)...
        # 11 * 11 = 121 unique sequences
        all_pairs = list(itertools.product(primitives_for_2gram, repeat=2))
        
        # Convert tuples to lists: [('straight', 'turn_left'), ...] -> [['straight', 'turn_left'], ...]
        transition_curriculum = [list(pair) for pair in all_pairs]
        
        # --- ADD ISOLATED MANEUVERS (Self-loops) ---
        # We want extra practice on just doing one thing perfectly.
        isolated_curriculum = [[p] for p in primitives]
        
        # --- OPTIONAL: ADD 3-STEP CHAINS (Sparse Sampling) ---
        # Generating ALL 3-step chains is 11*11*11 = 1331 episodes. Too many!
        # Let's just generate 50 random 3-step chains to test long-term stability.
        long_chains = []
        for _ in range(50):
            long_chains.append([random.choice(primitives) for _ in range(3)])

        # --- BUILD THE MASTER PLAN ---
        # How many times to repeat each unique transition?
        # 10 times per pair = 1210 episodes. This is a very good dataset size.
        repeats_per_pair = 7 
        repeats_per_isolated = 15 # Practice basics more
        transition_speeds = np.linspace(30, 60, repeats_per_pair) 
        isolated_speeds = np.linspace(30, 60, repeats_per_isolated)
        #cruise_speed_kph = random.uniform(30, 60) 
        run_out_dist = random.uniform(50.0, 200.0)
        isolated_straight = random.uniform(150,200.0)
        full_episode_plan = []
        
        # 1. Add Transitions
        for seq in transition_curriculum:
            for spd in transition_speeds:
                full_episode_plan.append((seq, float(spd)))
            
        # 2. Add Isolated
        for seq in isolated_curriculum:
            for spd in isolated_speeds:
                full_episode_plan.append((seq, float(spd)))
            
        # 3. Add Long Chains
        for seq in long_chains:
            full_episode_plan.append((seq, float(random.uniform(30, 60))))
        
        
        # Shuffle to ensure diverse batches during training
        random.shuffle(full_episode_plan)
        
        print(f"Total Episodes Planned: {len(full_episode_plan)}")
        print(f"Unique Transitions Covered: {len(transition_curriculum)}")
        
        # --- EXECUTE ---
        self.visualizer = PygameVisualizer(window_size=(1000, 500))
        for episode, (sequence, cruise_speed_kph) in enumerate(full_episode_plan):
            transition_chain_value_map={
                'straight': [random.uniform(30.0, 60.0)],
                'turn_left': [random.uniform(15, 30)],
                'turn_right': [random.uniform(15, 30)],
                'hairpin_left': [random.uniform(10, 15)],
                'hairpin_right': [random.uniform(10, 15)],
                'lane_change_left': [random.uniform(2,10), random.uniform(30, 80)], # (start_y, target_y, length)
                'lane_change_right': [random.uniform(2,10), random.uniform(30, 80)],
                'chicane_left': [random.uniform(2.0, 4.0), random.uniform(30.0, 50.0)], # (width, length)
                'chicane_right': [random.uniform(2.0, 4.0), random.uniform(30.0, 50.0)],
                's_curve_left': [random.uniform(20, 40)],
                's_curve_right': [random.uniform(20, 40)]
            }
            
            if len(sequence)==1 and sequence[0]=='straight':
                # For isolated straights, we can randomize the length more to get more speed diversity in the dataset
                transition_chain_value_map['straight'] = [isolated_straight]

            print(f"Ep {episode}: Generating a '{sequence}' at {cruise_speed_kph:.1f} kph")
            ghost_path = generate_specific_chain(cruise_speed_kph, sequence, transition_chain_value_map,run_up=30.0, run_out=run_out_dist)
            utils = ControllerUtils(data=ghost_path, lookahead_dist=25.0) # For error calculations and lookahead features
            # 3. Setup Controller
            controller = PathFollower(direct_data=ghost_path)
            
            bp = self.bp_lib.filter('model3')[0]
            # Spawn at x=10, y=-1.75 (Lane -1 Center), z=0.5
            
            # Introduce random spawn offsets
            y_offset = random.uniform(-2.5, 2.5)  # Spawn up to 2.5m off center
            yaw_offset = random.uniform(-30, 30) # Spawn with up to 30 degrees of heading error
            ideal_start_x = ghost_path[0][0] # This will now be 1000.0
            ideal_start_y = ghost_path[0][1] # This will now be -1.75
            start_transform = carla.Transform(
                carla.Location(x=ideal_start_x, y=ideal_start_y+y_offset, z=2.0),
                carla.Rotation(yaw=yaw_offset)
            )
            # 4. Spawn Vehicle 
            self.vehicle = self.world.try_spawn_actor(bp, start_transform)
            if not self.vehicle:
                print("Spawn failed, retrying...")
                continue
            for _ in range(20): 
                self.world.tick()
            v_loc = self.vehicle.get_transform().location
            start_dist = np.linalg.norm(ghost_path[0][:2] - np.array([v_loc.x, v_loc.y]))
            end_dist = np.linalg.norm(ghost_path[-1][:2] - np.array([v_loc.x, v_loc.y]))
            print(f"   Spawn Checks: Dist to Start={start_dist:.2f}m, Dist to End={end_dist:.2f}m")
            print(f"   Initial Spawn Location: x={v_loc.x:.2f}, y={v_loc.y:.2f}, z={v_loc.z:.2f}")
            print(f"   Initial ghost start point: x={ghost_path[0][0]:.2f}, y={ghost_path[0][1]:.2f}")
            start_dist = np.linalg.norm(ghost_path[0][:2] - np.array([v_loc.x, v_loc.y]))
            if start_dist > 10.0:
                print(f"   >>> ERROR: Car is {start_dist:.1f}m away from path start! Skipping...")
                self.vehicle.destroy()
                break
            # Settle
            for _ in range(10): self.world.tick()
            spectator = self.world.get_spectator()
            self.set_spectator_pitch_up(carla.Transform(carla.Location(x=ideal_start_x, y=ideal_start_y, z=0.5)), spectator, distance=50.0, pitch=-30.0)
            
            episode_data = []
            self.visualizer.set_path(ghost_path)
            try:
                while True:
                    # Current Physics State
                    v_trans = self.vehicle.get_transform()
                    remaining_points = len(ghost_path) - controller.last_closest_idx
                    if len(episode_data) > 20 and speed_ms < 0.1:
                        print("   -> End Reason: Car Stuck (Speed ~ 0)")
                        break
                    if v_trans.location.z < -1.0:
                        print("   -> End Reason: Car fell through map")
                        break
                    vel = self.vehicle.get_velocity()
                    speed_ms = math.sqrt(vel.x**2 + vel.y**2)
                    # Yaw Rate (rad/s)
                    ang_vel = self.vehicle.get_angular_velocity()
                    yaw_rate = math.radians(ang_vel.z) 
                    
                    # Lateral Acceleration (m/s^2)
                    # We can use the formula: a_lat = v * yaw_rate
                    # Or get it from IMU (acc.y in local frame). Calculation is cleaner for now.
                    lat_accel = speed_ms * yaw_rate 

                    #cornering_limit_ms = controller.get_curvature_based_speed(
                    #    v_trans.location, 
                    #    max_lat_accel=100.0 # Limit lateral Gs to ~0.5g
                    #)
                    speed_error = cruise_speed_kph/3.6 - speed_ms

                    # The actual target is the minimum of "Cruise Setpoint" and "Physics Limit"
                    # final_target_ms = min(cruise_speed_kph/3.6, cornering_limit_ms)
                    final_target_ms = cruise_speed_kph/3.6 # For data diversity, we can keep the cruise speed as the target and let the controller learn to slow down for corners based on the path curvature features.

                    # End Condition: Speed too high or Path Finished
                    if controller.last_closest_idx >= len(ghost_path) - 7:
                        print(f"   -> End Reason: Path Finished (Points left: {len(ghost_path) - controller.last_closest_idx}), Ep {episode}: {'->'.join(sequence)}")
                        # stop the iteration and save the data, but we don't want to break here because we want to ensure the finally block runs to save the episode data. So we can use a flag or just let it naturally exit the loop on the next iteration when it checks the condition again. For simplicity, we'll just let it exit naturally.
                        break

                        
                    # --- GET INPUTS (Features) ---
                    cte, he, fut_cte, sp, sp_err, last_closest_idx, future_path_curvature, fut_yaw = utils.calculate_relative_errors(self.vehicle)
                    
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
                    flat_waypoints = utils.get_local_waypoints_dynamic(
                        vehicle=self.vehicle,
                        closest_idx=last_closest_idx,      
                        num_points=10,
                        path_resolution=0.5                
                    )
                    # --- APPLY CONTROL ---
                    self.vehicle.apply_control(carla.VehicleControl(
                        throttle=float(thr), steer=float(steer_cmd), brake=float(brk)
                    ))
                    # keep the spectator above the vehicle, slightly behind and looking down
                    #self.set_spectator(v_trans, spectator)
                    # --- SAVE ROW ---
                    row = [episode, '->'.join(sequence), 
                           speed_ms, speed_error, cte, he, fut_cte, 
                           yaw_rate, lat_accel,future_path_curvature,fut_yaw,
                           steer_cmd, thr, brk]
                    row.extend(flat_waypoints)
                    episode_data.append(row)
                    self.visualizer.render(self.vehicle, final_target_ms, speed_ms)

                    #print(f"Ep {episode} | Target Spd: {final_target_ms:.1f} m/s | Speed: {speed_ms:.1f} m/s | CTE: {cte:.2f} m | HE: {math.degrees(he):.1f} deg | Fut CTE: {fut_cte:.2f} m | Yaw Rate: {math.degrees(yaw_rate):.1f} deg/s | Lat Accel: {lat_accel:.2f} m/s²")
                    self.world.tick()
            
            except Exception as e:
                print(f"Error: {e}")
                break
            finally:
                # Save and Destroy
                if len(episode_data) > MIN_STEPS and len(episode_data) > 0:
                    with open(OUTPUT_FILE_TRANSITION, 'a', newline='') as f:
                        csv.writer(f).writerows(episode_data)
                        print(f"")
                        if remaining_points < 10:
                            print(f"   -> End Reason: Path Finished (Points left: {remaining_points}), Ep {episode}: Saved {len(episode_data)} rows.")
                    
                else:
                    print(f"Ep {episode}, {'->'.join(sequence)}: DISCARDED (Too short: {len(episode_data)} rows). Check logs!")
                    break
                self.vehicle.destroy()
                # clean up memory
                del self.vehicle
                del controller
                del utils
                del episode_data
                del ghost_path


def generate_presentation_images(output_dir=current_dir.parent / "Map_Layouts" / "presentation_images"):
    """
    Generates and saves high-resolution images of individual driving maneuvers 
    and a sample transition chain for presentation purposes.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    speed_kph = 50.0
    run_out = 50.0
    primitives_for_2gram = [
        'turn_left', 'turn_right',
        'hairpin_left', 'hairpin_right',
        'lane_change_left', 'lane_change_right',
        'chicane_left', 'chicane_right',
        #'s_curve_left', 's_curve_right' 
    ]
    # all primitives including isolated ones (for self-loops in the transition matrix)
    primitives = primitives_for_2gram + ['straight', 's_curve_left', 's_curve_right'] 
    all_pairs = list(itertools.product(primitives_for_2gram, repeat=2))
    transition_curriculum = [list(pair) for pair in all_pairs]  
    transition_chain_value_map={
                'straight': [random.uniform(30.0, 60.0)],
                'turn_left': [random.uniform(15, 30)],
                'turn_right': [random.uniform(15, 30)],
                'hairpin_left': [random.uniform(10, 15)],
                'hairpin_right': [random.uniform(10, 15)],
                'lane_change_left': [random.uniform(2,10), random.uniform(30, 80)], # (start_y, target_y, length)
                'lane_change_right': [random.uniform(2,10), random.uniform(30, 80)],
                'chicane_left': [random.uniform(2.0, 4.0), random.uniform(30.0, 50.0)], # (width, length)
                'chicane_right': [random.uniform(2.0, 4.0), random.uniform(30.0, 50.0)],
                's_curve_left': [random.uniform(20, 40)],
                's_curve_right': [random.uniform(20, 40)]
            }  
    # 1. Define the unique paths you want to visualize
    #    Parameters are chosen to look aesthetically pleasing in a plot.
    paths = {
        "Straight": generate_straight_path(speed_kph, length=100.0),
        
        "90-Degree Turn (Left)": generate_90_degree_turn_path(
            speed_kph, turn_radius=20, direction='left', run_out=run_out
        ),
        "90-Degree Turn (Right)": generate_90_degree_turn_path(
            speed_kph, turn_radius=20, direction='right', run_out=run_out
        ),
        "Hairpin Turn (Right)": generate_hairpin_turn_path(
            speed_kph, turn_radius=12, direction='right', run_out=run_out
        ),
         "Hairpin Turn (Left)": generate_hairpin_turn_path(
            speed_kph, turn_radius=12, direction='left', run_out=run_out
        ),
        "S-Curve (Left)": generate_s_curve_path(
            speed_kph, turn_radius=25, direction='left', run_out=run_out
        ),
        "S-Curve (Right)": generate_s_curve_path(
            speed_kph, turn_radius=25, direction='right', run_out=run_out
        ),
        "Lane Change (Left to Right)": generate_lane_change_path(
            speed_kph, lc_length=50, start_y=-1.75, target_y=1.75, run_out=run_out
        ),
        "Lane Change (Left to Right)": generate_lane_change_path(
            speed_kph, lc_length=50, start_y=-1.75, target_y=1.75, run_out=run_out
        ),
        "Lane Change (Right to Left)": generate_lane_change_path(
            speed_kph, lc_length=50, start_y=1.75, target_y=-1.75, run_out=run_out
        ),
        "Chicane (Left)": generate_chicane_path(
            speed_kph, width=3.5, length=40.0, direction='left', run_out=run_out
        ),
        "Chicane (Right)": generate_chicane_path(
            speed_kph, width=3.5, length=40.0, direction='right', run_out=run_out
        )
    }

    # Optional: Generate one complex transition chain image
    chain_sequence = ['s_curve_left', 'turn_right', 'lane_change_left']
    chain_map = {
        'straight': [50.0],
        'chicane_left': [3.0, 40.0], # width, length
        'turn_right': [20.0]         # radius
    }
    paths['_'.join(chain_sequence)] = generate_specific_chain(
        speed_kph, chain_sequence, transition_chain_value_map, run_up=30.0, run_out=run_out
    )
    for seq in transition_curriculum:
        paths['_'.join(seq)] = generate_specific_chain(60,seq, transition_chain_value_map, run_up=30.0, run_out=50.0)
    
    print(f"Generating images in '{output_dir}/'...")

    # 2. Plot and save each maneuver
    for name, data in paths.items():
        # Your path generators likely return a list of lists/tuples: [[x, y, yaw, ...], ...]
        data_np = np.array(data)
        x = data_np[:, 0]
        y = data_np[:, 1]
        
        # Setup Figure suitable for slides (16:9-ish ratio)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Path
        ax.plot(x, y, color='#0052cc', linewidth=3.5, label='Reference Path', zorder=2)
        
        # Plot Start and End points
        ax.scatter([x[0]], [y[0]], color='#009900', s=150, zorder=3, label="Start", edgecolors='black')
        ax.scatter([x[-1]], [y[-1]], color='#cc0000', s=150, zorder=3, label="End", edgecolors='black')

        # Formatting
        ax.set_title(f"Maneuver: {name}", fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel("X Position (meters)", fontsize=14)
        ax.set_ylabel("Y Position (meters)", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add a subtle grid
        ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
        
        # Legend
        ax.legend(fontsize=12, loc='best')
        
        # CRITICAL: Equal aspect ratio ensures turns are circular, not elliptical
        ax.set_aspect('equal', adjustable='datalim')
        
        # Save high quality
        safe_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") + ".png"
        filepath = os.path.join(output_dir, safe_filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" -> Saved {filepath}")

    print("All images generated successfully!")

if __name__ == '__main__':
    generate_presentation_images()
    
    #try:
    #    orch = FlatTrackOrchestrator()
    #    orch.run_transitions()
    #except KeyboardInterrupt:
    #    print("Cancelled by user")

# CarlaUE4.exe /Game/Maps/RaceTrack -windowed -carla-server -benchmark -fps=30        