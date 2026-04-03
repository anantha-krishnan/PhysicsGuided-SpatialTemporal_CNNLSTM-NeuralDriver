# lanechangeorchestrator.py
from pathlib import Path
import carla
import random
import numpy as np
import math
import csv
import os
import itertools
from controller import PathFollower 
from utility_fncs_train_inference import ControllerUtils
from minimap import PygameVisualizer

from path_generators import generate_specific_chain

# --- CONFIGURATION ---
current_file_path = Path(os.path.abspath(__file__))
current_dir = current_file_path.parent
xodr_file_path = current_dir.parent / "Map_Layouts" / "flattesttrack.xodr" 
OUTPUT_FILE_MASTER = current_dir.parent / "Map_Layouts" / "master_dataset.csv"
MIN_STEPS = 50  

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

    def set_spectator_pitch_up(self, target_transform, spectator, distance=120.0, pitch=-15.0):
        target_loc = target_transform.location
        pitch_rad = math.radians(abs(pitch))
        back_offset = distance * math.cos(pitch_rad)
        height_offset = distance * math.sin(pitch_rad)
        
        cam_loc = carla.Location(
            x=target_loc.x - back_offset, 
            y=target_loc.y,               
            z=target_loc.z + height_offset
        )
        cam_rot = carla.Rotation(pitch=pitch, yaw=0.0, roll=0.0)
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))
    
    def build_master_plan(self):
        """
        Builds the 1:1 Doubled Curriculum (~1326 total episodes).
        Every unique scenario is run once perfectly (Pristine) and once with errors (Recovery).
        """
        primitives_for_2gram = [
            'turn_left', 'turn_right', 'hairpin_left', 'hairpin_right',
            'lane_change_left', 'lane_change_right', 'chicane_left', 'chicane_right'
        ]
        all_primitives = primitives_for_2gram + ['straight', 's_curve_left', 's_curve_right'] 

        # 1. Define the core scenarios
        all_pairs = list(itertools.product(primitives_for_2gram, repeat=2)) # 64 Bigrams
        isolated_curriculum = [[p] for p in all_primitives]                 # 11 Primitives
        long_chains = [[random.choice(all_primitives) for _ in range(3)] for _ in range(50)] # 50 Trigrams

        repeats_per_pair = 7 
        repeats_per_isolated = 15 
        
        transition_speeds = np.linspace(30, 60, repeats_per_pair) 
        isolated_speeds = np.linspace(30, 60, repeats_per_isolated)

        base_scenarios = []
        
        # Add Transitions (Bigrams)
        for seq in all_pairs:
            for spd in transition_speeds:
                base_scenarios.append((list(seq), float(spd)))
            
        # Add Isolated (1-grams)
        for seq in isolated_curriculum:
            for spd in isolated_speeds:
                base_scenarios.append((seq, float(spd)))
            
        # Add Long Chains (Trigrams)
        for seq in long_chains:
            base_scenarios.append((seq, float(random.uniform(30, 60))))

        print(f"Unique Scenarios Generated: {len(base_scenarios)}")

        # 2. Duplicate into 1:1 Pristine vs. Recovery Plan
        master_plan = []
        for seq, speed in base_scenarios:
            # RUN 1: Pristine (Zero Error)
            master_plan.append({
                'dataset_group': 'pristine',
                'sequence': seq,
                'speed': speed,
                'y_off': 0.0,
                'yaw_off': 0.0
            })
            
            # RUN 2: Recovery (Random Error Injection)
            master_plan.append({
                'dataset_group': 'recovery',
                'sequence': seq,
                'speed': speed,
                'y_off': random.uniform(-2.5, 2.5),
                'yaw_off': random.uniform(-30.0, 30.0)
            })

        random.shuffle(master_plan)
        return master_plan

    def run_master_generation(self):
        print("Starting Data Generation on Flat Track...")
        
        # 1. Define Master Headers
        header = [
            "episode_id", "dataset_group", "maneuver", 
            "target_speed_kph", "spawn_y_offset", "spawn_yaw_offset",
            "speed_input", "speed_error_input", "cte_input", "heading_error_input", 
            "future_cte_input", "yaw_rate_input", "lat_accel_input", 
            "future_path_curvature_input", "future_heading_error_input",
            "steer_cmd", "throttle_cmd", "brake_cmd"
        ]
        
        # Add Dynamic and Fixed Horizon Waypoint columns (100 total columns)
        for prefix in ['wp_dyn', 'wp_5m', 'wp_10m', 'wp_20m', 'wp_30m']:
            for i in range(10):
                header.extend([f"{prefix}_{i}_x", f"{prefix}_{i}_y"])

        with open(OUTPUT_FILE_MASTER, 'w', newline='') as f:
            csv.writer(f).writerow(header)
            print(f"Created new dataset file: {OUTPUT_FILE_MASTER}")

        master_plan = self.build_master_plan()
        print(f"Total Episodes Planned: {len(master_plan)} (Perfect 1:1 Ratio)")
        
        self.visualizer = PygameVisualizer(window_size=(1000, 500))
        run_out_dist = random.uniform(50.0, 200.0)
        isolated_straight_len = random.uniform(150, 200.0)

        spectator = self.world.get_spectator()

        for episode, config in enumerate(master_plan):
            dataset_group = config['dataset_group']
            sequence = config['sequence']
            cruise_speed_kph = config['speed']
            y_offset = config['y_off']
            yaw_offset = config['yaw_off']
            
            # Parameter Map for the path generator
            transition_chain_value_map = {
                'straight': [random.uniform(30.0, 60.0)],
                'turn_left': [random.uniform(15, 30)], 'turn_right': [random.uniform(15, 30)],
                'hairpin_left': [random.uniform(10, 15)], 'hairpin_right': [random.uniform(10, 15)],
                'lane_change_left': [random.uniform(2,10), random.uniform(30, 80)], 
                'lane_change_right': [random.uniform(2,10), random.uniform(30, 80)],
                'chicane_left': [random.uniform(2.0, 4.0), random.uniform(30.0, 50.0)], 
                'chicane_right': [random.uniform(2.0, 4.0), random.uniform(30.0, 50.0)],
                's_curve_left': [random.uniform(20, 40)], 's_curve_right': [random.uniform(20, 40)]
            }
            
            if len(sequence) == 1 and sequence[0] == 'straight':
                transition_chain_value_map['straight'] = [isolated_straight_len]

            print(f"Ep {episode}/{len(master_plan)} [{dataset_group.upper()}]: {'->'.join(sequence)} @ {cruise_speed_kph:.1f} kph")
            
            ghost_path = generate_specific_chain(cruise_speed_kph, sequence, transition_chain_value_map, run_up=30.0, run_out=run_out_dist)
            utils = ControllerUtils(data=ghost_path, lookahead_dist=25.0) 
            controller = PathFollower(direct_data=ghost_path)
            
            bp = self.bp_lib.filter('model3')[0]
            
            ideal_start_x = ghost_path[0][0] 
            ideal_start_y = ghost_path[0][1] 
            
            # INJECT OFFSETS HERE
            start_transform = carla.Transform(
                carla.Location(x=ideal_start_x, y=ideal_start_y + y_offset, z=2.0),
                carla.Rotation(yaw=yaw_offset)
            )
            
            self.vehicle = self.world.try_spawn_actor(bp, start_transform)
            if not self.vehicle:
                print("Spawn failed, retrying...")
                continue
                
            for _ in range(20): self.world.tick()
            
            v_loc = self.vehicle.get_transform().location
            start_dist = np.linalg.norm(ghost_path[0][:2] - np.array([v_loc.x, v_loc.y]))
            if start_dist > 10.0:
                print(f"   >>> ERROR: Car is {start_dist:.1f}m away from path start! Skipping...")
                self.vehicle.destroy()
                continue
                
            for _ in range(10): self.world.tick()
            self.set_spectator_pitch_up(carla.Transform(carla.Location(x=ideal_start_x, y=ideal_start_y, z=0.5)), spectator, distance=50.0, pitch=-30.0)
            
            episode_data = []
            self.visualizer.set_path(ghost_path)
            
            try:
                while True:
                    v_trans = self.vehicle.get_transform()
                    remaining_points = len(ghost_path) - controller.last_closest_idx
                    
                    vel = self.vehicle.get_velocity()
                    speed_ms = math.sqrt(vel.x**2 + vel.y**2)
                    
                    if len(episode_data) > 20 and speed_ms < 0.1:
                        print("   -> End Reason: Car Stuck")
                        break
                    if v_trans.location.z < -1.0:
                        break
                        
                    yaw_rate = math.radians(self.vehicle.get_angular_velocity().z) 
                    lat_accel = speed_ms * yaw_rate 
                    speed_error = (cruise_speed_kph/3.6) - speed_ms
                    final_target_ms = cruise_speed_kph/3.6 

                    if controller.last_closest_idx >= len(ghost_path) - 7:
                        print(f"   -> End Reason: Path Finished")
                        break

                    # --- GET BASE INPUTS ---
                    cte, he, fut_cte, sp, sp_err, last_closest_idx, future_path_curvature, fut_yaw = utils.calculate_relative_errors(self.vehicle)
                    
                    steer_rad = controller.get_pure_pursuit_steering(
                        v_trans.location.x, v_trans.location.y, math.radians(v_trans.rotation.yaw), speed_ms
                    )
                    steer_cmd = np.clip(steer_rad / 1.22, -1.0, 1.0) 
                    
                    sim_time = self.world.get_snapshot().timestamp.elapsed_seconds
                    thr, brk = controller.get_long_vel(speed_ms, final_target_ms, sim_time)
                    
                    # --- GET DYNAMIC AND FIXED HORIZONS ---
                    wp_dyn = utils.get_local_waypoints_dynamic(self.vehicle)
                    wp_5m  = utils.get_local_waypoints_dynamic(self.vehicle, lookahead_dis=5.0)
                    wp_10m = utils.get_local_waypoints_dynamic(self.vehicle, lookahead_dis=10.0)
                    wp_20m = utils.get_local_waypoints_dynamic(self.vehicle, lookahead_dis=20.0)
                    wp_30m = utils.get_local_waypoints_dynamic(self.vehicle, lookahead_dis=30.0)
                    
                    self.vehicle.apply_control(carla.VehicleControl(
                        throttle=float(thr), steer=float(steer_cmd), brake=float(brk)
                    ))
                    
                    # --- BUILD ROW ---
                    row = [
                        episode, dataset_group, '-'.join(sequence), 
                        cruise_speed_kph, y_offset, yaw_offset,
                        speed_ms, speed_error, cte, he, fut_cte, 
                        yaw_rate, lat_accel, future_path_curvature, fut_yaw,
                        steer_cmd, thr, brk
                    ]
                    
                    # Append all 5 spatial horizons
                    row.extend(wp_dyn)
                    row.extend(wp_5m)
                    row.extend(wp_10m)
                    row.extend(wp_20m)
                    row.extend(wp_30m)
                    
                    episode_data.append(row)
                    self.visualizer.render(self.vehicle, final_target_ms, speed_ms, cte=cte)
                    self.world.tick()
            
            except Exception as e:
                print(f"Error: {e}")
                break
            finally:
                if len(episode_data) > MIN_STEPS and len(episode_data) > 0:
                    with open(OUTPUT_FILE_MASTER, 'a', newline='') as f:
                        csv.writer(f).writerows(episode_data)
                else:
                    print(f"   -> DISCARDED (Too short)")
                
                self.vehicle.destroy()

        self.visualizer.destroy()
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)
        print("Done.")

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
    #generate_presentation_images()
    
    try:
        orch = FlatTrackOrchestrator()
        orch.run_master_generation()
    except KeyboardInterrupt:
        print("Cancelled by user")
# cd cd D:\Workspaces\github\NeuralDriver\carla_0_9\CARLA_0.9.15\WindowsNoEditor\
# CarlaUE4.exe /Game/Maps/RaceTrack -windowed -carla-server -benchmark -fps=30        
