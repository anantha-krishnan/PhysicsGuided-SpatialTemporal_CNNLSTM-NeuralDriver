import carla
import matplotlib.pyplot as plt
import os
import csv
import glob
import sys
import time
import math
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from pathlib import Path
from controller import PathFollower
try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure it is installed')


output_folder = "D:/PaperWork/personal/AI/LLM_Engg_GenAI_Rag_Lora_Agent/personal_works/MS/Driver/carla_0_9/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla-python-examples-main/Map_Layouts"

def list_maps():
    """
    Connects to a running CARLA server and prints a list of available maps.
    """
    try:
        # Connect to the server
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0) # Set a timeout to avoid waiting forever

        # Get the list of available map names
        available_maps = client.get_available_maps()

        print("Success! Connected to the CARLA server.")
        print("The following maps are available in this version:")
        for map_name in sorted(available_maps):
            print(f"- {map_name}")

    except Exception as e:
        print(f"Error: Could not connect to the CARLA server.")
        print(f"Please make sure the CARLA simulator is running.")
        print(f"Details: {e}")

# --- CONFIGURATION ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MAP_SIZE = 200        # Pixel size of the mini-map (square)
MAP_OPACITY = 200     # 0-255 (255 is solid, <255 is transparent)
STEER_SPEED = 0.01      # How fast the wheel turns (0.01 = slow, 0.1 = fast)
STEER_RETURN = 0.05     # How fast the wheel returns to center when you let go
MAX_STEER = 0.5         # Maximum steering angle
THROTTLE_SPEED = 0.02    # How fast gas pedal goes down (approx 1 sec to full)
THROTTLE_RETURN = 0.05   # How fast gas pedal releases
BRAKE_SPEED = 0.05       # Brakes apply faster than gas
BRAKE_RETURN = 0.1       # Brakes release quickly

def parse_image(display, image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))

def gen_centerline():
    """
    Connects to CARLA, generates all waypoints for the current map,
    and plots them using matplotlib.
    """
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Get current world (whatever you loaded manually)
        world = client.get_world()
        carla_map = world.get_map()
        map_name = carla_map.name.split('/')[-1] # e.g. "Town04"
        
        print(f"Generating map for: {map_name}")

        # 2. Generate Waypoints (approx every 2 meters)
        # We filter for Driving lanes to avoid parking spots and sidewalks
        waypoints = carla_map.generate_waypoints(distance=2.0)
        drivable_waypoints = [w for w in waypoints if w.lane_type == carla.LaneType.Driving]

        x_coords = [w.transform.location.x for w in drivable_waypoints]
        y_coords = [w.transform.location.y for w in drivable_waypoints]

        # 3. Plotting
        plt.figure(figsize=(12, 12)) # Make it big
        
        # s=1 makes the dots small enough to look like lines
        plt.scatter(x_coords, y_coords, s=1, c='blue') 
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Waypoint Map of {map_name}")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.grid(True)

        # 4. Save the file
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        filename = f"{output_folder}\\{map_name}.png"
        
        # *** SAVE BEFORE SHOWING ***
        print(f"Saving to {filename}...")
        plt.savefig(filename, dpi=150) 
        
        # 5. Show (Optional - you can remove this if you just want to save)
        print("Displaying plot...")
        plt.show() 
        plt.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the CARLA simulator is running.")


def gen_all_centerlines():
    # 1. Setup
    output_folder = "D:\\PaperWork\\personal\\AI\\LLM_Engg_GenAI_Rag_Lora_Agent\\personal_works\\MS\\Driver\\carla_0_9\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla-python-examples-main\\Map_Layouts"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        # 2. Connect to Client
        print("Connecting to CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0) # High timeout because loading maps takes time
        
        # 3. Get all available maps
        available_maps = client.get_available_maps()
        # Filter mostly for "Town" maps to avoid test/base maps that might crash or be empty
        town_maps = [m for m in available_maps if "Town" in m]
        
        print(f"Found {len(town_maps)} maps. Starting generation...\n")

        # 4. Loop through each map
        for map_name in sorted(town_maps):
            
            clean_name = map_name.split('/')[-1] # e.g., "Town04"
            print(f"--- Processing: {clean_name} ---")

            try:
                # Load the map
                print("   Loading world (this may take a few seconds)...")
                client.load_world(map_name)
                
                # Get the map object
                world = client.get_world()
                carla_map = world.get_map()

                # 5. Generate Waypoints (Center of Driving Lanes)
                # distance=2.0 means one point every 2 meters
                print("   Generating waypoints...")
                waypoints = carla_map.generate_waypoints(distance=2.0)

                # Filter for DRIVING lanes only (removes sidewalks, etc)
                # This ensures we get the 'center line' of the road.
                drivable_waypoints = [
                    w for w in waypoints 
                    if w.lane_type == carla.LaneType.Driving
                ]

                if not drivable_waypoints:
                    print("   Warning: No drivable waypoints found. Skipping plot.")
                    continue

                # Extract X and Y coordinates
                x_coords = [w.transform.location.x for w in drivable_waypoints]
                y_coords = [w.transform.location.y for w in drivable_waypoints]

                # 6. Plotting
                print(f"   Plotting {len(drivable_waypoints)} points...")
                plt.figure(figsize=(10, 10))
                
                # s=0.5 makes the dots very small to look like a line
                plt.scatter(x_coords, y_coords, s=0.5, c='black') 
                
                plt.title(f"Road Network: {clean_name}")
                plt.xlabel("X (meters)")
                plt.ylabel("Y (meters)")
                plt.axis('equal') # Keeps the aspect ratio correct
                plt.grid(True, alpha=0.3)

                # Save the figure
                filename = f"{output_folder}/{clean_name}.png"
                plt.savefig(filename, dpi=150)
                plt.close() # Close memory to prevent crashing on loop
                
                print(f"   Saved to: {filename}")

            except Exception as e:
                print(f"   ERROR processing {map_name}: {e}")

        print("\nAll Done! Check the 'Map_Layouts' folder.")

    except Exception as e:
        print(f"Fatal Error: {e}")
        print("Ensure CARLA is running.")

def smoothen_way_points(waypoints=None, input_csv="new_waypoints.txt", output_csv="new_waypoints_Processed.txt"):
    """
    Update the waypoints to match resolution of 0.5m as used in training data.    
    Uses B Splines to create a smooth path through the original waypoints, then samples new points every 0.5m along the curve.
    """
    INPUT_CSV = input_csv  # Your current file
    OUTPUT_CSV = output_csv # The file to use in neural_driver.py
    TARGET_RESOLUTION = 0.5
    print(f"Loading {INPUT_CSV}...")
    # Load data (Assuming no header, or skiprows if needed)
    # Adjust column names based on your file structure
    # Based on your snippet: x, y, z, speed
    if waypoints is None:
        try:
            df = pd.read_csv(INPUT_CSV)
            # rename columns to standard names
            df.columns = ['x', 'y', 'z', 'speed']
        except:
            print(f"Error loading {INPUT_CSV}. Ensure it exists and has the correct format.")
            return
    else:
        df = pd.DataFrame(waypoints, columns=['x', 'y', 'z', 'speed'])

    raw_len = len(df)
    print(f"Original points: {raw_len}")

    # 1. PRE-FILTERING: Remove points that are too close (car stopped or barely moving)
    # This prevents the Spline from going crazy with knotted loops
    clean_data = []
    prev_x, prev_y = -99999, -99999
    
    for i in range(len(df)):
        curr_x = df.iloc[i]['x']
        curr_y = df.iloc[i]['y']
        print(f"Processing point {i+1}/{len(df)}: ({curr_x}, {curr_y}), prev: ({prev_x}, {prev_y})")
        dist = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        
        # Keep point only if it moved at least 0.1m from the last kept point
        if dist > 0.1: 
            clean_data.append(df.iloc[i].values)
            prev_x, prev_y = curr_x, curr_y
            
    clean_data = np.array(clean_data)
    print(f"Points after distance filtering: {len(clean_data)}")

    if len(clean_data) < 10:
        print("Error: Not enough points remaining to interpolate!")
        return

    # Extract arrays
    x = clean_data[:, 0]
    y = clean_data[:, 1]
    z = clean_data[:, 2]
    speed = clean_data[:, 3]

    # 2. FIT SPLINE
    # s is the smoothing factor. 
    # s=0.0 means "pass through every point" (noisy).
    # s=5.0 allows the spline to ignore small jitters (smooth).
    tck, u = splprep([x, y, z], s=2.0) 

    # 3. RESAMPLE AT 0.5 METERS
    # We first generate a fine line to measure distance
    u_fine = np.linspace(0, 1, len(x) * 10)
    x_fine, y_fine, z_fine = splev(u_fine, tck)
    
    # Calculate cumulative distance along the fine path
    dist_steps = np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2)
    cum_dist = np.insert(np.cumsum(dist_steps), 0, 0)
    total_length = cum_dist[-1]
    
    print(f"Total Track Length: {total_length:.2f} meters")
    
    # Create target distances: 0, 0.5, 1.0, 1.5 ...
    target_distances = np.arange(0, total_length, TARGET_RESOLUTION)
    
    # Interpolate X, Y, Z at these exact distances
    x_final = np.interp(target_distances, cum_dist, x_fine)
    y_final = np.interp(target_distances, cum_dist, y_fine)
    z_final = np.interp(target_distances, cum_dist, z_fine)
    
    # Interpolate Speed (Linear) - Spline for speed can overshoot to negative values
    # We map the original speeds to the new distances
    # Estimate distance of original cleaned points
    orig_dist_steps = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    orig_cum_dist = np.insert(np.cumsum(orig_dist_steps), 0, 0)
    
    speed_final = np.interp(target_distances, orig_cum_dist, speed)

    # 4. SAVE
    output_data = np.column_stack((x_final, y_final, z_final, speed_final))
    
    # Save with header
    np.savetxt(OUTPUT_CSV, output_data, delimiter=',', header="x,y,z,speed", comments='')
    print(f"Saved processed route to {OUTPUT_CSV}")
    print(f"Final point count: {len(output_data)}")

    # Optional: Plot to check smoothness
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'r.', label='Original (Filtered)', markersize=2)
    plt.plot(x_final, y_final, 'b-', label='Splined & Resampled', linewidth=1)
    plt.legend()
    plt.title("Path Processing Result")
    plt.axis('equal')
    plt.show()

def generate_way_points():
    pygame.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Smooth Driving (Steer + Gas) + Mini-Map")
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0) 

    actor_list = []

    try:
        world = client.get_world()
        carla_map = world.get_map()
        
        # --- MAP GENERATION ---
        print("Generating Map Texture...")
        waypoints = carla_map.generate_waypoints(2.0)
        map_x = [w.transform.location.x for w in waypoints]
        map_y = [w.transform.location.y for w in waypoints]
        min_x, max_x = min(map_x), max(map_x)
        min_y, max_y = min(map_y), max(map_y)
        world_width = max_x - min_x
        world_height = max_y - min_y
        scale = MAP_SIZE / max(world_width, world_height)
        
        map_surface = pygame.Surface((MAP_SIZE, MAP_SIZE))
        map_surface.set_alpha(MAP_OPACITY) 
        map_surface.fill((0, 0, 0))
        for x, y in zip(map_x, map_y):
            px = int((x - min_x) * scale)
            py = int((y - min_y) * scale)
            pygame.draw.circle(map_surface, (150, 150, 150), (px, py), 1)
        print("Map generated!")

        # --- SPAWN ACTORS ---
        bp = world.get_blueprint_library().filter('model3')[0]
        spawn_point = carla_map.get_spawn_points()[0]
        vehicle = world.spawn_actor(bp, spawn_point)
        actor_list.append(vehicle)

        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('fov', '100')
        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-15))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        camera.listen(lambda image: parse_image(display, image))

        # --- MAIN LOOP ---
        clock = pygame.time.Clock()
        control = carla.VehicleControl()
        recorded_points = []
        
        # Initialize physics states
        current_steer = 0.0
        current_throttle = 0.0
        current_brake = 0.0
        reverse_gear = False
        r_pressed_last_frame = False

        print("Controls: WASD (Smooth). 'R' to toggle Reverse.")

        while True:
            clock.tick_busy_loop(60)
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == K_ESCAPE):
                    return

            # --- 1. STEERING LOGIC ---
            if keys[K_a]:
                current_steer -= STEER_SPEED
            elif keys[K_d]:
                current_steer += STEER_SPEED
            else:
                if current_steer > 0:
                    current_steer -= STEER_RETURN
                    if current_steer < 0: current_steer = 0
                elif current_steer < 0:
                    current_steer += STEER_RETURN
                    if current_steer > 0: current_steer = 0
            
            # Clamp Steering
            current_steer = max(-MAX_STEER, min(MAX_STEER, current_steer))

            # --- 2. THROTTLE LOGIC ---
            if keys[K_w]:
                current_throttle += THROTTLE_SPEED
                current_brake = 0.0 # Release brake if pressing gas
            else:
                current_throttle -= THROTTLE_RETURN
            
            # Clamp Throttle
            current_throttle = max(0.0, min(0.7, current_throttle))

            # --- 3. BRAKE LOGIC ---
            if keys[K_s]:
                current_brake += BRAKE_SPEED
                current_throttle = 0.0 # Cut gas if braking
            else:
                current_brake -= BRAKE_RETURN
            
            # Clamp Brake
            current_brake = max(0.0, min(1.0, current_brake))

            # --- 4. REVERSE LOGIC ---
            

            # --- APPLY CONTROLS ---
            control.steer = current_steer
            control.throttle = current_throttle
            control.brake = current_brake
            
            control.hand_brake = False
            vehicle.apply_control(control)

            # --- DATA RECORDING ---
            loc = vehicle.get_location()
            vel = vehicle.get_velocity()
            speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            recorded_points.append([loc.x, loc.y, loc.z, speed_ms])

            # --- DRAWING HUD ---
            # 1. Mini-Map
            car_px = int((loc.x - min_x) * scale)
            car_py = int((loc.y - min_y) * scale)
            display.blit(map_surface, (10, 10))
            pygame.draw.rect(display, (255, 255, 255), (10, 10, MAP_SIZE, MAP_SIZE), 2)
            pygame.draw.circle(display, (255, 0, 0), (10 + car_px, 10 + car_py), 3)

            # 2. Steering Bar (Bottom Center - Yellow)
            bar_width = 200
            pygame.draw.rect(display, (50, 50, 50), (WINDOW_WIDTH//2 - bar_width//2, WINDOW_HEIGHT - 30, bar_width, 10))
            indicator_pos = (WINDOW_WIDTH//2) + (current_steer * (bar_width//2))
            pygame.draw.circle(display, (255, 255, 0), (int(indicator_pos), WINDOW_HEIGHT - 25), 8)

            # 3. Throttle Bar (Right Side - Green)
            # Box height 150px
            t_bar_h = 150
            t_fill = int(current_throttle * t_bar_h)
            pygame.draw.rect(display, (50, 50, 50), (WINDOW_WIDTH - 40, WINDOW_HEIGHT - 200, 20, t_bar_h)) # Back
            pygame.draw.rect(display, (0, 255, 0), (WINDOW_WIDTH - 40, WINDOW_HEIGHT - 200 + (t_bar_h - t_fill), 20, t_fill)) # Fill
            
            # 4. Brake Bar (Right Side - Red)
            b_fill = int(current_brake * t_bar_h)
            pygame.draw.rect(display, (50, 50, 50), (WINDOW_WIDTH - 70, WINDOW_HEIGHT - 200, 20, t_bar_h)) # Back
            pygame.draw.rect(display, (255, 0, 0), (WINDOW_WIDTH - 70, WINDOW_HEIGHT - 200 + (t_bar_h - b_fill), 20, b_fill)) # Fill

            # 5. Gear Indicator
            if reverse_gear:
                label = pygame.font.SysFont('monospace', 30).render("R", True, (255, 0, 0))
                display.blit(label, (WINDOW_WIDTH - 55, WINDOW_HEIGHT - 230))
            else:
                label = pygame.font.SysFont('monospace', 30).render("D", True, (255, 255, 255))
                display.blit(label, (WINDOW_WIDTH - 55, WINDOW_HEIGHT - 230))

            pygame.display.flip()

    finally:
        if len(recorded_points) > 0:
            with open(os.path.join(output_folder, "new_waypoints.txt"), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["x", "y", "z", "speed_mps"])
                writer.writerows(recorded_points)
            print(f"Saved {len(recorded_points)} points.")

        for actor in actor_list:
            actor.destroy()
        pygame.quit()

def create_empty_flat_map():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # This XML string defines a standard OpenDRIVE road
    # It creates a straight road: 
    # - Length: 2000 meters
    # - Width: 3 lanes left, 3 lanes right (approx 20m total width)
    # - Geometry: Perfectly straight, flat (0 elevation)
    
    opendrive_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <OpenDRIVE>
        <header revMajor="1" revMinor="4" name="FlatTrack" version="1.00" date="Tue Feb 10 2026 12:00:00 GMT+0530 (India Standard Time)" north="0.0" south="0.0" east="0.0" west="0.0">
        </header>
        <road name="StraightRoad" length="2000.0" id="1" junction="-1">
            <link>
            </link>
            <planView>
                <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="2000.0">
                    <line/>
                </geometry>
            </planView>
            <elevationProfile>
                <elevation s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>
            </elevationProfile>
            <lateralProfile>
            </lateralProfile>
            <lanes>
                <laneSection s="0.0">
                    <left>
                        <lane id="1" type="driving" level="false">
                            <link>
                            </link>
                            <width sOffset="0.0" a="10.0" b="0.0" c="0.0" d="0.0"/>
                        </lane>
                        <lane id="2" type="driving" level="false">
                            <link>
                            </link>
                            <width sOffset="0.0" a="10.0" b="0.0" c="0.0" d="0.0"/>
                        </lane>
                    </left>
                    <center>
                        <lane id="0" type="none" level="false">
                            <link>
                            </link>
                        </lane>
                    </center>
                    <right>
                        <lane id="-1" type="driving" level="false">
                            <link>
                            </link>
                            <width sOffset="0.0" a="10.0" b="0.0" c="0.0" d="0.0"/>
                        </lane>
                        <lane id="-2" type="driving" level="false">
                            <link>
                            </link>
                            <width sOffset="0.0" a="10.0" b="0.0" c="0.0" d="0.0"/>
                        </lane>
                    </right>
                </laneSection>
            </lanes>
        </road>
    </OpenDRIVE>
    """

    print("Generating Empty Flat Map from OpenDRIVE...")
    
    # 2.0 is the vertex distance (resolution of the mesh)
    # The smaller the number, the smoother the road, but heavier on GPU
    client.generate_opendrive_world(opendrive_xml, parameters=carla.OpendriveGenerationParameters(
        vertex_distance=10.0,
        max_road_length=50.0,
        wall_height=1.0,
        additional_width=0.6,
        smooth_junctions=True,
        enable_mesh_visibility=True
    ))
    
    print("Map Loaded! You are now on a flat infinite void with a single 2km runway.")
    
    # OPTIONAL: Set weather to Clear Noon (Perfect visibility)
    world = client.get_world()
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    # Verify spawn points
    spawn_points = world.get_map().get_spawn_points()
    print(f"Map has {len(spawn_points)} spawn points.")

def load_custom_map(xodr_file="flattesttrack.xodr"):
    """
    Loads a custom OpenDRIVE map from a file.
    This is the MotionSolve equivalent of loading a .road file.
    """
    # get path of current file
    current_file_path = Path(os.path.abspath(__file__))
    current_dir = current_file_path.parent
    xodr_file_path = current_dir.parent / "Map_Layouts" / xodr_file

    client = carla.Client('localhost', 2000)
    if not xodr_file_path.exists():
        print(f"Error: Map file '{xodr_file}' not found.")
        return

    print(f"Loading map from {xodr_file}...")
    
    # Read the file content
    with open(xodr_file_path, 'r') as f:
        xodr_content = f.read()

    # Generate the world
    # vertex_distance=2.0 ensures smooth physics mesh
    try:
        client.generate_opendrive_world(xodr_content, parameters=carla.OpendriveGenerationParameters(
            vertex_distance=2.0,
            max_road_length=50.0,
            wall_height=0.5, 
            additional_width=0.6,
            smooth_junctions=True, 
            enable_mesh_visibility=True
        ))
        print("Map Loaded Successfully.")
    except Exception as e:
        print(f"Failed to load map: {e}")

    
def get_current_state(vehicle, world):
    snapshot = world.get_snapshot()
    timestamp = snapshot.timestamp.elapsed_seconds
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    x = location.x
    y = location.y
    yaw_radians = math.radians(rotation.yaw)
    vel = vehicle.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    return timestamp, x, y, yaw_radians, speed

def get_transform_for_spectator(vehicle_transform):
    forward_vector = vehicle_transform.get_forward_vector()
    location = vehicle_transform.location + carla.Location(
        x=-10.0 * forward_vector.x, 
        y=-10.0 * forward_vector.y, 
        z=5.0
    )
    rotation = carla.Rotation(pitch=-20.0, yaw=vehicle_transform.rotation.yaw)
    return carla.Transform(location, rotation)

def auto_driver():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    
    # --- LOAD MAP FILE ---
    # Ensure this points to your generated file
    CSV_FILE = os.path.join(output_folder, "new_waypoints.txt") 

    try:
        path_follower = PathFollower(CSV_FILE)
    except Exception as e:
        print(f"FAILED TO LOAD PATH: {e}")
        return

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    
    # Spawn at the start of the recorded path + small Z offset
    start_x, start_y = path_follower.waypoints_xy[0]
    spawn_point = carla.Transform(carla.Location(x=start_x, y=start_y, z=1.5))
    
    # Try spawning (if collision, fallback to map spawn)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        print("Could not spawn at waypoint start. Using default map spawn.")
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    spectator = world.get_spectator()
    print("Vehicle spawned. Starting Control Loop...")

    #debug = world.debug

    try:
        while True:
            # 1. Get State
            current_time, x, y, yaw, v = get_current_state(vehicle, world)            
            
            # 2. Get Pure Pursuit Steering (Now passing 'v' for dynamic lookahead)
            steer_radians = path_follower.get_pure_pursuit_steering(x, y, yaw, v)
            
            # 3. Convert Radians to [-1, 1] for Carla
            MAX_STEER_DEGREES = 70.0
            max_steer_radians = math.radians(MAX_STEER_DEGREES)
            steer_cmd = steer_radians / max_steer_radians
            steer_cmd = np.clip(steer_cmd, -1.0, 1.0)
            
            # 4. Throttle Logic (Simple P-Controller for speed)
            throttle_output, brake_output = path_follower.get_long_vel(v, path_follower.current_target_speed, current_time)

            # Apply Control
            control = carla.VehicleControl()
            control.steer = float(steer_cmd)
            control.throttle = float(throttle_output)
            control.brake = float(brake_output)
            vehicle.apply_control(control)
            
            # Update Camera
            spectator.set_transform(get_transform_for_spectator(vehicle.get_transform()))
            
            # Visualize the "Rear Axle" location used by the controller
            #rear_x = x - (WHEELBASE / 2.0) * np.cos(yaw)
            #rear_y = y - (WHEELBASE / 2.0) * np.sin(yaw)
            #debug.draw_string(carla.Location(x=rear_x, y=rear_y, z=2.0), 'O', 
            #                  draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.05)

            # Wait for tick (Keep physics synced)
            world.wait_for_tick()
            if path_follower.end_path:
                raise KeyboardInterrupt
                

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if vehicle:
            vehicle.destroy()
        print("Actors destroyed.")


if __name__ == '__main__':
    generate_way_points()
    smoothen_way_points(input_csv=os.path.join(output_folder, "new_waypoints.txt"), output_csv=os.path.join(output_folder, "new_waypoints_Processed.txt"))

# cd D:\PaperWork\personal\AI\LLM_Engg_GenAI_Rag_Lora_Agent\personal_works\MS\Driver\carla_0_9\CARLA_0.9.15\WindowsNoEditor\PythonAPI\util
# python config.py --map Town04