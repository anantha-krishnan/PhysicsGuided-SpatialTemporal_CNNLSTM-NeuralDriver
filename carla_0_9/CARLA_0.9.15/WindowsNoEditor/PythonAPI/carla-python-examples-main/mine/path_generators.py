import random
import numpy as np
import math

class PathFactory:
    """
    A class to procedurally generate a sequence of waypoints by chaining together
    path primitives using relative coordinates.
    """
    def __init__(self, start_x=1000.0, start_y=-1.75, start_yaw_deg=0.0, resolution=0.5):
        self.resolution = resolution
        self.points = []
        
        # State tracking
        self.current_x = start_x
        self.current_y = start_y
        self.current_yaw_rad = math.radians(start_yaw_deg)

    def _add_point(self, x, y, speed_ms):
        self.points.append([x, y, 0.0, speed_ms])

    def add_straight(self, length, speed_kph):
        speed_ms = speed_kph / 3.6
        n_points = int(length / self.resolution)
        
        cos_yaw = math.cos(self.current_yaw_rad)
        sin_yaw = math.sin(self.current_yaw_rad)

        for i in range(1, n_points + 1):
            dist = i * self.resolution
            x = self.current_x + dist * cos_yaw
            y = self.current_y + dist * sin_yaw
            self._add_point(x, y, speed_ms)
        
        # Update State
        self.current_x += length * cos_yaw
        self.current_y += length * sin_yaw

    def add_turn(self, radius, angle_deg, speed_kph, direction):
        if direction not in ['left', 'right']: raise ValueError("Invalid direction")
        speed_ms = speed_kph / 3.6
        
        # Standard Trig: Left turn increases angle, Right turn decreases
        turn_dir = 1.0 if direction == 'left' else -1.0
        angle_rad = math.radians(angle_deg)
        
        # Calculate Center of Rotation (90 deg offset from current heading)
        cx = self.current_x + radius * math.cos(self.current_yaw_rad + turn_dir * math.pi / 2)
        cy = self.current_y + radius * math.sin(self.current_yaw_rad + turn_dir * math.pi / 2)
        
        start_angle = self.current_yaw_rad - turn_dir * math.pi / 2
        arc_length = radius * angle_rad
        n_points = int(arc_length / self.resolution)
        
        final_x, final_y = self.current_x, self.current_y

        for i in range(1, n_points + 1):
            frac = i / n_points
            theta = start_angle + turn_dir * frac * angle_rad
            final_x = cx + radius * math.cos(theta)
            final_y = cy + radius * math.sin(theta)
            self._add_point(final_x, final_y, speed_ms)

        # Update State
        self.current_x = final_x
        self.current_y = final_y
        self.current_yaw_rad += turn_dir * angle_rad

    def add_lane_change(self, width, length, direction, speed_kph):
        """
        Adds a sigmoid path offset.
        Width: How far to move laterally (meters).
        Direction: 'left' (+Y relative) or 'right' (-Y relative).
        """
        speed_ms = speed_kph / 3.6
        n_points = int(length / self.resolution)
        
        cos_yaw = math.cos(self.current_yaw_rad)
        sin_yaw = math.sin(self.current_yaw_rad)
        
        # Lateral Direction: +1 for Left, -1 for Right (Standard Trig/Carla Logic)
        lat_dir = 1.0 if direction == 'left' else -1.0
        
        start_x, start_y = self.current_x, self.current_y
        final_x, final_y = start_x, start_y

        for i in range(1, n_points + 1):
            l_dist = i * self.resolution # Longitudinal distance
            p = l_dist / length
            
            # Sigmoid (0 to 1) -> Scale by Width -> Apply Direction
            lat_offset = width * (1 - math.cos(p * math.pi)) / 2.0
            lat_offset *= lat_dir
            
            # Rotate Local (Long, Lat) to Global
            # GlobalX = StartX + (Long * cos - Lat * sin)
            # GlobalY = StartY + (Long * sin + Lat * cos)
            final_x = start_x + (l_dist * cos_yaw - lat_offset * sin_yaw)
            final_y = start_y + (l_dist * sin_yaw + lat_offset * cos_yaw)
            
            self._add_point(final_x, final_y, speed_ms)
            
        self.current_x = final_x
        self.current_y = final_y
        # Yaw remains technically the same after a perfect parallel lane change

    def add_chicane(self, width, length, direction, speed_kph):
        speed_ms = speed_kph / 3.6
        n_points = int(length / self.resolution)
        
        cos_yaw = math.cos(self.current_yaw_rad)
        sin_yaw = math.sin(self.current_yaw_rad)
        lat_dir = 1.0 if direction == 'left' else -1.0
        
        start_x, start_y = self.current_x, self.current_y
        final_x, final_y = start_x, start_y

        for i in range(1, n_points + 1):
            l_dist = i * self.resolution
            p = i / n_points
            
            # Sine Wave
            lat_offset = width * math.sin(2 * math.pi * p) * lat_dir
            
            final_x = start_x + (l_dist * cos_yaw - lat_offset * sin_yaw)
            final_y = start_y + (l_dist * sin_yaw + lat_offset * cos_yaw)
            
            self._add_point(final_x, final_y, speed_ms)
            
        self.current_x = final_x
        self.current_y = final_y
    
    def get_path(self):
        return np.array(self.points)

def generate_specific_chain(speed_kph, maneuver_list, transition_chain_value_map, run_up=30.0, run_out=50.0):
    factory = PathFactory(start_x=1000.0, start_y=-1.75, start_yaw_deg=0.0)
    
    factory.add_straight(run_up, speed_kph)
    custome_speed_kph = speed_kph
    for move in maneuver_list:
        if move in ['straight', 'turn_left', 'turn_right', 'hairpin_left', 'hairpin_right', 's_curve_left', 's_curve_right'] and len(transition_chain_value_map[move])>1:
            custome_speed_kph = transition_chain_value_map[move][1] # reset to default for these moves unless overridden
        elif move in ['lane_change_left', 'lane_change_right', 'chicane_left', 'chicane_right'] and len(transition_chain_value_map[move])>2:
            custome_speed_kph = transition_chain_value_map[move][2] # reset to default for lane changes unless overridden
        if move == 'straight':
            factory.add_straight(transition_chain_value_map['straight'][0], custome_speed_kph)
        elif move == 'turn_left':
            factory.add_turn(transition_chain_value_map['turn_left'][0], 90.0, custome_speed_kph, 'left')
        elif move == 'turn_right':
            factory.add_turn(transition_chain_value_map['turn_right'][0], 90.0, custome_speed_kph, 'right')
        elif move == 'hairpin_left':
            factory.add_turn(transition_chain_value_map['hairpin_left'][0], 180.0, speed_kph, 'left')
        elif move == 'hairpin_right':
            factory.add_turn(transition_chain_value_map['hairpin_right'][0], 180.0, speed_kph, 'right')
        elif move == 'lane_change_left':
            # Map values: [width, length]
            w, l = transition_chain_value_map['lane_change_left']
            factory.add_lane_change(w, l, 'left', speed_kph)
        elif move == 'lane_change_right':
            # Map values: [width, length]
            w, l = transition_chain_value_map['lane_change_right']
            factory.add_lane_change(w, l, 'right', speed_kph)
        elif move == 'chicane_left':
            w, l = transition_chain_value_map['chicane_left']
            factory.add_chicane(w, l, 'left', speed_kph)
        elif move == 'chicane_right':
            w, l = transition_chain_value_map['chicane_right']
            factory.add_chicane(w, l, 'right', speed_kph)
        elif move == 's_curve_left':
            r = transition_chain_value_map['s_curve_left'][0]
            factory.add_turn(r, 90.0, speed_kph, 'left')
            factory.add_turn(r, 90.0, speed_kph, 'right')
        elif move == 's_curve_right':
            r = transition_chain_value_map['s_curve_right'][0]
            factory.add_turn(r, 90.0, speed_kph, 'right')
            factory.add_turn(r, 90.0, speed_kph, 'left')

    factory.add_straight(run_out, speed_kph)
    return factory.get_path()
# ==============================================================================
# HIGH-LEVEL SCENARIO GENERATORS
# These functions use the PathFactory to build complete, usable paths.
# ==============================================================================
def generate_chicane_path(speed_kph, width, length, direction='left', run_up=30.0, run_out=50.0):
    """
    Generates a path for a simple chicane (a quick left-right or right-left swerve).
    This strongly teaches the "temporary deviation and return" pattern.
    """
    factory = PathFactory()
    factory.add_straight(run_up, speed_kph)

    # We can model a chicane with two very short, shallow turns
    # Using cosine interpolation is smoother than arcs for this
    speed_ms = speed_kph / 3.6
    resolution = factory.resolution
    n_points = int(length / resolution)
    
    start_x = factory.current_x
    start_y = factory.current_y

    for i in range(1, n_points + 1):
        p = i / n_points
        # A full sine wave cycle creates a swerve out and back
        y_offset = width * math.sin(p * math.pi) 
        
        # This assumes starting yaw is 0, which is fine after the run-up
        x = start_x + (i * resolution)
        y = start_y + y_offset if direction == 'left' else start_y - y_offset
        factory._add_point(x, y, speed_ms)

    # Manually update the factory state to the end of the chicane
    factory.current_x = start_x + length
    factory.current_y = start_y # We end up back on the center line
    
    # Add the run_out
    factory.add_straight(run_out, speed_kph)
    return factory.get_path()

def generate_straight_path(speed_kph, length=200.0):
    """Generates a simple straight-line path."""
    factory = PathFactory()
    factory.add_straight(length, speed_kph)
    return factory.get_path()

def generate_90_degree_turn_path(speed_kph, turn_radius, direction, run_up=30.0, run_out=50.0):
    """Generates a path with a single 90-degree turn."""
    factory = PathFactory()
    factory.add_straight(run_up, speed_kph)
    factory.add_turn(turn_radius, 90.0, speed_kph, direction)
    factory.add_straight(run_out, speed_kph)
    return factory.get_path()

def generate_hairpin_turn_path(speed_kph, turn_radius, direction, run_up=30.0, run_out=50.0):
    """Generates a path with a 180-degree hairpin turn."""
    factory = PathFactory()
    factory.add_straight(run_up, speed_kph)
    factory.add_turn(turn_radius, 180.0, speed_kph, direction)
    factory.add_straight(run_out, speed_kph)
    return factory.get_path()

def generate_s_curve_path(speed_kph, turn_radius, direction, run_up=30.0, run_out=50.0):
    """Generates a path with a classic S-curve."""
    factory = PathFactory()
    opposite_direction = 'right' if direction == 'left' else 'left'
    
    factory.add_straight(run_up, speed_kph)
    factory.add_turn(turn_radius, 90.0, speed_kph, direction)
    factory.add_turn(turn_radius, 90.0, speed_kph, opposite_direction)
    factory.add_straight(run_out, speed_kph)
    return factory.get_path()

# Keep your original lane change generator for continuity
def generate_lane_change_path(speed_kph, lc_length, start_y, target_y, run_up=30.0, run_out=50.0):
    """
    Generates a Sigmoid path for a lane change.
    """
    speed_ms = speed_kph / 3.6
    points = []
    start_x_offset = 10.0
    resolution = 0.5
    total_dist = run_up + lc_length + run_out
    n_points = int(total_dist / resolution)

    for i in range(n_points):
        x_dist = i * resolution
        global_x = start_x_offset + x_dist
        
        if x_dist < run_up:
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
        
        points.append([global_x, current_y, 0.0, speed_ms])
        
    return np.array(points)
