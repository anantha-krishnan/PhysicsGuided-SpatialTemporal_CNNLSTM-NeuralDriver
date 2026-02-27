import numpy as np
import math

class PathFactory:
    """
    A class to procedurally generate a sequence of waypoints by chaining together
    path primitives like straights and constant-radius turns.
    """
    def __init__(self, start_x=1000.0, start_y=-1.75, start_yaw_deg=0.0, resolution=0.5):
        self.resolution = resolution
        self.points = []
        
        # Initial state
        self.current_x = start_x
        self.current_y = start_y
        self.current_yaw_rad = math.radians(start_yaw_deg)

    def _add_point(self, x, y, speed_ms):
        """Appends a single waypoint to the path list."""
        self.points.append([x, y, 0.0, speed_ms])

    def add_straight(self, length, speed_kph):
        """Adds a straight line segment to the path from the current pose."""
        speed_ms = speed_kph / 3.6
        n_points = int(length / self.resolution)

        for i in range(1, n_points + 1):
            dist = i * self.resolution
            x = self.current_x + dist * math.cos(self.current_yaw_rad)
            y = self.current_y + dist * math.sin(self.current_yaw_rad)
            self._add_point(x, y, speed_ms)
        
        # Update the factory's end-state pose
        self.current_x += length * math.cos(self.current_yaw_rad)
        self.current_y += length * math.sin(self.current_yaw_rad)
        
    def add_turn(self, radius, angle_deg, speed_kph, direction):
        """Adds a constant-radius arc segment to the path."""
        if direction not in ['left', 'right']:
            raise ValueError("Direction must be 'left' or 'right'.")

        speed_ms = speed_kph / 3.6
        turn_direction = 1.0 if direction == 'left' else -1.0
        angle_rad = math.radians(angle_deg)
        
        # Calculate the center of the turning circle
        cx = self.current_x + radius * math.cos(self.current_yaw_rad + turn_direction * math.pi / 2)
        cy = self.current_y + radius * math.sin(self.current_yaw_rad + turn_direction * math.pi / 2)
        
        # Calculate the starting angle of the car relative to the circle center
        start_angle = self.current_yaw_rad - turn_direction * math.pi / 2
        
        arc_length = radius * angle_rad
        n_points = int(arc_length / self.resolution)
        
        for i in range(1, n_points + 1):
            frac = i / n_points
            current_arc_angle = start_angle + turn_direction * frac * angle_rad
            
            x = cx + radius * math.cos(current_arc_angle)
            y = cy + radius * math.sin(current_arc_angle)
            self._add_point(x, y, speed_ms)

        # Update the factory's end-state pose
        self.current_x = x
        self.current_y = y
        self.current_yaw_rad += turn_direction * angle_rad

    def get_path(self):
        """Returns the generated path as a NumPy array."""
        return np.array(self.points)
    

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
    (This is your original, refactored to fit the new structure)
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