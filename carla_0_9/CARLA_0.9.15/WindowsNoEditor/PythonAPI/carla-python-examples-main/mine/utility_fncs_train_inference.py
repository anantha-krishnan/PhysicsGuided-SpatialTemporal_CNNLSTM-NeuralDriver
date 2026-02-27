# utility_fncs_train_inference.py
# Utility functions for training and inference of the controller
import numpy as np
import math
class ControllerUtils:
    def __init__(self, data=None, lookahead_time=None, lookahead_dist=None, resolution=0.5):
        self.waypoints_xy = data[:, 0:2] if data is not None else None
        self.target_speed = data[:, 3] if data is not None else None
        self.last_closest_idx = 0
        self.resolution = resolution # Meters between waypoints, used for lookahead index calculation
        if lookahead_time and lookahead_dist:
            raise ValueError("Both lookahead_time and lookahead_dist cannot be provided together.")
        elif not lookahead_time and not lookahead_dist:
            raise ValueError("Either lookahead_time or lookahead_dist must be provided.")
        elif lookahead_dist:
            self.lookahead_pts = int(lookahead_dist / self.resolution)
        elif lookahead_time:
            self.lookahead_pts = int(self.target_speed[0] * lookahead_time / self.resolution)
        self.lookahead_time = lookahead_time
        self.lookahead_dist = lookahead_dist
        
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

    def calculate_relative_errors(self, vehicle, path_points=None):
        """
        Calculates Inputs for the AI:
        1. CTE (Lateral Error)
        2. Heading Error
        3. Future Curvature (Lookahead Error)
        """
        if not path_points:
            path_points = self.waypoints_xy
            target_speed = self.target_speed
        else:
            path_points = path_points[:, 0:2]
            target_speed = path_points[:, 3]
        v_trans = vehicle.get_transform()
        v_loc = v_trans.location
        # Current Speed
        vel = vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2)
        # Find closest point based on lookahead_dist or lookahead_time
        search_start = self.last_closest_idx
        # lookahead_dist
        if self.lookahead_dist:
            search_end = min(self.last_closest_idx + self.lookahead_pts, len(path_points))
        # lookahead_time
        else:
            lookahead_pts_time = int((speed * self.lookahead_time) / self.resolution)
            search_end = min(self.last_closest_idx + lookahead_pts_time, len(path_points))
        search_points = path_points[search_start:search_end, :2]
        if len(search_points) < 5:
            print("Warning: Not enough points in search window for error calculation. End of track may be near.")
            # return 0, 0, 0, 0, 0, -1  # Not enough points to calculate errors
        dists = np.linalg.norm(search_points - np.array([v_loc.x, v_loc.y]), axis=1)
        min_idx = np.argmin(dists) + search_start        
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

        speed_error = target_speed[min_idx] -speed
        # 3. Lookahead/Future Error (The "Scalability" Feature)
        # What is the CTE 1.0 seconds ahead?        
        lookahead_dist = max(10.0, speed * 1.0) # Look 1s ahead
        
        # Find index approx lookahead_dist away
        look_idx = min_idx
        dist_accum = 0
        while look_idx < len(path_points)-1 and dist_accum < lookahead_dist:
            dist_accum += self.resolution
            look_idx += 1
            
        future_pt = path_points[look_idx]
        # Future CTE approximation (simple Y diff for this straight track)
        #future_cte = v_loc.y - future_pt[1]
        future_cte = self.get_cte(vehicle, future_pt, path_points[look_idx+1] if look_idx+1 < len(path_points) else future_pt)
        future_path_curvature = self.calculate_signed_path_curvature(min_idx, speed)
        self.last_closest_idx = min_idx

        return cte, he, future_cte, speed, speed_error, self.last_closest_idx, future_path_curvature
    
    def calculate_min_radius(self, start_idx, end_idx):
        """
        Scans the path ahead to find the sharpest curve and returns its radius.
        """
        min_radius = float('inf')
            
        # 3-Point Circle Fit to find Radius
        for i in range(start_idx, end_idx, 2): # Step by 2 to reduce noise
            p1 = self.waypoints_xy[i]
            p2 = self.waypoints_xy[i+1]
            p3 = self.waypoints_xy[i+2]
            radius = self.calculate_radius_of_curvature(p1, p2, p3)
            if radius < min_radius:
                min_radius = radius
        return min_radius
    def calculate_signed_path_curvature(self, current_idx, current_speed_ms, path_points=None):
        if path_points is None:
            path_points = self.waypoints_xy
        # Dynamic Lookahead
        lookahead_dist = max(10.0, current_speed_ms * 1.0)
        points_to_look_ahead = int(lookahead_dist / self.resolution)
        
        start_idx = current_idx
        end_idx = min(start_idx + points_to_look_ahead, len(path_points) - 2)

        if start_idx >= end_idx - 1:
            return 0.0

        # 1. Calculate the vector of the path at the START
        p_start = path_points[start_idx, :2]
        p_next  = path_points[start_idx+1, :2]
        vec_start = p_next - p_start
        
        # 2. Calculate the vector of the path at the END (Lookahead)
        p_end      = path_points[end_idx, :2]
        p_end_prev = path_points[end_idx-1, :2]
        vec_end    = p_end - p_end_prev
        
        # 3. Calculate the angle change (Heading difference)
        angle_start = math.atan2(vec_start[1], vec_start[0])
        angle_end   = math.atan2(vec_end[1], vec_end[0])
        
        angle_diff = angle_end - angle_start
        
        # Normalize to [-pi, pi]
        while angle_diff > math.pi: angle_diff -= 2 * math.pi
        while angle_diff < -math.pi: angle_diff += 2 * math.pi
        
        # 4. Calculate approximate Curvature (K = dTheta / dS)
        # This automatically gives us the SIGN (+ for Left, - for Right)
        dist = (end_idx - start_idx) * self.resolution # 0.5 is resolution
        
        if dist < 1.0: return 0.0
        
        curvature = angle_diff / dist
        
        # Clip extreme values (e.g. 90 deg turn in 1 meter) to keep inputs stable
        return np.clip(curvature, -0.2, 0.2)
    def calculate_path_curvature(self, current_idx, current_speed_ms, path_points=None):
        """
        Scans the path ahead of the vehicle to find the maximum curvature (1/radius).
        The lookahead distance is DYNAMIC based on speed.
        """
        if path_points is None:
            path_points = self.waypoints_xy
        # DYNAMIC LOOKAHEAD: Look ahead 1.0 seconds.
        # We add a minimum distance (e.g., 10m) to ensure stability at low speeds.
        lookahead_dist = max(10.0, current_speed_ms * 1.0)

        # Path resolution is 0.5 meters per point
        points_to_look_ahead = int(lookahead_dist / self.resolution)
        
        start_idx = current_idx
        end_idx = min(start_idx + points_to_look_ahead, len(path_points) - 3)
        min_radius = self.calculate_min_radius(start_idx, end_idx)
        
        if min_radius == float('inf'):
            return 0.0
        else:
            return 1.0 / min_radius
    def calculate_radius_of_curvature(self, p1, p2, p3):
        """
        Calculate radius of curvature given three points using the formula:
        K = (length of sides product) / (2 * Area of triangle formed by points)
        """
        # Triangle Area method to find curvature
        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
        
        # Side lengths
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p2 - p3)
        c = np.linalg.norm(p3 - p1)
        
        # R = (abc) / (4 * Area)
        if area > 1e-4: # Avoid division by zero (straight line)
            radius = (a * b * c) / (4.0 * area)
        else:
            radius = float('inf') # Straight line, infinite radius
        return radius
    def calculate_future_heading_error(self, vehicle, current_idx):
        """
        Calculates the difference between the car's current heading and 
        the heading of the path at a fixed lookahead distance.
        """
        path_points = self.waypoints_xy
        # Fixed lookahead of 15-20 meters is robust for city speeds
        lookahead_dist = 20.0 
        points_to_look_ahead = int(lookahead_dist / self.resolution)
        
        # Target index
        target_idx = min(current_idx + points_to_look_ahead, len(path_points) - 2)
        
        # 1. Get Car Yaw
        v_trans = vehicle.get_transform()
        v_yaw_rad = math.radians(v_trans.rotation.yaw)
        
        # 2. Get Path Yaw at Lookahead Point
        # We use the vector between target and target+1
        p_now = path_points[target_idx, :2]
        p_next = path_points[target_idx + 1, :2]
        
        dx = p_next[0] - p_now[0]
        dy = p_next[1] - p_now[1]
        path_yaw_rad = math.atan2(dy, dx)
        
        # 3. Calculate Diff
        diff = path_yaw_rad - v_yaw_rad
        
        # Normalize to [-pi, pi]
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        
        return diff