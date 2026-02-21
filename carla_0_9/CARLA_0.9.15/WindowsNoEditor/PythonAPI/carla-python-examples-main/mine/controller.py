# controller.py
import math
import numpy as np
import os
# --- CONFIGURATION ---
# Tesla Model 3 physical dimensions for Controller

WHEELBASE = 2.875  # Meters (Real value for Model 3, 1.5 is too short)

class PathFollower:
    def __init__(self, filepath=None, direct_data=None):
        self.last_closest_idx = 0  
        self._pi = math.pi
        self._2pi = 2.0 * math.pi
        self.t_previous = 0.0
        self.error_previous = 0.0
        self.integral_error = 0.0
        self.Kp = 0.2
        self.Ki = 0.05
        self.Kd = 0.3
        if direct_data is not None:
            self.data = direct_data
            self.waypoints_xy = self.data[:, 0:2]
            self.target_speeds = self.data[:, 3]
            print(f"Loaded waypoints from memory. Total points: {len(self.data)}")
        elif filepath:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Could not find {filepath}")
            
            print(f"Loading waypoints from {filepath}...")
            try:
                self.data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            except Exception as e:
                print(f"Error loading CSV: {e}")
                self.data = []

            if len(self.data) == 0:
                raise ValueError("Waypoint file is empty!")

            self.waypoints_xy = self.data[:, 0:2] 
            self.target_speeds = self.data[:, 3]
        self.current_target_speed = self.target_speeds[0]
        self.end_path=False
    
    def get_curvature_based_speed(self, vehicle_location, lookahead_dist=10.0, max_lat_accel=4.0):
        """
        Scans the ghost path ahead to find the sharpest curve.
        Returns a speed limit that keeps Lateral Accel under control.
        
        max_lat_accel: 4.0 m/s^2 is a comfortable limit (approx 0.4g). 
                       Set to 7.0 or 8.0 for racing.
        """
        # Find current index
        search_range = 20 # Look ahead 20 points (approx 10 meters)
        start_idx = self.last_closest_idx
        end_idx = min(start_idx + search_range, len(self.waypoints_xy) - 2)
        
        min_radius = float('inf')
        
        # 3-Point Circle Fit to find Radius
        for i in range(start_idx, end_idx, 2): # Step by 2 to reduce noise
            p1 = self.waypoints_xy[i]
            p2 = self.waypoints_xy[i+1]
            p3 = self.waypoints_xy[i+2]
            
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
                if radius < min_radius:
                    min_radius = radius
        
        # Physics Formula: V = sqrt(a_lat * R)
        if min_radius == float('inf'):
            return 999.0 # No limit
            
        target_speed = math.sqrt(max_lat_accel * min_radius)
        return target_speed
    def get_pure_pursuit_steering(self, x, y, yaw, v):
        """
        Calculates steering angle using Pure Pursuit.
        Crucially, it searches for the target STARTING from the last known position.
        """
        k_dd = 0.5  # Look-ahead gain
        # Dynamic lookahead: increases with speed
        look_ahead_dis = np.clip(k_dd * v, 3.0, 20.0) 

        # 1. Find the Closest Waypoint (Starting search from previous index)
        # We optimize by searching only the next 500 points, not the whole file
        search_start = self.last_closest_idx
        search_end = min(self.last_closest_idx + 500, len(self.waypoints_xy))
        # check if we have enough points ahead, if not, it means we are near the end, so we should end the search at the end of the list and not wrap around. 
        if search_end - search_start < 10:  # we have reached the end. inform user and end search at the end of the list
            print("Warning: Reached end of waypoints. Ending search at the end of the list.")
            search_end = len(self.waypoints_xy)
            self.end_path = True

        # Calculate distances only for the subset
        distances = np.linalg.norm(self.waypoints_xy[search_start:search_end] - np.array([x, y]), axis=1)
        
        # Get local index of minimum, add start offset to get global index
        local_min_idx = np.argmin(distances)
        self.last_closest_idx = search_start + local_min_idx

        # 2. Find Target Waypoint (Lookahead)
        # We iterate forward from the closest index until distance > look_ahead
        target_wp = None
        
        # Start loop from current closest
        for i in range(self.last_closest_idx, len(self.waypoints_xy)):
            dist = np.linalg.norm(self.waypoints_xy[i] - np.array([x, y]))
            if dist >= look_ahead_dis:
                target_wp = self.waypoints_xy[i]
                self.current_target_speed = self.target_speeds[i]
                break
        
        # If we ran out of points (end of track), use the very last point
        if target_wp is None:
            target_wp = self.waypoints_xy[-1]
            self.current_target_speed = self.target_speeds[-1]

        # 3. Calculate Heading Error (Alpha)
        # Move axle reference to Rear Axle
        rear_x = x - (WHEELBASE / 2.0) * np.cos(yaw)
        rear_y = y - (WHEELBASE / 2.0) * np.sin(yaw)

        path_vector_x = target_wp[0] - rear_x
        path_vector_y = target_wp[1] - rear_y
        
        # Standard atan2 (y, x)
        path_yaw = np.arctan2(path_vector_y, path_vector_x)
        
        alpha = path_yaw - yaw
        
        # Normalize angle to [-pi, pi]
        while alpha > self._pi: alpha -= self._2pi
        while alpha < -self._pi: alpha += self._2pi

        # 4. Pure Pursuit Steering Law
        # steer = atan( 2 * L * sin(alpha) / look_ahead )
        steer_output = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), look_ahead_dis)
        
        return steer_output
    
    def get_long_vel(self, v, v_desired, t):
        ######################################################
        ######################################################
        # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
        ######################################################
        ######################################################
        """
            Implement a longitudinal controller here. Remember that you can
            access the persistent variables declared above here. For
            example, can treat self.vars.v_previous like a "global variable".
        """
        dt = t - self.t_previous
        v_error = v_desired - v

        Kp, Ki, Kd = self.Kp, self.Ki, self.Kd
        
        D = 0.0
        if dt > 1e-6:
            D = Kd * (v_error - self.error_previous) / dt

        self.integral_error += v_error * dt
        self.integral_error = np.clip(self.integral_error, -5.0, 5.0)

        pid_output = (Kp * v_error) + (Ki * self.integral_error) + D

        if pid_output > 0:
            throttle_output = np.fmin(pid_output, 1.0)
            brake_output = 0.0
        else:
            throttle_output = 0.0
            brake_output = np.fmin(abs(pid_output), 1.0)

        # Change these outputs with the longitudinal controller. Note that
        # brake_output is optional and is not required to pass the
        # assignment, as the car will naturally slow down over time.
        
        #print(f"Speed: {v:.2f} | Target: {v_desired:.2f} | Error: {v_error:.2f} | Throttle: {throttle_output:.2f} | Brake: {brake_output:.2f}")
        self.error_previous = v_error
        return throttle_output, brake_output
