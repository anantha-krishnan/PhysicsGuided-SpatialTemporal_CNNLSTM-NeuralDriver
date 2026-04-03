# minimap.py
import pygame
import numpy as np
from collections import deque
import csv
# ==============================================================================
# -- PygameVisualizer ----------------------------------------------------------
# ==============================================================================

class PygameVisualizer:
    def __init__(self, window_size=(800, 400), background_color=(28, 28, 28), log_frame=False):
        pygame.init()
        self.width, self.height = window_size
        self.screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Real-time Trajectory, Velocity, and Steering Monitor")
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.background_color = background_color
        self.log_frame=log_frame
        # --- Dynamic Layout Calculation ---
        margin = 20
        gap = 20
        
        # Minimap is a square on the left, its size is based on the window's height.
        minimap_size = self.height - 2 * margin
        self.minimap_rect = pygame.Rect(margin, margin, minimap_size, minimap_size)
        
        # Right Side Area for Plots
        plot_left = self.minimap_rect.right + gap
        plot_width = self.width - plot_left - margin
        
        plot_top_margin = 40
        plot_bottom_margin = 30
        plot_area_height = self.height - plot_top_margin - plot_bottom_margin
        
        #split vertical space for THREE plots: Velocity, Steering, CTE
        plot_gap = 70 # Space between the two plots to fit legends and titles
        single_plot_height = (plot_area_height - 2 * plot_gap) // 3
        
        # Rectangles for plotting areas
        self.vel_plot_rect = pygame.Rect(plot_left, plot_top_margin, plot_width, single_plot_height)

        self.steer_plot_rect = pygame.Rect(
            plot_left,
            self.vel_plot_rect.bottom + plot_gap,
            plot_width,
            single_plot_height,
        )

        self.cte_plot_rect = pygame.Rect(
            plot_left,
            self.steer_plot_rect.bottom + plot_gap,
            plot_width,
            single_plot_height,
        )
        # --- Minimap Data ---
        self.world_path = None
        self.screen_path = None
        
        # --- Plot Histories ---
        self.VEL_PLOT_MAX_KPH = 120.0
        self.VEL_PLOT_MAX_MS = self.VEL_PLOT_MAX_KPH / 3.6
        
        # History length dynamically matches pixel width of plots
        history_len = self.vel_plot_rect.width
        self.target_vel_history = deque(maxlen=history_len)
        self.actual_vel_history = deque(maxlen=history_len)
        self.steer_history = deque(maxlen=history_len) # NEW: Steering history
        self.cte_history = deque(maxlen=history_len) # NEW: CTE history
        self.CTE_PLOT_MAX = 10.0 # +/- meters — adjust as needed
        
        self.log_file = open("telemetry_log.csv", "w", newline="")
        self.log_writer = csv.writer(self.log_file)

        # Header row
        self.log_writer.writerow(["target_vel_ms", "actual_vel_ms", "steer", "cte", "he", "fut_yaw", "future_path_curvature", "yaw_rate"])
        self.log_file.flush()

    def set_path(self, ghost_path):
        """Pre-calculates the transformation for the minimap."""
        self.world_path = np.array(ghost_path)[:, :2]
        self.target_vel_history.clear()
        self.actual_vel_history.clear()
        self.steer_history.clear() # Clear steering on new path

        if self.world_path.size == 0: return

        min_x, max_x = np.min(self.world_path[:, 0]), np.max(self.world_path[:, 0])
        min_y, max_y = np.min(self.world_path[:, 1]), np.max(self.world_path[:, 1])
        
        world_width = max(1, max_x - min_x)
        world_height = max(1, max_y - min_y)
        
        x_scale = self.minimap_rect.width / world_width
        y_scale = self.minimap_rect.height / world_height
        self.scale = min(x_scale, y_scale)
        
        self.x_offset = min_x
        self.y_offset = min_y
        
        self.screen_path = [self._world_to_screen(p[0], p[1]) for p in self.world_path]

    def _world_to_screen(self, x, y):
        """Converts a CARLA world coordinate to a minimap screen coordinate."""
        screen_x = (x - self.x_offset) * self.scale
        screen_y = (y - self.y_offset) * self.scale
        
        path_render_width = (np.max(self.world_path[:, 0]) - self.x_offset) * self.scale
        path_render_height = (np.max(self.world_path[:, 1]) - self.y_offset) * self.scale
        x_padding = (self.minimap_rect.width - path_render_width) / 2
        y_padding = (self.minimap_rect.height - path_render_height) / 2

        return (int(self.minimap_rect.left + screen_x + x_padding), 
                int(self.minimap_rect.top + screen_y + y_padding))

    def _draw_velocity_plot(self):
        """Draws the Velocity graph section."""
        pygame.draw.rect(self.screen, (40, 40, 40), self.vel_plot_rect.inflate(20, 40), border_radius=5)
        title_surf = self.font.render("Velocity (km/h)", True, (255, 255, 255))
        self.screen.blit(title_surf, (self.vel_plot_rect.centerx - title_surf.get_width() // 2, self.vel_plot_rect.top - 30))

        pygame.draw.line(self.screen, (150, 150, 150), self.vel_plot_rect.bottomleft, self.vel_plot_rect.topleft, 2)
        pygame.draw.line(self.screen, (150, 150, 150), self.vel_plot_rect.bottomleft, self.vel_plot_rect.bottomright, 2)
        
        max_vel_surf = self.small_font.render(f"{self.VEL_PLOT_MAX_KPH:.0f}", True, (200, 200, 200))
        self.screen.blit(max_vel_surf, (self.vel_plot_rect.left - 35, self.vel_plot_rect.top - 8))
        min_vel_surf = self.small_font.render("0", True, (200, 200, 200))
        self.screen.blit(min_vel_surf, (self.vel_plot_rect.left - 20, self.vel_plot_rect.bottom - 10))

        grid_color = (70, 70, 70)
        for i in range(1, 4):
            vel_kph = i * (self.VEL_PLOT_MAX_KPH / 4)
            y = self.vel_plot_rect.bottom - int((vel_kph / self.VEL_PLOT_MAX_KPH) * self.vel_plot_rect.height)
            pygame.draw.line(self.screen, grid_color, (self.vel_plot_rect.left, y), (self.vel_plot_rect.right, y), 1)
            label_surf = self.small_font.render(f"{vel_kph:.0f}", True, (200, 200, 200))
            self.screen.blit(label_surf, (self.vel_plot_rect.left - 35, y - 8))

        if len(self.actual_vel_history) > 1:
            target_points = []
            for i, vel in enumerate(self.target_vel_history):
                x = self.vel_plot_rect.left + i
                y = self.vel_plot_rect.bottom - int((vel / self.VEL_PLOT_MAX_MS) * self.vel_plot_rect.height)
                target_points.append((x, y))
            pygame.draw.lines(self.screen, (100, 100, 255), False, target_points, 2)

            actual_points = []
            for i, vel in enumerate(self.actual_vel_history):
                x = self.vel_plot_rect.left + i
                y = self.vel_plot_rect.bottom - int((vel / self.VEL_PLOT_MAX_MS) * self.vel_plot_rect.height)
                actual_points.append((x, y))
            pygame.draw.lines(self.screen, (255, 80, 80), False, actual_points, 2)

            last_actual_vel_ms = self.actual_vel_history[-1]
            last_actual_vel_kph = last_actual_vel_ms * 3.6
            y_actual = self.vel_plot_rect.bottom - int((last_actual_vel_ms / self.VEL_PLOT_MAX_MS) * self.vel_plot_rect.height)

            line_color = (255, 80, 80, 150)
            for x in range(self.vel_plot_rect.left, self.vel_plot_rect.right, 10):
                pygame.draw.line(self.screen, line_color, (x, y_actual), (x + 5, y_actual), 1)

            last_point_x = self.vel_plot_rect.left + len(self.actual_vel_history) - 1
            pygame.draw.circle(self.screen, (255, 255, 255), (last_point_x, y_actual), 5)
            pygame.draw.circle(self.screen, (255, 80, 80), (last_point_x, y_actual), 3)

            readout_text = f"{last_actual_vel_kph:.1f} km/h"
            text_surf = self.font.render(readout_text, True, (255, 255, 255))
            
            text_rect = text_surf.get_rect(topright=(self.vel_plot_rect.right - 10, y_actual + 5))
            
            bg_rect = text_rect.inflate(10, 6)
            pygame.draw.rect(self.screen, (20, 20, 20), bg_rect, border_radius=3)
            pygame.draw.rect(self.screen, (255, 80, 80), bg_rect, width=1, border_radius=3)
            
            self.screen.blit(text_surf, text_rect)

        legend_y = self.vel_plot_rect.bottom + 25
        pygame.draw.line(self.screen, (100, 100, 255), (self.vel_plot_rect.right - 200, legend_y), (self.vel_plot_rect.right - 180, legend_y), 2)
        target_legend = self.small_font.render("Target", True, (255, 255, 255))
        self.screen.blit(target_legend, (self.vel_plot_rect.right - 170, legend_y - 7))

        pygame.draw.line(self.screen, (255, 80, 80), (self.vel_plot_rect.right - 100, legend_y), (self.vel_plot_rect.right - 80, legend_y), 2)
        actual_legend = self.small_font.render("Actual", True, (255, 255, 255))
        self.screen.blit(actual_legend, (self.vel_plot_rect.right - 70, legend_y - 7))

    def _draw_steer_plot(self):
        """Draws the Steering Command graph section."""
        pygame.draw.rect(self.screen, (40, 40, 40), self.steer_plot_rect.inflate(20, 40), border_radius=5)
        title_surf = self.font.render("Steer Command", True, (255, 255, 255))
        self.screen.blit(title_surf, (self.steer_plot_rect.centerx - title_surf.get_width() // 2, self.steer_plot_rect.top - 30))

        pygame.draw.line(self.screen, (150, 150, 150), self.steer_plot_rect.bottomleft, self.steer_plot_rect.topleft, 2)
        pygame.draw.line(self.screen, (150, 150, 150), self.steer_plot_rect.bottomleft, self.steer_plot_rect.bottomright, 2)
        
        max_steer = 1.0
        min_steer = -1.0
        
        # Grid lines and labels for -1.0, -0.5, 0.0, 0.5, 1.0
        grid_color = (70, 70, 70)
        for val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            y = self.steer_plot_rect.bottom - int(((val - min_steer) / (max_steer - min_steer)) * self.steer_plot_rect.height)
            pygame.draw.line(self.screen, grid_color, (self.steer_plot_rect.left, y), (self.steer_plot_rect.right, y), 1)
            label_surf = self.small_font.render(f"{val:.1f}", True, (200, 200, 200))
            self.screen.blit(label_surf, (self.steer_plot_rect.left - 30, y - 8))

        if len(self.steer_history) > 1:
            steer_points = []
            for i, st in enumerate(self.steer_history):
                x = self.steer_plot_rect.left + i
                st_clamped = max(min_steer, min(max_steer, st))
                y = self.steer_plot_rect.bottom - int(((st_clamped - min_steer) / (max_steer - min_steer)) * self.steer_plot_rect.height)
                steer_points.append((x, y))
                
            pygame.draw.lines(self.screen, (80, 255, 80), False, steer_points, 2)

            last_st = self.steer_history[-1]
            st_clamped = max(min_steer, min(max_steer, last_st))
            y_actual = self.steer_plot_rect.bottom - int(((st_clamped - min_steer) / (max_steer - min_steer)) * self.steer_plot_rect.height)

            line_color = (80, 255, 80, 150)
            for x in range(self.steer_plot_rect.left, self.steer_plot_rect.right, 10):
                pygame.draw.line(self.screen, line_color, (x, y_actual), (x + 5, y_actual), 1)

            last_point_x = self.steer_plot_rect.left + len(self.steer_history) - 1
            pygame.draw.circle(self.screen, (255, 255, 255), (last_point_x, y_actual), 5)
            pygame.draw.circle(self.screen, (80, 255, 80), (last_point_x, y_actual), 3)

            readout_text = f"{last_st:.2f}"
            text_surf = self.font.render(readout_text, True, (255, 255, 255))
            
            # Position text box securely
            text_rect = text_surf.get_rect(topright=(self.steer_plot_rect.right - 10, y_actual + 5))
            if text_rect.bottom > self.steer_plot_rect.bottom + 10:
                text_rect = text_surf.get_rect(bottomright=(self.steer_plot_rect.right - 10, y_actual - 5))
            
            bg_rect = text_rect.inflate(10, 6)
            pygame.draw.rect(self.screen, (20, 20, 20), bg_rect, border_radius=3)
            pygame.draw.rect(self.screen, (80, 255, 80), bg_rect, width=1, border_radius=3)
            
            self.screen.blit(text_surf, text_rect)

    def _draw_cte_plot(self):
        """Draws the Cross Track Error graph section."""
        pygame.draw.rect(self.screen, (40, 40, 40),
                        self.cte_plot_rect.inflate(20, 40), border_radius=5)

        # Title
        title = self.font.render("Cross Track Error (m)", True, (255, 255, 255))
        self.screen.blit(
            title,
            (self.cte_plot_rect.centerx - title.get_width() // 2,
            self.cte_plot_rect.top - 30)
        )

        # Axes
        pygame.draw.line(self.screen, (150, 150, 150),
                        self.cte_plot_rect.bottomleft, self.cte_plot_rect.topleft, 2)
        pygame.draw.line(self.screen, (150, 150, 150),
                        self.cte_plot_rect.bottomleft, self.cte_plot_rect.bottomright, 2)

        max_cte = self.CTE_PLOT_MAX
        min_cte = -self.CTE_PLOT_MAX

        # Grid lines every 1m
        grid_color = (70, 70, 70)
        for val in np.linspace(min_cte, max_cte, 5):  # -max, -half, 0, half, +max
            y = self.cte_plot_rect.bottom - int(((val - min_cte)/(max_cte-min_cte)) *
                                                self.cte_plot_rect.height)
            pygame.draw.line(self.screen, grid_color,
                            (self.cte_plot_rect.left, y),
                            (self.cte_plot_rect.right, y), 1)
            label = self.small_font.render(f"{val:.1f}", True, (200, 200, 200))
            self.screen.blit(label, (self.cte_plot_rect.left - 32, y - 8))

        # Draw CTE line
        if len(self.cte_history) > 1:
            points = []
            for i, c in enumerate(self.cte_history):
                c = max(min_cte, min(max_cte, c))  # clamp
                x = self.cte_plot_rect.left + i
                y = self.cte_plot_rect.bottom - int(((c - min_cte) /
                                                    (max_cte - min_cte)) *
                                                    self.cte_plot_rect.height)
                points.append((x, y))

            pygame.draw.lines(self.screen, (255, 220, 0), False, points, 2)

            # Last point highlight
            last_cte = self.cte_history[-1]
            lc = max(min_cte, min(max_cte, last_cte))
            y_last = self.cte_plot_rect.bottom - int(((lc - min_cte)/(max_cte-min_cte)) *
                                                    self.cte_plot_rect.height)

            last_x = self.cte_plot_rect.left + len(self.cte_history) - 1
            pygame.draw.circle(self.screen, (255, 255, 255), (last_x, y_last), 5)
            pygame.draw.circle(self.screen, (255, 220, 0), (last_x, y_last), 3)

            # Floating value box
            text = self.font.render(f"{last_cte:.2f} m", True, (255, 255, 255))
            rect = text.get_rect(topright=(self.cte_plot_rect.right - 10, y_last + 5))
            bg = rect.inflate(10, 6)

            pygame.draw.rect(self.screen, (20, 20, 20), bg, border_radius=3)
            pygame.draw.rect(self.screen, (255, 220, 0), bg, width=1, border_radius=3)
            self.screen.blit(text, rect)
    def render(self, vehicle, target_vel_ms, actual_vel_ms, steer=0.0, cte=0.0, he=0.0, fut_yaw=0.0, future_path_curvature=0.0, yaw_rate=0.0):
        """Main render loop. Draws all components."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        if self.log_frame:
            self._log_frame(target_vel_ms, actual_vel_ms, steer, cte, he, fut_yaw, future_path_curvature, yaw_rate) # Log telemetry data
        self.target_vel_history.append(target_vel_ms)
        self.actual_vel_history.append(actual_vel_ms)
        self.steer_history.append(steer) # Log steering data
        self.cte_history.append(cte)
        self.screen.fill(self.background_color)
        
        pygame.draw.rect(self.screen, (40, 40, 40), self.minimap_rect.inflate(10, 10), border_radius=5)
        if self.screen_path and len(self.screen_path) > 1:
            pygame.draw.lines(self.screen, (0, 180, 0), False, self.screen_path, 2)
        if vehicle:
            v_loc = vehicle.get_transform().location
            vehicle_pos_screen = self._world_to_screen(v_loc.x, v_loc.y)
            pygame.draw.circle(self.screen, (255, 100, 100), vehicle_pos_screen, 5)
        
        self._draw_velocity_plot()
        self._draw_steer_plot() # Render the new plot
        self._draw_cte_plot()
        pygame.display.flip()
        return True
    def _log_frame(self, target_vel, actual_vel, steer, cte, he, fut_yaw, future_path_curvature, yaw_rate):
        self.log_writer.writerow([target_vel, actual_vel, steer, cte, he, fut_yaw, future_path_curvature, yaw_rate])
        self.log_file.flush()  # ensures data is NOT lost on crash

    def destroy(self):
        self.log_file.close()
        pygame.quit()