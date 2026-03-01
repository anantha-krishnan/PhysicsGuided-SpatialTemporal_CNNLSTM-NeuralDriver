# minimap.py
import pygame
import numpy as np
from collections import deque

# ==============================================================================
# -- PygameVisualizer ----------------------------------------------------------
# ==============================================================================

class PygameVisualizer:
    def __init__(self, window_size=(800, 400), background_color=(28, 28, 28)):
        pygame.init()
        self.width, self.height = window_size
        self.screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Real-time Trajectory and Velocity Monitor")
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.background_color = background_color
        
        # --- NEW: Dynamic Layout Calculation ---
        # This section calculates the layout based on the window size.
        margin = 20
        
        # Minimap is a square on the left, its size is based on the window's height.
        minimap_size = self.height - 2 * margin
        self.minimap_rect = pygame.Rect(margin, margin, minimap_size, minimap_size)
        
        # Plot takes the remaining space on the right.
        gap = 20
        plot_left = self.minimap_rect.right + gap
        plot_top_margin = 40  # Space for title
        plot_bottom_margin = 40 # Space for legend
        
        plot_width = self.width - plot_left - margin
        plot_height = self.height - plot_top_margin - plot_bottom_margin - (margin / 2)
        
        self.plot_rect = pygame.Rect(plot_left, plot_top_margin, plot_width, plot_height)
        
        # --- Minimap Data ---
        self.world_path = None
        self.screen_path = None
        
        # --- Velocity Plot Data ---
        self.VEL_PLOT_MAX_KPH = 120.0
        self.VEL_PLOT_MAX_MS = self.VEL_PLOT_MAX_KPH / 3.6
        # History length is now dynamically set by the plot's width.
        history_len = self.plot_rect.width
        self.target_vel_history = deque(maxlen=history_len)
        self.actual_vel_history = deque(maxlen=history_len)

    def set_path(self, ghost_path):
        """Pre-calculates the transformation for the minimap."""
        self.world_path = np.array(ghost_path)[:, :2]
        self.target_vel_history.clear()
        self.actual_vel_history.clear()

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
        """Draws the entire velocity graph section."""
        # 1. Draw Background and Title
        pygame.draw.rect(self.screen, (40, 40, 40), self.plot_rect.inflate(20, 40), border_radius=5)
        title_surf = self.font.render("Velocity (km/h)", True, (255, 255, 255))
        self.screen.blit(title_surf, (self.plot_rect.centerx - title_surf.get_width() // 2, self.plot_rect.top - 30))

        # 2. Draw Axes and Labels
        pygame.draw.line(self.screen, (150, 150, 150), self.plot_rect.bottomleft, self.plot_rect.topleft, 2)
        pygame.draw.line(self.screen, (150, 150, 150), self.plot_rect.bottomleft, self.plot_rect.bottomright, 2)
        
        max_vel_surf = self.small_font.render(f"{self.VEL_PLOT_MAX_KPH:.0f}", True, (200, 200, 200))
        self.screen.blit(max_vel_surf, (self.plot_rect.left - 35, self.plot_rect.top - 8))
        min_vel_surf = self.small_font.render("0", True, (200, 200, 200))
        self.screen.blit(min_vel_surf, (self.plot_rect.left - 20, self.plot_rect.bottom - 10))

        grid_color = (70, 70, 70)
        for i in range(1, 4):
            vel_kph = i * (self.VEL_PLOT_MAX_KPH / 4)
            y = self.plot_rect.bottom - int((vel_kph / self.VEL_PLOT_MAX_KPH) * self.plot_rect.height)
            pygame.draw.line(self.screen, grid_color, (self.plot_rect.left, y), (self.plot_rect.right, y), 1)
            label_surf = self.small_font.render(f"{vel_kph:.0f}", True, (200, 200, 200))
            self.screen.blit(label_surf, (self.plot_rect.left - 35, y - 8))

        # 3. Transform and Draw Velocity Lines
        if len(self.actual_vel_history) > 1:
            target_points = []
            for i, vel in enumerate(self.target_vel_history):
                x = self.plot_rect.left + i
                y = self.plot_rect.bottom - int((vel / self.VEL_PLOT_MAX_MS) * self.plot_rect.height)
                target_points.append((x, y))
            pygame.draw.lines(self.screen, (100, 100, 255), False, target_points, 2)

            actual_points = []
            for i, vel in enumerate(self.actual_vel_history):
                x = self.plot_rect.left + i
                y = self.plot_rect.bottom - int((vel / self.VEL_PLOT_MAX_MS) * self.plot_rect.height)
                actual_points.append((x, y))
            pygame.draw.lines(self.screen, (255, 80, 80), False, actual_points, 2)

            last_actual_vel_ms = self.actual_vel_history[-1]
            last_actual_vel_kph = last_actual_vel_ms * 3.6
            y_actual = self.plot_rect.bottom - int((last_actual_vel_ms / self.VEL_PLOT_MAX_MS) * self.plot_rect.height)

            line_color = (255, 80, 80, 150)
            for x in range(self.plot_rect.left, self.plot_rect.right, 10):
                pygame.draw.line(self.screen, line_color, (x, y_actual), (x + 5, y_actual), 1)

            last_point_x = self.plot_rect.left + len(self.actual_vel_history) - 1
            pygame.draw.circle(self.screen, (255, 255, 255), (last_point_x, y_actual), 5)
            pygame.draw.circle(self.screen, (255, 80, 80), (last_point_x, y_actual), 3)

            readout_text = f"{last_actual_vel_kph:.1f} km/h"
            text_surf = self.font.render(readout_text, True, (255, 255, 255))
            
            # --- FIXED HERE: Position text box *below* the indicator line ---
            # Anchor the text's TOP RIGHT to a point 5px below the line.
            text_rect = text_surf.get_rect(topright=(self.plot_rect.right - 10, y_actual + 5))
            
            bg_rect = text_rect.inflate(10, 6)
            pygame.draw.rect(self.screen, (20, 20, 20), bg_rect, border_radius=3)
            pygame.draw.rect(self.screen, (255, 80, 80), bg_rect, width=1, border_radius=3)
            
            self.screen.blit(text_surf, text_rect)

        # 4. Draw Legend
        legend_y = self.plot_rect.bottom + 25
        pygame.draw.line(self.screen, (100, 100, 255), (self.plot_rect.right - 200, legend_y), (self.plot_rect.right - 180, legend_y), 2)
        target_legend = self.small_font.render("Target", True, (255, 255, 255))
        self.screen.blit(target_legend, (self.plot_rect.right - 170, legend_y - 7))

        pygame.draw.line(self.screen, (255, 80, 80), (self.plot_rect.right - 100, legend_y), (self.plot_rect.right - 80, legend_y), 2)
        actual_legend = self.small_font.render("Actual", True, (255, 255, 255))
        self.screen.blit(actual_legend, (self.plot_rect.right - 70, legend_y - 7))

    def render(self, vehicle, target_vel_ms, actual_vel_ms):
        """Main render loop. Draws all components."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.target_vel_history.append(target_vel_ms)
        self.actual_vel_history.append(actual_vel_ms)

        self.screen.fill(self.background_color)
        
        pygame.draw.rect(self.screen, (40, 40, 40), self.minimap_rect.inflate(10, 10), border_radius=5)
        if self.screen_path and len(self.screen_path) > 1:
            pygame.draw.lines(self.screen, (0, 180, 0), False, self.screen_path, 2)
        if vehicle:
            v_loc = vehicle.get_transform().location
            vehicle_pos_screen = self._world_to_screen(v_loc.x, v_loc.y)
            pygame.draw.circle(self.screen, (255, 100, 100), vehicle_pos_screen, 5)
        
        self._draw_velocity_plot()
        
        pygame.display.flip()
        return True

    def destroy(self):
        pygame.quit()