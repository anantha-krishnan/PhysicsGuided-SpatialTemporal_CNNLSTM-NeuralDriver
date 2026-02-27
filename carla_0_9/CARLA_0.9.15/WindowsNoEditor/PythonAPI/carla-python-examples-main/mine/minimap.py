# minimap.py
import pygame
import numpy as np
from collections import deque

# ==============================================================================
# -- PygameVisualizer ----------------------------------------------------------
# ==============================================================================

class PygameVisualizer:
    def __init__(self, window_size=(1000, 500), background_color=(28, 28, 28)):
        pygame.init()
        self.width, self.height = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Real-time Trajectory and Velocity Monitor")
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.background_color = background_color
        
        # Define drawing areas for the two plots
        self.minimap_rect = pygame.Rect(20, 20, 460, 460)
        self.plot_rect = pygame.Rect(520, 40, 460, 420) # Main area for graph lines

        # --- Minimap Data ---
        self.world_path = None
        self.screen_path = None
        
        # --- Velocity Plot Data ---
        self.VEL_PLOT_MAX_KPH = 120.0  # Set a fixed upper limit for the Y-axis (in kph)
        self.VEL_PLOT_MAX_MS = self.VEL_PLOT_MAX_KPH / 3.6
        history_len = self.plot_rect.width - 20 # Number of points to show on the x-axis
        self.target_vel_history = deque(maxlen=history_len)
        self.actual_vel_history = deque(maxlen=history_len)

    def set_path(self, ghost_path):
        """Pre-calculates the transformation for the minimap."""
        self.world_path = np.array(ghost_path)[:, :2]
        # Reset velocity history for the new episode
        self.target_vel_history.clear()
        self.actual_vel_history.clear()

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
        pygame.draw.rect(self.screen, (40, 40, 40), self.plot_rect.inflate(40, 40), border_radius=5)
        title_surf = self.font.render("Velocity (km/h)", True, (255, 255, 255))
        self.screen.blit(title_surf, (self.plot_rect.centerx - title_surf.get_width() // 2, self.plot_rect.top - 30))

        # 2. Draw Axes and Labels
        pygame.draw.line(self.screen, (150, 150, 150), self.plot_rect.bottomleft, self.plot_rect.topleft, 2)
        pygame.draw.line(self.screen, (150, 150, 150), self.plot_rect.bottomleft, self.plot_rect.bottomright, 2)
        
        max_vel_surf = self.small_font.render(f"{self.VEL_PLOT_MAX_KPH:.0f}", True, (200, 200, 200))
        self.screen.blit(max_vel_surf, (self.plot_rect.left - 30, self.plot_rect.top - 5))
        min_vel_surf = self.small_font.render("0", True, (200, 200, 200))
        self.screen.blit(min_vel_surf, (self.plot_rect.left - 20, self.plot_rect.bottom - 10))

        # 3. Transform and Draw Velocity Lines
        if len(self.actual_vel_history) > 1:
            # Target Velocity (Dashed Blue)
            target_points = []
            for i, vel in enumerate(self.target_vel_history):
                x = self.plot_rect.left + i
                y = self.plot_rect.bottom - int((vel / self.VEL_PLOT_MAX_MS) * self.plot_rect.height)
                target_points.append((x, y))
            pygame.draw.lines(self.screen, (100, 100, 255), False, target_points, 2)

            # Actual Velocity (Solid Red)
            actual_points = []
            for i, vel in enumerate(self.actual_vel_history):
                x = self.plot_rect.left + i
                y = self.plot_rect.bottom - int((vel / self.VEL_PLOT_MAX_MS) * self.plot_rect.height)
                actual_points.append((x, y))
            pygame.draw.lines(self.screen, (255, 80, 80), False, actual_points, 2)

        # 4. Draw Legend
        pygame.draw.line(self.screen, (100, 100, 255), (self.plot_rect.right - 200, self.plot_rect.bottom + 25), (self.plot_rect.right - 180, self.plot_rect.bottom + 25), 2)
        target_legend = self.small_font.render("Target", True, (255, 255, 255))
        self.screen.blit(target_legend, (self.plot_rect.right - 170, self.plot_rect.bottom + 18))

        pygame.draw.line(self.screen, (255, 80, 80), (self.plot_rect.right - 100, self.plot_rect.bottom + 25), (self.plot_rect.right - 80, self.plot_rect.bottom + 25), 2)
        actual_legend = self.small_font.render("Actual", True, (255, 255, 255))
        self.screen.blit(actual_legend, (self.plot_rect.right - 70, self.plot_rect.bottom + 18))

    def render(self, vehicle, target_vel_ms, actual_vel_ms):
        """Main render loop. Draws all components."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # --- Update Data ---
        self.target_vel_history.append(target_vel_ms)
        self.actual_vel_history.append(actual_vel_ms)

        # --- Drawing ---
        self.screen.fill(self.background_color)
        
        # 1. Draw Minimap
        pygame.draw.rect(self.screen, (40, 40, 40), self.minimap_rect.inflate(20, 20), border_radius=5)
        if self.screen_path and len(self.screen_path) > 1:
            pygame.draw.lines(self.screen, (0, 180, 0), False, self.screen_path, 2)
        if vehicle:
            v_loc = vehicle.get_transform().location
            vehicle_pos_screen = self._world_to_screen(v_loc.x, v_loc.y)
            pygame.draw.circle(self.screen, (255, 100, 100), vehicle_pos_screen, 5)
        
        # 2. Draw Velocity Plot
        self._draw_velocity_plot()
        
        # Update Display
        pygame.display.flip()
        return True

    def destroy(self):
        """Properly close Pygame."""
        pygame.quit()