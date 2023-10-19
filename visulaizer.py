import pygame
import numpy as np
import path_planner

import environment as env
# import planner

# Color Constants:
WHITE = (75, 75, 75)
BLACK = (30, 30, 35)
RED = (220, 80, 80)
GREEN = (0, 175, 39)
BLUE = (0, 157, 196)
ORANGE = (255, 165, 0)
# PURPLE = (150, 55, 230)
PURPLE = (0, 84, 170)

COLOR_DICT = {
    env.EMPTY_SQUARE: WHITE,
    env.OBSTACLE: BLACK,
    env.OPEN_SET: GREEN,
    env.CLOSED_SET: RED,
    env.START: BLUE,
    env.GOAL: ORANGE,
    env.PATH: PURPLE
}


class EnvVisualizer:

    def __init__(self, win_title: str = "Path Planner",
                 win_width: int = 800, win_height: int = 800,
                 fps: int = 120) -> None:

        pygame.init()

        self.screen = pygame.display.set_mode((win_width, win_height))
        pygame.display.set_caption(win_title)
        self.clock = pygame.time.Clock()
        self.fps = fps

        self.win_width = win_width
        self.win_height = win_height

        self.edit_square_type: int = env.OBSTACLE
        self.mouse_down = False
        self.run_sim = False
        self.is_running = True

    def display_env(self, grid_env: env.GridEnv):
        square_width: int = self.win_width // grid_env.num_cols
        square_height: int = self.win_height // grid_env.num_rows

        self.screen.fill(WHITE)

        for row in range(grid_env.shape[0]):
            for col in range(grid_env.shape[1]):

                if row == grid_env.start_pos[0] and col == grid_env.start_pos[1]:
                    square_color = COLOR_DICT[env.START]
                elif row == grid_env.end_pos[0] and col == grid_env.end_pos[1]:
                    square_color = COLOR_DICT[env.GOAL]
                elif (grid_env.overlay_grid is not None and grid_env[(row, col)] == env.EMPTY_SQUARE and grid_env.overlay_grid[(row, col)] != 0):
                    square_color = COLOR_DICT[grid_env.overlay_grid[(
                        row, col)]]
                else:
                    square_color = COLOR_DICT[grid_env[(row, col)]]

                square_params = (row * square_height, col * square_width,
                                 square_height, square_width)

                pygame.draw.rect(self.screen, square_color, square_params, 0)
                pygame.draw.rect(self.screen, BLACK, square_params, 1)

        pygame.display.flip()

        self.clock.tick(self.fps)

    def event_handler(self, grid_env: env.GridEnv) -> list[str]:

        flags = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_rusnning = False
                pygame.quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False

            if event.type == pygame.KEYDOWN:
                flags += self.keyboard_handler(grid_env)

        if self.mouse_down:
            flags += self.mouse_handler(grid_env)

        return flags

    def keyboard_handler(self, grid_env: env.GridEnv) -> list[str]:
        flags = []

        keys = pygame.key.get_pressed()

        if keys[pygame.K_SPACE]:
            is_empty = self.edit_square_type == env.EMPTY_SQUARE
            self.edit_square_type = env.OBSTACLE if is_empty else env.EMPTY_SQUARE
        if keys[pygame.K_1]:
            self.edit_square_type = env.START
        if keys[pygame.K_2]:
            self.edit_square_type = env.GOAL
        if keys[pygame.K_TAB]:
            flags.append('start_sim')

        if keys[pygame.K_r]:
            grid_env.generate_random_grid(
                grid_env.num_rows, grid_env.num_cols, grid_env.obstacle_density)

        return flags

    def mouse_handler(self, grid_env: env.GridEnv) -> list[str]:
        flags = []

        square_width: int = self.win_width // grid_env.num_cols
        square_height: int = self.win_height // grid_env.num_rows

        mouse_pos: tuple[int, int] = pygame.mouse.get_pos()
        grid_pos: tuple[int, int] = (int(np.floor(mouse_pos[0] / square_width)),
                                     int(np.floor(mouse_pos[1] / square_height)))

        if self.edit_square_type == env.START:
            grid_env.start_pos = grid_pos
        elif self.edit_square_type == env.GOAL:
            grid_env.end_pos = grid_pos
        else:
            grid_env[grid_pos] = self.edit_square_type

        return flags
