# Imports
import pygame
import numpy as np
from a_star import AStar

from typing import Tuple

# Initialize pygame
pygame.init()

# Color Constants
WHITE = (75, 75, 75)
BLACK = (30, 30, 35)
RED = (220, 80, 80)
GREEN = (0, 175, 39)
BLUE = (0, 157, 196)
ORANGE = (255, 165, 0)
# PURPLE = (150, 55, 230)
PURPLE = (0, 84, 170)

# Grid Square Types
EMPTY_SQUARE = 0
OBSTACLE = -1
OPEN_SET = 1
CLOSED_SET = 2
START = 3
GOAL = 4
PATH = 5

# Dictionary that maps grid type to color to display in environment
COLOR_DICT = {EMPTY_SQUARE: WHITE,
              OBSTACLE: BLACK,
              OPEN_SET: GREEN,
              CLOSED_SET: RED,
              START: BLUE,
              GOAL: ORANGE,
              PATH: PURPLE
              }

FPS = 1500


class BasicGridEnv():
    """
    Represents a basic grid environment for path planning simulations.
    """

    def __init__(self, win_title: str = "Grid", win_width: int = 800, win_height: int = 800,
                 num_rows: int = 25, num_cols: int = 25, obstacle_density: float = 0.0) -> None:
        """
        Initializes the grid environment with given parameters.
        """

        # -------------- Logic Variables ---------------
        self.grid: np.array = self.generate_random_grid(
            num_rows, num_cols, obstacle_density)
        # TODO: refactor the overlay grid system

        self.obstacle_density: int = obstacle_density

        # For a simple editor, tells the type that the grid square will be replaced with upon clicking
        self.edit_square_type: int = OBSTACLE

        self.path_planner = None

        # ------------- Graphics Variables -------------
        self.screen = pygame.display.set_mode((win_width, win_height))
        pygame.display.set_caption(win_title)
        self.clock = pygame.time.Clock()

        self.square_width: int = win_width // num_cols
        self.square_height: int = win_height // num_rows

        self.is_sim_complete = False
        self.mouse_down = False
        self.run = True

    def generate_random_grid(self, num_rows: int, num_cols: int, obstacle_density: float = None) -> np.array:
        """
        Generates a random grid based on given parameters.
        """

        if obstacle_density is None:
            obstacle_density = self.obstacle_density

        grid = np.random.choice([EMPTY_SQUARE, OBSTACLE], size=(
            num_rows, num_cols), p=(1-obstacle_density, obstacle_density))

        # grid[(0, 0)] = EMPTY_SQUARE
        # grid[(-1, -1)] = EMPTY_SQUARE

        start_pos = (np.random.randint(
            0, grid.shape[0]), np.random.randint(0, grid.shape[1]))
        while True:
            end_pos = (np.random.randint(
                0, grid.shape[0]), np.random.randint(0, grid.shape[1]))
            if start_pos[0] != end_pos[0] or start_pos[1] != end_pos[1]:
                break

        grid[start_pos] = START
        grid[end_pos] = GOAL

        return grid

    def reset(self):
        self.grid = self.generate_random_grid(
            self.grid.shape[0], self.grid.shape[1])

        self.path_planner = None

    def event_handler(self) -> None:
        """
        Handles pygame events like quitting, mouse clicks, and key presses.
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False

            if event.type == pygame.KEYDOWN:
                self.keyboard_handler()

        if self.mouse_down:
            self.mouse_handler()

    def keyboard_handler(self) -> None:
        """
        Handles key press events.
        """

        keys = pygame.key.get_pressed()

        if keys[pygame.K_SPACE]:
            self.edit_square_type = OBSTACLE if self.edit_square_type == EMPTY_SQUARE else EMPTY_SQUARE
        if keys[pygame.K_1]:
            self.edit_square_type = START
        if keys[pygame.K_2]:
            self.edit_square_type = GOAL
        if keys[pygame.K_TAB]:
            self.path_planner = AStar(self.grid)

        if keys[pygame.K_r]:
            self.reset()

    def mouse_handler(self):
        """
        Handles mouse click events to edit the grid.
        """

        mouse_pos: Tuple[int] = pygame.mouse.get_pos()
        grid_pos: Tuple[int] = (int(np.floor(mouse_pos[0] / self.square_width)),
                                int(np.floor(mouse_pos[1] / self.square_height)))

        self.grid[grid_pos] = self.edit_square_type

    def draw(self, overlay_grid: np.array = None) -> None:
        """
        Renders the grid on the pygame surface.
        """

        self.screen.fill(WHITE)

        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                if overlay_grid is not None and self.grid[row][col] == EMPTY_SQUARE and overlay_grid[row][col] > 0:
                    square_color = COLOR_DICT[overlay_grid[row][col]]
                else:
                    square_color = COLOR_DICT[self.grid[row][col]]
                square_params = (row * self.square_height, col * self.square_width,
                                 self.square_height, self.square_width)

                pygame.draw.rect(self.screen, square_color, square_params, 0)
                pygame.draw.rect(self.screen, BLACK, square_params, 1)

        pygame.display.flip()

        self.clock.tick(FPS)

    def update(self) -> bool:
        """
        Updates the grid environment, handles events, and draws the grid.
        """

        self.event_handler()

        overlay_grid = None
        if self.path_planner is not None:
            overlay_grid, self.is_sim_complete = self.path_planner.step()

        if self.is_sim_complete:
            self.draw(overlay_grid)


if __name__ == '__main__':
    env = BasicGridEnv("Grid", 1000, 1000, 50, 50, 0.5)

    while env.run:
        env.update()
