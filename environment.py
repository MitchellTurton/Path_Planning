# Imports
import pygame
import numpy as np
from a_star import AStar

from typing import Tuple

# Initialize pygame
pygame.init()

# Color Constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (196, 39, 0)
GREEN = (0, 196, 39)
BLUE = (0, 157, 196)
ORANGE = (255, 165, 0)

# Grid Square Types
EMPTY_SQUARE = 0
OBSTACLE = -1
OPEN_SET = 1
CLOSED_SET = 2
START = 3
GOAL = 4

# Dictionary that maps grid type to color to display in environment
COLOR_DICT = {EMPTY_SQUARE: WHITE,
              OBSTACLE: BLACK,
              OPEN_SET: GREEN,
              CLOSED_SET: RED,
              START: BLUE,
              GOAL: ORANGE
              }

FPS = 100


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

        self.run_sim = True
        self.mouse_down = False
        self.is_path_planning = False

    def generate_random_grid(self, num_rows: int, num_cols: int, obstacle_density: float = None) -> np.array:
        """
        Generates a random grid based on given parameters.
        """

        if obstacle_density is None:
            obstacle_density = self.obstacle_density

        return np.random.choice([EMPTY_SQUARE, OBSTACLE], size=(num_rows, num_cols),
                                p=(1-obstacle_density, obstacle_density))

    def event_handler(self) -> None:
        """
        Handles pygame events like quitting, mouse clicks, and key presses.
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run_sim = False

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

        keys = pygame.ket.get_pressed()

        if keys[pygame.K_SPACE]:
            self.edit_square_type = OBSTACLE if self.edit_square_type == EMPTY_SQUARE else EMPTY_SQUARE
        if keys[pygame.K_1]:
            self.edit_square_type = START
        if keys[pygame.K_2]:
            self.edit_square_type = GOAL
        if keys[pygame.K_KP_ENTER]:
            self.is_path_planning = True
            self.path_planner = AStar(self.grid)

        if keys[pygame.K_r]:
            self.generate_random_grid(self.grid.shape[0], self.grid.shape[1])

    def mouse_handler(self):
        """
        Handles mouse click events to edit the grid.
        """

        mouse_pos: Tuple[int] = pygame.mouse.get_pos()
        grid_pos: Tuple[int] = (np.floor(mouse_pos[0] / self.square_width),
                                np.floor(mouse_pos[1] / self.square_height))

        self.grid[grid_pos] = self.editor_square_type

    def draw(self) -> None:
        """
        Renders the grid on the pygame surface.
        """

        self.screen.fill(WHITE)

        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
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

        output = self.path_planner.step()  # TODO: Think on this interface more
