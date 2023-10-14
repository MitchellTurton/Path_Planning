# Importts
import pygame
import numpy as np
import math
from a_star import *

# Initializng pygame
pygame.init()

# Color Constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (196, 39, 0)
GREEN = (0, 196, 39)
BLUE = (0, 157, 196)
ORANGE = (255, 165, 0)

# Grid Square Types
OPEN_SPACE = 0
OBSTACLE = -1
SEARCHED = 1
EVALUATED = 2
START = 3
GOAL = 4

# Dictionaries mapping grid type to color
color_dict = {OPEN_SPACE: WHITE,
              OBSTACLE: BLACK,
              SEARCHED: GREEN,
              EVALUATED: RED,
              START: BLUE,
              GOAL: ORANGE
              }


class BasicGridEnv():
    """
    Class containing a basic representation of a grid world environment,
    as well as methods to display and generate this environment.
    """

    def __init__(self, win_title: str = "Grid", win_width: int = 800,
                 win_height: str = 800, num_rows: int = 25, num_cols: int = 25, obstacle_density: float = 0.0) -> None:
        """
        Initialize the grid environment class.

        Args:
            win_title (str, optional): Title of screen. Defaults to "Grid".
            win_width (int, optional): Width of screen. Defaults to 800.
            win_height (str, optional): Height of screen. Defaults to 800.
            num_rows (int, optional): Num of rows of squares. Defaults to 25.
            num_cols (int, optional): Num of columns of squares. Defaults to 25.
            obstacle_density (float, optional): Percentage of obstacles to open spaces. Defaults to 0.0.
        """

        self.win_width = win_width
        self.win_height = win_height
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.square_width = self.win_width // self.num_cols
        self.square_height = self.win_height // self.num_rows

        self.obstacle_density = obstacle_density
        self.editor_square_type = OBSTACLE

        self.grid = self.generate_random_grid()

        self.screen = pygame.display.set_mode(
            (self.win_width, self.win_height))
        pygame.display.set_caption(win_title)
        self.clock = pygame.time.Clock()

    def draw(self, overlay_grid: np.array = None):
        """
        Visualizes a representation of the grid environment
        """

        if overlay_grid is not None:
            display_grid = self.grid

            for i in range(overlay_grid.shape[0]):
                for j in range(overlay_grid.shape[1]):
                    if overlay_grid[i][j] == 1 and self.grid[i][j] != OBSTACLE:
                        display_grid[i][j] = SEARCHED
                    if overlay_grid[i][j] == -1 and self.grid[i][j] != OBSTACLE:
                        display_grid[i][j] = EVALUATED

        self.screen.fill(WHITE)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                color = color_dict[self.grid[i][j]]

                pygame.draw.rect(self.screen, color,
                                 (i * self.square_height, j * self.square_width,
                                  self.square_height, self.square_width), 0)

                pygame.draw.rect(self.screen, BLACK,
                                 (i * self.square_height, j * self.square_width,
                                  self.square_height, self.square_width), 1)

        pygame.display.flip()

        self.clock.tick(60)

    def mouse_handler(self):
        """
        Handles user mouse input, used to edit the environment
        """

        mouse_pos = pygame.mouse.get_pos()
        row = math.floor(mouse_pos[0] / self.square_width)
        col = math.floor(mouse_pos[1] / self.square_height)
        self.grid[row][col] = self.editor_square_type

    def generate_random_grid(self, obstacle_density: float = None) -> np.array:
        """
        Using the current grid size, create a new random grid of clear spaces
        and obstacles.

        Args:
            obstacle_density (float, optional): 
               -Percentage of spaces that are obstacles in the grid compared to clear 
                spaces. 
               -Defaults to None.
                   -If None sets obstacle density to class obstacle density
        """

        if obstacle_density is None:
            obstacle_density = self.obstacle_density

        output = np.random.choice([OPEN_SPACE, OBSTACLE], size=(
            self.num_rows, self.num_cols), p=[1 - obstacle_density, obstacle_density])

        return output


if __name__ == "__main__":
    env = BasicGridEnv("Basic Grid", 800, 800, 25, 25, 0.15)
    a_star = AStar(env.grid)

    mouse_down = False
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True

            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False

            elif event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    env.editor_square_type = OBSTACLE
                if keys[pygame.K_1]:
                    env.editor_square_type = SEARCHED
                if keys[pygame.K_2]:
                    env.editor_square_type = EVALUATED
                if keys[pygame.K_3]:
                    env.editor_square_type = START
                if keys[pygame.K_4]:
                    env.editor_square_type = GOAL
                if keys[pygame.K_0]:
                    env.editor_square_type = OPEN_SPACE
                # if keys[pygame.K_TAB]:

                if keys[pygame.K_r]:
                    env.generate_random_grid()

        if mouse_down:
            env.mouse_handler()

        env.draw()

    pygame.quit()
