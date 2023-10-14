# Imports
import numpy as np
import math
from typing import List, Tuple
from environment import *

# Constants
MAX_FLOAT32 = np.finfo(np.float32).max


# Heuristic Functions
def h_dist(start_point: Tuple[int], end_point: Tuple[int]) -> float:
    return np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)


class AStar():

    def __init__(self, env_grid: np.array, heuristic_func=h_dist) -> None:
        self.env_grid: np.array = env_grid

        self.heuristic_func: function = heuristic_func

        self.start_point: Tuple[int] = None
        self.end_point: Tuple[int] = None

        for row in range(self.env_grid.shape[0]):
            for col in range(self.env_grid.shape[1]):
                if self.env_grid[row][col] == 3:
                    self.start_point = (row, col)
                elif self.env_grid[row][col] == 4:
                    self.end_point = (row, col)

                if self.start_point and self.end_point:
                    break
            if self.start_point and self.end_point:
                break

        if not self.start_point:
            self.start_point = (0, 0)

        if not self.end_point:
            self.end_point = (
                self.env_grid.shape[0] - 1, self.env_grid.shape[1] - 1)

        self.open_close_map = np.full(self.env_grid.shape, np.NaN)
        self.open_close_map[self.start_point[0]][self.start_point[1]] = 1
        self.came_from = np.full(self.env_grid.shape, np.NaN)

        self.g_score = np.full(env.shape, np.inf)
        self.g_score[self.start_point[0]][self.start_point[1]] = 0

        self.f_score = np.full(env.shape, np.inf)
        self.f_score[self.start_point[0]][self.start_point] = self.h_func(
            self.start_point, self.end_point)

    def reconstruct_path(self) -> List[Tuple[int]]:
        pass

    def step(self) -> np.array:
        current = np.unravel_indexed(
            np.argmax(self.f_score[self.open_close_map == 1]), self.env_grid.shape)

        if current == self.goal:
            return self.reconstruct_path()

        self.open_close_map[current[0]][current[1]] = -1

        for i in range(-1, 1):
            for j in range(-1, 1):

                if i == 0 and j == 0:
                    continue

                neighbor = (current[0] + i, current[1] + j)
                grid_dist = 1 if i == 0 and j == 0 else np.sqrt(2)
                temp_g = self.g_score[current[0]][current[1]] + grid_dist

                if temp_g < self.g_score[neighbor[0]][neighbor[1]]:
                    self.came_from = np.ravel_multi_index(
                        current, self.env_grid.shape)

                    self.g_score[neighbor[0]][neighbor[1]] = temp_g

                    self.f_score[neighbor[0]][neighbor[1]] = temp_g + \
                        self.h_func(neighbor, self.end_point)

                    self.open_close_map[neighbor[0]][neighbor[1]] = 1

        return self.open_close_map

    def solve(self) -> List[Tuple[int]]:
        pass

    # def gen_heuristic_grid(self, env_grid: np.array = None, heuristic_function=None,
    #                        start_point: Tuple[int] = None, end_point: Tuple[int] = None) -> np.array:
    #     """
    #     Generates the "cost" in order to make an educated guess, in this case a simple distance formula
    #     """

    #     if env_grid == None:
    #         env_grid = self.env_grid

    #     if heuristic_function == None:
    #         heuristic_function = self.heuristic_func

    #     if start_point == None:
    #         start_point = self.start_point

    #     if end_point == None:
    #         end_point = self.end_point

    #     print(f'Start: {start_point}, End: {end_point}')

    #     h_grid = np.zeros(env_grid.shape, float)

    #     for row in range(env_grid.shape[0]):
    #         for col in range(env_grid.shape[1]):
    #             if env_grid[row][col] == -1:
    #                 # If the position is an obstacle in the environment assign it a value of INF
    #                 h_grid[row][col] = MAX_FLOAT32
    #             else:
    #                 # If not an obstacle calulate the distance to node
    #                 h_grid[row][col] = heuristic_function(row, col, end_point)

    #     return h_grid


if __name__ == "__main__":
    env = BasicGridEnv(num_rows=5, num_cols=5, obstacle_density=0)
    a_star = AStar(env.grid, h_dist)

    print(a_star.gen_heuristic_grid())
