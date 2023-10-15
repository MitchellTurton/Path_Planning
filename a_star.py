# Imports
import numpy as np
import math
from typing import List, Tuple
import time


# Heuristic Functions
def h_dist(start_point: Tuple[int], end_point: Tuple[int]) -> float:
    return np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)


class AStar():

    def __init__(self, env_grid: np.array, heuristic_func=h_dist) -> None:
        self.env_grid: np.array = env_grid

        self.h_func: function = heuristic_func

        self.update_grid()

        self.open_close_map = np.full(self.env_grid.shape, np.NaN)
        self.open_close_map[self.start_point[0]][self.start_point[1]] = 1
        self.came_from = np.full(self.env_grid.shape, np.NaN, dtype=np.int32)

        self.g_score = np.full(self.env_grid.shape, np.inf)
        self.g_score[self.start_point[0]][self.start_point[1]] = 0

        self.f_score = np.full(self.env_grid.shape, np.inf)
        self.f_score[self.start_point[0]][self.start_point[1]] = self.h_func(
            self.start_point, self.end_point)

    def update_grid(self):
        self.start_point: Tuple[int] = None
        self.end_point: Tuple[int] = None

        print(self.env_grid)

        for row in range(self.env_grid.shape[0]):
            for col in range(self.env_grid.shape[1]):
                if self.env_grid[row][col] == 3:
                    print('new start')
                    self.start_point = (row, col)
                elif self.env_grid[row][col] == 4:
                    print('new end')
                    self.end_point = (row, col)

                if self.start_point is not None and self.end_point is not None:
                    break
            if self.start_point is not None and self.end_point is not None:
                break

        if self.start_point is None:
            print('no new start')
            self.start_point = (0, 0)

        if self.end_point is None:
            print('no new end')
            self.end_point = (
                self.env_grid.shape[0] - 1, self.env_grid.shape[1] - 1)

    def reconstruct_path(self, current=None, came_from: np.array = None) -> List[Tuple[int]]:
        # print(f"came_from: \n{self.came_from}\n\n")
        if current is None:
            current = self.end_point

        if came_from is None:
            came_from = self.came_from

        path = [current]
        for _ in range(self.env_grid.size):
            # print(f"i: {current[0]}, j: {current[1]}")
            # print(f'came_from_val: {self.came_from[current[0]][current[1]]}')
            current = np.unravel_index(
                self.came_from[current[0]][current[1]], self.env_grid.shape)
            # print(f'new_current: {current}')
            path.append(current)

            if current == self.start_point:
                break

        return path

    def step(self) -> np.array:
        # print("----------------------------------------------------------------")
        masked_arr = self.f_score.copy()
        masked_arr[self.open_close_map != 1] = np.Inf
        current = np.unravel_index(
            np.argmin(masked_arr), self.env_grid.shape)

        # print(f'\nCurrent: {current}\n\n')
        # print(f'Mask: \n\n{self.open_close_map == 1}\n\n\n')
        # print(
        #     f'Masked: \n\n{masked_arr}\n\n\n')

        if current == self.end_point:
            path = self.reconstruct_path()
            output = self.open_close_map.copy()

            for i, j in path:
                output[i][j] = 4

            return output

        self.open_close_map[current[0]][current[1]] = -1

        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbor = (current[0] + i, current[1] + j)

                if (i == 0 and j == 0) or (
                        neighbor[0] < 0 or neighbor[1] < 0 or
                        neighbor[0] > self.env_grid.shape[0] - 1 or
                        neighbor[1] > self.env_grid.shape[1] - 1) or (
                        self.env_grid[neighbor[0]][neighbor[1]] == -1):
                    continue

                neighbor = (current[0] + i, current[1] + j)
                grid_dist = 1 if i == 0 or j == 0 else np.sqrt(2)
                temp_g = self.g_score[current[0]][current[1]] + grid_dist

                if temp_g < self.g_score[neighbor[0]][neighbor[1]]:
                    self.came_from[neighbor[0]][neighbor[1]] = np.ravel_multi_index(
                        current, self.env_grid.shape)

                    self.g_score[neighbor[0]][neighbor[1]] = temp_g

                    self.f_score[neighbor[0]][neighbor[1]] = temp_g + \
                        self.h_func(neighbor, self.end_point)

                    self.open_close_map[neighbor[0]][neighbor[1]] = 1

        # print(f'open_close_map: \n\n{self.open_close_map}\n\n\n')
        # print(f'f_score       : \n\n{self.f_score}\n\n\n')
        # print(f'g_score       : \n\n{self.g_score}\n\n\n')

        return self.open_close_map

    def solve(self, disp_env=None) -> List[Tuple[int]]:
        while np.any(self.open_close_map == 1):
            step_grid = self.step()
            print(step_grid)
            print("\n\n")
            if disp_env is not None:
                disp_env.overlay_grid = step_grid
            else:
                # print(step_grid)
                # print()
                pass

            time.sleep(0.125)


if __name__ == "__main__":
    a_star = AStar(np.zeros((5, 5)))
    a_star.solve()
