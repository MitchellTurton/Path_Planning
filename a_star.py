# Imports
import numpy as np
import time
from typing import List, Tuple

# Heuristic Functions


def euclidian_dist(start_point: Tuple[int], end_point: Tuple[int]) -> float:
    """
    Gives the normal 2d Euclidian distance between 2 points
    """
    a = end_point[0] - start_point[0]
    b = end_point[1] - start_point[1]

    return np.sqrt(a*a + b*b)


class AStar():
    """
    Implementation of the A* pathfinding algorithm on a grid.
    """

    def __init__(self, env_grid: np.array, heuristic_func=euclidian_dist) -> None:
        """
        Initialize the A* algorithm with a given environment grid and heuristic function.

        Args:
            env_grid (np.array): 2D array representing the environment.
            heuristic_func (Callable): Function to compute the heuristic. Defaults to euclidian_dist.
        """

        self.h_func = heuristic_func

        self.update_grid(env_grid)

    def update_grid(self, env_grid):
        """
        Update the internal state with a new environment grid.

        Args:
            env_grid (np.array): 2D array representing the environment.
        """

        self.env_grid: np.array = env_grid

        start_row, start_col = np.where(env_grid == 3)
        if start_row.size > 0 and start_col.size > 0:
            self.start_point = (start_row[0], start_col[0])
        else:
            self.start_point = (0, 0)

        end_row, end_col = np.where(env_grid == 4)
        if end_row.size > 0 and end_col.size > 0:
            self.end_point = (end_row[0], end_col[0])
        else:
            self.end_point = (env_grid.shape[0] - 1, env_grid.shape[1] - 1)

        self.open_close_set: np.array = np.full(env_grid.shape, np.NaN)
        self.open_close_set[self.start_point] = 1

        self.came_from = np.full(env_grid.shape, np.NaN, dtype=np.int32)

        self.g_score = np.full(env_grid.shape, np.inf)
        self.g_score[self.start_point] = 0

        self.f_score = np.full(env_grid.shape, np.inf)
        self.f_score[self.start_point] = self.h_func(
            self.start_point, self.end_point)

    def solve(self, is_logging=False) -> List[Tuple[int]]:
        """
        Solve the pathfinding problem using the A* algorithm.

        Args:
            is_logging (bool): If True, logs intermediate steps. Defaults to False.

        Returns:
            List[Tuple[int]]: List of grid positions forming the path from start to goal.
        """

        while np.any(self.open_close_set == 1):
            step_grid, done = self.step()

            if is_logging:
                print(f'Step Grid: \n{step_grid}\n\n')

            if done:
                return step_grid

        return self.open_close_set

    def step(self) -> np.array:  # Returns an np.array of the current closed and open set
        """
        Perform one step of the A* algorithm.

        Returns:
            np.array: 2D array representing the current state of the open and closed sets.
        """

        f_score_mask = self.f_score.copy()
        f_score_mask[self.open_close_set != 1] = np.Inf

        current = np.unravel_index(
            np.argmin(f_score_mask), self.env_grid.shape)

        if current == self.end_point:
            path = self.reconstruct_path()
            output = self.open_close_set.copy()

            for pos in path:
                output[pos] = 4

            return output, True

        self.open_close_set[current] = -1

        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbor = (current[0] + i, current[1] + j)

                if (i == 0 and j == 0) or (
                    neighbor[0] < 0 or neighbor[1] < 0 or
                    neighbor[0] >= self.env_grid.shape[0] or
                    self.env_grid[neighbor] == -1
                ):
                    continue

                grid_dist = 1 if i == 0 or j == 0 else np.sqrt(2)
                temp_g = self.g_score[current] + grid_dist

                if temp_g < self.g_score[neighbor]:
                    self.came_from[neighbor] = np.ravel_multi_index(
                        current, self.env_grid.shape)

                    self.g_score[neighbor] = temp_g

                    self.f_score[neighbor] = temp_g + \
                        self.h_func(neighbor, self.end_point)

                    self.open_close_set[neighbor] = 1

        return self.open_close_set, False

    def reconstruct_path(self) -> List[Tuple[int]]:
        """
        Reconstruct the path from the goal to the start using the internal state.

        Returns:
            List[Tuple[int]]: List of grid positions forming the path from start to goal.
        """

        current = self.end_point
        path = [current]

        while path[-1] != self.start_point:
            current = np.unravel_index(
                self.came_from[current], self.env_grid.shape)
            path.append(current)

        return path


if __name__ == '__main__':
    a_star = AStar(np.zeros((5, 5)))
    a_star.solve(is_logging=True)
