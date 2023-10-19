import numpy as np
import environment as env

# Constants
MAX_INT32: int = np.iinfo(np.int32).max

# Heuristic functions:


def euclidian_dist(start_point: tuple[int, int], end_point: tuple[int, int]) -> float:
    """
    Gives the normal 2d Euclidian distance between 2 points
    """

    a = end_point[0] - start_point[0]
    b = end_point[1] - start_point[1]

    return np.sqrt(a*a + b*b)

# Path Planning Algorithms:


def a_star(grid_env: env.GridEnv, start_pos: tuple[int, int],
           end_pos: tuple[int, int], h_func: callable = euclidian_dist) -> list[tuple[int, int]]:

    open_close_set: np.array = np.full(grid_env.shape, np.NaN)
    open_close_set[start_pos] = env.OPEN_SET

    came_from: np.array = np.full(grid_env.shape, np.NaN, dtype=np.int32)

    g_score = np.full(grid_env.shape, MAX_INT32, dtype=np.int32)
    g_score[start_pos] = 0

    f_score = np.full(grid_env.shape, np.Inf)
    f_score[start_pos] = h_func(start_pos, end_pos)

    while np.any(open_close_set == env.OPEN_SET):
        f_mask = f_score.copy()
        f_mask[open_close_set != 1] = np.Inf

        current = np.unravel_index(np.argmin(f_mask), grid_env.shape)

        if current == end_pos:
            path = reconstruct_path(
                came_from, start_pos, end_pos, grid_env.shape)
            grid_env.overlay_grid = open_close_set
            grid_env.overlay_grid[np.isnan(grid_env.overlay_grid)] = 0
            for pos in path:
                grid_env.overlay_grid[pos] = env.PATH

            return path

        open_close_set[current] = 2

        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbor = (current[0] + i, current[1] + j)

                if (i == 0 and j == 0) or not is_valid_neighbor(grid_env, neighbor):
                    continue

                # dist of 1 if directly next to or sqrt(2) if diagonal
                grid_dist = 1 if (i == 0 or j == 0) else np.sqrt(2)
                temp_g = g_score[current] + grid_dist

                if temp_g < g_score[neighbor]:
                    came_from[neighbor] = np.ravel_multi_index(
                        current, grid_env.shape)

                    g_score[neighbor] = temp_g

                    f_score[neighbor] = temp_g + h_func(neighbor, end_pos)

                    open_close_set[neighbor] = 1

    grid_env.overlay_grid = open_close_set
    grid_env.overlay_grid[np.isnan(grid_env.overlay_grid)] = 0
    return None


def is_valid_neighbor(grid_env: env.GridEnv, neighbor: tuple[int, int]):
    # Checking bounds
    for i in range(len(neighbor)):
        if neighbor[i] < 0 or neighbor[i] >= grid_env.shape[i]:
            return False

    if grid_env[neighbor] == env.OBSTACLE:
        return False

    return True


def reconstruct_path(came_from: np.array, start_pos: tuple[int, int],
                     end_pos: tuple[int, int], grid_shape: tuple[int, int]) -> list[tuple[int, int]]:

    current = end_pos
    path = [current]

    while path[-1] != start_pos:
        current = np.unravel_index(came_from[current], grid_shape)
        path.append(current)

    return path


if __name__ == '__main__':
    grid_env = env.GridEnv(np.array([[0, 1, 0, 0, 0],
                                     [1, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 1],
                                     [0, 0, 0, 1, 1]]),
                           start_pos=(0, 0), end_pos=(4, 4))
    # grid_env.generate_random_grid(5, 5, 0.2)
    print(grid_env)

    path = a_star(grid_env, grid_env.start_pos, grid_env.end_pos)
    print(path)
