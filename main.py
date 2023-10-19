import numpy as np

import environment as env
import path_planner
import visulaizer

if __name__ == '__main__':
    grid_env = env.GridEnv(np.zeros((1, 1)))
    grid_env.generate_random_grid(50, 50, 0.5)
    path = []

    display = visulaizer.EnvVisualizer()

    while display.is_running:
        flags = display.event_handler(grid_env)

        for flag in flags:
            if flag == 'start_sim':
                path = path_planner.a_star(
                    grid_env, grid_env.start_pos, grid_env.end_pos)

        display.display_env(grid_env)
