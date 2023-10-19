import numpy as np
from dataclasses import dataclass, field

# Grid Square Types:

EMPTY_SQUARE = 0
OBSTACLE = -1
OPEN_SET = 1
CLOSED_SET = 2
START = 3
GOAL = 4
PATH = 5


@dataclass(slots=True)
class GridEnv:
    grid: np.array
    overlay_grid: np.array = field(init=False)
    start_pos: tuple[int, int] = (0, 0)
    end_pos: tuple[int, int] = (-1, -1)
    obstacle_density: float = 0.0

    def __post_init__(self):
        self.overlay_grid = np.zeros(self.grid.shape)

    def generate_random_grid(self, num_rows: int, num_cols: int,
                             obstacle_density: float) -> None:

        self.grid = np.random.choice([EMPTY_SQUARE, OBSTACLE],
                                     size=(num_rows, num_cols),
                                     p=(1-obstacle_density, obstacle_density))
        self.obstacle_density = obstacle_density
        # return GridEnv(grid, obstacle_density=obstacle_density)

        self.overlay_grid = np.zeros(self.grid.shape)

        self.generate_random_end_points()

    def generate_random_end_points(self):
        self.start_pos = (np.random.randint(
            0, self.grid.shape[0]), np.random.randint(0, self.grid.shape[1]))
        self.grid[self.start_pos] = EMPTY_SQUARE
        while True:
            self.end_pos = (np.random.randint(
                0, self.grid.shape[0]), np.random.randint(0, self.grid.shape[1]))
            if self.start_pos[0] != self.end_pos[0] or self.start_pos[1] != self.end_pos[1]:
                self.grid[self.end_pos] = EMPTY_SQUARE
                break

    @property
    def num_rows(self) -> int:
        return self.grid.shape[0]

    @property
    def num_cols(self) -> int:
        return self.grid.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def __getitem__(self, index: tuple[int, int]) -> int:
        return self.grid[index]

    def __setitem__(self, index: tuple[int, int], val: int) -> None:
        self.grid[index] = val


if __name__ == "__main__":
    # env: GridEnv = GridEnv.generate_random_grid(10, 10, 0.2)
    env = GridEnv(np.zeros((1, 1)))
    env.generate_random_grid(10, 10, 0.2)
    print(env)
