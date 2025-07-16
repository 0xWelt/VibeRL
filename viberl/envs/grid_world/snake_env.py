from enum import Enum
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class SnakeGameEnv(gym.Env):
    metadata: ClassVar[dict[str, int | list[str]]] = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10,
    }

    def __init__(self, render_mode: str | None = None, grid_size: int = 20):
        super().__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode

        # Action space: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = spaces.Discrete(4)

        # Observation space: grid with snake body, food, and empty spaces
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(grid_size, grid_size), dtype=np.uint8
        )

        # Initialize game state
        self.reset()

        # Pygame setup for rendering
        self.window = None
        self.clock = None
        self.cell_size = 20
        self.window_size = self.grid_size * self.cell_size

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Initialize snake at center of grid
        center = self.grid_size // 2
        self.snake = [(center - 1, center), (center, center), (center + 1, center)]
        self.direction = Direction.RIGHT

        # Place food
        self.food = self._place_food()

        # Game state
        self.game_over = False
        self.score = 0
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 4

        return self._get_observation(), self._get_info()

    def _place_food(self) -> tuple[int, int]:
        """Place food at a random empty location."""
        empty_cells = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) not in self.snake
        ]

        if not empty_cells:
            return (0, 0)  # Should not happen in normal gameplay

        return empty_cells[np.random.randint(len(empty_cells))]

    def _get_observation(self) -> np.ndarray:
        """Get current state as a grid."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Mark snake body (1)
        for segment in self.snake[:-1]:
            grid[segment[0], segment[1]] = 1

        # Mark snake head (2)
        if self.snake:
            head = self.snake[-1]
            grid[head[0], head[1]] = 2

        # Mark food (3)
        if self.food:
            grid[self.food[0], self.food[1]] = 3

        return grid

    def _get_info(self) -> dict[str, Any]:
        """Get additional info about the current state."""
        return {
            'score': self.score,
            'steps': self.steps,
            'snake_length': len(self.snake),
            'game_over': self.game_over,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1

        # Convert action to direction (ensure snake doesn't reverse into itself)
        new_direction = Direction(action)

        # Prevent moving directly opposite to current direction
        if (
            (new_direction == Direction.UP and self.direction == Direction.DOWN)
            or (new_direction == Direction.DOWN and self.direction == Direction.UP)
            or (new_direction == Direction.LEFT and self.direction == Direction.RIGHT)
            or (new_direction == Direction.RIGHT and self.direction == Direction.LEFT)
        ):
            new_direction = self.direction

        self.direction = new_direction

        # Move snake
        head_x, head_y = self.snake[-1]

        if self.direction == Direction.UP:
            new_head = (head_x - 1, head_y)
        elif self.direction == Direction.RIGHT:
            new_head = (head_x, head_y + 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x + 1, head_y)
        elif self.direction == Direction.LEFT:
            new_head = (head_x, head_y - 1)

        # Check collision with walls
        if (
            new_head[0] < 0
            or new_head[0] >= self.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.grid_size
        ):
            self.game_over = True
            return self._get_observation(), -10.0, True, False, self._get_info()

        # Check collision with self
        if new_head in self.snake[:-1]:
            self.game_over = True
            return self._get_observation(), -10.0, True, False, self._get_info()

        # Move snake
        self.snake.append(new_head)

        reward = 0.0

        # Check if food eaten
        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self.food = self._place_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop(0)

        # Give negative reward to encourage faster completion
        reward += -0.1

        # Check if maximum steps reached
        if self.steps >= self.max_steps:
            self.game_over = True
            return self._get_observation(), reward, True, False, self._get_info()

        return self._get_observation(), reward, False, False, self._get_info()

    def render(self) -> None | np.ndarray:
        if self.render_mode is None:
            return None

        if self.render_mode == 'rgb_array':
            return self._render_frame()
        if self.render_mode == 'human':
            self._render_frame()

    def _render_frame(self) -> None | np.ndarray:
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption('Snake Game')

        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((50, 50, 50))  # Dark gray background

        # Draw grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                (100, 100, 100),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.window_size),
                1,
            )

        for y in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                (100, 100, 100),
                (0, y * self.cell_size),
                (self.window_size, y * self.cell_size),
                1,
            )

        # Draw snake
        for i, segment in enumerate(self.snake):
            color = (
                (0, 200, 0) if i < len(self.snake) - 1 else (0, 255, 0)
            )  # Green body, brighter head
            pygame.draw.rect(
                canvas,
                color,
                (
                    segment[1] * self.cell_size + 1,
                    segment[0] * self.cell_size + 1,
                    self.cell_size - 2,
                    self.cell_size - 2,
                ),
            )

        # Draw food
        if self.food:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),  # Red
                (
                    self.food[1] * self.cell_size + 1,
                    self.food[0] * self.cell_size + 1,
                    self.cell_size - 2,
                    self.cell_size - 2,
                ),
            )

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        canvas.blit(score_text, (10, 10))

        if self.game_over:
            game_over_text = font.render('Game Over! Press R to restart', True, (255, 255, 0))
            text_rect = game_over_text.get_rect(
                center=(self.window_size // 2, self.window_size // 2)
            )
            canvas.blit(game_over_text, text_rect)

        if self.render_mode == 'human':
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
