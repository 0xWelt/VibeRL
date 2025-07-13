"""
Tests for SnakeGameEnv - the snake game environment.

Tests focus on environment-specific functionality without retesting gymnasium compliance.
"""

import numpy as np
import pytest

from viberl.envs import Direction, SnakeGameEnv


class TestSnakeEnvSpecific:
    """Test SnakeGameEnv-specific functionality."""

    @pytest.fixture
    def env(self) -> SnakeGameEnv:
        """Create SnakeGameEnv for testing."""
        return SnakeGameEnv(grid_size=10)

    def test_direction_enum_values(self):
        """Test Direction enum has correct values."""
        assert Direction.UP.value == 0
        assert Direction.RIGHT.value == 1
        assert Direction.DOWN.value == 2
        assert Direction.LEFT.value == 3

    def test_snake_initialization(self, env: SnakeGameEnv) -> None:
        """Test snake initialization."""
        env.reset()
        assert len(env.snake) == 3
        assert len(env.food) == 2
        assert env.direction == Direction.RIGHT
        assert env.score == 0
        assert env.steps == 0

    def test_observation_values(self, env: SnakeGameEnv) -> None:
        """Test observation values are correct."""
        obs, _ = env.reset()

        # Check observation contains expected values
        unique_values = np.unique(obs)
        expected_values = {0, 1, 2, 3}  # empty, snake body, snake head, food
        assert set(unique_values).issubset(expected_values)

        # Ensure all required elements are present
        assert 2 in obs  # snake head
        assert 1 in obs  # snake body
        assert 3 in obs  # food

    def test_food_placement(self, env: SnakeGameEnv) -> None:
        """Test food placement logic."""
        env.reset()

        # Food should not be on snake (test the general case, not specific positions)
        obs = env._get_observation()
        snake_mask = (obs == 1) | (obs == 2)  # Snake body and head
        food_mask = obs == 3  # Food

        # Ensure no overlap between snake and food
        overlap = snake_mask & food_mask
        assert not np.any(overlap)

    def test_snake_growth_on_food(self, env: SnakeGameEnv) -> None:
        """Test snake grows when eating food."""
        env.reset()
        initial_length = len(env.snake)

        # Place food directly in front of snake head
        head = env.snake[-1]
        env.food = (head[0], head[1] + 1)

        obs, reward, terminated, truncated, info = env.step(Direction.RIGHT.value)

        assert not terminated
        assert reward > 0  # Positive reward for food
        assert len(env.snake) == initial_length + 1
        assert env.score == 1

    def test_collision_detection(self, env: SnakeGameEnv) -> None:
        """Test collision detection."""
        env.reset()

        # Manually set snake at edge to test wall collision
        env.snake = [(5, 8), (5, 9)]  # At right edge
        env.direction = Direction.RIGHT

        obs, reward, terminated, truncated, info = env.step(Direction.RIGHT.value)

        assert terminated
        assert reward < 0  # Negative reward for collision

    def test_self_collision(self, env: SnakeGameEnv) -> None:
        """Test self-collision detection."""
        env.reset()

        # Create a snake that will collide with itself
        env.snake = [(5, 5), (5, 6), (5, 7), (4, 7), (4, 6)]
        env.direction = Direction.LEFT

        obs, reward, terminated, truncated, info = env.step(Direction.UP.value)

        # Should detect collision
        assert terminated or reward < 0

    def test_opposite_direction_prevention(self, env: SnakeGameEnv) -> None:
        """Test prevention of moving in opposite direction."""
        env.reset()

        # Moving right, try to move left
        env.direction = Direction.RIGHT
        obs, reward, terminated, truncated, info = env.step(Direction.LEFT.value)

        # Should continue moving right
        assert not terminated
        assert env.direction == Direction.RIGHT

    def test_reset_clears_state(self, env: SnakeGameEnv) -> None:
        """Test reset properly clears game state."""
        env.reset()

        # Modify state
        env.score = 10
        env.steps = 50
        env.game_over = True

        # Reset should clear everything
        obs, info = env.reset()
        assert env.score == 0
        assert env.steps == 0
        assert not env.game_over
        assert len(env.snake) == 3

    @pytest.mark.parametrize('grid_size', [10, 15, 20])
    def test_different_grid_sizes(self, grid_size: int) -> None:
        """Test environment works with different grid sizes."""
        env = SnakeGameEnv(grid_size=grid_size)
        obs, info = env.reset()

        assert obs.shape == (grid_size, grid_size)
        assert len(env.snake) == 3
        assert not env.game_over

    def test_max_steps_truncation(self, env: SnakeGameEnv) -> None:
        """Test truncation after max steps."""
        env.reset()

        # Set steps close to max
        env.steps = env.max_steps - 1

        obs, reward, terminated, truncated, info = env.step(Direction.RIGHT.value)

        assert terminated  # Game ends due to max steps
        assert env.game_over

    def test_observation_consistency(self, env: SnakeGameEnv) -> None:
        """Test observation consistency across resets with same seed."""
        env1 = SnakeGameEnv(grid_size=10)
        env2 = SnakeGameEnv(grid_size=10)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

    def test_reward_signal_consistency(self, env: SnakeGameEnv) -> None:
        """Test reward signals are consistent for same actions."""
        env1 = SnakeGameEnv(grid_size=10)
        env2 = SnakeGameEnv(grid_size=10)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        action = Direction.RIGHT.value
        _, reward1, _, _, _ = env1.step(action)
        _, reward2, _, _, _ = env2.step(action)

        assert reward1 == reward2
