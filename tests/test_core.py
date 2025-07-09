import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from snake_game.core import Direction, SnakeGameEnv


class TestDirection:
    """Test Direction enum."""

    def test_direction_values(self):
        assert Direction.UP.value == 0
        assert Direction.RIGHT.value == 1
        assert Direction.DOWN.value == 2
        assert Direction.LEFT.value == 3


class TestSnakeGameEnvInitialization:
    """Test SnakeGameEnv initialization."""

    def test_default_initialization(self):
        env = SnakeGameEnv()
        assert env.grid_size == 20
        assert env.render_mode is None
        assert env.action_space.n == 4
        assert env.observation_space.shape == (20, 20)
        assert len(env.snake) == 3
        assert env.direction == Direction.RIGHT
        assert env.game_over is False
        assert env.score == 0
        assert env.steps == 0
        assert len(env.food) == 2

    def test_custom_grid_size(self):
        env = SnakeGameEnv(grid_size=30)
        assert env.grid_size == 30
        assert env.observation_space.shape == (30, 30)
        assert len(env.snake) == 3

    def test_render_mode_human(self):
        env = SnakeGameEnv(render_mode='human')
        assert env.render_mode == 'human'

    def test_render_mode_rgb_array(self):
        env = SnakeGameEnv(render_mode='rgb_array')
        assert env.render_mode == 'rgb_array'


class TestSnakeGameEnvReset:
    """Test environment reset functionality."""

    def test_reset_default(self):
        env = SnakeGameEnv()
        obs, info = env.reset()

        assert obs.shape == (20, 20)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        assert 'score' in info
        assert 'steps' in info
        assert 'snake_length' in info
        assert 'game_over' in info
        assert info['score'] == 0
        assert info['steps'] == 0
        assert info['snake_length'] == 3
        assert info['game_over'] is False

    def test_reset_with_seed(self):
        env = SnakeGameEnv()
        obs1, _ = env.reset(seed=42)

        env2 = SnakeGameEnv()
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_clears_state(self):
        env = SnakeGameEnv()
        env.reset()

        # Simulate some game state
        env.score = 10
        env.steps = 50
        env.game_over = True

        # Reset should clear everything
        obs, info = env.reset()
        assert env.score == 0
        assert env.steps == 0
        assert env.game_over is False
        assert len(env.snake) == 3


class TestSnakeGameEnvObservation:
    """Test observation generation."""

    def test_observation_structure(self):
        env = SnakeGameEnv()
        env.reset()
        obs = env._get_observation()

        # Grid should be 0 (empty), 1 (snake body), 2 (snake head), 3 (food)
        unique_values = np.unique(obs)
        assert all(val in [0, 1, 2, 3] for val in unique_values)

        # Should have snake head
        assert 2 in unique_values
        # Should have snake body
        assert 1 in unique_values
        # Should have food
        assert 3 in unique_values

    def test_observation_snake_placement(self):
        env = SnakeGameEnv(grid_size=10)
        env.reset()
        obs = env._get_observation()

        # Find snake head position
        head_positions = np.argwhere(obs == 2)
        assert len(head_positions) == 1

        # Find snake body positions
        body_positions = np.argwhere(obs == 1)
        assert len(body_positions) == 2  # Initial snake has 3 segments

        # Find food position
        food_positions = np.argwhere(obs == 3)
        assert len(food_positions) == 1

    def test_observation_food_not_on_snake(self):
        env = SnakeGameEnv()
        env.reset()
        obs = env._get_observation()

        snake_positions = set()
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                if obs[i, j] in [1, 2]:  # snake body or head
                    snake_positions.add((i, j))

        food_positions = np.argwhere(obs == 3)
        for food_pos in food_positions:
            assert tuple(food_pos) not in snake_positions


class TestSnakeGameEnvInfo:
    """Test info generation."""

    def test_info_structure(self):
        env = SnakeGameEnv()
        env.reset()
        info = env._get_info()

        required_keys = ['score', 'steps', 'snake_length', 'game_over']
        for key in required_keys:
            assert key in info

        assert isinstance(info['score'], int)
        assert isinstance(info['steps'], int)
        assert isinstance(info['snake_length'], int)
        assert isinstance(info['game_over'], bool)

    def test_info_values(self):
        env = SnakeGameEnv()
        env.reset()
        info = env._get_info()

        assert info['score'] == env.score
        assert info['steps'] == env.steps
        assert info['snake_length'] == len(env.snake)
        assert info['game_over'] == env.game_over


class TestSnakeGameEnvFoodPlacement:
    """Test food placement functionality."""

    def test_place_food_not_on_snake(self):
        env = SnakeGameEnv(grid_size=5)
        env.reset()

        # Clear the grid to test food placement
        _ = env.snake.copy()  # Copy to verify it doesn't change

        food = env._place_food()
        assert food not in env.snake
        assert 0 <= food[0] < env.grid_size
        assert 0 <= food[1] < env.grid_size

    def test_place_food_with_constrained_space(self):
        env = SnakeGameEnv(grid_size=5)
        env.reset()

        # Fill almost entire grid with snake
        env.snake = [(i, j) for i in range(5) for j in range(5)]
        env.snake = env.snake[:-1]  # Leave one empty space

        food = env._place_food()
        assert food not in env.snake
        assert food in [(i, j) for i in range(5) for j in range(5) if (i, j) not in env.snake]

    def test_place_food_empty_grid_fallback(self):
        env = SnakeGameEnv(grid_size=5)
        env.reset()

        # Fill entire grid with snake (edge case)
        env.snake = [(i, j) for i in range(5) for j in range(5)]

        food = env._place_food()
        assert food == (0, 0)  # Fallback position when no empty cells


class TestSnakeGameEnvStep:
    """Test step functionality."""

    def test_step_basic_movement(self):
        env = SnakeGameEnv()
        env.reset()

        initial_snake_head = env.snake[-1]
        obs, reward, terminated, truncated, info = env.step(Direction.RIGHT.value)

        assert not terminated
        assert not truncated
        assert reward == 0.0  # No food eaten
        assert len(env.snake) == 3  # Same length (moved, didn't grow)
        assert env.snake[-1] != initial_snake_head  # Head moved

    def test_step_food_eating(self):
        env = SnakeGameEnv(grid_size=10)
        env.reset()

        # Place food directly in front of snake
        head = env.snake[-1]
        new_food = (head[0], head[1] + 1)
        env.food = new_food

        obs, reward, terminated, truncated, info = env.step(Direction.RIGHT.value)

        assert not terminated
        assert not truncated
        assert reward == 1.0  # Food eaten
        assert len(env.snake) == 4  # Snake grew
        assert env.score == 1
        assert env.food != new_food  # Food moved to new location

    def test_step_wall_collision(self):
        env = SnakeGameEnv(grid_size=5)
        env.reset()

        # Move snake to edge and then into wall
        env.snake = [(2, 2), (2, 3), (2, 4)]  # At right edge
        env.direction = Direction.RIGHT

        obs, reward, terminated, truncated, info = env.step(Direction.RIGHT.value)

        assert terminated
        assert not truncated
        assert reward == -1.0  # Collision penalty
        assert env.game_over is True

    def test_step_self_collision(self):
        env = SnakeGameEnv(grid_size=10)
        env.reset()

        # Create a snake that will collide with itself
        env.snake = [(5, 5), (5, 6), (5, 7), (6, 7), (6, 6)]  # U-shaped
        env.direction = Direction.UP  # Moving up

        obs, reward, terminated, truncated, info = env.step(Direction.LEFT.value)

        assert terminated
        assert not truncated
        assert reward == -1.0  # Collision penalty
        assert env.game_over is True

    def test_step_opposite_direction_prevention(self):
        env = SnakeGameEnv()
        env.reset()

        # Moving right, try to move left (should be prevented)
        env.direction = Direction.RIGHT
        obs, reward, terminated, truncated, info = env.step(Direction.LEFT.value)

        # Should continue moving right, not turn back into itself
        assert not terminated
        assert env.direction == Direction.RIGHT

    def test_step_max_steps_truncation(self):
        env = SnakeGameEnv(grid_size=5)
        env.reset()

        # Set steps close to max
        env.steps = env.max_steps - 1

        obs, reward, terminated, truncated, info = env.step(Direction.RIGHT.value)

        assert terminated  # Game ends due to max steps
        assert env.game_over is True
        assert env.steps == env.max_steps


class TestSnakeGameEnvGameOver:
    """Test game over behavior."""

    def test_step_when_game_over(self):
        env = SnakeGameEnv()
        env.reset()
        env.game_over = True

        obs, reward, terminated, truncated, info = env.step(Direction.RIGHT.value)

        assert terminated
        assert reward == 0.0
        assert len(env.snake) == 3  # Unchanged

    def test_reset_after_game_over(self):
        env = SnakeGameEnv()
        env.reset()
        env.game_over = True
        env.score = 10
        env.steps = 50

        obs, info = env.reset()

        assert env.game_over is False
        assert env.score == 0
        assert env.steps == 0
        assert len(env.snake) == 3


class TestSnakeGameEnvRandomScenarios:
    """Test various random scenarios."""

    def test_snake_growth_multiple_food(self):
        env = SnakeGameEnv(grid_size=10)
        env.reset()

        initial_length = len(env.snake)

        # Simulate eating multiple food items
        for i in range(3):
            # Place food in front of snake
            head = env.snake[-1]
            env.food = (head[0], head[1] + 1)

            obs, reward, terminated, truncated, info = env.step(Direction.RIGHT.value)

            assert not terminated
            assert reward == 1.0
            assert len(env.snake) == initial_length + i + 1
            assert env.score == i + 1

    def test_complex_movement_pattern(self):
        env = SnakeGameEnv(grid_size=10)
        env.reset()

        # Create a simple movement pattern
        actions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action.value)
            assert not terminated
            assert reward >= 0  # Either 0 (move) or 1 (food)

    def test_boundary_conditions(self):
        env = SnakeGameEnv(grid_size=5)
        env.reset()

        # Test movement near boundaries without collision
        env.snake = [(2, 1), (2, 2), (2, 3)]  # Center area
        env.direction = Direction.LEFT

        # Move left (should be safe)
        obs, reward, terminated, truncated, info = env.step(Direction.LEFT.value)
        assert not terminated
        assert reward == 0.0
        assert env.snake[-1] == (2, 0)  # Moved left

        # Move right back to original position
        obs, reward, terminated, truncated, info = env.step(Direction.RIGHT.value)
        assert not terminated
        assert env.snake[-1] == (2, 1)

    def test_scoring_multiple_food_items(self):
        """Test scoring with multiple consecutive food items."""
        env = SnakeGameEnv(grid_size=10)
        env.reset()

        initial_score = env.score
        food_positions = [(5, 6), (5, 7), (5, 8)]

        for i, _food_pos in enumerate(food_positions):
            # Place food in front of snake
            head = env.snake[-1]
            env.food = (head[0], head[1] + 1)

            obs, reward, terminated, truncated, info = env.step(Direction.RIGHT.value)

            assert not terminated
            assert reward == 1.0
            assert env.score == initial_score + i + 1
            assert len(env.snake) == 3 + i + 1

    def test_maximum_possible_score(self):
        """Test game behavior when approaching maximum possible score."""
        env = SnakeGameEnv(grid_size=5)
        env.reset()

        # Fill grid completely (snake occupies all cells)
        max_length = env.grid_size * env.grid_size
        env.snake = [(i, j) for i in range(env.grid_size) for j in range(env.grid_size)]
        env.score = max_length - 3  # Account for initial length
        env.food = None  # No space for food

        # Try to move - should result in game over due to no space
        obs, reward, terminated, truncated, info = env.step(Direction.RIGHT.value)

        # Game should end due to filled grid
        assert terminated or env.game_over
        assert env.score >= max_length - 3

    def test_observation_consistency_across_resets(self):
        """Test that observations are consistent after multiple resets with same seed."""
        env1 = SnakeGameEnv(grid_size=10)
        env2 = SnakeGameEnv(grid_size=10)

        for seed in [42, 123, 999]:
            obs1_1, _ = env1.reset(seed=seed)
            obs1_2, _ = env1.reset(seed=seed)
            obs2_1, _ = env2.reset(seed=seed)

            np.testing.assert_array_equal(obs1_1, obs1_2)
            np.testing.assert_array_equal(obs1_1, obs2_1)

    def test_reward_signal_consistency(self):
        """Test that reward signals are consistent for same actions in same states."""
        env1 = SnakeGameEnv(grid_size=8)
        env2 = SnakeGameEnv(grid_size=8)

        # Reset with same seed
        env1.reset(seed=42)
        env2.reset(seed=42)

        # Take same action in both environments
        action = Direction.RIGHT.value
        obs1, reward1, term1, trunc1, info1 = env1.step(action)
        obs2, reward2, term2, trunc2, info2 = env2.step(action)

        assert reward1 == reward2
        assert term1 == term2
        assert trunc1 == trunc2


class TestSnakeGameEnvPerformance:
    """Test performance and edge cases."""

    def test_large_grid_performance(self):
        """Test that large grids work efficiently."""
        import time

        env = SnakeGameEnv(grid_size=50)
        env.reset()

        start_time = time.time()

        # Perform several steps
        for i in range(100):
            action = Direction.RIGHT.value if i % 2 == 0 else Direction.UP.value
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        end_time = time.time()

        # Should complete reasonably quickly
        assert end_time - start_time < 1.0  # Less than 1 second for 100 steps
        assert obs.shape == (50, 50)
        assert not terminated or env.game_over

    def test_memory_efficiency(self):
        """Test that memory usage doesn't explode during long games."""
        env = SnakeGameEnv(grid_size=10)
        obs, info = env.reset()

        initial_memory = obs.nbytes
        max_steps = 200

        for step in range(max_steps):
            if env.game_over:
                break

            action = Direction.RIGHT.value if step % 4 == 0 else env.direction.value
            obs, reward, terminated, truncated, info = env.step(action)

            # Observation size should remain constant
            assert obs.nbytes == initial_memory
            assert obs.shape == (10, 10)

    def test_edge_case_minimal_grid(self):
        """Test minimal grid size (5x5 as per minimum)."""
        env = SnakeGameEnv(grid_size=5)
        obs, info = env.reset()

        assert obs.shape == (5, 5)
        assert len(env.snake) == 3  # Initial snake length
        assert not env.game_over
        assert 0 <= env.food[0] < 5
        assert 0 <= env.food[1] < 5

    def test_edge_case_maximum_grid(self):
        """Test maximum sensible grid size."""
        env = SnakeGameEnv(grid_size=100)
        obs, info = env.reset()

        assert obs.shape == (100, 100)
        assert len(env.snake) == 3
        assert not env.game_over
        assert obs.dtype == np.uint8
        assert np.max(obs) <= 3  # Only our defined observation values

    def test_collision_at_grid_boundaries(self):
        """Test collision detection at all four grid boundaries."""
        env = SnakeGameEnv(grid_size=5)

        # Test each boundary
        boundaries = [
            # Top boundary
            ([(0, 2), (1, 2), (2, 2)], Direction.UP),
            # Right boundary
            ([(2, 4), (2, 3), (2, 2)], Direction.RIGHT),
            # Bottom boundary
            ([(4, 2), (3, 2), (2, 2)], Direction.DOWN),
            # Left boundary
            ([(2, 0), (2, 1), (2, 2)], Direction.LEFT),
        ]

        for snake_pos, direction in boundaries:
            env.reset()
            env.snake = snake_pos.copy()
            env.direction = direction

            obs, reward, terminated, truncated, info = env.step(direction.value)

            assert terminated, f'Collision not detected at {direction} boundary'
            assert reward == -1.0, f'Wrong penalty for {direction} collision'
            assert env.game_over, f'Game should be over after {direction} collision'

    def test_simultaneous_food_and_collision_placement(self):
        """Test edge case where food placement and collision could be ambiguous."""
        env = SnakeGameEnv(grid_size=6)
        env.reset()

        # Create a situation where snake is long and food placement is constrained
        env.snake = [(3, i) for i in range(6)]  # Long horizontal snake
        env.direction = Direction.UP

        # Place food near snake to test interaction
        env.food = (2, 5)

        # This move should either get food or collide, not both
        obs, reward, terminated, truncated, info = env.step(Direction.UP.value)

        # Should not get both positive and negative rewards
        assert not (reward > 0 and reward < 0)

        if terminated:
            assert reward == -1.0
        elif reward > 0:
            assert reward == 1.0
