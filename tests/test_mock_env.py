"""
Tests for MockEnv - the mock environment for testing RL algorithms.
"""

import numpy as np
import pytest
from gymnasium import spaces

from viberl.utils.mock_env import MockEnv


class TestMockEnvInitialization:
    """Test MockEnv initialization and basic properties."""

    def test_default_initialization(self):
        """Test default MockEnv initialization."""
        env = MockEnv()

        assert env.state_size == 4
        assert env.action_size == 2
        assert env.max_episode_steps == 100
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.observation_space.shape == (4,)
        assert env.action_space.n == 2

    def test_custom_initialization(self):
        """Test MockEnv with custom parameters."""
        env = MockEnv(state_size=8, action_size=4, max_episode_steps=50)

        assert env.state_size == 8
        assert env.action_size == 4
        assert env.max_episode_steps == 50
        assert env.observation_space.shape == (8,)
        assert env.action_space.n == 4

    def test_observation_space_bounds(self):
        """Test observation space bounds are correct."""
        env = MockEnv()
        obs_space = env.observation_space

        assert obs_space.low.tolist() == [-1.0] * 4
        assert obs_space.high.tolist() == [1.0] * 4
        assert obs_space.dtype == np.float32


class TestMockEnvReset:
    """Test MockEnv reset functionality."""

    def test_reset_returns_correct_format(self):
        """Test reset returns correct format (obs, info)."""
        env = MockEnv()
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        assert 'step' in info
        assert 'episode' in info
        assert 'random_metric' in info

    def test_reset_with_seed(self):
        """Test reset with seed produces reproducible results."""
        env1 = MockEnv()
        env2 = MockEnv()

        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)
        assert info1['episode'] == info2['episode']
        assert info1['step'] == info2['step']

    def test_reset_resets_step_counter(self):
        """Test reset resets the step counter."""
        env = MockEnv()
        env.reset()

        # Take some steps
        for _ in range(5):
            env.step(0)

        assert env.current_step == 5

        # Reset should reset step counter
        env.reset()
        assert env.current_step == 0


class TestMockEnvStep:
    """Test MockEnv step functionality."""

    def test_step_returns_correct_format(self):
        """Test step returns correct format (obs, reward, terminated, truncated, info)."""
        env = MockEnv()
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_increments_step_counter(self):
        """Test step increments the step counter."""
        env = MockEnv()
        env.reset()

        assert env.current_step == 0
        env.step(0)
        assert env.current_step == 1
        env.step(1)
        assert env.current_step == 2

    def test_step_with_invalid_action_raises(self):
        """Test step with invalid action raises assertion."""
        env = MockEnv(action_size=2)
        env.reset()

        # Valid actions
        env.step(0)
        env.step(1)

        # Invalid action should raise assertion
        with pytest.raises(AssertionError):
            env.step(2)

    def test_step_truncation_after_max_steps(self):
        """Test truncation occurs after max episode steps."""
        env = MockEnv(max_episode_steps=5)
        env.reset()

        # Take 4 steps (should not be truncated yet)
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step(0)
            assert not truncated
            assert not terminated

        # 5th step should be truncated (step 5 >= max_episode_steps=5)
        _, _, terminated, truncated, _ = env.step(0)
        assert truncated  # Should be truncated due to max steps


class TestMockEnvRandomness:
    """Test MockEnv randomness properties."""

    def test_observations_within_bounds(self):
        """Test observations are always within bounds."""
        env = MockEnv()
        env.reset()

        for _ in range(100):
            obs, _, _, _, _ = env.step(0)
            assert env.observation_space.contains(obs)

    def test_rewards_within_bounds(self):
        """Test rewards are within expected bounds."""
        env = MockEnv()
        env.reset()

        for _ in range(100):
            _, reward, _, _, _ = env.step(0)
            assert -1.0 <= reward <= 1.0

    def test_randomness_across_episodes(self):
        """Test that different episodes produce different observations."""
        env = MockEnv()
        observations = []

        for _ in range(10):
            obs, _ = env.reset()
            observations.append(obs.copy())

        # Check that we have some variety in observations
        # (not all observations are identical)
        unique_obs = [tuple(obs) for obs in observations]
        assert len(set(unique_obs)) > 1


class TestMockEnvCompatibility:
    """Test MockEnv compatibility with gymnasium interface."""

    def test_gymnasium_compatibility(self):
        """Test MockEnv follows gymnasium.Env interface."""
        env = MockEnv()

        # Test basic interface
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'close')
        assert hasattr(env, 'render')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')

    def test_full_episode_cycle(self):
        """Test a complete episode cycle."""
        env = MockEnv(max_episode_steps=10)

        # Reset
        obs, info = env.reset()
        assert env.observation_space.contains(obs)

        episode_length = 0
        terminated = truncated = False

        # Run episode until termination
        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

            episode_length += 1
            assert episode_length <= 11  # Allow up to max_steps + 1 due to truncation

        # Reset should work after episode ends
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert env.current_step == 0


class TestMockEnvWithAgents:
    """Test MockEnv can be used with agent training."""

    def test_mock_env_with_training_loop(self):
        """Test MockEnv can be used in a simple training loop."""
        from viberl.agents.dqn import DQNAgent

        env = MockEnv(state_size=4, action_size=2, max_episode_steps=10)
        agent = DQNAgent(state_size=4, action_size=2)

        # Run a few episodes
        for _episode in range(3):
            state, info = env.reset()
            total_reward = 0

            terminated = truncated = False
            while not terminated and not truncated:
                action_obj = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action_obj.action)

                total_reward += reward
                state = next_state

            assert total_reward != 0  # Should accumulate some reward
            assert env.current_step <= 11  # Allow up to max_steps + 1 due to truncation

    def test_mock_env_consistency(self):
        """Test MockEnv produces consistent results with same seed."""
        env1 = MockEnv(state_size=4, action_size=2, max_episode_steps=10)
        env2 = MockEnv(state_size=4, action_size=2, max_episode_steps=10)

        # Reset with same seed
        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

        # Take same actions
        actions = [0, 1, 0, 1]
        for action in actions:
            obs1, reward1, term1, trunc1, info1 = env1.step(action)
            obs2, reward2, term2, trunc2, info2 = env2.step(action)

            np.testing.assert_array_equal(obs1, obs2)
            assert reward1 == reward2
            assert term1 == term2
            assert trunc1 == trunc2
