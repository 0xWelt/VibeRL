"""Tests for vector environment utilities."""

from unittest.mock import Mock

import numpy as np
import pytest

from viberl.agents.base import Agent
from viberl.envs import SnakeGameEnv
from viberl.typing import Action, Transition
from viberl.utils.vector_env import VectorEnvSampler, create_vector_sampler


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock(spec=Agent)
    agent.act.return_value = Action(action=0, log_prob=0.0)
    return agent


@pytest.fixture
def snake_env_fn():
    """Create snake environment factory."""

    def _create_env():
        return SnakeGameEnv(grid_size=5)

    return _create_env


class TestVectorEnvSampler:
    """Test VectorEnvSampler class."""

    def test_initialization(self, snake_env_fn, mock_agent):
        """Test VectorEnvSampler initialization."""
        sampler = VectorEnvSampler(
            env_fn=snake_env_fn,
            num_envs=2,
            agent=mock_agent,
            max_steps=100,
        )

        assert sampler.num_envs == 2
        assert sampler.max_steps == 100
        assert sampler.agent == mock_agent

        sampler.close()

    def test_single_env_sampling(self, snake_env_fn, mock_agent):
        """Test sampling with single environment."""
        sampler = VectorEnvSampler(
            env_fn=snake_env_fn,
            num_envs=1,
            agent=mock_agent,
            max_steps=50,
        )

        results = sampler.collect_trajectory_batch(1)

        assert len(results) == 1
        trajectory, reward = results[0]
        assert len(trajectory.transitions) > 0
        assert isinstance(reward, int | float)

        sampler.close()

    def test_multi_env_sampling(self, snake_env_fn, mock_agent):
        """Test sampling with multiple environments."""
        sampler = VectorEnvSampler(
            env_fn=snake_env_fn,
            num_envs=2,
            agent=mock_agent,
            max_steps=50,
        )

        results = sampler.collect_trajectory_batch(2)

        assert len(results) == 2
        for trajectory, reward in results:
            assert len(trajectory.transitions) > 0
            assert isinstance(reward, int | float)

        sampler.close()

    def test_batch_size_larger_than_envs(self, snake_env_fn, mock_agent):
        """Test when batch size is larger than number of environments."""
        sampler = VectorEnvSampler(
            env_fn=snake_env_fn,
            num_envs=2,
            agent=mock_agent,
            max_steps=50,
        )

        results = sampler.collect_trajectory_batch(4)

        assert len(results) == 4
        for trajectory, reward in results:
            assert len(trajectory.transitions) > 0
            assert isinstance(reward, int | float)

        sampler.close()

    def test_observation_preprocessing(self):
        """Test observation preprocessing."""
        agent = Mock(spec=Agent)
        agent.act.return_value = Action(action=0, log_prob=0.0)
        sampler = VectorEnvSampler(
            env_fn=lambda: SnakeGameEnv(grid_size=3),
            num_envs=1,
            agent=agent,
            max_steps=10,
        )

        # Test 2D observation preprocessing
        obs_2d = np.random.rand(1, 3, 3)
        processed = sampler._preprocess_observations(obs_2d)
        assert processed.shape == (1, 9)

        # Test 1D observation (no preprocessing needed)
        obs_1d = np.random.rand(1, 5)
        processed = sampler._preprocess_observations(obs_1d)
        assert processed.shape == (1, 5)

        sampler.close()

    def test_context_manager(self, snake_env_fn, mock_agent):
        """Test VectorEnvSampler as context manager."""
        with VectorEnvSampler(
            env_fn=snake_env_fn,
            num_envs=1,
            agent=mock_agent,
            max_steps=10,
        ) as sampler:
            results = sampler.collect_trajectory_batch(1)
            assert len(results) == 1

    def test_create_vector_sampler(self, snake_env_fn, mock_agent):
        """Test create_vector_sampler factory function."""
        sampler = create_vector_sampler(
            env_fn=snake_env_fn,
            num_envs=3,
            agent=mock_agent,
            max_steps=100,
            device='cpu',
        )

        assert sampler.num_envs == 3
        assert sampler.max_steps == 100

        sampler.close()


@pytest.mark.integration
class TestVectorEnvIntegration:
    """Integration tests for vector environment sampling."""

    def test_real_agent_with_vector_env(self):
        """Test with real agent implementation."""
        from viberl.agents import REINFORCEAgent

        state_size = 25  # 5x5 grid
        action_size = 4

        agent = REINFORCEAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_size=32,
            num_hidden_layers=1,
        )

        def env_fn():
            return SnakeGameEnv(grid_size=5)

        sampler = VectorEnvSampler(
            env_fn=env_fn,
            num_envs=2,
            agent=agent,
            max_steps=20,
        )

        results = sampler.collect_trajectory_batch(2)

        assert len(results) == 2
        for trajectory, reward in results:
            assert len(trajectory.transitions) > 0
            assert isinstance(reward, int | float)
            # Check that all transitions have the expected structure
            for transition in trajectory.transitions:
                assert isinstance(transition, Transition)
                assert len(transition.state) == state_size
                assert transition.action.action in range(action_size)

        sampler.close()

    def test_performance_comparison(self):
        """Test performance comparison between sequential and parallel sampling."""
        import time

        def env_fn():
            return SnakeGameEnv(grid_size=4)

        agent = Mock(spec=Agent)
        agent.act.return_value = Action(action=0, log_prob=0.0)

        # Test sequential sampling (1 env)
        start_time = time.time()
        sampler_seq = VectorEnvSampler(env_fn, 1, agent, max_steps=50)
        results_seq = sampler_seq.collect_trajectory_batch(4)
        seq_time = time.time() - start_time
        sampler_seq.close()

        # Test parallel sampling (2 envs)
        start_time = time.time()
        sampler_par = VectorEnvSampler(env_fn, 2, agent, max_steps=50)
        results_par = sampler_par.collect_trajectory_batch(4)
        par_time = time.time() - start_time
        sampler_par.close()

        assert len(results_seq) == 4
        assert len(results_par) == 4

        # Both should collect the same number of trajectories
        assert len(results_seq) == len(results_par)

        # Performance should be better with parallel (though this is environment-dependent)
        # We don't assert this as it depends on the system, but we log it
        print(f'Sequential time: {seq_time:.3f}s')
        print(f'Parallel time: {par_time:.3f}s')
        if par_time < seq_time:
            speedup = seq_time / par_time
            print(f'Speedup: {speedup:.2f}x')
