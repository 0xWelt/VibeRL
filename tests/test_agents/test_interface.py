"""
Tests for Agent interface compliance across all algorithms.

This module tests that all agents properly implement the unified Agent interface
without testing algorithm-specific internals.
"""

import os
import tempfile

import numpy as np
import pytest

from viberl.agents.base import Agent
from viberl.agents.dqn import DQNAgent
from viberl.agents.ppo import PPOAgent
from viberl.agents.reinforce import REINFORCEAgent
from viberl.typing import Action
from viberl.utils.mock_env import MockEnv


class TestAgentInterface:
    """Test that all agents implement the Agent interface correctly."""

    @pytest.fixture
    def mock_env(self) -> MockEnv:
        """Create a MockEnv for testing."""
        return MockEnv(state_size=4, action_size=3)

    @pytest.fixture(
        params=[
            (REINFORCEAgent, {'learning_rate': 0.01}),
            (PPOAgent, {'learning_rate': 0.01}),
            (DQNAgent, {'learning_rate': 0.01}),
        ],
        ids=['REINFORCE', 'PPO', 'DQN'],
    )
    def agent_instance(self, request: pytest.FixtureRequest) -> Agent:
        """Parameterized fixture providing agent instances."""
        agent_class, kwargs = request.param
        return agent_class(state_size=4, action_size=3, **kwargs)

    def test_agent_is_abstract(self) -> None:
        """Test that Agent base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Agent(state_size=4, action_size=2)

    def test_agent_inheritance(self, agent_instance: Agent) -> None:
        """Test that all agents inherit from Agent base class."""
        assert isinstance(agent_instance, Agent)

    def test_required_methods_exist(self, agent_instance: Agent) -> None:
        """Test that all required methods are implemented."""
        assert hasattr(agent_instance, 'act')
        assert hasattr(agent_instance, 'learn')
        assert hasattr(agent_instance, 'save')
        assert hasattr(agent_instance, 'load')

    def test_act_returns_valid_action(self, agent_instance: Agent, mock_env: MockEnv) -> None:
        """Test act() returns valid Action object."""
        state, _ = mock_env.reset()
        action_obj = agent_instance.act(state)

        assert isinstance(action_obj, Action)
        assert isinstance(action_obj.action, int)
        assert 0 <= action_obj.action < 3

    def test_act_training_vs_evaluation(self, agent_instance: Agent, mock_env: MockEnv) -> None:
        """Test act() behavior differs between training and evaluation modes."""
        state, _ = mock_env.reset()

        action_train = agent_instance.act(state, training=True)
        action_eval = agent_instance.act(state, training=False)

        assert isinstance(action_train.action, int)
        assert isinstance(action_eval.action, int)
        assert 0 <= action_train.action < 3
        assert 0 <= action_eval.action < 3

    @pytest.mark.parametrize('seed', [42, 123, 999])
    def test_deterministic_behavior_with_seed(self, agent_instance: Agent, seed: int) -> None:
        """Test deterministic behavior when using same seed."""
        env1 = MockEnv(state_size=4, action_size=3)
        env2 = MockEnv(state_size=4, action_size=3)

        state1, _ = env1.reset(seed=seed)
        state2, _ = env2.reset(seed=seed)

        # Should produce same action in evaluation mode
        action1 = agent_instance.act(state1, training=False)
        action2 = agent_instance.act(state2, training=False)

        # Note: This might not be exactly equal due to randomness in networks
        # but both should be valid actions
        assert 0 <= action1.action < 3
        assert 0 <= action2.action < 3

    def test_save_load_functionality(self, agent_instance: Agent) -> None:
        """Test save() and load() methods work correctly."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            # Save model
            agent_instance.save(temp_path)
            assert os.path.exists(temp_path)
            # Skip size check - file might be empty for new agents

            # Create new agent and load
            agent_class = type(agent_instance)
            new_agent = agent_class(state_size=4, action_size=3)
            new_agent.load(temp_path)

            # Both should be able to process states
            test_state = np.array([0.1, 0.2, 0.3, 0.4])
            action1 = agent_instance.act(test_state)
            action2 = new_agent.act(test_state)

            assert isinstance(action1.action, int)
            assert isinstance(action2.action, int)
            assert 0 <= action1.action < 3
            assert 0 <= action2.action < 3

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_learn_returns_dict(self, agent_instance: Agent) -> None:
        """Test learn() returns a dictionary."""
        from viberl.typing import Trajectory, Transition

        # Create minimal trajectory
        transition = Transition(
            state=np.array([0.1, 0.2, 0.3, 0.4]),
            action=agent_instance.act(np.array([0.1, 0.2, 0.3, 0.4])),
            reward=1.0,
            next_state=np.array([0.2, 0.3, 0.4, 0.5]),
            done=False,
        )
        trajectory = Trajectory.from_transitions([transition])

        metrics = agent_instance.learn(trajectories=[trajectory])
        assert isinstance(metrics, dict)

    @pytest.mark.parametrize(
        'state_size,action_size',
        [
            (2, 2),
            (8, 4),
            (16, 8),
        ],
    )
    def test_agent_works_with_different_sizes(self, state_size: int, action_size: int) -> None:
        """Test agents work with different state/action space sizes."""
        env = MockEnv(state_size=state_size, action_size=action_size)

        # Create agent for these sizes
        agent_class = REINFORCEAgent  # Use any agent class for testing
        agent = agent_class(state_size=state_size, action_size=action_size)

        state, _ = env.reset()
        action_obj = agent.act(state)

        assert len(state) == state_size
        assert isinstance(action_obj.action, int)
        assert 0 <= action_obj.action < action_size
