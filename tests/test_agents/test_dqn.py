"""
Tests for DQN algorithm-specific functionality.

Tests focus on DQN-specific features without retesting interface compliance.
"""

import numpy as np
import pytest

from viberl.agents.dqn import DQNAgent
from viberl.typing import Trajectory, Transition


class TestDQNSpecific:
    """Test DQN-specific functionality."""

    @pytest.fixture
    def agent(self) -> DQNAgent:
        """Create DQN agent for testing."""
        return DQNAgent(
            state_size=4,
            action_size=3,
            learning_rate=0.01,
            epsilon_start=1.0,
            epsilon_end=0.01,
            memory_size=100,
            batch_size=4,
        )

    def test_q_network_output_shape(self, agent: DQNAgent) -> None:
        """Test Q network outputs correct Q-values."""
        state = np.array([0.1, 0.2, 0.3, 0.4])

        q_values = agent.q_network.get_q_values(state)
        assert q_values.shape == (1, 3)

    def test_epsilon_greedy_exploration(self, agent: DQNAgent) -> None:
        """Test epsilon-greedy exploration behavior."""
        state = np.array([0.1, 0.2, 0.3, 0.4])

        # With epsilon=1.0 (fully random)
        original_epsilon = agent.epsilon
        agent.epsilon = 1.0

        actions = [agent.act(state, training=True).action for _ in range(100)]
        unique_actions = set(actions)
        assert len(unique_actions) == 3  # Should explore all actions

        # Reset epsilon
        agent.epsilon = original_epsilon

    def test_epsilon_decay(self, agent: DQNAgent) -> None:
        """Test epsilon decay over learning steps."""
        initial_epsilon = agent.epsilon

        # Create trajectory with enough transitions for batch
        transitions = [
            Transition(
                state=np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]),
                action=agent.act(np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i])),
                reward=1.0,
                next_state=np.array([0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1), 0.4 * (i + 1)]),
                done=False,
            )
            for i in range(10)
        ]

        trajectory = Trajectory.from_transitions(transitions)

        # Learn and check epsilon decay
        agent.learn(trajectory)
        assert agent.epsilon <= initial_epsilon
        assert agent.epsilon >= agent.epsilon_end

    def test_experience_replay_memory(self, agent: DQNAgent) -> None:
        """Test experience replay memory functionality."""
        # Add transitions to memory
        for i in range(10):
            transition = Transition(
                state=np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]),
                action=agent.act(np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i])),
                reward=1.0,
                next_state=np.array([0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1), 0.4 * (i + 1)]),
                done=False,
            )
            trajectory = Trajectory.from_transitions([transition])
            agent.learn(trajectory)

        assert len(agent.memory) > 0
        assert len(agent.memory) <= agent.memory_size

    def test_batch_learning(self, agent: DQNAgent) -> None:
        """Test learning with batch sampling."""
        # Fill memory with enough transitions
        for i in range(agent.batch_size + 1):
            transition = Transition(
                state=np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]),
                action=agent.act(np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i])),
                reward=1.0,
                next_state=np.array([0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1), 0.4 * (i + 1)]),
                done=False,
            )
            trajectory = Trajectory.from_transitions([transition])
            agent.learn(trajectory)

        # Should have enough data for batch learning
        assert len(agent.memory) >= agent.batch_size

    def test_q_value_consistency(self, agent: DQNAgent) -> None:
        """Test Q-value consistency between evaluations."""
        state = np.array([0.1, 0.2, 0.3, 0.4])

        # Evaluate mode should be deterministic
        action1 = agent.act(state, training=False)
        action2 = agent.act(state, training=False)

        assert action1.action == action2.action

    @pytest.mark.parametrize(
        'epsilon_config',
        [
            {'epsilon_start': 1.0, 'epsilon_end': 0.1},
            {'epsilon_start': 0.5, 'epsilon_end': 0.01},
            {'epsilon_start': 0.1, 'epsilon_end': 0.001},
        ],
    )
    def test_different_epsilon_configs(self, epsilon_config: dict[str, float]) -> None:
        """Test different epsilon configurations."""
        agent = DQNAgent(state_size=2, action_size=2, **epsilon_config)

        assert agent.epsilon == epsilon_config['epsilon_start']
        assert agent.epsilon_end == epsilon_config['epsilon_end']

        # Test learning works
        transition = Transition(
            state=np.array([0.1, 0.2]),
            action=agent.act(np.array([0.1, 0.2])),
            reward=1.0,
            next_state=np.array([0.2, 0.3]),
            done=False,
        )
        trajectory = Trajectory.from_transitions([transition])
        metrics = agent.learn(trajectory)

        assert isinstance(metrics, dict)

    def test_memory_overflow_handling(self, agent: DQNAgent) -> None:
        """Test memory overflow handling."""
        agent.memory_size = 5  # Small memory size
        agent.memory = agent.memory.__class__(maxlen=agent.memory_size)

        # Add more transitions than memory size
        for i in range(10):
            transition = Transition(
                state=np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]),
                action=agent.act(np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i])),
                reward=1.0,
                next_state=np.array([0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1), 0.4 * (i + 1)]),
                done=False,
            )
            trajectory = Trajectory.from_transitions([transition])
            agent.learn(trajectory)

        # Memory should not exceed max size
        assert len(agent.memory) <= 5
