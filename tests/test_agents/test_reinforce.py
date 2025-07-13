"""
Tests for REINFORCE algorithm-specific functionality.

Tests focus on REINFORCE-specific features without retesting interface compliance.
"""

import numpy as np
import pytest

from viberl.agents.reinforce import REINFORCEAgent
from viberl.typing import Trajectory, Transition
from viberl.utils.mock_env import MockEnv


class TestREINFORCESpecific:
    """Test REINFORCE-specific functionality."""

    @pytest.fixture
    def agent(self) -> REINFORCEAgent:
        """Create REINFORCE agent for testing."""
        return REINFORCEAgent(state_size=4, action_size=3, learning_rate=0.01)

    def test_policy_network_output_shape(self, agent: REINFORCEAgent) -> None:
        """Test policy network outputs correct action probabilities."""
        state = np.array([0.1, 0.2, 0.3, 0.4])

        # Test via act method
        action_obj = agent.act(state)
        assert isinstance(action_obj.action, int)
        assert 0 <= action_obj.action < 3

    def test_returns_calculation(self, agent: REINFORCEAgent) -> None:
        """Test returns calculation for REINFORCE."""
        rewards = [1.0, 2.0, 3.0, -1.0]

        # Test returns calculation via learning
        transitions = []
        for i, reward in enumerate(rewards):
            state = np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i])
            transitions.append(
                Transition(
                    state=state,
                    action=agent.act(state),
                    reward=reward,
                    next_state=np.array(
                        [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1), 0.4 * (i + 1)]
                    ),
                    done=(i == len(rewards) - 1),
                )
            )

        trajectory = Trajectory.from_transitions(transitions)
        metrics = agent.learn(trajectory)

        assert isinstance(metrics, dict)
        assert 'reinforce/policy_loss' in metrics

    def test_learning_with_single_episode(self, agent: REINFORCEAgent) -> None:
        """Test learning process with single episode data."""
        env = MockEnv(state_size=4, action_size=3, max_episode_steps=5)

        # Collect episode data
        state, _ = env.reset()
        transitions = []

        for _ in range(3):
            action_obj = agent.act(state)
            next_state, reward, done, _, _ = env.step(action_obj.action)

            transitions.append(
                Transition(
                    state=state,
                    action=action_obj,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
            )

            state = next_state
            if done:
                break

        trajectory = Trajectory.from_transitions(transitions)

        # Should be able to learn from trajectory
        metrics = agent.learn(trajectory)
        assert isinstance(metrics, dict)
        assert 'reinforce/policy_loss' in metrics
        assert isinstance(metrics['reinforce/policy_loss'], float)

    def test_learning_with_multiple_episodes(self, agent: REINFORCEAgent) -> None:
        """Test learning with data from multiple episodes."""
        # Create trajectory with multiple episodes
        transitions = []

        # Episode 1
        transitions.extend(
            [
                Transition(
                    state=np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]),
                    action=agent.act(np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i])),
                    reward=1.0,
                    next_state=np.array(
                        [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1), 0.4 * (i + 1)]
                    ),
                    done=(i == 2),
                )
                for i in range(3)
            ]
        )

        # Episode 2
        transitions.extend(
            [
                Transition(
                    state=np.array([0.5 * i, 0.6 * i, 0.7 * i, 0.8 * i]),
                    action=agent.act(np.array([0.5 * i, 0.6 * i, 0.7 * i, 0.8 * i])),
                    reward=-0.5,
                    next_state=np.array(
                        [0.5 * (i + 1), 0.6 * (i + 1), 0.7 * (i + 1), 0.8 * (i + 1)]
                    ),
                    done=(i == 1),
                )
                for i in range(2)
            ]
        )

        trajectory = Trajectory.from_transitions(transitions)
        metrics = agent.learn(trajectory)

        assert isinstance(metrics, dict)
        assert 'reinforce/policy_loss' in metrics

    def test_action_sampling_behavior(self, agent: REINFORCEAgent) -> None:
        """Test action sampling behavior in training vs evaluation."""
        state = np.array([0.1, 0.2, 0.3, 0.4])

        # Test that actions are valid
        action_train = agent.act(state, training=True)
        action_eval = agent.act(state, training=False)

        assert isinstance(action_train.action, int)
        assert isinstance(action_eval.action, int)
        assert 0 <= action_train.action < 3
        assert 0 <= action_eval.action < 3

    @pytest.mark.parametrize('learning_rate', [0.1, 0.01, 0.001])
    def test_different_learning_rates(self, learning_rate: float) -> None:
        """Test agent works with different learning rates."""
        agent = REINFORCEAgent(state_size=2, action_size=2, learning_rate=learning_rate)

        # Create simple trajectory
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
        assert 'reinforce/policy_loss' in metrics

    def test_empty_trajectory_handling(self, agent: REINFORCEAgent) -> None:
        """Test handling of empty trajectory."""
        trajectory = Trajectory.from_transitions([])

        # Should handle empty trajectory gracefully
        metrics = agent.learn(trajectory)
        assert isinstance(metrics, dict)

    def test_single_transition_trajectory(self, agent: REINFORCEAgent) -> None:
        """Test learning with single transition."""
        transition = Transition(
            state=np.array([0.1, 0.2, 0.3, 0.4]),
            action=agent.act(np.array([0.1, 0.2, 0.3, 0.4])),
            reward=1.0,
            next_state=np.array([0.2, 0.3, 0.4, 0.5]),
            done=True,
        )
        trajectory = Trajectory.from_transitions([transition])

        metrics = agent.learn(trajectory)
        assert isinstance(metrics, dict)
        assert 'reinforce/policy_loss' in metrics
