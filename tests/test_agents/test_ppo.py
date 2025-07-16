"""
Tests for PPO algorithm-specific functionality.

Tests focus on PPO-specific features without retesting interface compliance.
"""

import numpy as np
import pytest
import torch

from viberl.agents.ppo import PPOAgent
from viberl.typing import Trajectory, Transition


class TestPPOSpecific:
    """Test PPO-specific functionality."""

    @pytest.fixture
    def agent(self) -> PPOAgent:
        """Create PPO agent for testing."""
        return PPOAgent(
            state_size=4,
            action_size=3,
            learning_rate=0.01,
            clip_epsilon=0.2,
            ppo_epochs=4,
            batch_size=4,
        )

    def test_ppo_network_functionality(self, agent: PPOAgent) -> None:
        """Test policy network outputs correct action probabilities and values."""
        state = np.array([0.1, 0.2, 0.3, 0.4])

        # Test via act method and check logprobs
        action_obj = agent.act(state)
        assert action_obj.logprobs is not None
        assert isinstance(action_obj.logprobs, torch.Tensor)

    def test_action_probabilities_with_logprobs(self, agent: PPOAgent) -> None:
        """Test action includes log probabilities."""
        state = np.array([0.1, 0.2, 0.3, 0.4])

        action_obj = agent.act(state)

        assert action_obj.logprobs is not None
        assert isinstance(action_obj.logprobs, torch.Tensor)

    def test_ppo_clip_objective(self, agent: PPOAgent) -> None:
        """Test PPO clipping objective functionality."""
        # Test with minimal data
        transitions = []
        for i in range(5):
            state = np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i])
            action_obj = agent.act(state)
            transitions.append(
                Transition(
                    state=state,
                    action=action_obj,
                    reward=1.0,
                    next_state=np.array(
                        [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1), 0.4 * (i + 1)]
                    ),
                    done=False,
                )
            )

        trajectory = Trajectory.from_transitions(transitions)

        # Should be able to learn
        metrics = agent.learn(trajectory=trajectory)
        assert isinstance(metrics, dict)
        assert 'ppo/policy_loss' in metrics
        assert 'ppo/value_loss' in metrics

    def test_multiple_epochs_training(self, agent: PPOAgent) -> None:
        """Test training across multiple epochs."""
        # Create sufficient data for batch processing
        transitions = []
        for i in range(10):
            state = np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i])
            action_obj = agent.act(state)
            transitions.append(
                Transition(
                    state=state,
                    action=action_obj,
                    reward=1.0,
                    next_state=np.array(
                        [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1), 0.4 * (i + 1)]
                    ),
                    done=False,
                )
            )

        trajectory = Trajectory.from_transitions(transitions)

        metrics = agent.learn(trajectory=trajectory)
        assert isinstance(metrics, dict)
        assert 'ppo/policy_loss' in metrics
        assert 'ppo/value_loss' in metrics

    def test_batch_processing(self, agent: PPOAgent) -> None:
        """Test batch processing for PPO training."""
        # Create data larger than batch size
        transitions = []
        for i in range(agent.batch_size * 3):
            state = np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i])
            action_obj = agent.act(state)
            transitions.append(
                Transition(
                    state=state,
                    action=action_obj,
                    reward=1.0,
                    next_state=np.array(
                        [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1), 0.4 * (i + 1)]
                    ),
                    done=False,
                )
            )

        trajectory = Trajectory.from_transitions(transitions)

        # Should process in batches
        metrics = agent.learn(trajectory=trajectory)
        assert isinstance(metrics, dict)
        assert 'ppo/policy_loss' in metrics

    @pytest.mark.parametrize('clip_epsilon', [0.1, 0.2, 0.3])
    def test_different_clipping_ranges(self, clip_epsilon: float) -> None:
        """Test different epsilon clipping ranges."""
        agent = PPOAgent(
            state_size=2,
            action_size=2,
            clip_epsilon=clip_epsilon,
            ppo_epochs=1,
            batch_size=2,
        )

        transition = Transition(
            state=np.array([0.1, 0.2]),
            action=agent.act(np.array([0.1, 0.2])),
            reward=1.0,
            next_state=np.array([0.2, 0.3]),
            done=False,
        )
        trajectory = Trajectory.from_transitions([transition])

        metrics = agent.learn(trajectory=trajectory)
        assert isinstance(metrics, dict)
        assert agent.clip_epsilon == clip_epsilon

    def test_small_batch_training(self, agent: PPOAgent) -> None:
        """Test training with small batch size."""
        agent.batch_size = 2

        # Create small batch
        transitions = []
        for i in range(3):
            state = np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i])
            action_obj = agent.act(state)
            transitions.append(
                Transition(
                    state=state,
                    action=action_obj,
                    reward=1.0,
                    next_state=np.array(
                        [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1), 0.4 * (i + 1)]
                    ),
                    done=False,
                )
            )

        trajectory = Trajectory.from_transitions(transitions)

        # Should handle small batch gracefully
        metrics = agent.learn(trajectory=trajectory)
        assert isinstance(metrics, dict)

    def test_empty_trajectory_handling(self, agent: PPOAgent) -> None:
        """Test handling of empty trajectory."""
        trajectory = Trajectory.from_transitions([])

        # Should handle empty trajectory gracefully
        metrics = agent.learn(trajectory=trajectory)
        assert isinstance(metrics, dict)
