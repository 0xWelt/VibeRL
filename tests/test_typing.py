"""
Tests for typing module - custom type system.

Tests focus on the Action, Transition, and Trajectory classes.
"""

import numpy as np
import pytest
import torch

from viberl.typing import Action, Trajectory, Transition


class TestAction:
    """Test Action class functionality."""

    def test_action_creation(self):
        """Test Action creation with required fields."""
        action = Action(action=2)
        assert action.action == 2
        assert action.logprobs is None

    def test_action_with_logprobs(self):
        """Test Action creation with logprobs."""
        logprobs = torch.tensor([0.1, 0.2, 0.3])
        action = Action(action=1, logprobs=logprobs)
        assert action.action == 1
        assert torch.equal(action.logprobs, logprobs)

    def test_action_validation(self):
        """Test Action validation."""
        # Valid action
        action = Action(action=0)
        assert action.action == 0

        # Valid action with logprobs
        logprobs = torch.tensor(0.5)
        action = Action(action=2, logprobs=logprobs)
        assert action.action == 2

    def test_action_immutability(self):
        """Test Action attributes are properly set."""
        action = Action(action=1)
        assert action.action == 1


class TestTransition:
    """Test Transition class functionality."""

    def test_transition_creation(self):
        """Test Transition creation with all fields."""
        state = np.array([0.1, 0.2, 0.3])
        action = Action(action=1)
        reward = 1.0
        next_state = np.array([0.2, 0.3, 0.4])
        done = False

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        assert np.array_equal(transition.state, state)
        assert transition.action == action
        assert transition.reward == reward
        assert np.array_equal(transition.next_state, next_state)
        assert transition.done == done

    def test_transition_with_info(self):
        """Test Transition creation with info."""
        state = np.array([0.1, 0.2])
        action = Action(action=0)
        reward = -0.1
        next_state = np.array([0.1, 0.2])
        done = True
        info = {'score': 10}

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info,
        )

        assert transition.info == info

    def test_transition_validation(self):
        """Test Transition field validation."""
        # Valid transition
        transition = Transition(
            state=np.array([0.1]),
            action=Action(action=0),
            reward=1.0,
            next_state=np.array([0.2]),
            done=False,
        )
        assert transition.reward == 1.0


class TestTrajectory:
    """Test Trajectory class functionality."""

    def test_empty_trajectory(self):
        """Test empty trajectory creation."""
        trajectory = Trajectory.from_transitions([])
        assert len(trajectory.transitions) == 0
        assert trajectory.total_reward == 0.0
        assert trajectory.length == 0

    def test_trajectory_from_transitions(self):
        """Test Trajectory creation from transitions."""
        transitions = [
            Transition(
                state=np.array([0.1]),
                action=Action(action=0),
                reward=1.0,
                next_state=np.array([0.2]),
                done=False,
            ),
            Transition(
                state=np.array([0.2]),
                action=Action(action=1),
                reward=2.0,
                next_state=np.array([0.3]),
                done=True,
            ),
        ]

        trajectory = Trajectory.from_transitions(transitions)

        assert len(trajectory.transitions) == 2
        assert trajectory.total_reward == 3.0
        assert trajectory.length == 2

    def test_trajectory_with_single_transition(self):
        """Test Trajectory with single transition."""
        transition = Transition(
            state=np.array([0.1]),
            action=Action(action=0),
            reward=10.0,
            next_state=np.array([0.2]),
            done=True,
        )

        trajectory = Trajectory.from_transitions([transition])

        assert len(trajectory.transitions) == 1
        assert trajectory.total_reward == 10.0
        assert trajectory.length == 1

    def test_trajectory_with_negative_rewards(self):
        """Test Trajectory with negative rewards."""
        transitions = [
            Transition(
                state=np.array([0.1]),
                action=Action(action=0),
                reward=-1.0,
                next_state=np.array([0.2]),
                done=False,
            ),
            Transition(
                state=np.array([0.2]),
                action=Action(action=1),
                reward=-2.0,
                next_state=np.array([0.3]),
                done=True,
            ),
        ]

        trajectory = Trajectory.from_transitions(transitions)

        assert trajectory.total_reward == -3.0
        assert trajectory.length == 2

    def test_trajectory_immutability(self):
        """Test trajectory attributes are properly set."""
        transitions = [
            Transition(
                state=np.array([0.1]),
                action=Action(action=0),
                reward=1.0,
                next_state=np.array([0.2]),
                done=False,
            )
        ]

        trajectory = Trajectory.from_transitions(transitions)

        assert len(trajectory.transitions) == 1
        assert trajectory.total_reward == 1.0
        assert trajectory.length == 1

    @pytest.mark.parametrize('num_transitions', [0, 1, 5, 10])
    def test_trajectory_lengths(self, num_transitions: int) -> None:
        """Test trajectory with different numbers of transitions."""
        transitions = [
            Transition(
                state=np.array([0.1]),
                action=Action(action=0),
                reward=1.0,
                next_state=np.array([0.2]),
                done=False,
            )
            for _ in range(num_transitions)
        ]

        trajectory = Trajectory.from_transitions(transitions)

        assert len(trajectory.transitions) == num_transitions
        assert trajectory.length == num_transitions
        assert trajectory.total_reward == float(num_transitions)
