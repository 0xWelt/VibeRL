"""Tests for the unified Agent interface."""

import numpy as np
import pytest
import torch

from viberl.agents.base import Agent
from viberl.agents.dqn import DQNAgent
from viberl.agents.ppo import PPOAgent
from viberl.agents.reinforce import REINFORCEAgent
from viberl.typing import Action, Trajectory, Transition


class TestAgentInterface:
    """Test suite for Agent interface."""

    def test_agent_abstract_class(self):
        """Test that Agent is an abstract class."""
        with pytest.raises(TypeError):
            Agent(state_size=4, action_size=2)

    def test_reinforce_agent_interface(self):
        """Test REINFORCEAgent implements Agent interface."""
        agent = REINFORCEAgent(state_size=4, action_size=2)

        # Test act method
        state = np.array([1, 2, 3, 4])
        action_obj = agent.act(state)
        assert isinstance(action_obj, Action)
        assert isinstance(action_obj.action, int)
        assert 0 <= action_obj.action < 2

        # Test act with training parameter
        action_training = agent.act(state, training=False)
        assert isinstance(action_training, Action)

        # Test learn method with trajectory
        transitions = [
            Transition(
                state=np.array([1, 2, 3, 4]),
                action=Action(action=0),
                reward=1.0,
                next_state=np.array([2, 3, 4, 5]),
                done=False,
            ),
            Transition(
                state=np.array([2, 3, 4, 5]),
                action=Action(action=1),
                reward=2.0,
                next_state=np.array([3, 4, 5, 6]),
                done=True,
            ),
        ]
        trajectory = Trajectory.from_transitions(transitions)
        metrics = agent.learn(trajectory=trajectory)
        assert isinstance(metrics, dict)

    def test_dqn_agent_interface(self):
        """Test DQNAgent implements Agent interface."""
        agent = DQNAgent(state_size=4, action_size=2)

        # Test act method
        state = np.array([1, 2, 3, 4])
        action_obj = agent.act(state)
        assert isinstance(action_obj, Action)
        assert isinstance(action_obj.action, int)
        assert 0 <= action_obj.action < 2

        # Test act with training parameter
        action_training = agent.act(state, training=False)
        assert isinstance(action_training, Action)

        # Test learn method with trajectory
        transitions = [
            Transition(
                state=np.array([1, 2, 3, 4]),
                action=Action(action=0),
                reward=1.0,
                next_state=np.array([2, 3, 4, 5]),
                done=False,
            )
        ] * 100
        trajectory = Trajectory.from_transitions(transitions)
        metrics = agent.learn(trajectory=trajectory)
        assert isinstance(metrics, dict)

    def test_ppo_agent_interface(self):
        """Test PPOAgent implements Agent interface."""
        agent = PPOAgent(state_size=4, action_size=2)

        # Test act method
        state = np.array([1, 2, 3, 4])
        action_obj = agent.act(state)
        assert isinstance(action_obj, Action)
        assert isinstance(action_obj.action, int)
        assert 0 <= action_obj.action < 2

        # Test act with training parameter
        action_training = agent.act(state, training=False)
        assert isinstance(action_training, Action)

        # Test learn method with trajectory
        transitions = [
            Transition(
                state=np.array([1, 2, 3, 4]),
                action=Action(action=0, logprobs=torch.tensor(-0.1)),
                reward=1.0,
                next_state=np.array([2, 3, 4, 5]),
                done=False,
            ),
            Transition(
                state=np.array([2, 3, 4, 5]),
                action=Action(action=1, logprobs=torch.tensor(-0.2)),
                reward=2.0,
                next_state=np.array([3, 4, 5, 6]),
                done=True,
            ),
        ]
        trajectory = Trajectory.from_transitions(transitions)
        metrics = agent.learn(trajectory=trajectory)
        assert isinstance(metrics, dict)

    def test_agent_save_load(self):
        """Test save and load functionality."""
        import os
        import tempfile

        agent = DQNAgent(state_size=4, action_size=2)

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            # Test save
            agent.save(temp_path)
            assert os.path.exists(temp_path)

            # Test load
            agent.load(temp_path)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_agent_inheritance(self):
        """Test that all agents inherit from Agent base class."""
        agents = [
            REINFORCEAgent(state_size=4, action_size=2),
            DQNAgent(state_size=4, action_size=2),
            PPOAgent(state_size=4, action_size=2),
        ]

        for agent in agents:
            assert isinstance(agent, Agent)
            assert hasattr(agent, 'act')
            assert hasattr(agent, 'learn')
            assert hasattr(agent, 'save')
            assert hasattr(agent, 'load')
