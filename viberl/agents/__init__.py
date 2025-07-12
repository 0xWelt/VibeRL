"""
Agents module - Collection of reinforcement learning algorithms.
"""

from .dqn import DQNAgent
from .policy_gradient.reinforce import REINFORCEAgent


__all__ = ['DQNAgent', 'REINFORCEAgent']
