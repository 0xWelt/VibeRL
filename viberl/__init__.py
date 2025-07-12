"""
Tiny RL - A simple reinforcement learning framework for research and education.

This framework provides:
- Environments: Classic control and grid world environments
- Agents: Policy gradient and value-based RL algorithms
- Utilities: Common RL utilities and tools
- Examples: Ready-to-run examples and tutorials
"""

__version__ = '0.2.0'
__author__ = 'Tiny RL Team'

from .agents import DQNAgent, REINFORCEAgent
from .envs import Direction, SnakeGameEnv
from .utils import evaluate_agent, train_agent


__all__ = [
    'DQNAgent',
    'Direction',
    'REINFORCEAgent',
    'SnakeGameEnv',
    'evaluate_agent',
    'train_agent',
]
