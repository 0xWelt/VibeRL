"""
Mock environment for testing RL algorithms.

Provides a gymnasium-compatible environment that returns random valid values
for all methods, useful for testing agents without complex environment setup.
"""

from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces


class MockEnv(Env):
    """
    A mock environment that returns random valid values for testing.

    This environment provides:
    - Random observations within observation space
    - Random rewards within reward range
    - Random terminal states
    - Random info dictionaries

    Args:
        state_size: Size of the observation space
        action_size: Number of discrete actions
        max_episode_steps: Maximum steps before truncation
    """

    def __init__(
        self,
        state_size: int = 4,
        action_size: int = 2,
        max_episode_steps: int = 100,
    ) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.max_episode_steps = max_episode_steps

        # Define spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(state_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(action_size)

        # Internal state
        self.current_step = 0
        self._np_random = np.random.RandomState()

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment with random initial state."""
        super().reset(seed=seed)

        if seed is not None:
            self._np_random = np.random.RandomState(seed)
            # Set numpy's global random state for gymnasium's sample() method
            np.random.seed(seed)

        self.current_step = 0

        # Generate random observation using our seeded random state
        obs = self._np_random.uniform(
            low=self.observation_space.low,
            high=self.observation_space.high,
            size=self.observation_space.shape,
        ).astype(np.float32)

        # Return with random info
        info = {'episode': 0, 'step': 0, 'random_metric': self._np_random.random()}

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take a step with random outcomes.

        Args:
            action: The action to take

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f'Invalid action: {action}'

        # Check if we've reached max steps (truncation happens after max_episode_steps steps)
        if self.current_step >= self.max_episode_steps:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = True
            truncated = True
            info = {'step': self.current_step, 'truncated': True}
            return obs, reward, terminated, truncated, info

        self.current_step += 1

        # Generate random observation using seeded random state
        obs = self._np_random.uniform(
            low=self.observation_space.low,
            high=self.observation_space.high,
            size=self.observation_space.shape,
        ).astype(np.float32)

        # Generate random reward (-1 to 1)
        reward = float(self._np_random.uniform(-1.0, 1.0))

        # Random termination (5% chance per step)
        terminated = bool(self._np_random.random() < 0.05)

        # Truncation happens when we reach max_episode_steps
        truncated = self.current_step >= self.max_episode_steps

        # Random info
        info = {
            'step': self.current_step,
            'action_taken': action,
            'random_info': self._np_random.random(),
            'episode_complete': terminated or truncated,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Mock render - does nothing."""

    def close(self) -> None:
        """Mock close - does nothing."""

    def seed(self, seed: int | None = None) -> None:
        """Set random seed for reproducibility."""
        self._np_random = np.random.RandomState(seed)
