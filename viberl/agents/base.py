"""Base Agent class for all RL agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import gymnasium as gym
    import numpy as np

    from viberl.typing import Action


class Agent(ABC):
    """Abstract base class for all RL agents."""

    def __init__(self, state_size: int, action_size: int, **kwargs):
        """Initialize the agent.

        Args:
            state_size: Size of the state space
            action_size: Size of the action space
            **kwargs: Additional configuration parameters
        """
        self.state_size = state_size
        self.action_size = action_size

    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> Action:
        """Select an action given the current state.

        Args:
            state: Current state observation
            training: Whether in training mode (affects exploration)

        Returns:
            Action object containing the selected action and optional metadata
        """

    @abstractmethod
    def learn(self, **kwargs) -> dict[str, float]:
        """Perform one learning step.

        Args:
            **kwargs: Learning-specific parameters

        Returns:
            Dictionary of step metrics (e.g., loss values)
        """

    def reset(self):
        """Reset agent state for new episode."""

    def get_metrics(self) -> dict[str, float]:
        """Get current training metrics.

        Returns:
            Dictionary of metrics
        """
        return {}

    def save(self, filepath: str):
        """Save agent state to file.

        Args:
            filepath: Path to save the model
        """

    def load(self, filepath: str):
        """Load agent state from file.

        Args:
            filepath: Path to load the model from
        """

    def setup_training(self, env: gym.Env):
        """Setup agent for training in given environment.

        Args:
            env: Training environment
        """

    def setup_evaluation(self):
        """Setup agent for evaluation mode."""
