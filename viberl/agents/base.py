"""Base Agent class for all RL agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import numpy as np

    from viberl.typing import Action, Trajectory


class Agent(ABC):
    """Abstract base class for all RL agents."""

    def __init__(self, state_size: int, action_size: int) -> None:
        """Initialize the agent.

        Args:
            state_size: Size of the state space
            action_size: Size of the action space
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
    def learn(self, trajectories: list[Trajectory]) -> dict[str, float]:
        """Perform one learning step.

        Args:
            trajectories: List of trajectories to learn from

        Returns:
            Dictionary of step metrics (e.g., loss values)
        """

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save agent state to file.

        Args:
            filepath: Path to save the model
        """

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load agent state from file.

        Args:
            filepath: Path to load the model from
        """
