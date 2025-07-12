"""Custom typing classes for reinforcement learning using Pydantic."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict


class Action(BaseModel):
    """An action taken by an agent, optionally with log probabilities."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: int
    logprobs: torch.Tensor | None = None


class Transition(BaseModel):
    """A single transition in an episode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: np.ndarray
    action: Action
    reward: float
    next_state: np.ndarray
    done: bool
    info: dict[str, Any] = {}


class Trajectory(BaseModel):
    """A complete trajectory (episode) consisting of multiple transitions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transitions: list[Transition]
    total_reward: float
    length: int

    @classmethod
    def from_transitions(cls, transitions: list[Transition]) -> Trajectory:
        """Create a trajectory from a list of transitions."""
        total_reward = sum(t.reward for t in transitions)
        return cls(transitions=transitions, total_reward=total_reward, length=len(transitions))

    def to_dict(self) -> dict:
        """Convert trajectory to dictionary format for agent learning."""
        return {
            'states': [t.state for t in self.transitions],
            'actions': [t.action.action for t in self.transitions],
            'rewards': [t.reward for t in self.transitions],
            'next_states': [t.next_state for t in self.transitions],
            'dones': [t.done for t in self.transitions],
            'logprobs': [
                t.action.logprobs for t in self.transitions if t.action.logprobs is not None
            ],
            'infos': [t.info for t in self.transitions],
        }
