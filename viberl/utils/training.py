"""Training utilities for VibeRL framework.

This module provides backward compatibility for the old training interface
while internally using the new Trainer class.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from viberl.trainer import Trainer


if TYPE_CHECKING:
    import gymnasium as gym

    from viberl.agents.base import Agent


def train_agent(
    env: gym.Env,
    agent: Agent,
    num_episodes: int = 1000,
    max_steps: int = 1000,
    render_interval: int | None = None,
    save_interval: int | None = None,
    save_path: str | None = None,
    verbose: bool = True,
    log_dir: str | None = None,
    eval_interval: int = 100,
    eval_episodes: int = 10,
    log_interval: int = 1000,
) -> list[float]:
    """
    Generic training function for RL agents with periodic evaluation.

    .. deprecated:: 1.0
        Use :class:`viberl.trainer.Trainer` instead.

    Args:
        env: Gymnasium environment
        agent: RL agent with select_action, store_transition, and update_policy methods
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        render_interval: Render every N episodes
        save_interval: Save model every N episodes
        save_path: Path to save models
        verbose: Print training progress
        log_dir: Directory for TensorBoard logs
        eval_interval: Evaluate every N episodes
        eval_episodes: Number of evaluation episodes

    Returns:
        List of episode rewards
    """
    warnings.warn(
        'train_agent is deprecated and will be removed in a future version. '
        'Use viberl.trainer.Trainer instead.',
        DeprecationWarning,
        stacklevel=2,
    )

    # Create trainer using new interface
    trainer = Trainer(
        env=env,
        agent=agent,
        max_steps=max_steps,
        log_dir=log_dir,
    )

    # Train using new trainer
    return trainer.train(
        num_episodes=num_episodes,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        save_interval=save_interval,
        save_path=save_path,
        render_interval=render_interval,
        log_interval=log_interval,
        verbose=verbose,
    )


def evaluate_agent(
    env: gym.Env,
    agent: Agent,
    num_episodes: int = 10,
    render: bool = False,
    max_steps: int = 1000,
) -> tuple[list[float], list[int]]:
    """
    Generic evaluation function for RL agents.

    Args:
        env: Gymnasium environment
        agent: RL agent with select_action method
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        max_steps: Maximum steps per episode

    Returns:
        Tuple of (episode_rewards, episode_lengths)
    """
    trainer = Trainer(env=env, agent=agent, max_steps=max_steps)
    return trainer.evaluate(num_episodes=num_episodes, render=render)
