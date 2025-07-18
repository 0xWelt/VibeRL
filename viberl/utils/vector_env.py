"""Vector environment utilities for parallel sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from loguru import logger

from viberl.typing import Action, Trajectory, Transition


if TYPE_CHECKING:
    from typing import Self

    from viberl.agents.base import Agent


class VectorEnvSampler:
    """Vector environment sampler for parallel trajectory collection."""

    def __init__(
        self,
        env_fn: callable,
        num_envs: int,
        agent: Agent,
        max_steps: int = 1000,
        device: str = 'cpu',
    ) -> None:
        """Initialize vector environment sampler.

        Args:
            env_fn: Function that creates a single environment
            num_envs: Number of parallel environments
            agent: RL agent to use for action selection
            max_steps: Maximum steps per episode
            device: Device for tensor operations
        """
        self.num_envs = num_envs
        self.agent = agent
        self.max_steps = max_steps
        self.device = torch.device(device)

        # Create vector environment
        try:
            from gymnasium.vector import AsyncVectorEnv

            self.env = AsyncVectorEnv([env_fn for _ in range(num_envs)])
            logger.info(f'Created AsyncVectorEnv with {num_envs} environments')
        except ImportError:
            logger.warning('AsyncVectorEnv not available, using SyncVectorEnv')
            from gymnasium.vector import SyncVectorEnv

            self.env = SyncVectorEnv([env_fn for _ in range(num_envs)])

        # Initialize tracking variables
        self.active_trajectories: list[list[Transition]] = [[] for _ in range(num_envs)]
        self.active_episode_rewards = np.zeros(num_envs)
        self.completed_trajectories: list[tuple[Trajectory, float]] = []

    def reset(self) -> np.ndarray:
        """Reset all environments.

        Returns:
            Initial observations from all environments
        """
        observations, _ = self.env.reset()
        return observations

    def collect_batch_trajectories(
        self, num_trajectories: int, render: bool = False
    ) -> list[tuple[Trajectory, float]]:
        """Collect a batch of trajectories using parallel environments.

        Args:
            num_trajectories: Number of trajectories to collect
            render: Whether to render the environments

        Returns:
            List of (trajectory, episode_reward) tuples
        """
        completed_trajectories = []
        collected_count = 0

        # Reset environments if needed
        observations, _ = self.env.reset()
        observations = self._preprocess_observations(observations)

        # Initialize tracking for active episodes
        active_trajectories = [[] for _ in range(self.num_envs)]
        active_rewards = np.zeros(self.num_envs)
        active_masks = np.ones(self.num_envs, dtype=bool)  # Track active environments

        while collected_count < num_trajectories:
            # Select actions for all active environments
            actions = []
            valid_observations = observations[active_masks]

            if len(valid_observations) > 0:
                # Get actions from agent for valid observations
                for obs in valid_observations:
                    action_obj = self.agent.act(obs)
                    actions.append(action_obj.action)

                # Step environments
                next_observations, rewards, terminations, truncations, infos = self.env.step(
                    np.array(actions)
                )
                next_observations = self._preprocess_observations(next_observations)

                # Process results for active environments
                action_idx = 0
                for env_idx in range(self.num_envs):
                    if not active_masks[env_idx]:
                        continue

                    # Create transition
                    transition = Transition(
                        state=observations[env_idx],
                        action=Action(
                            action=actions[action_idx],
                            log_prob=getattr(
                                self.agent.act(valid_observations[action_idx]),
                                'log_prob',
                                None,
                            ),
                        ),
                        reward=float(rewards[env_idx]),
                        next_state=next_observations[env_idx],
                        done=bool(terminations[env_idx] or truncations[env_idx]),
                        info=infos
                        if isinstance(infos, dict)
                        else (infos[env_idx] if isinstance(infos, list | tuple) else {}),
                    )

                    active_trajectories[env_idx].append(transition)
                    active_rewards[env_idx] += rewards[env_idx]
                    action_idx += 1

                    # Check if episode is done
                    if terminations[env_idx] or truncations[env_idx]:
                        # Complete trajectory
                        trajectory = Trajectory.from_transitions(active_trajectories[env_idx])
                        completed_trajectories.append((trajectory, active_rewards[env_idx]))

                        # Reset environment
                        active_trajectories[env_idx] = []
                        active_rewards[env_idx] = 0.0
                        collected_count += 1

                        # Reset this specific environment
                        obs, _ = self.env.reset()
                        next_observations[env_idx] = self._preprocess_observations(obs)[env_idx]

                # Update observations for next step
                observations = next_observations

                # Render if requested
                if render:
                    self.env.render()

        return completed_trajectories[:num_trajectories]

    def collect_trajectory_batch(
        self, batch_size: int, render: bool = False
    ) -> list[tuple[Trajectory, float]]:
        """Collect a batch of trajectories using parallel sampling.

        Args:
            batch_size: Number of trajectories to collect
            render: Whether to render environments

        Returns:
            List of (trajectory, episode_reward) tuples
        """
        return self.collect_batch_trajectories(batch_size, render=render)

    def _preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        """Preprocess observations for vectorized processing.

        Args:
            observations: Raw observations from vector environment

        Returns:
            Preprocessed observations
        """
        # Handle both single and multiple observations
        if len(observations.shape) > 2:
            # Grid-based observations (batch, height, width)
            return observations.reshape(observations.shape[0], -1)
        return observations

    def close(self) -> None:
        """Close the vector environment."""
        self.env.close()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


def create_vector_sampler(
    env_fn: callable,
    num_envs: int,
    agent: Agent,
    max_steps: int = 1000,
    device: str = 'cpu',
) -> Self:
    """Create a vector environment sampler.

    Args:
        env_fn: Function that creates a single environment
        num_envs: Number of parallel environments
        agent: RL agent to use for action selection
        max_steps: Maximum steps per episode
        device: Device for tensor operations

    Returns:
        VectorEnvSampler instance
    """
    return VectorEnvSampler(
        env_fn=env_fn,
        num_envs=num_envs,
        agent=agent,
        max_steps=max_steps,
        device=device,
    )
