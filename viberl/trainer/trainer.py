"""Trainer class for unified RL agent training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

from viberl.typing import Trajectory, Transition
from viberl.utils.writer import UnifiedWriter


if TYPE_CHECKING:
    import gymnasium as gym

    from viberl.agents.base import Agent


class Trainer:
    """Unified trainer for RL agents with comprehensive training and evaluation capabilities.

    This class encapsulates the training logic that was previously in training.py,
    providing a clean object-oriented interface for training any RL agent.
    """

    def __init__(
        self,
        env: gym.Env,
        agent: Agent,
        max_steps: int = 1000,
        log_dir: str | None = None,
        device: str = 'auto',
        eval_env: gym.Env | None = None,
        enable_tensorboard: bool = True,
        enable_wandb: bool = False,
        wandb_config: dict | None = None,
        run_name: str | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            env: The training environment
            agent: The RL agent to train
            max_steps: Maximum steps per episode
            log_dir: Directory for logs
            device: Device to use for training ("auto", "cpu", "cuda")
            eval_env: Optional evaluation environment. If None, will create a deep copy of the training environment.
            enable_tensorboard: Whether to enable TensorBoard logging
            enable_wandb: Whether to enable Weights & Biases logging
            wandb_config: Configuration dict for wandb
        """
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.log_dir = log_dir
        self.eval_env = eval_env

        # Device configuration
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize tracking variables
        self.episode_rewards: list[float] = []
        self.eval_rewards: list[float] = []
        self.best_eval_score = float('-inf')
        self.writer: UnifiedWriter | None = None

        # Initialize unified writer
        if log_dir is not None:
            self.writer = UnifiedWriter(
                log_dir=log_dir,
                enable_tensorboard=enable_tensorboard,
                enable_wandb=enable_wandb,
                wandb_config=wandb_config,
                run_name=run_name,
            )

    def train(
        self,
        num_episodes: int = 1000,
        eval_interval: int = 100,
        eval_episodes: int = 10,
        save_interval: int | None = None,
        save_path: str | None = None,
        render_interval: int | None = None,
        log_interval: int = 1000,
        verbose: bool = True,
    ) -> list[float]:
        """Train the agent for specified number of episodes.

        Args:
            num_episodes: Number of training episodes
            eval_interval: Evaluate every N episodes
            eval_episodes: Number of evaluation episodes
            save_interval: Save model every N episodes
            save_path: Path to save models
            render_interval: Render environment every N episodes
            log_interval: Log progress every N episodes
            verbose: Whether to print progress

        Returns:
            List of episode rewards during training
        """
        # Validate interval alignment
        if save_interval is not None:
            assert save_interval % eval_interval == 0, (
                f'save_interval ({save_interval}) must be a multiple of eval_interval ({eval_interval})'
            )

        assert log_interval % eval_interval == 0, (
            f'log_interval ({log_interval}) must be a multiple of eval_interval ({eval_interval})'
        )

        # Create evaluation environment
        eval_env = self._create_eval_env()

        try:
            for episode in range(num_episodes):
                # Run training episode
                episode_reward = self._train_episode(episode, render_interval)
                self.episode_rewards.append(episode_reward)

                # Evaluation
                eval_mean = 0.0
                eval_rewards = []
                eval_lengths = []

                if episode % eval_interval == 0:
                    eval_rewards, eval_lengths = self.evaluate(
                        eval_env, num_episodes=eval_episodes, render=False
                    )
                    eval_mean = np.mean(eval_rewards)
                    self.eval_rewards.extend(eval_rewards)

                # Save checkpoint
                if save_interval is not None and episode % save_interval == 0:
                    self._save_checkpoint(episode, eval_rewards, save_path)

                # Logging
                if episode % log_interval == 0:
                    self._log_progress(episode, num_episodes, eval_rewards, eval_mean, verbose)

        finally:
            eval_env.close()
            if self.writer is not None:
                self.writer.close()

        return self.episode_rewards

    def _train_episode(self, episode: int, render_interval: int | None) -> float:
        """Run a single training episode.

        Args:
            episode: Current episode number
            render_interval: Interval for rendering

        Returns:
            Total reward for the episode
        """
        state, _ = self.env.reset()
        state = self._preprocess_state(state)

        episode_reward = 0
        transitions = []

        for _step in range(self.max_steps):
            # Select action
            action_obj = self.agent.act(state)
            action = action_obj.action

            # Take action in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state = self._preprocess_state(next_state)
            done = terminated or truncated

            # Create transition
            transition = Transition(
                state=state,
                action=action_obj,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info,
            )
            transitions.append(transition)

            episode_reward += reward
            state = next_state

            # Render if specified
            if render_interval and episode % render_interval == 0:
                self.env.render()

            if done:
                break

        # Update agent policy
        trajectory = Trajectory.from_transitions(transitions)
        learn_metrics = self.agent.learn(trajectory=trajectory)

        # Unified logging
        if self.writer is not None:
            self.writer.log_scalar('rollout_train/average_return', episode_reward, episode)
            self.writer.log_scalar('rollout_train/episode_length', len(transitions), episode)
            if learn_metrics:
                metrics_dict = {f'learn/{k}': v for k, v in learn_metrics.items()}
                self.writer.log_scalars(metrics_dict, episode)

        return episode_reward

    def evaluate(
        self,
        eval_env: gym.Env | None = None,
        num_episodes: int = 10,
        render: bool = False,
    ) -> tuple[list[float], list[int]]:
        """Evaluate the agent.

        Args:
            eval_env: Environment for evaluation (uses training env if None)
            num_episodes: Number of evaluation episodes
            render: Whether to render the environment

        Returns:
            Tuple of (episode_rewards, episode_lengths)
        """
        if eval_env is None:
            eval_env = self._create_eval_env()

        scores = []
        lengths = []

        try:
            for _episode in range(num_episodes):
                state, _ = eval_env.reset()
                state = self._preprocess_state(state)

                episode_reward = 0
                episode_length = 0

                for _step in range(self.max_steps):
                    # Use agent in evaluation mode
                    action_obj = self.agent.act(state, training=False)
                    action = action_obj.action

                    next_state, reward, terminated, truncated, _ = eval_env.step(action)
                    next_state = self._preprocess_state(next_state)
                    done = terminated or truncated

                    episode_reward += reward
                    episode_length += 1
                    state = next_state

                    if render:
                        eval_env.render()

                    if done:
                        break

                scores.append(episode_reward)
                lengths.append(episode_length)

        finally:
            if eval_env is not self.env:
                eval_env.close()

        return scores, lengths

    def _create_eval_env(self) -> gym.Env:
        """Create evaluation environment.

        Returns:
            Evaluation environment
        """
        # If eval_env was provided, use it directly
        if self.eval_env is not None:
            return self.eval_env

        try:
            # Try to create a deep copy of the training environment
            eval_env = deepcopy(self.env)
        except (TypeError, ValueError, RuntimeError) as e:
            logger.warning(
                f'Failed to deepcopy environment for evaluation: {e}. '
                f'Using training environment for evaluation.'
            )
            return self.env
        else:
            return eval_env

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Preprocess state for training.

        Args:
            state: Raw state from environment

        Returns:
            Preprocessed state
        """
        # Only flatten if state is 2D (e.g., from grid environments)
        if len(state.shape) > 1:
            return state.flatten()
        return state

    def _save_checkpoint(
        self, episode: int, eval_rewards: list[float], save_path: str | None
    ) -> None:
        """Save model checkpoint.

        Args:
            episode: Current episode number
            eval_rewards: Evaluation rewards
            save_path: Base path for saving models
        """
        if save_path is None:
            save_path = 'models'

        models_dir = Path(save_path)
        models_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = models_dir / f'model_episode_{episode}.pth'
        self.agent.save(str(checkpoint_path))

        # Update best model if this is the best so far
        if eval_rewards:
            eval_mean = np.mean(eval_rewards)
            if eval_mean > self.best_eval_score:
                self.best_eval_score = eval_mean
                best_path = models_dir / 'best_model.pth'
                if best_path.exists() or best_path.is_symlink():
                    best_path.unlink()
                best_path.symlink_to(checkpoint_path.name)

    def _log_progress(
        self,
        episode: int,
        num_episodes: int,
        eval_rewards: list[float],
        eval_mean: float,
        verbose: bool,
    ) -> None:
        """Log training progress.

        Args:
            episode: Current episode number
            num_episodes: Total number of episodes
            eval_rewards: Evaluation rewards
            eval_mean: Mean evaluation reward
            verbose: Whether to print verbose logs
        """
        if not verbose:
            return

        # Calculate rollout statistics
        recent_scores = self.episode_rewards[-min(1000, len(self.episode_rewards)) :]
        rollout_stats = {
            'avg_score': np.mean(recent_scores) if recent_scores else 0,
            'std_score': np.std(recent_scores) if recent_scores else 0,
            'min_score': np.min(recent_scores) if recent_scores else 0,
            'max_score': np.max(recent_scores) if recent_scores else 0,
        }

        # Log unified statistics
        logger.info('=' * 80)
        logger.info(
            f'📊 TRAINING SUMMARY - Episode {episode:,}/{num_episodes:,} ({episode / num_episodes * 100:.1f}%)'
        )
        logger.info('=' * 80)

        # Rollout stats
        logger.info('🎮 ROLLOUT:')
        logger.info(
            f'  Average Score: {rollout_stats["avg_score"]:.2f} ± {rollout_stats["std_score"]:.2f}'
        )
        logger.info(
            f'  Score Range: {rollout_stats["min_score"]:.2f} - {rollout_stats["max_score"]:.2f}'
        )

        # Evaluation stats
        if eval_rewards:
            eval_std = np.std(eval_rewards)
            logger.info('🔍 EVALUATION:')
            logger.info(f'  Average Score: {eval_mean:.2f} ± {eval_std:.2f}')

        logger.info('=' * 80)

        # Unified logging for evaluation
        if self.writer is not None and eval_rewards:
            eval_metrics = {
                'rollout_eval/average_return': eval_mean,
                'rollout_eval/episode_length': np.mean([len(eval_rewards)]),
            }
            self.writer.log_scalars(eval_metrics, episode)
