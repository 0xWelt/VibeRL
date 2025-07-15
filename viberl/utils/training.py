from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from viberl.typing import Trajectory, Transition


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

    Raises:
        AssertionError: If intervals are not properly aligned
    """
    # Parameter validation for interval alignment
    if save_interval is not None:
        assert save_interval % eval_interval == 0, (
            f'save_interval ({save_interval}) must be a multiple of eval_interval ({eval_interval})'
        )

    assert log_interval % eval_interval == 0, (
        f'log_interval ({log_interval}) must be a multiple of eval_interval ({eval_interval})'
    )
    scores = []
    eval_scores = []
    best_eval_score = float('-inf')

    # Initialize TensorBoard writer
    writer = None
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    # Create evaluation environment
    # Handle both gym.make() environments and custom environments
    from viberl.envs import SnakeGameEnv

    # Check if this is a snake environment
    if hasattr(env, 'grid_size'):
        eval_env = SnakeGameEnv(render_mode=None, grid_size=env.grid_size)
    else:
        # For MockEnv and other test environments, use the same instance
        eval_env = env

    for episode in range(num_episodes):
        state, _ = env.reset()
        # Only flatten if state is 2D (e.g., from grid environments)
        if len(state.shape) > 1:
            state = state.flatten()
        episode_reward = 0

        # Collect transitions for this episode
        transitions = []

        from viberl.agents.ppo import PPOAgent

        for _step in range(max_steps):
            # Select action using unified Agent interface
            action_obj = agent.act(state)
            action = action_obj.action

            # For PPO, collect additional information
            log_prob = None
            if isinstance(agent, PPOAgent):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    action_probs = agent.policy_network(state_tensor)
                    dist = torch.distributions.Categorical(action_probs)
                    log_prob = dist.log_prob(torch.tensor(action))
                    action_obj.logprobs = log_prob

            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Flatten next state if 2D
            if len(next_state.shape) > 1:
                next_state = next_state.flatten()

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
                env.render()

            if done:
                break

        # Create trajectory and update policy
        trajectory = Trajectory.from_transitions(transitions)
        learn_metrics = agent.learn(trajectory=trajectory)

        scores.append(episode_reward)

        # 1. Evaluation check: iter % eval_interval == 0
        eval_rewards = []
        eval_lengths = []
        eval_mean = 0.0

        if episode % eval_interval == 0:
            eval_rewards, eval_lengths = evaluate_agent(
                eval_env, agent, num_episodes=eval_episodes, render=False, max_steps=max_steps
            )
            eval_mean = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)
            eval_scores.extend(eval_rewards)

        # 2. Checkpoint saving check: iter % save_interval == 0
        if save_interval is not None and episode % save_interval == 0:
            # Save checkpoint to experiment directory
            if log_dir:
                models_dir = Path(log_dir).parent / 'models'
            else:
                models_dir = Path('models')
            models_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = models_dir / f'model_episode_{episode}.pth'
            agent.save(str(checkpoint_path))

            # Update best model if this is the best so far (reuse eval results)
            if eval_rewards and eval_mean > best_eval_score:
                best_eval_score = eval_mean
                best_path = models_dir / 'best_model.pth'
                if best_path.exists() or best_path.is_symlink():
                    best_path.unlink()
                best_path.symlink_to(checkpoint_path.name)

        # 3. Logging check: iter % log_interval == 0
        if episode % log_interval == 0:
            # Calculate rollout statistics
            recent_scores = scores[-min(1000, len(scores)) :] if scores else [0]
            rollout_stats = {
                'avg_score': np.mean(recent_scores),
                'std_score': np.std(recent_scores),
                'min_score': np.min(recent_scores),
                'max_score': np.max(recent_scores),
                'avg_length': np.mean([len(transitions)] * min(1000, len(scores)))
                if transitions
                else 0,
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
            logger.info(f'  Average Episode Length: {rollout_stats["avg_length"]:.1f}')

            # Evaluation stats (only if we have evaluation data)
            if eval_rewards:
                logger.info('🔍 EVALUATION:')
                logger.info(f'  Average Score: {eval_mean:.2f} ± {eval_std:.2f}')
                logger.info(f'  Average Length: {np.mean(eval_lengths):.1f}')

            # Training stats
            if learn_metrics:
                logger.info('🎯 TRAIN:')
                for metric_name, metric_value in learn_metrics.items():
                    logger.info(f'  {metric_name}: {metric_value:.6f}')

            logger.info('=' * 80)

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('rollout_train/average_return', episode_reward, episode)
            writer.add_scalar('rollout_train/episode_length', _step + 1, episode)
            if eval_rewards:
                writer.add_scalar('rollout_eval/average_return', eval_mean, episode)
                writer.add_scalar('rollout_eval/episode_length', np.mean(eval_lengths), episode)
            if learn_metrics:
                for metric_name, metric_value in learn_metrics.items():
                    writer.add_scalar(f'learn/{metric_name}', metric_value, episode)

    # Close environments
    if writer is not None:
        writer.close()
    eval_env.close()

    return scores


def evaluate_agent(
    env: gym.Env, agent: Agent, num_episodes: int = 10, render: bool = False, max_steps: int = 1000
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
    scores = []
    lengths = []

    for _episode in range(num_episodes):
        state, _ = env.reset()
        # Only flatten if state is 2D (e.g., from grid environments)
        if len(state.shape) > 1:
            state = state.flatten()
        episode_reward = 0
        episode_length = 0

        for _step in range(max_steps):
            # Use unified Agent interface
            action_obj = agent.act(state, training=False)
            action = action_obj.action

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Flatten next state if 2D
            if len(next_state.shape) > 1:
                next_state = next_state.flatten()

            episode_reward += reward
            episode_length += 1
            state = next_state

            if render:
                env.render()

            if done:
                break

        scores.append(episode_reward)
        lengths.append(episode_length)

    return scores, lengths
