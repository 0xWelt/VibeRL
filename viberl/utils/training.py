import os

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

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
    """
    scores = []
    eval_scores = []

    # Initialize TensorBoard writer
    writer = None
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    # Create evaluation environment
    # Handle both gym.make() environments and custom environments
    from viberl.envs import SnakeGameEnv

    if hasattr(env, 'grid_size'):
        eval_env = SnakeGameEnv(render_mode=None, grid_size=env.grid_size)
    else:
        eval_env = SnakeGameEnv(render_mode=None)

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()  # Flatten 2D grid to 1D vector
        episode_reward = 0

        # Collect trajectories for this episode
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        log_probs = []
        values = []

        from viberl.agents.ppo import PPOAgent

        for _step in range(max_steps):
            # Select action using unified Agent interface
            action = agent.act(state)

            # For PPO, collect additional information
            if isinstance(agent, PPOAgent):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    action_probs = agent.policy_network(state_tensor)
                    dist = torch.distributions.Categorical(action_probs)
                    log_prob = dist.log_prob(torch.tensor(action)).item()
                    value = agent.value_network(state_tensor).squeeze(-1).item()
                    log_probs.append(log_prob)
                    values.append(value)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Flatten next state
            next_state = next_state.flatten()

            # Store trajectory data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            episode_reward += reward
            state = next_state

            # Render if specified
            if render_interval and episode % render_interval == 0:
                env.render()

            if done:
                break

        # Update policy using unified Agent interface with collected trajectories
        learn_metrics = {}
        if (episode + 1) % 10 == 0:  # Update every 10 episodes
            # Prepare arguments based on agent type
            learn_kwargs = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones,
            }

            # Add PPO-specific arguments
            if isinstance(agent, PPOAgent) and log_probs and values:
                learn_kwargs.update(
                    {
                        'log_probs': log_probs,
                        'values': values,
                    }
                )

            learn_metrics = agent.learn(**learn_kwargs)
            if verbose and learn_metrics:
                # Display returned metrics
                metrics_str = ', '.join(f'{k}: {v:.4f}' for k, v in learn_metrics.items())
                print(f'Update - {metrics_str}')

        # Log learn/ metrics
        if writer is not None and learn_metrics:
            for metric_name, metric_value in learn_metrics.items():
                writer.add_scalar(f'learn/{metric_name}', metric_value, episode)

        scores.append(episode_reward)

        # Log training metrics to TensorBoard
        if writer is not None:
            # rollout_train/ metrics - training environment metrics
            writer.add_scalar('rollout_train/average_return', episode_reward, episode)
            writer.add_scalar('rollout_train/episode_length', _step + 1, episode)

        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(
                f'Episode {episode + 1}/{num_episodes}, Average Score (last 100): {avg_score:.2f}'
            )

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            eval_rewards, eval_lengths = evaluate_agent(
                eval_env, agent, num_episodes=eval_episodes, render=False, max_steps=max_steps
            )
            eval_mean = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)
            eval_scores.extend(eval_rewards)

            if writer is not None:
                # rollout_eval/ metrics - evaluation environment metrics
                writer.add_scalar('rollout_eval/average_return', eval_mean, episode)
                writer.add_scalar(
                    'rollout_eval/episode_length',
                    np.mean(eval_lengths) if eval_lengths else 0,
                    episode,
                )

            if verbose:
                print(
                    f'Evaluation at episode {episode + 1}: Mean={eval_mean:.2f}, Std={eval_std:.2f}'
                )

        # Save model if specified
        if (
            save_interval
            and save_path
            and (episode + 1) % save_interval == 0
            and hasattr(agent, 'save_policy')
        ):
            # Ensure save_path is a directory path
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            # Create proper filename
            filename = os.path.join(save_dir, f'model_episode_{episode + 1}.pth')
            agent.save(filename)

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

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()  # Flatten 2D grid to 1D vector
        episode_reward = 0
        episode_length = 0

        for _step in range(max_steps):
            # Use unified Agent interface
            action = agent.act(state, training=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Flatten next state
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

        agent_name = agent.__class__.__name__ if hasattr(agent, '__class__') else 'Agent'

        print(
            f'{agent_name} - Evaluation Episode {episode + 1}: Score = {episode_reward}, Length = {episode_length}'
        )

    print(
        f'Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}, Average Length: {np.mean(lengths):.2f}'
    )
    return scores, lengths
