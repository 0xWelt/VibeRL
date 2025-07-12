import os
from typing import Any

import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def train_agent(
    env: gym.Env,
    agent: Any,
    num_episodes: int = 1000,
    max_steps: int = 1000,
    render_interval: int | None = None,
    save_interval: int | None = None,
    save_path: str | None = None,
    verbose: bool = True,
    log_dir: str | None = None,
) -> list[float]:
    """
    Generic training function for RL agents.

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

    Returns:
        List of episode rewards
    """
    scores = []

    # Initialize TensorBoard writer
    writer = None
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()  # Flatten 2D grid to 1D vector
        episode_reward = 0

        # Handle PPO agent's experience collection
        if hasattr(agent, 'act'):  # PPO agent
            action, log_prob, value = agent.act(state)
        else:  # REINFORCE/DQN agent
            action = agent.select_action(state)
            log_prob, value = 0.0, 0.0

        for _step in range(max_steps):
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Flatten next state
            next_state = next_state.flatten()

            # Handle different agent types
            if hasattr(agent, 'store_experience'):  # PPO agent
                agent.store_experience(state, action, reward, done, log_prob, value)
            elif hasattr(agent, 'store_transition'):  # DQN/REINFORCE agents
                # Handle different agent interfaces
                if hasattr(agent, 'memory'):  # DQN agent
                    agent.store_transition(state, action, reward, next_state, done)
                else:  # REINFORCE agent
                    agent.store_transition(state, action, reward)

            episode_reward += reward
            state = next_state

            # Handle PPO's next action
            if hasattr(agent, 'act'):  # PPO agent
                if not done:
                    action, log_prob, value = agent.act(state)
                else:
                    # Store final state with value 0
                    agent.store_experience(state, action, 0.0, done, 0.0, 0.0)

            # Render if specified
            if render_interval and episode % render_interval == 0:
                env.render()

            if done:
                break

        # Update policy based on agent type
        if hasattr(agent, 'update'):  # PPO agent
            if (episode + 1) % 10 == 0:  # Update every 10 episodes
                metrics = agent.update()
                if verbose and metrics:
                    print(
                        f'PPO Update - Policy Loss: {metrics.get("policy_loss", 0):.4f}, '
                        f'Value Loss: {metrics.get("value_loss", 0):.4f}'
                    )
        elif hasattr(agent, 'update_policy'):  # REINFORCE agent
            agent.update_policy()
        elif hasattr(agent, 'update_target_network') and (episode + 1) % 10 == 0:  # DQN agent
            agent.update_target_network()

        scores.append(episode_reward)

        # Print progress and log to TensorBoard
        if verbose and (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(
                f'Episode {episode + 1}/{num_episodes}, Average Score (last 100): {avg_score:.2f}'
            )

        # Log metrics to TensorBoard
        if writer is not None:
            # Only log final return (episode reward) and episode length
            writer.add_scalar('final_return', episode_reward, episode)
            writer.add_scalar('episode_length', _step + 1, episode)

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
            agent.save_policy(filename)

    # Close TensorBoard writer
    if writer is not None:
        writer.close()

    return scores


def evaluate_agent(
    env: gym.Env, agent: Any, num_episodes: int = 10, render: bool = False, max_steps: int = 1000
) -> list[float]:
    """
    Generic evaluation function for RL agents.

    Args:
        env: Gymnasium environment
        agent: RL agent with select_action method
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        max_steps: Maximum steps per episode

    Returns:
        List of episode rewards
    """
    scores = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()  # Flatten 2D grid to 1D vector
        episode_reward = 0

        for _step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Flatten next state
            next_state = next_state.flatten()

            episode_reward += reward
            state = next_state

            if render:
                env.render()

            if done:
                break

        scores.append(episode_reward)

        agent_name = agent.__class__.__name__ if hasattr(agent, '__class__') else 'Agent'

        print(f'{agent_name} - Evaluation Episode {episode + 1}: Score = {episode_reward}')

    print(f'Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}')
    return scores
