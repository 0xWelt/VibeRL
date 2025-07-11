from typing import Any

import gymnasium as gym
import numpy as np


def train_agent(
    env: gym.Env,
    agent: Any,
    num_episodes: int = 1000,
    max_steps: int = 1000,
    render_interval: int | None = None,
    save_interval: int | None = None,
    save_path: str | None = None,
    verbose: bool = True,
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

    Returns:
        List of episode rewards
    """
    scores = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()  # Flatten 2D grid to 1D vector
        episode_reward = 0

        for _step in range(max_steps):
            # Select action
            action = agent.select_action(state)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Flatten next state
            next_state = next_state.flatten()

            # Store transition (if agent supports it)
            if hasattr(agent, 'store_transition'):
                agent.store_transition(state, action, reward)

            episode_reward += reward
            state = next_state

            # Render if specified
            if render_interval and episode % render_interval == 0:
                env.render()

            if done:
                break

        # Update policy (if agent supports it)
        if hasattr(agent, 'update_policy'):
            agent.update_policy()

        scores.append(episode_reward)

        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(
                f'Episode {episode + 1}/{num_episodes}, Average Score (last 100): {avg_score:.2f}'
            )

        # Save model if specified
        if (
            save_interval
            and save_path
            and (episode + 1) % save_interval == 0
            and hasattr(agent, 'save_policy')
        ):
            agent.save_policy(f'{save_path}_episode_{episode + 1}.pth')

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
