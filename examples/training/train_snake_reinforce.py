#!/usr/bin/env python3
"""
Example: Train Snake Game with REINFORCE Algorithm

This example demonstrates how to train a Snake game agent using the REINFORCE
policy gradient algorithm.
"""

import argparse
import time

import numpy as np

from viberl.agents import REINFORCEAgent
from viberl.envs import SnakeGameEnv
from viberl.utils import evaluate_agent, get_device, set_seed, train_agent, create_experiment


def main():
    parser = argparse.ArgumentParser(description='Train Snake with REINFORCE')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('--grid-size', type=int, default=15, help='Grid size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Evaluation episodes')
    parser.add_argument('--name', type=str, default='snake_reinforce', help='Experiment name for automatic directory creation')

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Get device
    device = get_device()
    print(f'Using device: {device}')

    # Create environment
    env = SnakeGameEnv(grid_size=args.grid_size)

    # Calculate state and action sizes
    state_size = args.grid_size * args.grid_size
    action_size = 4  # UP, RIGHT, DOWN, LEFT

    # Create agent
    agent = REINFORCEAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
    )

    # Move agent to device
    agent.policy_network.to(device)

    # Create experiment with automatic directory structure
    exp_manager = create_experiment(args.name)
    tb_logs_dir = str(exp_manager.get_tb_logs_path())
    models_dir = exp_manager.get_models_path()
    save_path = str(models_dir / 'model')

    print('Training REINFORCE agent on Snake game...')
    print(f'Episodes: {args.episodes}')
    print(f'Grid size: {args.grid_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Gamma: {args.gamma}')
    print(f'TensorBoard logs: {tb_logs_dir}')
    print(f'Experiment directory: {exp_manager.get_experiment_path()}')

    # Train agent
    start_time = time.time()

    scores = train_agent(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        render_interval=args.episodes // 10,  # Render every 10% of episodes
        save_interval=args.episodes // 5,  # Save every 20% of episodes
        save_path=save_path,
        verbose=True,
        log_dir=tb_logs_dir,
    )

    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.1f} seconds')

    # Save final model
    final_model_path = str(models_dir / 'final_model.pth')
    agent.save_policy(final_model_path)
    print(f'Final model saved to {final_model_path}')
    print(f'All experiment files saved in: {exp_manager.get_experiment_path()}')

    # Evaluate agent
    print(f'\nEvaluating agent over {args.eval_episodes} episodes...')

    eval_env = SnakeGameEnv(render_mode='human', grid_size=args.grid_size)
    eval_scores = evaluate_agent(
        env=eval_env, agent=agent, num_episodes=args.eval_episodes, render=True
    )

    eval_env.close()
    env.close()

    print('\nFinal Results:')
    print(f'Average training score (last 100 episodes): {np.mean(scores[-100:]):.2f}')
    print(f'Average evaluation score: {np.mean(eval_scores):.2f}')
    print(f'Training time: {training_time:.1f} seconds')


if __name__ == '__main__':
    main()
