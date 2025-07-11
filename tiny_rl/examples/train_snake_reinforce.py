#!/usr/bin/env python3
"""
Example: Train Snake Game with REINFORCE Algorithm

This example demonstrates how to train a Snake game agent using the REINFORCE
policy gradient algorithm.
"""

import argparse
import time

import numpy as np

from tiny_rl.agents import REINFORCEAgent
from tiny_rl.envs import SnakeGameEnv
from tiny_rl.utils import evaluate_agent, get_device, set_seed, train_agent


def main():
    parser = argparse.ArgumentParser(description='Train Snake with REINFORCE')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('--grid-size', type=int, default=15, help='Grid size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Evaluation episodes')
    parser.add_argument('--save-path', type=str, default='snake_reinforce', help='Model save path')
    parser.add_argument('--log-dir', type=str, help='Directory for TensorBoard logs')

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

    print('Training REINFORCE agent on Snake game...')
    print(f'Episodes: {args.episodes}')
    print(f'Grid size: {args.grid_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Gamma: {args.gamma}')
    if args.log_dir:
        print(f'TensorBoard logs: {args.log_dir}')

    # Train agent
    start_time = time.time()

    scores = train_agent(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        render_interval=args.episodes // 10,  # Render every 10% of episodes
        save_interval=args.episodes // 5,  # Save every 20% of episodes
        save_path=args.save_path,
        verbose=True,
        log_dir=args.log_dir,
    )

    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.1f} seconds')

    # Save final model
    agent.save_policy(f'{args.save_path}_final.pth')
    print(f'Final model saved to {args.save_path}_final.pth')

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
