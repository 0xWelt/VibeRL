#!/usr/bin/env python3
"""
Command-line interface for Tiny RL.
"""

import argparse
import sys

import numpy as np
import torch

from viberl.agents import DQNAgent, REINFORCEAgent
from viberl.envs import SnakeGameEnv
from viberl.utils import evaluate_agent, get_device, set_seed, train_agent
from viberl.utils.experiment_manager import create_experiment


def train_main():
    """Main training CLI entry point."""
    parser = argparse.ArgumentParser(description='Train RL agents')
    parser.add_argument('--env', choices=['snake'], default='snake', help='Environment to train on')
    parser.add_argument(
        '--alg',
        choices=['reinforce', 'dqn'],
        default='reinforce',
        help='Reinforcement learning algorithm',
    )
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--grid-size', type=int, default=15, help='Grid size for snake environment')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--num-hidden-layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--name', type=str, help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--render-interval', type=int, help='Render every N episodes')
    parser.add_argument('--save-interval', type=int, help='Save model every N episodes')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='auto', help='Device to use')
    parser.add_argument(
        '--epsilon-start', type=float, default=1.0, help='Initial exploration rate (DQN)'
    )
    parser.add_argument(
        '--epsilon-end', type=float, default=0.01, help='Final exploration rate (DQN)'
    )
    parser.add_argument(
        '--epsilon-decay', type=float, default=0.995, help='Exploration decay rate (DQN)'
    )
    parser.add_argument('--memory-size', type=int, default=10000, help='Replay memory size (DQN)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (DQN)')
    parser.add_argument(
        '--target-update', type=int, default=10, help='Target network update frequency (DQN)'
    )
    parser.add_argument('--eval-episodes', type=int, default=10, help='Evaluation episodes')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation after training')
    parser.add_argument('--quiet', action='store_true', help='Suppress training progress output')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device() if args.device == 'auto' else torch.device(args.device)

    print(f'Using device: {device}')

    # Create environment
    if args.env == 'snake':
        env = SnakeGameEnv(grid_size=args.grid_size)
        state_size = args.grid_size * args.grid_size
        action_size = 4
    else:
        raise ValueError(f'Unknown environment: {args.env}')

    # Create agent
    base_params = {
        'state_size': state_size,
        'action_size': action_size,
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.num_hidden_layers,
    }

    if args.alg == 'reinforce':
        agent = REINFORCEAgent(**base_params)
    elif args.alg == 'dqn':
        agent = DQNAgent(
            **base_params,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            target_update=args.target_update,
        )
    else:
        raise ValueError(f'Unknown algorithm: {args.alg}')

    # Move agent to device
    if hasattr(agent, 'policy_network'):
        agent.policy_network.to(device)
    if hasattr(agent, 'q_network'):
        agent.q_network.to(device)
        agent.target_network.to(device)

    print(f'Training {args.alg} agent on {args.env} environment...')
    print(f'Episodes: {args.episodes}')
    print(f'Grid size: {args.grid_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Gamma: {args.gamma}')

    # Create experiment with automatic directory structure
    experiment_name = args.name or f'{args.alg}_{args.env}'
    exp_manager = create_experiment(experiment_name)
    tb_logs_dir = str(exp_manager.get_tb_logs_path())
    models_dir = exp_manager.get_models_path()
    save_path = str(models_dir / 'model')

    if tb_logs_dir:
        print(f'TensorBoard logs: {tb_logs_dir}')

    # Train agent
    train_agent(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        render_interval=args.render_interval,
        save_interval=args.save_interval,
        save_path=save_path,
        verbose=True,
        log_dir=tb_logs_dir,
    )

    # Save final model
    final_model_path = str(models_dir / 'final_model.pth')
    agent.save_policy(final_model_path)
    print(f'Final model saved to {final_model_path}')
    print(f'All experiment files saved in: {exp_manager.get_experiment_path()}')

    env.close()


def eval_main():
    """Main evaluation CLI entry point."""
    parser = argparse.ArgumentParser(description='Evaluate trained RL agents')
    parser.add_argument(
        '--env', choices=['snake'], default='snake', help='Environment to evaluate on'
    )
    parser.add_argument(
        '--agent', choices=['reinforce'], default='reinforce', help='Agent algorithm'
    )
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--grid-size', type=int, default=15, help='Grid size for snake environment')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='auto', help='Device to use')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device() if args.device == 'auto' else torch.device(args.device)

    print(f'Using device: {device}')

    # Create environment
    if args.env == 'snake':
        env = SnakeGameEnv(render_mode='human' if args.render else None, grid_size=args.grid_size)
        state_size = args.grid_size * args.grid_size
        action_size = 4
    else:
        raise ValueError(f'Unknown environment: {args.env}')

    # Create agent
    if args.agent == 'reinforce':
        agent = REINFORCEAgent(state_size=state_size, action_size=action_size)
    else:
        raise ValueError(f'Unknown agent: {args.agent}')

    # Move agent to device
    agent.policy_network.to(device)

    # Load trained model
    try:
        agent.load_policy(args.model_path)
        print(f'Loaded model from {args.model_path}')
    except OSError as e:
        print(f'Failed to load model: {e}')
        return

    # Evaluate agent
    print(f'Evaluating {args.agent} agent on {args.env} environment...')

    scores = evaluate_agent(env=env, agent=agent, num_episodes=args.episodes, render=args.render)

    print('\nEvaluation Results:')
    print(f'Average score: {np.mean(scores):.2f} ± {np.std(scores):.2f}')
    print(f'Min score: {np.min(scores):.2f}')
    print(f'Max score: {np.max(scores):.2f}')

    env.close()


def demo_main():
    """Demo CLI entry point."""
    parser = argparse.ArgumentParser(description='Run RL framework demos')
    parser.add_argument('--env', choices=['snake'], default='snake', help='Environment to demo')
    parser.add_argument('--episodes', type=int, default=5, help='Number of demo episodes')
    parser.add_argument('--grid-size', type=int, default=15, help='Grid size for snake environment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create environment
    if args.env == 'snake':
        env = SnakeGameEnv(render_mode='human', grid_size=args.grid_size)
    else:
        raise ValueError(f'Unknown environment: {args.env}')

    print(f'Running {args.env} demo for {args.episodes} episodes...')

    # Run demo with random actions
    for episode in range(args.episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0

        print(f'\nEpisode {episode + 1}/{args.episodes}')

        while True:
            action = env.action_space.sample()  # Random action
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                print(
                    f'Episode finished! Score: {info.get("score", 0)}, Steps: {steps}, Total reward: {total_reward}'
                )
                break

    env.close()
    print('\nDemo completed!')


if __name__ == '__main__':
    # Simple dispatcher
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        train_main()
    elif len(sys.argv) > 1 and sys.argv[1] == 'eval':
        sys.argv.pop(1)
        eval_main()
    elif len(sys.argv) > 1 and sys.argv[1] == 'demo':
        sys.argv.pop(1)
        demo_main()
    else:
        print('Usage: python -m viberl.cli [train|eval|demo] [options...]')
        print('\nExamples:')
        print('  python -m viberl.cli train --episodes 1000 --env snake')
        print('  python -m viberl.cli eval --model-path model.pth --render')
        print('  python -m viberl.cli demo --episodes 5')
