#!/usr/bin/env python3
"""
Command-line interface for Tiny RL.
"""

import argparse
import sys

import numpy as np
import torch
from loguru import logger

from viberl.agents import DQNAgent, PPOAgent, REINFORCEAgent
from viberl.envs import SnakeGameEnv
from viberl.trainer import Trainer
from viberl.utils import evaluate_agent, get_device, set_seed
from viberl.utils.experiment_manager import create_experiment


def train_main():
    """Main training CLI entry point."""
    parser = argparse.ArgumentParser(description='Train RL agents')
    parser.add_argument('--env', choices=['snake'], default='snake', help='Environment to train on')
    parser.add_argument(
        '--alg',
        choices=['reinforce', 'dqn', 'ppo'],
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
    parser.add_argument('--save-interval', type=int, help='Save model every N episodes (optional)')
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
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument(
        '--target-update', type=int, default=10, help='Target network update frequency (DQN)'
    )
    parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO clipping parameter')
    parser.add_argument('--ppo-epochs', type=int, default=4, help='PPO epochs per update')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda parameter (PPO)')
    parser.add_argument(
        '--value-loss-coef', type=float, default=0.5, help='Value loss coefficient (PPO)'
    )
    parser.add_argument(
        '--entropy-coef', type=float, default=0.01, help='Entropy coefficient (PPO)'
    )
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm (PPO)')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Evaluation episodes')
    parser.add_argument(
        '--eval-interval', type=int, default=100, help='Evaluation interval during training'
    )
    parser.add_argument(
        '--log-interval', type=int, default=1000, help='Log summary interval during training'
    )
    parser.add_argument(
        '--trajectory-batch',
        type=int,
        default=8,
        help='Number of trajectories to collect per training iteration',
    )
    parser.add_argument(
        '--num-envs',
        type=int,
        default=1,
        help='Number of parallel environments for sampling (use 1 for sequential)',
    )
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation after training')
    parser.add_argument('--quiet', action='store_true', help='Suppress training progress output')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device() if args.device == 'auto' else torch.device(args.device)

    logger.info(f'Using device: {device}')

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
    elif args.alg == 'ppo':
        agent = PPOAgent(
            **base_params,
            clip_epsilon=args.clip_epsilon,
            ppo_epochs=args.ppo_epochs,
            lam=args.gae_lambda,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f'Unknown algorithm: {args.alg}')

    # Move agent to device
    if hasattr(agent, 'policy_network'):
        agent.policy_network.to(device)
    if hasattr(agent, 'q_network'):
        agent.q_network.to(device)
        agent.target_network.to(device)
    if hasattr(agent, 'value_network'):
        agent.value_network.to(device)

    logger.info(f'Training {args.alg} agent on {args.env} environment...')
    logger.info(f'Episodes: {args.episodes}')
    logger.info(f'Grid size: {args.grid_size}')
    logger.info(f'Learning rate: {args.lr}')
    logger.info(f'Gamma: {args.gamma}')

    # Create experiment with automatic directory structure
    experiment_name = args.name or f'{args.alg}_{args.env}'
    exp_manager = create_experiment(experiment_name)
    tb_logs_dir = str(exp_manager.get_tb_logs_path())

    # Configure file logging and log command line arguments
    exp_manager.configure_file_logging(log_level='INFO')
    exp_manager.log_command_line_args(args)

    if tb_logs_dir:
        logger.info(f'TensorBoard logs: {tb_logs_dir}')

    # Create trainer and train agent
    trainer = Trainer(
        env=env,
        agent=agent,
        max_steps=args.max_steps,
        log_dir=tb_logs_dir,
        device=device,
        enable_wandb=args.wandb,
        wandb_config=vars(args),
        run_name=experiment_name,
        batch_size=args.trajectory_batch,
        num_envs=args.num_envs,
    )

    trainer.train(
        num_episodes=args.episodes,
        render_interval=args.render_interval,
        save_interval=args.save_interval,
        save_path=str(exp_manager.get_models_path()),
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        log_interval=args.log_interval,
        verbose=not args.quiet,
    )

    # Save final model
    models_dir = exp_manager.get_models_path()
    final_model_path = str(models_dir / 'final_model.pth')
    agent.save(final_model_path)
    logger.success(f'Final model saved to {final_model_path}')
    logger.success(f'All experiment files saved in: {exp_manager.get_experiment_path()}')

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

    logger.info(f'Using device: {device}')

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
        agent.load(args.model_path)
        logger.success(f'Loaded model from {args.model_path}')
    except OSError as e:
        logger.error(f'Failed to load model: {e}')
        return

    # Evaluate agent
    logger.info(f'Evaluating {args.agent} agent on {args.env} environment...')

    scores = evaluate_agent(env=env, agent=agent, num_episodes=args.episodes, render=args.render)

    logger.info('\nEvaluation Results:')
    logger.success(f'Average score: {np.mean(scores):.2f} ± {np.std(scores):.2f}')
    logger.info(f'Min score: {np.min(scores):.2f}')
    logger.info(f'Max score: {np.max(scores):.2f}')

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

    logger.info(f'Running {args.env} demo for {args.episodes} episodes...')

    # Run demo with random actions
    for episode in range(args.episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0

        logger.info(f'\nEpisode {episode + 1}/{args.episodes}')

        while True:
            action = env.action_space.sample()  # Random action
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                logger.success(
                    f'Episode finished! Score: {info.get("score", 0)}, Steps: {steps}, Total reward: {total_reward}'
                )
                break

    env.close()
    logger.success('\nDemo completed!')


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
        logger.info('Usage: python -m viberl.cli [train|eval|demo] [options...]')
        logger.info('\nExamples:')
        logger.info('  python -m viberl.cli train --episodes 1000 --env snake')
        logger.info('  python -m viberl.cli eval --model-path model.pth --render')
        logger.info('  python -m viberl.cli demo --episodes 5')
