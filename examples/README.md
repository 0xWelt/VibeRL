# VibeRL Examples

This directory contains convenient shell scripts for running VibeRL experiments.

## Quick Start

### Training
```bash
# Train with default settings
./train.sh

# Train DQN for 1000 episodes
./train.sh dqn snake 1000

# Train REINFORCE with custom parameters
./train.sh reinforce snake 2000 --lr 0.001 --grid-size 20
```

### Evaluation
```bash
# Evaluate a trained model
./eval.sh dqn snake experiments/dqn_snake_20250712/final_model.pth 10

# Evaluate with rendering
./eval.sh reinforce snake experiments/reinforce_snake_20250712/final_model.pth 5 --render
```

### Demo
```bash
# Run demo with random actions
./demo.sh random snake 5
```

## Advanced Usage

All scripts support additional arguments that are passed directly to the underlying CLI:

```bash
# DQN with custom parameters
./train.sh dqn snake 500 --lr 0.0005 --epsilon-start 0.5 --memory-size 50000

# REINFORCE with custom network
./train.sh reinforce snake 1000 --hidden-size 256 --num-hidden-layers 3
```

## Output Structure

All experiments are saved in the `experiments/` directory with the format:
```
experiments/
├── {alg}_{env}_20250712_143000/
│   ├── tb_logs/          # TensorBoard logs
│   ├── models/           # Saved models
│   └── final_model.pth   # Final trained model
```

## Available Algorithms

- **reinforce**: Policy gradient method
- **dqn**: Deep Q-Network with experience replay

## Available Environments

- **snake**: Snake game environment (grid-based)

## Usage Examples

### Basic Training
```bash
# Default: REINFORCE on Snake for 1000 episodes
./train.sh

# DQN training
./train.sh dqn

# Custom episode count
./train.sh reinforce snake 5000
```

### With Custom Name
```bash
# Use custom experiment name
./train.sh dqn snake 2000 --name my_dqn_experiment
```

### With Additional Parameters
```bash
# DQN with larger memory and smaller learning rate
./train.sh dqn snake 1000 --lr 0.0001 --memory-size 20000

# REINFORCE with larger network
./train.sh reinforce snake 2000 --hidden-size 256 --num-hidden-layers 4
```
