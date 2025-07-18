# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Instructions for Claude Code

1. You should implement tests for each part of the code and try to coverage as much as possible.
2. You can run commands to verify your code. Note:
   1. Do not use commands that may block program execution, such as `mkdocs serve` or `tensorboard`.
   2. You can create temporary files to test your code and delete them after you are done.
3. After you have done your changes:
   1.  Make sure `pytest -n 8` passes. If it doesn't, fix the issues and run again.
   2.  Make sure `pre-commit run --all` passes. If it doesn't, fix the issues and run again.
   3.  Check if `CLAUDE.md, README.md, pyproject.toml` are up to date, change them according to the changes you made.
4. When creating a git commit:
   1. Scan all newly-added and modified files; if any of them shouldn't be committed, update .gitignore to exclude them, then run `git add .` to stage all changes.
   2. Always create a new commit, do not use `git commit --amend` to modify the previous commit.
   3. Never commit with `--no-verify`, make sure to pass pre-commit. The pre-commit hook has auto-fix enabled so you may run `pre-commit run --all` twice to check if there are any issues remaining.

## Project Overview

VibeRL - A modern Reinforcement Learning framework built with type safety and modern Python practices. Features three algorithms (REINFORCE, PPO, DQN) with a unified agent interface and modern type system using pydantic.

## Current Architecture (Updated)

The codebase follows a clean 5-layer architecture:

- **`viberl/typing.py`**: Modern type system (Action, Transition, Trajectory) using pydantic
- **`viberl/agents/`**: RL algorithms with unified interface (REINFORCE, PPO, DQN)
- **`viberl/envs/`**: Environments (SnakeGameEnv implementing gymnasium.Env)
- **`viberl/networks/`**: Neural network implementations (base, policy, value networks)
- **`viberl/cli.py`**: Command-line interface with train/eval/demo modes
- **`viberl/utils/`**: Training utilities and experiment management
- **`viberl/utils/vector_env.py`**: Parallel sampling with AsyncVectorEnv

### Key Classes & Components

- **Types** (`viberl.typing`):
  - `Action`: Type-safe action with optional log probabilities (pydantic model)
  - `Transition`: Single step data structure (pydantic model)
  - `Trajectory`: Complete episode data (pydantic model with helper methods)

- **Agents** (`viberl.agents`):
  - `Agent`: Abstract base class for all RL agents
  - `REINFORCEAgent`, `PPOAgent`, `DQNAgent`: Algorithm implementations

- **Networks** (`viberl.networks`):
  - `BaseNetwork`: Base neural network class
  - `PolicyNetwork`: Policy network implementations
  - `ValueNetwork`: Value network implementations

- **Environment** (`viberl.envs`):
  - `SnakeGameEnv`: Gymnasium-compatible snake game environment

- **CLI** (`viberl.cli`):
  - `viberl-train`: Comprehensive training with all algorithms
  - `viberl-eval`: Model evaluation with rendering support
  - `viberl-demo`: Random action demonstrations

## Common Commands

### Development Setup
```bash
# Install dependencies
uv pip install -e ".[dev]"

# Format and lint
uv run ruff format viberl/
uv run ruff check viberl/ --fix

# Run tests
uv run pytest -n 8

# Type checking (if mypy is configured)
uv run mypy viberl/
```

### CLI Commands
```bash
# Train agents with full parameter support
viberl-train --alg [reinforce|dqn|ppo] --episodes 1000 --grid-size 15 --lr 0.001

# Evaluate trained models
viberl-eval --model-path experiments/reinforce_snake/final_model.pth --episodes 10 --render

# Run demo
viberl-demo --episodes 5 --grid-size 10

# Play human
python examples/human_play/play_human.py
```

### Example Usage
```bash
# Train REINFORCE
viberl-train --alg reinforce --episodes 1000 --grid-size 10 --lr 0.001

# Train REINFORCE with batch learning (multiple trajectories per iteration)
viberl-train --alg reinforce --episodes 1000 --grid-size 10 --lr 0.001 --trajectory-batch 4

# Train DQN with experience replay
viberl-train --alg dqn --episodes 2000 --grid-size 15 --memory-size 10000 --batch-size 64

# Train PPO with batch learning and advanced parameters
viberl-train --alg ppo --episodes 1000 --grid-size 12 --ppo-epochs 4 --clip-epsilon 0.2 --trajectory-batch 8

# Evaluate model
viberl-eval --model-path experiments/ppo_snake_20241231_120000/final_model.pth --episodes 10 --render
```

## Key Features

- **Type Safety**: Pydantic-based type system with Action, Transition, Trajectory
- **Unified Interface**: All agents inherit from base Agent class with act() and learn() methods
- **Modern Python**: 3.12+ with type hints and future annotations
- **Comprehensive Testing**: 50+ tests covering all components
- **Experiment Management**: Automatic directory structure with TensorBoard logging
- **CLI Interface**: Complete command-line interface with training, evaluation, and demo modes
- **Network Layer**: Separate neural network implementations for policy and value functions

## Dependencies

Core: gymnasium, numpy, pygame, torch, pydantic, tensorboard, mkdocs (for docs)
Dev: pytest, pytest-xdist, pytest-cov, pre-commit, ruff

## Extension Points

### Add New Algorithm
1. Inherit from `Agent` class in `viberl/agents/base.py`
2. Implement `act()` and `learn()` methods
3. Use provided `Action`, `Transition`, `Trajectory` types from `viberl.typing`
4. Add new agent to `viberl/agents/__init__.py`
5. Update CLI in `viberl/cli.py` (if new algorithm-specific parameters needed)

### Add New Environment
1. Implement gymnasium.Env interface in `viberl/envs/`
2. Use standard observation/action spaces
3. Integrate with existing training pipeline via CLI parameters

## Directory Structure
```
viberl/
├── agents/           # RL algorithm implementations
│   ├── base.py       # Base Agent class
│   ├── reinforce.py  # REINFORCE agent
│   ├── dqn.py        # DQN agent
│   └── ppo.py        # PPO agent
├── envs/             # Environment implementations
│   └── grid_world/   # Snake game environment
├── networks/         # Neural network components
│   ├── base_network.py
│   ├── policy_network.py
│   └── value_network.py
├── utils/            # Training utilities
│   ├── training.py   # Core training loops
│   ├── experiment_manager.py  # Experiment tracking
│   └── common.py     # Utility functions
├── typing.py         # Modern type system
└── cli.py           # Command-line interface

# Supporting files
├── pyproject.toml    # Project configuration
├── README.md        # Project overview
├── docs/            # Documentation
├── examples/        # Usage examples
└── tests/           # Test suite
```
