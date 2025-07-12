# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Instructions for Claude Code

1. After you have made changes to the code:
   1.  Make sure `pre-commit run --all` passes. If it doesn't, fix the issues and run again. You should always fix the issues and do not use `noqa` to skip any issues.
   2.  Make sure `pytest` passes. If it doesn't, fix the issues and run again.
   3.  Always keep `CLAUDE.md, README.md, pyproject.toml` up to date.
2. You should implement tests for each part of the code and try to coverage as much as possible.
3. When creating a git commit, scan all newly-added and modified files; if any of them shouldn't be committed, update .gitignore to exclude them, then run git add . to stage all changes.
4. Never commit with --no-verify, make sure to pass pre-commit

## Project Overview

VibeRL - A modern Reinforcement Learning framework built with type safety and modern Python practices. Features three algorithms (REINFORCE, PPO, DQN) with a unified agent interface and modern type system using pydantic.

## Architecture

The codebase follows a clean 4-layer architecture:

- **`viberl/typing.py`**: Modern type system (Action, Transition, Trajectory) using pydantic
- **`viberl/agents/`**: RL algorithms with unified interface (REINFORCE, PPO, DQN)
- **`viberl/envs/`**: Environments (SnakeGameEnv implementing gymnasium.Env)
- **`viberl/cli.py`**: Command-line interface with train/eval/demo modes
- **`viberl/utils/`**: Training utilities and experiment management

### Key Classes

- `Action`: Type-safe action with optional log probabilities
- `Transition`: Single step data structure
- `Trajectory`: Complete episode data
- `Agent`: Abstract base class for all RL agents
- `SnakeGameEnv`: Gymnasium-compatible snake game environment
- `REINFORCEAgent`, `PPOAgent`, `DQNAgent`: Algorithm implementations

## Common Commands

### Development Setup
```bash
# Install dependencies
uv pip install -e ".[dev]"

# Format and lint
uv run ruff format viberl/
uv run ruff check viberl/ --fix

# Run tests
uv run pytest

# Type checking
uv run mypy viberl/
```

### CLI Commands
```bash
# Train agents
viberl-train --alg [reinforce|dqn|ppo] --episodes 1000 --grid-size 15

# Evaluate trained models
viberl-eval --model-path path/to/model.pth --episodes 10

# Run demo
viberl-demo --episodes 5 --grid-size 10

# Play human
python examples/human_play/play_human.py
```

## Key Features

- **Type Safety**: Pydantic-based type system with Action, Transition, Trajectory
- **Unified Interface**: All agents inherit from base Agent class
- **Modern Python**: 3.12+ with type hints and future annotations
- **Comprehensive Testing**: 49+ tests covering all components
- **Experiment Management**: Automatic directory structure with TensorBoard logging
- **Extensible**: Easy to add new algorithms and environments

## Dependencies

Core: gymnasium, numpy, pygame, torch, pydantic
Dev: pytest, ruff, mypy, pre-commit

## Extension Points

### Add New Algorithm
1. Inherit from `Agent` class
2. Implement `act()` and `learn()` methods
3. Use provided `Action`, `Transition`, `Trajectory` types
4. Integrate with training pipeline

### Add New Environment
1. Implement gymnasium.Env interface
2. Use standard observation/action spaces
3. Integrate with existing training pipeline
