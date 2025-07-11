# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Instructions for Claude Code

1. After you have made changes to the code:
   1.  Make sure `pre-commit run --all` passes. If it doesn't, fix the issues and run again. You should always fix the issues and do not use `noqa` to skip any issues.
   2.  Make sure `pytest -n 8` passes. If it doesn't, fix the issues and run again.
   3.  Always keep `CLAUDE.md, README.md, pyproject.toml` up to date.
2. You should implement tests for each part of the code and try to coverage as much as possible.
3. When creating a git commit, scan all newly-added and modified files; if any of them shouldn't be committed, update .gitignore to exclude them, then run git add . to stage all changes.

## Project Overview

VibeRL - A Reinforcement Learning framework built essentially through vibe coding. Built with Python using pygame, numpy, and PyTorch.

## Architecture

The codebase follows a simple 3-layer structure:

- **`viberl/envs/`**: Environments including SnakeGameEnv implementing gymnasium.Env interface
- **`viberl/cli.py`**: Command-line interface with different play modes (train, eval, demo)
- **`viberl/`**: Main package with agents, utils, and examples

### Key Classes

- `SnakeGameEnv`: Core Gymnasium environment with observation/action spaces
- `REINFORCEAgent`: Policy gradient agent for training
- `PolicyNetwork`: Neural network for policy approximation

## Common Commands

### Development Setup
```bash
# Install dependencies
uv pip install -e ".[dev]"

# Format code with ruff
uv run ruff format src/

# Lint code with ruff
uv run ruff check src/ --fix

# Run tests
uv run pytest

# Lint code
uv run flake8 src/
```

### CLI Commands
```bash
# Train an agent
viberl-train --episodes 1000 --env snake --agent reinforce

# Evaluate a trained model
viberl-eval --model-path model.pth --render

# Run demo with random actions
viberl-demo --episodes 5

# Run example training script
python examples/training/train_snake_reinforce.py --episodes 1000
```

### Mode-Specific Options

**Training Mode**: Configure episodes, learning rate, grid size, etc.
**Evaluation Mode**: Load trained models, set render options
**Demo Mode**: Random agent with configurable episodes and rendering

## Key Features

- **Dual Interface**: Human play + AI training via Gymnasium API
- **Policy Gradient**: REINFORCE algorithm implementation
- **Flexible**: Configurable grid size, rendering, episode count
- **Clean API**: Standard gymnasium.Env implementation for RL algorithms
- **Examples**: Ready-to-run training scripts

## Dependencies

Core: gymnasium, numpy, pygame, torch
Dev: pytest, ruff, mypy
