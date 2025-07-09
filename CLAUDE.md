# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Snake game with Gymnasium interface for both human play and AI training. Supports GUI and text-based play modes, built with Python using pygame and numpy.

## Architecture

The codebase follows a simple 3-layer structure:

- **`core.py`**: Main SnakeGameEnv class implementing gymnasium.Env interface
- **`cli.py`**: Command-line interface with different play modes (human, AI, text)
- **`__init__.py`**: Package exports

### Key Classes

- `SnakeGameEnv`: Core Gymnasium environment with observation/action spaces
- `TextPlayableSnake`: Terminal-based gameplay without GUI
- `HumanPlayableSnake`: GUI-based gameplay with keyboard controls

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

### Game Commands
```bash
# Run human mode with GUI
snake-game human --grid-size 20
snake-human --grid-size 20

# Run text mode (no GUI)
snake-game text --grid-size 15

# Run AI demo
snake-game ai --episodes 5
snake-ai --episodes 5

# Quick aliases
snake-game play      # Same as human
snake-game demo      # AI with 3 episodes
```

### Mode-Specific Options

**Human Mode**: Arrow keys to move, R to restart, Q to quit
**Text Mode**: WASD/HJKL to move, R to restart, Q to quit
**AI Mode**: Random agent, configurable episodes and rendering

## Key Features

- **Dual Interface**: Human play + AI training via Gymnasium API
- **Text Mode**: Terminal play without GUI dependencies
- **Flexible**: Configurable grid size, rendering, episode count
- **Clean API**: Standard gymnasium.Env implementation for RL algorithms

## Dependencies

Core: gymnasium, numpy, pygame
Dev: pytest, black, mypy, flake8
