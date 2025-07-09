# 🐍 Snake Game with Gymnasium Interface

A fully functional Snake game with dual interface, designed for both human play and AI training with **UV package management**.

- **Human mode**: Playable with arrow keys
- **AI mode**: Standard Gymnasium environment for reinforcement learning

## Features

- 🎮 **Human Playable**: Use arrow keys to control the snake
- 🤖 **Gymnasium API**: Standard RL environment interface
- 🐍 **Classic Gameplay**: Food eating, snake growth, collision detection
- 🎨 **Visual Rendering**: Real-time graphics with pygame
- 📱 **Text Mode**: Play in terminal without GUI
- 📦 **UV Support**: Modern Python packaging with UV

## Quick Start

### 1. Install with UV

```bash
# Install the package
uv pip install -e .

# Or with development dependencies
uv pip install -e ".[dev]"
```

### 2. Play the Game

**Text Mode** - Play in terminal (no GUI required):
```bash
# Text-based CLI game
snake-game text

# with custom grid size
snake-game text --grid-size 15
```

**Human Mode** - Play with GUI:
```bash
# GUI-based game
snake-human

# With custom grid size
snake-human --grid-size 30
```

**AI Mode** - Watch AI play:
```bash
# Demo with random AI
snake-ai

# Multiple episodes
snake-ai --episodes 5
```

**Using the main command**:
```bash
# Text-based play (terminal only)
snake-game text --grid-size 20

# GUI-based play
snake-game human --grid-size 25

# AI demo mode
snake-game ai --episodes 3 --grid-size 20
```

## Installation Options

### With UV (Recommended)
```bash
# Clone the repository
git clone https://github.com/snake-game/snake-game.git
cd snake-game

# Install in development mode
uv pip install -e .

# Or with development tools
uv pip install -e ".[dev]"
```

### With pip
```bash
pip install -e .
pip install -e ".[dev]"  # With dev dependencies
```

## CLI Commands

### Text Mode (No GUI Required)
Play snake game in terminal - perfect for headless environments:
- **W/A/S/D** - Move snake (up/left/down/right)
- **H/J/K/L** - Vim-style movement
- **Space** - Move up
- **R** - Restart game
- **Q** - Quit game

```bash
snake-game text [OPTIONS]

Options:
  --grid-size INT     Grid size for the game (default: 15, max: 25)

# Quick text mode
snake-game text
```

### Human Mode (GUI)
Play snake game with graphics:
- **Arrow Keys** - Move snake (up/down/left/right)
- **R** - Restart game
- **Q** - Quit game

```bash
snake-human [OPTIONS]
snake-game human [OPTIONS]

Options:
  --grid-size INT     Grid size for the game (default: 20)
```

### AI Demo Mode
Watch AI play with random actions:

```bash
snake-ai [OPTIONS]
snake-game ai [OPTIONS]

Options:
  --grid-size INT     Grid size for the game (default: 20)
  --episodes INT      Number of episodes to run (default: 1)
  --render/--no-render Enable/disable rendering (default: enabled)
```

### Unified Interface
Main command with multiple game modes:

```bash
snake-game COMMAND [OPTIONS]

Commands:
  text     Play text-based game (terminal only)
  human    Play GUI-based game
  ai       Run AI demo mode
  play     Alias for human command
  demo     Alias for ai command with 3 episodes
```

## Environment Details

**Observation Space:**
- Grid: `grid_size × grid_size` (default: 20×20)
- Values: 0=empty, 1=snake body, 2=snake head, 3=food

**Action Space:**
- 0: UP
- 1: RIGHT
- 2: DOWN
- 3: LEFT

**Rewards:**
- +1: Eating food
- -1: Collision (game over)
- 0: Regular move

## Integrating with Reinforcement Learning

```python
from snake_game import SnakeGameEnv, Direction

# Create environment
env = SnakeGameEnv(render_mode="human")

# Run episode
observation, info = env.reset()
total_reward = 0

while True:
    action = env.action_space.sample()  # Your RL policy here
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

print(f"Score: {info['score']}, Total reward: {total_reward}")
env.close()
```

### Custom Environment

```python
# Custom grid size
env = SnakeGameEnv(grid_size=30, render_mode="human")

# Inherit and extend
class CustomSnakeEnv(SnakeGameEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom features like different rewards, obstacles, etc.

# Use with popular RL frameworks
import stable_baselines3 as sb3
from snake_game import SnakeGameEnv

env = SnakeGameEnv()
model = sb3.PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Development Setup

### Setup Development Environment

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo-url>
cd snake-game
uv venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
uv pip install -e ".[dev]"
```

### Development Tools

```bash
# Format code
uv run black src/

# Type checking
uv run mypy src/snake_game/

# Run tests
uv run pytest

# Lint code
uv run flake8 src/
```

## Game Logic

- 🐍 **Snake**: Green segments (body and head)
- 🍎 **Food**: Red squares
- 🎯 **Objective**: Eat food to grow and increase score
- 💀 **Game Over**: Collision with walls, own body, or max steps reached
- ⚡ **Controls**: Various options available
  - **GUI Mode**: Arrow keys, R/Q for restart/quit
  - **Text Mode**: WASD/HJKL, Space/R/Q
  - **AI Mode**: Zero direct user input

## Project Structure

```
snake-game/
├── pyproject.toml          # UV project configuration
├── LICENSE                 # MIT license
├── README.md              # This file
├── src/
│   └── snake_game/
│       ├── __init__.py    # Package exports
│       ├── core.py        # Main Snake game environment
│       └── cli.py         # Command-line interface
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `uv pip install -e ".[dev]"`
4. Make your changes and add tests
5. Run tests: `uv run pytest`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
