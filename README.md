# 🚀 VibeRL

A Reinforcement Learning framework built essentially through vibe coding. VibeRL provides a simple yet powerful platform for training and evaluating RL agents using modern Python tools.

- **Vibe Coding**: Built through intuitive development practices
- **Gymnasium Compatible**: Standard RL environment interface
- **Human Playable**: Interactive gameplay with keyboard controls
- **Research Ready**: Configurable environments and algorithms

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

### 2. Get Started with VibeRL

**Demo Mode** - Watch random AI play:
```bash
# Run demo with 5 episodes
viberl-demo --episodes 5

# Custom grid size
viberl-demo --episodes 3 --grid-size 15
```

**Training Mode** - Train your first agent:
```bash
# Train REINFORCE agent
viberl-train --episodes 1000 --env snake --agent reinforce

# With custom parameters
viberl-train --episodes 500 --grid-size 20 --lr 1e-4
```

**Evaluation Mode** - Test trained models:
```bash
# Evaluate trained model
viberl-eval --model-path model.pth --render

# Multiple evaluation episodes
viberl-eval --model-path model.pth --episodes 10 --render
```

## Installation Options

### With UV (Recommended)
```bash
# Clone the repository
git clone https://github.com/0xWelt/VibeRL.git
cd VibeRL

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

### Demo Mode
Watch random AI agent play:
```bash
viberl-demo [OPTIONS]

Options:
  --episodes INT      Number of episodes to run (default: 5)
  --grid-size INT     Grid size for the game (default: 15)
```

### Training Mode
Train RL agents using various algorithms:
```bash
viberl-train [OPTIONS]

Options:
  --episodes INT      Number of training episodes (default: 1000)
  --grid-size INT     Grid size for the game (default: 15)
  --lr FLOAT          Learning rate (default: 1e-3)
  --gamma FLOAT       Discount factor (default: 0.99)
  --save-path PATH    Model save path (default: 'trained_model')
```

### Evaluation Mode
Evaluate trained RL models:
```bash
viberl-eval [OPTIONS]

Options:
  --model-path PATH   Path to trained model (required)
  --episodes INT      Number of evaluation episodes (default: 10)
  --grid-size INT     Grid size for the game (default: 15)
  --render            Enable rendering during evaluation
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

## Development Setup

### Setup Development Environment

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/0xWelt/VibeRL.git
cd VibeRL
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Development Tools

```bash
# Format code
uv run ruff format viberl/

# Type checking
uv run mypy viberl/

# Run tests
uv run pytest

# Lint code
uv run ruff check viberl/ --fix
```

## Project Structure

```
VibeRL/
├── pyproject.toml          # UV project configuration
├── LICENSE                 # MIT license
├── README.md              # This file
├── viberl/
│   ├── __init__.py        # Package exports
│   ├── cli.py             # Command-line interface
│   ├── envs/              # RL environments
│   ├── agents/            # RL algorithms
│   ├── utils/             # Utilities and helpers
│   └── examples/          # Training examples
├── tests/                 # Test suite
└── docs/                  # Documentation
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
