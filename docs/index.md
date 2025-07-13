# VibeRL Documentation

Welcome to **VibeRL** - A modern Reinforcement Learning framework built with type safety and modern Python practices.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/0xWelt/VibeRL.git
cd VibeRL

# Install using uv
uv pip install -e "."

# Or install with development dependencies
uv pip install -e ".[dev]"
```

### Your First Training

```bash
# Train a REINFORCE agent
viberl-train --alg reinforce --episodes 1000 --grid-size 10

# Train a DQN agent
viberl-train --alg dqn --episodes 2000 --grid-size 15 --memory-size 10000

# Train a PPO agent
viberl-train --alg ppo --episodes 1000 --grid-size 12 --ppo-epochs 4
```

### Evaluate and Demo

```bash
# Evaluate a trained model
viberl-eval --model-path experiments/reinforce_snake/final_model.pth --episodes 10 --render

# Run demo with random actions
viberl-demo --episodes 5 --grid-size 15
```

## Python API

### Basic Training

```python
from viberl.agents.reinforce import REINFORCEAgent
from viberl.envs import SnakeGameEnv
from viberl.utils.training import train_agent

# Create environment
env = SnakeGameEnv(grid_size=10)

# Create agent
agent = REINFORCEAgent(
    state_size=100,  # 10x10 grid
    action_size=4,   # 4 directions
    learning_rate=0.001
)

# Train the agent
train_agent(
    agent=agent,
    env=env,
    episodes=1000,
    save_path="models/reinforce_snake.pth"
)
```

### Custom Training Loop

```python
import numpy as np
from viberl.typing import Trajectory, Transition
from viberl.agents.dqn import DQNAgent

env = SnakeGameEnv(grid_size=10)
agent = DQNAgent(state_size=100, action_size=4)

for episode in range(1000):
    state, _ = env.reset()
    transitions = []

    while True:
        action = agent.act(state, training=True)
        next_state, reward, done, truncated, info = env.step(action.action)

        transitions.append(Transition(
            state=state, action=action, reward=reward,
            next_state=next_state, done=done
        ))

        state = next_state
        if done or truncated:
            break

    trajectory = Trajectory.from_transitions(transitions)
    metrics = agent.learn(trajectory)

    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {trajectory.total_reward}")
```

## Features

- **Modern Type System**: Pydantic-based Action, Transition, Trajectory classes
- **Three Algorithms**: REINFORCE, DQN, PPO with unified interface
- **Type Safety**: Full type annotations throughout
- **CLI Interface**: Complete training, evaluation, and demo commands
- **Experiment Management**: Automatic directory structure with TensorBoard logging
- **50+ Tests**: Comprehensive test suite

## Architecture

The framework follows a clean architecture:

- **`viberl/typing.py`**: Modern type system
- **`viberl/agents/`**: RL algorithms (REINFORCE, DQN, PPO)
- **`viberl/envs/`**: Environments (SnakeGameEnv)
- **`viberl/networks/`**: Neural network implementations
- **`viberl/utils/`**: Training utilities and experiment management
- **`viberl/cli.py`**: Command-line interface

## Algorithms

### REINFORCE
Policy gradient method using Monte Carlo returns.

### DQN
Deep Q-Network with experience replay and target networks.

### PPO
Proximal Policy Optimization with clipping and multiple epochs.

## Contributing

See the [Contributing Guide](contributing.md) for information on how to contribute to VibeRL.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/0xWelt/VibeRL/blob/main/LICENSE) file for details.
