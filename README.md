# 🚀 VibeRL

A modern Reinforcement Learning framework for education and research, built with type safety and modern Python practices.

## 🎯 Quick Start

### Install
```bash
# Using UV (recommended)
uv pip install -e ".[dev]"

# Using pip
pip install -e ".[dev]"
```

### Run Experiments
```bash
# Train agents
viberl-train --alg=dqn --episodes 1000 --grid-size 15
viberl-train --alg=ppo --episodes 500 --lr 3e-4
viberl-train --alg=reinforce --episodes 1000 --grid-size 10

# Evaluate trained models
viberl-eval --model-path experiments/*/models/final_model.pth --episodes 10

# Play as human
viberl-demo --episodes 5 --grid-size 20
```

## 🏗️ Architecture

- **Environments**: `viberl.envs.SnakeGameEnv` - gymnasium-compatible snake game
- **Agents**: REINFORCE, PPO, DQN implementations with unified interface
- **Types**: Modern pydantic-based type system (`Action`, `Transition`, `Trajectory`)
- **CLI**: Simple commands for training, evaluation, and demo

## 🔧 Extend & Contribute

### Add New Algorithm
```python
from viberl.agents.base import Agent
from viberl.typing import Action, Trajectory

class MyAgent(Agent):
    def act(self, state, training=True) -> Action:
        return Action(action=self.policy(state))

    def learn(self, trajectory: Trajectory) -> dict[str, float]:
        # Your algorithm here
        return {"loss": loss.item()}
```

### Add New Environment
```python
from gymnasium import Env, spaces

class MyEnv(Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(...)
```

## 📊 Features

| Algorithm | Status | Key Features |
|-----------|--------|--------------|
| **REINFORCE** | ✅ | Policy gradient, simple implementation |
| **PPO** | ✅ | Clipped objective, stable training |
| **DQN** | ✅ | Experience replay, target networks |

## 📦 Project Structure
```
viberl/
├── cli.py              # Command-line interface
├── typing.py           # Modern type system (Action, Transition, Trajectory)
├── agents/             # RL algorithms
├── envs/               # Environments
└── utils/              # Training utilities
```

## 🛠️ Development
```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Quality checks
uv run pytest tests/
uv run ruff check viberl/ --fix
uv run ruff format viberl/
```

## 📈 Quick Metrics
- **Python**: 3.12+
- **Dependencies**: gymnasium, pytorch, pygame, pydantic
- **Test Coverage**: 49 tests passing
- **CLI Commands**: train, eval, demo

## 🤝 Contributors

Thanks to all the contributors who have helped make VibeRL better!

<a href="https://github.com/0xWelt/VibeRL/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=0xWelt/VibeRL" />
</a>

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=0xWelt/VibeRL&type=Date)](https://star-history.com/#0xWelt/VibeRL&Date)

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) for deep learning
- [Gymnasium](https://gymnasium.farama.org/) for RL environment interface
- [Pydantic](https://docs.pydantic.dev/) for type safety
- [UV](https://docs.astral.sh/uv/) for modern Python packaging
