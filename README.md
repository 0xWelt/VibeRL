# 🚀 VibeRL

[![Documentation](https://img.shields.io/badge/docs-0xwelt.github.io%2FVibeRL-blue)](https://0xwelt.github.io/VibeRL/)
[![CI](https://img.shields.io/github/actions/workflow/status/0xWelt/VibeRL/docs.yml?branch=main)](https://github.com/0xWelt/VibeRL/actions)
[![Tests](https://img.shields.io/github/actions/workflow/status/0xWelt/VibeRL/pytest.yml?branch=main)](https://github.com/0xWelt/VibeRL/actions/workflows/pytest.yml)
[![Coverage](https://img.shields.io/codecov/c/github/0xWelt/VibeRL)](https://codecov.io/gh/0xWelt/VibeRL)
[![Python](https://img.shields.io/badge/python-3.12+-3776ab)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-008000)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

A modern Reinforcement Learning framework for education and research, built with type safety and modern Python practices.

## 🤖 Creation Story

**VibeRL** was generated using **Kimi K2** (Moonshot AI's trillion-parameter open-source MoE model released July 2025) in combination with **Claude Code** for refinement and implementation. This project demonstrates the power of modern AI-assisted development, combining Kimi K2's advanced reasoning capabilities with Claude Code's precision in code generation and project management.

> **[Kimi K2](https://github.com/MoonshotAI/Kimi-K2)**: A 1+ trillion parameter Mixture-of-Experts model focused on long context understanding, code generation, complex reasoning, and agentic behavior. Released under Apache 2.0 license by Moonshot AI in July 2025.
>
> **[Claude Code](https://claude.ai/code)**: Anthropic's AI-powered development tool for precise code generation and project refinement.
>
> **[Kimi CC](https://github.com/LLM-Red-Team/kimi-cc)**: A shell script tool by LLM-Red-Team that enables using Kimi K2 (kimi-k2-0711-preview) to power Claude Code development workflows.

This framework was built through collaborative AI development, showcasing how next-generation language models can accelerate research and educational tool creation in reinforcement learning.

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
├── trainer/            # Unified training framework
│   ├── __init__.py
│   └── trainer.py      # Main Trainer class
└── utils/              # Training utilities
```

## 🛠️ Development
```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Quality checks
uv run pytest tests/ -n 8
uv run ruff check viberl/ --fix
uv run ruff format viberl/
```

## 🎯 Trainer Usage

### Basic Training
```python
from viberl.trainer import Trainer
from viberl.agents import DQNAgent
from viberl.envs import SnakeGameEnv

# Create environment and agent
env = SnakeGameEnv(grid_size=15)
agent = DQNAgent(
    state_size=15*15,
    action_size=4,
    learning_rate=0.001,
    memory_size=10000
)

# Create trainer
trainer = Trainer(
    env=env,
    agent=agent,
    max_steps=1000,
    log_dir="logs/experiment_1"
)

# Train the agent
rewards = trainer.train(
    num_episodes=1000,
    eval_interval=100,
    save_interval=200,
    save_path="models/"
)
```

### Custom Evaluation Environment
```python
# Use different environment for evaluation
train_env = SnakeGameEnv(grid_size=15)
eval_env = SnakeGameEnv(grid_size=15)  # Can be different configuration

trainer = Trainer(
    env=train_env,
    agent=agent,
    eval_env=eval_env,  # Optional: uses deepcopy if not provided
    max_steps=1000
)
```

### Programmatic Evaluation
```python
# Evaluate trained agent
scores, lengths = trainer.evaluate(
    num_episodes=10,
    render=True
)
print(f"Average score: {np.mean(scores):.2f}")
```

## 📈 Quick Metrics
- **Python**: 3.12+
- **Dependencies**: gymnasium, pytorch, pygame, pydantic, tensorboard
- **Test Coverage**: 50+ tests passing
- **CLI Commands**: train, eval, demo
- **Algorithms**: REINFORCE, DQN, PPO
- **Environment**: SnakeGameEnv (gymnasium-compatible)

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
