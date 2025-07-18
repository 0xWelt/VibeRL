<p align="center">
  <img src="docs/VibeRL LOGO.png" alt="VibeRL Logo" width="260"/>
</p>

<p align="center">
  <a href="https://0xwelt.github.io/VibeRL/"><img src="https://img.shields.io/badge/docs-0xwelt.github.io%2FVibeRL-blue" alt="Documentation"></a>
  <a href="https://github.com/0xWelt/VibeRL/actions"><img src="https://img.shields.io/github/actions/workflow/status/0xWelt/VibeRL/docs.yml?branch=main" alt="CI"></a>
  <a href="https://github.com/0xWelt/VibeRL/actions/workflows/pytest.yml"><img src="https://img.shields.io/github/actions/workflow/status/0xWelt/VibeRL/pytest.yml?branch=main" alt="Tests"></a>
  <a href="https://codecov.io/gh/0xWelt/VibeRL"><img src="https://img.shields.io/codecov/c/github/0xWelt/VibeRL" alt="Coverage"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-3776ab" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-ee4c2c" alt="PyTorch"></a>
  <a href="https://gymnasium.farama.org/"><img src="https://img.shields.io/badge/Gymnasium-008000" alt="Gymnasium"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

<p align="center">
  <a href="./README.md">English</a> | <a href="./README.zh.md">中文</a> | <a href="./README.ja.md">日本語</a>
</p>

---

A modern Reinforcement Learning framework for education and research, built with type safety and modern Python practices.

Document: https://0xwelt.github.io/VibeRL

## 🤖 Creation Story

**VibeRL** was generated using **Kimi K2** (Moonshot AI's trillion-parameter open-source MoE model released July 2025) in combination with **Claude Code** for refinement and implementation.

> **[Kimi K2](https://github.com/MoonshotAI/Kimi-K2)**: A 1+ trillion parameter Mixture-of-Experts model focused on long context understanding, code generation, complex reasoning, and agentic behavior. Released under Apache 2.0 license by Moonshot AI in July 2025.
>
> **[Claude Code](https://claude.ai/code)**: Anthropic's AI-powered development tool for precise code generation and project refinement.

This project showcases modern AI-assisted development, combining Kimi K2's reasoning capabilities with Claude Code's precision to accelerate reinforcement learning research and education.

## ✨ Features

### 🔧 **Modern Development Stack**
- **[UV](https://docs.astral.sh/uv/)**: Lightning-fast Python package manager that replaces pip and poetry with Rust-powered dependency resolution and virtual environment management
- **[Ruff](https://docs.astral.sh/ruff/)**: Ultra-fast Python linter and formatter written in Rust that enforces PEP 8 standards and catches code issues before runtime
- **[Pytest](https://docs.pytest.org/)**: Comprehensive testing framework with 50+ unit tests covering all algorithms, environments, and utilities with fixtures and parametrization
- **[Pydantic](https://docs.pydantic.dev/)**: Runtime type validation using Python type hints that ensures data integrity across actions, transitions, and trajectories
- **[Loguru](https://loguru.readthedocs.io/)**: Elegant logging library with automatic file rotation, structured logging, and colored console output for debugging and monitoring
- **[MkDocs](https://www.mkdocs.org/)**: Static site generator that transforms Markdown into beautiful documentation with search, navigation, and responsive design
- **[TensorBoard](https://www.tensorflow.org/tensorboard)**: Real-time training visualization dashboard showing loss curves, reward trends, and hyperparameter sweeps
- **[Weights & Biases](https://wandb.ai/)**: Advanced experiment tracking platform that logs hyperparameters, metrics, artifacts, and enables collaborative ML research

### 🤖 **Reinforcement Learning Algorithms**
- **[REINFORCE](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)**: Policy gradient method with Monte Carlo returns
- **[PPO](https://arxiv.org/abs/1707.06347)**: Proximal Policy Optimization with clipping
- **[DQN](https://www.nature.com/articles/nature14236)**: Deep Q-Network with experience replay

### 🐍 **Environment**
- **SnakeGame**: Custom environment with [Gymnasium](https://gymnasium.farama.org/) integration

## 🎯 Quick Start

```bash
# Install
[uv](https://docs.astral.sh/uv/) pip install -e ".[dev]"

# Train agents
viberl-train --alg=dqn --episodes 1000 --grid-size 15
viberl-train --alg=ppo --episodes 500 --lr 3e-4
viberl-train --alg=reinforce --episodes 1000 --grid-size 10

# Train with [Weights & Biases](https://wandb.ai/) logging
viberl-train --alg=dqn --episodes 1000 --wandb --name my_experiment

# 🚀 Parallel Training with AsyncVectorEnv
viberl-train --alg=reinforce --episodes 1000 --num-envs 4 --trajectory-batch 8
viberl-train --alg=ppo --episodes 500 --lr 3e-4 --num-envs 2 --trajectory-batch 4

# Evaluate
viberl-eval --model-path experiments/*/models/final_model.pth --episodes 10

# Demo
viberl-demo --episodes 5 --grid-size 20
```


## 🤝 Contributors

Thanks to all the contributors who have helped make VibeRL better!

<a href="https://github.com/0xWelt/VibeRL/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=0xWelt/VibeRL" />
</a>

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=0xWelt/VibeRL&type=Date)](https://star-history.com/#0xWelt/VibeRL&Date)

## 📖 Citation

If you use VibeRL in your research, please cite:

```bibtex
@software{viberl2025,
  title={VibeRL: Modern Reinforcement Learning with Vibe Coding},
  author={0xWelt},
  year={2025},
  url={https://github.com/0xWelt/VibeRL},
}
```
