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

Document: https://0xwelt.github.io/VibeRL

## 🤖 Creation Story

**VibeRL** was generated using **Kimi K2** (Moonshot AI's trillion-parameter open-source MoE model released July 2025) in combination with **Claude Code** for refinement and implementation.

> **[Kimi K2](https://github.com/MoonshotAI/Kimi-K2)**: A 1+ trillion parameter Mixture-of-Experts model focused on long context understanding, code generation, complex reasoning, and agentic behavior. Released under Apache 2.0 license by Moonshot AI in July 2025.
>
> **[Claude Code](https://claude.ai/code)**: Anthropic's AI-powered development tool for precise code generation and project refinement.

This project showcases modern AI-assisted development, combining Kimi K2's reasoning capabilities with Claude Code's precision to accelerate reinforcement learning research and education.

## ✨ Features

- **3 Algorithms**: REINFORCE, PPO, DQN with unified interface
- **Type-Safe**: Pydantic models for actions, transitions, trajectories
- **CLI Tools**: Simple `viberl-train`, `viberl-eval`, `viberl-demo` commands
- **Modern Python**: 3.12+ with full type hints and UV support
- **TensorBoard**: Built-in training metrics and visualization

## 🎯 Quick Start

```bash
# Install
uv pip install -e ".[dev]"

# Train agents
viberl-train --alg=dqn --episodes 1000 --grid-size 15
viberl-train --alg=ppo --episodes 500 --lr 3e-4
viberl-train --alg=reinforce --episodes 1000 --grid-size 10

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
