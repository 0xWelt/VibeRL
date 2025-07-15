# 🚀 VibeRL

[![Documentation](https://img.shields.io/badge/docs-0xwelt.github.io%2FVibeRL-blue)](https://0xwelt.github.io/VibeRL/)
[![CI](https://img.shields.io/github/actions/workflow/status/0xWelt/VibeRL/docs.yml?branch=main)](https://github.com/0xWelt/VibeRL/actions)
[![Tests](https://img.shields.io/github/actions/workflow/status/0xWelt/VibeRL/pytest.yml?branch=main)](https://github.com/0xWelt/VibeRL/actions/workflows/pytest.yml)
[![Coverage](https://img.shields.io/codecov/c/github/0xWelt/VibeRL)](https://codecov.io/gh/0xWelt/VibeRL)
[![Python](https://img.shields.io/badge/python-3.12+-3776ab)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-008000)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

[🇺🇸 English](./README.md) | 🇨🇳 中文 | [🇯🇵 日本語](./README.ja.md)

一个现代化的强化学习框架，专为教育和研究设计，采用类型安全和现代 Python 实践构建。

文档: https://0xwelt.github.io/VibeRL

## 🤖 创造故事

**VibeRL** 使用 **Kimi K2**（月之暗面于2025年7月发布的万亿参数开源MoE模型）结合 **Claude Code** 进行优化和实施，展示了现代AI辅助开发的强大能力。

> **[Kimi K2](https://github.com/MoonshotAI/Kimi-K2)**: 月之暗面的万亿参数MoE模型，专注于长上下文理解、代码生成、复杂推理和智能体行为。2025年7月以Apache 2.0许可证开源发布。
>
> **[Claude Code](https://claude.ai/code)**: Anthropic的AI驱动开发工具，提供精确的代码生成和项目优化功能。

本项目展示了现代AI辅助开发的能力，结合 Kimi K2 的推理能力与 Claude Code 的精确性，加速强化学习研究和教育。

## ✨ 特性

- **3种算法**: REINFORCE、PPO、DQN 统一接口
- **类型安全**: 使用 Pydantic 模型处理动作、转换、轨迹
- **CLI工具**: 简单的 `viberl-train`、`viberl-eval`、`viberl-demo` 命令
- **现代Python**: 3.12+ 完整类型提示和 UV 支持
- **TensorBoard**: 内置训练指标和可视化

## 🎯 快速开始

```bash
# 安装
uv pip install -e ".[dev]"

# 训练智能体
viberl-train --alg=dqn --episodes 1000 --grid-size 15
viberl-train --alg=ppo --episodes 500 --lr 3e-4
viberl-train --alg=reinforce --episodes 1000 --grid-size 10

# 评估
viberl-eval --model-path experiments/*/models/final_model.pth --episodes 10

# 演示
viberl-demo --episodes 5 --grid-size 20
```

## 🤝 贡献者

感谢所有帮助改进VibeRL的贡献者！

<a href="https://github.com/0xWelt/VibeRL/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=0xWelt/VibeRL" />
</a>

## ⭐ Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=0xWelt/VibeRL&type=Date)](https://star-history.com/#0xWelt/VibeRL&Date)

## 📖 引用

如果您在研究中使用VibeRL，请引用：

```bibtex
@software{viberl2025,
  title={VibeRL: Modern Reinforcement Learning with Vibe Coding},
  author={0xWelt},
  year={2025},
  url={https://github.com/0xWelt/VibeRL},
}
```
