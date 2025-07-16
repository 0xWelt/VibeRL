<p align="center">
  <img src="docs/VibeRL LOGO.png" alt="VibeRL Logo" width="260"/>
</p>

<p align="center">
  <a href="https://0xwelt.github.io/VibeRL/">
    <img src="https://img.shields.io/badge/docs-0xwelt.github.io%2FVibeRL-blue" alt="Documentation"/>
  </a>
  <a href="https://github.com/0xWelt/VibeRL/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/0xWelt/VibeRL/docs.yml?branch=main" alt="CI"/>
  </a>
  <a href="https://github.com/0xWelt/VibeRL/actions/workflows/pytest.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/0xWelt/VibeRL/pytest.yml?branch=main" alt="Tests"/>
  </a>
  <a href="https://codecov.io/gh/0xWelt/VibeRL">
    <img src="https://img.shields.io/codecov/c/github/0xWelt/VibeRL" alt="Coverage"/>
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.12+-3776ab" alt="Python"/>
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-ee4c2c" alt="PyTorch"/>
  </a>
  <a href="https://gymnasium.farama.org/">
    <img src="https://img.shields.io/badge/Gymnasium-008000" alt="Gymnasium"/>
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
  </a>
</p>

<p align="center">
  <a href="./README.md">🇺🇸 English</a> | 🇨🇳 中文 | <a href="./README.ja.md">🇯🇵 日本語</a>
</p>

---

一个现代化的强化学习框架，专为教育和研究设计，采用类型安全和现代 Python 实践构建。

文档: https://0xwelt.github.io/VibeRL

## 🤖 创造故事

**VibeRL** 使用 **Kimi K2**（月之暗面于2025年7月发布的万亿参数开源MoE模型）结合 **Claude Code** 进行优化和实施，展示了现代AI辅助开发的强大能力。

> **[Kimi K2](https://github.com/MoonshotAI/Kimi-K2)**: 月之暗面的万亿参数MoE模型，专注于长上下文理解、代码生成、复杂推理和智能体行为。2025年7月以Apache 2.0许可证开源发布。
>
> **[Claude Code](https://claude.ai/code)**: Anthropic的AI驱动开发工具，提供精确的代码生成和项目优化功能。

本项目展示了现代AI辅助开发的能力，结合 Kimi K2 的推理能力与 Claude Code 的精确性，加速强化学习研究和教育。

## ✨ 特性

### 🔧 **现代化开发栈**
- **[UV](https://docs.astral.sh/uv/)**: 基于Rust的极速Python包管理器，完全替代pip和poetry，提供毫秒级依赖解析和虚拟环境管理
- **[Ruff](https://docs.astral.sh/ruff/)**: 用Rust编写的超快Python代码检查器，严格执行PEP 8标准并在运行时前捕获代码问题
- **[Pytest](https://docs.pytest.org/)**: 包含50+单元测试的综合测试框架，覆盖所有算法、环境和工具，支持固件和参数化测试
- **[Pydantic](https://docs.pydantic.dev/)**: 使用Python类型提示的运行时类型验证，确保动作、转换、轨迹的数据完整性
- **[Loguru](https://loguru.readthedocs.io/)**: 优雅的日志库，支持自动文件轮转、结构化日志记录和彩色控制台输出，便于调试和监控
- **[MkDocs](https://www.mkdocs.org/)**: 将Markdown转换为精美文档的静态站点生成器，内置搜索、导航和响应式设计
- **[TensorBoard](https://www.tensorflow.org/tensorboard)**: 实时训练可视化仪表板，展示损失曲线、奖励趋势和超参数扫描
- **[Weights & Biases](https://wandb.ai/)**: 高级实验跟踪平台，记录超参数、指标、工件，支持协作式机器学习研究

### 🤖 **强化学习算法**
- **[REINFORCE](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)**: 蒙特卡洛收益的策略梯度方法
- **[PPO](https://arxiv.org/abs/1707.06347)**: 带裁剪的近端策略优化
- **[DQN](https://www.nature.com/articles/nature14236)**: 经验回放的深度Q网络

### 🐍 **环境**
- **SnakeGame**: 自定义环境，集成 [Gymnasium](https://gymnasium.farama.org/)

## 🎯 快速开始

```bash
# 安装
[uv](https://docs.astral.sh/uv/) pip install -e ".[dev]"

# 训练智能体
viberl-train --alg=dqn --episodes 1000 --grid-size 15
viberl-train --alg=ppo --episodes 500 --lr 3e-4
viberl-train --alg=reinforce --episodes 1000 --grid-size 10

# 使用 [Weights & Biases](https://wandb.ai/) 进行训练
viberl-train --alg=dqn --episodes 1000 --wandb --name my_experiment

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
