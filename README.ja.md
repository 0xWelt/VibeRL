# 🚀 VibeRL

[![Documentation](https://img.shields.io/badge/docs-0xwelt.github.io%2FVibeRL-blue)](https://0xwelt.github.io/VibeRL/)
[![CI](https://img.shields.io/github/actions/workflow/status/0xWelt/VibeRL/docs.yml?branch=main)](https://github.com/0xWelt/VibeRL/actions)
[![Tests](https://img.shields.io/github/actions/workflow/status/0xWelt/VibeRL/pytest.yml?branch=main)](https://github.com/0xWelt/VibeRL/actions/workflows/pytest.yml)
[![Coverage](https://img.shields.io/codecov/c/github/0xWelt/VibeRL)](https://codecov.io/gh/0xWelt/VibeRL)
[![Python](https://img.shields.io/badge/python-3.12+-3776ab)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-008000)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

[🇺🇸 English](./README.md) | [🇨🇳 中文](./README.zh.md) | 🇯🇵 日本語

教育と研究のためのモダンな強化学習フレームワークで、型安全性と最新のPython実践を採用しています。

ドキュメント: https://0xwelt.github.io/VibeRL

## 🤖 創作ストーリー

**VibeRL**は、**Kimi K2**（2025年7月にMoonshot AIが公開した1兆パラメータのオープンソースMoEモデル）と**Claude Code**を組み合わせて生成され、最新のAI支援開発の威力を示しています。

> **[Kimi K2](https://github.com/MoonshotAI/Kimi-K2)**: Moonshot AIの1兆パラメータMoEモデル。長文脈理解、コード生成、複雑な推論、エージェント動作に特化。2025年7月にApache 2.0ライセンスで公開。
>
> **[Claude Code](https://claude.ai/code)**: AnthropicのAI駆動開発ツール。正確なコード生成とプロジェクト最適化を提供。

このプロジェクトは、Kimi K2の推論能力とClaude Codeの精度を組み合わせて、強化学習研究と教育を加速する最新のAI支援開発を示しています。

## ✨ 特徴

- **3種類のアルゴリズム**: REINFORCE、PPO、DQNを統一インターフェースで
- **型安全性**: アクション、遷移、軌跡のためのPydanticモデル
- **CLIツール**: シンプルな `viberl-train`、`viberl-eval`、`viberl-demo` コマンド
- **モダンPython**: 3.12+ 完全な型ヒントとUVサポート
- **TensorBoard**: 組み込みのトレーニング指標と可視化

## 🎯 クイックスタート

```bash
# インストール
uv pip install -e ".[dev]"

# エージェントのトレーニング
viberl-train --alg=dqn --episodes 1000 --grid-size 15
viberl-train --alg=ppo --episodes 500 --lr 3e-4
viberl-train --alg=reinforce --episodes 1000 --grid-size 10

# 評価
viberl-eval --model-path experiments/*/models/final_model.pth --episodes 10

# デモ
viberl-demo --episodes 5 --grid-size 20
```

## 🤝 貢献者

VibeRLをより良くするために協力してくれたすべての貢献者に感謝します！

<a href="https://github.com/0xWelt/VibeRL/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=0xWelt/VibeRL" />
</a>

## ⭐ Star 履歴

[![Star History Chart](https://api.star-history.com/svg?repos=0xWelt/VibeRL&type=Date)](https://star-history.com/#0xWelt/VibeRL&Date)

## 📖 引用

研究でVibeRLを使用する場合は、以下のように引用してください：

```bibtex
@software{viberl2025,
  title={VibeRL: Modern Reinforcement Learning with Vibe Coding},
  author={0xWelt},
  year={2025},
  url={https://github.com/0xWelt/VibeRL},
}
```
