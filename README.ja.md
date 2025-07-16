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
  <a href="./README.md">English</a> | <a href="./README.zh.md">中文</a> | 日本語
</p>

---

教育と研究のためのモダンな強化学習フレームワークで、型安全性と最新のPython実践を採用しています。

ドキュメント: https://0xwelt.github.io/VibeRL

## 🤖 創作ストーリー

**VibeRL**は、**Kimi K2**（2025年7月にMoonshot AIが公開した1兆パラメータのオープンソースMoEモデル）と**Claude Code**を組み合わせて生成され、最新のAI支援開発の威力を示しています。

> **[Kimi K2](https://github.com/MoonshotAI/Kimi-K2)**: Moonshot AIの1兆パラメータMoEモデル。長文脈理解、コード生成、複雑な推論、エージェント動作に特化。2025年7月にApache 2.0ライセンスで公開。
>
> **[Claude Code](https://claude.ai/code)**: AnthropicのAI駆動開発ツール。正確なコード生成とプロジェクト最適化を提供。

このプロジェクトは、Kimi K2の推論能力とClaude Codeの精度を組み合わせて、強化学習研究と教育を加速する最新のAI支援開発を示しています。

## ✨ 特徴

### 🔧 **モダン開発スタック**
- **[UV](https://docs.astral.sh/uv/)**: Rust製の高速Pythonパッケージマネージャーで、pipとpoetryを完全に置き換え、ミリ秒単位の依存関係解決と仮想環境管理を提供
- **[Ruff](https://docs.astral.sh/ruff/)**: Rustで書かれた超高速Pythonリンターで、PEP 8標準を厳格に実行し、実行時の前にコードの問題を検出
- **[Pytest](https://docs.pytest.org/)**: 全アルゴリズム、環境、ユーティリティをカバーする50以上のユニットテストを含む包括的テストフレームワーク
- **[Pydantic](https://docs.pydantic.dev/)**: Python型ヒントを使用したランタイム型検証で、アクション、遷移、軌跡全体のデータ整合性を確保
- **[Loguru](https://loguru.readthedocs.io/)**: 自動ファイルローテーション、構造化ロギング、カラー化コンソール出力を備えたエレガントなログライブラリ
- **[MkDocs](https://www.mkdocs.org/)**: Markdownを美しいドキュメントに変換する静的サイトジェネレーターで、検索、ナビゲーション、レスポンシブデザインを内蔵
- **[TensorBoard](https://www.tensorflow.org/tensorboard)**: 損失曲線、報酬トレンド、ハイパーパラメータスイープを表示するリアルタイムトレーニング可視化ダッシュボード
- **[Weights & Biases](https://wandb.ai/)**: ハイパーパラメータ、メトリクス、アーティファクトを記録し、協働的ML研究を可能にする高度な実験追跡プラットフォーム

### 🤖 **強化学習アルゴリズム**
- **[REINFORCE](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)**: モンテカルロ収益の政策勾配法
- **[PPO](https://arxiv.org/abs/1707.06347)**: クリッピング付き近接政策最適化
- **[DQN](https://www.nature.com/articles/nature14236)**: 経験再生の深層Qネットワーク

### 🐍 **環境**
- **SnakeGame**: カスタム環境、[Gymnasium](https://gymnasium.farama.org/)統合

## 🎯 クイックスタート

```bash
# インストール
[uv](https://docs.astral.sh/uv/) pip install -e ".[dev]"

# エージェントのトレーニング
viberl-train --alg=dqn --episodes 1000 --grid-size 15
viberl-train --alg=ppo --episodes 500 --lr 3e-4
viberl-train --alg=reinforce --episodes 1000 --grid-size 10

# [Weights & Biases](https://wandb.ai/)でトレーニング
viberl-train --alg=dqn --episodes 1000 --wandb --name my_experiment

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
