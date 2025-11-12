# othello-zero (minimal AlphaZero-style scaffold)

A compact, **from-scratch** Othello (Reversi) self-play + training loop using
a small ResNet policy/value network + PUCT MCTS. No external game engines required.

### Features
- Pure-Python Othello environment (legal moves, flips, pass, 65th action for pass).
- Small ResNet (default: 8 residual blocks × 64 channels).
- PUCT MCTS with Dirichlet noise, temperature annealing, tree reuse.
- Self-play → Replay buffer → SGD training loop (policy CE + value MSE).
- Gating matches (new vs current) with simple Elo-like report.
- Sane defaults to run on CPU/GPU (configurable via `config.yaml`).

> Goal: **hit run and start self-play immediately**. Defaults are lightweight.

### Quickstart

```bash
# (1) Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate

# (2) Install deps
pip install -r requirements.txt

# (3) Run self-play + training (CPU by default)
python scripts/self_play_train.py --config config.yaml

# (4) Optional: play a quick human-vs-AI game (CLI)
python scripts/play_human.py --config config.yaml
```

### File layout
```
src/
  othello/
    board.py        # Rules, legal moves, apply flips, encoding, symmetries
    game.py         # Game manager (self-play helpers)
  net/
    model.py        # Small ResNet policy/value network
  mcts/
    mcts.py         # PUCT MCTS
  train/
    replay.py       # Ring replay buffer (disk-backed)
    selfplay.py     # Generate games with MCTS
    trainer.py      # SGD training loop
    evaluator.py    # Head-to-head gating matches
  utils/
    config.py       # YAML config loader
    logger.py       # Simple TSV logger + pretty prints
    seed.py         # Seeding helpers
scripts/
  self_play_train.py # Orchestrates SP → Train → Gate loop
  play_human.py      # Human vs AI in terminal
data/
  checkpoints/       # Saved model checkpoints
  replay/            # Replay buffer shards
  openings/          # (optional) Opening suites
```

### Notes
- Defaults are conservative (short games/iter, small sims) to be runnable on CPU.
- Turn on GPU by setting `device: cuda` in `config.yaml` if you have a CUDA build.
- This is a **teaching scaffold**: clarity over micro-optimizations.

MIT License.
