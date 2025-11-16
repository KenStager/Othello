# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a minimal AlphaZero-style Othello (Reversi) self-play training scaffold. It implements self-play using PUCT MCTS combined with a ResNet policy/value network, trained via SGD on a replay buffer. The system includes a gating mechanism that promotes improved models to "champion" status.

## Common Commands

### Development and Training
```bash
# Setup environment (Python 3.10+ recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run self-play + training loop (main training workflow)
python scripts/self_play_train.py --config config.yaml

# Play against the AI interactively
python scripts/play_human.py --config config.yaml
```

### Configuration
- All hyperparameters are in `config.yaml`
- Key settings: MCTS simulations, batch size, learning rate, gating win rate threshold
- Set `device: cuda` in config.yaml if you have CUDA available (default is CPU)

## Architecture Overview

### Training Loop (scripts/self_play_train.py)
The main training loop orchestrates three phases per iteration:
1. **Self-play**: Generate games using current network + MCTS
2. **Training**: Sample from replay buffer and perform SGD updates
3. **Gating**: Pit current network vs champion; promote if win rate ≥ 55%

### Core Components

**src/othello/board.py**
- Pure-Python Othello rules engine
- Encoding: 4 planes (current player, opponent, valid moves, player-to-move indicator)
- Action space: 65 actions (64 board cells + 1 pass action at index 64)
- Implements 8 dihedral symmetries for data augmentation

**src/othello/features.py**
- Computes Othello-specific auxiliary features used for multi-task learning:
  - **Mobility**: legal move counts (normalized)
  - **Stability**: iteratively identifies stable discs (cannot be flipped)
  - **Corner control**: 4 corner ownership indicators
  - **Parity**: empty square parity (overall + 4 quadrants)
- `augment_state()`: generates all 8 symmetries with corresponding feature transformations

**src/net/model.py**
- `OthelloNet`: ResNet architecture with specialized heads
  - **Trunk**: Residual blocks interleaved with `LineAwareBlock` (axial convolutions for rank/file dependencies)
  - **Policy head**: 65-dim action logits
  - **Dual value head**: outputs both win probability and score differential predictions
  - **Auxiliary heads**: mobility, stability map, corner control, parity (multi-task learning signals)
- `NetworkOutput` dataclass structures all network outputs

**src/mcts/mcts.py**
- PUCT-based MCTS with Dirichlet noise at root for exploration
- `reuse_tree=True`: reuses tree across moves (currently simplified to always re-expand root)
- Value backup uses proper perspective flipping when players switch
- Returns visit count distribution as policy target

**src/train/replay.py**
- Ring buffer for experience replay (disk-backed shards)
- Stores: state planes, MCTS policy, game outcome (win/score), auxiliary features

**src/train/selfplay.py**
- `generate_selfplay()`: runs N games with temperature annealing
  - Temperature τ=1 for first `temperature_moves`, then τ→0 (deterministic)
  - Computes auxiliary features (mobility, stability, corners, parity) for each position
  - Stores augmented samples (8 symmetries) in replay buffer

**src/train/trainer.py**
- SGD training with AdamW optimizer
- Loss function combines:
  - Policy loss (cross-entropy with MCTS visit distribution)
  - Value losses (win probability + score differential)
  - Auxiliary losses (mobility, stability, corner, parity) with weighting: 0.3, 0.2, 0.2, 0.1, 0.1
- Gradient clipping enabled

**src/train/evaluator.py**
- `play_match()`: head-to-head evaluation between two networks
- Used for gating: new model must achieve ≥55% win rate vs champion to get promoted

## Key Design Decisions

### Multi-Task Learning Architecture
The network predicts multiple Othello-specific signals beyond policy/value:
- **Dual value head**: Predicts both win probability and score differential to provide richer game outcome signals
- **Auxiliary heads**: Mobility, stability, corner control, and parity features provide additional learning signals that encode domain knowledge about strong Othello positions
- These auxiliary tasks act as regularizers and guide the network to learn useful intermediate representations

### MCTS Integration
- Dirichlet noise (α=0.15, frac=0.25) added at root during self-play for exploration
- PUCT formula balances exploitation (Q) vs exploration (prior × visit-count-based bonus)
- Temperature annealing: stochastic sampling early game, deterministic late game

### Training Details
- Data augmentation via 8 dihedral symmetries applied during self-play
- Replay buffer retains 200k samples by default
- Training starts only after 5k samples collected
- Gating uses reduced MCTS simulations (50% of training sims) for efficiency

## Development Notes

### When Modifying the Model
- Network output structure is defined in `src/net/__init__.py` (NetworkOutput dataclass)
- If adding/removing heads, update:
  1. `OthelloNet.forward()` in src/net/model.py
  2. Loss computation in src/train/trainer.py
  3. Feature extraction in src/othello/features.py
  4. Replay buffer storage in src/train/selfplay.py

### When Adjusting MCTS
- MCTS parameters in config.yaml: `cpuct`, `simulations`, `reuse_tree`
- Root noise controlled by `dirichlet_alpha` and `dirichlet_frac`
- Temperature schedule: `temperature_moves` controls stochastic-to-deterministic transition

### Testing Changes
- Quick sanity check: run 1-2 iterations with reduced settings (games_per_iter=2, simulations=50)
- Monitor logs/train.tsv for training metrics
- Checkpoints saved to data/checkpoints/

## File References

- Board encoding: src/othello/board.py:111-119
- MCTS simulation loop: src/mcts/mcts.py:100-151
- Training loss computation: src/train/trainer.py:74-81
- Self-play generation: src/train/selfplay.py
- Network architecture: src/net/model.py:46-154
