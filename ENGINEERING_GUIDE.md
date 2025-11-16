# Othello Learning System — Engineering Intent & Implementation Guide

## 1) Purpose & North Star

* **Goal:** Train a self-play agent that **never loses** to strong classical engines at modest time controls on 8×8 Othello (Reversi), while remaining compute-practical on a single workstation.
* **Approach:** AlphaZero-style **policy+value** network with **MCTS**, enhanced by **Othello-aware inductive biases** (line-aware ops, dihedral symmetry) and a **perfect/near-perfect endgame oracle** (Edax or our own alpha-beta).
* **Why this matters:** Othello has rich midgame tactics (mobility, stability, parity) and a near-deterministic endgame. We'll learn the opening/midgame policy, anchor the endgame with exact play, and evaluate against strong baselines.

## 2) Success Criteria (measurable)

* ≥ **55% win-rate** vs previous checkpoint in 800-game gates (balanced color & openings).
* **Loss rate** does not worsen vs prior champ at equal time controls (draws OK).
* Vs **Edax** at fixed budget (e.g., 0.5 s/move): **<2% losses** over 1,000 games with diverse openings.
* Training reproducibility: seed-locked runs vary **<15 Elo** across 3 reruns.
* Inference speed: < **30 ms** policy+value forward pass on an A-class consumer GPU for batch=1.

## 3) Non-Goals (guardrails)

* We are **not** solving Othello from first principles.
* We are **not** building a large distributed trainer; single node + light multiprocessing is enough.
* We are **not** optimizing GUI/UX; CLI first, optional web viewer later.

---

## 4) System Design (high level)

**Loop:** `Self-Play → Replay Buffer → Train → Candidate → Gate → Promote → Eval vs Engines`

**Key choices shaped by gameplay**

* **Corners/lines:** add **line-aware ops** (axial conv or attention) to complement small ResNet.
* **Symmetry:** heavy **D8 (dihedral) augmentation**; optional group-equivariant convs later.
* **Midgame currency:** auxiliary heads for **mobility, stability map, corner control, parity**.
* **Endgame:** when empties ≤ 14 (tunable), delegate rollouts/labels to **oracle** for perfect play.

---

## 5) Repository Layout

```
othello/
  README.md
  CLAUDE.md              # Current codebase documentation for Claude Code
  ENGINEERING_GUIDE.md   # This file - forward-looking design & intent
  config.yaml            # Main configuration
  requirements.txt

  src/
    othello/
      board.py           # Numpy-based board (will stay; bitboards deferred to v2)
      game.py
      features.py        # Auxiliary feature extraction
    net/
      model.py           # ResNet + LineAwareBlock + auxiliary heads
      __init__.py        # NetworkOutput dataclass
    mcts/
      mcts.py            # PUCT MCTS
      zobrist.py         # NEW: Zobrist hashing for transposition table
    train/
      replay.py          # Ring buffer with phase tagging
      selfplay.py        # Self-play generation
      trainer.py         # SGD training with phase-weighted losses
      evaluator.py       # Head-to-head evaluation
      oracle.py          # NEW: Edax bridge for endgame oracle
    utils/
      config.py, logger.py, seed.py

  scripts/
    self_play_train.py            # Main training loop
    play_human.py                 # Human vs AI
    bootstrap_il_from_edax.py     # NEW: IL data generation
    make_opening_suite.py         # NEW: Opening book generation
    eval_vs_edax.py               # NEW: External evaluation
    run_ablation.py               # NEW: Ablation experiments

  data/
    checkpoints/         # Model checkpoints with metadata
    replay/              # Replay buffer shards
    openings/            # Opening suites (JSON)
    il_bootstrap/        # IL training data

  tests/
    test_phase_tagging.py
    test_temperature_schedule.py
    test_zobrist.py
    test_oracle_bridge.py
    test_oracle_determinism.py
    test_gating_criteria.py
    test_symmetry.py
```

---

## 6) Core Data Structures & APIs

### 6.1 Board / Moves

* **Current:** Numpy arrays (`np.int8` 8×8); side-to-move flag.
* **Future (v2):** Bitboard (`uint64` x2) for 5-10× speedup (deferred until profiling shows bottleneck).
* **Legal mask:** Currently computed via directional checking; action space = 65 (64 squares + pass at index 64).
* **Pass handling:** if `legal_mask==0` and opponent has moves → "pass" action (index 64).

```python
# policy index mapping
# 0..63 -> board squares (row-major), 64 -> pass
```

### 6.2 Model IO

```python
# Inputs (float32, B x C x 8 x 8)
# C=4: [my_discs, opp_discs, legal_mask, side_to_play_plane]
# Optional: phase plane or empties count plane.

# Outputs (NetworkOutput dataclass)
policy_logits: (B, 65)
value_win:     (B,)      # tanh in [-1, 1] for win/loss
value_score:   (B,)      # normalized disc diff [-1, 1]
mobility:      (B, 2)    # predicted legal counts (me, opp)
stability_map: (B, 2, 8, 8)  # per-cell stability (black, white)
corner:        (B, 4)    # 4 corners ownership probability
parity:        (B, 5)    # overall + 4 quadrants parity
```

### 6.3 Loss (weights configurable)

```python
L = CE(policy, π_MCTS)
  + MSE(value_win, z_win)
  + α(empties) * MSE(value_score, z_score)    # α = 0.3 * (1 - empties/64)
  + 0.2 * BCE(mobility)
  + 0.2 * BCE(stability_map)
  + 0.1 * BCE(corner)
  + 0.1 * BCE(parity)
  + wd * L2
```

**Phase-weighted score loss:** Score differential matters more in endgame; weight increases as empties decrease.

---

## 7) Network Architecture

### 7.1 Small (default - current)

* **Stem:** 3×3 conv, 64 ch.
* **Backbone:** 8 residual blocks (64 ch).
* **Line-aware:** after every 2 blocks, add **LineAwareBlock** with axial convolutions (1×3 + 3×1 depthwise).
* **Heads:** policy (65), dual value (win + score), auxiliaries (mobility, stability, corner, parity).
* **Params:** ~1–2M; **fast on laptops**.

### 7.2 Strong (future workstation variant)

* 12–16 residual blocks, 96 ch.
* **Axial attention** (4 heads, 96 dim) at blocks 4/8/12.
* **Params:** ~6–9M; **800–1200 sims/move** midgame.

---

## 8) Search (PUCT MCTS)

### 8.1 Defaults (editable in `config.yaml`)

* **sims/move:** 200 (current lightweight default) → 400 (opening), 800 (midgame), 200 (endgame with oracle).
* **PUCT c:** 1.5 (range 1.25–2.5).
* **Dirichlet root noise:** α=0.15; ε=0.25 (first 12 plies).
* **Temperature:**
  - τ=1.0 (plies 1–12): stochastic exploration
  - τ=0.25 (plies 13–20): reduced randomness
  - τ→0.0 (plies 21+): deterministic best move
* **Parallelism:** Parallel actors (separate games); no shared-tree MCTS (no virtual loss needed).
* **Tree reuse:** Per-actor transposition table using Zobrist hashing.

### 8.2 Transposition Table

* **Zobrist keys:** Random 64-bit hashes for (side, 64 squares, pass).
* **Storage:** `TT[zobrist_key] → MCTSNode` per actor.
* **Reuse:** When board position repeats across moves, reuse subtree instead of re-expanding.

---

## 9) Self-Play & Curriculum

### 9.1 Generation

* N parallel actors (8–32) generate games using current net + MCTS.
* Store `(state, π_MCTS, z_win, z_score, aux_targets, phase, empties)` with **8-way dihedral augmentation**.

### 9.2 Phase Tagging

Based on empties count:
* **Opening:** empties ∈ [45, 64] (plies ≤ 19)
* **Midgame:** empties ∈ [15, 44]
* **Endgame:** empties ≤ 14 (oracle territory)

Used for:
* Balanced batch sampling (40% opening / 40% midgame / 20% endgame)
* Phase-weighted score loss
* Phase-specific metrics (entropy, calibration)

### 9.3 Bootstraps

* **Imitation Learning (IL) warm-start**: 200k positions from Edax at tiny time limit (5-10ms). Train 10–20 epochs to imitate policy/value.
* Mix IL data at **20% ratio** for first 20 RL iterations, then fade to 0%.

### 9.4 Endgame Oracle

* If empties ≤ 14 (configurable): **Edax oracle** supplies exact value (and optionally best move).
* **Benefits:**
  - Anchors value targets, reduces drift
  - Provides perfect endgame labels
  - Makes training steadier
* **Implementation:** Subprocess CLI calls to Edax binary (FFI upgrade later if bottleneck).

---

## 10) Training

### 10.1 Optim & Schedule

* **AdamW**; base LR **1e-2** with **cosine decay** to **1e-3**; weight decay **1e-4**; grad clip **1.0**.
* **Batch:** 256 (small model) / 512 (strong); **buffer:** 200k samples (FIFO).
* **Sampling balance:** 40% opening / 40% midgame / 20% endgame per batch.
* **AMP:** Enable automatic mixed precision with **FP32 policy head** to prevent softmax underflow.

### 10.2 Gating (champion promotion)

* Play **800 games** vs current champ (opening suite × color balance).
* Promote if:
  - **win-rate ≥ 55%** **AND**
  - **loss-rate_new ≤ 1.10 × loss-rate_champ** (10% slack for draws)
* This prevents draw-collapse and ensures steady improvement.

### 10.3 Checkpointing

* Save every K iters; store:
  - Model state_dict
  - Optimizer state
  - RNG seeds (numpy, torch)
  - Oracle metadata (Edax commit hash, config flags, 10-position checksum)

---

## 11) Evaluation & Reporting

### 11.1 Engines & Suites

* **Vs Edax** at fixed TC (e.g., 0.5 s/move and 2.0 s/move).
* **Opening suite:**
  - 32 hand-curated positions from opening databases
  - 32 auto-generated "hot" positions (high-temp self-play)
  - D8 symmetries expand to 256 positions for gates
  - Refresh 25% every N gates using Edax to filter for "interesting" lines

### 11.2 Metrics (tracked per phase: opening/mid/end)

* Win/Draw/Loss; **loss rate** primary KPI.
* Elo vs Edax; Elo vs previous champs.
* Policy entropy (early plies), unique openings coverage.
* **Value calibration** curves: predicted value vs realized outcome (10 bins from -1 to +1).
* Oracle hit-rate (% of endgame positions delegated).

### 11.3 Artifacts

* TensorBoard logs: loss curves, entropy by phase, Q-value histograms, calibration plots.
* JSON reports: W/L/D stats, Elo estimates, phase breakdowns.
* HTML summaries: training history, ablation comparisons.

---

## 12) Configuration (YAML)

`config.yaml` (enhanced)

```yaml
seed: 42
device: "cpu"   # "cuda" if available

paths:
  checkpoint_dir: "data/checkpoints"
  replay_dir: "data/replay"
  opening_suite: "data/openings/rot64.json"
  il_data: "data/il_bootstrap"

game:
  board_size: 8
  dirichlet_alpha: 0.15
  dirichlet_frac: 0.25

mcts:
  cpuct: 1.5
  simulations: 200        # Will increase for strong variant
  num_threads: 1
  reuse_tree: true
  tt_enabled: true        # NEW: transposition table
  zobrist: true           # NEW: zobrist hashing

selfplay:
  games_per_iter: 20
  max_moves: 120
  save_every_iters: 1
  temp_schedule:          # NEW: 3-phase temperature
    open_to: 12           # Plies 1-12: τ=1.0
    mid_to: 20            # Plies 13-20: τ=0.25
    open_tau: 1.0
    mid_tau: 0.25
    late_tau: 0.0         # Plies 21+: deterministic

model:
  channels: 64
  residual_blocks: 8
  l2_weight_decay: 1.0e-4

train:
  batch_size: 256
  steps_per_iter: 200
  lr: 1.0e-2
  lr_min: 1.0e-3
  weight_decay: 1.0e-4
  grad_clip: 1.0
  replay_capacity: 200000
  min_replay_to_train: 5000
  phase_mix: [0.4, 0.4, 0.2]     # NEW: opening/mid/end sampling
  il_mixing:                      # NEW: IL bootstrap
    enabled: true
    ratio: 0.2
    iters: 20                     # Fade out over 20 iterations
  amp_enabled: true               # NEW: automatic mixed precision

gate:
  eval_games: 800                 # Increased for statistical significance
  promote_win_rate: 0.55
  max_loss_rate_multiplier: 1.10 # NEW: loss rate check
  time_limit_ms: 50

oracle:                           # NEW: endgame oracle
  use: true
  empties_threshold: 14
  edax_path: "third_party/edax/bin/edax"
  time_limit_ms: 100

logging:                          # NEW: enhanced logging
  tensorboard: true
  wandb: false
  metrics:
    - entropy_by_phase
    - q_hist_by_phase
    - aux_losses
    - calibration
```

---

## 13) CLI Workflow

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (Optional) Build Edax oracle
# See third_party/edax/README for build instructions

# 1) Generate IL bootstrap data (200k positions from Edax)
python scripts/bootstrap_il_from_edax.py \
  --positions 200000 \
  --tc 5 \
  --out data/il_bootstrap/edax_200k.pkl

# 2) Pre-train on IL data (10-15 epochs)
python scripts/train_loop.py \
  --config config.yaml \
  --il data/il_bootstrap/edax_200k.pkl \
  --epochs 15

# 3) Start self-play + training loop (with IL mixing for first 20 iters)
python scripts/self_play_train.py --config config.yaml

# 4) Evaluate champion vs Edax
python scripts/eval_vs_edax.py \
  --config config.yaml \
  --checkpoint data/checkpoints/champion_latest.pt \
  --tc 0.5 \
  --games 1000

# 5) Run ablation study
python scripts/run_ablation.py \
  --config config.yaml \
  --ablate aux_heads \
  --games 20000
```

---

## 14) Testing & QA

* **Rules & masks:** legal move generator vs known test positions; pass sequences; no illegal expansions in MCTS.
* **Symmetry checks:** network equivariance under D8 (up to augmentation noise).
* **Oracle parity:** identical endgame results from oracle bridge vs reference engine CLI.
* **Determinism:** seed-locked replay → identical loss curves ± tolerance.
* **Phase tagging:** verify opening/mid/end classification based on empties.
* **Temperature schedule:** test 3-phase function with edge cases.
* **Zobrist hashing:** collision rate, hash consistency across board copies.
* **Gating criteria:** win rate + loss rate logic with edge cases.

---

## 15) Performance & Tuning Playbook

* If early **draw-collapse**: raise `score_weight` base to 0.4, extend τ=1.0 to 14 plies, diversify opening suite.
* If **sacrifice-phobia** (won't give discs early): keep progressive bias a bit longer; modestly raise root noise ε to 0.35 for first 8 plies.
* If **value over-confidence** in opening: switch value MSE→Huber; reduce base LR to 5e-3; add label smoothing ε=0.05 on policy.
* If **slow self-play**: lower sims to 300/600/200; enable half precision (AMP) for model; batch actor evals.
* If **oracle bottleneck**: upgrade from CLI subprocess to FFI (requires C bindings).

---

## 16) Ablation Plan

**Mini-ablations (20k games each, ~2-3 days):**
1. **Aux heads ON vs OFF**
2. **Line-aware ON vs OFF**
3. **3-phase τ vs binary τ**
4. **Phase-weighted score loss vs constant 0.3**

**Full ablations (50k games, ~1 week):**
5. **Oracle ≤14 vs none**

Each ablation: fixed seeds, report Elo deltas, loss rates, and training stability.

---

## 17) Roadmap (6–8 weeks, single-GPU cadence)

* **Week 1:**
  - Repo enhancements: phase tagging, temp schedule, gating criteria
  - TensorBoard integration
  - Zobrist hashing + transposition table

* **Week 2:**
  - Opening suite generation (rot64)
  - Edax bridge (CLI subprocess)
  - Oracle determinism tests

* **Week 3:**
  - IL bootstrap: generate 200k positions
  - Pre-train on IL data (10-15 epochs)
  - Start RL with IL mixing (20%)

* **Week 4:**
  - First robust gating with new criteria
  - Eval harness vs Edax @ 0.5s/move
  - Calibration plots, phase metrics

* **Weeks 5-6:**
  - Mini-ablations: aux heads, line-aware, temp schedule, score weighting
  - Tune hyperparameters based on results
  - Reproducibility checks (3 reruns, <15 Elo variance)

* **Weeks 7-8:**
  - Full ablation: oracle integration
  - Endurance eval (1k+ games vs Edax)
  - HTML/JSON reports, write-up

---

## 18) Deliverables

* **Trainable codebase** with configs and scripts.
* **Pretrained checkpoints** (IL-bootstrapped, self-play trained, strong variant).
* **Evaluation reports** (JSON/HTML/TensorBoard) showing KPI attainment.
* **Opening suites** (rot64 JSON, auto-generated variants).
* **Test suite** with >90% coverage on critical paths.
* **Documentation:** CLAUDE.md (current state), ENGINEERING_GUIDE.md (this file), README with quick-start.

---

## 19) Risks & Mitigations

* **Compute bottleneck:** keep model small; tune sims; AMP; async actors. **Defer bitboards until profiling confirms need.**
* **Oracle dependency drift:** pin Edax version; snapshot engine settings; 10-position determinism checksum.
* **Overfitting to engine style:** cap IL mixing to 20% post-bootstrap; diversify openings; inject aggressive starts.
* **Stochastic regressions:** gate on **loss rate**; run 2× repeats for promotions; track variance across seeds.
* **Temperature schedule complexity:** log policy entropy by phase; verify exploration doesn't collapse too early.

---

## 20) "Definition of Done"

* Passes all tests (CI green).
* Deterministic seeds verified (3 reruns, <15 Elo variance).
* Achieves target **loss rate ≤ 2%** vs Edax @ 0.5s/move (1k games).
* Gating criteria stable (2 consecutive promotions with win-rate ≥ 55%, loss-rate ≤ 1.10×).
* Full report + exported checkpoints available.
* Ablation results documented with statistical significance.

---

## 21) Key Decisions Locked

1. **Bitboards:** Deferred to v2 (after profiling).
2. **MCTS:** Parallel actors (no shared-tree, no virtual loss). Transposition table per actor.
3. **Phase tagging:** By empties count (opening 45-64, mid 15-44, end ≤14).
4. **IL warm-start:** Both policy and value, 200k positions, 20% mixing for 20 iters.
5. **Opening suite:** Hybrid (32 human + 32 auto), color-balanced, D8 symmetries.
6. **Gating:** Win-rate ≥ 55% AND loss-rate ≤ 1.10× champ.
7. **Score loss:** Phase-weighted (0.3 × (1 - empties/64)).
8. **Temperature:** 3-phase (1.0 / 0.25 / 0.0 at plies 12/20/21+).
9. **AMP:** Enabled with FP32 policy head.
10. **Oracle:** CLI subprocess to Edax (upgrade to FFI if bottleneck), threshold ≤14 empties.
11. **Logging:** TensorBoard (W&B optional).
12. **Ablations:** Mini (20k) first, full (50k) for top 2 effects.

---

## Appendix: Code Snippets

### A) Phase tagging in self-play

```python
# src/train/selfplay.py
empties = 64 - np.count_nonzero(board.board)
if empties >= 45:
    phase = "opening"
elif empties <= 14:
    phase = "endgame"
else:
    phase = "midgame"

sample = {
    "state": board.encode(),
    "policy": pi,
    "value_win": z_win,
    "value_score": z_score,
    "phase": phase,
    "empties": empties,
    # ... aux features
}
```

### B) Phase-weighted score loss

```python
# src/train/trainer.py
empties = batch["empties"].float()
score_weight = 0.3 * (1.0 - empties / 64.0)
score_loss = F.mse_loss(outputs.value_score, targets.value_score, reduction='none')
score_loss = (score_loss.squeeze() * score_weight).mean()
```

### C) 3-phase temperature

```python
# src/train/selfplay.py
def move_temperature(ply, cfg):
    if ply <= cfg['temp_schedule']['open_to']:
        return cfg['temp_schedule']['open_tau']
    elif ply <= cfg['temp_schedule']['mid_to']:
        return cfg['temp_schedule']['mid_tau']
    else:
        return cfg['temp_schedule']['late_tau']
```

### D) Gating with loss rate

```python
# scripts/self_play_train.py
new_winrate = wins / max(1, wins + losses)
new_loss_rate = losses / max(1, wins + losses + draws)
champ_loss_rate = historical_loss_rate  # from prior gate

if (new_winrate >= cfg['gate']['promote_win_rate'] and
    new_loss_rate <= cfg['gate']['max_loss_rate_multiplier'] * champ_loss_rate):
    promote_to_champion(net)
```

### E) Zobrist hashing

```python
# src/mcts/zobrist.py
import numpy as np

rng = np.random.RandomState(12345)
Z_SIDE = rng.randint(0, 2**64, dtype=np.uint64)
Z_BLACK = rng.randint(0, 2**64, size=64, dtype=np.uint64)
Z_WHITE = rng.randint(0, 2**64, size=64, dtype=np.uint64)
Z_PASS = rng.randint(0, 2**64, dtype=np.uint64)

def zobrist_hash(board):
    h = np.uint64(0)
    if board.player == 1:  # BLACK
        h ^= Z_SIDE
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            if board.board[r, c] == 1:    # BLACK
                h ^= Z_BLACK[idx]
            elif board.board[r, c] == -1: # WHITE
                h ^= Z_WHITE[idx]
    # Handle pass state if needed
    return h
```

---

## Contact & Contributions

For questions, issues, or improvements, see the main README.md.

MIT License.
