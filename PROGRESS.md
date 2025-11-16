# Implementation Progress

## Summary

This document tracks the implementation of enhancements to the Othello AlphaZero learning system based on the ENGINEERING_GUIDE.md design.

**Date:** 2025-11-12
**Status:** Phase 1-2 Complete (Core Infrastructure + Search Enhancements)

---

## ✅ Completed (Days 1-3 equivalent)

### Documentation
- [x] Created ENGINEERING_GUIDE.md with comprehensive design and rationale
- [x] Updated README.md to reference both CLAUDE.md and ENGINEERING_GUIDE.md
- [x] Created PROGRESS.md (this file)

### Phase 1: Core Infrastructure
- [x] **Phase tagging**: Added `phase` and `empties` fields to replay samples
  - Modified `src/train/selfplay.py` to tag samples as opening/midgame/endgame based on empties count
  - Updated `src/train/replay.py` to support phase-balanced sampling with `phase_mix` parameter
- [x] **Phase-weighted score loss**: Implemented in `src/train/trainer.py`
  - Score loss weight increases as game progresses: `0.3 * (1 - empties/64)`
  - Emphasizes score prediction more in endgame where it matters most
- [x] **3-phase temperature schedule**: Implemented in `src/train/selfplay.py`
  - Plies 1-12: τ=1.0 (stochastic exploration)
  - Plies 13-20: τ=0.25 (reduced randomness)
  - Plies 21+: τ=0.0 (deterministic best move)
  - Added `move_temperature()` and `apply_temperature()` helper functions
- [x] **Gating criteria with loss rate**: Updated `scripts/self_play_train.py`
  - Promotion requires: `win_rate >= 55%` AND `loss_rate <= 1.10 * champ_loss_rate`
  - Prevents draw-collapse by ensuring losses don't increase
- [x] **Updated config.yaml**: Added all new parameters
  - `selfplay.temp_schedule` with 3-phase settings
  - `train.phase_mix` for balanced sampling
  - `train.phase_weighted_score` flag
  - `gate.max_loss_rate_multiplier`
  - `oracle.*` settings (placeholder)
  - `logging.*` settings (placeholder)

### Phase 2: Search Enhancements
- [x] **Zobrist hashing**: Created `src/mcts/zobrist.py`
  - Fixed-seed random hash generation for reproducibility
  - Hash includes: side-to-move, piece positions, optional pass state
  - Includes test function for consistency verification
- [x] **Transposition table**: Enhanced `src/mcts/mcts.py`
  - Per-actor TT dictionary: `zobrist_hash → MCTSNode`
  - Reuses subtrees across moves for same positions
  - Tracks TT hit/miss statistics
  - Added `clear_tt()` and `get_tt_stats()` methods
- [x] **Opening suite generation**: Created `scripts/make_opening_suite.py`
  - Generates random opening positions with center/corner bias
  - Applies D8 symmetries for coverage
  - Exports to JSON format for evaluation
  - Supports loading human opening databases (placeholder)

### Phase 3: Oracle Integration (Partial)
- [x] **Oracle bridge**: Created `src/train/oracle.py`
  - Subprocess interface to Edax for exact endgame solving
  - `EdaxOracle` class with caching and statistics
  - `DummyOracle` fallback for testing without Edax
  - Factory function `create_oracle(cfg)` for easy instantiation
  - **Note**: Edax output parsing is placeholder - needs adjustment based on actual Edax CLI

---

## 🚧 In Progress / TODO

### Phase 3: Oracle Integration (Remaining)
- [ ] **Edax setup**: Download/build Edax and document setup process
  - Add Edax as git submodule OR document build instructions
  - Test oracle bridge with actual Edax binary
  - Update `_parse_edax_score()` and `_parse_edax_best_move()` with correct parsing logic
- [ ] **Integrate oracle into self-play**: Modify selfplay to use oracle for endgame positions
  - Call oracle when `empties <= threshold`
  - Use oracle value for leaf evaluation in MCTS
  - Store oracle-labeled samples with metadata
- [ ] **Checkpoint metadata**: Extend checkpoint saving to include oracle info
  - Edax commit hash, config flags, 10-position checksum
  - `tests/test_oracle_determinism.py`

### Phase 4: IL Bootstrap
- [ ] **IL data generation**: Create `scripts/bootstrap_il_from_edax.py`
  - Generate 200k positions from Edax at low time control (5-10ms)
  - Store as pickle with (state, edax_policy, edax_value, aux_features)
- [ ] **IL training mode**: Add supervised pre-training to training loop
  - Train for 10-15 epochs on IL data before RL
  - Save IL-bootstrapped checkpoint
- [ ] **IL mixing in RL**: Implement 20% IL data mixing for first 20 iterations
  - Modify replay sampling to include IL samples with fade-out
  - Track IL vs RL sample ratios in logs

### Phase 5: Evaluation & Reporting
- [ ] **TensorBoard integration**: Add TensorBoard logging to training loop
  - Policy entropy by phase
  - Q-value histograms by phase
  - Auxiliary head losses
  - Calibration curves (predicted vs realized value)
- [ ] **Edax evaluation harness**: Create `scripts/eval_vs_edax.py`
  - Pit checkpoint vs Edax at fixed time controls (0.5s, 2.0s)
  - Use opening suite for balanced starts
  - Report W/L/D, Elo, loss rate by phase
- [ ] **Calibration logging**: Implement value calibration tracking
  - Bucket predicted values into 10 bins
  - Track realized outcomes per bin per phase
  - Visualize as calibration curves in TensorBoard
- [ ] **Ablation framework**: Create `scripts/run_ablation.py`
  - Support config overrides for ablation experiments
  - Run mini-ablations (20k games each)
  - Generate comparison reports
- [ ] **HTML reporting**: Generate comprehensive HTML reports
  - Training curves, gating history, Edax benchmarks
  - Ablation comparisons with statistical significance

### Testing
- [ ] **Test suite**: Create comprehensive tests
  - `tests/test_phase_tagging.py` - Verify opening/mid/end classification
  - `tests/test_temperature_schedule.py` - Check 3-phase function
  - `tests/test_zobrist.py` - Hash consistency and collision rate
  - `tests/test_oracle_bridge.py` - Edax subprocess I/O
  - `tests/test_oracle_determinism.py` - Fixed positions → fixed results
  - `tests/test_gating_criteria.py` - Win rate + loss rate logic
  - Update `tests/test_symmetry.py` if needed

---

## 🎯 Next Steps (Priority Order)

1. **Test current implementation** (HIGHEST PRIORITY)
   - Run a smoke test: 2-3 iterations with current enhancements
   - Verify phase tagging, temperature schedule, gating work correctly
   - Check TT statistics (hit rate should be >10% after a few games)
   - Monitor training loss and ensure phase-weighted score loss doesn't explode

2. **Oracle setup** (HIGH PRIORITY - unblocks endgame improvements)
   - Download and build Edax
   - Test oracle bridge with a few endgame positions
   - Fix parsing logic in `oracle.py`
   - Document setup process in README or separate EDAX_SETUP.md

3. **IL bootstrap** (MEDIUM PRIORITY - 2-3 week speedup)
   - Implement `bootstrap_il_from_edax.py`
   - Generate 200k positions (takes ~1-2 hours)
   - Pre-train for 10-15 epochs
   - Start RL with IL mixing

4. **TensorBoard logging** (MEDIUM PRIORITY - essential for debugging)
   - Add TensorBoard writer to training loop
   - Log phase-specific metrics
   - Monitor calibration curves

5. **Evaluation harness** (MEDIUM PRIORITY - validates improvements)
   - Implement `eval_vs_edax.py`
   - Run benchmark vs Edax at 0.5s/move
   - Establish baseline before further enhancements

6. **Ablation studies** (LOW PRIORITY - for publication/validation)
   - Run mini-ablations once baseline is solid
   - Focus on aux-heads and line-aware blocks first

---

## 📊 Metrics to Track

### Training Metrics
- [x] Replay buffer size
- [x] Training loss (policy + value + aux)
- [x] Gate win/loss/draw rates
- [x] Gate loss rate (NEW)
- [ ] Policy entropy by phase
- [ ] Q-value distribution by phase
- [ ] Auxiliary head losses separately
- [ ] Value calibration error

### Performance Metrics
- [ ] Self-play games/hour
- [ ] MCTS simulations/second
- [ ] TT hit rate (NEW)
- [ ] Oracle call rate (when empties <= 14)

### Evaluation Metrics
- [ ] Win rate vs Edax @ 0.5s/move
- [ ] Loss rate vs Edax @ 0.5s/move (PRIMARY KPI)
- [ ] Elo estimate
- [ ] Loss rate by phase (opening/mid/end)

---

## 🔧 Configuration Changes

### Before (Minimal Defaults)
```yaml
game:
  temperature_moves: 12

train:
  batch_size: 256
  replay_capacity: 200000

gate:
  promote_win_rate: 0.55
```

### After (Enhanced)
```yaml
selfplay:
  temp_schedule:
    open_to: 12
    mid_to: 20
    open_tau: 1.0
    mid_tau: 0.25
    late_tau: 0.0

train:
  phase_mix: [0.4, 0.4, 0.2]
  phase_weighted_score: true
  score_weight_base: 0.3

gate:
  promote_win_rate: 0.55
  max_loss_rate_multiplier: 1.10

mcts:
  tt_enabled: true
  zobrist: true

oracle:
  use: false  # Set true after Edax setup
  empties_threshold: 14
```

---

## 📝 Notes & Observations

### Design Decisions Made
1. **Phase boundaries**: Opening (45-64 empties), Midgame (15-44), Endgame (≤14)
   - Simple, robust, aligns with oracle threshold
2. **Score loss weighting**: Linear ramp from 0 at game start to 0.3 at endgame
   - Could later switch to step function (only apply score loss in endgame)
3. **Temperature schedule**: 3-phase with intermediate τ=0.25
   - Rationale: Othello midgame has many near-equivalent moves, softening collapse helps
4. **Gating loss rate**: 10% slack (1.10×)
   - Allows some variance but prevents significant regression
5. **TT per-actor**: Simple, no virtual loss needed
   - Future: could share TT across actors with locking

### Potential Issues to Watch
- **Phase-weighted score loss explosion**: If empties calculation is wrong, could blow up training
  - Mitigation: Added safeguards in trainer, defaults to empties=32 if missing
- **TT memory usage**: Could grow large over long games
  - Mitigation: Clear TT between games with `mcts.clear_tt()`
- **Oracle parsing fragility**: Placeholder parsing may break with actual Edax output
  - Mitigation: Fallback to dummy oracle, extensive testing needed
- **Temperature schedule complexity**: More hyperparams to tune
  - Mitigation: Defaults match AlphaZero-style proven schedules

### Future Optimizations (v2)
- Bitboard implementation (5-10× speedup in board ops)
- Shared-tree MCTS with virtual loss (if multi-threading within tree)
- FFI to Edax instead of subprocess (faster oracle calls)
- AMP (automatic mixed precision) for GPU training
- Group-equivariant convolutions for symmetry handling

---

## 🏆 Success Criteria Checklist

- [ ] ≥55% win-rate vs previous checkpoint in 800-game gates
- [ ] Loss rate ≤ prior champion at equal time controls
- [ ] Vs Edax @ 0.5s/move: <2% losses over 1,000 games
- [ ] Seed-locked runs vary <15 Elo across 3 reruns
- [ ] Inference speed: <30ms forward pass (batch=1, GPU)
- [ ] All tests passing (pytest)
- [ ] Documentation complete (CLAUDE.md, ENGINEERING_GUIDE.md, README)

---

## 📚 References

- **ENGINEERING_GUIDE.md**: Full design document with rationale
- **CLAUDE.md**: Current codebase documentation for Claude Code
- **README.md**: Quick-start and overview
- **config.yaml**: All hyperparameters with comments

---

**Last Updated**: 2025-11-12
**Next Review**: After smoke test of current implementation
