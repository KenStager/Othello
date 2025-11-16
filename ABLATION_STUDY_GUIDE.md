# Ablation Study Guide: High Score Margins Investigation

## Purpose

Scientifically determine which factors contribute to high score margins (±27-28 discs) observed at iteration 45.

## Background

**Observation**: Games ending with excessive score margins (±27-28 discs)
- Expected: ±8-15 discs (healthy competitive games)
- Current: ±27-28 discs (blowouts, snowballing)

**Hypotheses**:
1. **Temperature schedule too aggressive**: Deterministic play (τ=0.0) starting at move 21 (33% into game) may cause snowballing
2. **MCTS too weak**: 150 simulations insufficient to find defensive resources, crude evaluations miss tactical nuances

**Goal**: Isolate the contribution of each factor via controlled experiments

---

## Experimental Design

### 4 Configurations (Progressive MCTS Scaling)

**Research Insight**: Literature recommends 200 simulations as standard baseline (not 150). Progressive scaling (150→200→300) is more scientifically sound than jumping directly to 400.

| Config | MCTS Sims | Temperature Schedule | Expected Outcome |
|--------|-----------|---------------------|------------------|
| **A** (Baseline) | 150 | τ=0.0 at move 21 | ±27-28 discs (control) |
| **B** (Standard MCTS) | 200 | τ=0.0 at move 21 | ±20-23 discs (15-25% improvement) |
| **C** (Temp Fix) | 200 | τ=0.05 extended to move 40 | ±15-18 discs (30-40% improvement) |
| **D** (Progressive MCTS + Temp) | 300 | τ=0.05 extended to move 40 | ±12-15 discs (40-55% improvement) |

### Metrics

**Primary**: Average score margin over 500 games (100 games × 5 iterations)

**Secondary**:
- Win rate distribution (how lopsided are results?)
- Game length (blowouts end faster)
- Q-value calibration (MAE, correlation)
- Policy entropy by phase

**Training-to-Evaluation Ratio**: 100:40 = 2.5:1 (research-aligned, matches alpha-zero-general)

---

## Running the Ablation Study

### Setup (One Time)

All 4 configuration files are already created:
- `config_ablation_a.yaml` (baseline)
- `config_ablation_b.yaml` (research-standard MCTS)
- `config_ablation_c.yaml` (temperature fix at standard MCTS)
- `config_ablation_d.yaml` (progressive MCTS + temperature fix)

Each config uses isolated directories (`data/ablation_a/`, `data/ablation_b/`, etc.) to prevent contamination.

### Running Each Configuration

Run each config for **5 iterations** (100 games per iteration = 500 games total):

```bash
# Config A: Baseline (150 sims, aggressive temp)
PYTHONPATH=. python scripts/self_play_train.py --config config_ablation_a.yaml
# Stop after iteration 5 (Ctrl+C)

# Config B: Research-standard MCTS (200 sims, aggressive temp)
PYTHONPATH=. python scripts/self_play_train.py --config config_ablation_b.yaml
# Stop after iteration 5 (Ctrl+C)

# Config C: Temperature fix at standard MCTS (200 sims, gentle temp)
PYTHONPATH=. python scripts/self_play_train.py --config config_ablation_c.yaml
# Stop after iteration 5 (Ctrl+C)

# Config D: Progressive MCTS + temp fix (300 sims, gentle temp)
PYTHONPATH=. python scripts/self_play_train.py --config config_ablation_d.yaml
# Stop after iteration 5 (Ctrl+C)
```

### Time Estimates (With 100 Training Games + 40 Eval Games)

| Config | Sims | Self-Play (100g) | Gating (40g) | Per Iteration | 5 Iterations |
|--------|------|------------------|--------------|---------------|--------------|
| A | 150 | ~7.6 min | ~3.1 min | ~10.7 min | ~54 min |
| B | 200 | ~10.1 min | ~3.1 min | ~13.2 min | ~66 min |
| C | 200 | ~10.1 min | ~3.1 min | ~13.2 min | ~66 min |
| D | 300 | ~15.2 min | ~3.1 min | ~18.3 min | ~92 min |

**Total**: ~5.5 hours (can run sequentially or in parallel)

**Training-to-Evaluation Ratio**: 100:40 = 2.5:1 (research-standard)
- **Before**: 50:60 = 0.83:1 (inverted - more time on evaluation than training!)
- **After**: 100:40 = 2.5:1 (aligned with alpha-zero-general for Othello)
- **Benefit**: 2x training data generation per iteration (100 games × 8 augmentations = 800 positions)

---

## Analyzing Results

After all 4 configs complete 5 iterations, analyze the results:

```bash
python scripts/analyze_ablation_results.py
```

### Sample Output

```
================================================================================
ABLATION STUDY RESULTS ANALYSIS
================================================================================

Config A: BASELINE (sims=150, tau=0.0)
  Iterations analyzed: 5
  Mean score margin: ±27.3 discs
  Median score margin: ±27.1 discs

Config B: RESEARCH-STANDARD MCTS (sims=200, tau=0.0)
  Iterations analyzed: 5
  Mean score margin: ±21.5 discs  (-21.2% vs baseline)
  Median score margin: ±21.2 discs

Config C: TEMPERATURE FIX AT STANDARD MCTS (sims=200, tau=0.05 extended)
  Iterations analyzed: 5
  Mean score margin: ±16.8 discs  (-38.5% vs baseline)
  Median score margin: ±16.5 discs

Config D: PROGRESSIVE MCTS + TEMP (sims=300, tau=0.05 extended)
  Iterations analyzed: 5
  Mean score margin: ±13.2 discs  (-51.6% vs baseline)
  Median score margin: ±13.0 discs

================================================================================
CONCLUSIONS
================================================================================

Moving to research-standard MCTS (150→200): -5.8 discs (21.2%)
Temperature fix at standard MCTS: -4.7 discs additional (17.3%)
Progressive MCTS increase (200→300): -3.6 discs additional (13.2%)

✅ Primary cause: Below-standard MCTS depth (150 sims insufficient)
✅ Secondary factor: Aggressive temperature schedule (synergistic with MCTS)
✅ Target achieved: Score margins in healthy range (±12-15 discs)
```

---

## Decision Tree

### Case 1: Config B (200 sims) achieves ±15 discs or better
**Conclusion**: Moving to research-standard MCTS is sufficient
**Action**: Update main `config.yaml` with `simulations: 200`
**Trade-off**: Only 30% slower (9 min → 10 min per iteration)
**Next step**: Monitor for 10-20 iterations; consider 300 if further improvement needed

### Case 2: Config C (200 + temp) significantly better than B
**Conclusion**: Temperature schedule is key factor at standard MCTS
**Action**: Update main `config.yaml` with 200 sims + extended temperature schedule
**Trade-off**: 30% slower, but healthier game diversity
**Next step**: Full training run with these settings

### Case 3: Config D (300 + temp) provides best results
**Conclusion**: Both factors important, progressive MCTS scaling justified
**Action**: Update main `config.yaml` with 300 sims + extended temperature schedule
**Trade-off**: 67% slower (9 min → 15 min per iteration)
**Future**: Consider 400 sims after 50+ iterations when model is mature

### Case 4: Config B shows improvement but insufficient (<±20 discs)
**Conclusion**: Need higher MCTS, but 300 may not be enough
**Action**: Run follow-up ablation with 400 sims (Config E)
**Note**: This suggests MCTS depth is the dominant factor

### Case 5: All configs have similar margins (no improvement)
**Conclusion**: Root cause is elsewhere (network architecture, training dynamics)
**Action**: Deep dive into:
- Value network calibration (MAE, correlation)
- Auxiliary head weights (mobility, stability, corner, parity)
- Network capacity (increase channels or residual blocks)
- Training data quality and diversity

---

## Configuration Details

### Temperature Schedules

**Baseline (A, C)**:
- Plies 1-12: τ=1.0 (fully stochastic)
- Plies 13-20: τ=0.25 (reduced randomness)
- Plies 21+: τ=0.0 (fully deterministic)
- Problem: Deterministic for 70% of game (moves 21-60)

**Extended (B, D)**:
- Plies 1-20: τ=1.0 (extended opening exploration)
- Plies 21-40: τ=0.15 (gentle midgame)
- Plies 41+: τ=0.05 (minimal exploration, matches eval)
- Benefit: Stochastic exploration through 67% of game

### MCTS Simulation Counts

**Below-Standard (A)**: 150 simulations
- Fast: 4.6s per game
- Weak: Crude evaluations, misses tactics
- Status: Below research baseline

**Research-Standard (B, C)**: 200 simulations
- Moderate: 6.0s per game (30% slower)
- Standard: Most research implementations use this baseline
- Status: Recommended minimum for quality training

**Progressive (D)**: 300 simulations
- Slower: ~9.2s per game (2x baseline speed)
- Strong: Better tactical awareness, finds defensive resources
- Status: Middle ground between standard and aggressive

**Future Option**: 400+ simulations
- Much slower: 12.3s per game (2.67x slower)
- Very strong: AlphaZero evaluation standard
- Status: Reserve for mature model stages (iteration 50+)

---

## FAQ

**Q: Why start with 200 sims instead of jumping to 400?**
A: Research literature (MiniZero, OLIVAW) recommends 200 as standard baseline. Progressive scaling (150→200→300→400) is more scientifically sound and computationally efficient than jumping directly to 400. AlphaZero used 800, but had massive TPU infrastructure.

**Q: Can I run configs in parallel?**
A: Yes, each config uses isolated directories. Run in separate terminals if you have compute to spare (total 3.5 hours if parallel).

**Q: Do I need to run all 4 configs?**
A: Recommended, but prioritize:
- Config A (baseline) - essential for comparison
- Config B (200 sims) - tests if research-standard MCTS is sufficient
- Config C (200 + temp) - tests combined effect at standard MCTS
- Config D (300 + temp) - tests if progressive scaling provides further benefit

**Q: What if Config B (200 sims) already achieves ±15 discs?**
A: Then 200 sims is sufficient! No need for 300 or 400. Update main config.yaml and continue training.

**Q: What if none of the configs improve margins?**
A: Root cause is likely network architecture or training dynamics, not MCTS/temperature. Investigate value calibration, auxiliary head weights, or network capacity.

**Q: Can I use fewer iterations?**
A: Yes, 3 iterations (300 games) provides reasonable signal, but 5 iterations (500 games) is more robust for statistical confidence.

**Q: Should I test 400 simulations if 300 isn't enough?**
A: Yes, create Config E with 400 sims. But if 300 doesn't help, the problem may be elsewhere (network, not search depth).

---

## Next Steps After Ablation Study

1. **Analyze results**: Run `python scripts/analyze_ablation_results.py`

2. **Update main config**: Based on conclusions, update `config.yaml` with winning configuration

3. **Resume full training**: Continue from iteration 45+ with improved settings

4. **Monitor progress**: Check if score margins drop to healthy range (±8-15 discs)

5. **Optional**: Run longer ablation (10-20 iterations) if results are ambiguous

---

## Troubleshooting

**Problem**: Config C/D takes too long (>25 min per iteration)
- **Cause**: MPS throughput lower than expected with 400 simulations
- **Solution**: Reduce to 300 simulations as compromise

**Problem**: Score margins don't improve with any config
- **Cause**: Root cause is elsewhere (network, training dynamics)
- **Solution**: Investigate value calibration, auxiliary head weights, network capacity

**Problem**: Results are inconsistent across iterations
- **Cause**: High variance, need more data
- **Solution**: Run 10 iterations instead of 5

**Problem**: Config D is worse than Config C
- **Cause**: Negative interference between fixes
- **Solution**: Use Config C settings (MCTS fix only)

---

## Files Reference

**Configs**:
- `config_ablation_a.yaml` - Baseline
- `config_ablation_b.yaml` - Temperature fix
- `config_ablation_c.yaml` - MCTS fix
- `config_ablation_d.yaml` - Both fixes

**Scripts**:
- `scripts/self_play_train.py` - Main training loop (use with --config)
- `scripts/analyze_ablation_results.py` - Extract and compare results

**Data Directories** (auto-created):
- `data/ablation_a/` - Baseline data
- `data/ablation_b/` - Temperature fix data
- `data/ablation_c/` - MCTS fix data
- `data/ablation_d/` - Both fixes data

---

## Research Justification for Revised Approach

### Critical Fix: Training-to-Evaluation Ratio (100:40 = 2.5:1)

**Problem Identified**: Original config had 50 training games and 60 evaluation games (0.83:1 ratio)
- **Symptom**: Spending MORE time on evaluation than generating training data
- **Impact**: Slow learning, inefficient resource allocation

**Research Standards**:
| Implementation | Training Games | Eval Games | Ratio |
|----------------|----------------|------------|-------|
| AlphaGo Zero   | 25,000        | 400        | 62.5:1 |
| alpha-zero-general (Othello) | 100 | ~40 | 2.5:1 |
| **Original (Wrong)** | **50** | **60** | **0.83:1** ❌ |
| **Fixed** | **100** | **40** | **2.5:1** ✅ |

**Fix Benefits**:
- ✅ 2x training data per iteration (100 games × 8 augmentations = 800 positions)
- ✅ Proper time allocation: 76% training, 24% evaluation
- ✅ Aligned with alpha-zero-general's proven Othello implementation
- ✅ Faster learning curve with more diverse training data

**Evaluation Still Statistically Sound**:
- 40 games at 55% threshold = ±15.5% confidence interval
- Sufficient for promotion decisions (need 22/40 wins)
- Reduced from 60 games saves ~1.5 min per iteration

---

### Why Progressive Scaling (150→200→300) Instead of Jumping to 400?

**Literature Evidence**:
1. **Original AlphaZero**: Used 800 simulations for training with 5,000 TPUs
2. **MiniZero (2023)**: Uses 200 simulations as standard baseline for Othello
3. **OLIVAW Project (2021)**: Progressive scaling strategy (100→200→400 across generations)
4. **ELF OpenGo**: Doubling rollouts yields ~250-400 Elo improvement (diminishing returns)

**Key Findings**:
- 150 simulations is **below research standard** (most use 200+)
- 200 simulations is only **30% slower** than 150 (6.0s vs 4.6s per game)
- Jumping from 150→400 is a **2.67x increase** without intermediate data
- Progressive increases allow **data-driven decisions** rather than guesswork
- 400+ simulations show **exponentially diminishing returns** beyond 200-300

**Cost-Benefit**:
- Testing 200 first: **30% time increase**, likely substantial quality gain
- If 200 sufficient: **Save 2.2x computational cost** vs running 400 throughout
- If 200 insufficient: Test 300 (67% slower) before committing to 400 (2.67x slower)

**Verdict**: Start conservative, scale progressively based on empirical results. This mirrors successful strategies in published research (OLIVAW) and avoids premature optimization.

---

**Ready to start the ablation study!** Run the first config (baseline) to establish the control group. 🔬
