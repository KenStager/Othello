# Training Commands

## 🚀 Run Full Training (Production)

### Option 1: Direct Command
```bash
PYTHONPATH=. python scripts/self_play_train.py --config config.yaml
```

### Option 2: Using Shell Script
```bash
./RUN_TRAINING.sh
```

### Settings (config.yaml)
- **20 games** per iteration
- **200 MCTS simulations** per move
- **256 batch size**
- **200 training steps** per iteration
- **Expected time**: 30-60 minutes per iteration (CPU)

---

## ⚡ Run Fast Testing (Smoke Test)

```bash
PYTHONPATH=. python scripts/self_play_train.py --config config_smoke_test.yaml
```

### Settings (config_smoke_test.yaml)
- **2 games** per iteration (10× faster)
- **50 MCTS simulations** per move (4× faster)
- **64 batch size** (4× faster)
- **20 training steps** per iteration (10× faster)
- **Expected time**: 2-5 minutes per iteration (CPU)

---

## 🎮 Play Against AI

```bash
PYTHONPATH=. python scripts/play_human.py --config config.yaml
```

---

## 🧪 Test All Enhancements

```bash
PYTHONPATH=. python test_enhancements.py
```

Expected output:
```
============================================================
TESTING OTHELLO ENHANCEMENTS
============================================================

[1/8] Testing configuration loading... ✓
[2/8] Testing Zobrist hashing... ✓
[3/8] Testing phase tagging... ✓
[4/8] Testing MCTS with transposition table... ✓
[5/8] Testing replay buffer with phase tagging... ✓
[6/8] Testing phase-weighted loss... ✓
[7/8] Testing oracle bridge... ✓
[8/8] Testing gating criteria... ✓

============================================================
ALL TESTS PASSED ✓
============================================================
```

---

## 📊 Monitor Training

### Watch Live Logs
```bash
tail -f logs/train.tsv
```

### View Training Output
```bash
# If running in background, check output:
tail -20 logs/train.tsv
```

### Check Checkpoints
```bash
ls -lh data/checkpoints/
```

---

## 🔧 Generate Opening Suite

```bash
PYTHONPATH=. python scripts/make_opening_suite.py \
  --auto_count 32 \
  --out data/openings/rot64.json
```

This creates 256 positions (32 × 8 symmetries) for balanced evaluation.

---

## 🏗️ GPU Training (Faster)

### Option A: Apple Silicon (MPS) - **Default Configuration**

**Prerequisites**: macOS with Apple Silicon (M1/M2/M3), PyTorch 2.2+

**Already configured!** The default `config.yaml` uses MPS device.

#### Environment Setup (Optional - for advanced tuning)
```bash
# Add to ~/.zshrc or ~/.bash_profile
export PYTORCH_ENABLE_MPS_FALLBACK=1      # CPU fallback for unsupported ops
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.4  # Memory limit (conservative)
```

#### Run training
```bash
PYTHONPATH=. python scripts/self_play_train.py --config config.yaml
```

**Expected speedup**: 2-3× faster than CPU
**Memory management**: Automatically uses 75% of recommended memory (~45-48GB on 64GB system)
**System buffer**: Leaves 15-20GB free for macOS

#### Verify MPS is working
Look for in training output:
```
Using device: mps
  MPS recommended max memory: 51.20 GB
  MPS memory fraction: 0.75 (target: 38.40 GB)
```

#### Troubleshooting
- **OOM errors**: Reduce `batch_size` in config.yaml (256 → 128)
- **Slow training**: Check Activity Monitor for memory pressure (yellow/red)
- **Fallback to CPU**: If MPS unavailable, automatically uses CPU

---

### Option B: NVIDIA GPU (CUDA)

**Prerequisites**: NVIDIA GPU, CUDA toolkit, PyTorch with CUDA support

#### 1. Update config.yaml
```yaml
device: "cuda"  # Change from "mps"

train:
  amp_enabled: true  # Enable automatic mixed precision
```

#### 2. Run training
```bash
PYTHONPATH=. python scripts/self_play_train.py --config config.yaml
```

**Expected speedup**: 5-10× faster than CPU (RTX 3090 or equivalent)

---

## 📈 TensorBoard (Optional)

### 1. Install TensorBoard
```bash
pip install tensorboard
```

### 2. Enable in config.yaml
```yaml
logging:
  tensorboard: true
```

### 3. Start training
```bash
PYTHONPATH=. python scripts/self_play_train.py --config config.yaml
```

### 4. View in browser (separate terminal)
```bash
tensorboard --logdir=logs/tensorboard
```

Then open: http://localhost:6006

---

## 🔬 Edax Oracle Setup (Optional)

### 1. Download and build Edax
```bash
# See Edax documentation for build instructions
# https://github.com/abulmo/edax-reversi
```

### 2. Update config.yaml
```yaml
oracle:
  use: true
  edax_path: "third_party/edax/bin/edax"
  empties_threshold: 14
```

### 3. Run training
```bash
PYTHONPATH=. python scripts/self_play_train.py --config config.yaml
```

**Benefit**: Perfect endgame play when ≤14 empties

---

## 🎯 Common Use Cases

### Quick Test (Verify Everything Works)
```bash
PYTHONPATH=. python test_enhancements.py
PYTHONPATH=. python scripts/self_play_train.py --config config_smoke_test.yaml
# Wait for 2-3 iterations, then Ctrl+C
```

### Overnight Training Session
```bash
# Use nohup to keep running after logout
nohup PYTHONPATH=. python scripts/self_play_train.py --config config.yaml > training.log 2>&1 &

# Check progress
tail -f training.log
```

### Resume Training (Continues from Last Checkpoint)
```bash
# Just run the same command - checkpoints and replay buffer persist
PYTHONPATH=. python scripts/self_play_train.py --config config.yaml
```

---

## 🛠️ Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Always set PYTHONPATH
```bash
PYTHONPATH=. python scripts/self_play_train.py --config config.yaml
```

### Slow Training
**Problem**: Training taking too long

**Solution 1**: Use smoke test config
```bash
PYTHONPATH=. python scripts/self_play_train.py --config config_smoke_test.yaml
```

**Solution 2**: Reduce settings in config.yaml
```yaml
mcts:
  simulations: 100  # Down from 200

selfplay:
  games_per_iter: 10  # Down from 20
```

**Solution 3**: Use GPU
```yaml
device: "cuda"
train:
  amp_enabled: true
```

### Memory Issues
**Problem**: Out of memory

**Solution**: Reduce batch size and buffer
```yaml
train:
  batch_size: 128  # Down from 256
  replay_capacity: 100000  # Down from 200000
```

### Checkpoints Not Saving
**Problem**: No checkpoints in data/checkpoints/

**Solution**: Check paths in config.yaml
```yaml
paths:
  checkpoint_dir: "data/checkpoints"  # Ensure this exists
```

---

## 📝 Full Training Workflow

```bash
# 1. Test everything works
PYTHONPATH=. python test_enhancements.py

# 2. Quick smoke test (5-10 iterations)
PYTHONPATH=. python scripts/self_play_train.py --config config_smoke_test.yaml

# 3. Run full training
PYTHONPATH=. python scripts/self_play_train.py --config config.yaml

# 4. Monitor progress (separate terminal)
tail -f logs/train.tsv

# 5. Play against trained model
PYTHONPATH=. python scripts/play_human.py --config config.yaml
```

---

## 🎓 Understanding Output

### Training Output Format
```
=== Iteration 5 ===
Self-play added samples: 640 (buffer size=3200)
Train loss: 4.1882
Gating (mini): W/L/D = 0/0/0 (winrate=0.00%, loss_rate=0.00%)
Saved current checkpoint: data/checkpoints/current_iter5.pt
```

### What Each Line Means
- **Iteration N**: Training iteration number
- **Self-play added samples**: New positions (includes 8 symmetries)
- **Buffer size**: Total samples available for training
- **Train loss**: Combined loss (should decrease over time)
- **W/L/D**: Wins/Losses/Draws in evaluation games
- **Winrate**: % of decisive games won (threshold: 55%)
- **Loss rate**: % of all games lost (NEW - prevents regression)
- **Checkpoint**: Auto-saved model file

### Healthy Training Signs
✅ Loss decreasing over iterations
✅ Buffer size growing steadily
✅ No crashes or error messages
✅ Checkpoints being saved regularly
✅ Win rate eventually > 55% (triggers promotion)

---

## 📚 Documentation Reference

- **QUICKSTART.md** - Quick start guide
- **ENGINEERING_GUIDE.md** - Complete design document
- **IMPLEMENTATION_SUMMARY.md** - What was implemented
- **TRAINING_IN_ACTION.md** - Live training analysis
- **PROGRESS.md** - Implementation tracking
- **CLAUDE.md** - Code architecture

---

**Last Updated**: 2025-11-12
