#!/bin/bash
# Othello AlphaZero Training Script
# Run this to start full production training

echo "=========================================="
echo "Starting Othello AlphaZero Training"
echo "=========================================="
echo ""
echo "Configuration: config.yaml (Full Training)"
echo "Settings:"
echo "  - 20 games per iteration"
echo "  - 200 MCTS simulations per move"
echo "  - 256 batch size"
echo "  - 200 training steps per iteration"
echo ""
echo "Expected time per iteration: 30-60 minutes (CPU)"
echo "Tip: Set device: cuda in config.yaml for GPU training"
echo ""
echo "Press Ctrl+C to stop training at any time"
echo "Checkpoints auto-saved to: data/checkpoints/"
echo "Logs written to: logs/train.tsv"
echo ""
echo "=========================================="
echo ""

# Set PYTHONPATH and run training
PYTHONPATH=. python scripts/self_play_train.py --config config.yaml
