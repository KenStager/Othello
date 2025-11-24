#!/bin/bash
# Vast.ai Onstart Script for Othello AlphaZero Training
# Runs inside Docker container on instance startup

set -e

echo "==========================================="
echo "Othello AlphaZero - Vast.ai Setup"
echo "==========================================="
echo ""

# Update package lists
echo "1. Updating system packages..."
apt-get update -qq > /dev/null 2>&1
apt-get install -y git wget curl > /dev/null 2>&1
echo "   ✓ System packages updated"
echo ""

# Clone repository
echo "2. Cloning Othello repository..."
cd /workspace
if [ -d "Othello" ]; then
    echo "   Repository already exists, updating..."
    cd Othello
    git pull
else
    git clone https://github.com/KenStager/Othello.git
    cd Othello
fi
echo "   ✓ Repository cloned to /workspace/Othello"
echo ""

# Install Python dependencies
echo "3. Installing Python dependencies..."
pip install pyyaml matplotlib pandas --quiet
echo "   ✓ Dependencies installed"
echo ""

# Setup Edax oracle binary
echo "4. Setting up Edax oracle..."
if [ -f "third_party/edax/bin/edax" ]; then
    chmod +x third_party/edax/bin/edax
    echo "   ✓ Edax binary permissions set"
else
    echo "   ⚠ Edax binary not found (may need to download separately)"
fi
echo ""

# Create data directories
echo "5. Creating data directories..."
mkdir -p data/checkpoints data/replay logs runs
echo "   ✓ Directories created"
echo ""

# Start training in background
echo "6. Starting training..."
echo "   Config: config_cloud_aws.yaml"
echo "   Log file: /workspace/train.log"
echo ""

nohup python -u scripts/self_play_train.py --config config_cloud_aws.yaml > /workspace/train.log 2>&1 &
TRAIN_PID=$!

# Start TensorBoard server in background
echo "7. Starting TensorBoard server..."
echo "   Port: 6006"
echo "   Logdir: runs/"
echo ""

nohup tensorboard --logdir=runs --port=6006 --host=0.0.0.0 --bind_all > /workspace/tensorboard.log 2>&1 &
TB_PID=$!

echo "==========================================="
echo "✓ Setup Complete!"
echo "==========================================="
echo ""
echo "Training started (PID: $TRAIN_PID)"
echo "TensorBoard started (PID: $TB_PID)"
echo ""
echo "Monitor training with:"
echo "  tail -f /workspace/train.log"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Training progress:"
echo "  ls -lh /workspace/Othello/data/checkpoints/"
echo ""
echo "Access TensorBoard:"
echo "  From local machine: ssh -p <PORT> root@<HOST> -L 6006:localhost:6006"
echo "  Then open: http://localhost:6006"
echo ""
echo "==========================================="
