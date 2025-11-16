#!/bin/bash
# Bootstrap script for EC2 instance
# Runs automatically on instance launch via user data
# Sets up environment and starts Othello training

set -e

# Log everything
exec > >(tee /var/log/othello-bootstrap.log)
exec 2>&1

echo "==========================================="
echo "Othello Training Bootstrap Starting"
echo "Time: $(date)"
echo "==========================================="

# Activate PyTorch conda environment
echo "1. Activating PyTorch environment..."
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch || conda activate pytorch_latest_p310 || source activate pytorch

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Clone repository
echo ""
echo "2. Cloning repository..."
cd /home/ubuntu
if [ -d "Othello" ]; then
    echo "Repository already exists, pulling latest..."
    cd Othello && git pull && cd ..
else
    git clone https://github.com/KenStager/Othello.git
fi
cd Othello

echo "Current branch: $(git branch --show-current)"
echo "Latest commit: $(git log -1 --oneline)"

# Install Python dependencies
echo ""
echo "3. Installing Python dependencies..."
pip install pyyaml matplotlib pandas --quiet

# Setup Edax oracle
echo ""
echo "4. Setting up Edax oracle..."
if [ -f "third_party/edax/bin/edax" ]; then
    chmod +x third_party/edax/bin/edax
    echo "Edax binary ready: $(ls -lh third_party/edax/bin/edax)"
else
    echo "Warning: Edax binary not found at third_party/edax/bin/edax"
    echo "Oracle will be disabled"
fi

# Create data directories
echo ""
echo "5. Creating data directories..."
mkdir -p data/checkpoints data/replay data/openings data/il_bootstrap logs runs

# Check for existing checkpoint to resume
echo ""
echo "6. Checking for existing checkpoints..."
CHECKPOINT_COUNT=$(find data/checkpoints -name "*.pt" 2>/dev/null | wc -l)
if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
    LATEST_CHECKPOINT=$(ls -t data/checkpoints/*.pt 2>/dev/null | head -1)
    echo "Found $CHECKPOINT_COUNT checkpoint(s)"
    echo "Latest: $LATEST_CHECKPOINT"
    echo "Training will resume from checkpoint"
else
    echo "No checkpoints found - starting fresh"
fi

# Start Spot termination handler in background
echo ""
echo "7. Starting Spot termination handler..."
cat > /home/ubuntu/spot_termination_handler.sh <<'HANDLER_EOF'
#!/bin/bash
# Monitor for Spot termination notice

while true; do
    # Check metadata endpoint for termination notice
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "X-aws-ec2-metadata-token: $(curl -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' -s http://169.254.169.254/latest/api/token)" \
        http://169.254.169.254/latest/meta-data/spot/instance-action 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" == "200" ]; then
        echo "$(date): Spot termination notice received! Saving checkpoint..."

        # Send SIGTERM to training process (triggers checkpoint save)
        pkill -SIGTERM -f "self_play_train.py"

        # Wait for graceful shutdown
        sleep 30

        echo "$(date): Checkpoint saved. Instance will terminate soon."
        break
    fi

    # Check every 5 seconds
    sleep 5
done
HANDLER_EOF

chmod +x /home/ubuntu/spot_termination_handler.sh
nohup /home/ubuntu/spot_termination_handler.sh > /var/log/spot-termination.log 2>&1 &
echo "Termination handler started (PID: $!)"

# Start training
echo ""
echo "8. Starting training..."
echo "Command: python -u scripts/self_play_train.py --config config_cloud_aws.yaml"
echo "Logs: /home/ubuntu/Othello/train.log"
echo ""

cd /home/ubuntu/Othello
nohup python -u scripts/self_play_train.py --config config_cloud_aws.yaml > train.log 2>&1 &
TRAIN_PID=$!

echo "Training started (PID: $TRAIN_PID)"
echo ""

# Wait for training to start
sleep 10

# Check if training is running
if ps -p $TRAIN_PID > /dev/null; then
    echo "✓ Training process is running"
    echo ""
    echo "Initial log output:"
    head -20 train.log
else
    echo "✗ Training process failed to start"
    echo "Check logs: /home/ubuntu/Othello/train.log"
fi

echo ""
echo "==========================================="
echo "Bootstrap Complete!"
echo "Time: $(date)"
echo "==========================================="
echo ""
echo "To monitor training:"
echo "  ssh ubuntu@$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4) tail -f Othello/train.log"
echo ""
echo "TensorBoard:"
echo "  Port 6006 (set up SSH tunnel to access)"
echo "==========================================="
