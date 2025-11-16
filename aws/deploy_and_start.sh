#!/bin/bash
# Complete deployment workflow
# Convenience script that runs: check → launch → monitor

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "==========================================="
echo "Othello AlphaZero - AWS Deployment"
echo "==========================================="
echo ""

# Step 1: Prerequisites check
echo "Step 1: Checking prerequisites..."
"$SCRIPT_DIR/check_prerequisites.sh"
if [ $? -ne 0 ]; then
    echo "Prerequisites check failed. Please fix issues and try again."
    exit 1
fi
echo ""

# Step 2: Check if setup is done
if [ ! -f "$SCRIPT_DIR/config.json" ]; then
    echo "Step 2: Running one-time setup..."
    "$SCRIPT_DIR/setup_aws_resources.sh"
    echo ""
else
    echo "Step 2: Setup already complete (skip)"
    echo ""
fi

# Step 3: Launch instance
echo "Step 3: Launching Spot instance..."
"$SCRIPT_DIR/launch_spot_instance.sh"
echo ""

# Get instance info
INSTANCE_INFO_FILE="$SCRIPT_DIR/instance_info.json"
if [ ! -f "$INSTANCE_INFO_FILE" ]; then
    echo "Error: Instance launch failed"
    exit 1
fi

PUBLIC_IP=$(jq -r '.public_ip' "$INSTANCE_INFO_FILE")
KEY_FILE=$(jq -r '.key_file' "$SCRIPT_DIR/config.json")

# Step 4: Wait for bootstrap to complete
echo "Step 4: Waiting for bootstrap to complete (3 minutes)..."
echo "  The instance is:"
echo "    - Cloning GitHub repository"
echo "    - Installing dependencies"
echo "    - Setting up Edax oracle"
echo "    - Starting training"
echo ""

for i in {1..180}; do
    printf "\r  Elapsed: %d seconds / 180 seconds" $i
    sleep 1
done
echo ""
echo ""

# Step 5: Check if training started
echo "Step 5: Verifying training started..."
SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

TRAINING_RUNNING=$(ssh $SSH_OPTS ubuntu@$PUBLIC_IP "pgrep -f self_play_train.py" 2>/dev/null || echo "")

if [ -n "$TRAINING_RUNNING" ]; then
    echo -e "${GREEN}✓${NC} Training is running (PID: $TRAINING_RUNNING)"
else
    echo -e "${YELLOW}⚠${NC} Training process not found"
    echo "  Checking bootstrap log..."
    ssh $SSH_OPTS ubuntu@$PUBLIC_IP "tail -20 /var/log/othello-bootstrap.log" || true
fi
echo ""

# Step 6: Show initial output
echo "Step 6: Initial training output:"
echo "==========================================="
ssh $SSH_OPTS ubuntu@$PUBLIC_IP "tail -30 Othello/train.log 2>/dev/null" || echo "Log file not ready yet"
echo "==========================================="
echo ""

# Final instructions
echo "==========================================="
echo -e "${GREEN}✓ Deployment complete!${NC}"
echo ""
echo "Training is running on AWS!"
echo ""
echo "Monitoring options:"
echo "  1. Stream logs:"
echo "       ./monitor_training.sh  (then select option 1)"
echo "  2. Manual SSH:"
echo "       ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo "       cd Othello && tail -f train.log"
echo "  3. TensorBoard (in separate terminal):"
echo "       ssh -i $KEY_FILE -L 6006:localhost:6006 ubuntu@$PUBLIC_IP"
echo "       Then open: http://localhost:6006"
echo ""
echo "When finished:"
echo "  ./shutdown_instance.sh --download-checkpoints --terminate"
echo ""
echo "Spot instance will save checkpoints every iteration."
echo "If interrupted, just run this script again to resume!"
echo "==========================================="
