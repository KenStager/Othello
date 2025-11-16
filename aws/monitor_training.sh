#!/bin/bash
# Monitor training on AWS EC2 instance
# SSH into instance and display logs, GPU usage, progress

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.json"
INSTANCE_INFO_FILE="$SCRIPT_DIR/instance_info.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Load instance info
if [ ! -f "$INSTANCE_INFO_FILE" ]; then
    echo "Error: Instance info not found"
    echo "Run: ./launch_spot_instance.sh first"
    exit 1
fi

PUBLIC_IP=$(jq -r '.public_ip' "$INSTANCE_INFO_FILE")
INSTANCE_ID=$(jq -r '.instance_id' "$INSTANCE_INFO_FILE")
KEY_FILE=$(jq -r '.key_file' "$CONFIG_FILE")

echo "=========================================="
echo "Monitoring Othello Training"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""

# SSH options
SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Function to check if instance is running
check_instance() {
    aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text 2>/dev/null || echo "terminated"
}

STATE=$(check_instance)
if [ "$STATE" != "running" ]; then
    echo "Error: Instance is not running (state: $STATE)"
    exit 1
fi

# Menu
echo "Select monitoring option:"
echo "  1) Stream training logs (tail -f)"
echo "  2) Show GPU usage (nvidia-smi)"
echo "  3) Show training progress (checkpoints, iteration)"
echo "  4) Show TensorBoard URL (SSH tunnel)"
echo "  5) Interactive SSH session"
echo "  6) All-in-one dashboard"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "Streaming training logs (Ctrl+C to exit)..."
        echo "----------------------------------------"
        ssh $SSH_OPTS ubuntu@$PUBLIC_IP "tail -f Othello/train.log"
        ;;

    2)
        echo ""
        echo "GPU Usage (refreshing every 2 seconds, Ctrl+C to exit)..."
        echo "----------------------------------------"
        ssh $SSH_OPTS ubuntu@$PUBLIC_IP "watch -n 2 nvidia-smi"
        ;;

    3)
        echo ""
        echo "Training Progress:"
        echo "----------------------------------------"
        ssh $SSH_OPTS ubuntu@$PUBLIC_IP << 'EOF'
cd Othello
echo "Latest checkpoints:"
ls -lht data/checkpoints/*.pt 2>/dev/null | head -5 || echo "  No checkpoints yet"
echo ""
echo "Replay buffer:"
ls -lht data/replay/*.pkl 2>/dev/null | head -3 || echo "  No replay data yet"
echo ""
echo "Recent log entries:"
tail -20 train.log | grep -E "(Iteration|Training|Gating|promoted)" || echo "  No training output yet"
EOF
        ;;

    4)
        echo ""
        echo "TensorBoard Access:"
        echo "----------------------------------------"
        echo "To access TensorBoard running on the instance:"
        echo ""
        echo "1. In a separate terminal, create SSH tunnel:"
        echo "     ssh -i $KEY_FILE -L 6006:localhost:6006 ubuntu@$PUBLIC_IP"
        echo ""
        echo "2. Open in browser:"
        echo "     http://localhost:6006"
        echo ""
        echo "(TensorBoard starts automatically with training)"
        ;;

    5)
        echo ""
        echo "Opening SSH session..."
        echo "----------------------------------------"
        ssh $SSH_OPTS ubuntu@$PUBLIC_IP
        ;;

    6)
        echo ""
        echo "All-in-One Dashboard"
        echo "=========================================="
        ssh $SSH_OPTS ubuntu@$PUBLIC_IP << 'EOF'
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "=== Training Process ==="
pgrep -a python | grep self_play_train || echo "Training not running"

echo ""
echo "=== Latest Checkpoint ==="
ls -lt Othello/data/checkpoints/*.pt 2>/dev/null | head -1 || echo "No checkpoints yet"

echo ""
echo "=== Recent Training Output (last 10 lines) ==="
tail -10 Othello/train.log 2>/dev/null || echo "No log file yet"

echo ""
echo "=== Disk Usage ==="
df -h /dev/nvme0n1p1 | tail -1

echo ""
echo "=== Uptime ==="
uptime
EOF
        echo ""
        echo "=========================================="
        echo "Dashboard complete. Run again to refresh."
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
