#!/bin/bash
# Vast.ai Monitoring Script for Othello Training
# Interactive monitoring of running instance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_ID_FILE="$SCRIPT_DIR/instance_id.txt"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

if [ ! -f "$INSTANCE_ID_FILE" ]; then
    echo -e "${RED}✗${NC} No running instance found"
    echo "  Deploy first with: ./deploy.sh"
    exit 1
fi

INSTANCE_ID=$(cat "$INSTANCE_ID_FILE")

echo "==========================================="
echo "Othello Training Monitor"
echo "==========================================="
echo ""
echo "Instance ID: $INSTANCE_ID"
echo ""

# Get SSH connection info
SSH_INFO=$(vastai ssh-url $INSTANCE_ID 2>/dev/null)

if [ -z "$SSH_INFO" ]; then
    echo -e "${RED}✗${NC} Could not get SSH connection info"
    echo "  Instance may have been destroyed or is not ready"
    echo "  Check status: vastai show instance $INSTANCE_ID"
    exit 1
fi

SSH_CMD=$(echo "$SSH_INFO" | sed 's/ssh //')

echo "What would you like to monitor?"
echo ""
echo "1) Stream training logs (tail -f)"
echo "2) Show GPU usage (nvidia-smi)"
echo "3) Show training progress (checkpoint count)"
echo "4) Show instance status"
echo "5) Open interactive SSH session"
echo "6) Show all (combined dashboard)"
echo "7) TensorBoard status & access instructions"
echo ""
read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo ""
        echo "=== Training Logs ===="
        echo "Press Ctrl+C to exit"
        echo ""
        ssh $SSH_CMD "tail -f /workspace/train.log"
        ;;

    2)
        echo ""
        echo "=== GPU Usage ==="
        echo "Refreshing every 2 seconds (Ctrl+C to exit)"
        echo ""
        while true; do
            ssh $SSH_CMD "nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits" 2>/dev/null || break
            echo "---"
            sleep 2
        done
        ;;

    3)
        echo ""
        echo "=== Training Progress ==="
        ssh $SSH_CMD "cd /workspace/Othello && echo 'Checkpoints:' && ls -lh data/checkpoints/ 2>/dev/null | tail -10 && echo '' && echo 'Latest iteration:' && ls data/checkpoints/ | grep -oE 'iter_[0-9]+' | sort -V | tail -1"
        ;;

    4)
        echo ""
        echo "=== Instance Status ==="
        vastai show instance $INSTANCE_ID
        ;;

    5)
        echo ""
        echo "=== Opening SSH Session ==="
        echo "Run 'exit' to return to local terminal"
        echo ""
        ssh $SSH_CMD "cd /workspace/Othello && exec bash -l"
        ;;

    6)
        echo ""
        echo "=== Combined Dashboard ==="
        echo ""

        # Instance status
        echo -e "${BLUE}[Instance Status]${NC}"
        INSTANCE_INFO=$(vastai show instance $INSTANCE_ID --raw 2>/dev/null)
        STATUS=$(echo "$INSTANCE_INFO" | jq -r '.actual_status')
        GPU=$(echo "$INSTANCE_INFO" | jq -r '.gpu_name' | sed 's/_/ /g')
        PRICE=$(echo "$INSTANCE_INFO" | jq -r '.dph_total')
        echo "Status: $STATUS"
        echo "GPU: $GPU"
        echo "Price: \$$PRICE/hour"
        echo ""

        # GPU usage
        echo -e "${BLUE}[GPU Usage]${NC}"
        ssh $SSH_CMD "nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits" 2>/dev/null || echo "Not available"
        echo ""

        # Training progress
        echo -e "${BLUE}[Training Progress]${NC}"
        ssh $SSH_CMD "cd /workspace/Othello 2>/dev/null && ls data/checkpoints/ 2>/dev/null | grep -oE 'iter_[0-9]+' | sort -V | tail -1 | sed 's/iter_/Latest iteration: /'" 2>/dev/null || echo "No checkpoints yet"
        ssh $SSH_CMD "cd /workspace/Othello 2>/dev/null && ls data/checkpoints/*.pt 2>/dev/null | wc -l | sed 's/^/Total checkpoints: /'" 2>/dev/null
        echo ""

        # Recent logs
        echo -e "${BLUE}[Recent Logs (last 10 lines)]${NC}"
        ssh $SSH_CMD "tail -10 /workspace/train.log 2>/dev/null" || echo "Training log not available"
        echo ""

        echo "==========================================="
        echo "For live monitoring, choose option 1 or 2"
        echo "==========================================="
        ;;

    7)
        echo ""
        echo "=== TensorBoard Status ==="
        echo ""

        # Check if TensorBoard is running
        TB_STATUS=$(ssh $SSH_CMD "ps aux | grep tensorboard | grep -v grep" 2>/dev/null)

        if [ -n "$TB_STATUS" ]; then
            echo -e "${GREEN}✓${NC} TensorBoard is running"
            echo ""
            echo "Process:"
            echo "$TB_STATUS" | head -1
        else
            echo -e "${RED}✗${NC} TensorBoard is not running"
            echo ""
            echo "To start manually:"
            echo "  ssh to instance and run:"
            echo "  nohup tensorboard --logdir=runs --port=6006 --host=0.0.0.0 &"
        fi

        echo ""
        echo "=== Access Instructions ==="
        echo ""
        echo "Option 1 - Quick access (recommended):"
        echo "  ./tensorboard.sh"
        echo ""
        echo "Option 2 - Manual SSH tunnel:"
        echo "  $SSH_INFO -L 6006:localhost:6006"
        echo "  Then open: http://localhost:6006"
        echo ""
        echo "=== Metrics Being Logged ==="
        echo ""
        echo "  Training: loss, learning_rate"
        echo "  Gating: win_rate, loss_rate, avg_moves, promotions"
        echo "  Diagnostics: value_correlation, policy_entropy"
        echo "  Data: buffer_size"
        echo ""
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
