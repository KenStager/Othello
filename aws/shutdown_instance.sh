#!/bin/bash
# Shutdown and cleanup AWS EC2 instance
# Options: download checkpoints, terminate instance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.json"
INSTANCE_INFO_FILE="$SCRIPT_DIR/instance_info.json"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
DOWNLOAD_CHECKPOINTS=false
TERMINATE=false
COST_REPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --download-checkpoints)
            DOWNLOAD_CHECKPOINTS=true
            shift
            ;;
        --terminate)
            TERMINATE=true
            shift
            ;;
        --cost-report)
            COST_REPORT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--download-checkpoints] [--terminate] [--cost-report]"
            exit 1
            ;;
    esac
done

# Load instance info
if [ ! -f "$INSTANCE_INFO_FILE" ]; then
    echo "Error: Instance info not found"
    echo "No active instance to shutdown"
    exit 1
fi

INSTANCE_ID=$(jq -r '.instance_id' "$INSTANCE_INFO_FILE")
PUBLIC_IP=$(jq -r '.public_ip' "$INSTANCE_INFO_FILE")
LAUNCH_TIME=$(jq -r '.launch_time' "$INSTANCE_INFO_FILE")
INSTANCE_TYPE=$(jq -r '.instance_type' "$INSTANCE_INFO_FILE")
AWS_REGION=$(jq -r '.region' "$CONFIG_FILE")
KEY_FILE=$(jq -r '.key_file' "$CONFIG_FILE")

echo "=========================================="
echo "Shutdown AWS Instance"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "Type: $INSTANCE_TYPE"
echo ""

# Check instance state
STATE=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $AWS_REGION \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text 2>/dev/null || echo "not-found")

if [ "$STATE" == "not-found" ] || [ "$STATE" == "terminated" ]; then
    echo "Instance is already terminated"
    rm -f "$INSTANCE_INFO_FILE"
    exit 0
fi

echo "Current state: $STATE"
echo ""

# Download checkpoints if requested
if [ "$DOWNLOAD_CHECKPOINTS" == true ]; then
    echo "1. Downloading checkpoints..."
    DOWNLOAD_DIR="$SCRIPT_DIR/downloaded_checkpoints_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$DOWNLOAD_DIR"

    SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

    # Download latest checkpoint
    echo "  Downloading latest checkpoint..."
    scp $SSH_OPTS -r ubuntu@$PUBLIC_IP:Othello/data/checkpoints/*.pt "$DOWNLOAD_DIR/" 2>/dev/null || true

    # Download training log
    echo "  Downloading training log..."
    scp $SSH_OPTS ubuntu@$PUBLIC_IP:Othello/train.log "$DOWNLOAD_DIR/" 2>/dev/null || true

    # Download TensorBoard logs
    echo "  Downloading TensorBoard logs..."
    scp $SSH_OPTS -r ubuntu@$PUBLIC_IP:Othello/runs "$DOWNLOAD_DIR/" 2>/dev/null || true

    CHECKPOINT_COUNT=$(ls "$DOWNLOAD_DIR"/*.pt 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} Downloaded $CHECKPOINT_COUNT checkpoint(s) to: $DOWNLOAD_DIR"
    echo ""
fi

# Cost report if requested
if [ "$COST_REPORT" == true ]; then
    echo "2. Generating cost report..."

    # Calculate runtime
    CURRENT_TIME=$(date -u +%s)
    LAUNCH_TIME_EPOCH=$(date -u -d "$LAUNCH_TIME" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "$LAUNCH_TIME" +%s)
    RUNTIME_SECONDS=$((CURRENT_TIME - LAUNCH_TIME_EPOCH))
    RUNTIME_HOURS=$(echo "scale=2; $RUNTIME_SECONDS / 3600" | bc)

    # Get Spot price history
    AVG_SPOT_PRICE=$(aws ec2 describe-spot-price-history \
        --instance-types $INSTANCE_TYPE \
        --product-descriptions "Linux/UNIX" \
        --start-time "$LAUNCH_TIME" \
        --region $AWS_REGION \
        --query 'SpotPriceHistory[*].SpotPrice' \
        --output text | awk '{sum+=$1; count++} END {if(count>0) printf "%.4f", sum/count; else print "0.20"}')

    ESTIMATED_COST=$(echo "$RUNTIME_HOURS * $AVG_SPOT_PRICE" | bc -l)

    echo "  Launch time: $LAUNCH_TIME"
    echo "  Current time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf "  Runtime: %.2f hours\n" $RUNTIME_HOURS
    printf "  Avg Spot price: \$%.4f/hour\n" $AVG_SPOT_PRICE
    printf "  Estimated cost: \$%.2f\n" $ESTIMATED_COST
    echo ""
fi

# Terminate if requested
if [ "$TERMINATE" == true ]; then
    read -p "Are you sure you want to terminate instance $INSTANCE_ID? (yes/no): " confirm
    if [ "$confirm" == "yes" ]; then
        echo ""
        echo "3. Terminating instance..."
        aws ec2 terminate-instances \
            --instance-ids $INSTANCE_ID \
            --region $AWS_REGION \
            --output text > /dev/null

        echo -e "${GREEN}✓${NC} Termination initiated"
        echo ""
        echo "Waiting for instance to terminate..."
        aws ec2 wait instance-terminated \
            --instance-ids $INSTANCE_ID \
            --region $AWS_REGION

        echo -e "${GREEN}✓${NC} Instance terminated"

        # Clean up instance info file
        rm -f "$INSTANCE_INFO_FILE"
    else
        echo "Termination cancelled"
    fi
else
    echo "Note: Instance is still running"
    echo "To terminate, run with --terminate flag"
fi

echo ""
echo "=========================================="
echo "Shutdown complete"
echo "=========================================="
