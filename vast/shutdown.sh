#!/bin/bash
# Vast.ai Shutdown Script for Othello Training
# Downloads checkpoints and destroys instance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_ID_FILE="$SCRIPT_DIR/instance_id.txt"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "==========================================="
echo "Othello Training Shutdown"
echo "==========================================="
echo ""

if [ ! -f "$INSTANCE_ID_FILE" ]; then
    echo -e "${RED}✗${NC} No running instance found"
    echo "  No instance to shutdown"
    exit 0
fi

INSTANCE_ID=$(cat "$INSTANCE_ID_FILE")
echo "Instance ID: $INSTANCE_ID"
echo ""

# Get instance info for cost calculation
INSTANCE_INFO=$(vastai show instance $INSTANCE_ID --raw 2>/dev/null)

if [ -z "$INSTANCE_INFO" ] || [ "$INSTANCE_INFO" == "null" ]; then
    echo -e "${YELLOW}⚠${NC} Instance not found (may already be destroyed)"
    rm "$INSTANCE_ID_FILE"
    exit 0
fi

STATUS=$(echo "$INSTANCE_INFO" | jq -r '.actual_status')
PRICE=$(echo "$INSTANCE_INFO" | jq -r '.dph_total')
GPU=$(echo "$INSTANCE_INFO" | jq -r '.gpu_name' | sed 's/_/ /g')

echo "Status: $STATUS"
echo "GPU: $GPU"
echo "Price: \$$PRICE/hour"
echo ""

# Parse command line arguments
DOWNLOAD_CHECKPOINTS=1
DESTROY_INSTANCE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-download)
            DOWNLOAD_CHECKPOINTS=0
            shift
            ;;
        --no-destroy)
            DESTROY_INSTANCE=0
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-download] [--no-destroy]"
            exit 1
            ;;
    esac
done

# Get SSH connection info
SSH_INFO=$(vastai ssh-url $INSTANCE_ID 2>/dev/null)

if [ -z "$SSH_INFO" ]; then
    echo -e "${YELLOW}⚠${NC} Could not get SSH connection (instance may be stopped)"
    DOWNLOAD_CHECKPOINTS=0
fi

# Step 1: Download checkpoints
if [ $DOWNLOAD_CHECKPOINTS -eq 1 ]; then
    echo "Step 1: Downloading checkpoints..."
    echo ""

    # Extract SSH port and host
    SSH_PORT=$(echo "$SSH_INFO" | grep -oP '(?<=-p )\d+')
    SSH_HOST=$(echo "$SSH_INFO" | grep -oP 'root@\S+' | sed 's/root@//')

    # Create local checkpoint directory
    LOCAL_CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints_vast_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOCAL_CHECKPOINT_DIR"

    echo "Downloading to: $LOCAL_CHECKPOINT_DIR"
    echo ""

    # Download checkpoints
    if scp -P "$SSH_PORT" -r "root@$SSH_HOST:/workspace/Othello/data/checkpoints/*" "$LOCAL_CHECKPOINT_DIR/" 2>/dev/null; then
        CHECKPOINT_COUNT=$(ls -1 "$LOCAL_CHECKPOINT_DIR"/*.pt 2>/dev/null | wc -l)
        if [ $CHECKPOINT_COUNT -gt 0 ]; then
            TOTAL_SIZE=$(du -sh "$LOCAL_CHECKPOINT_DIR" | cut -f1)
            echo -e "${GREEN}✓${NC} Downloaded $CHECKPOINT_COUNT checkpoints ($TOTAL_SIZE)"
        else
            echo -e "${YELLOW}⚠${NC} No checkpoint files found"
        fi
    else
        echo -e "${YELLOW}⚠${NC} Could not download checkpoints"
        echo "  Try manually: scp -P $SSH_PORT root@$SSH_HOST:/workspace/Othello/data/checkpoints/* ./"
    fi
    echo ""

    # Also download training log
    echo "Downloading training log..."
    if scp -P "$SSH_PORT" "root@$SSH_HOST:/workspace/train.log" "$LOCAL_CHECKPOINT_DIR/train.log" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Downloaded training log"
    else
        echo -e "${YELLOW}⚠${NC} Could not download training log"
    fi
    echo ""
fi

# Step 2: Calculate costs
echo "Step 2: Calculating costs..."
echo ""

START_TIME=$(echo "$INSTANCE_INFO" | jq -r '.start_date')
CURRENT_TIME=$(date +%s)

if [ -n "$START_TIME" ] && [ "$START_TIME" != "null" ]; then
    RUNTIME_SECONDS=$((CURRENT_TIME - START_TIME))
    RUNTIME_HOURS=$(echo "scale=2; $RUNTIME_SECONDS / 3600" | bc)
    ESTIMATED_COST=$(echo "scale=2; $RUNTIME_HOURS * $PRICE" | bc)

    echo "Runtime: ${RUNTIME_HOURS} hours"
    echo "Estimated cost: \$${ESTIMATED_COST}"
else
    echo "Could not calculate runtime"
fi
echo ""

# Step 3: Destroy instance
if [ $DESTROY_INSTANCE -eq 1 ]; then
    echo "Step 3: Destroying instance..."
    echo ""

    read -p "Are you sure you want to destroy instance $INSTANCE_ID? (yes/no): " confirm

    if [ "$confirm" == "yes" ]; then
        if vastai destroy instance $INSTANCE_ID 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Instance destroyed"
            rm "$INSTANCE_ID_FILE"
        else
            echo -e "${RED}✗${NC} Failed to destroy instance"
            echo "  Try manually: vastai destroy instance $INSTANCE_ID"
            exit 1
        fi
    else
        echo "Shutdown cancelled"
        exit 0
    fi
else
    echo "Step 3: Skipping instance destruction (--no-destroy)"
fi

echo ""
echo "==========================================="
echo "Shutdown complete!"
echo "==========================================="

if [ $DOWNLOAD_CHECKPOINTS -eq 1 ] && [ -d "$LOCAL_CHECKPOINT_DIR" ]; then
    echo ""
    echo "Checkpoints saved to:"
    echo "  $LOCAL_CHECKPOINT_DIR"
fi

echo ""
