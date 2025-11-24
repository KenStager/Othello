#!/bin/bash
# Vast.ai Deployment Script for Othello AlphaZero Training
# Complete automated deployment workflow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAST_API_KEY="5757d13648f3ae8b6e70e8fccb97af6b47a59289da5a20d123e7a75a19cda503"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "==========================================="
echo "Othello AlphaZero - Vast.ai Deployment"
echo "==========================================="
echo ""

# Step 1: Install/verify Vast.ai CLI
echo "Step 1: Checking Vast.ai CLI..."
if ! command -v vastai &> /dev/null; then
    echo "  Installing Vast.ai CLI..."
    pip install vastai --quiet
    echo -e "${GREEN}✓${NC} Vast.ai CLI installed"
else
    VAST_VERSION=$(vastai --version 2>&1 | head -1 || echo "unknown")
    echo -e "${GREEN}✓${NC} Vast.ai CLI already installed: $VAST_VERSION"
fi
echo ""

# Step 2: Set API key
echo "Step 2: Setting API key..."
vastai set api-key $VAST_API_KEY > /dev/null 2>&1
echo -e "${GREEN}✓${NC} API key configured"
echo ""

# Step 3: Search for GPU instances
echo "Step 3: Searching for available GPU instances..."
echo "  Filters: ≤$0.15/hr, US/EU only, reliability > 0.95, GPU RAM >= 16GB, disk >= 50GB"
echo "  Sort: Best DL Performance/$ (primary), CPU cores (secondary)"
echo ""

# Search for instances and parse results
SEARCH_JSON=$(vastai search offers 'dph <= 0.15 geolocation in [US,CA,GB,DE,FR,IT,ES,NL,BE,AT,SE,NO,DK,IE,CH,CZ,PL] reliability > 0.95 num_gpus=1 gpu_ram>=16 disk_space>=50' -o 'dlperf_usd-,num_cpus-' --raw 2>/dev/null)

if [ -z "$SEARCH_JSON" ] || [ "$SEARCH_JSON" == "[]" ]; then
    echo -e "${RED}✗${NC} No instances found matching criteria"
    echo "Try relaxing filters or check Vast.ai marketplace manually"
    exit 1
fi

# Display results
echo "Top 5 available instances:"
echo "----------------------------------------------------------------------------------------------"
printf "%-8s %-15s %-8s %-8s %-6s %-10s %-8s %-8s\n" "ID" "GPU" "$/hr" "Perf/$" "CPUs" "CUDA" "RAM" "Reliability"
echo "----------------------------------------------------------------------------------------------"

OFFER_IDS=()
COUNT=0
while IFS= read -r line; do
    [ $COUNT -ge 5 ] && break

    OFFER_ID=$(echo "$line" | jq -r '.id')
    GPU_NAME=$(echo "$line" | jq -r '.gpu_name' | sed 's/_/ /g')
    PRICE=$(echo "$line" | jq -r '.dph_total // 0')
    DLPERF_USD=$(echo "$line" | jq -r '.dlperf_usd // 0')
    CPU_CORES=$(echo "$line" | jq -r '.num_cpus // 0')
    CUDA_VER=$(echo "$line" | jq -r '.cuda_max_good // "N/A"')
    GPU_RAM=$(echo "$line" | jq -r '.gpu_ram // 0')
    RELIABILITY=$(echo "$line" | jq -r '.reliability2 // 0')

    printf "%-8s %-15s %-8.3f %-8.2f %-6s %-10s %-8.0fGB %-8.2f\n" "$OFFER_ID" "$GPU_NAME" "$PRICE" "$DLPERF_USD" "$CPU_CORES" "$CUDA_VER" "$GPU_RAM" "$RELIABILITY"
    OFFER_IDS+=("$OFFER_ID")
    COUNT=$((COUNT + 1))
done < <(echo "$SEARCH_JSON" | jq -c '.[]')

echo "----------------------------------------------------------------------------------------------"
echo ""

# Step 4: Select instance (use best performance/$ by default)
SELECTED_OFFER="${OFFER_IDS[0]}"

echo "Step 4: Select instance"
echo "  Auto-selecting best performance/$ instance (≤$0.15/hr): ID $SELECTED_OFFER"
echo "  (or Ctrl+C and edit script to choose different instance)"
echo ""
sleep 2

# Step 5: Create instance
echo "Step 5: Creating instance..."
echo "  Image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
echo "  Disk: 50GB"
echo "  Onstart script: $SCRIPT_DIR/setup.sh"
echo ""

# Upload setup script to a temporary accessible location
# For now, use raw GitHub URL (requires pushing setup.sh to repo first)
# Alternative: Use vastai's --onstart-cmd with inline script
SETUP_URL="https://raw.githubusercontent.com/KenStager/Othello/main/vast/setup.sh"

CREATE_RESULT=$(vastai create instance $SELECTED_OFFER \
    --image pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime \
    --disk 50 \
    --ssh \
    --onstart-cmd "wget -q $SETUP_URL -O /tmp/setup.sh && chmod +x /tmp/setup.sh && bash /tmp/setup.sh || (cd /workspace && git clone https://github.com/KenStager/Othello.git && cd Othello && pip install pyyaml matplotlib pandas && chmod +x third_party/edax/bin/edax && mkdir -p data/checkpoints data/replay logs runs && nohup python -u scripts/self_play_train.py --config config_cloud_aws.yaml > /workspace/train.log 2>&1 &)" \
    --env 'TZ=UTC' \
    --raw \
    2>&1)

INSTANCE_ID=$(echo "$CREATE_RESULT" | jq -r '.new_contract' 2>/dev/null || echo "")

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" == "null" ]; then
    echo -e "${RED}✗${NC} Failed to create instance"
    echo "Error: $CREATE_RESULT"
    exit 1
fi

echo -e "${GREEN}✓${NC} Instance created!"
echo "  Instance ID: $INSTANCE_ID"
echo ""

# Save instance info
echo "$INSTANCE_ID" > "$SCRIPT_DIR/instance_id.txt"

# Step 6: Wait for instance to be ready
echo "Step 6: Waiting for instance to be ready..."
echo "  This may take 2-5 minutes (Docker image pull + setup)"
echo ""

MAX_WAIT=300  # 5 minutes
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    INSTANCE_STATUS=$(vastai show instance $INSTANCE_ID --raw 2>/dev/null | jq -r '.actual_status' || echo "unknown")

    if [ "$INSTANCE_STATUS" == "running" ]; then
        echo -e "${GREEN}✓${NC} Instance is running!"
        break
    fi

    echo "  Status: $INSTANCE_STATUS (waited ${WAITED}s)"
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${YELLOW}⚠${NC} Instance taking longer than expected"
    echo "  Check status with: vastai show instance $INSTANCE_ID"
fi

echo ""

# Step 7: Get SSH connection info
echo "Step 7: Getting SSH connection info..."
SSH_INFO=$(vastai ssh-url $INSTANCE_ID 2>/dev/null)

if [ -n "$SSH_INFO" ]; then
    echo -e "${GREEN}✓${NC} SSH connection ready"
    echo ""
    echo "==========================================="
    echo "INSTANCE READY"
    echo "==========================================="
    echo ""
    echo "Instance ID: $INSTANCE_ID"
    echo ""
    echo "Connect via SSH:"
    echo "  $SSH_INFO"
    echo ""
    echo "Monitor training:"
    echo "  ./monitor.sh"
    echo ""
    echo "Access TensorBoard:"
    echo "  ./tensorboard.sh"
    echo "  Then open: http://localhost:6006"
    echo ""
    echo "Or manually:"
    echo "  $SSH_INFO"
    echo "  cd /workspace/Othello"
    echo "  tail -f /workspace/train.log"
    echo ""
    echo "Shutdown and retrieve checkpoints:"
    echo "  ./shutdown.sh"
    echo ""
    echo "==========================================="
else
    echo -e "${YELLOW}⚠${NC} Could not get SSH info yet"
    echo "  Wait a minute and try: vastai ssh-url $INSTANCE_ID"
fi

echo ""
echo "Waiting 30 seconds for setup to complete..."
sleep 30

# Step 8: Show initial training output
echo ""
echo "Step 8: Checking training startup..."
if [ -n "$SSH_INFO" ]; then
    SSH_CMD=$(echo "$SSH_INFO" | sed 's/ssh //')
    echo "Attempting to show initial training output..."
    ssh $SSH_CMD "tail -20 /workspace/train.log 2>/dev/null || echo 'Training log not ready yet. Wait a minute and check manually.'" || echo "Could not connect yet"
fi

echo ""
echo "==========================================="
echo "Deployment complete!"
echo "==========================================="
