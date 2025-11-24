#!/bin/bash
# Vast.ai TensorBoard SSH Tunnel Helper
# Establishes SSH tunnel for TensorBoard access

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_ID_FILE="$SCRIPT_DIR/instance_id.txt"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "==========================================="
echo "TensorBoard SSH Tunnel"
echo "==========================================="
echo ""

if [ ! -f "$INSTANCE_ID_FILE" ]; then
    echo -e "${RED}✗${NC} No running instance found"
    echo "  Deploy first with: ./deploy.sh"
    exit 1
fi

INSTANCE_ID=$(cat "$INSTANCE_ID_FILE")

# Get SSH connection info
SSH_INFO=$(vastai ssh-url $INSTANCE_ID 2>/dev/null)

if [ -z "$SSH_INFO" ]; then
    echo -e "${RED}✗${NC} Could not get SSH connection info"
    echo "  Instance may have been destroyed or is not ready"
    echo "  Check status: vastai show instance $INSTANCE_ID"
    exit 1
fi

# Parse SSH URL format: ssh://root@host:port
# Extract host and port from URL
SSH_HOST=$(echo "$SSH_INFO" | sed -E 's|ssh://(.*)@([^:]+):([0-9]+)|\2|')
SSH_PORT=$(echo "$SSH_INFO" | sed -E 's|ssh://(.*)@([^:]+):([0-9]+)|\3|')
SSH_USER=$(echo "$SSH_INFO" | sed -E 's|ssh://(.*)@([^:]+):([0-9]+)|\1|')

# Find SSH key
SSH_KEY=""
if [ -f "$HOME/.ssh/id_ed25519_vastai" ]; then
    SSH_KEY="$HOME/.ssh/id_ed25519_vastai"
elif [ -f "$HOME/.ssh/id_rsa" ]; then
    SSH_KEY="$HOME/.ssh/id_rsa"
fi

echo -e "${GREEN}✓${NC} Establishing SSH tunnel to TensorBoard..."
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Connection: $SSH_USER@$SSH_HOST:$SSH_PORT"
echo "Tunnel: localhost:6006 → instance:16006 (TensorBoard)"
echo ""
echo -e "${BLUE}TensorBoard will be available at:${NC}"
echo "  http://localhost:6006"
echo ""
echo "Press Ctrl+C to disconnect"
echo "==========================================="
echo ""

# Establish SSH tunnel (blocking, -N means no remote command)
# Note: Vast.ai runs TensorBoard on port 16006, not 6006
if [ -n "$SSH_KEY" ]; then
    ssh -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" -L 6006:localhost:16006 -N
else
    ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" -L 6006:localhost:16006 -N
fi
