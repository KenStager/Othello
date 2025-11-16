#!/bin/bash
# One-time AWS resource setup for Othello training
# Creates: security group, SSH key pair, saves configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "AWS Resources Setup (One-Time)"
echo "=========================================="
echo ""

# Get AWS region
AWS_REGION=$(aws configure get region)
if [ -z "$AWS_REGION" ]; then
    echo "No default region set. Using us-east-1"
    AWS_REGION="us-east-1"
fi
echo "Region: $AWS_REGION"
echo ""

# 1. Create Security Group
echo "1. Creating security group..."
SG_NAME="othello-training-sg"
SG_DESC="Security group for Othello AlphaZero training instances"

# Check if security group already exists
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region $AWS_REGION 2>/dev/null || echo "")

if [ "$SG_ID" != "" ] && [ "$SG_ID" != "None" ]; then
    echo -e "${YELLOW}⚠${NC} Security group already exists: $SG_ID"
else
    # Create security group
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SG_NAME \
        --description "$SG_DESC" \
        --region $AWS_REGION \
        --query 'GroupId' \
        --output text)

    echo -e "${GREEN}✓${NC} Created security group: $SG_ID"

    # Get your public IP
    echo "  Detecting your public IP..."
    MY_IP=$(curl -s https://checkip.amazonaws.com || echo "0.0.0.0")
    echo "  Your IP: $MY_IP"

    # Add SSH rule
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr ${MY_IP}/32 \
        --region $AWS_REGION \
        --output text

    echo -e "${GREEN}✓${NC} Added SSH rule for your IP: ${MY_IP}/32"
fi
echo ""

# 2. Create SSH Key Pair
echo "2. Creating SSH key pair..."
KEY_NAME="othello-training-key"
KEY_FILE="$SCRIPT_DIR/${KEY_NAME}.pem"

# Check if key already exists in AWS
KEY_EXISTS=$(aws ec2 describe-key-pairs \
    --key-names $KEY_NAME \
    --region $AWS_REGION \
    --query 'KeyPairs[0].KeyName' \
    --output text 2>/dev/null || echo "")

if [ "$KEY_EXISTS" == "$KEY_NAME" ]; then
    echo -e "${YELLOW}⚠${NC} Key pair already exists in AWS: $KEY_NAME"

    if [ ! -f "$KEY_FILE" ]; then
        echo -e "${YELLOW}⚠${NC} Local key file not found: $KEY_FILE"
        echo "  If you have the key file, move it to: $KEY_FILE"
        echo "  Otherwise, delete the key from AWS and re-run this script"
    else
        echo -e "${GREEN}✓${NC} Local key file exists: $KEY_FILE"
    fi
else
    # Create new key pair
    aws ec2 create-key-pair \
        --key-name $KEY_NAME \
        --region $AWS_REGION \
        --query 'KeyMaterial' \
        --output text > "$KEY_FILE"

    chmod 400 "$KEY_FILE"
    echo -e "${GREEN}✓${NC} Created SSH key pair: $KEY_NAME"
    echo -e "${GREEN}✓${NC} Saved to: $KEY_FILE"
fi
echo ""

# 3. Save configuration
echo "3. Saving configuration..."
cat > "$CONFIG_FILE" <<EOF
{
  "region": "$AWS_REGION",
  "security_group_id": "$SG_ID",
  "security_group_name": "$SG_NAME",
  "key_name": "$KEY_NAME",
  "key_file": "$KEY_FILE",
  "setup_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo -e "${GREEN}✓${NC} Configuration saved to: $CONFIG_FILE"
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Created resources:"
echo "  Security Group: $SG_NAME ($SG_ID)"
echo "  SSH Key Pair: $KEY_NAME"
echo "  Key File: $KEY_FILE"
echo ""
echo "Next step:"
echo "  Run: ./launch_spot_instance.sh"
echo "=========================================="
