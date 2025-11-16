#!/bin/bash
# Launch AWS EC2 g4dn.xlarge Spot instance for Othello training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.json"
INSTANCE_INFO_FILE="$SCRIPT_DIR/instance_info.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Launch Spot Instance"
echo "=========================================="
echo ""

# Load configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration not found"
    echo "Run: ./setup_aws_resources.sh first"
    exit 1
fi

AWS_REGION=$(jq -r '.region' "$CONFIG_FILE")
SG_ID=$(jq -r '.security_group_id' "$CONFIG_FILE")
KEY_NAME=$(jq -r '.key_name' "$CONFIG_FILE")

echo "Configuration loaded:"
echo "  Region: $AWS_REGION"
echo "  Security Group: $SG_ID"
echo "  Key Name: $KEY_NAME"
echo ""

# Query latest Deep Learning AMI
echo "1. Querying latest Deep Learning AMI..."
AMI_ID=$(aws ssm get-parameters \
    --names /aws/service/deep-learning-ami/pytorch-gpu-2-4/latest/ubuntu-22-04/x86_64 \
    --region $AWS_REGION \
    --query 'Parameters[0].Value' \
    --output text 2>/dev/null)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
    echo -e "${YELLOW}⚠${NC} Deep Learning AMI parameter not found, using fallback query..."
    # Fallback: Find any recent Deep Learning AMI with PyTorch
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu*" "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --region $AWS_REGION \
        --output text)
fi

echo -e "${GREEN}✓${NC} AMI ID: $AMI_ID"
echo ""

# Read bootstrap script
BOOTSTRAP_SCRIPT=$(cat "$SCRIPT_DIR/bootstrap.sh" | base64)

# Spot instance specification
SPOT_PRICE="0.40"  # Max price cap (safety limit)
INSTANCE_TYPE="g4dn.xlarge"

echo "2. Requesting Spot instance..."
echo "  Instance type: $INSTANCE_TYPE"
echo "  Max price: \$$SPOT_PRICE/hour"
echo ""

# Launch Spot instance
LAUNCH_RESULT=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --instance-market-options \
        "MarketType=spot,SpotOptions={MaxPrice=$SPOT_PRICE,SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}" \
    --block-device-mappings \
        "DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3,DeleteOnTermination=true}" \
    --metadata-options \
        "HttpTokens=required,HttpPutResponseHopLimit=1" \
    --user-data "$BOOTSTRAP_SCRIPT" \
    --tag-specifications \
        "ResourceType=instance,Tags=[{Key=Name,Value=othello-training},{Key=Project,Value=AlphaZero}]" \
    --region $AWS_REGION)

INSTANCE_ID=$(echo $LAUNCH_RESULT | jq -r '.Instances[0].InstanceId')

echo -e "${GREEN}✓${NC} Spot request accepted!"
echo "  Instance ID: $INSTANCE_ID"
echo ""

# Wait for instance to be running
echo "3. Waiting for instance to start (this may take 1-2 minutes)..."
aws ec2 wait instance-running \
    --instance-ids $INSTANCE_ID \
    --region $AWS_REGION

echo -e "${GREEN}✓${NC} Instance is running!"
echo ""

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $AWS_REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo -e "${GREEN}✓${NC} Public IP: $PUBLIC_IP"
echo ""

# Save instance info
cat > "$INSTANCE_INFO_FILE" <<EOF
{
  "instance_id": "$INSTANCE_ID",
  "public_ip": "$PUBLIC_IP",
  "instance_type": "$INSTANCE_TYPE",
  "region": "$AWS_REGION",
  "launch_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "ami_id": "$AMI_ID"
}
EOF

echo -e "${GREEN}✓${NC} Instance info saved to: $INSTANCE_INFO_FILE"
echo ""

# SSH connection info
KEY_FILE=$(jq -r '.key_file' "$CONFIG_FILE")

echo "=========================================="
echo -e "${GREEN}✓ Instance launched successfully!${NC}"
echo ""
echo "Instance details:"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo "  Type: $INSTANCE_TYPE"
echo ""
echo "Bootstrap is running (cloning repo, installing deps, starting training)..."
echo "This will take 2-3 minutes."
echo ""
echo "Next steps:"
echo "  1. Wait 3 minutes for bootstrap to complete"
echo "  2. Monitor training:"
echo "       ssh -i $KEY_FILE ubuntu@$PUBLIC_IP tail -f train.log"
echo "  3. Or use: ./monitor_training.sh"
echo ""
echo "Spot instance will auto-save checkpoints every iteration."
echo "=========================================="
