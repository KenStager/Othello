#!/bin/bash
# Check prerequisites for AWS EC2 deployment
# Verifies AWS CLI setup, credentials, and permissions

set -e

echo "=========================================="
echo "AWS Deployment Prerequisites Check"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SUCCESS=0

# 1. Check AWS CLI is installed
echo "1. Checking AWS CLI..."
if command -v aws &> /dev/null; then
    AWS_VERSION=$(aws --version 2>&1 | cut -d' ' -f1 | cut -d'/' -f2)
    echo -e "${GREEN}✓${NC} AWS CLI installed: v$AWS_VERSION"

    # Check if it's v2
    MAJOR_VERSION=$(echo $AWS_VERSION | cut -d'.' -f1)
    if [ "$MAJOR_VERSION" -ge 2 ]; then
        echo -e "${GREEN}✓${NC} Version 2.x detected (recommended)"
    else
        echo -e "${YELLOW}⚠${NC} Version 1.x detected. Consider upgrading to v2"
    fi
else
    echo -e "${RED}✗${NC} AWS CLI not found"
    echo "  Install: https://aws.amazon.com/cli/"
    SUCCESS=1
fi
echo ""

# 2. Check AWS credentials are configured
echo "2. Checking AWS credentials..."
if aws sts get-caller-identity &> /dev/null; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
    echo -e "${GREEN}✓${NC} AWS credentials configured"
    echo "  Account ID: $ACCOUNT_ID"
    echo "  User/Role: $USER_ARN"
else
    echo -e "${RED}✗${NC} AWS credentials not configured or invalid"
    echo "  Run: aws configure"
    echo "  You'll need:"
    echo "    - AWS Access Key ID"
    echo "    - AWS Secret Access Key"
    echo "    - Default region (e.g., us-east-1)"
    SUCCESS=1
fi
echo ""

# 3. Check default region is set
echo "3. Checking AWS region..."
AWS_REGION=$(aws configure get region)
if [ -n "$AWS_REGION" ]; then
    echo -e "${GREEN}✓${NC} Default region: $AWS_REGION"
else
    echo -e "${YELLOW}⚠${NC} No default region set"
    echo "  Run: aws configure set region us-east-1"
    echo "  (or your preferred region)"
fi
echo ""

# 4. Check required IAM permissions
echo "4. Checking IAM permissions..."
echo "   Testing EC2 permissions..."

# Test EC2 describe permission
if aws ec2 describe-instances --max-results 5 &> /dev/null; then
    echo -e "${GREEN}✓${NC} EC2 describe permission"
else
    echo -e "${RED}✗${NC} Missing EC2 permissions"
    echo "  Required: ec2:DescribeInstances, ec2:RunInstances, ec2:TerminateInstances"
    SUCCESS=1
fi

# Test SSM parameter access (for AMI lookup)
if aws ssm get-parameters --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id --region ${AWS_REGION:-us-east-1} &> /dev/null; then
    echo -e "${GREEN}✓${NC} SSM parameter access"
else
    echo -e "${YELLOW}⚠${NC} Cannot access SSM parameters (needed for AMI lookup)"
    echo "  May need: ssm:GetParameter permission"
fi
echo ""

# 5. Check GitHub repository access
echo "5. Checking GitHub repository..."
if git ls-remote https://github.com/KenStager/Othello.git &> /dev/null; then
    echo -e "${GREEN}✓${NC} GitHub repository accessible (public)"
else
    echo -e "${YELLOW}⚠${NC} Cannot access GitHub repository"
    echo "  Repo may be private or network issue"
    echo "  If private, you'll need to set up deploy keys on AWS instance"
fi
echo ""

# 6. Check SSH client
echo "6. Checking SSH client..."
if command -v ssh &> /dev/null; then
    SSH_VERSION=$(ssh -V 2>&1 | cut -d' ' -f1)
    echo -e "${GREEN}✓${NC} SSH installed: $SSH_VERSION"
else
    echo -e "${RED}✗${NC} SSH not found"
    echo "  Install SSH client for your system"
    SUCCESS=1
fi
echo ""

# 7. Check current Spot pricing
echo "7. Checking current g4dn.xlarge Spot pricing..."
if [ -n "$AWS_REGION" ]; then
    SPOT_PRICE=$(aws ec2 describe-spot-price-history \
        --instance-types g4dn.xlarge \
        --product-descriptions "Linux/UNIX" \
        --max-results 1 \
        --region ${AWS_REGION} \
        --query 'SpotPriceHistory[0].SpotPrice' \
        --output text 2>/dev/null || echo "N/A")

    if [ "$SPOT_PRICE" != "N/A" ]; then
        echo -e "${GREEN}✓${NC} Current Spot price in ${AWS_REGION}: \$$SPOT_PRICE/hour"

        # Calculate estimated costs
        COST_PER_ITER=$(echo "$SPOT_PRICE * 0.05" | bc -l)  # 3 min = 0.05 hours
        COST_100_ITERS=$(echo "$SPOT_PRICE * 5" | bc -l)    # 100 iters ≈ 5 hours

        printf "  Estimated cost per iteration: \$%.4f\n" $COST_PER_ITER
        printf "  Estimated cost for 100 iterations: \$%.2f\n" $COST_100_ITERS
    else
        echo -e "${YELLOW}⚠${NC} Could not fetch Spot price"
    fi
else
    echo -e "${YELLOW}⚠${NC} Skipped (no region configured)"
fi
echo ""

# Summary
echo "=========================================="
if [ $SUCCESS -eq 0 ]; then
    echo -e "${GREEN}✓ All prerequisite checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./setup_aws_resources.sh (one-time setup)"
    echo "  2. Run: ./deploy_and_start.sh (launch training)"
else
    echo -e "${RED}✗ Some checks failed${NC}"
    echo "  Please address the issues above before proceeding"
fi
echo "=========================================="

exit $SUCCESS
