#!/bin/bash
# IAM Permission Setup Guide for Othello AWS Deployment
# This script provides instructions and automation for setting up IAM permissions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_FILE="$SCRIPT_DIR/iam_policy.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "IAM Permission Setup for Othello AWS"
echo "=========================================="
echo ""

# Check if policy file exists
if [ ! -f "$POLICY_FILE" ]; then
    echo "Error: IAM policy file not found: $POLICY_FILE"
    exit 1
fi

echo "This script will help you set up IAM permissions for AWS deployment."
echo ""
echo "You have three options:"
echo "  1) Create a new IAM user with required permissions (recommended)"
echo "  2) Add permissions to existing IAM user"
echo "  3) Show instructions for manual setup via AWS Console"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Creating new IAM user with permissions...${NC}"
        echo ""

        # Get username
        read -p "Enter new IAM username (e.g., othello-training-user): " USERNAME

        if [ -z "$USERNAME" ]; then
            echo "Error: Username cannot be empty"
            exit 1
        fi

        echo ""
        echo "Step 1: Creating IAM user..."

        # Create user
        if aws iam create-user --user-name "$USERNAME" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Created user: $USERNAME"
        else
            echo -e "${YELLOW}⚠${NC} User may already exist, continuing..."
        fi

        echo ""
        echo "Step 2: Creating IAM policy..."

        # Create policy
        POLICY_NAME="OthelloTrainingPolicy"
        POLICY_ARN=$(aws iam create-policy \
            --policy-name "$POLICY_NAME" \
            --policy-document file://"$POLICY_FILE" \
            --query 'Policy.Arn' \
            --output text 2>/dev/null || \
            aws iam list-policies --query "Policies[?PolicyName=='$POLICY_NAME'].Arn" --output text)

        if [ -n "$POLICY_ARN" ]; then
            echo -e "${GREEN}✓${NC} Policy ARN: $POLICY_ARN"
        else
            echo -e "${YELLOW}⚠${NC} Could not create or find policy"
        fi

        echo ""
        echo "Step 3: Attaching policy to user..."

        # Attach policy to user
        if aws iam attach-user-policy \
            --user-name "$USERNAME" \
            --policy-arn "$POLICY_ARN" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Policy attached to user"
        else
            echo -e "${YELLOW}⚠${NC} Could not attach policy (may already be attached)"
        fi

        echo ""
        echo "Step 4: Creating access key..."

        # Create access key
        ACCESS_KEY_OUTPUT=$(aws iam create-access-key --user-name "$USERNAME" 2>/dev/null)

        if [ -n "$ACCESS_KEY_OUTPUT" ]; then
            ACCESS_KEY_ID=$(echo "$ACCESS_KEY_OUTPUT" | jq -r '.AccessKey.AccessKeyId')
            SECRET_ACCESS_KEY=$(echo "$ACCESS_KEY_OUTPUT" | jq -r '.AccessKey.SecretAccessKey')

            echo -e "${GREEN}✓${NC} Access key created"
            echo ""
            echo "=========================================="
            echo -e "${YELLOW}IMPORTANT: Save these credentials securely${NC}"
            echo "=========================================="
            echo "Access Key ID: $ACCESS_KEY_ID"
            echo "Secret Access Key: $SECRET_ACCESS_KEY"
            echo ""
            echo "Configure AWS CLI with these credentials:"
            echo "  aws configure"
            echo ""
            echo "Or set environment variables:"
            echo "  export AWS_ACCESS_KEY_ID=$ACCESS_KEY_ID"
            echo "  export AWS_SECRET_ACCESS_KEY=$SECRET_ACCESS_KEY"
            echo "=========================================="

            # Save to file
            CREDS_FILE="$SCRIPT_DIR/credentials_${USERNAME}.txt"
            cat > "$CREDS_FILE" <<EOF
AWS Credentials for $USERNAME
Created: $(date)

Access Key ID: $ACCESS_KEY_ID
Secret Access Key: $SECRET_ACCESS_KEY

Configure with:
  aws configure

Or use environment variables:
  export AWS_ACCESS_KEY_ID=$ACCESS_KEY_ID
  export AWS_SECRET_ACCESS_KEY=$SECRET_ACCESS_KEY
EOF
            chmod 600 "$CREDS_FILE"
            echo ""
            echo -e "${GREEN}✓${NC} Credentials saved to: $CREDS_FILE (keep secure!)"
        else
            echo -e "${YELLOW}⚠${NC} Could not create access key"
            echo "Create manually with: aws iam create-access-key --user-name $USERNAME"
        fi

        echo ""
        echo -e "${GREEN}✓ IAM user setup complete!${NC}"
        ;;

    2)
        echo ""
        echo -e "${BLUE}Adding permissions to existing user...${NC}"
        echo ""

        # List existing users
        echo "Existing IAM users:"
        aws iam list-users --query 'Users[].UserName' --output table
        echo ""

        read -p "Enter username to add permissions to: " USERNAME

        if [ -z "$USERNAME" ]; then
            echo "Error: Username cannot be empty"
            exit 1
        fi

        # Verify user exists
        if ! aws iam get-user --user-name "$USERNAME" &>/dev/null; then
            echo "Error: User $USERNAME not found"
            exit 1
        fi

        echo ""
        echo "Step 1: Creating/finding policy..."

        POLICY_NAME="OthelloTrainingPolicy"
        POLICY_ARN=$(aws iam create-policy \
            --policy-name "$POLICY_NAME" \
            --policy-document file://"$POLICY_FILE" \
            --query 'Policy.Arn' \
            --output text 2>/dev/null || \
            aws iam list-policies --query "Policies[?PolicyName=='$POLICY_NAME'].Arn" --output text)

        if [ -n "$POLICY_ARN" ]; then
            echo -e "${GREEN}✓${NC} Policy ARN: $POLICY_ARN"
        else
            echo "Error: Could not create or find policy"
            exit 1
        fi

        echo ""
        echo "Step 2: Attaching policy to user..."

        if aws iam attach-user-policy \
            --user-name "$USERNAME" \
            --policy-arn "$POLICY_ARN"; then
            echo -e "${GREEN}✓${NC} Policy attached successfully"
        else
            echo -e "${YELLOW}⚠${NC} Could not attach policy (may already be attached)"
        fi

        echo ""
        echo -e "${GREEN}✓ Permissions added to user: $USERNAME${NC}"
        ;;

    3)
        echo ""
        echo -e "${BLUE}Manual Setup Instructions (AWS Console)${NC}"
        echo ""
        echo "=========================================="
        echo "1. Go to AWS Console: https://console.aws.amazon.com/iam/"
        echo ""
        echo "2. Create new IAM user:"
        echo "   - Click 'Users' → 'Add users'"
        echo "   - Username: othello-training-user"
        echo "   - Access type: Programmatic access"
        echo "   - Click 'Next'"
        echo ""
        echo "3. Set permissions:"
        echo "   - Click 'Attach policies directly'"
        echo "   - Click 'Create policy'"
        echo "   - Switch to 'JSON' tab"
        echo "   - Copy contents from: $POLICY_FILE"
        echo "   - Paste into the editor"
        echo "   - Click 'Review policy'"
        echo "   - Name: OthelloTrainingPolicy"
        echo "   - Click 'Create policy'"
        echo ""
        echo "4. Attach policy to user:"
        echo "   - Go back to user creation"
        echo "   - Refresh policies"
        echo "   - Search for 'OthelloTrainingPolicy'"
        echo "   - Check the box"
        echo "   - Click 'Next' → 'Create user'"
        echo ""
        echo "5. Save credentials:"
        echo "   - Download CSV or copy Access Key ID and Secret Access Key"
        echo "   - IMPORTANT: Save these securely!"
        echo ""
        echo "6. Configure AWS CLI:"
        echo "   Run: aws configure"
        echo "   Enter your Access Key ID and Secret Access Key"
        echo ""
        echo "=========================================="
        echo ""
        echo "Policy file location: $POLICY_FILE"
        echo ""
        echo "After setup, run: ./check_prerequisites.sh"
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Next steps:"
echo "  1. Verify setup: ./check_prerequisites.sh"
echo "  2. Deploy to AWS: ./deploy_and_start.sh"
