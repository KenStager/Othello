#!/bin/bash
# Estimate AWS costs for Othello training
# Queries current Spot prices and calculates estimated costs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "==========================================="
echo "AWS Cost Estimation for Othello Training"
echo "==========================================="
echo ""

# Get region
if [ -f "$CONFIG_FILE" ]; then
    AWS_REGION=$(jq -r '.region' "$CONFIG_FILE")
else
    AWS_REGION=$(aws configure get region)
    if [ -z "$AWS_REGION" ]; then
        AWS_REGION="us-east-1"
    fi
fi

echo "Region: $AWS_REGION"
echo ""

# Instance type
INSTANCE_TYPE="g4dn.xlarge"

# Query current Spot price
echo "1. Current Spot Pricing"
echo "-------------------------------------------"
CURRENT_SPOT=$(aws ec2 describe-spot-price-history \
    --instance-types $INSTANCE_TYPE \
    --product-descriptions "Linux/UNIX" \
    --max-results 1 \
    --region $AWS_REGION \
    --query 'SpotPriceHistory[0].SpotPrice' \
    --output text)

ON_DEMAND_PRICE="0.526"  # g4dn.xlarge on-demand price

printf "Current Spot price: \$%.4f/hour\n" $CURRENT_SPOT
printf "On-Demand price: \$%.3f/hour\n" $ON_DEMAND_PRICE

SAVINGS=$(echo "scale=1; (1 - $CURRENT_SPOT / $ON_DEMAND_PRICE) * 100" | bc)
printf "Savings: %.0f%%\n" $SAVINGS
echo ""

# Spot price history (last 24 hours)
echo "2. Spot Price History (last 24 hours)"
echo "-------------------------------------------"
PRICES=$(aws ec2 describe-spot-price-history \
    --instance-types $INSTANCE_TYPE \
    --product-descriptions "Linux/UNIX" \
    --start-time $(date -u -v-24H +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) \
    --region $AWS_REGION \
    --query 'SpotPriceHistory[*].SpotPrice' \
    --output text)

if [ -n "$PRICES" ]; then
    AVG_PRICE=$(echo "$PRICES" | tr '\t' '\n' | awk '{sum+=$1; count++} END {printf "%.4f", sum/count}')
    MIN_PRICE=$(echo "$PRICES" | tr '\t' '\n' | sort -n | head -1)
    MAX_PRICE=$(echo "$PRICES" | tr '\t' '\n' | sort -n | tail -1)

    printf "Average: \$%.4f/hour\n" $AVG_PRICE
    printf "Min: \$%.4f/hour\n" $MIN_PRICE
    printf "Max: \$%.4f/hour\n" $MAX_PRICE
else
    AVG_PRICE=$CURRENT_SPOT
    echo "No price history available, using current price"
fi
echo ""

# Training cost estimates
echo "3. Training Cost Estimates"
echo "-------------------------------------------"
echo "Assumptions:"
echo "  - Iteration time: 3 minutes (with 8 workers)"
echo "  - Using average Spot price: \$$AVG_PRICE/hour"
echo ""

# Per iteration
ITER_TIME_HOURS=$(echo "3 / 60" | bc -l)  # 3 minutes
COST_PER_ITER=$(echo "$ITER_TIME_HOURS * $AVG_PRICE" | bc -l)
printf "Cost per iteration: \$%.4f\n" $COST_PER_ITER

# 10 iterations
COST_10=$(echo "$COST_PER_ITER * 10" | bc -l)
printf "Cost for 10 iterations: \$%.3f (30 minutes)\n" $COST_10

# 100 iterations
COST_100=$(echo "$COST_PER_ITER * 100" | bc -l)
HOURS_100=$(echo "$ITER_TIME_HOURS * 100" | bc -l)
printf "Cost for 100 iterations: \$%.2f (%.1f hours)\n" $COST_100 $HOURS_100

# 1000 iterations
COST_1000=$(echo "$COST_PER_ITER * 1000" | bc -l)
HOURS_1000=$(echo "$ITER_TIME_HOURS * 1000" | bc -l)
printf "Cost for 1000 iterations: \$%.2f (%.0f hours)\n" $COST_1000 $HOURS_1000
echo ""

# Storage costs
echo "4. Storage Costs"
echo "-------------------------------------------"
STORAGE_GB=30
STORAGE_COST_PER_GB=0.10  # gp3 per GB-month

MONTHLY_STORAGE=$(echo "$STORAGE_GB * $STORAGE_COST_PER_GB" | bc -l)
printf "EBS (30GB gp3): \$%.2f/month\n" $MONTHLY_STORAGE
printf "Daily storage cost: \$%.3f\n" $(echo "$MONTHLY_STORAGE / 30" | bc -l)
echo ""

# Total estimates
echo "5. Complete Training Scenarios"
echo "-------------------------------------------"

echo "Scenario A: Quick experiment (50 iterations)"
COMPUTE_A=$(echo "$COST_PER_ITER * 50" | bc -l)
HOURS_A=$(echo "$ITER_TIME_HOURS * 50" | bc -l)
STORAGE_A=$(echo "$MONTHLY_STORAGE / 30 * 1" | bc -l)  # 1 day
TOTAL_A=$(echo "$COMPUTE_A + $STORAGE_A" | bc -l)
printf "  Compute: \$%.2f (%.1f hours)\n" $COMPUTE_A $HOURS_A
printf "  Storage: \$%.2f (1 day)\n" $STORAGE_A
printf "  Total: \$%.2f\n" $TOTAL_A
echo ""

echo "Scenario B: Full training run (500 iterations)"
COMPUTE_B=$(echo "$COST_PER_ITER * 500" | bc -l)
HOURS_B=$(echo "$ITER_TIME_HOURS * 500" | bc -l)
DAYS_B=$(echo "$HOURS_B / 24" | bc -l)
STORAGE_B=$(echo "$MONTHLY_STORAGE / 30 * $DAYS_B" | bc -l)
TOTAL_B=$(echo "$COMPUTE_B + $STORAGE_B" | bc -l)
printf "  Compute: \$%.2f (%.0f hours = %.1f days)\n" $COMPUTE_B $HOURS_B $DAYS_B
printf "  Storage: \$%.2f\n" $STORAGE_B
printf "  Total: \$%.2f\n" $TOTAL_B
echo ""

# Comparison with on-demand
echo "6. Spot vs On-Demand Comparison (100 iterations)"
echo "-------------------------------------------"
SPOT_COST_100=$(echo "$COST_PER_ITER * 100" | bc -l)
ON_DEMAND_COST_100=$(echo "$HOURS_100 * $ON_DEMAND_PRICE" | bc -l)
SAVINGS_AMOUNT=$(echo "$ON_DEMAND_COST_100 - $SPOT_COST_100" | bc -l)

printf "Spot cost: \$%.2f\n" $SPOT_COST_100
printf "On-Demand cost: \$%.2f\n" $ON_DEMAND_COST_100
printf "You save: \$%.2f (%.0f%%)\n" $SAVINGS_AMOUNT $SAVINGS
echo ""

echo "==========================================="
echo "Notes:"
echo "  - Spot prices vary by region and time"
echo "  - Set max price cap to avoid price spikes"
echo "  - Checkpoints save every iteration (fault-tolerant)"
echo "  - Data transfer costs not included (minimal)"
echo "==========================================="
