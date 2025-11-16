# AWS Deployment Guide for Othello AlphaZero Training

Complete guide for deploying and running Othello AlphaZero training on AWS EC2 Spot instances using automated CLI tools.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Detailed Setup](#detailed-setup)
5. [Monitoring](#monitoring)
6. [Cost Management](#cost-management)
7. [Fault Tolerance](#fault-tolerance)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Overview

### What This Does

- **Automates** AWS EC2 Spot instance deployment for GPU-accelerated training
- **Reduces costs** by ~70% using Spot pricing vs On-Demand
- **Fault-tolerant** with automatic checkpoint saving
- **Zero-configuration** deployment with a single command
- **8× faster** training with parallel self-play on NVIDIA T4 GPU

### Architecture

```
Local Machine (macOS)
    ↓ (AWS CLI)
AWS EC2 g4dn.xlarge Spot Instance
    - Ubuntu 22.04
    - NVIDIA T4 GPU (16GB)
    - 4 vCPUs, 16GB RAM
    - Deep Learning AMI (PyTorch)
    ↓ (Auto-configured)
Training System
    - 8 parallel self-play workers (CUDA)
    - Batched MCTS inference
    - Checkpoint every iteration
    - TensorBoard monitoring
```

### Performance & Cost

| Metric | Local (M4 Max) | AWS (g4dn.xlarge) |
|--------|----------------|-------------------|
| **Iteration time** | ~14 minutes | ~2-3 minutes |
| **Speedup** | 1× | 6-8× |
| **Cost** | Free | ~$0.20/hour (Spot) |
| **100 iterations** | ~24 hours | ~5 hours (~$1.00) |

---

## Prerequisites

### 1. AWS Account Setup

✅ **AWS account** with billing enabled
✅ **IAM user** with programmatic access
✅ **Required IAM permissions**:
- `ec2:RunInstances`, `ec2:DescribeInstances`, `ec2:TerminateInstances`
- `ec2:CreateSecurityGroup`, `ec2:AuthorizeSecurityGroupIngress`
- `ec2:CreateKeyPair`, `ec2:DescribeKeyPairs`
- `ssm:GetParameter` (for AMI lookup)

### 2. Local Tools

✅ **AWS CLI v2** - Already installed (v2.31.27)
✅ **jq** - JSON processor for parsing AWS responses
```bash
brew install jq  # macOS
```

✅ **SSH client** - Already available on macOS

### 3. AWS CLI Configuration

Run `aws configure` and provide:
```bash
AWS Access Key ID: [your-access-key-id]
AWS Secret Access Key: [your-secret-access-key]
Default region name: us-east-1  # or preferred region
Default output format: json
```

**Test configuration:**
```bash
aws sts get-caller-identity
```

### 4. Verify Prerequisites

```bash
cd /Users/kstager/Desktop/Othello/aws
./check_prerequisites.sh
```

---

## Quick Start

### One-Command Deployment

```bash
cd /Users/kstager/Desktop/Othello/aws
./deploy_and_start.sh
```

This script automatically:
1. Checks prerequisites
2. Sets up AWS resources (one-time)
3. Launches Spot instance
4. Deploys code and starts training
5. Shows initial output

**That's it!** Training is now running on AWS.

---

## Detailed Setup

### Step 1: One-Time AWS Resource Setup

```bash
./setup_aws_resources.sh
```

Creates:
- **Security group** `othello-training-sg`
  - Allows SSH (port 22) from your IP only
- **SSH key pair** `othello-training-key`
  - Saves private key to `othello-training-key.pem`

**Configuration saved to:** `config.json`

### Step 2: Launch Spot Instance

```bash
./launch_spot_instance.sh
```

**What happens:**
1. Queries latest Deep Learning AMI with PyTorch
2. Requests g4dn.xlarge Spot instance
   - Max price cap: $0.40/hour (safety limit)
   - Current price: ~$0.15-0.25/hour
3. Attaches bootstrap script to auto-configure on boot
4. Waits for instance to be running
5. Returns instance ID and public IP

**Instance info saved to:** `instance_info.json`

### Step 3: Bootstrap (Automatic)

The instance automatically:
1. Activates PyTorch conda environment
2. Clones GitHub repository
3. Installs Python dependencies
4. Sets up Edax oracle
5. Starts Spot termination handler
6. Launches training in background

**Bootstrap time:** ~2-3 minutes

### Step 4: Verify Training Started

```bash
./monitor_training.sh
# Select option 1: Stream training logs
```

You should see:
```
=== Iteration 1 ===
  Game 1/100: BLACK wins +16 (50 moves, 2.1s)
  Game 2/100: WHITE wins +12 (52 moves, 2.2s)
  ...
```

---

## Monitoring

### Real-Time Log Streaming

```bash
./monitor_training.sh
```

**Options:**
1. **Stream training logs** - Live tail of train.log
2. **Show GPU usage** - nvidia-smi with auto-refresh
3. **Show training progress** - Checkpoints and iteration status
4. **Show TensorBoard URL** - Instructions for SSH tunnel
5. **Interactive SSH** - Full shell access
6. **All-in-one dashboard** - Combined view

### TensorBoard Access

**In one terminal:**
```bash
ssh -i aws/othello-training-key.pem -L 6006:localhost:6006 ubuntu@<PUBLIC_IP>
```

**In browser:**
```
http://localhost:6006
```

### Manual SSH Access

```bash
ssh -i aws/othello-training-key.pem ubuntu@<PUBLIC_IP>
cd Othello
tail -f train.log
```

---

## Cost Management

### Current Pricing

```bash
./estimate_costs.sh
```

**Sample Output:**
```
Current Spot price: $0.2000/hour
On-Demand price: $0.526/hour
Savings: 62%

Cost per iteration: $0.0100
Cost for 100 iterations: $1.00 (5 hours)
```

### Cost Optimization Tips

1. **Use Spot instances** (70% cheaper than On-Demand)
2. **Set max price cap** ($0.40/hour prevents price spikes)
3. **Monitor idle time** (shutdown when not training)
4. **Choose right region** (prices vary by region)
5. **Use gp3 storage** (cheaper than gp2)

### Example Scenarios

**Scenario A: Quick experiment (50 iterations)**
- Time: ~2.5 hours
- Cost: ~$0.50

**Scenario B: Full training run (500 iterations)**
- Time: ~25 hours
- Cost: ~$5.00

**Scenario C: Production training (2000 iterations)**
- Time: ~100 hours (~4 days)
- Cost: ~$20.00

---

## Fault Tolerance

### Spot Interruption Handling

**What happens if Spot instance is terminated:**

1. **2-minute warning** - AWS sends termination notice
2. **Automatic checkpoint** - Handler sends SIGTERM to training
3. **Graceful shutdown** - Training saves checkpoint and exits
4. **Resume training** - Just run `./deploy_and_start.sh` again
5. **Zero data loss** - Resumes from latest checkpoint

### Checkpoint System

- **Saved every iteration** - `data/checkpoints/current_iter_N.pt`
- **Includes optimizer state** - Momentum, LR schedule preserved
- **Automatic resume** - Training detects latest checkpoint on startup

### Manual Backup

**Download checkpoints before shutdown:**
```bash
./shutdown_instance.sh --download-checkpoints
```

**Saved to:** `downloaded_checkpoints_YYYYMMDD_HHMMSS/`

---

## Troubleshooting

### Training Not Starting

**Check bootstrap log:**
```bash
ssh -i aws/othello-training-key.pem ubuntu@<PUBLIC_IP>
cat /var/log/othello-bootstrap.log
```

**Common issues:**
- Edax binary not found: Check `third_party/edax/bin/edax` exists
- Python dependencies: Check pip install succeeded
- GPU not detected: Run `nvidia-smi` to verify GPU

### SSH Connection Refused

**Wait for instance to fully boot:**
```bash
aws ec2 describe-instance-status --instance-ids <INSTANCE_ID>
```

**Check security group:**
```bash
# Your IP may have changed
aws ec2 describe-security-groups --group-ids <SG_ID>
```

**Update security group with new IP:**
```bash
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
    --group-id <SG_ID> \
    --protocol tcp --port 22 --cidr ${MY_IP}/32
```

### High Costs

**Check current Spot price:**
```bash
./estimate_costs.sh
```

**If price spiked above $0.40:**
- Instance auto-terminates (max price cap)
- Checkpoint is saved
- Relaunch when prices drop

**Monitor active instances:**
```bash
aws ec2 describe-instances --filters "Name=instance-state-name,Values=running"
```

### Spot Instance Terminated Unexpectedly

**Check termination reason:**
```bash
aws ec2 describe-spot-instance-requests \
    --filters "Name=instance-id,Values=<INSTANCE_ID>"
```

**Common reasons:**
- Price exceeded max ($0.40)
- Capacity constraints in AZ
- AWS maintenance

**Solution:** Relaunch (checkpoints are saved)

---

## Best Practices

### Security

✅ **Limit SSH access to your IP only** (automatic in setup)
✅ **Use IMDSv2** for metadata access (enforced in launch)
✅ **Rotate SSH keys** periodically
✅ **Delete instances when done** (avoid forgotten instances)
✅ **Monitor AWS CloudTrail** for unexpected activity

### Performance

✅ **Use gp3 storage** (faster, cheaper than gp2)
✅ **Enable AMP** (automatic mixed precision) for NVIDIA
✅ **Batch size 128** for MCTS (optimal for T4)
✅ **8 workers** for g4dn.xlarge (matches vCPU count)
✅ **Monitor GPU usage** (`nvidia-smi`) to verify utilization

### Cost Control

✅ **Set billing alerts** in AWS console
✅ **Tag resources** (Name=othello-training)
✅ **Terminate when idle** (don't leave running overnight)
✅ **Use Spot for experiments** (70% cheaper)
✅ **Reserved instances for production** (if running 24/7)

### Data Management

✅ **Backup checkpoints** before termination
✅ **Delete old checkpoints** on instance (save space)
✅ **Use S3 for long-term storage** (optional)
✅ **Replay buffer auto-cleanup** (keeps last 3 shards)

---

## Shutdown and Cleanup

### Graceful Shutdown

```bash
./shutdown_instance.sh --download-checkpoints --terminate
```

**Steps:**
1. Downloads latest checkpoints to local machine
2. Downloads training logs
3. Downloads TensorBoard logs
4. Terminates instance
5. Cleans up `instance_info.json`

### Cost Report

```bash
./shutdown_instance.sh --cost-report
```

Shows:
- Total runtime
- Average Spot price
- Estimated total cost

### Manual Cleanup

**Delete security group:**
```bash
aws ec2 delete-security-group --group-id <SG_ID>
```

**Delete SSH key:**
```bash
aws ec2 delete-key-pair --key-name othello-training-key
rm aws/othello-training-key.pem
```

---

## Advanced Usage

### Custom Configuration

**Modify cloud config before deploying:**
```bash
vi config_cloud_aws.yaml

# Example changes:
num_workers: 12          # More parallelism
batch_size: 256          # Larger batches
games_per_iter: 200      # More data per iteration
```

### Multiple Regions

**Change default region:**
```bash
aws configure set region us-west-2
```

**Check prices in different regions:**
```bash
for region in us-east-1 us-west-2 eu-west-1; do
  price=$(aws ec2 describe-spot-price-history \
    --instance-types g4dn.xlarge \
    --product-descriptions "Linux/UNIX" \
    --max-results 1 \
    --region $region \
    --query 'SpotPriceHistory[0].SpotPrice' \
    --output text)
  echo "$region: \$$price/hour"
done
```

### S3 Checkpoint Backup (Optional)

**On instance, sync checkpoints to S3:**
```bash
aws s3 sync data/checkpoints s3://my-bucket/othello-checkpoints/
```

**In bootstrap.sh, add:**
```bash
# Sync from S3 before training
aws s3 sync s3://my-bucket/othello-checkpoints/ data/checkpoints/
```

---

## Support

### Documentation

- **AWS CLI Reference**: https://docs.aws.amazon.com/cli/
- **EC2 Spot Instances**: https://aws.amazon.com/ec2/spot/
- **Deep Learning AMI**: https://aws.amazon.com/machine-learning/amis/

### GitHub Issues

https://github.com/KenStager/Othello/issues

---

## Summary

### Complete Workflow

```bash
# One-time setup
cd /Users/kstager/Desktop/Othello/aws
./check_prerequisites.sh
./setup_aws_resources.sh

# Deploy and train (repeatable)
./deploy_and_start.sh

# Monitor
./monitor_training.sh

# Shutdown
./shutdown_instance.sh --download-checkpoints --terminate
```

### Key Files

- `check_prerequisites.sh` - Verify AWS CLI setup
- `setup_aws_resources.sh` - One-time resource creation
- `launch_spot_instance.sh` - Launch EC2 instance
- `bootstrap.sh` - Instance auto-configuration
- `deploy_and_start.sh` - Complete deployment workflow
- `monitor_training.sh` - Training monitoring
- `shutdown_instance.sh` - Terminate and cleanup
- `estimate_costs.sh` - Cost estimation
- `config.json` - AWS resource configuration
- `instance_info.json` - Current instance details

---

**Your training is now fully automated on AWS! 🚀☁️**
