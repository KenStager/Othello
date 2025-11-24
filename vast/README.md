# Vast.ai Deployment for Othello AlphaZero

Automated deployment scripts for running Othello AlphaZero training on Vast.ai GPU marketplace.

## Quick Start

```bash
# Complete deployment in one command
cd vast
./deploy.sh
```

Training will start on Vast.ai in ~3-5 minutes.

## What's Included

| Script | Purpose |
|--------|---------|
| `deploy.sh` | **START HERE** - Complete deployment workflow |
| `setup.sh` | Onstart script (runs inside container) |
| `monitor.sh` | Interactive monitoring dashboard |
| `tensorboard.sh` | SSH tunnel for TensorBoard access |
| `shutdown.sh` | Download checkpoints and destroy instance |

## Prerequisites

- Vast.ai account with API key
- Python with pip installed
- SSH client
- ~$0.30-0.50/hour budget for GPU

## Performance & Cost

### Performance Comparison

| Metric | Local (M4 Max) | AWS (g4dn.xlarge) | Vast.ai (RTX 3090) |
|--------|----------------|-------------------|-------------------|
| Iteration time | 14 min | 2-3 min | 2-3 min |
| 100 iterations | 24 hours | 5 hours | 5 hours |
| Cost | Free | ~$1.00 (Spot) | ~$1.50-2.50 |

### Cost Breakdown

**RTX 3090 On-Demand:**
- Price: ~$0.30-0.50/hour
- Per iteration: ~$0.015-0.025
- 100 iterations: ~$1.50-2.50

**RTX 4090 On-Demand:**
- Price: ~$0.45-0.65/hour
- Per iteration: ~$0.0225-0.0325
- 100 iterations: ~$2.25-3.25

**Hidden costs:**
- Storage: ~$0.01-0.02/GB/month (even when stopped)
- Bandwidth: Varies by host (check before renting)

## Detailed Workflow

### 1. Deploy

```bash
./deploy.sh
```

**What it does:**
1. Installs Vast.ai CLI (`pip install vastai`)
2. Sets your API key
3. Searches for GPUs (reliability > 0.95, 16GB+ RAM)
4. Displays top 5 options with pricing
5. Creates instance with PyTorch Docker image
6. Runs setup script (clone repo, install deps)
7. Starts training automatically
8. Shows SSH connection info

**Expected output:**
```
Instance ID: 1234567
Connect via SSH:
  ssh -p 12345 root@ssh.vast.ai
Monitor training:
  ./monitor.sh
```

### 2. Monitor

```bash
./monitor.sh
```

**Options:**
1. Stream training logs (tail -f)
2. GPU usage (nvidia-smi with live updates)
3. Training progress (checkpoint count)
4. Instance status
5. Interactive SSH session
6. All-in-one dashboard

**Example dashboard:**
```
[Instance Status]
Status: running
GPU: RTX 3090
Price: $0.35/hour

[GPU Usage]
GPU 0: RTX 3090 | Util: 95% | Mem: 18.2GB/24GB | Temp: 72°C

[Training Progress]
Latest iteration: iter_23
Total checkpoints: 23

[Recent Logs]
Iteration 23/2000 complete
Policy loss: 2.341
Value loss: 0.542
```

### 3. Shutdown

```bash
./shutdown.sh
```

**What it does:**
1. Downloads all checkpoints to local directory
2. Downloads training log
3. Calculates total runtime and cost
4. Destroys instance (with confirmation)
5. Cleans up instance tracking file

**Options:**
```bash
# Download but don't destroy
./shutdown.sh --no-destroy

# Destroy without downloading
./shutdown.sh --no-download
```

## TensorBoard Monitoring

Your training automatically logs comprehensive metrics to TensorBoard for real-time visualization.

### Quick Access

```bash
# Establish SSH tunnel to TensorBoard
./tensorboard.sh
```

Then open in your browser:
```
http://localhost:6006
```

### Metrics Available

Your training automatically logs:

**Training Metrics:**
- `train/loss` - Combined training loss
- `train/learning_rate` - Current learning rate

**Gating Metrics:**
- `gate/win_rate` - Win rate vs champion (promotion at ≥55%)
- `gate/loss_rate` - Loss rate during gating matches
- `gate/avg_moves` - Average game length
- `gate/avg_score_margin` - Score differential
- `gate/promoted` - Binary flag when model is promoted

**Diagnostics:**
- `diagnostics/value_correlation` - Value prediction accuracy
- `diagnostics/policy_entropy` - Policy diversity

**Data:**
- `data/buffer_size` - Replay buffer utilization

### Manual Tunnel (Alternative)

If you prefer to set up the SSH tunnel manually:

```bash
# Get instance SSH info
vastai ssh-url $(cat instance_id.txt)

# Establish tunnel (replace with actual port/host)
ssh -p 12345 root@ssh.vast.ai -L 6006:localhost:6006 -N
```

Keep this terminal open and access TensorBoard at `http://localhost:6006`

### TensorBoard Features

- **Real-time updates** - Metrics refresh every 30 seconds
- **Interactive charts** - Zoom, pan, hover for details
- **Compare runs** - View multiple training runs side-by-side
- **Smoothing** - Adjust smoothing slider for clearer trends
- **Download data** - Export metrics as CSV

### Troubleshooting

**TensorBoard not accessible?**

Check if it's running:
```bash
./monitor.sh
# Select option 7
```

**Manually start TensorBoard:**
```bash
# SSH into instance
vastai ssh-url $(cat instance_id.txt)

# Start TensorBoard
cd /workspace/Othello
nohup tensorboard --logdir=runs --port=6006 --host=0.0.0.0 &
```

## Manual Operations

### Search for GPUs

```bash
vastai search offers 'reliability > 0.95 num_gpus=1 gpu_ram>=16' -o 'dph_total+' | head -10
```

### Check instance status

```bash
vastai show instances
```

### Connect via SSH

```bash
# Get SSH command
vastai ssh-url <instance_id>

# Connect
ssh -p <port> root@ssh.vast.ai

# Navigate to training
cd /workspace/Othello
tail -f /workspace/train.log
```

### Stop vs Destroy

```bash
# Stop (pauses GPU, keeps storage)
vastai stop instance <id>

# Resume later
vastai start instance <id>

# Destroy permanently (irreversible!)
vastai destroy instance <id>
```

## Instance Selection Tips

### Recommended Filters

- **Reliability:** > 0.95 (avoid flaky hosts)
- **GPU RAM:** >= 16GB (for batch training)
- **Disk:** >= 50GB (for checkpoints)
- **CUDA:** >= 11.8 (PyTorch compatibility)

### GPU Recommendations

**Best Value:**
- RTX 3090 (24GB, ~$0.30-0.50/hr)
- RTX 4080 (16GB, ~$0.25-0.40/hr)

**High Performance:**
- RTX 4090 (24GB, ~$0.45-0.65/hr)
- A100 (40GB/80GB, ~$1.50-3.00/hr) - overkill for this project

**Avoid:**
- Consumer GPUs with < 16GB VRAM
- Hosts with reliability < 0.9
- Interruptible instances (training can be interrupted)

### Instance Type

**Always use On-Demand (not Interruptible):**
- Training runs for hours
- Interruption would lose progress
- On-demand ensures uninterrupted training

## Troubleshooting

### "No instances found"

**Solutions:**
- Relax filters (try reliability > 0.90)
- Check different GPU models
- Try different times (availability varies)
- Browse Vast.ai marketplace manually

### "Instance not starting"

**Check:**
1. Instance status: `vastai show instance <id>`
2. Wait longer (Docker pull can take 5-10 min)
3. Check Vast.ai dashboard for errors
4. Destroy and retry with different instance

### "Cannot connect via SSH"

**Solutions:**
1. Wait 2-3 minutes (instance still initializing)
2. Check instance status is "running"
3. Verify SSH port: `vastai ssh-url <id>`
4. Check firewall/network settings

### "Training not starting"

**Debug:**
```bash
# Connect to instance
ssh -p <port> root@ssh.vast.ai

# Check setup log
cat /tmp/setup.sh.log

# Check training log
tail -f /workspace/train.log

# Check if Python process is running
ps aux | grep python

# Manually start if needed
cd /workspace/Othello
python scripts/self_play_train.py --config config_cloud_aws.yaml
```

### "Checkpoints not downloading"

**Solutions:**
1. Verify instance still running
2. Check SSH connection
3. Manual download:
   ```bash
   scp -P <port> root@ssh.vast.ai:/workspace/Othello/data/checkpoints/* ./
   ```

## Advanced Usage

### Custom GPU Search

```bash
# Specific GPU model
vastai search offers 'gpu_name=RTX_4090' -o 'dph_total+'

# Price limit
vastai search offers 'dph_total < 0.40' -o 'reliability-'

# Multiple GPUs
vastai search offers 'num_gpus>=2' -o 'dph_total+'
```

### Custom Docker Image

Edit `deploy.sh` line with `--image`:
```bash
--image your-username/custom-othello:latest
```

### Different Config

Edit `setup.sh` to use different config:
```bash
python scripts/self_play_train.py --config your_config.yaml
```

### Persistent Volumes

For long-term storage:
```bash
# Create volume
vastai create volume --name othello-data --size 100

# Attach when creating instance
vastai create instance <offer_id> \
  --volume <volume_id>:/mnt/data \
  ...
```

## Cost Management

### Minimize Costs

1. **Destroy when done** - Don't leave idle
2. **Choose cheapest GPU** - Sort by price
3. **Download checkpoints** - Storage costs add up
4. **Monitor spending** - Check Vast.ai dashboard
5. **Set budget alerts** - In account settings

### Estimate Costs

**Before deploying:**
```bash
# Check current prices
vastai search offers 'gpu_name=RTX_3090' -o 'dph_total+' | head -1

# Calculate for your needs
# Example: 5 hours @ $0.35/hour = $1.75
```

## Architecture Notes

### Vast.ai vs AWS

| Aspect | AWS | Vast.ai |
|--------|-----|---------|
| Environment | EC2 VM | Docker container |
| Setup | User data script | Onstart script |
| User | ubuntu | root |
| Working dir | /home/ubuntu | /workspace |
| Persistence | EBS volumes | Ephemeral (48hr) |
| Networking | VPC/Security groups | Automatic |
| Pricing | Fixed + Spot | Marketplace |

### Docker Environment

- Base image: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
- PyTorch pre-installed at `/opt/conda`
- Working directory: `/workspace`
- Onstart script runs at `/tmp/setup.sh`
- Training log: `/workspace/train.log`

### Storage

**Instance storage (ephemeral):**
- Allocated at creation (default: 50GB)
- Deleted 48 hours after instance expires
- **Always download checkpoints before destroying!**

**Persistent volumes (optional):**
- Create separately
- Attach to instances
- Survives instance destruction
- Billed continuously

## Support

**Vast.ai Documentation:**
- https://vast.ai/docs

**Vast.ai CLI:**
```bash
vastai --help
vastai <command> --help
```

**Issues with this deployment:**
- https://github.com/KenStager/Othello/issues

## Quick Reference

```bash
# Deploy
./deploy.sh

# Monitor
./monitor.sh

# Shutdown
./shutdown.sh

# Manual SSH
vastai ssh-url $(cat instance_id.txt)

# Instance status
vastai show instance $(cat instance_id.txt)

# List all instances
vastai show instances

# Destroy instance
vastai destroy instance $(cat instance_id.txt)
```

---

**Ready to deploy?** Run `./deploy.sh` to get started!
