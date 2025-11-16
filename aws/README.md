# AWS Deployment Automation for Othello AlphaZero

Automated deployment scripts for running Othello AlphaZero training on AWS EC2 Spot instances.

## Quick Start

```bash
# Complete deployment in one command
./deploy_and_start.sh
```

That's it! Training will start on AWS in ~5 minutes.

## What's Included

| Script | Purpose |
|--------|---------|
| `deploy_and_start.sh` | **🚀 START HERE** - Complete deployment workflow |
| `check_prerequisites.sh` | Verify AWS CLI setup and permissions |
| `setup_aws_resources.sh` | One-time setup (security group, SSH key) |
| `launch_spot_instance.sh` | Launch g4dn.xlarge Spot instance |
| `bootstrap.sh` | Auto-runs on instance (setup & start training) |
| `monitor_training.sh` | Monitor logs, GPU usage, progress |
| `shutdown_instance.sh` | Download checkpoints and terminate |
| `estimate_costs.sh` | Check current Spot prices and cost estimates |

## Documentation

📖 **Complete guide:** [AWS_DEPLOYMENT_GUIDE.md](./AWS_DEPLOYMENT_GUIDE.md)

## Prerequisites

- AWS account with billing enabled
- AWS CLI v2 configured (`aws configure`)
- IAM permissions for EC2, SSM

## Performance

| Metric | Local | AWS (Spot) |
|--------|-------|------------|
| Iteration time | 14 min | 2-3 min |
| 100 iterations | 24 hours | 5 hours |
| Cost | Free | ~$1.00 |

## Fault Tolerance

✅ Checkpoints save every iteration
✅ Auto-resume after Spot termination
✅ Zero data loss

## Support

Questions? See [AWS_DEPLOYMENT_GUIDE.md](./AWS_DEPLOYMENT_GUIDE.md) or [open an issue](https://github.com/KenStager/Othello/issues).
