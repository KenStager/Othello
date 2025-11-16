import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ReplayDataset(Dataset):
    def __init__(self, replay):
        self.replay = replay

    def __len__(self):
        return self.replay.size()

    def __getitem__(self, idx):
        sample = self.replay.buffer[idx]
        return (
            torch.from_numpy(sample["state"]),
            torch.from_numpy(sample["policy"]),
            torch.tensor(sample["value_win"], dtype=torch.float32),
            torch.tensor(sample["value_score"], dtype=torch.float32),
            torch.from_numpy(sample["mobility"]),
            torch.from_numpy(sample["stability"]),
            torch.from_numpy(sample["corner"]),
            torch.from_numpy(sample["parity"]),
            torch.tensor(sample.get("empties", 32), dtype=torch.int32),
            sample.get("phase", "midgame"),  # Include phase for diagnostics
        )

def train_steps(net, replay, device, steps, batch_size, optimizer=None, lr=None, lr_min=None, weight_decay=1e-4, grad_clip=1.0,
                phase_weighted_score=True, score_weight_base=0.3, il_dataset=None, il_ratio=0.0,
                verbose=True, log_interval=50, diagnostics=None, stability_monitor=None):
    """
    Train network for specified steps (Phase 1: Accepts persistent optimizer).

    Args:
        optimizer: Pre-created optimizer (new). If None, creates one from lr (backward compat).
        lr: Learning rate (used only if optimizer is None)
        ... (rest of args)
    """
    if replay.size() == 0:
        return 0.0

    # Backward compatibility: create optimizer if not provided
    if optimizer is None:
        if lr is None:
            raise ValueError("Must provide either optimizer or lr")
        opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = optimizer

    # Choose sampling strategy based on IL mixing
    use_il_mixing = (il_dataset is not None and il_ratio > 0.0)

    if use_il_mixing:
        # Manual sampling with IL mixing (no DataLoader)
        from .replay import sample_mixed_batch
        loader_it = None
    else:
        # Standard DataLoader approach (no IL mixing)
        ds = ReplayDataset(replay)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        loader_it = iter(loader)
    value_loss_fn = nn.MSELoss(reduction='none' if phase_weighted_score else 'mean')
    bce_loss_fn = nn.BCELoss()

    net.train()
    total_loss = 0.0
    for step in range(steps):
        # Sample batch (with or without IL mixing)
        if use_il_mixing:
            # Sample mixed batch from IL + replay
            batch_tuple = sample_mixed_batch(replay, il_dataset, batch_size, il_ratio)
            s, pi, z_win, z_score, mobility, stability, corner, parity, empties, phases = batch_tuple
            # Convert to tensors
            s = torch.from_numpy(s)
            pi = torch.from_numpy(pi)
            z_win = torch.from_numpy(z_win)
            z_score = torch.from_numpy(z_score)
            mobility = torch.from_numpy(mobility)
            stability = torch.from_numpy(stability)
            corner = torch.from_numpy(corner)
            parity = torch.from_numpy(parity)
            empties = torch.from_numpy(empties)
            # phases is already numpy array of strings
        else:
            # Use DataLoader
            try:
                batch = next(loader_it)
            except StopIteration:
                loader_it = iter(loader)
                batch = next(loader_it)
            (s, pi, z_win, z_score, mobility, stability, corner, parity, empties, phases) = batch

        s = s.to(device=device, dtype=torch.float32)
        pi = pi.to(device=device, dtype=torch.float32)
        z_win = z_win.to(device=device)
        z_score = z_score.to(device=device)
        mobility = mobility.to(device=device, dtype=torch.float32)
        stability = stability.to(device=device, dtype=torch.float32)
        corner = corner.to(device=device, dtype=torch.float32)
        parity = parity.to(device=device, dtype=torch.float32)
        empties = empties.to(device=device, dtype=torch.float32)

        outputs = net(s)
        logits = outputs.policy_logits
        v_win = outputs.value_win
        v_score = outputs.value_score

        # CrossEntropy expects class indices; we have a distribution target. Use KL or CE-with-soft labels.
        # Use -sum(p * logsoftmax)
        logp = torch.log_softmax(logits, dim=1)
        pol_loss = -(pi * logp).sum(dim=1).mean()

        val_loss = value_loss_fn(v_win, z_win)
        if phase_weighted_score:
            val_loss = val_loss.mean()

        # Phase-weighted score loss: weight increases as empties decrease
        if phase_weighted_score:
            score_weight = score_weight_base * (1.0 - empties / 64.0)
            score_loss_unreduced = value_loss_fn(v_score, z_score).squeeze()
            score_loss = (score_loss_unreduced * score_weight).mean()
        else:
            if value_loss_fn.reduction == 'none':
                score_loss = value_loss_fn(v_score, z_score).mean()
            else:
                score_loss = value_loss_fn(v_score, z_score)
            score_loss = score_weight_base * score_loss

        mobility_loss = bce_loss_fn(outputs.mobility, mobility)
        stability_loss = bce_loss_fn(outputs.stability_map, stability)
        corner_loss = bce_loss_fn(outputs.corner, corner)
        parity_loss = bce_loss_fn(outputs.parity, parity)

        loss = (
            pol_loss
            + val_loss
            + score_loss
            + 0.2 * mobility_loss
            + 0.2 * stability_loss
            + 0.1 * corner_loss
            + 0.1 * parity_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Record gradients for stability monitoring (before clipping)
        if stability_monitor is not None:
            gradients = {name: param.grad.clone() if param.grad is not None else None
                        for name, param in net.named_parameters()}
            stability_monitor.record_step(loss.item(), gradients=gradients, outputs=outputs)

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()

        total_loss += loss.item()

        # Record diagnostics
        if diagnostics is not None:
            batch_dict = {
                'state': s,
                'policy': pi,
                'value_win': z_win,
                'value_score': z_score,
                'phase': phases
            }
            losses_dict = {
                'total': loss,
                'policy': pol_loss,
                'value_win': val_loss,
                'value_score': score_loss,
                'loss_mobility': mobility_loss,
                'loss_stability': stability_loss,
                'loss_corner': corner_loss,
                'loss_parity': parity_loss
            }
            diagnostics.record_batch(batch_dict, outputs, losses_dict)

        # Verbose logging
        if verbose and (step + 1) % log_interval == 0:
            aux_total = (0.2 * mobility_loss.item() + 0.2 * stability_loss.item() +
                         0.1 * corner_loss.item() + 0.1 * parity_loss.item())
            print(f"    Step {step+1}/{steps}: loss={loss.item():.4f} "
                  f"[pol={pol_loss.item():.3f}, val={val_loss.item():.3f}, "
                  f"score={score_loss.item():.3f}, aux={aux_total:.3f}]")

            # Print diagnostics summary periodically
            if diagnostics is not None and (step + 1) % (log_interval * 4) == 0:
                print()
                diagnostics.print_summary(step=step+1)

                # Print health score
                health, issues = diagnostics.get_training_health_score()
                if health < 0.8:
                    print(f"⚠️  Training health: {health:.2f}/1.0")
                    for issue in issues:
                        print(f"   - {issue}")
                    print()

            # Check stability and print alerts if needed
            if stability_monitor is not None and (step + 1) % (log_interval * 2) == 0:
                if stability_monitor.has_critical_issues():
                    print()
                    print("🚨 STABILITY ALERT:")
                    stability_monitor.print_alerts(recent_only=True)
                    recommendations = stability_monitor.get_recommendations()
                    if recommendations:
                        print("Recommendations:")
                        for rec in recommendations[:3]:  # Top 3 recommendations
                            print(f"  • {rec}")
                    print()

        # MPS memory management: clear cache periodically
        if device.type == "mps" and (step + 1) % 100 == 0:
            torch.mps.empty_cache()
            # Log memory usage in verbose mode (every 500 steps)
            if verbose and (step + 1) % 500 == 0:
                try:
                    allocated = torch.mps.current_allocated_memory() / 1e9
                    driver = torch.mps.driver_allocated_memory() / 1e9
                    print(f"    MPS memory: {allocated:.2f} GB allocated, {driver:.2f} GB driver")
                except:
                    pass  # Some PyTorch versions may not have these APIs

    return total_loss / max(1, steps)
