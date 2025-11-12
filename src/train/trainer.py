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
        )

def train_steps(net, replay, device, steps, batch_size, lr, lr_min, weight_decay, grad_clip):
    if replay.size() == 0:
        return 0.0
    ds = ReplayDataset(replay)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(loader)

    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    value_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss()

    net.train()
    total_loss = 0.0
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        (s, pi, z_win, z_score, mobility, stability, corner, parity) = batch

        s = s.to(device=device, dtype=torch.float32)
        pi = pi.to(device=device, dtype=torch.float32)
        z_win = z_win.to(device=device)
        z_score = z_score.to(device=device)
        mobility = mobility.to(device=device, dtype=torch.float32)
        stability = stability.to(device=device, dtype=torch.float32)
        corner = corner.to(device=device, dtype=torch.float32)
        parity = parity.to(device=device, dtype=torch.float32)

        outputs = net(s)
        logits = outputs.policy_logits
        v_win = outputs.value_win
        v_score = outputs.value_score

        # CrossEntropy expects class indices; we have a distribution target. Use KL or CE-with-soft labels.
        # Use -sum(p * logsoftmax)
        logp = torch.log_softmax(logits, dim=1)
        pol_loss = -(pi * logp).sum(dim=1).mean()

        val_loss = value_loss_fn(v_win, z_win)
        score_loss = value_loss_fn(v_score, z_score)

        mobility_loss = bce_loss_fn(outputs.mobility, mobility)
        stability_loss = bce_loss_fn(outputs.stability_map, stability)
        corner_loss = bce_loss_fn(outputs.corner, corner)
        parity_loss = bce_loss_fn(outputs.parity, parity)

        loss = (
            pol_loss
            + val_loss
            + 0.3 * score_loss
            + 0.2 * mobility_loss
            + 0.2 * stability_loss
            + 0.1 * corner_loss
            + 0.1 * parity_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()

        total_loss += loss.item()
    return total_loss / max(1, steps)
