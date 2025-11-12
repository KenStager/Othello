import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ..utils.logger import log_print

class ReplayDataset(Dataset):
    def __init__(self, replay):
        self.replay = replay
    def __len__(self):
        return self.replay.size()
    def __getitem__(self, idx):
        s, pi, z = self.replay.buffer[idx]
        return torch.from_numpy(s), torch.from_numpy(pi), torch.tensor(z, dtype=torch.float32)

def train_steps(net, replay, device, steps, batch_size, lr, lr_min, weight_decay, grad_clip):
    if replay.size() == 0:
        return 0.0
    ds = ReplayDataset(replay)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(loader)

    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    net.train()
    total_loss = 0.0
    for step in range(steps):
        try:
            s, pi, z = next(it)
        except StopIteration:
            it = iter(loader)
            s, pi, z = next(it)
        s = s.to(device=device, dtype=torch.float32)
        pi = pi.to(device=device, dtype=torch.float32)
        z = z.to(device=device)

        logits, v = net(s)           # logits: (B,65), v: (B,)
        # CrossEntropy expects class indices; we have a distribution target. Use KL or CE-with-soft labels.
        # Use -sum(p * logsoftmax)
        logp = torch.log_softmax(logits, dim=1)
        pol_loss = -(pi * logp).sum(dim=1).mean()

        val_loss = value_loss_fn(v, z)

        loss = pol_loss + val_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()

        total_loss += loss.item()
    return total_loss / max(1, steps)
