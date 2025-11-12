import os, pickle, random, numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, save_dir):
        self.capacity = capacity
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.buffer = deque(maxlen=capacity)
        self.total_added = 0

    def add(self, sample):
        self.buffer.append(sample)
        self.total_added += 1
        if self.total_added % 5000 == 0:
            self._save_shard()

    def _save_shard(self):
        path = os.path.join(self.save_dir, f"replay_{self.total_added}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.stack([b["state"] for b in batch], axis=0)
        policies = np.stack([b["policy"] for b in batch], axis=0)
        value_win = np.array([b["value_win"] for b in batch], dtype=np.float32)
        value_score = np.array([b["value_score"] for b in batch], dtype=np.float32)
        mobility = np.stack([b["mobility"] for b in batch], axis=0)
        stability = np.stack([b["stability"] for b in batch], axis=0)
        corner = np.stack([b["corner"] for b in batch], axis=0)
        parity = np.stack([b["parity"] for b in batch], axis=0)
        return states, policies, value_win, value_score, mobility, stability, corner, parity
