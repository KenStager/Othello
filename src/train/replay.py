import os, pickle, random, numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, save_dir):
        self.capacity = capacity
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.buffer = deque(maxlen=capacity)
        self.total_added = 0

    def add(self, s, pi, z):
        # s: (4,8,8) float32, pi: (65,) float32, z: float scalar in [-1,1]
        self.buffer.append((s.astype(np.float32), pi.astype(np.float32), float(z)))
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
        s = np.stack([b[0] for b in batch], axis=0)
        pi = np.stack([b[1] for b in batch], axis=0)
        z = np.array([b[2] for b in batch], dtype=np.float32)
        return s, pi, z
