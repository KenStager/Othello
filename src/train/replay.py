import os, pickle, random, numpy as np, re, glob
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, save_dir, auto_load=True, cleanup_enabled=True,
                 cleanup_keep_recent=3, cleanup_keep_milestone_every=50000):
        self.capacity = capacity
        self.save_dir = save_dir
        self.cleanup_enabled = cleanup_enabled
        self.cleanup_keep_recent = cleanup_keep_recent
        self.cleanup_keep_milestone_every = cleanup_keep_milestone_every
        os.makedirs(save_dir, exist_ok=True)
        self.buffer = deque(maxlen=capacity)
        self.total_added = 0

        # Auto-load latest shard if available
        if auto_load:
            latest_path, latest_count = self._find_latest_shard()
            if latest_path:
                self._load_shard(latest_path, latest_count)
                print(f"Loaded replay buffer: {os.path.basename(latest_path)} ({len(self.buffer):,} samples, total_added={self.total_added:,})")

    def add(self, sample):
        self.buffer.append(sample)
        self.total_added += 1
        if self.total_added % 5000 == 0:
            self._save_shard()

    def _save_shard(self):
        path = os.path.join(self.save_dir, f"replay_{self.total_added}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)

        # Auto-cleanup old shards if enabled
        if self.cleanup_enabled:
            self._cleanup_old_shards()

    def _cleanup_old_shards(self):
        """
        Remove old replay shards to prevent unbounded disk usage.

        Keeps:
        - N most recent shards (self.cleanup_keep_recent)
        - Milestone shards at specified intervals (self.cleanup_keep_milestone_every)

        Deletes all other shards.
        """
        pattern = os.path.join(self.save_dir, "replay_*.pkl")
        shard_files = glob.glob(pattern)

        # Extract counts and sort
        shards = []
        for path in shard_files:
            basename = os.path.basename(path)
            match = re.search(r'replay_(\d+)\.pkl', basename)
            if match:
                count = int(match.group(1))
                shards.append((path, count))

        shards.sort(key=lambda x: x[1], reverse=True)

        if len(shards) <= self.cleanup_keep_recent:
            return  # Not enough shards to clean

        # Determine which to keep
        keep_paths = set()

        # Keep N most recent
        for path, count in shards[:self.cleanup_keep_recent]:
            keep_paths.add(path)

        # Keep milestone shards
        for path, count in shards:
            if count % self.cleanup_keep_milestone_every == 0:
                keep_paths.add(path)

        # Delete the rest
        deleted_count = 0
        freed_mb = 0
        for path, count in shards:
            if path not in keep_paths:
                try:
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    os.remove(path)
                    deleted_count += 1
                    freed_mb += size_mb
                except Exception as e:
                    print(f"Warning: Failed to delete {os.path.basename(path)}: {e}")

        if deleted_count > 0:
            print(f"  Cleaned up {deleted_count} old replay shards, freed {freed_mb:.1f} MB")

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, phase_mix=None):
        """
        Sample from replay buffer with optional phase balancing.

        Args:
            batch_size: Total number of samples
            phase_mix: Optional [opening, midgame, endgame] ratios (e.g., [0.4, 0.4, 0.2])
        """
        if phase_mix is None or len(self.buffer) < batch_size:
            # Simple random sampling
            batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        else:
            # Phase-balanced sampling
            opening_samples = [s for s in self.buffer if s.get("phase") == "opening"]
            midgame_samples = [s for s in self.buffer if s.get("phase") == "midgame"]
            endgame_samples = [s for s in self.buffer if s.get("phase") == "endgame"]

            n_opening = int(batch_size * phase_mix[0])
            n_midgame = int(batch_size * phase_mix[1])
            n_endgame = batch_size - n_opening - n_midgame

            batch = []
            if opening_samples:
                batch.extend(random.sample(opening_samples, min(n_opening, len(opening_samples))))
            if midgame_samples:
                batch.extend(random.sample(midgame_samples, min(n_midgame, len(midgame_samples))))
            if endgame_samples:
                batch.extend(random.sample(endgame_samples, min(n_endgame, len(endgame_samples))))

            # Fill remainder with random if needed
            while len(batch) < batch_size and len(self.buffer) > len(batch):
                batch.append(random.choice(self.buffer))

        states = np.stack([b["state"] for b in batch], axis=0)
        policies = np.stack([b["policy"] for b in batch], axis=0)
        value_win = np.array([b["value_win"] for b in batch], dtype=np.float32)
        value_score = np.array([b["value_score"] for b in batch], dtype=np.float32)
        mobility = np.stack([b["mobility"] for b in batch], axis=0)
        stability = np.stack([b["stability"] for b in batch], axis=0)
        corner = np.stack([b["corner"] for b in batch], axis=0)
        parity = np.stack([b["parity"] for b in batch], axis=0)
        empties = np.array([b.get("empties", 32) for b in batch], dtype=np.int32)
        phases = np.array([b.get("phase", "midgame") for b in batch])
        return states, policies, value_win, value_score, mobility, stability, corner, parity, empties, phases

    def _find_latest_shard(self):
        """
        Find the most recent replay shard file dynamically.

        Returns:
            (path, count) tuple if found, else (None, None)
        """
        pattern = os.path.join(self.save_dir, "replay_*.pkl")
        shard_files = glob.glob(pattern)

        if not shard_files:
            return None, None

        # Extract counts from filenames: replay_20000.pkl → 20000
        shards_with_counts = []
        for path in shard_files:
            basename = os.path.basename(path)
            match = re.search(r'replay_(\d+)\.pkl', basename)
            if match:
                count = int(match.group(1))
                shards_with_counts.append((path, count))

        if not shards_with_counts:
            return None, None

        # Return shard with highest count (most recent)
        latest_path, latest_count = max(shards_with_counts, key=lambda x: x[1])
        return latest_path, latest_count

    def _load_shard(self, path, count):
        """
        Load samples from a replay shard file.

        Args:
            path: Path to the .pkl file
            count: The sample count (from filename, e.g., 20000)
        """
        try:
            with open(path, 'rb') as f:
                samples = pickle.load(f)

            # Load samples into buffer (respects maxlen capacity)
            for sample in samples:
                self.buffer.append(sample)

            # Restore total_added counter
            self.total_added = count

        except Exception as e:
            print(f"Warning: Failed to load replay shard {path}: {e}")
            # Continue with empty buffer on error


class ILDataset:
    """
    Imitation Learning dataset for expert game samples.

    Loads IL samples from pickle shards (generated by parse_wthor.py).
    Supports random sampling for mixing with self-play data.
    """

    def __init__(self, data_dir):
        """
        Load IL samples from data directory.

        Args:
            data_dir: Directory containing il_shard_*.pkl files
        """
        self.data_dir = data_dir
        self.samples = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"IL data directory not found: {data_dir}")

        # Load all IL shard files
        pattern = os.path.join(data_dir, "il_shard_*.pkl")
        shard_files = sorted(glob.glob(pattern))

        if len(shard_files) == 0:
            raise FileNotFoundError(f"No IL shard files found in {data_dir}")

        print(f"Loading IL data from {len(shard_files)} shard files...")

        for shard_path in shard_files:
            try:
                with open(shard_path, 'rb') as f:
                    shard_samples = pickle.load(f)
                self.samples.extend(shard_samples)
                print(f"  Loaded {os.path.basename(shard_path)}: {len(shard_samples)} samples")
            except Exception as e:
                print(f"  Warning: Failed to load {shard_path}: {e}")

        if len(self.samples) == 0:
            raise ValueError("No IL samples loaded from data directory")

        # Validate sample structure
        self._validate_samples()

        print(f"IL dataset ready: {len(self.samples):,} samples")

    def _validate_samples(self):
        """Validate that samples have required fields."""
        required_fields = {
            "state", "policy", "value_win", "value_score",
            "mobility", "stability", "corner", "parity",
            "phase", "empties"
        }

        if len(self.samples) == 0:
            return

        sample = self.samples[0]
        missing_fields = required_fields - set(sample.keys())

        if missing_fields:
            raise ValueError(f"IL samples missing required fields: {missing_fields}")

    def __len__(self):
        """Return number of IL samples."""
        return len(self.samples)

    def sample(self, batch_size, phase_mix=None):
        """
        Sample a batch of IL data.

        Args:
            batch_size: Number of samples to return
            phase_mix: Optional [opening, midgame, endgame] ratios

        Returns:
            Same format as ReplayBuffer.sample()
        """
        if batch_size > len(self.samples):
            batch_size = len(self.samples)

        if phase_mix is None:
            # Simple random sampling without replacement
            batch = random.sample(self.samples, batch_size)
        else:
            # Phase-balanced sampling
            opening_samples = [s for s in self.samples if s.get("phase") == "opening"]
            midgame_samples = [s for s in self.samples if s.get("phase") == "midgame"]
            endgame_samples = [s for s in self.samples if s.get("phase") == "endgame"]

            n_opening = int(batch_size * phase_mix[0])
            n_midgame = int(batch_size * phase_mix[1])
            n_endgame = batch_size - n_opening - n_midgame

            batch = []
            if opening_samples:
                batch.extend(random.sample(opening_samples, min(n_opening, len(opening_samples))))
            if midgame_samples:
                batch.extend(random.sample(midgame_samples, min(n_midgame, len(midgame_samples))))
            if endgame_samples:
                batch.extend(random.sample(endgame_samples, min(n_endgame, len(endgame_samples))))

            # Fill remainder with random if needed
            while len(batch) < batch_size:
                batch.append(random.choice(self.samples))

        # Convert to same format as ReplayBuffer.sample()
        states = np.stack([b["state"] for b in batch], axis=0)
        policies = np.stack([b["policy"] for b in batch], axis=0)
        value_win = np.array([b["value_win"] for b in batch], dtype=np.float32)
        value_score = np.array([b["value_score"] for b in batch], dtype=np.float32)
        mobility = np.stack([b["mobility"] for b in batch], axis=0)
        stability = np.stack([b["stability"] for b in batch], axis=0)
        corner = np.stack([b["corner"] for b in batch], axis=0)
        parity = np.stack([b["parity"] for b in batch], axis=0)
        empties = np.array([b.get("empties", 32) for b in batch], dtype=np.int32)
        phases = np.array([b.get("phase", "midgame") for b in batch])

        return states, policies, value_win, value_score, mobility, stability, corner, parity, empties, phases


def sample_mixed_batch(replay_buffer, il_dataset, batch_size, il_ratio, phase_mix=None):
    """
    Sample a mixed batch from both IL data and self-play replay buffer.

    Args:
        replay_buffer: ReplayBuffer instance (self-play data)
        il_dataset: ILDataset instance (expert games)
        batch_size: Total batch size
        il_ratio: Fraction of IL samples (0.0-1.0)
        phase_mix: Optional [opening, midgame, endgame] ratios

    Returns:
        Combined batch in same format as ReplayBuffer.sample()
    """
    # Determine split
    n_il = int(batch_size * il_ratio)
    n_replay = batch_size - n_il

    # Sample from both sources
    il_batch = il_dataset.sample(n_il, phase_mix=phase_mix) if n_il > 0 else None
    replay_batch = replay_buffer.sample(n_replay, phase_mix=phase_mix) if n_replay > 0 else None

    # Combine batches
    if il_batch is None:
        return replay_batch
    elif replay_batch is None:
        return il_batch
    else:
        # Concatenate all arrays
        combined = tuple(
            np.concatenate([il_arr, replay_arr], axis=0)
            for il_arr, replay_arr in zip(il_batch, replay_batch)
        )
        return combined
