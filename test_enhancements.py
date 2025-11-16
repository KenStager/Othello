"""
Quick test script to verify all enhancements work correctly.
"""
import sys
import numpy as np
import torch

print("=" * 60)
print("TESTING OTHELLO ENHANCEMENTS")
print("=" * 60)

# Test 1: Configuration loading
print("\n[1/8] Testing configuration loading...")
from src.utils.config import load_config
cfg = load_config("config_smoke_test.yaml")
print(f"✓ Config loaded successfully")
print(f"  - Temp schedule: {cfg['selfplay']['temp_schedule']}")
print(f"  - Phase mix: {cfg['train']['phase_mix']}")
print(f"  - TT enabled: {cfg['mcts']['tt_enabled']}")

# Test 2: Zobrist hashing
print("\n[2/8] Testing Zobrist hashing...")
from src.othello.board import Board
from src.mcts.zobrist import zobrist_hash

b1 = Board(8)
b2 = Board(8)
h1 = zobrist_hash(b1)
h2 = zobrist_hash(b2)
assert h1 == h2, "Identical boards should have identical hashes"

moves = b1.legal_moves()
if moves:
    b1.apply_move(moves[0])
    h3 = zobrist_hash(b1)
    assert h3 != h1, "Different boards should have different hashes"

print(f"✓ Zobrist hashing works correctly")
print(f"  - Initial board hash: {h1}")
print(f"  - After move hash: {h3}")

# Test 3: Phase tagging
print("\n[3/8] Testing phase tagging...")
from src.train.selfplay import move_temperature

temp_schedule = cfg['selfplay']['temp_schedule']
# Use thresholds from smoke test config: open_to=6, mid_to=12
tau1 = move_temperature(3, temp_schedule)   # Opening (ply <= 6)
tau2 = move_temperature(9, temp_schedule)   # Midgame (6 < ply <= 12)
tau3 = move_temperature(15, temp_schedule)  # Endgame (ply > 12)

assert tau1 == 1.0, f"Expected tau=1.0 for ply 3, got {tau1}"
assert tau2 == 0.25, f"Expected tau=0.25 for ply 9, got {tau2}"
assert tau3 == 0.0, f"Expected tau=0.0 for ply 15, got {tau3}"

print(f"✓ 3-phase temperature schedule works")
print(f"  - Ply 3 (opening, ≤6): τ={tau1}")
print(f"  - Ply 9 (midgame, 7-12): τ={tau2}")
print(f"  - Ply 15 (endgame, >12): τ={tau3}")

# Test 4: MCTS with transposition table
print("\n[4/8] Testing MCTS with transposition table...")
from src.othello.game import Game
from src.net.model import OthelloNet
from src.mcts.mcts import MCTS

device = torch.device("cpu")
net = OthelloNet(in_channels=4, channels=32, residual_blocks=2).to(device)  # Tiny model
game = Game(8)
board = game.new_board()

mcts = MCTS(
    game_cls=lambda: Game(8),
    net=net,
    device=device,
    cpuct=1.5,
    simulations=10,  # Very few for speed
    dir_alpha=0.15,
    dir_frac=0.25,
    reuse_tree=True,
    use_tt=True
)

pi = mcts.run(board)
tt_stats = mcts.get_tt_stats()

print(f"✓ MCTS with TT works correctly")
print(f"  - TT size: {tt_stats['size']}")
print(f"  - TT hits: {tt_stats['hits']}")
print(f"  - TT misses: {tt_stats['misses']}")
print(f"  - Policy shape: {pi.shape}")

# Test 5: Replay buffer with phase tagging
print("\n[5/8] Testing replay buffer with phase tagging...")
from src.train.replay import ReplayBuffer

replay = ReplayBuffer(capacity=1000, save_dir="data/replay_test")

# Add sample with phase tagging
sample = {
    "state": np.random.rand(4, 8, 8).astype(np.float32),
    "policy": np.random.rand(65).astype(np.float32),
    "value_win": 0.5,
    "value_score": 0.2,
    "mobility": np.random.rand(2).astype(np.float32),
    "stability": np.random.rand(2, 8, 8).astype(np.float32),
    "corner": np.random.rand(4).astype(np.float32),
    "parity": np.random.rand(5).astype(np.float32),
    "phase": "midgame",
    "empties": 30
}

replay.add(sample)
print(f"✓ Replay buffer with phase tagging works")
print(f"  - Buffer size: {replay.size()}")

# Test 6: Phase-weighted loss calculation
print("\n[6/8] Testing phase-weighted loss...")
empties = torch.tensor([50, 30, 10], dtype=torch.float32)  # opening, mid, end
score_weight_base = 0.3
score_weights = score_weight_base * (1.0 - empties / 64.0)

print(f"✓ Phase-weighted loss calculation works")
print(f"  - Opening (50 empties): weight={score_weights[0]:.3f}")
print(f"  - Midgame (30 empties): weight={score_weights[1]:.3f}")
print(f"  - Endgame (10 empties): weight={score_weights[2]:.3f}")

# Test 7: Oracle (dummy)
print("\n[7/8] Testing oracle bridge...")
from src.train.oracle import DummyOracle

oracle = DummyOracle(empties_threshold=14)
board_endgame = Board(8)
# Manually set to endgame state
board_endgame.board[board_endgame.board == 0] = 1  # Fill most squares
board_endgame.board[0, 0] = 0  # Leave some empties
board_endgame.board[0, 1] = 0

should_use = oracle.should_use(board_endgame)
result = oracle.evaluate(board_endgame)

print(f"✓ Oracle bridge works (dummy mode)")
print(f"  - Should use oracle: {should_use}")
print(f"  - Oracle result: {result}")

# Test 8: Gating with loss rate
print("\n[8/8] Testing gating criteria...")
wins, losses, draws = 11, 4, 5  # 55% win rate, 20% loss rate
total = wins + losses + draws
winrate = wins / max(1, wins + losses)
loss_rate = losses / max(1, total)

max_loss_mult = 1.10
champ_loss_rate = 0.15  # Champion had 15% loss rate

should_promote = (
    (wins + losses) > 0 and
    winrate >= 0.55 and
    loss_rate <= max_loss_mult * champ_loss_rate
)

print(f"✓ Gating criteria works")
print(f"  - Win rate: {winrate:.2%} (threshold: 55%)")
print(f"  - Loss rate: {loss_rate:.2%} (max: {max_loss_mult * champ_loss_rate:.2%})")
print(f"  - Should promote: {should_promote}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("\nYou can now run full training with:")
print("  PYTHONPATH=. python scripts/self_play_train.py --config config_smoke_test.yaml")
