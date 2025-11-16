"""Test paired position evaluation in gating."""
import sys
sys.path.insert(0, '.')

import json
from src.othello.game import Game
from scripts.make_opening_suite import dict_to_board

# Load opening suite
with open('data/openings/rot64.json', 'r') as f:
    opening_suite = json.load(f)

print("=== Testing Paired Position Logic ===\n")

# Simulate paired evaluation with 4 games
games = 4
num_positions = games // 2
print(f"Games: {games}")
print(f"Positions to sample: {num_positions}")
print(f"Expected: Each position played twice\n")

# Sample positions once
import random
random.seed(42)  # For reproducibility
sampled_positions = random.sample(opening_suite, num_positions)

print("Sampled positions:")
for i, pos in enumerate(sampled_positions):
    print(f"  Position {i}: {id(pos)}")

print("\nGame-to-Position mapping:")
position_hashes = []
for g in range(games):
    pos_idx = g // 2  # 0, 0, 1, 1
    opening_dict = sampled_positions[pos_idx]
    b = dict_to_board(opening_dict)
    board_hash = hash(b.board.tobytes())
    position_hashes.append(board_hash)

    color_swap = g % 2
    a_is_black = (g % 2 == 0)

    print(f"  Game {g}: Position {pos_idx} (hash: {board_hash}) | A is {'BLACK' if a_is_black else 'WHITE'}")

print("\nVerification:")
# Check pairing: games 0 and 1 should be same position
if position_hashes[0] == position_hashes[1]:
    print(f"  ✅ Games 0 and 1 use SAME position (paired)")
else:
    print(f"  ❌ Games 0 and 1 use DIFFERENT positions (NOT paired)")

# Check pairing: games 2 and 3 should be same position
if position_hashes[2] == position_hashes[3]:
    print(f"  ✅ Games 2 and 3 use SAME position (paired)")
else:
    print(f"  ❌ Games 2 and 3 use DIFFERENT positions (NOT paired)")

# Check different pairs use different positions
if position_hashes[0] != position_hashes[2]:
    print(f"  ✅ Position pairs 0-1 and 2-3 are DIFFERENT")
else:
    print(f"  ❌ Position pairs 0-1 and 2-3 are SAME (should be different)")

print("\n=== Test Complete ===")
