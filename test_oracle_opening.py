"""Quick test for oracle and opening suite integration."""
import sys
sys.path.insert(0, '.')

from src.utils.config import load_config
from src.train.oracle import create_oracle
from src.othello.game import Game
from src.othello.board import EMPTY
import json

# Load config
cfg = load_config('config.yaml')

print("=== Testing Oracle Integration ===")
# Test oracle
oracle = create_oracle(cfg)
print(f"Oracle created: {oracle}")
print(f"Oracle should_use enabled: {cfg['oracle']['use']}")

# Create an endgame position (14 empties)
game = Game()
b = game.new_board()

# Play some moves to reach endgame
moves = [27, 19, 18, 26, 34, 35, 43, 42]  # Sample opening moves
for move in moves:
    valid_mask = b.valid_action_mask()
    if valid_mask[move]:
        b.step_action_index(move)
    else:
        print(f"Move {move} not valid, skipping")

empties = int((b.board == EMPTY).sum())
print(f"Board empties: {empties}")
print(f"Should use oracle: {oracle.should_use(b)}")

if oracle.should_use(b):
    result = oracle.query(b)
    print(f"Oracle result: {result}")

print("\n=== Testing Opening Suite ===")
# Test opening suite
with open(cfg['paths']['opening_suite'], 'r') as f:
    opening_suite = json.load(f)

print(f"Opening suite loaded: {len(opening_suite)} positions")
print(f"Sample opening keys: {list(opening_suite[0].keys())}")

import random
from scripts.make_opening_suite import dict_to_board
opening_dict = random.choice(opening_suite)
opening_board = dict_to_board(opening_dict)
print(f"Sample opening board created, empties: {int((opening_board.board == EMPTY).sum())}")
print(f"Sample opening player: {opening_board.player}")

print("\n=== All Tests Passed ===")
