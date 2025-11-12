import numpy as np
from ..othello.board import Board
from ..mcts.mcts import MCTS
import torch

def play_one_game(game_cls, net, device, mcts_cfg, temp_moves=12, max_moves=120, dir_alpha=0.15, dir_frac=0.25):
    game = game_cls()
    b = game.new_board()
    mcts = MCTS(game_cls, net, device,
                cpuct=mcts_cfg['cpuct'],
                simulations=mcts_cfg['simulations'],
                dir_alpha=dir_alpha,
                dir_frac=dir_frac,
                reuse_tree=mcts_cfg.get('reuse_tree', True))
    states, policies, players = [], [], []
    move_count = 0
    while not b.is_terminal() and move_count < max_moves:
        pi = mcts.run(b)
        # Temperature
        if move_count < temp_moves:
            probs = pi
        else:
            # argmax
            probs = np.zeros_like(pi)
            probs[np.argmax(pi)] = 1.0

        states.append(b.encode())
        policies.append(probs)
        players.append(b.player)

        # Sample (or pick argmax after temp phase)
        a = int(np.random.choice(len(probs), p=probs))
        b.step_action_index(a)
        move_count += 1

        # If no legal move for next player AND current also had none earlier, pass logic is handled in board

    # Game end
    winner, diff = b.result()
    # z from each player's perspective
    traj = []
    for s, pi, p in zip(states, policies, players):
        z = 0.0
        if winner == 0:
            z = 0.0
        elif winner == p:
            z = 1.0
        else:
            z = -1.0
        # Symmetry augmentation
        for ss, ppi in Board.symmetries(s, pi):
            traj.append((ss, ppi, z))
    return traj

def generate_selfplay(replay, game_cls, net, device, mcts_cfg, games=20, temp_moves=12, max_moves=120, dir_alpha=0.15, dir_frac=0.25):
    total = 0
    for _ in range(games):
        traj = play_one_game(game_cls, net, device, mcts_cfg, temp_moves, max_moves, dir_alpha, dir_frac)
        for s, pi, z in traj:
            replay.add(s, pi, z)
            total += 1
    return total
