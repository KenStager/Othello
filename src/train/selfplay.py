import numpy as np

from ..mcts.mcts import MCTS
from ..othello.board import BLACK
from ..othello.features import (
    augment_state,
    compute_corner_control,
    compute_mobility_features,
    compute_parity_features,
    compute_stability_map,
)

def play_one_game(game_cls, net, device, mcts_cfg, temp_moves=12, max_moves=120, dir_alpha=0.15, dir_frac=0.25):
    game = game_cls()
    b = game.new_board()
    mcts = MCTS(game_cls, net, device,
                cpuct=mcts_cfg['cpuct'],
                simulations=mcts_cfg['simulations'],
                dir_alpha=dir_alpha,
                dir_frac=dir_frac,
                reuse_tree=mcts_cfg.get('reuse_tree', True))
    states, policies, players, board_snapshots = [], [], [], []
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
        board_snapshots.append(b.copy())

        # Sample (or pick argmax after temp phase)
        a = int(np.random.choice(len(probs), p=probs))
        b.step_action_index(a)
        move_count += 1

        # If no legal move for next player AND current also had none earlier, pass logic is handled in board

    # Game end
    winner, diff = b.result()
    # z from each player's perspective
    traj = []
    board_area = board_snapshots[0].n * board_snapshots[0].n if board_snapshots else 64
    base_score = diff / float(board_area) if board_area else 0.0

    for planes, pi, player, snapshot in zip(states, policies, players, board_snapshots):
        if winner == 0:
            z_win = 0.0
            z_score = 0.0
        elif winner == player:
            z_win = 1.0
            z_score = base_score
        else:
            z_win = -1.0
            z_score = -base_score

        stability_raw = compute_stability_map(snapshot)
        if player == BLACK:
            stability = np.stack((stability_raw[0], stability_raw[1]), axis=0)
        else:
            stability = np.stack((stability_raw[1], stability_raw[0]), axis=0)

        mobility = compute_mobility_features(snapshot)
        corner = compute_corner_control(snapshot)
        parity = compute_parity_features(snapshot)

        for aug_planes, aug_policy, aug_stability, aug_corner, aug_parity in augment_state(
            planes, pi, stability, corner, parity
        ):
            sample = {
                "state": aug_planes.astype(np.float32),
                "policy": aug_policy.astype(np.float32),
                "value_win": float(z_win),
                "value_score": float(z_score),
                "mobility": mobility.astype(np.float32),
                "stability": aug_stability.astype(np.float32),
                "corner": aug_corner.astype(np.float32),
                "parity": aug_parity.astype(np.float32),
            }
            traj.append(sample)
    return traj

def generate_selfplay(replay, game_cls, net, device, mcts_cfg, games=20, temp_moves=12, max_moves=120, dir_alpha=0.15, dir_frac=0.25):
    total = 0
    for _ in range(games):
        traj = play_one_game(game_cls, net, device, mcts_cfg, temp_moves, max_moves, dir_alpha, dir_frac)
        for sample in traj:
            replay.add(sample)
            total += 1
    return total
