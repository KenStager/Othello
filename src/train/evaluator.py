import numpy as np
from ..mcts.mcts import MCTS

def play_match(game_cls, net_a, net_b, device, mcts_cfg, games=20):
    # Alternate colors for fairness
    results = []  # +1 = A win, 0 = draw, -1 = A loss
    for g in range(games):
        game = game_cls()
        b = game.new_board()
        mcts_a = MCTS(game_cls, net_a, device, **mcts_cfg)
        mcts_b = MCTS(game_cls, net_b, device, **mcts_cfg)
        nets = {1: mcts_a, -1: mcts_b} if g % 2 == 0 else {1: mcts_b, -1: mcts_a}

        move_count = 0
        while not b.is_terminal() and move_count < 120:
            mcts = nets[b.player]
            pi = mcts.run(b)
            a = int(np.argmax(pi))  # deterministic for eval
            b.step_action_index(a)
            move_count += 1

        winner, _ = b.result()
        if g % 2 == 0:
            # BLACK is net_a in even games
            if winner == 1: results.append(+1)
            elif winner == -1: results.append(-1)
            else: results.append(0)
        else:
            # BLACK is net_b in odd games, so invert
            if winner == 1: results.append(-1)
            elif winner == -1: results.append(+1)
            else: results.append(0)
    wins = sum(1 for r in results if r == +1)
    losses = sum(1 for r in results if r == -1)
    draws = sum(1 for r in results if r == 0)
    return wins, losses, draws, results
