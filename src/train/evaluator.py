import numpy as np
from ..mcts.mcts import MCTS

def play_match(game_cls, net_a, net_b, device, mcts_cfg, games=20, oracle=None, opening_suite=None, verbose=True):
    # Alternate colors for fairness
    # IMPORTANT: Uses paired position evaluation - each position played twice (once per color)
    results = []  # +1 = A win, 0 = draw, -1 = A loss
    game_details = []  # Store details for each game

    # Sample positions ONCE for entire match (paired evaluation)
    if opening_suite:
        import random
        from scripts.make_opening_suite import dict_to_board
        num_positions = games // 2  # Each position played twice
        sampled_positions = random.sample(opening_suite, num_positions)
        if verbose:
            print(f"    Paired evaluation: {num_positions} positions × 2 colors = {games} games")
    else:
        sampled_positions = None

    for g in range(games):
        game = game_cls()

        # Get opening position (paired: same position for g and g+1 if g is even)
        if sampled_positions:
            pos_idx = g // 2  # Position index (0, 0, 1, 1, 2, 2, ...)
            opening_dict = sampled_positions[pos_idx]
            b = dict_to_board(opening_dict)
        else:
            b = game.new_board()

        mcts_a = MCTS(game_cls, net_a, device, **mcts_cfg)
        mcts_b = MCTS(game_cls, net_b, device, **mcts_cfg)
        nets = {1: mcts_a, -1: mcts_b} if g % 2 == 0 else {1: mcts_b, -1: mcts_a}

        move_count = 0
        while not b.is_terminal() and move_count < 120:
            # Check if oracle should be used for this position
            if oracle and oracle.should_use(b):
                oracle_result = oracle.evaluate(b)
                if oracle_result.get('is_exact') and oracle_result.get('best_move') is not None:
                    # Use oracle move
                    a = oracle_result['best_move']
                else:
                    # Oracle failed, fallback to MCTS
                    mcts = nets[b.player]
                    pi = mcts.run(b)
                    a = int(np.argmax(pi))
            else:
                # Normal MCTS
                mcts = nets[b.player]
                pi = mcts.run(b)
                # FIX: Gating must be deterministic - always pick best move (argmax)
                a = int(np.argmax(pi))
            b.step_action_index(a)
            move_count += 1

        winner, score_diff = b.result()
        a_is_black = (g % 2 == 0)

        if a_is_black:
            # BLACK is net_a in even games
            if winner == 1:
                results.append(+1)
                result_str = "A wins"
            elif winner == -1:
                results.append(-1)
                result_str = "B wins"
            else:
                results.append(0)
                result_str = "Draw"
            a_score = score_diff
        else:
            # BLACK is net_b in odd games, so invert
            if winner == 1:
                results.append(-1)
                result_str = "B wins"
            elif winner == -1:
                results.append(+1)
                result_str = "A wins"
            else:
                results.append(0)
                result_str = "Draw"
            a_score = -score_diff

        game_details.append({
            'game_num': g + 1,
            'result': results[-1],
            'result_str': result_str,
            'score': a_score,
            'moves': move_count,
            'a_is_black': a_is_black
        })

        if verbose:
            color_a = "BLACK" if a_is_black else "WHITE"
            color_b = "WHITE" if a_is_black else "BLACK"
            score_str = f"{a_score:+d}" if winner != 0 else "+0"
            print(f"    Game {g+1}/{games}: {result_str} {score_str} ({move_count} moves) [A={color_a}, B={color_b}]")

    wins = sum(1 for r in results if r == +1)
    losses = sum(1 for r in results if r == -1)
    draws = sum(1 for r in results if r == 0)
    return wins, losses, draws, results, game_details
