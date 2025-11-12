import argparse, torch, numpy as np
from src.utils.config import load_config
from src.othello.game import Game
from src.net.model import OthelloNet
from src.mcts.mcts import MCTS

def render(board):
    sym = {0:'.', 1:'B', -1:'W'}
    print("  " + " ".join(map(str, range(8))))
    for r in range(8):
        print(r, end=" ")
        for c in range(8):
            print(sym[int(board.board[r,c])], end=" ")
        print()
    print(f"To move: {'B' if board.player==1 else 'W'}")
    print()

def main(cfg_path, checkpoint=None):
    cfg = load_config(cfg_path)
    device = torch.device(cfg['device']) if torch.cuda.is_available() or cfg['device']=="cpu" else torch.device("cpu")
    net = OthelloNet(in_channels=4, channels=cfg['model']['channels'],
                     residual_blocks=cfg['model']['residual_blocks']).to(device)
    if checkpoint:
        sd = torch.load(checkpoint, map_location=device)
        net.load_state_dict(sd)
    net.eval()

    game_cls = lambda: Game(8)
    b = game_cls().new_board()

    mcts = MCTS(game_cls, net, device,
                cpuct=cfg['mcts']['cpuct'],
                simulations=cfg['mcts']['simulations'],
                dir_alpha=None, dir_frac=0.0, reuse_tree=False)

    while not b.is_terminal():
        render(b)
        if b.player == 1:
            # Human plays BLACK
            moves = b.legal_moves(1)
            if len(moves)==0:
                print("No legal moves. PASS.")
                b.step_action_index(64)
                continue
            try:
                s = input("Enter move as r c (e.g., 2 3), or 'p' to pass: ").strip()
                if s.lower().startswith('p') and not moves:
                    a = 64
                else:
                    r, c = map(int, s.split())
                    a = r*8 + c
                b.step_action_index(a)
            except Exception as e:
                print("Invalid input.", e)
        else:
            pi = mcts.run(b)
            a = int(np.argmax(pi))
            b.step_action_index(a)

    render(b)
    winner, diff = b.result()
    if winner == 1: print("BLACK wins!")
    elif winner == -1: print("WHITE wins!")
    else: print("Draw.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
