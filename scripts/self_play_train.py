import argparse, os, copy, torch
from src.utils.config import load_config
from src.utils.logger import log_print, TSVLogger
from src.utils.seed import set_seed
from src.othello.game import Game
from src.net.model import OthelloNet
from src.train.replay import ReplayBuffer
from src.train.selfplay import generate_selfplay
from src.train.trainer import train_steps
from src.train.evaluator import play_match

def main(cfg_path):
    cfg = load_config(cfg_path)
    set_seed(cfg['seed'])

    device = torch.device(cfg['device']) if torch.cuda.is_available() or cfg['device']=="cpu" else torch.device("cpu")
    log_print(f"Using device: {device}")

    # Model
    net = OthelloNet(in_channels=4, channels=cfg['model']['channels'],
                     residual_blocks=cfg['model']['residual_blocks']).to(device)
    champion = copy.deepcopy(net).to(device)

    # Replay
    replay = ReplayBuffer(capacity=cfg['train']['replay_capacity'],
                          save_dir=cfg['paths']['replay_dir'])

    # Logging
    os.makedirs("logs", exist_ok=True)
    tsv = TSVLogger("logs/train.tsv")

    game_cls = lambda: Game(cfg['game']['board_size'])
    mcts_cfg = dict(
        cpuct=cfg['mcts']['cpuct'],
        simulations=cfg['mcts']['simulations'],
        dir_alpha=cfg['game']['dirichlet_alpha'],
        dir_frac=cfg['game']['dirichlet_frac'],
        reuse_tree=cfg['mcts']['reuse_tree'],
    )

    it = 0
    while True:
        it += 1
        log_print(f"=== Iteration {it} ===")

        # 1) Self-play
        added = generate_selfplay(
            replay=replay,
            game_cls=game_cls,
            net=net,
            device=device,
            mcts_cfg=mcts_cfg,
            games=cfg['selfplay']['games_per_iter'],
            temp_moves=cfg['game']['temperature_moves'],
            max_moves=cfg['selfplay']['max_moves'],
            dir_alpha=cfg['game']['dirichlet_alpha'],
            dir_frac=cfg['game']['dirichlet_frac']
        )
        log_print(f"Self-play added samples: {added} (buffer size={replay.size()})")
        tsv.log("replay_size", replay.size())

        # 2) Train (if enough data)
        if replay.size() >= cfg['train']['min_replay_to_train']:
            avg_loss = train_steps(
                net=net, replay=replay, device=device,
                steps=cfg['train']['steps_per_iter'],
                batch_size=cfg['train']['batch_size'],
                lr=cfg['train']['lr'],
                lr_min=cfg['train']['lr_min'],
                weight_decay=cfg['train']['weight_decay'],
                grad_clip=cfg['train']['grad_clip'],
            )
            log_print(f"Train loss: {avg_loss:.4f}")
            tsv.log("train_loss", avg_loss)
        else:
            log_print("Not enough data to train yet.")

        # 3) Gate: pit current net vs champion
        wins, losses, draws, _ = play_match(
            game_cls=game_cls,
            net_a=net,
            net_b=champion,
            device=device,
            mcts_cfg=dict(cpuct=cfg['mcts']['cpuct'],
                          simulations=max(50, cfg['mcts']['simulations']//2),
                          dir_alpha=None, dir_frac=0.0, reuse_tree=False),
            games=cfg['gate']['eval_games']//10  # keep light per-iter
        )
        total = wins + losses + draws
        winrate = wins / max(1, (wins+losses))
        log_print(f"Gating (mini): W/L/D = {wins}/{losses}/{draws} (winrate={winrate:.2%})")
        tsv.log("gate_wins", wins); tsv.log("gate_losses", losses); tsv.log("gate_draws", draws)

        # Promote if good
        if (wins + losses) > 0 and winrate >= cfg['gate']['promote_win_rate']:
            champion = copy.deepcopy(net).to(device)
            ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], f"champion_iter{it}.pt")
            os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)
            torch.save(champion.state_dict(), ckpt_path)
            log_print(f"Promoted to champion! Saved: {ckpt_path}")
        elif it % cfg['selfplay']['save_every_iters'] == 0:
            # Save current net anyway
            ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], f"current_iter{it}.pt")
            os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)
            torch.save(net.state_dict(), ckpt_path)
            log_print(f"Saved current checkpoint: {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
