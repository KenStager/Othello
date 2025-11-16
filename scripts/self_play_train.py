import argparse, os, copy, torch, re, glob
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from src.utils.config import load_config
from src.utils.logger import log_print, TSVLogger
from src.utils.seed import set_seed
from src.othello.game import Game
from src.net.model import OthelloNet
from src.train.replay import ReplayBuffer, ILDataset
from src.train.selfplay import generate_selfplay
from src.train.trainer import train_steps
from src.train.evaluator import play_match
from src.train.diagnostics import Diagnostics
from src.train.stability_monitor import StabilityMonitor
from src.train.oracle import create_oracle

def setup_device(cfg):
    """
    Setup compute device with MPS support and memory management.
    Fallback order: CUDA → MPS → CPU
    """
    device_cfg = cfg['device']

    # CUDA (NVIDIA GPU)
    if torch.cuda.is_available() and device_cfg == "cuda":
        device = torch.device("cuda")
        log_print(f"Using device: {device}")
        return device

    # MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available() and device_cfg == "mps":
        device = torch.device("mps")

        # Set memory fraction to leave buffer for system (use 75% of recommended max)
        torch.mps.set_per_process_memory_fraction(0.75)

        # Enable fallback to CPU for unsupported ops
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

        log_print(f"Using device: {device}")

        # Log memory info if verbose
        if cfg.get('logging', {}).get('verbose', False):
            try:
                recommended = torch.mps.recommended_max_memory() / 1e9
                log_print(f"  MPS recommended max memory: {recommended:.2f} GB")
                log_print(f"  MPS memory fraction: 0.75 (target: {recommended * 0.75:.2f} GB)")
            except:
                pass  # Some PyTorch versions may not have these APIs

        return device

    # CPU (fallback)
    device = torch.device("cpu")
    if device_cfg != "cpu":
        log_print(f"Warning: {device_cfg} not available, falling back to CPU")
    else:
        log_print(f"Using device: {device}")

    return device

def find_latest_checkpoint(checkpoint_dir):
    """
    Dynamically find the most recent checkpoint file.

    Returns:
        (path, iteration) tuple if found, else (None, None)
    """
    pattern = os.path.join(checkpoint_dir, "current_iter*.pt")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        return None, None

    # Extract iteration numbers from filenames: current_iter2.pt → 2
    checkpoints_with_iters = []
    for path in checkpoint_files:
        basename = os.path.basename(path)
        match = re.search(r'current_iter(\d+)\.pt', basename)
        if match:
            iteration = int(match.group(1))
            checkpoints_with_iters.append((path, iteration))

    if not checkpoints_with_iters:
        return None, None

    # Return checkpoint with highest iteration number (most recent)
    latest_path, latest_iter = max(checkpoints_with_iters, key=lambda x: x[1])
    return latest_path, latest_iter

def verify_checkpoint(checkpoint, net, strict=False):
    """
    Validate checkpoint integrity (Phase 1: Checkpoint verification).

    Args:
        checkpoint: Loaded checkpoint dict
        net: Network to validate against
        strict: If True, raise errors; if False, return warnings

    Returns:
        (valid, warnings) tuple
    """
    warnings = []

    # Check required keys
    required = ['model_state', 'iteration']
    for key in required:
        if key not in checkpoint:
            msg = f"Missing required key: {key}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

    # Check state_dict compatibility
    if 'model_state' in checkpoint:
        model_keys = set(net.state_dict().keys())
        ckpt_keys = set(checkpoint['model_state'].keys())

        if model_keys != ckpt_keys:
            missing = model_keys - ckpt_keys
            extra = ckpt_keys - model_keys
            if missing:
                warnings.append(f"Missing keys in checkpoint: {list(missing)[:5]}")
            if extra:
                warnings.append(f"Extra keys in checkpoint: {list(extra)[:5]}")

        # Check for NaN/Inf
        for name, tensor in checkpoint['model_state'].items():
            if torch.is_tensor(tensor):
                if torch.isnan(tensor).any():
                    msg = f"NaN detected in {name}"
                    if strict:
                        raise ValueError(msg)
                    warnings.append(msg)
                if torch.isinf(tensor).any():
                    msg = f"Inf detected in {name}"
                    if strict:
                        raise ValueError(msg)
                    warnings.append(msg)

    return len(warnings) == 0, warnings

def main(cfg_path):
    print("\n" + "=" * 80, flush=True)
    print("🚀 RUNNING EDITED CODE - DEBUG VERSION 2.0 - LINES 391-395 SHOULD PRINT!", flush=True)
    print("=" * 80 + "\n", flush=True)
    cfg = load_config(cfg_path)
    set_seed(cfg['seed'])

    device = setup_device(cfg)

    # Model
    net = OthelloNet(in_channels=4, channels=cfg['model']['channels'],
                     residual_blocks=cfg['model']['residual_blocks']).to(device)
    champion = copy.deepcopy(net).to(device)

    # Check for existing checkpoint to resume from
    ckpt_path, latest_iter = find_latest_checkpoint(cfg['paths']['checkpoint_dir'])

    it = 0
    champ_loss_rate = 0.5  # Initial estimate for first gate
    champion_iter = 0  # Stage 2: Track which iteration champion comes from

    if ckpt_path:
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)

            # Handle both old (state_dict only) and new (dict with metadata) formats
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                # New format with metadata - verify if enabled
                if cfg.get('checkpoint', {}).get('verify_on_load', True):
                    valid, warnings = verify_checkpoint(checkpoint, net, strict=False)
                    if warnings:
                        log_print(f"Checkpoint validation warnings:")
                        for w in warnings:
                            log_print(f"  - {w}")

                # Load model state
                net.load_state_dict(checkpoint['model_state'])
                it = checkpoint['iteration']
                champ_loss_rate = checkpoint.get('champ_loss_rate', 0.5)
                log_print(f"Resumed from checkpoint: {os.path.basename(ckpt_path)} (iteration {it})")
                log_print(f"  Champion loss rate: {champ_loss_rate:.2%}")
            else:
                # Old format (just state_dict) - load weights but start iteration at 0
                net.load_state_dict(checkpoint)
                log_print(f"Loaded checkpoint (old format): {os.path.basename(ckpt_path)}")
                log_print(f"  Warning: No metadata found, starting iteration counter at 0")

            # Find and load latest champion checkpoint
            champion_files = glob.glob(os.path.join(cfg['paths']['checkpoint_dir'], "champion_iter*.pt"))
            if champion_files:
                # Extract iteration numbers and find max
                champion_iters = []
                for f in champion_files:
                    match = re.search(r'champion_iter(\d+)', f)
                    if match:
                        champion_iters.append((f, int(match.group(1))))

                if champion_iters:
                    latest_champ_path, latest_champ_iter = max(champion_iters, key=lambda x: x[1])

                    # Load latest champion
                    champ_checkpoint = torch.load(latest_champ_path, map_location=device)
                    champ_state = champ_checkpoint['model_state'] if isinstance(champ_checkpoint, dict) else champ_checkpoint
                    champion.load_state_dict(champ_state)
                    champion_iter = latest_champ_iter
                    log_print(f"  Loaded champion checkpoint: {os.path.basename(latest_champ_path)}")
                    log_print(f"  Champion age: {it - champion_iter} iterations")
                    if it - champion_iter > 20:
                        log_print(f"  ⚠️  WARNING: Champion is very old (>{it - champion_iter} iterations)")
                else:
                    # No valid champion files
                    champion = copy.deepcopy(net).to(device)
                    champion_iter = it
                    log_print(f"  No valid champion checkpoint found, using current as champion")
            else:
                # No champion checkpoints exist
                champion = copy.deepcopy(net).to(device)
                champion_iter = it
                log_print(f"  No champion checkpoint found, using current as champion")

        except Exception as e:
            log_print(f"Warning: Failed to load checkpoint {ckpt_path}: {e}")
            log_print(f"  Starting fresh from iteration 0")
            it = 0
            champ_loss_rate = 0.5
    else:
        log_print("No checkpoint found, starting fresh from iteration 0")

    # Replay (will auto-load latest shard)
    cleanup_cfg = cfg['train'].get('replay_cleanup', {})
    replay = ReplayBuffer(
        capacity=cfg['train']['replay_capacity'],
        save_dir=cfg['paths']['replay_dir'],
        cleanup_enabled=cleanup_cfg.get('enabled', True),
        cleanup_keep_recent=cleanup_cfg.get('keep_recent', 3),
        cleanup_keep_milestone_every=cleanup_cfg.get('keep_milestone_every', 50000)
    )

    # IL Dataset (Imitation Learning bootstrap from expert games)
    il_data = None
    il_cfg = cfg['train'].get('il_mixing', {})
    if il_cfg.get('enabled', False):
        try:
            il_data = ILDataset(cfg['paths']['il_data'])
            log_print(f"IL mixing enabled: {len(il_data):,} expert samples loaded")
            log_print(f"  Base ratio: {il_cfg.get('ratio', 0.2):.1%}, fade-out over {il_cfg.get('iters', 20)} iterations")
        except Exception as e:
            log_print(f"Warning: Failed to load IL data: {e}")
            log_print(f"  Continuing without IL mixing")
            il_data = None

    # Oracle (Endgame exact solver using Edax)
    oracle = create_oracle(cfg)
    if cfg.get('oracle', {}).get('use', False):
        log_print(f"Oracle enabled: Edax solver for endgame positions")
        log_print(f"  Threshold: empties <= {cfg['oracle']['empties_threshold']}")

    # Opening Suite (diverse starting positions for self-play)
    opening_suite = None
    opening_path = cfg['paths'].get('opening_suite', 'data/openings/rot64.json')
    if os.path.exists(opening_path):
        import json
        with open(opening_path, 'r') as f:
            opening_suite = json.load(f)
        log_print(f"Opening suite loaded: {len(opening_suite)} positions from {os.path.basename(opening_path)}")
    else:
        log_print(f"Warning: Opening suite not found at {opening_path}, using standard start only")

    # Logging
    os.makedirs("logs", exist_ok=True)
    tsv = TSVLogger("logs/train.tsv")

    # TensorBoard (Phase 1: Real-time monitoring)
    tb_writer = None
    if cfg.get('logging', {}).get('tensorboard', False):
        tb_dir = cfg.get('logging', {}).get('tensorboard_dir', 'runs')
        from datetime import datetime
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tb_writer = SummaryWriter(os.path.join(tb_dir, run_name))
        log_print(f"TensorBoard logging enabled: {tb_dir}/{run_name}")
        log_print(f"  View with: tensorboard --logdir={tb_dir} --port=6006")

    # Diagnostics (Phase 2.1: Enhanced training visibility)
    diagnostics = Diagnostics(window_size=1000)

    # Stability Monitor (Phase 2.2: Training stability tracking)
    stability_monitor = StabilityMonitor(window_size=100)

    # Optimizer (Phase 1: Persistent across iterations to retain AdamW momentum)
    import torch.optim as optim
    optimizer = optim.AdamW(net.parameters(),
                            lr=cfg['train']['lr'],
                            weight_decay=cfg['train']['weight_decay'])
    log_print(f"Optimizer: AdamW(lr={cfg['train']['lr']:.2e}, weight_decay={cfg['train']['weight_decay']:.2e})")

    # LR Scheduler (Phase 1: Cosine annealing with warm restarts for stability)
    scheduler = None
    scheduler_cfg = cfg.get('train', {}).get('scheduler', {})
    if scheduler_cfg.get('enabled', False):
        scheduler_type = scheduler_cfg.get('type', 'cosine_warmrestarts')

        if scheduler_type == 'cosine_warmrestarts':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            T_0 = scheduler_cfg.get('T_0', 10)
            T_mult = scheduler_cfg.get('T_mult', 1)
            eta_min = scheduler_cfg.get('eta_min', cfg['train']['lr_min'])

            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )
            log_print(f"LR Scheduler: CosineAnnealingWarmRestarts(T_0={T_0}, T_mult={T_mult}, eta_min={eta_min:.2e})")
        else:
            log_print(f"Warning: Unknown scheduler type '{scheduler_type}', using fixed LR")
    else:
        log_print("LR Scheduler: disabled (using fixed LR)")

    # Load optimizer and scheduler state from checkpoint (if exists)
    if ckpt_path:
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            if isinstance(checkpoint, dict):
                # Load optimizer state
                if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] and cfg.get('checkpoint', {}).get('save_optimizer', True):
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state'])
                        log_print(f"  Loaded optimizer state from checkpoint")
                    except Exception as e:
                        log_print(f"  Warning: Failed to load optimizer state: {e}")

                # Load scheduler state
                if 'scheduler_state' in checkpoint and checkpoint['scheduler_state'] and scheduler and cfg.get('checkpoint', {}).get('save_scheduler', True):
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state'])
                        log_print(f"  Loaded scheduler state from checkpoint")
                    except Exception as e:
                        log_print(f"  Warning: Failed to load scheduler state: {e}")
        except Exception as e:
            log_print(f"Warning: Failed to load optimizer/scheduler state: {e}")

    game_cls = lambda: Game(cfg['game']['board_size'])
    mcts_cfg = dict(
        cpuct=cfg['mcts']['cpuct'],
        simulations=cfg['mcts']['simulations'],
        dir_alpha=cfg['game']['dirichlet_alpha'],
        dir_frac=cfg['game']['dirichlet_frac'],
        reuse_tree=cfg['mcts']['reuse_tree'],
        tt_enabled=cfg['mcts'].get('tt_enabled', True),
        batch_size=cfg['mcts'].get('batch_size', 64),
        use_batching=cfg['mcts'].get('use_batching', False),
    )

    # Temperature schedule (3-phase)
    temp_schedule = cfg.get('selfplay', {}).get('temp_schedule', {
        'open_to': cfg['game'].get('temperature_moves', 12),
        'mid_to': 20,
        'open_tau': 1.0,
        'mid_tau': 0.25,
        'late_tau': 0.0
    })

    verbose = cfg.get('logging', {}).get('verbose', True)

    try:
        while True:
            it += 1
            import time
            iter_start = time.time()

            log_print(f"=== Iteration {it} ===")

            # 1) Self-play
            selfplay_start = time.time()
            added = generate_selfplay(
            replay=replay,
            game_cls=game_cls,
            net=net,
            device=device,
            mcts_cfg=mcts_cfg,
            games=cfg['selfplay']['games_per_iter'],
            temp_schedule=temp_schedule,
            max_moves=cfg['selfplay']['max_moves'],
            dir_alpha=cfg['game']['dirichlet_alpha'],
            dir_frac=cfg['game']['dirichlet_frac'],
            oracle=oracle,
            opening_suite=opening_suite,
            num_workers=cfg['selfplay'].get('num_workers', 1),
            verbose=verbose
            )
            print(f"🔍 DEBUG: generate_selfplay returned, added={added}", flush=True)
            selfplay_time = time.time() - selfplay_start
            print(f"🔍 DEBUG: About to log self-play summary", flush=True)
            log_print(f"Self-play added samples: {added} (buffer size={replay.size()}) [{selfplay_time:.1f}s]")
            print(f"🔍 DEBUG: Logged self-play summary", flush=True)
            tsv.log("replay_size", replay.size())

            # 2) Train (if enough data)
            print(f"🔍 DEBUG: Checking training condition: replay.size()={replay.size()}, min={cfg['train']['min_replay_to_train']}", flush=True)
            if replay.size() >= cfg['train']['min_replay_to_train']:
                print(f"🔍 DEBUG: Training condition MET, entering training block", flush=True)
                train_start = time.time()

                # Stage 2: Adaptive training steps - gentle start, ramp up
                if it <= 10:
                    train_steps_actual = 50
                elif it <= 20:
                    train_steps_actual = 100
                else:
                    train_steps_actual = cfg['train']['steps_per_iter']

                # Compute IL mixing ratio with fade-out
                il_ratio = 0.0
                if il_data and it <= il_cfg.get('iters', 20):
                    base_ratio = il_cfg.get('ratio', 0.2)
                    fade_factor = 1.0 - (it / il_cfg.get('iters', 20))
                    il_ratio = base_ratio * fade_factor

                if verbose:
                    if il_ratio > 0:
                        log_print(f"Training: {train_steps_actual} steps (batch size {cfg['train']['batch_size']}, IL ratio {il_ratio:.1%})")
                    else:
                        log_print(f"Training: {train_steps_actual} steps (batch size {cfg['train']['batch_size']})")

                avg_loss = train_steps(
                    net=net, replay=replay, device=device,
                    steps=train_steps_actual,
                    batch_size=cfg['train']['batch_size'],
                    optimizer=optimizer,  # Phase 1: Pass persistent optimizer
                    grad_clip=cfg['train']['grad_clip'],
                    phase_weighted_score=cfg['train'].get('phase_weighted_score', True),
                    score_weight_base=cfg['train'].get('score_weight_base', 0.3),
                    il_dataset=il_data,
                    il_ratio=il_ratio,
                    verbose=verbose,
                    log_interval=cfg.get('logging', {}).get('train_log_interval', 50),
                    diagnostics=diagnostics,
                    stability_monitor=stability_monitor
                )
                train_time = time.time() - train_start
                log_print(f"Train loss: {avg_loss:.4f} [{train_time:.1f}s]")
                tsv.log("train_loss", avg_loss)

                # Step LR scheduler (Phase 1)
                if scheduler:
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    log_print(f"  Learning rate: {current_lr:.2e}")

                    # TensorBoard logging
                    if tb_writer:
                        tb_writer.add_scalar('train/learning_rate', current_lr, it)

                # TensorBoard logging
                if tb_writer:
                    tb_writer.add_scalar('train/loss', avg_loss, it)

                # Stage 2: Diagnostic logging - value correlation and policy entropy
                if verbose and replay.size() >= 1000:
                    net.eval()
                    with torch.no_grad():
                        # Sample batch for diagnostics
                        sample_size = min(1000, replay.size())
                        import random
                        sample_indices = random.sample(range(replay.size()), sample_size)
                        sample_states = []
                        sample_values = []
                        for idx in sample_indices:
                            s = replay.buffer[idx]
                            sample_states.append(torch.from_numpy(s["state"]))
                            sample_values.append(s["value_win"])

                        states_batch = torch.stack(sample_states).to(device=device, dtype=torch.float32)
                        values_true = np.array(sample_values)

                        outputs = net(states_batch)
                        values_pred = outputs.value_win.cpu().numpy()

                        # Compute value correlation
                        if len(values_pred) > 1:
                            correlation = np.corrcoef(values_pred, values_true)[0, 1]
                            log_print(f"  Diagnostics: Value correlation={correlation:.3f}")
                            tsv.log("value_correlation", correlation)

                        # Compute policy entropy
                        policy_logits = outputs.policy_logits
                        policy_probs = torch.softmax(policy_logits, dim=1)
                        entropy = -(policy_probs * torch.log(policy_probs + 1e-10)).sum(dim=1).mean().item()
                        log_print(f"  Diagnostics: Policy entropy={entropy:.3f}")
                        tsv.log("policy_entropy", entropy)

                        # TensorBoard logging
                        if tb_writer:
                            tb_writer.add_scalar('diagnostics/value_correlation', correlation, it)
                            tb_writer.add_scalar('diagnostics/policy_entropy', entropy, it)

                    net.train()
            else:
                print(f"🔍 DEBUG: Training condition NOT MET, skipping training", flush=True)
                log_print("Not enough data to train yet.")

            # 3) Gate: pit current net vs champion
            gate_games = cfg['gate']['eval_games']
            verbose = cfg.get('logging', {}).get('verbose', True)

            if verbose and gate_games > 0:
                log_print(f"Gating: Running {gate_games} games (current iter {it} vs champion iter {champion_iter})")

            wins, losses, draws, _, game_details = play_match(
                game_cls=game_cls,
                net_a=net,
                net_b=champion,
                device=device,
                mcts_cfg=dict(cpuct=cfg['mcts']['cpuct'],
                              simulations=max(50, cfg['mcts']['simulations']//2),
                              dir_alpha=None, dir_frac=0.0, reuse_tree=False),
                games=gate_games,
                oracle=oracle,
                opening_suite=opening_suite,
                verbose=verbose
            )
            total = wins + losses + draws
            winrate = wins / max(1, (wins+losses)) if (wins+losses) > 0 else 0.0
            new_loss_rate = losses / max(1, total)

            if verbose and gate_games > 0:
                avg_moves = sum(d['moves'] for d in game_details) / len(game_details) if game_details else 0
                avg_score = sum(abs(d['score']) for d in game_details if d['result'] != 0) / max(1, wins + losses)
                log_print(f"\n[Gating Summary]")
                log_print(f"  Results: {wins}W / {losses}L / {draws}D")
                log_print(f"  Win rate: {winrate:.1%} (threshold: {cfg['gate']['promote_win_rate']:.1%})")
                log_print(f"  Loss rate: {new_loss_rate:.1%} (max: {cfg['gate'].get('max_loss_rate_multiplier', 1.10) * champ_loss_rate:.1%})")
                log_print(f"  Avg moves: {avg_moves:.1f}, Avg score margin: ±{avg_score:.1f}\n")
            else:
                log_print(f"Gating (mini): W/L/D = {wins}/{losses}/{draws} (winrate={winrate:.2%}, loss_rate={new_loss_rate:.2%})")

            # Performance degradation detection
            warmup_iters = 5  # Define before use
            if it > warmup_iters and winrate < champ_loss_rate - 0.15:  # >15% performance drop
                log_print(f"⚠️  WARNING: Significant performance degradation detected!")
                log_print(f"  Previous champion loss rate: {champ_loss_rate:.1%}")
                log_print(f"  Current win rate vs champion: {winrate:.1%}")
                log_print(f"  Performance dropped by {(champ_loss_rate - winrate):.1%}")
                log_print(f"  Consider reverting to checkpoint iteration {champion_iter}\n")

            tsv.log("gate_wins", wins)
            tsv.log("gate_losses", losses)
            tsv.log("gate_draws", draws)
            tsv.log("gate_loss_rate", new_loss_rate)

            # TensorBoard logging
            if tb_writer:
                tb_writer.add_scalar('gate/win_rate', winrate, it)
                tb_writer.add_scalar('gate/loss_rate', new_loss_rate, it)
                tb_writer.add_scalar('data/buffer_size', replay.size(), it)
                if game_details:
                    avg_moves = sum(d['moves'] for d in game_details) / len(game_details)
                    avg_score = sum(abs(d['score']) for d in game_details if d['result'] != 0) / max(1, wins + losses)
                    tb_writer.add_scalar('gate/avg_moves', avg_moves, it)
                    tb_writer.add_scalar('gate/avg_score_margin', avg_score, it)

            # Promote if good: win rate >= threshold AND loss rate doesn't worsen
            # Stage 2: Champion warm-up - skip promotion for first 5 iterations (warmup_iters defined above)
            max_loss_mult = cfg['gate'].get('max_loss_rate_multiplier', 1.10)
            should_promote = (
                it > warmup_iters and  # Skip promotion during warm-up
                (wins + losses) > 0 and
                winrate >= cfg['gate']['promote_win_rate'] and
                new_loss_rate <= max_loss_mult * champ_loss_rate
            )

            if should_promote:
                champion = copy.deepcopy(net).to(device)
                champion_iter = it  # Stage 2: Update champion iteration tracking
                champ_loss_rate = new_loss_rate  # Update champion baseline
                ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], f"champion_iter{it}.pt")
                os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)

                # Save checkpoint with enhanced metadata (Phase 1)
                checkpoint = {
                    'model_state': champion.state_dict(),
                    'optimizer_state': optimizer.state_dict() if cfg.get('checkpoint', {}).get('save_optimizer', True) else None,
                    'scheduler_state': scheduler.state_dict() if scheduler and cfg.get('checkpoint', {}).get('save_scheduler', True) else None,
                    'iteration': it,
                    'champ_loss_rate': champ_loss_rate,
                    'buffer_size': replay.size(),
                    'timestamp': time.time(),
                    'config_snapshot': {  # Save key config for verification
                        'lr': cfg['train']['lr'],
                        'channels': cfg['model']['channels'],
                        'residual_blocks': cfg['model']['residual_blocks']
                    }
                }
                torch.save(checkpoint, ckpt_path)
                log_print(f"Promoted to champion! Saved: {ckpt_path} (loss_rate: {champ_loss_rate:.2%})")

                # TensorBoard logging
                if tb_writer:
                    tb_writer.add_scalar('gate/promoted', 1, it)

            # Flush TensorBoard metrics to disk after each iteration (Phase 1: Fix)
            if tb_writer:
                tb_writer.flush()

            if it % cfg['selfplay']['save_every_iters'] == 0:
                # Save current net anyway
                ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], f"current_iter{it}.pt")
                os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)

                # Save checkpoint with enhanced metadata (Phase 1)
                checkpoint = {
                    'model_state': net.state_dict(),
                    'optimizer_state': optimizer.state_dict() if cfg.get('checkpoint', {}).get('save_optimizer', True) else None,
                    'scheduler_state': scheduler.state_dict() if scheduler and cfg.get('checkpoint', {}).get('save_scheduler', True) else None,
                    'iteration': it,
                    'champ_loss_rate': champ_loss_rate,
                    'buffer_size': replay.size(),
                    'timestamp': time.time(),
                    'config_snapshot': {  # Save key config for verification
                        'lr': cfg['train']['lr'],
                        'channels': cfg['model']['channels'],
                        'residual_blocks': cfg['model']['residual_blocks']
                    }
                }
                torch.save(checkpoint, ckpt_path)
                log_print(f"Saved current checkpoint: {ckpt_path}")

                # Iteration summary
                iter_time = time.time() - iter_start
                if verbose:
                    log_print(f"[Iteration {it} complete in {iter_time/60:.1f} minutes]\n")
                else:
                    log_print("")
    finally:
        # Close TensorBoard writer on exit
        if tb_writer:
            tb_writer.close()
            log_print("TensorBoard writer closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
