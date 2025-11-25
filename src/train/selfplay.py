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

def move_temperature(ply, temp_schedule):
    """
    Returns temperature based on 3-phase schedule.

    Args:
        ply: Current ply (move number)
        temp_schedule: Dict with 'open_to', 'mid_to', 'open_tau', 'mid_tau', 'late_tau'
    """
    if ply <= temp_schedule.get('open_to', 12):
        return temp_schedule.get('open_tau', 1.0)
    elif ply <= temp_schedule.get('mid_to', 20):
        return temp_schedule.get('mid_tau', 0.25)
    else:
        return temp_schedule.get('late_tau', 0.0)

def apply_temperature(pi, tau):
    """Apply temperature to policy distribution."""
    if tau <= 0.0:
        # Deterministic: pick argmax
        probs = np.zeros_like(pi)
        probs[np.argmax(pi)] = 1.0
        return probs
    elif tau == 1.0:
        # No change
        return pi
    else:
        # Apply temperature: p_i^(1/tau) / sum(p_j^(1/tau))
        # Equivalent to: softmax(log(p) / tau)
        log_pi = np.log(pi + 1e-10)
        scaled = log_pi / tau
        scaled = scaled - scaled.max()  # numerical stability
        exp_scaled = np.exp(scaled)
        return exp_scaled / exp_scaled.sum()

def _play_games_worker(args):
    """
    Worker function for parallel self-play (ProcessPool - legacy).

    Each worker process:
    1. Reconstructs the neural network on CPU from state_dict
    2. Creates its own oracle instance (if enabled)
    3. Sets unique random seeds for reproducibility
    4. Plays N games using play_one_game()
    5. Returns all samples and game statistics

    Args:
        args: Tuple of (worker_id, num_games, net_state_dict, net_config,
              board_size, mcts_cfg, temp_schedule, max_moves, dir_alpha,
              dir_frac, oracle_cfg, opening_suite, base_seed)

    Returns:
        (all_samples, all_stats): Lists of samples and game metadata
    """
    import torch
    import random
    import time
    from ..net.model import OthelloNet
    from ..othello.board import BLACK, WHITE
    from ..othello.game import Game

    # Unpack arguments
    (worker_id, num_games, net_state_dict, net_config, board_size, mcts_cfg,
     temp_schedule, max_moves, dir_alpha, dir_frac, oracle_cfg,
     opening_suite, base_seed) = args

    # Create game_cls callable
    game_cls = lambda: Game(board_size)

    # Reconstruct model in worker process
    # Auto-detect CUDA (works on AWS/cloud) or fall back to CPU (for MPS compatibility)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = OthelloNet(
        in_channels=net_config.get('in_channels', 4),
        channels=net_config.get('channels', 64),
        residual_blocks=net_config.get('residual_blocks', 8),
        action_size=net_config.get('action_size', 65)
    ).to(device)
    net.load_state_dict(net_state_dict)
    net.eval()

    # Reconstruct oracle if enabled
    oracle = None
    if oracle_cfg and oracle_cfg.get('use'):
        from .oracle import EdaxOracle
        oracle = EdaxOracle(
            edax_path=oracle_cfg['edax_path'],
            empties_threshold=oracle_cfg['empties_threshold'],
            time_limit_ms=oracle_cfg.get('time_limit_ms', 100)
        )

    # Set unique random seed for this worker
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)

    # Play games
    all_samples = []
    all_stats = []

    for game_idx in range(num_games):
        start_time = time.time()
        traj, meta = play_one_game(
            game_cls, net, device, mcts_cfg, temp_schedule,
            max_moves, dir_alpha, dir_frac, oracle, opening_suite
        )
        elapsed = time.time() - start_time

        all_samples.extend(traj)
        all_stats.append({**meta, 'time': elapsed})

    return all_samples, all_stats


def _play_games_worker_threaded(args):
    """
    Thread-based worker function for parallel self-play with InferenceServer.

    Uses QueuedBatchEvaluator to submit inference requests to shared InferenceServer,
    enabling cross-game batching for 1.5-2× speedup.

    Each worker thread:
    1. Receives InferenceClient (no model reconstruction needed)
    2. Creates QueuedBatchEvaluator for MCTS
    3. Sets unique random seeds for reproducibility
    4. Plays N games using play_one_game_threaded()
    5. Returns all samples and game statistics

    Args:
        args: Tuple of (worker_id, num_games, inference_client, board_size,
              mcts_cfg, temp_schedule, max_moves, dir_alpha, dir_frac,
              oracle_cfg, opening_suite, base_seed)

    Returns:
        (all_samples, all_stats): Lists of samples and game metadata
    """
    import random
    import time
    from ..othello.game import Game
    from ..mcts.batch_evaluator import QueuedBatchEvaluator

    # Unpack arguments
    (worker_id, num_games, inference_client, board_size, mcts_cfg,
     temp_schedule, max_moves, dir_alpha, dir_frac, oracle_cfg,
     opening_suite, base_seed) = args

    # Create game_cls callable
    game_cls = lambda: Game(board_size)

    # Create evaluator from inference client
    evaluator = QueuedBatchEvaluator(inference_client)

    # Reconstruct oracle if enabled
    oracle = None
    if oracle_cfg and oracle_cfg.get('use'):
        from .oracle import EdaxOracle
        oracle = EdaxOracle(
            edax_path=oracle_cfg['edax_path'],
            empties_threshold=oracle_cfg['empties_threshold'],
            time_limit_ms=oracle_cfg.get('time_limit_ms', 100)
        )

    # Set unique random seed for this worker
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)

    # Play games
    all_samples = []
    all_stats = []

    for game_idx in range(num_games):
        start_time = time.time()
        traj, meta = play_one_game_threaded(
            game_cls, evaluator, mcts_cfg, temp_schedule,
            max_moves, dir_alpha, dir_frac, oracle, opening_suite
        )
        elapsed = time.time() - start_time

        all_samples.extend(traj)
        all_stats.append({**meta, 'time': elapsed})

    return all_samples, all_stats

def play_one_game(game_cls, net, device, mcts_cfg, temp_schedule=None, max_moves=120, dir_alpha=0.15, dir_frac=0.25, oracle=None, opening_suite=None):
    """
    Play one self-play game.

    Args:
        temp_schedule: Dict with temperature parameters. If None, uses simple threshold at move 12.
        oracle: Optional endgame oracle for exact evaluation when empties <= threshold.
        opening_suite: Optional list of opening positions to sample from.
    """
    if temp_schedule is None:
        # Fallback to old behavior
        temp_schedule = {'open_to': 12, 'mid_to': 20, 'open_tau': 1.0, 'mid_tau': 0.25, 'late_tau': 0.0}

    game = game_cls()

    # Sample random opening position if suite available
    if opening_suite:
        import random
        from scripts.make_opening_suite import dict_to_board
        opening_dict = random.choice(opening_suite)
        b = dict_to_board(opening_dict)
    else:
        b = game.new_board()
    mcts = MCTS(game_cls, net, device,
                cpuct=mcts_cfg['cpuct'],
                simulations=mcts_cfg['simulations'],
                dir_alpha=dir_alpha,
                dir_frac=dir_frac,
                reuse_tree=mcts_cfg.get('reuse_tree', True),
                use_tt=mcts_cfg.get('tt_enabled', True),
                batch_size=mcts_cfg.get('batch_size', 32),
                use_batching=mcts_cfg.get('use_batching', True))
    states, policies, players, board_snapshots = [], [], [], []
    move_count = 0
    while not b.is_terminal() and move_count < max_moves:
        # Check if oracle should be used for this position
        if oracle and oracle.should_use(b):
            # Use oracle for exact endgame evaluation
            oracle_result = oracle.evaluate(b)
            if oracle_result.get('is_exact') and oracle_result.get('best_move') is not None:
                # Create one-hot policy for oracle move
                pi = np.zeros(65, dtype=np.float32)
                pi[oracle_result['best_move']] = 1.0
            else:
                # Oracle failed, fallback to MCTS
                pi = mcts.run(b)
        else:
            # Normal MCTS
            pi = mcts.run(b)

        # Apply 3-phase temperature schedule
        tau = move_temperature(move_count, temp_schedule)
        probs = apply_temperature(pi, tau)

        states.append(b.encode())
        policies.append(probs)
        players.append(b.player)
        board_snapshots.append(b.copy())

        # Sample from temperature-adjusted distribution
        a = int(np.random.choice(len(probs), p=probs))
        b.step_action_index(a)
        move_count += 1

        # If no legal move for next player AND current also had none earlier, pass logic is handled in board

    # Game end
    winner, diff = b.result()

    # Collect game metadata
    game_length = move_count
    phase_counts = {'opening': 0, 'midgame': 0, 'endgame': 0}
    for snapshot in board_snapshots:
        empties = int(np.sum(snapshot.board == 0))
        if empties >= 45:
            phase_counts['opening'] += 1
        elif empties <= 14:
            phase_counts['endgame'] += 1
        else:
            phase_counts['midgame'] += 1

    # Get MCTS statistics if available
    tt_stats = mcts.get_tt_stats() if hasattr(mcts, 'get_tt_stats') else {}

    game_meta = {
        'winner': winner,
        'score_diff': diff,
        'length': game_length,
        'phase_counts': phase_counts,
        'tt_stats': tt_stats
    }

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

        # Phase tagging based on empties
        empties = int(np.sum(snapshot.board == 0))
        if empties >= 45:
            phase = "opening"
        elif empties <= 14:
            phase = "endgame"
        else:
            phase = "midgame"

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
                "phase": phase,
                "empties": empties,
            }
            traj.append(sample)
    return traj, game_meta


def play_one_game_threaded(game_cls, evaluator, mcts_cfg, temp_schedule=None, max_moves=120, dir_alpha=0.15, dir_frac=0.25, oracle=None, opening_suite=None):
    """
    Play one self-play game using a pre-configured evaluator (for threaded execution).

    Args:
        game_cls: Game class constructor
        evaluator: Batch evaluator instance (QueuedBatchEvaluator with InferenceClient)
        mcts_cfg: MCTS configuration dict
        temp_schedule: Dict with temperature parameters. If None, uses simple threshold at move 12.
        max_moves: Maximum moves before declaring draw
        dir_alpha: Dirichlet noise alpha parameter
        dir_frac: Dirichlet noise fraction at root
        oracle: Optional endgame oracle for exact evaluation when empties <= threshold
        opening_suite: Optional list of opening positions to sample from

    Returns:
        (trajectory, game_metadata) tuple
    """
    if temp_schedule is None:
        # Fallback to old behavior
        temp_schedule = {'open_to': 12, 'mid_to': 20, 'open_tau': 1.0, 'mid_tau': 0.25, 'late_tau': 0.0}

    game = game_cls()

    # Sample random opening position if suite available
    if opening_suite:
        import random
        from scripts.make_opening_suite import dict_to_board
        opening_dict = random.choice(opening_suite)
        b = dict_to_board(opening_dict)
    else:
        b = game.new_board()

    # Create MCTS with evaluator instead of (net, device)
    mcts = MCTS(game_cls, net=None, device=None,
                cpuct=mcts_cfg['cpuct'],
                simulations=mcts_cfg['simulations'],
                dir_alpha=dir_alpha,
                dir_frac=dir_frac,
                batch_evaluator=evaluator,
                reuse_tree=mcts_cfg.get('reuse_tree', True),
                use_tt=mcts_cfg.get('tt_enabled', True))

    states, policies, players, board_snapshots = [], [], [], []
    move_count = 0
    while not b.is_terminal() and move_count < max_moves:
        # Check if oracle should be used for this position
        if oracle and oracle.should_use(b):
            # Use oracle for exact endgame evaluation
            oracle_result = oracle.evaluate(b)
            if oracle_result.get('is_exact') and oracle_result.get('best_move') is not None:
                # Create one-hot policy for oracle move
                pi = np.zeros(65, dtype=np.float32)
                pi[oracle_result['best_move']] = 1.0
            else:
                # Oracle failed, fallback to MCTS
                pi = mcts.run(b)
        else:
            # Normal MCTS
            pi = mcts.run(b)

        # Apply 3-phase temperature schedule
        tau = move_temperature(move_count, temp_schedule)
        probs = apply_temperature(pi, tau)

        states.append(b.encode())
        policies.append(probs)
        players.append(b.player)
        board_snapshots.append(b.copy())

        # Sample from temperature-adjusted distribution
        a = int(np.random.choice(len(probs), p=probs))
        b.step_action_index(a)
        move_count += 1

    # Game end
    winner, diff = b.result()

    # Collect game metadata
    game_length = move_count
    phase_counts = {'opening': 0, 'midgame': 0, 'endgame': 0}
    for snapshot in board_snapshots:
        empties = int(np.sum(snapshot.board == 0))
        if empties >= 45:
            phase_counts['opening'] += 1
        elif empties <= 14:
            phase_counts['endgame'] += 1
        else:
            phase_counts['midgame'] += 1

    # Get MCTS statistics if available
    tt_stats = mcts.get_tt_stats() if hasattr(mcts, 'get_tt_stats') else {}

    game_meta = {
        'winner': winner,
        'score_diff': diff,
        'length': game_length,
        'phase_counts': phase_counts,
        'tt_stats': tt_stats
    }

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

        # Phase tagging based on empties
        empties = int(np.sum(snapshot.board == 0))
        if empties >= 45:
            phase = "opening"
        elif empties <= 14:
            phase = "endgame"
        else:
            phase = "midgame"

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
                "phase": phase,
                "empties": empties,
            }
            traj.append(sample)
    return traj, game_meta


def generate_selfplay(replay, game_cls, net, device, mcts_cfg, games=20, temp_schedule=None, max_moves=120, dir_alpha=0.15, dir_frac=0.25, oracle=None, opening_suite=None, num_workers=1, model_cfg=None, verbose=True, use_inference_server=False):
    """
    Generate self-play games.

    Args:
        temp_schedule: Dict with 'open_to', 'mid_to', 'open_tau', 'mid_tau', 'late_tau'.
                      If None, uses default 3-phase schedule.
        oracle: Optional endgame oracle for exact evaluation
        opening_suite: Optional list of opening positions to sample from
        num_workers: Number of parallel workers (1=serial, >1=parallel)
        model_cfg: Model configuration dict (for ProcessPool worker reconstruction)
        verbose: If True, print game-by-game progress
        use_inference_server: If True, use ThreadPoolExecutor + InferenceServer (Phase 2.2)
                             If False, use ProcessPoolExecutor (legacy mode)
    """
    import time
    from ..othello.board import BLACK, WHITE

    # Parallel path with InferenceServer (Phase 2.2)
    if num_workers > 1 and use_inference_server:
        from concurrent.futures import ThreadPoolExecutor
        from ..mcts.inference_server import InferenceServer, InferenceClient

        # Distribute games across workers
        games_per_worker = [games // num_workers] * num_workers
        remainder = games % num_workers
        for i in range(remainder):
            games_per_worker[i] += 1

        # Create centralized InferenceServer
        # All workers will share this single GPU model via thread-safe queuing
        if verbose:
            print(f"  Starting InferenceServer (max_batch_size={mcts_cfg.get('batch_size', 64)}, batch_timeout=10ms)")

        inference_server = InferenceServer(
            model=net,
            device=device,
            max_batch_size=mcts_cfg.get('batch_size', 64),
            batch_timeout_ms=10.0
        )
        inference_server.start()

        # Oracle configuration
        oracle_cfg = None
        if oracle and hasattr(oracle, 'edax_path'):
            # Real EdaxOracle (DummyOracle doesn't have edax_path)
            oracle_cfg = {
                'use': True,
                'edax_path': oracle.edax_path,
                'empties_threshold': oracle.empties_threshold,
                'time_limit_ms': oracle.time_limit_ms
            }

        base_seed = int(time.time() * 1000) % (2**31)

        # Extract board_size from game_cls
        try:
            test_game = game_cls()
            board_size = test_game.n
        except:
            board_size = 8

        # Prepare worker arguments
        # Each worker gets its own InferenceClient instance
        worker_args = [
            (worker_id, games_per_worker[worker_id], InferenceClient(inference_server),
             board_size, mcts_cfg, temp_schedule, max_moves, dir_alpha, dir_frac,
             oracle_cfg, opening_suite, base_seed)
            for worker_id in range(num_workers)
        ]

        # Execute in parallel with threads
        if verbose:
            print(f"  Threaded self-play: {num_workers} workers, {games} games total")

        parallel_start = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_play_games_worker_threaded, worker_args))
        parallel_time = time.time() - parallel_start

        # Shutdown InferenceServer
        server_stats = inference_server.get_stats()
        inference_server.shutdown()

        if verbose:
            print(f"  InferenceServer stats: {server_stats['total_requests']} requests, "
                  f"{server_stats['total_batches']} batches, "
                  f"avg_batch_size={server_stats['avg_batch_size']:.1f}")

        # Aggregate results and add to replay buffer
        total_samples = 0
        game_stats = []
        for samples_list, stats_list in results:
            for sample in samples_list:
                replay.add(sample)
                total_samples += 1
            game_stats.extend(stats_list)

        if verbose:
            print(f"  Threaded execution completed in {parallel_time:.1f}s ({parallel_time/games:.1f}s per game)")

        # Fall through to summary printing

    # Parallel path: multiprocess self-play with CPU workers (legacy)
    elif num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        # Distribute games across workers
        games_per_worker = [games // num_workers] * num_workers
        remainder = games % num_workers
        for i in range(remainder):
            games_per_worker[i] += 1

        # Prepare worker arguments
        # IMPORTANT: Move model to CPU before state_dict to avoid MPS pickling issues
        net_cpu = net.cpu()
        net_state_dict = net_cpu.state_dict()
        # Move model back to original device for main process training
        net.to(device)

        # Use model_cfg if provided, otherwise use defaults
        if model_cfg:
            net_config = {
                'in_channels': model_cfg.get('in_channels', 4),
                'channels': model_cfg.get('channels', 64),
                'residual_blocks': model_cfg.get('residual_blocks', 8),
                'action_size': model_cfg.get('action_size', 65)
            }
        else:
            net_config = {
                'in_channels': 4,
                'channels': 64,
                'residual_blocks': 8,
                'action_size': 65
            }

        # Oracle configuration
        oracle_cfg = None
        if oracle and hasattr(oracle, 'edax_path'):
            # Real EdaxOracle (DummyOracle doesn't have edax_path)
            oracle_cfg = {
                'use': True,
                'edax_path': oracle.edax_path,
                'empties_threshold': oracle.empties_threshold,
                'time_limit_ms': oracle.time_limit_ms
            }

        base_seed = int(time.time() * 1000) % (2**31)

        # Extract board_size from game_cls callable (it's a lambda)
        # We need to pass it separately since lambdas can't be pickled
        try:
            # Try to get board_size from a test call
            test_game = game_cls()
            board_size = test_game.n
        except:
            # Fallback to default
            board_size = 8

        worker_args = [
            (worker_id, games_per_worker[worker_id], net_state_dict, net_config,
             board_size, mcts_cfg, temp_schedule, max_moves, dir_alpha, dir_frac,
             oracle_cfg, opening_suite, base_seed)
            for worker_id in range(num_workers)
        ]

        # Execute in parallel
        if verbose:
            print(f"  Parallel self-play: {num_workers} workers, {games} games total")

        parallel_start = time.time()
        # Use 'spawn' method for CUDA compatibility in worker processes
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            results = list(executor.map(_play_games_worker, worker_args))
        parallel_time = time.time() - parallel_start

        # Aggregate results and add to replay buffer
        total_samples = 0
        game_stats = []
        for samples_list, stats_list in results:
            for sample in samples_list:
                replay.add(sample)
                total_samples += 1
            game_stats.extend(stats_list)

        if verbose:
            print(f"  Parallel execution completed in {parallel_time:.1f}s ({parallel_time/games:.1f}s per game)")

        # Print summary statistics (reuse existing summary code below)
        # Fall through to summary printing

    # Serial path: original single-threaded implementation
    else:
        total_samples = 0
        game_stats = []

        for game_num in range(games):
            start_time = time.time()
            traj, meta = play_one_game(game_cls, net, device, mcts_cfg, temp_schedule, max_moves, dir_alpha, dir_frac, oracle, opening_suite)
            elapsed = time.time() - start_time

            # Add samples to replay
            for sample in traj:
                replay.add(sample)
                total_samples += 1

            # Store stats
            game_stats.append({**meta, 'time': elapsed})

            # Verbose output
            if verbose:
                winner = meta['winner']
                score = meta['score_diff']
                length = meta['length']
                phases = meta['phase_counts']

                winner_str = "BLACK wins" if winner == BLACK else ("WHITE wins" if winner == WHITE else "Draw")
                score_str = f"{score:+d}" if winner != 0 else "+0"
                phase_str = f"[{phases['opening']}/{phases['midgame']}/{phases['endgame']}]"

                print(f"  Game {game_num+1}/{games}: {winner_str} {score_str} ({length} moves, {elapsed:.1f}s) {phase_str}")

    # Summary statistics
    if verbose and games > 0:
        black_wins = sum(1 for s in game_stats if s['winner'] == BLACK)
        white_wins = sum(1 for s in game_stats if s['winner'] == WHITE)
        draws = sum(1 for s in game_stats if s['winner'] == 0)
        avg_length = sum(s['length'] for s in game_stats) / len(game_stats)
        avg_score = sum(abs(s['score_diff']) for s in game_stats if s['score_diff'] != 0) / max(1, black_wins + white_wins)
        total_time = sum(s['time'] for s in game_stats)

        # Phase distribution across all games
        total_opening = sum(s['phase_counts']['opening'] for s in game_stats)
        total_midgame = sum(s['phase_counts']['midgame'] for s in game_stats)
        total_endgame = sum(s['phase_counts']['endgame'] for s in game_stats)
        total_moves = total_opening + total_midgame + total_endgame

        # MCTS stats (if available)
        tt_hits = sum(s['tt_stats'].get('hits', 0) for s in game_stats if s['tt_stats'])
        tt_misses = sum(s['tt_stats'].get('misses', 0) for s in game_stats if s['tt_stats'])
        tt_hit_rate = tt_hits / max(1, tt_hits + tt_misses) * 100

        print(f"\n[Self-Play Summary]")
        print(f"  Games: {games} ({black_wins} BLACK, {white_wins} WHITE, {draws} draws)")
        print(f"  Avg length: {avg_length:.1f} moves")
        print(f"  Avg score margin: ±{avg_score:.1f} discs")
        print(f"  Phases: opening {total_opening/total_moves*100:.1f}%, midgame {total_midgame/total_moves*100:.1f}%, endgame {total_endgame/total_moves*100:.1f}%")
        if tt_hits + tt_misses > 0:
            print(f"  MCTS stats: TT hit rate {tt_hit_rate:.1f}%")
        print(f"  Samples added: {total_samples} (with 8× symmetries)")
        print(f"  Time: {total_time/60:.1f} minutes\n")

    print(f"🔥 DEBUG INSIDE generate_selfplay: About to return total_samples={total_samples}", flush=True)
    return total_samples
