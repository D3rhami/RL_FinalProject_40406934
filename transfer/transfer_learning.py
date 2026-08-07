import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.q_learning import QLearningAgent
from agents.value_iteration import ValueIteration
from environments.generator import generate_target_maps, load_map
from environments.maze import CellType, MazeEnv, load_config
from experiments.logger import EpisodeLogger
from experiments.run_experiments import (
    EVAL_REWARD_MODE, _ledger_append, _rolling_success_episode, _save_json,
    derive_seeds, evaluate_greedy_policy,
)

SOURCE_MAP = 'environments/maps/source_maze.json'
SIMILAR_MAP = 'environments/maps/target_similar.json'
DIFFERENT_MAP = 'environments/maps/target_different.json'


def build_initial_q(source_Q, scenario, n_states_target, beta=None,
                    unchanged_state_indices=None):
    if scenario == 'scratch':
        return np.zeros((n_states_target, 4))
    if scenario == 'full':
        return np.array(source_Q, dtype=np.float64, copy=True)
    if scenario == 'scaled':
        return float(beta) * np.array(source_Q, dtype=np.float64)
    if scenario == 'selective':
        Q = np.zeros((n_states_target, 4))
        if unchanged_state_indices is not None and len(unchanged_state_indices):
            idx = np.asarray(unchanged_state_indices, dtype=int)
            Q[idx] = source_Q[idx]
        return Q
    raise ValueError(scenario)


def _neighborhood_sig(grid, r, c):
    size = grid.shape[0]
    sig = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                sig.append(1 if int(grid[nr, nc]) == CellType.WALL else 0)
            else:
                sig.append(-1)
    return tuple(sig)


def unchanged_state_indices(source_env, target_env):
    src, tgt = source_env.grid, target_env.grid
    size = source_env.maze_size
    positions = []
    for r in range(size):
        for c in range(size):
            if int(src[r, c]) == CellType.WALL and int(tgt[r, c]) == CellType.WALL:
                continue
            if _neighborhood_sig(src, r, c) == _neighborhood_sig(tgt, r, c):
                positions.append((r, c))
    indices = []
    for r, c in positions:
        for has_key in (0, 1):
            for energy in range(source_env.n_energy_levels):
                indices.append(source_env.encode_state((r, c, has_key, energy)))
    return np.array(indices, dtype=int)


def _load_source_q_full(model_path, env):
    with open(model_path) as f:
        model = json.load(f)
    Q = np.zeros((env.n_states, 4), dtype=np.float64)
    for s, q in zip(model['states'], model['Q']):
        Q[env.encode_state(tuple(s))] = q
    return Q, model


def pick_best_shaped_ql_model(cfg):
    summary = pd.read_csv('results/raw_data/q_learning/q_learning_summary.csv')
    shaped = summary[summary['reward_mode'] == 'shaped']
    if shaped.empty:
        raise FileNotFoundError('No shaped Q-Learning runs in q_learning_summary.csv')
    best = shaped.loc[shaped['eval_success_rate'].idxmax()]
    path = (f"results/models/q_learning/"
            f"q_learning_shaped_{best['schedule']}_seed{int(best['seed'])}.json")
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return path, best


def ensure_target_maps(cfg):
    if not (Path(SIMILAR_MAP).exists() and Path(DIFFERENT_MAP).exists()):
        print('Generating target maps...')
        generate_target_maps(cfg)
    for path in (SIMILAR_MAP, DIFFERENT_MAP):
        data = load_map(path)
        if not Path(path).exists():
            raise FileNotFoundError(path)
        print(f"  {path}: key={data['key_pos']} door={data['door_pos']} "
              f"goal={data['goal_pos']} validated")


def _scenario_specs(tcfg):
    specs = [
        ('scratch', 'scratch', None),
        ('full', 'full', None),
    ]
    for beta in tcfg['beta_values']:
        specs.append((f'scaled_{beta}', 'scaled', float(beta)))
    specs.append(('selective', 'selective', None))
    return specs


def run_transfer_experiments(cfg):
    print('=== Transfer Learning (Q-Learning) ===')
    tcfg = cfg['transfer']
    ensure_target_maps(cfg)
    seeds, eval_seed = derive_seeds(cfg['base_seed'], n=cfg['experiment_grid']['n_seeds'])
    source_model_path, best_row = pick_best_shaped_ql_model(cfg)
    print(f"Source Q: {source_model_path} "
          f"(eval_success={best_row['eval_success_rate']:.2f})")

    source_env = MazeEnv(SOURCE_MAP, config=cfg, reward_mode=tcfg['reward_mode'])
    source_Q, source_model = _load_source_q_full(source_model_path, source_env)

    targets = [('similar', SIMILAR_MAP), ('different', DIFFERENT_MAP)]
    training_rows, summary_rows = [], []
    Path('results/models/transfer').mkdir(parents=True, exist_ok=True)
    Path('results/raw_data/transfer').mkdir(parents=True, exist_ok=True)

    for target_name, map_path in targets:
        target_env_probe = MazeEnv(map_path, config=cfg, reward_mode=tcfg['reward_mode'])
        unchanged = unchanged_state_indices(source_env, target_env_probe)
        print(f"  {target_name}: unchanged state indices={len(unchanged)}")

        for scenario_id, scenario, beta in _scenario_specs(tcfg):
            for seed in seeds:
                variant_id = f'transfer_{target_name}_{scenario_id}_seed{seed}'
                t0 = time.time()
                try:
                    env = MazeEnv(map_path, config=cfg,
                                  reward_mode=tcfg['reward_mode'], seed=seed)
                    initial_Q = build_initial_q(
                        source_Q, scenario, env.n_states, beta=beta,
                        unchanged_state_indices=unchanged)
                    agent = QLearningAgent(
                        env, alpha=tcfg['alpha'], gamma=tcfg['gamma'],
                        epsilon_start=tcfg['epsilon_start'],
                        epsilon_min=tcfg['epsilon_min'],
                        decay_type=tcfg['decay_type'],
                        decay_param=tcfg['decay_param'],
                        num_episodes=tcfg['num_episodes'],
                        reward_mode=tcfg['reward_mode'],
                        initial_Q=initial_Q, seed=seed,
                    )
                    logger = EpisodeLogger(
                        f'results/raw_data/transfer/_tmp_{variant_id}.csv')
                    agent.train(logger=logger)

                    per_success, per_return = [], []
                    for row in logger.summary_rows:
                        per_success.append(int(row['success']))
                        per_return.append(row['total_reward'])
                        training_rows.append({
                            'target': target_name, 'scenario': scenario_id,
                            'seed': seed, 'episode': row['episode_idx'],
                            'epsilon': row['epsilon_final'],
                            'steps': row['steps'], 'return': row['total_reward'],
                            'success': row['success'],
                            'wall_hits': row['wall_hits'],
                            'penalty_entries': row['penalty_entries'],
                            'termination_reason': row['termination_reason'],
                        })

                    eval_env = MazeEnv(map_path, config=cfg,
                                       reward_mode=EVAL_REWARD_MODE)

                    def act_fn(state, _a=agent):
                        return int(np.argmax(_a.Q[_a.env.encode_state(state)]))

                    eval_metrics = evaluate_greedy_policy(
                        eval_env, act_fn, eval_seed, tcfg['eval_episodes'])

                    states = env.all_states()
                    state_indices = [env.encode_state(s) for s in states]
                    model_path = f'results/models/transfer/{variant_id}.json'
                    _save_json(model_path, {
                        'algorithm': 'transfer_q_learning',
                        'target': target_name, 'scenario': scenario_id,
                        'beta': beta, 'seed': seed,
                        'source_model': source_model_path,
                        'reward_mode': tcfg['reward_mode'],
                        'episodes': tcfg['num_episodes'],
                        'epsilon_start': tcfg['epsilon_start'],
                        'epsilon_min': tcfg['epsilon_min'],
                        'decay_type': tcfg['decay_type'],
                        'decay_param': tcfg['decay_param'],
                        'n_unchanged_indices': int(len(unchanged)),
                        'states': [list(s) for s in states],
                        'Q': agent.Q[state_indices].tolist(),
                        'visits': agent.visits[state_indices].tolist(),
                    })

                    jump_n = min(100, len(per_return))
                    first_success = next(
                        (i for i, s in enumerate(per_success) if s == 1), None)
                    summary_rows.append({
                        'target': target_name, 'scenario': scenario_id,
                        'beta': beta, 'seed': seed,
                        'source_model': source_model_path,
                        'first_success_episode': first_success,
                        'episodes_to_90pct_success': _rolling_success_episode(per_success),
                        'jumpstart_mean_return': float(np.mean(per_return[:jump_n])),
                        'jumpstart_success_rate': float(np.mean(per_success[:jump_n])),
                        'final_train_success_rate': float(np.mean(per_success[-100:])),
                        'final_train_mean_return': float(np.mean(per_return[-100:])),
                        **eval_metrics,
                        'visited_states': int(np.sum(agent.visits > 0)),
                    })
                    Path(f'results/raw_data/transfer/_tmp_{variant_id}.csv').unlink(
                        missing_ok=True)
                    _ledger_append('transfer', variant_id, seed, 'success',
                                   time.time() - t0, output_files=model_path)
                    print(f"  {variant_id}: eval={eval_metrics['eval_success_rate']:.2f} "
                          f"jump={summary_rows[-1]['jumpstart_mean_return']:.1f} "
                          f"final_train={summary_rows[-1]['final_train_success_rate']:.2f}")
                except Exception as e:
                    _ledger_append('transfer', variant_id, seed, 'failed',
                                   time.time() - t0, error_message=str(e))
                    print(f"  FAILED {variant_id}: {e}")
                    traceback.print_exc()

    if training_rows:
        p = Path('results/raw_data/transfer/transfer_training.csv')
        with open(p, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(training_rows[0].keys()))
            writer.writeheader()
            writer.writerows(training_rows)
        print(f'  saved {p}')
    if summary_rows:
        p = Path('results/raw_data/transfer/transfer_summary.csv')
        with open(p, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f'  saved {p}')

    run_negative_transfer_case(cfg, source_Q, source_model_path)
    return summary_rows


def run_negative_transfer_case(cfg, source_Q=None, source_model_path=None):
    print('=== Negative-transfer case study (different target) ===')
    tcfg = cfg['transfer']
    ensure_target_maps(cfg)
    if source_Q is None or source_model_path is None:
        source_model_path, _ = pick_best_shaped_ql_model(cfg)
        source_env = MazeEnv(SOURCE_MAP, config=cfg, reward_mode=tcfg['reward_mode'])
        source_Q, _ = _load_source_q_full(source_model_path, source_env)

    source_data = load_map(SOURCE_MAP)
    old_key = tuple(source_data['key_pos'])

    target_env = MazeEnv(DIFFERENT_MAP, config=cfg, reward_mode='sparse')
    vi = ValueIteration(target_env, gamma=cfg['value_iteration']['gamma'],
                        theta=cfg['value_iteration']['theta'], reward_mode='sparse')
    V, n_iter, converged, _ = vi.run(max_iterations=2000)
    vi_policy = vi.extract_policy(V)
    print(f"  VI on different target: converged={converged} iters={n_iter}")

    worst = None
    for s, vi_a in vi_policy.items():
        r, c, hk, en = s
        if abs(r - old_key[0]) + abs(c - old_key[1]) > 2:
            continue
        if vi_a < 0:
            continue
        idx = target_env.encode_state(s)
        transferred_a = int(np.argmax(source_Q[idx]))
        if transferred_a == vi_a:
            continue
        gap = vi._q_value(V, s, vi_a) - vi._q_value(V, s, transferred_a)
        if worst is None or gap > worst['action_gap']:
            worst = {
                'state': list(s), 'vi_action': int(vi_a),
                'transferred_action': transferred_a,
                'action_gap': float(gap),
                'q_transferred': source_Q[idx].tolist(),
                'dist_to_old_key': abs(r - old_key[0]) + abs(c - old_key[1]),
            }

    if worst is None:
        for s, vi_a in vi_policy.items():
            if vi_a < 0:
                continue
            idx = target_env.encode_state(s)
            transferred_a = int(np.argmax(source_Q[idx]))
            if transferred_a == vi_a:
                continue
            gap = vi._q_value(V, s, vi_a) - vi._q_value(V, s, transferred_a)
            if worst is None or gap > worst['action_gap']:
                worst = {
                    'state': list(s), 'vi_action': int(vi_a),
                    'transferred_action': transferred_a,
                    'action_gap': float(gap),
                    'q_transferred': source_Q[idx].tolist(),
                    'dist_to_old_key': None,
                }

    if worst is None:
        print('  No disagreement found for negative-transfer case')
        return None

    case_state = tuple(worst['state'])
    case_idx = target_env.encode_state(case_state)
    checkpoints = [0, 100, 500, 1000, 2500, tcfg['num_episodes']]
    checkpoints = sorted({c for c in checkpoints if c <= tcfg['num_episodes']})

    train_env = MazeEnv(DIFFERENT_MAP, config=cfg,
                        reward_mode=tcfg['reward_mode'], seed=cfg['base_seed'])
    agent = QLearningAgent(
        train_env, alpha=tcfg['alpha'], gamma=tcfg['gamma'],
        epsilon_start=tcfg['epsilon_start'], epsilon_min=tcfg['epsilon_min'],
        decay_type=tcfg['decay_type'], decay_param=tcfg['decay_param'],
        num_episodes=tcfg['num_episodes'], reward_mode=tcfg['reward_mode'],
        initial_Q=source_Q, seed=cfg['base_seed'],
    )

    snap_rows = []
    next_cp = 0
    for ep in range(tcfg['num_episodes'] + 1):
        if next_cp < len(checkpoints) and ep == checkpoints[next_cp]:
            q = agent.Q[case_idx].tolist()
            snap_rows.append({
                'episode': ep,
                'state': list(case_state),
                'q0': q[0], 'q1': q[1], 'q2': q[2], 'q3': q[3],
                'greedy_action': int(np.argmax(q)),
                'vi_action': worst['vi_action'],
                'matches_vi': int(np.argmax(q)) == worst['vi_action'],
            })
            next_cp += 1
        if ep >= tcfg['num_episodes']:
            break
        epsilon = agent.epsilon_at(ep)
        state = agent.env.reset()
        done = False
        while not done:
            a = agent.select_action(state, epsilon)
            s_next, r, done, info = agent.env.step(a)
            agent.update(state, a, r, s_next, done)
            state = s_next

    out = {
        'old_key_pos': list(old_key),
        'new_key_pos': list(load_map(DIFFERENT_MAP)['key_pos']),
        'source_model': source_model_path,
        'vi_converged': converged,
        'vi_iterations': n_iter,
        'case': worst,
        'checkpoints': snap_rows,
    }
    path = Path('results/raw_data/transfer/negative_transfer_case.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  case state={worst['state']} gap={worst['action_gap']:.3f} "
          f"transferred_a={worst['transferred_action']} vi_a={worst['vi_action']}")
    print(f'  saved {path}')

    csv_path = Path('results/raw_data/transfer/negative_transfer_checkpoints.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(snap_rows[0].keys()))
        writer.writeheader()
        writer.writerows(snap_rows)
    print(f'  saved {csv_path}')
    return out
