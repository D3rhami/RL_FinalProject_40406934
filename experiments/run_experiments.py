import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.maze import MazeEnv, load_config
from experiments.logger import EpisodeLogger
from agents.value_iteration import ValueIteration
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent

MAP_PATH = 'environments/maps/source_maze.json'
LEDGER_PATH = Path('results/raw_data/run_ledger.csv')

# All evaluation (greedy rollouts) uses this reward_mode regardless of the
# variant's training reward_mode, so eval_mean_return is comparable across
# sparse- and shaped-trained agents (shaping bonus must not leak into eval).
EVAL_REWARD_MODE = 'sparse'

_ledger_rows = []


def derive_seeds(base_seed, n=3):
    """Generate n reproducible training seeds + 1 eval seed from base_seed."""
    rng = np.random.default_rng(base_seed)
    seeds = [int(x) for x in rng.integers(1, 100000, size=n)]
    eval_seed = int(rng.integers(1, 100000))
    return seeds, eval_seed


def _ledger_append(algorithm, variant_id, seed, status, runtime_seconds,
                    error_message='', output_files=''):
    _ledger_rows.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': algorithm, 'variant_id': variant_id, 'seed': seed,
        'status': status, 'runtime_seconds': round(runtime_seconds, 3),
        'error_message': error_message, 'output_files': output_files,
    })


def _flush_ledger():
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _ledger_rows:
        return
    write_header = not LEDGER_PATH.exists()
    with open(LEDGER_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(_ledger_rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(_ledger_rows)
    _ledger_rows.clear()


def _save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)


def evaluate_greedy_policy(env, act_fn, eval_seed, eval_episodes):
    successes, returns, steps_list = 0, [], []
    for i in range(eval_episodes):
        state = env.reset(seed=eval_seed + i)
        done, total_r, n_steps, info = False, 0.0, 0, {}
        while not done:
            a = act_fn(state)
            state, r, done, info = env.step(a)
            total_r += r
            n_steps += 1
        successes += int(info.get('termination_reason') == 'goal_reached')
        returns.append(total_r)
        steps_list.append(n_steps)
    return {
        'eval_success_rate': successes / eval_episodes,
        'eval_mean_return': float(np.mean(returns)),
        'eval_mean_steps': float(np.mean(steps_list)),
    }


def run_value_iteration(cfg):
    print("=== Value Iteration ===")
    grid_cfg = cfg['experiment_grid']['value_iteration']
    vi_cfg = cfg['value_iteration']
    reward_mode = grid_cfg['reward_mode']
    _, eval_seed = derive_seeds(cfg['base_seed'], n=cfg['experiment_grid']['n_seeds'])
    rows = []

    for gamma in grid_cfg['gammas']:
        variant_id = f"vi_{reward_mode}_gamma{gamma}"
        t0 = time.time()
        try:
            env = MazeEnv(MAP_PATH, config=cfg, reward_mode=reward_mode)
            vi = ValueIteration(env, gamma=gamma, theta=vi_cfg['theta'],
                                 reward_mode=reward_mode)
            V, n_iter, converged, history = vi.run(max_iterations=2000)
            policy = vi.extract_policy(V)
            states = env.all_states()

            eval_env = MazeEnv(MAP_PATH, config=cfg, reward_mode=EVAL_REWARD_MODE)
            eval_metrics = evaluate_greedy_policy(
                eval_env, lambda s, _p=policy: _p.get(s, 0),
                eval_seed, cfg['experiment_grid']['eval_episodes'],
            )

            model_path = f'results/models/vi/{variant_id}.json'
            _save_json(model_path, {
                'algorithm': 'value_iteration', 'gamma': gamma,
                'theta': vi_cfg['theta'], 'reward_mode': reward_mode,
                'iterations': n_iter, 'runtime_seconds': vi.runtime,
                'converged': converged, 'deltas': history,
                'states': [list(s) for s in states],
                'V': V.tolist(),
                'policy': [policy.get(s, -1) for s in states],
            })

            start_state = env.reset()
            rows.append({
                'gamma': gamma, 'iterations': n_iter,
                'runtime_seconds': vi.runtime, 'converged': converged,
                'v_start': float(V[env.encode_state(start_state)]),
                **eval_metrics,
            })
            _ledger_append('value_iteration', variant_id, '', 'success',
                           time.time() - t0, output_files=model_path)
            print(f"  gamma={gamma}: converged={converged} iters={n_iter} "
                  f"runtime={vi.runtime:.1f}s eval_success={eval_metrics['eval_success_rate']:.2f}")
        except Exception as e:
            _ledger_append('value_iteration', variant_id, '', 'failed',
                           time.time() - t0, error_message=str(e))
            print(f"  FAILED gamma={gamma}: {e}")
            traceback.print_exc()

    if rows:
        csv_path = Path('results/raw_data/vi/vi_gamma_sweep.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  saved {csv_path}")


def _rolling_success_episode(successes, threshold=0.9, window=100):
    if len(successes) < window:
        return None
    arr = np.array(successes, dtype=np.float64)
    csum = np.cumsum(arr)
    csum = np.insert(csum, 0, 0.0)
    rolling = (csum[window:] - csum[:-window]) / window
    hit = np.where(rolling >= threshold)[0]
    return int(hit[0] + window) if len(hit) else None


def run_q_learning(cfg):
    print("=== Q-Learning ===")
    grid = cfg['experiment_grid']
    qg = grid['q_learning']
    qcfg = cfg['q_learning']
    seeds, eval_seed = derive_seeds(cfg['base_seed'], n=grid['n_seeds'])
    trace_cfg = qg['trace_run']
    trace_seed = seeds[0]

    training_rows, summary_rows = [], []

    for reward_mode in qg['reward_modes']:
        for schedule in qg['schedules']:
            for seed in seeds:
                variant_id = f"q_learning_{reward_mode}_{schedule}_seed{seed}"
                t0 = time.time()
                try:
                    env = MazeEnv(MAP_PATH, config=cfg, reward_mode=reward_mode, seed=seed)
                    decay_param = (qcfg['linear_decay_episodes'] if schedule == 'linear'
                                   else qcfg['exponential_decay_rate'])
                    agent = QLearningAgent(
                        env, alpha=qcfg['alpha'], gamma=qcfg['gamma'],
                        epsilon_start=qcfg['epsilon_start'],
                        epsilon_min=qcfg['epsilon_min'],
                        decay_type=schedule, decay_param=decay_param,
                        num_episodes=qcfg['num_episodes'],
                        reward_mode=reward_mode, seed=seed,
                    )

                    is_trace = (reward_mode == trace_cfg['reward_mode']
                                and schedule == trace_cfg['schedule']
                                and seed == trace_seed)
                    trace_rows = []

                    def step_cb(row, _rows=trace_rows):
                        r, c, hk, en = row['state']
                        nr, nc, nhk, nen = row['next_state']
                        _rows.append({
                            'episode': row['episode'], 'step': row['step'],
                            'r': r, 'c': c, 'has_key': hk, 'energy': en,
                            'action': row['action'], 'reward': row['reward'],
                            'next_r': nr, 'next_c': nc,
                            'next_has_key': nhk, 'next_energy': nen,
                            'event': row['event'], 'q_before': row['q_before'],
                            'max_next_q': row['max_next_q'],
                            'td_target': row['td_target'],
                            'td_error': row['td_error'],
                            'q_after': row['q_after'],
                            'alpha': row['alpha'], 'gamma': row['gamma'],
                            'epsilon': row['epsilon'],
                        })

                    logger = EpisodeLogger(
                        f'results/raw_data/q_learning/_tmp_{variant_id}.csv')
                    agent.train(
                        logger=logger,
                        step_callback=step_cb if is_trace else None,
                        trace_episodes={trace_cfg['episode_index']} if is_trace else None,
                    )

                    per_episode_success, per_episode_return = [], []
                    for row in logger.summary_rows:
                        per_episode_success.append(int(row['success']))
                        per_episode_return.append(row['total_reward'])
                        training_rows.append({
                            'reward_mode': reward_mode, 'schedule': schedule,
                            'seed': seed, 'episode': row['episode_idx'],
                            'epsilon': row['epsilon_final'],
                            'steps': row['steps'], 'return': row['total_reward'],
                            'success': row['success'],
                            'wall_hits': row['wall_hits'],
                            'penalty_entries': row['penalty_entries'],
                            'door_attempts': row['door_attempts'],
                            'key_picked': row['key_pickup_step'] is not None,
                            'termination_reason': row['termination_reason'],
                        })

                    states = env.all_states()
                    state_indices = [env.encode_state(s) for s in states]
                    eval_env = MazeEnv(MAP_PATH, config=cfg, reward_mode=EVAL_REWARD_MODE)

                    def act_fn(state, _a=agent):
                        return int(np.argmax(_a.Q[_a.env.encode_state(state)]))

                    eval_metrics = evaluate_greedy_policy(
                        eval_env, act_fn, eval_seed,
                        grid['eval_episodes'],
                    )

                    model_path = f'results/models/q_learning/{variant_id}.json'
                    _save_json(model_path, {
                        'algorithm': 'q_learning', 'alpha': qcfg['alpha'],
                        'gamma': qcfg['gamma'], 'reward_mode': reward_mode,
                        'schedule': schedule, 'seed': seed,
                        'episodes': qcfg['num_episodes'],
                        'epsilon_start': qcfg['epsilon_start'],
                        'epsilon_min': qcfg['epsilon_min'],
                        'decay_param': decay_param,
                        'states': [list(s) for s in states],
                        'Q': agent.Q[state_indices].tolist(),
                        'visits': agent.visits[state_indices].tolist(),
                    })

                    if is_trace and trace_rows:
                        p = Path('results/raw_data/q_learning/q_update_trace.csv')
                        p.parent.mkdir(parents=True, exist_ok=True)
                        with open(p, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
                            writer.writeheader()
                            writer.writerows(trace_rows)
                        print(f"  saved {p} ({len(trace_rows)} rows)")

                    first_success = next(
                        (i for i, s in enumerate(per_episode_success) if s == 1), None)
                    summary_rows.append({
                        'reward_mode': reward_mode, 'schedule': schedule, 'seed': seed,
                        'first_success_episode': first_success,
                        'episodes_to_90pct_success': _rolling_success_episode(per_episode_success),
                        'final_train_success_rate': float(np.mean(per_episode_success[-100:])),
                        **eval_metrics,
                        'visited_states': int(np.sum(agent.visits > 0)),
                    })

                    Path(f'results/raw_data/q_learning/_tmp_{variant_id}.csv').unlink(missing_ok=True)
                    _ledger_append('q_learning', variant_id, seed, 'success',
                                   time.time() - t0, output_files=model_path)
                    print(f"  {variant_id}: "
                          f"eval={eval_metrics['eval_success_rate']:.2f} "
                          f"final_train={summary_rows[-1]['final_train_success_rate']:.2f}")
                except Exception as e:
                    _ledger_append('q_learning', variant_id, seed, 'failed',
                                   time.time() - t0, error_message=str(e))
                    print(f"  FAILED {variant_id}: {e}")
                    traceback.print_exc()

    if training_rows:
        p = Path('results/raw_data/q_learning/q_learning_training.csv')
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(training_rows[0].keys()))
            writer.writeheader()
            writer.writerows(training_rows)
        print(f"  saved {p}")
    if summary_rows:
        p = Path('results/raw_data/q_learning/q_learning_summary.csv')
        with open(p, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  saved {p}")


def run_sarsa_lambda(cfg):
    print("=== SARSA(lambda) ===")
    grid = cfg['experiment_grid']
    sg = grid['sarsa_lambda']
    scfg = cfg['sarsa_lambda']
    seeds, eval_seed = derive_seeds(cfg['base_seed'], n=grid['n_seeds'])
    reward_mode = sg['reward_mode']
    trace_cfg = sg['trace_run']
    trace_seed = seeds[0]

    training_rows, summary_rows = [], []

    for lam in sg['lambdas']:
        for seed in seeds:
            variant_id = f"sarsa_{reward_mode}_lambda{lam}_seed{seed}"
            t0 = time.time()
            try:
                env = MazeEnv(MAP_PATH, config=cfg, reward_mode=reward_mode, seed=seed)
                agent = SarsaLambdaAgent(
                    env, alpha=scfg['alpha'], gamma=scfg['gamma'], lam=lam,
                    trace_type=scfg['trace_type'],
                    epsilon_start=scfg['epsilon_start'],
                    epsilon_min=scfg['epsilon_min'],
                    decay_type='exponential',
                    decay_param=scfg['exponential_decay_rate'],
                    num_episodes=scfg['num_episodes'],
                    reward_mode=reward_mode, seed=seed,
                )

                is_trace = (lam == trace_cfg['lambda'] and seed == trace_seed)
                trace_rows, trace_dump_rows = [], []

                def step_cb(row, _rows=trace_rows):
                    r, c, hk, en = row['state']
                    _rows.append({
                        'episode': row['episode'], 'step': row['step'],
                        'r': r, 'c': c, 'has_key': hk, 'energy': en,
                        'action': row['action'], 'next_action': row['next_action'],
                        'reward': row['reward'], 'event': row['event'],
                        'delta': row['delta'],
                        'active_traces': row['active_traces'],
                        'alpha': row['alpha'], 'gamma': row['gamma'],
                        'lam': row['lam'], 'epsilon': row['epsilon'],
                    })

                def dump_cb(ep, step, s_idx, a_idx, e_val,
                            _rows=trace_dump_rows, _env=env):
                    r, c, hk, en = _env.decode_state(s_idx)
                    _rows.append({
                        'episode': ep, 'step': step,
                        'r': r, 'c': c, 'has_key': hk, 'energy': en,
                        'action': a_idx, 'E': e_val,
                    })

                logger = EpisodeLogger(
                    f'results/raw_data/sarsa/_tmp_{variant_id}.csv')
                agent.train(
                    logger=logger,
                    step_callback=step_cb if is_trace else None,
                    trace_episodes={trace_cfg['episode_index']} if is_trace else None,
                    trace_dump_callback=dump_cb if is_trace else None,
                )

                per_episode_success, per_episode_return = [], []
                for row in logger.summary_rows:
                    per_episode_success.append(int(row['success']))
                    per_episode_return.append(row['total_reward'])
                    training_rows.append({
                        'lambda': lam, 'seed': seed,
                        'episode': row['episode_idx'],
                        'epsilon': row['epsilon_final'],
                        'steps': row['steps'], 'return': row['total_reward'],
                        'success': row['success'],
                        'wall_hits': row['wall_hits'],
                        'penalty_entries': row['penalty_entries'],
                        'door_attempts': row['door_attempts'],
                        'key_picked': row['key_pickup_step'] is not None,
                        'termination_reason': row['termination_reason'],
                    })

                states = env.all_states()
                state_indices = [env.encode_state(s) for s in states]
                eval_env = MazeEnv(MAP_PATH, config=cfg, reward_mode=EVAL_REWARD_MODE)

                def act_fn(state, _a=agent):
                    return int(np.argmax(_a.Q[_a.env.encode_state(state)]))

                eval_metrics = evaluate_greedy_policy(
                    eval_env, act_fn, eval_seed, grid['eval_episodes'],
                )

                model_path = f'results/models/sarsa/{variant_id}.json'
                _save_json(model_path, {
                    'algorithm': 'sarsa_lambda', 'alpha': scfg['alpha'],
                    'gamma': scfg['gamma'], 'lambda': lam,
                    'trace_type': scfg['trace_type'],
                    'reward_mode': reward_mode, 'seed': seed,
                    'episodes': scfg['num_episodes'],
                    'states': [list(s) for s in states],
                    'Q': agent.Q[state_indices].tolist(),
                    'visits': agent.visits[state_indices].tolist(),
                })

                if is_trace and trace_rows:
                    p1 = Path('results/raw_data/sarsa/sarsa_step_trace.csv')
                    p1.parent.mkdir(parents=True, exist_ok=True)
                    with open(p1, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(trace_rows)
                    print(f"  saved {p1} ({len(trace_rows)} rows)")
                if is_trace and trace_dump_rows:
                    p2 = Path('results/raw_data/sarsa/sarsa_trace_dump.csv')
                    with open(p2, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=list(trace_dump_rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(trace_dump_rows)
                    print(f"  saved {p2} ({len(trace_dump_rows)} rows)")

                first_success = next(
                    (i for i, s in enumerate(per_episode_success) if s == 1), None)
                summary_rows.append({
                    'lambda': lam, 'seed': seed,
                    'first_success_episode': first_success,
                    'episodes_to_90pct_success': _rolling_success_episode(per_episode_success),
                    'final_train_success_rate': float(np.mean(per_episode_success[-100:])),
                    **eval_metrics,
                    'visited_states': int(np.sum(agent.visits > 0)),
                    'late_return_std': float(np.std(per_episode_return[-1000:])),
                })

                Path(f'results/raw_data/sarsa/_tmp_{variant_id}.csv').unlink(missing_ok=True)
                _ledger_append('sarsa_lambda', variant_id, seed, 'success',
                               time.time() - t0, output_files=model_path)
                print(f"  {variant_id}: "
                      f"eval={eval_metrics['eval_success_rate']:.2f} "
                      f"final_train={summary_rows[-1]['final_train_success_rate']:.2f}")
            except Exception as e:
                _ledger_append('sarsa_lambda', variant_id, seed, 'failed',
                               time.time() - t0, error_message=str(e))
                print(f"  FAILED {variant_id}: {e}")
                traceback.print_exc()

    if training_rows:
        p = Path('results/raw_data/sarsa/sarsa_training.csv')
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(training_rows[0].keys()))
            writer.writeheader()
            writer.writerows(training_rows)
        print(f"  saved {p}")
    if summary_rows:
        p = Path('results/raw_data/sarsa/sarsa_summary.csv')
        with open(p, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  saved {p}")


ALGO_RUNNERS = {
    'vi': run_value_iteration, 'value_iteration': run_value_iteration,
    'q_learning': run_q_learning,
    'sarsa': run_sarsa_lambda, 'sarsa_lambda': run_sarsa_lambda,
}


def main():
    args = sys.argv[1:]

    config_path = 'experiments/configs/default_config.json'
    if '--config' in args:
        i = args.index('--config')
        config_path = args[i + 1]
        args = args[:i] + args[i + 2:]

    dry_run = '--dry-run' in args
    if dry_run:
        args = [a for a in args if a != '--dry-run']

    fresh = '--fresh' in args
    if fresh:
        args = [a for a in args if a != '--fresh']

    requested = args
    cfg = load_config(config_path)

    if dry_run:
        cfg['q_learning']['num_episodes'] = 50
        cfg['sarsa_lambda']['num_episodes'] = 50
        cfg['experiment_grid']['eval_episodes'] = 10
        cfg['experiment_grid']['q_learning']['trace_run']['episode_index'] = 10
        cfg['experiment_grid']['sarsa_lambda']['trace_run']['episode_index'] = 10
        cfg['experiment_grid']['value_iteration']['gammas'] = [0.95]
        print("[dry-run] Using 50 episodes, 10 eval episodes, gamma=0.95 only")

    if fresh and LEDGER_PATH.exists():
        LEDGER_PATH.unlink()
        print(f"[fresh] Cleared {LEDGER_PATH}")

    seeds, eval_seed = derive_seeds(cfg['base_seed'], n=cfg['experiment_grid']['n_seeds'])
    print(f"Seeds (derived from base_seed={cfg['base_seed']}): {seeds}")
    print(f"Eval seed: {eval_seed}")

    to_run = []
    if not requested:
        to_run = [run_value_iteration, run_q_learning, run_sarsa_lambda]
    else:
        seen = set()
        for name in requested:
            fn = ALGO_RUNNERS.get(name)
            if fn is None:
                print(f"Unknown algorithm '{name}'. Choices: vi, q_learning, sarsa")
                continue
            if fn not in seen:
                to_run.append(fn)
                seen.add(fn)

    t_start = time.time()
    for fn in to_run:
        fn(cfg)
        _flush_ledger()

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    print(f"Run ledger: {LEDGER_PATH}")


if __name__ == '__main__':
    main()
