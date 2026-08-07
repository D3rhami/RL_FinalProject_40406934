
import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.maze import MazeEnv, load_config
from environments.generator import load_map, validate_map, _bfs_distance, _PRE_KEY_PASS, _POST_KEY_PASS
from agents.value_iteration import ValueIteration
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from experiments.run_experiments import derive_seeds, evaluate_greedy_policy

MAP_PATH = 'environments/maps/source_maze.json'
DEFAULT_BUDGETS = [50, 60, 100, 150]


def learnability_probe(cfg_e, seed, eval_seed, episodes, eval_episodes):
    """Fast (single-seed, short-budget) check of whether model-free learners
    can actually make progress at this energy, not just whether VI can.

    VI ignores sample efficiency entirely (it gets the exact transition model
    and computes the optimum in one shot), so a budget where VI succeeds
    100% of the time can still be nearly unlearnable for epsilon-greedy
    exploration if most early random rollouts burn all their energy before
    ever reaching the key. This probe trains a *short* shaped-reward QL and
    SARSA(lambda=0.3) run (single seed, exponential decay so it doesn't
    depend on total-episode count) at each candidate energy and evaluates
    greedy performance, to see where model-free learning actually gets
    traction within a realistic episode budget.
    """
    qcfg = cfg_e['q_learning']
    scfg = cfg_e['sarsa_lambda']
    out = {}

    t0 = time.time()
    ql_env = MazeEnv(MAP_PATH, config=cfg_e, reward_mode='shaped', seed=seed)
    ql_agent = QLearningAgent(
        ql_env, alpha=qcfg['alpha'], gamma=qcfg['gamma'],
        epsilon_start=qcfg['epsilon_start'], epsilon_min=qcfg['epsilon_min'],
        decay_type='exponential', decay_param=qcfg['exponential_decay_rate'],
        num_episodes=episodes, reward_mode='shaped', seed=seed)
    ql_agent.train()
    ql_eval_env = MazeEnv(MAP_PATH, config=cfg_e, reward_mode='shaped', seed=seed)
    ql_metrics = evaluate_greedy_policy(
        ql_eval_env, lambda s, _a=ql_agent: int(np.argmax(_a.Q[_a.env.encode_state(s)])),
        eval_seed, eval_episodes)
    out['ql_probe_eval_success'] = ql_metrics['eval_success_rate']
    out['ql_probe_eval_mean_return'] = ql_metrics['eval_mean_return']
    out['ql_probe_eval_mean_steps'] = ql_metrics['eval_mean_steps']
    out['ql_probe_train_seconds'] = round(time.time() - t0, 2)

    t0 = time.time()
    sa_env = MazeEnv(MAP_PATH, config=cfg_e, reward_mode='shaped', seed=seed)
    sa_agent = SarsaLambdaAgent(
        sa_env, alpha=scfg['alpha'], gamma=scfg['gamma'], lam=0.3,
        trace_type=scfg['trace_type'],
        epsilon_start=scfg['epsilon_start'], epsilon_min=scfg['epsilon_min'],
        decay_type='exponential', decay_param=scfg['exponential_decay_rate'],
        num_episodes=episodes, reward_mode='shaped', seed=seed)
    sa_agent.train()
    sa_eval_env = MazeEnv(MAP_PATH, config=cfg_e, reward_mode='shaped', seed=seed)
    sa_metrics = evaluate_greedy_policy(
        sa_eval_env, lambda s, _a=sa_agent: int(np.argmax(_a.Q[_a.env.encode_state(s)])),
        eval_seed, eval_episodes)
    out['sarsa_probe_eval_success'] = sa_metrics['eval_success_rate']
    out['sarsa_probe_eval_mean_return'] = sa_metrics['eval_mean_return']
    out['sarsa_probe_eval_mean_steps'] = sa_metrics['eval_mean_steps']
    out['sarsa_probe_train_seconds'] = round(time.time() - t0, 2)
    return out


def _cfg_with_energy(cfg, max_energy):
    out = copy.deepcopy(cfg)
    out['env']['max_energy'] = int(max_energy)
    return out


def random_policy_stats(env, n_episodes, seed_base):
    counts = {}
    returns, steps_list = [], []
    for i in range(n_episodes):
        env.reset(seed=seed_base + i)
        done, total_r, n_steps, info = False, 0.0, 0, {}
        while not done:
            _, r, done, info = env.step(int(env.rng.integers(0, 4)))
            total_r += r
            n_steps += 1
        reason = info.get('termination_reason', 'unknown')
        counts[reason] = counts.get(reason, 0) + 1
        returns.append(total_r)
        steps_list.append(n_steps)
    return {
        'random_energy_depleted_rate': counts.get('energy_depleted', 0) / n_episodes,
        'random_goal_rate': counts.get('goal_reached', 0) / n_episodes,
        'random_step_cap_rate': counts.get('step_cap', 0) / n_episodes,
        'random_mean_return': float(np.mean(returns)),
        'random_mean_steps': float(np.mean(steps_list)),
        'termination_counts': counts,
    }


def run_sweep(cfg, budgets, n_episodes=200, skip_vi=False,
              with_learners=False, learn_episodes=6000, learn_eval_episodes=60):
    data = load_map()
    d1 = _bfs_distance(data['grid'], data['start_pos'], data['key_pos'], _PRE_KEY_PASS)
    d2 = _bfs_distance(data['grid'], data['key_pos'], data['goal_pos'], _POST_KEY_PASS)
    shortest = d1 + d2
    print(f"Shortest feasible path length (start->key->goal): {shortest}")

    grid = cfg['experiment_grid']
    _, eval_seed = derive_seeds(cfg['base_seed'], n=grid['n_seeds'])
    ref_gamma = grid['value_iteration']['reference_gamma']
    reward_mode = grid['value_iteration']['reward_mode']
    rows = []

    for energy in budgets:
        print(f"\n=== max_energy={energy} ===")
        valid = validate_map(data, energy)
        print(f"  BFS-valid under budget: {valid} (need <= {energy}, path={shortest})")
        cfg_e = _cfg_with_energy(cfg, energy)
        env = MazeEnv(MAP_PATH, config=cfg_e, reward_mode=reward_mode, seed=0)
        print(f"  n_states={env.n_states}  MAX_ENERGY={env.MAX_ENERGY}")

        rand = random_policy_stats(env, n_episodes, seed_base=1000)
        print(f"  random({n_episodes}): "
              f"energy_depleted={rand['random_energy_depleted_rate']:.2f} "
              f"goal={rand['random_goal_rate']:.2f} "
              f"step_cap={rand['random_step_cap_rate']:.2f}")

        row = {
            'max_energy': energy,
            'shortest_path': shortest,
            'bfs_valid': valid,
            'n_states': env.n_states,
            'random_episodes': n_episodes,
            'random_energy_depleted_rate': rand['random_energy_depleted_rate'],
            'random_goal_rate': rand['random_goal_rate'],
            'random_step_cap_rate': rand['random_step_cap_rate'],
            'random_mean_return': rand['random_mean_return'],
            'random_mean_steps': rand['random_mean_steps'],
            'vi_gamma': ref_gamma,
            'vi_eval_success_rate': None,
            'vi_eval_mean_return': None,
            'vi_eval_mean_steps': None,
            'vi_iterations': None,
            'vi_runtime_seconds': None,
            'vi_converged': None,
            'ql_probe_eval_success': None,
            'ql_probe_eval_mean_return': None,
            'ql_probe_eval_mean_steps': None,
            'ql_probe_train_seconds': None,
            'sarsa_probe_eval_success': None,
            'sarsa_probe_eval_mean_return': None,
            'sarsa_probe_eval_mean_steps': None,
            'sarsa_probe_train_seconds': None,
        }

        if not skip_vi and valid:
            t0 = time.time()
            vi_env = MazeEnv(MAP_PATH, config=cfg_e, reward_mode=reward_mode)
            vi = ValueIteration(vi_env, gamma=ref_gamma,
                                theta=cfg['value_iteration']['theta'],
                                reward_mode=reward_mode)
            V, n_iter, converged, _ = vi.run(max_iterations=2000)
            policy = vi.extract_policy(V)
            eval_env = MazeEnv(MAP_PATH, config=cfg_e, reward_mode=reward_mode)
            metrics = evaluate_greedy_policy(
                eval_env, lambda s, _p=policy: _p.get(s, 0),
                eval_seed, grid['eval_episodes'],
            )
            row.update({
                'vi_eval_success_rate': metrics['eval_success_rate'],
                'vi_eval_mean_return': metrics['eval_mean_return'],
                'vi_eval_mean_steps': metrics['eval_mean_steps'],
                'vi_iterations': n_iter,
                'vi_runtime_seconds': round(time.time() - t0, 2),
                'vi_converged': converged,
            })
            print(f"  VI gamma={ref_gamma}: converged={converged} iters={n_iter} "
                  f"eval_success={metrics['eval_success_rate']:.2f} "
                  f"runtime={row['vi_runtime_seconds']:.1f}s")
        elif not valid:
            print("  skip VI (map not BFS-reachable within budget)")

        if with_learners and valid:
            probe_seed = derive_seeds(cfg['base_seed'], n=grid['n_seeds'])[0][0]
            t0 = time.time()
            probe = learnability_probe(cfg_e, probe_seed, eval_seed,
                                        learn_episodes, learn_eval_episodes)
            row.update(probe)
            print(f"  QL probe ({learn_episodes} eps, shaped, single seed): "
                  f"eval_success={probe['ql_probe_eval_success']:.2f} "
                  f"({probe['ql_probe_train_seconds']:.1f}s)")
            print(f"  SARSA(0.3) probe ({learn_episodes} eps, shaped, single seed): "
                  f"eval_success={probe['sarsa_probe_eval_success']:.2f} "
                  f"({probe['sarsa_probe_train_seconds']:.1f}s)")

        rows.append(row)

    out_csv = Path('results/raw_data/energy/energy_sweep.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {out_csv}")

    _plot_sweep(rows)
    return rows


def _plot_sweep(rows):
    energies = [r['max_energy'] for r in rows]
    has_probe = any(r.get('ql_probe_eval_success') is not None for r in rows)
    fig, axes = plt.subplots(1, 3 if has_probe else 2, figsize=(15 if has_probe else 10, 4))

    axes[0].plot(energies, [r['random_energy_depleted_rate'] for r in rows],
                 'o-', label='energy_depleted')
    axes[0].plot(energies, [r['random_goal_rate'] for r in rows],
                 's-', label='goal_reached')
    axes[0].plot(energies, [r['random_step_cap_rate'] for r in rows],
                 '^-', label='step_cap')
    axes[0].set_xlabel('max_energy')
    axes[0].set_ylabel('rate (random policy)')
    axes[0].set_title('Random-policy terminations vs energy budget')
    axes[0].legend()
    axes[0].set_ylim(-0.05, 1.05)

    vi_rates = [r['vi_eval_success_rate'] for r in rows]
    if any(v is not None for v in vi_rates):
        xs = [e for e, v in zip(energies, vi_rates) if v is not None]
        ys = [v for v in vi_rates if v is not None]
        axes[1].plot(xs, ys, 'o-', color='C2')
        axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel('max_energy')
    axes[1].set_ylabel('VI greedy eval success')
    axes[1].set_title('Optimal-policy success vs energy budget')

    if has_probe:
        ql_rates = [r.get('ql_probe_eval_success') for r in rows]
        sarsa_rates = [r.get('sarsa_probe_eval_success') for r in rows]
        xs_ql = [e for e, v in zip(energies, ql_rates) if v is not None]
        ys_ql = [v for v in ql_rates if v is not None]
        xs_sa = [e for e, v in zip(energies, sarsa_rates) if v is not None]
        ys_sa = [v for v in sarsa_rates if v is not None]
        axes[2].plot(xs_ql, ys_ql, 'o-', color='C0', label='Q-Learning (shaped)')
        axes[2].plot(xs_sa, ys_sa, 's-', color='C3', label='SARSA(0.3, shaped)')
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_xlabel('max_energy')
        axes[2].set_ylabel('greedy eval success (short probe)')
        axes[2].set_title('Model-free learnability vs energy budget')
        axes[2].legend()

    fig.tight_layout()
    out = Path('results/figures/energy/energy_sweep.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description='Energy budget sensitivity probe')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Random-policy episodes per energy budget')
    parser.add_argument('--skip-vi', action='store_true',
                        help='Skip VI (random policy only; much faster)')
    parser.add_argument('--budgets', type=int, nargs='+', default=DEFAULT_BUDGETS)
    parser.add_argument('--with-learners', action='store_true',
                        help='Also run a short single-seed QL + SARSA(0.3) probe per budget')
    parser.add_argument('--learn-episodes', type=int, default=6000,
                        help='Episode budget for the learnability probe (kept short for speed)')
    parser.add_argument('--learn-eval-episodes', type=int, default=60,
                        help='Greedy eval episodes for the learnability probe')
    args = parser.parse_args()

    cfg = load_config()
    budgets = args.budgets
    if budgets == DEFAULT_BUDGETS:
        budgets = cfg.get('env', {}).get('energy_sweep', DEFAULT_BUDGETS)
    run_sweep(cfg, budgets, n_episodes=args.episodes, skip_vi=args.skip_vi,
              with_learners=args.with_learners, learn_episodes=args.learn_episodes,
              learn_eval_episodes=args.learn_eval_episodes)


if __name__ == '__main__':
    main()
