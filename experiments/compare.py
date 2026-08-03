import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.maze import MazeEnv, load_config, CellType
from agents.value_iteration import ValueIteration
from experiments.run_experiments import derive_seeds

MAP_PATH = 'environments/maps/source_maze.json'


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _check_prereqs(cfg):
    grid = cfg['experiment_grid']
    seeds, _ = derive_seeds(cfg['base_seed'], n=grid['n_seeds'])
    ref_gamma = grid['value_iteration']['reference_gamma']
    vi_path = Path(f'results/models/vi/vi_sparse_gamma{ref_gamma}.json')
    missing = []
    if not vi_path.exists():
        missing.append(str(vi_path))

    for reward_mode in grid['q_learning']['reward_modes']:
        for schedule in grid['q_learning']['schedules']:
            for seed in seeds:
                p = Path(f'results/models/q_learning/q_learning_{reward_mode}_{schedule}_seed{seed}.json')
                if not p.exists():
                    missing.append(str(p))

    for lam in grid['sarsa_lambda']['lambdas']:
        for seed in seeds:
            rm = grid['sarsa_lambda']['reward_mode']
            p = Path(f'results/models/sarsa/sarsa_{rm}_lambda{lam}_seed{seed}.json')
            if not p.exists():
                missing.append(str(p))

    if missing:
        print("Missing required model files. Run `python experiments/run_experiments.py` first.")
        for m in missing[:20]:
            print(f"  - {m}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        sys.exit(1)


def _pick_best_ql_variant(cfg, seeds):
    summary = pd.read_csv('results/raw_data/q_learning/q_learning_summary.csv')
    agg = summary.groupby(['reward_mode', 'schedule'])['eval_success_rate'].mean().reset_index()
    best = agg.loc[agg['eval_success_rate'].idxmax()]
    subset = summary[(summary['reward_mode'] == best['reward_mode']) &
                      (summary['schedule'] == best['schedule'])]
    best_seed = int(subset.loc[subset['eval_success_rate'].idxmax(), 'seed'])
    return str(best['reward_mode']), str(best['schedule']), best_seed


def _pick_best_sarsa_variant(cfg, seeds):
    summary = pd.read_csv('results/raw_data/sarsa/sarsa_summary.csv')
    agg = summary.groupby('lambda')['eval_success_rate'].mean().reset_index()
    best_lam = float(agg.loc[agg['eval_success_rate'].idxmax(), 'lambda'])
    subset = summary[summary['lambda'] == best_lam]
    best_seed = int(subset.loc[subset['eval_success_rate'].idxmax(), 'seed'])
    return best_lam, best_seed


def build_comparison(cfg):
    grid = cfg['experiment_grid']
    seeds, eval_seed = derive_seeds(cfg['base_seed'], n=grid['n_seeds'])
    ref_gamma = grid['value_iteration']['reference_gamma']
    vi_reward_mode = grid['value_iteration']['reward_mode']

    vi_model = _load_json(f'results/models/vi/vi_sparse_gamma{ref_gamma}.json')
    vi_states = [tuple(s) for s in vi_model['states']]
    vi_policy = {s: a for s, a in zip(vi_states, vi_model['policy']) if a != -1}
    vi_V = np.array(vi_model['V'])
    state_to_vi_idx = {s: i for i, s in enumerate(vi_states)}

    env = MazeEnv(MAP_PATH, config=cfg, reward_mode=vi_reward_mode)
    vi = ValueIteration(env, gamma=ref_gamma, theta=cfg['value_iteration']['theta'],
                         reward_mode=vi_reward_mode)

    ql_rm, ql_sched, ql_seed = _pick_best_ql_variant(cfg, seeds)
    ql_model = _load_json(
        f'results/models/q_learning/q_learning_{ql_rm}_{ql_sched}_seed{ql_seed}.json')
    ql_states = [tuple(s) for s in ql_model['states']]
    ql_Q = np.array(ql_model['Q'])
    ql_visits = np.array(ql_model['visits'])

    sarsa_lam, sarsa_seed = _pick_best_sarsa_variant(cfg, seeds)
    sarsa_rm = grid['sarsa_lambda']['reward_mode']
    sarsa_model = _load_json(
        f'results/models/sarsa/sarsa_{sarsa_rm}_lambda{sarsa_lam}_seed{sarsa_seed}.json')
    sarsa_states = [tuple(s) for s in sarsa_model['states']]
    sarsa_Q = np.array(sarsa_model['Q'])
    sarsa_visits = np.array(sarsa_model['visits'])

    print(f"Reference VI: gamma={ref_gamma}")
    print(f"Reference QL: reward_mode={ql_rm}, schedule={ql_sched}, seed={ql_seed}")
    print(f"Reference SARSA: lambda={sarsa_lam}, seed={sarsa_seed}")

    def agreement_report(name, states, Q, visits):
        disagreements = []
        all_agree = all_total = visited_agree = visited_total = 0
        for i, s in enumerate(states):
            if s not in vi_policy:
                continue
            agent_action = int(np.argmax(Q[i]))
            vi_action = vi_policy[s]
            agree = agent_action == vi_action
            all_total += 1
            all_agree += int(agree)
            if visits[i] > 0:
                visited_total += 1
                visited_agree += int(agree)
                if not agree:
                    gap = (vi._q_value(vi_V, s, vi_action)
                           - vi._q_value(vi_V, s, agent_action))
                    disagreements.append({
                        'algorithm': name, 'state': s,
                        'vi_action': vi_action, 'agent_action': agent_action,
                        'action_gap': gap, 'visits': int(visits[i]),
                    })
        all_rate = all_agree / all_total if all_total else float('nan')
        visited_rate = visited_agree / visited_total if visited_total else float('nan')
        return all_rate, visited_rate, visited_total, disagreements

    ql_all, ql_vis, ql_n, ql_dis = agreement_report(
        'q_learning', ql_states, ql_Q, ql_visits)
    sarsa_all, sarsa_vis, sarsa_n, sarsa_dis = agreement_report(
        'sarsa_lambda', sarsa_states, sarsa_Q, sarsa_visits)

    print(f"QL    agreement: all={ql_all:.3f}  visited-only={ql_vis:.3f}  (n={ql_n})")
    print(f"SARSA agreement: all={sarsa_all:.3f}  visited-only={sarsa_vis:.3f}  (n={sarsa_n})")

    return {
        'vi': {'model': vi_model, 'policy': vi_policy, 'V': vi_V, 'vi_obj': vi},
        'ql': {'reward_mode': ql_rm, 'schedule': ql_sched, 'seed': ql_seed,
               'model': ql_model, 'states': ql_states, 'Q': ql_Q, 'visits': ql_visits,
               'all_agreement': ql_all, 'visited_agreement': ql_vis,
               'n_visited': ql_n, 'disagreements': ql_dis},
        'sarsa': {'lam': sarsa_lam, 'seed': sarsa_seed,
                  'model': sarsa_model, 'states': sarsa_states,
                  'Q': sarsa_Q, 'visits': sarsa_visits,
                  'all_agreement': sarsa_all, 'visited_agreement': sarsa_vis,
                  'n_visited': sarsa_n, 'disagreements': sarsa_dis},
        'env': env,
    }


def select_sample_states(disagreements, env):
    if not disagreements:
        return {}
    by_gap = sorted(disagreements, key=lambda d: d['action_gap'], reverse=True)
    nonzero = [d for d in by_gap if d['action_gap'] > 1e-6]
    return {
        'largest_action_gap': by_gap[0],
        'smallest_nonzero_gap': nonzero[-1] if nonzero else None,
        'nearest_penalty_cell': min(
            disagreements,
            key=lambda d: min(abs(d['state'][0]-pr)+abs(d['state'][1]-pc)
                              for pr, pc in env.penalty_cells),
        ),
    }


def plot_policy_diff(vi_policy, Q_by_state, states, maze_size, has_key, energy,
                      out_path, title):
    diff = np.full((maze_size, maze_size), np.nan)
    for s in states:
        r, c, hk, en = s
        if hk != has_key or en != energy or s not in vi_policy:
            continue
        agent_a = Q_by_state.get(s)
        if agent_a is not None:
            diff[r, c] = 1.0 if agent_a == vi_policy[s] else 0.0

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(diff, cmap=plt.get_cmap('RdYlGn'), vmin=0, vmax=1)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label='1=agree, 0=disagree')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def main():
    cfg = load_config()
    _check_prereqs(cfg)

    result = build_comparison(cfg)
    env = result['env']
    maze_size, max_energy = env.maze_size, env.MAX_ENERGY

    out_dir = Path('results/raw_data/comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    vi_csv = pd.read_csv('results/raw_data/vi/vi_gamma_sweep.csv')
    ref_gamma = cfg['experiment_grid']['value_iteration']['reference_gamma']
    vi_row = vi_csv[vi_csv['gamma'] == ref_gamma].iloc[0]

    ql_summary = pd.read_csv('results/raw_data/q_learning/q_learning_summary.csv')
    ql_subset = ql_summary[(ql_summary['reward_mode'] == result['ql']['reward_mode']) &
                            (ql_summary['schedule'] == result['ql']['schedule'])]
    sarsa_summary = pd.read_csv('results/raw_data/sarsa/sarsa_summary.csv')
    sarsa_subset = sarsa_summary[sarsa_summary['lambda'] == result['sarsa']['lam']]

    ql_gaps = [d['action_gap'] for d in result['ql']['disagreements']]
    sarsa_gaps = [d['action_gap'] for d in result['sarsa']['disagreements']]

    summary_rows = [
        {
            'algorithm': 'value_iteration', 'type': 'model-based',
            'progress_unit': 'bellman_iteration', 'main_output': 'V,policy',
            'runtime_seconds': vi_row['runtime_seconds'],
            'convergence': vi_row['converged'],
            'samples_needed': vi_row['iterations'],
            'stability': None,
            'eval_success_rate': vi_row['eval_success_rate'],
            'eval_mean_steps': vi_row['eval_mean_steps'],
            'vi_agreement_visited': 1.0,
            'median_action_gap': 0.0,
        },
        {
            'algorithm': 'q_learning', 'type': 'model-free/off-policy',
            'progress_unit': 'episode', 'main_output': 'Q,policy',
            'runtime_seconds': None, 'convergence': None,
            'samples_needed': ql_subset['episodes_to_90pct_success'].mean(),
            'stability': ql_subset['eval_success_rate'].std(),
            'eval_success_rate': ql_subset['eval_success_rate'].mean(),
            'eval_mean_steps': ql_subset['eval_mean_steps'].mean(),
            'vi_agreement_visited': result['ql']['visited_agreement'],
            'median_action_gap': float(np.median(ql_gaps)) if ql_gaps else 0.0,
        },
        {
            'algorithm': 'sarsa_lambda', 'type': 'model-free/on-policy',
            'progress_unit': 'episode', 'main_output': 'Q,E,policy',
            'runtime_seconds': None, 'convergence': None,
            'samples_needed': sarsa_subset['episodes_to_90pct_success'].mean(),
            'stability': sarsa_subset['late_return_std'].mean(),
            'eval_success_rate': sarsa_subset['eval_success_rate'].mean(),
            'eval_mean_steps': sarsa_subset['eval_mean_steps'].mean(),
            'vi_agreement_visited': result['sarsa']['visited_agreement'],
            'median_action_gap': float(np.median(sarsa_gaps)) if sarsa_gaps else 0.0,
        },
    ]

    with open(out_dir / 'comparison_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"saved {out_dir / 'comparison_summary.csv'}")

    sample_rows = []
    for algo, disagreements in (('q_learning', result['ql']['disagreements']),
                                 ('sarsa_lambda', result['sarsa']['disagreements'])):
        for cat, d in select_sample_states(disagreements, env).items():
            if d:
                sample_rows.append({
                    'algorithm': algo, 'category': cat, 'state': d['state'],
                    'vi_action': d['vi_action'], 'agent_action': d['agent_action'],
                    'action_gap': d['action_gap'], 'visits': d['visits'],
                })
    if sample_rows:
        with open(out_dir / 'comparison_sample_states.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_rows)
        print(f"saved {out_dir / 'comparison_sample_states.csv'}")

    for algo_key, label in (('ql', 'QL'), ('sarsa', 'SARSA')):
        algo_states = result[algo_key]['states']
        algo_Q = result[algo_key]['Q']
        Q_by_state = {s: int(np.argmax(q)) for s, q in zip(algo_states, algo_Q)}
        for has_key, tag in ((0, 'prekey'), (1, 'postkey')):
            plot_policy_diff(
                result['vi']['policy'], Q_by_state, algo_states,
                maze_size, has_key, max_energy,
                f'results/figures/comparison/{algo_key}_policy_diff_{tag}.png',
                f'{label} vs VI agreement (has_key={has_key}, energy={max_energy})',
            )


if __name__ == '__main__':
    main()
