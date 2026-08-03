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

MAP_PATH = 'environments/maps/source_maze.json'


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _check_prereqs(cfg):
    grid = cfg['experiment_grid']
    ref_gamma = grid['value_iteration']['reference_gamma']
    vi_path = Path(f'results/models/vi/vi_sparse_gamma{ref_gamma}.json')
    missing = []
    if not vi_path.exists():
        missing.append(str(vi_path))

    for reward_mode in grid['q_learning']['reward_modes']:
        for schedule in grid['q_learning']['schedules']:
            for seed in grid['seeds']:
                p = Path(f'results/models/q_learning/q_learning_{reward_mode}_{schedule}_seed{seed}.json')
                if not p.exists():
                    missing.append(str(p))

    for lam in grid['sarsa_lambda']['lambdas']:
        for seed in grid['seeds']:
            p = Path(f"results/models/sarsa/sarsa_{grid['sarsa_lambda']['reward_mode']}_lambda{lam}_seed{seed}.json")
            if not p.exists():
                missing.append(str(p))

    if missing:
        print("Missing required model files. Run `python experiments/run_experiments.py` first.")
        print("Missing:")
        for m in missing[:20]:
            print(f"  - {m}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        sys.exit(1)


def _pick_best_ql_variant(cfg):
    summary = pd.read_csv('results/raw_data/q_learning/q_learning_summary.csv')
    agg = summary.groupby(['reward_mode', 'schedule'])['eval_success_rate'].mean().reset_index()
    best = agg.loc[agg['eval_success_rate'].idxmax()]
    subset = summary[(summary['reward_mode'] == best['reward_mode']) &
                      (summary['schedule'] == best['schedule'])]
    best_seed = subset.loc[subset['eval_success_rate'].idxmax(), 'seed']
    return best['reward_mode'], best['schedule'], int(best_seed)


def _pick_best_sarsa_variant(cfg):
    summary = pd.read_csv('results/raw_data/sarsa/sarsa_summary.csv')
    agg = summary.groupby('lambda')['eval_success_rate'].mean().reset_index()
    best_lam = agg.loc[agg['eval_success_rate'].idxmax(), 'lambda']
    subset = summary[summary['lambda'] == best_lam]
    best_seed = int(subset.loc[subset['eval_success_rate'].idxmax(), 'seed'])
    return float(best_lam), best_seed


def _q_star_value(vi: ValueIteration, V, state, action):
    return vi._q_value(V, state, action)


def _mean_runtime_from_ledger(algorithm, variant_ids):
    """Average runtime_seconds across the given variant_ids (successful runs
    only), read from run_ledger.csv. Returns None if the ledger or the
    variants aren't found, rather than crashing -- runtime is a nice-to-have
    for the comparison table, not something that should block the rest of it."""
    ledger_path = Path('results/raw_data/run_ledger.csv')
    if not ledger_path.exists():
        return None
    ledger = pd.read_csv(ledger_path)
    subset = ledger[(ledger['algorithm'] == algorithm) &
                     (ledger['variant_id'].isin(variant_ids)) &
                     (ledger['status'] == 'success')]
    if subset.empty:
        return None
    return float(subset['runtime_seconds'].mean())


def build_comparison(cfg):
    grid = cfg['experiment_grid']
    ref_gamma = grid['value_iteration']['reference_gamma']
    vi_reward_mode = grid['value_iteration']['reward_mode']

    vi_model = _load_json(f'results/models/vi/vi_sparse_gamma{ref_gamma}.json')
    vi_states = [tuple(s) for s in vi_model['states']]
    vi_policy = {s: a for s, a in zip(vi_states, vi_model['policy']) if a != -1}
    vi_V = np.array(vi_model['V'])

    env = MazeEnv(MAP_PATH, config=cfg, reward_mode=vi_reward_mode)
    vi = ValueIteration(env, gamma=ref_gamma, theta=cfg['value_iteration']['theta'],
                         reward_mode=vi_reward_mode)

    ql_reward_mode, ql_schedule, ql_seed = _pick_best_ql_variant(cfg)
    ql_model = _load_json(f'results/models/q_learning/q_learning_{ql_reward_mode}_{ql_schedule}_seed{ql_seed}.json')
    ql_states = [tuple(s) for s in ql_model['states']]
    ql_Q = np.array(ql_model['Q'])
    ql_visits = np.array(ql_model['visits'])

    sarsa_lam, sarsa_seed = _pick_best_sarsa_variant(cfg)
    sarsa_model = _load_json(f"results/models/sarsa/sarsa_{grid['sarsa_lambda']['reward_mode']}_lambda{sarsa_lam}_seed{sarsa_seed}.json")
    sarsa_states = [tuple(s) for s in sarsa_model['states']]
    sarsa_Q = np.array(sarsa_model['Q'])
    sarsa_visits = np.array(sarsa_model['visits'])

    print(f"Reference VI: gamma={ref_gamma}")
    print(f"Reference QL: reward_mode={ql_reward_mode}, schedule={ql_schedule}, seed={ql_seed}")
    print(f"Reference SARSA: lambda={sarsa_lam}, seed={sarsa_seed}")

    # Runtime: pull the real per-run wall-clock from the ledger, averaged
    # across this variant's seeds. VI's runtime already lives directly in
    # its own model JSON (read separately in main()); QL/SARSA never wrote
    # runtime into their summary CSVs, so the ledger is the only place it
    # actually exists for them.
    ql_variant_ids = [f"q_learning_{ql_reward_mode}_{ql_schedule}_seed{s}" for s in grid['seeds']]
    ql_runtime = _mean_runtime_from_ledger('q_learning', ql_variant_ids)

    sarsa_variant_ids = [f"sarsa_{grid['sarsa_lambda']['reward_mode']}_lambda{sarsa_lam}_seed{s}"
                          for s in grid['seeds']]
    sarsa_runtime = _mean_runtime_from_ledger('sarsa_lambda', sarsa_variant_ids)

    def agreement_report(name, states, Q, visits):
        rows = []
        all_agree, all_total = 0, 0
        visited_agree, visited_total = 0, 0
        for i, s in enumerate(states):
            if s not in vi_policy:
                continue
            agent_action = int(np.argmax(Q[i]))
            vi_action = vi_policy[s]
            agree = agent_action == vi_action
            all_total += 1
            all_agree += int(agree)
            is_visited = visits[i] > 0
            if is_visited:
                visited_total += 1
                visited_agree += int(agree)
                if not agree:
                    gap = (_q_star_value(vi, vi_V, s, vi_action)
                           - _q_star_value(vi, vi_V, s, agent_action))
                    rows.append({
                        'algorithm': name, 'state': s, 'vi_action': vi_action,
                        'agent_action': agent_action, 'action_gap': gap,
                        'visits': int(visits[i]),
                    })
        all_rate = all_agree / all_total if all_total else float('nan')
        visited_rate = visited_agree / visited_total if visited_total else float('nan')
        return all_rate, visited_rate, visited_total, rows

    ql_all, ql_visited, ql_n_visited, ql_disagree = agreement_report('q_learning', ql_states, ql_Q, ql_visits)
    sarsa_all, sarsa_visited, sarsa_n_visited, sarsa_disagree = agreement_report(
        'sarsa_lambda', sarsa_states, sarsa_Q, sarsa_visits)

    print(f"QL agreement: all-states={ql_all:.3f}  visited-only={ql_visited:.3f} (n={ql_n_visited})")
    print(f"SARSA agreement: all-states={sarsa_all:.3f}  visited-only={sarsa_visited:.3f} (n={sarsa_n_visited})")

    return {
        'vi': {'model': vi_model, 'policy': vi_policy, 'V': vi_V, 'vi_obj': vi},
        'ql': {'reward_mode': ql_reward_mode, 'schedule': ql_schedule, 'seed': ql_seed,
               'model': ql_model, 'all_agreement': ql_all, 'visited_agreement': ql_visited,
               'n_visited': ql_n_visited, 'disagreements': ql_disagree, 'runtime_seconds': ql_runtime},
        'sarsa': {'lam': sarsa_lam, 'seed': sarsa_seed, 'model': sarsa_model,
                  'all_agreement': sarsa_all, 'visited_agreement': sarsa_visited,
                  'n_visited': sarsa_n_visited, 'disagreements': sarsa_disagree,
                  'runtime_seconds': sarsa_runtime},
        'env': env,
    }


def select_sample_states(disagreements, env):
    if not disagreements:
        return []
    by_gap = sorted(disagreements, key=lambda d: d['action_gap'], reverse=True)
    largest_gap = by_gap[0]

    nonzero = [d for d in disagreements if d['action_gap'] > 1e-6]
    smallest_nonzero = min(nonzero, key=lambda d: d['action_gap']) if nonzero else None

    def dist_to_penalty(state):
        r, c = state[0], state[1]
        return min(abs(r - pr) + abs(c - pc) for pr, pc in env.penalty_cells)

    nearest_penalty = min(disagreements, key=lambda d: dist_to_penalty(d['state']))

    picks = {'largest_action_gap': largest_gap,
             'smallest_nonzero_gap': smallest_nonzero,
             'nearest_penalty_cell': nearest_penalty}
    return picks


def plot_policy_diff(vi_policy, agent_Q_by_state, visits_by_state, states,
                      maze_size, has_key, energy, out_path, title):
    """Green/red = agree/disagree on a VISITED state; unvisited cells are left
    NaN (blank) rather than colored, since an unvisited state's Q-row is still
    at its initial (untrained) value and comparing it to VI is meaningless --
    it would just reflect argmax-of-zeros tie-breaking, not learned behavior."""
    diff = np.full((maze_size, maze_size), np.nan)
    for s in states:
        r, c, hk, en = s
        if hk != has_key or en != energy:
            continue
        if s not in vi_policy:
            continue
        if visits_by_state.get(s, 0) <= 0:
            continue  # unvisited -- leave as NaN, don't plot a misleading color
        agent_action = agent_Q_by_state.get(s)
        if agent_action is None:
            continue
        diff[r, c] = 1.0 if agent_action == vi_policy[s] else 0.0

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap('RdYlGn')
    cmap.set_bad(color='lightgray')  # unvisited/NaN cells render gray, not a diff color
    im = ax.imshow(diff, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label='1=agree, 0=disagree (gray=unvisited)')
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
    maze_size = env.maze_size
    max_energy = env.MAX_ENERGY

    out_dir = Path('results/raw_data/comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    vi_csv = pd.read_csv('results/raw_data/vi/vi_gamma_sweep.csv')
    ref_gamma = cfg['experiment_grid']['value_iteration']['reference_gamma']
    vi_row = vi_csv[vi_csv['gamma'] == ref_gamma].iloc[0]

    ql_summary = pd.read_csv('results/raw_data/q_learning/q_learning_summary.csv')
    ql_subset = ql_summary[(ql_summary['reward_mode'] == result['ql']['reward_mode']) &
                            (ql_summary['schedule'] == result['ql']['schedule'])]
    sarsa_summary = pd.read_csv('results/raw_data/sarsa/sarsa_summary.csv')
    sarsa_subset = sarsa_summary[sarsa_summary['lambda'] == result['sarsa']['lam']]

    summary_rows.append({
        'algorithm': 'value_iteration', 'type': 'model-based', 'progress_unit': 'bellman_iteration',
        'main_output': 'V,policy', 'runtime_seconds': vi_row['runtime_seconds'],
        'convergence': vi_row['converged'], 'samples_needed': vi_row['iterations'],
        'stability': None, 'eval_success_rate': vi_row['eval_success_rate'],
        'eval_mean_steps': vi_row['eval_mean_steps'], 'vi_agreement_visited': 1.0,
        'median_action_gap': 0.0,
    })
    ql_gaps = [d['action_gap'] for d in result['ql']['disagreements']]
    summary_rows.append({
        'algorithm': 'q_learning', 'type': 'model-free/off-policy', 'progress_unit': 'episode',
        'main_output': 'Q,policy', 'runtime_seconds': result['ql']['runtime_seconds'],
        'convergence': None, 'samples_needed': ql_subset['episodes_to_90pct_success'].mean(),
        'stability': ql_subset['eval_success_rate'].std(),
        'eval_success_rate': ql_subset['eval_success_rate'].mean(),
        'eval_mean_steps': ql_subset['eval_mean_steps'].mean(),
        'vi_agreement_visited': result['ql']['visited_agreement'],
        'median_action_gap': float(np.median(ql_gaps)) if ql_gaps else 0.0,
    })
    sarsa_gaps = [d['action_gap'] for d in result['sarsa']['disagreements']]
    summary_rows.append({
        'algorithm': 'sarsa_lambda', 'type': 'model-free/on-policy', 'progress_unit': 'episode',
        'main_output': 'Q,E,policy', 'runtime_seconds': result['sarsa']['runtime_seconds'],
        'convergence': None, 'samples_needed': sarsa_subset['episodes_to_90pct_success'].mean(),
        'stability': sarsa_subset['late_return_std'].mean(),
        'eval_success_rate': sarsa_subset['eval_success_rate'].mean(),
        'eval_mean_steps': sarsa_subset['eval_mean_steps'].mean(),
        'vi_agreement_visited': result['sarsa']['visited_agreement'],
        'median_action_gap': float(np.median(sarsa_gaps)) if sarsa_gaps else 0.0,
    })

    with open(out_dir / 'comparison_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"saved {out_dir / 'comparison_summary.csv'}")

    sample_rows = []
    for algo_name, disagreements in (('q_learning', result['ql']['disagreements']),
                                      ('sarsa_lambda', result['sarsa']['disagreements'])):
        picks = select_sample_states(disagreements, env)
        for category, d in picks.items():
            if d is None:
                continue
            sample_rows.append({
                'algorithm': algo_name, 'category': category,
                'state': d['state'], 'vi_action': d['vi_action'],
                'agent_action': d['agent_action'], 'action_gap': d['action_gap'],
                'visits': d['visits'],
            })
    if sample_rows:
        with open(out_dir / 'comparison_sample_states.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_rows)
        print(f"saved {out_dir / 'comparison_sample_states.csv'}")

    # Policy-difference maps -- now generated for BOTH Q-Learning and SARSA(λ)
    # against VI (previously only Q-Learning's were produced), and both use
    # the visits-based masking from plot_policy_diff so unvisited states show
    # as gray instead of a misleading agree/disagree color.
    ql_Q_by_state = {tuple(s): int(np.argmax(q)) for s, q in
                      zip(result['ql']['model']['states'], result['ql']['model']['Q'])}
    ql_visits_by_state = {tuple(s): v for s, v in
                           zip(result['ql']['model']['states'], result['ql']['model']['visits'])}

    sarsa_Q_by_state = {tuple(s): int(np.argmax(q)) for s, q in
                         zip(result['sarsa']['model']['states'], result['sarsa']['model']['Q'])}
    sarsa_visits_by_state = {tuple(s): v for s, v in
                              zip(result['sarsa']['model']['states'], result['sarsa']['model']['visits'])}

    for has_key, tag in ((0, 'prekey'), (1, 'postkey')):
        plot_policy_diff(
            result['vi']['policy'], ql_Q_by_state, ql_visits_by_state,
            [tuple(s) for s in result['ql']['model']['states']],
            maze_size, has_key, max_energy,
            f'results/figures/comparison/policy_diff_q_learning_{tag}.png',
            f'QL vs VI policy agreement (has_key={has_key}, energy={max_energy})',
        )
        plot_policy_diff(
            result['vi']['policy'], sarsa_Q_by_state, sarsa_visits_by_state,
            [tuple(s) for s in result['sarsa']['model']['states']],
            maze_size, has_key, max_energy,
            f'results/figures/comparison/policy_diff_sarsa_{tag}.png',
            f'SARSA(lambda) vs VI policy agreement (has_key={has_key}, energy={max_energy})',
        )


if __name__ == '__main__':
    main()