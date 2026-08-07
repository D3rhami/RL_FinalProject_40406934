import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.maze import MazeEnv, CellType, load_config
from agents.value_iteration import ValueIteration
from experiments.run_experiments import derive_seeds

MAP_PATH = 'environments/maps/source_maze.json'

# Match gui_micro_roadmap / CustomTkinter maze hex table
_CELL_COLOR = {
    CellType.WALL:    '#2D2D2D',
    CellType.EMPTY:   '#FFFFFF',
    CellType.PENALTY: '#FF6464',
    CellType.START:   '#64C864',
    CellType.KEY:     '#FFD700',
    CellType.DOOR:    '#8B5A2B',
    CellType.GOAL:    '#00C800',
}
_CELL_LABEL = {
    CellType.START:   ('S', '#003300'),
    CellType.KEY:     ('K', '#000000'),
    CellType.DOOR:    ('D', '#FFFFFF'),
    CellType.GOAL:    ('G', '#003300'),
    CellType.PENALTY: ('P', '#FFFFFF'),
}
_ARROW_SYM = {0: '\u2191', 1: '\u2193', 2: '\u2190', 3: '\u2192'}

VALUE_CMAP = LinearSegmentedColormap.from_list(
    'value_blue', ['#FFFFFF', '#C6DBEF', '#6BAED6', '#2171B5', '#08306B'])
VALUE_CMAP.set_bad(color=(1, 1, 1, 0))
AGREE_CMAP = LinearSegmentedColormap.from_list(
    'agree_rwg', ['#D73027', '#FFFFFF', '#1A9850'])
AGREE_CMAP.set_bad(color=(1, 1, 1, 0))


def _draw_maze_cells(ax, env):
    """Base grid only (no letters/arrows). Overlay comes after imshow."""
    for r in range(env.maze_size):
        for c in range(env.maze_size):
            ct = CellType(int(env.grid[r, c]))
            ax.add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor=_CELL_COLOR.get(ct, '#FFFFFF'),
                edgecolor='#B0B0B0', linewidth=0.35, zorder=1))
    ax.set_xlim(-0.5, env.maze_size - 0.5)
    ax.set_ylim(env.maze_size - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_facecolor('#FFFFFF')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def _overlay_cell_labels(ax, env):
    for r in range(env.maze_size):
        for c in range(env.maze_size):
            ct = CellType(int(env.grid[r, c]))
            if ct in _CELL_LABEL:
                txt, tc = _CELL_LABEL[ct]
                ax.text(c, r - 0.32, txt, ha='center', va='top',
                        fontsize=8, color=tc, fontweight='bold', zorder=5)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_ledger_runtime():
    path = Path('results/raw_data/run_ledger.csv')
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[df['status'] == 'success']
    return df.groupby('algorithm')['runtime_seconds'].mean().to_dict()


def _check_prereqs(cfg):
    grid = cfg['experiment_grid']
    seeds, _ = derive_seeds(cfg['base_seed'], n=grid['n_seeds'])
    ref_gamma = grid['value_iteration']['reference_gamma']
    vi_rm = grid['value_iteration']['reward_mode']
    vi_path = Path(f'results/models/vi/vi_{vi_rm}_gamma{ref_gamma}.json')
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

    vi_model = _load_json(f'results/models/vi/vi_{vi_reward_mode}_gamma{ref_gamma}.json')
    vi_states = [tuple(s) for s in vi_model['states']]
    vi_policy = {s: a for s, a in zip(vi_states, vi_model['policy']) if a != -1}
    vi_V = np.array(vi_model['V'])

    env = MazeEnv(MAP_PATH, config=cfg, reward_mode=vi_reward_mode)
    vi = ValueIteration(env, gamma=ref_gamma, theta=cfg['value_iteration']['theta'],
                         reward_mode=vi_reward_mode)

    ql_rm, ql_sched, ql_seed = _pick_best_ql_variant(cfg, seeds)
    ql_model = _load_json(f'results/models/q_learning/q_learning_{ql_rm}_{ql_sched}_seed{ql_seed}.json')
    ql_states = [tuple(s) for s in ql_model['states']]
    ql_Q = np.array(ql_model['Q'])
    ql_visits = np.array(ql_model['visits'])

    sarsa_lam, sarsa_seed = _pick_best_sarsa_variant(cfg, seeds)
    sarsa_rm = grid['sarsa_lambda']['reward_mode']
    sarsa_model = _load_json(f'results/models/sarsa/sarsa_{sarsa_rm}_lambda{sarsa_lam}_seed{sarsa_seed}.json')
    sarsa_states = [tuple(s) for s in sarsa_model['states']]
    sarsa_Q = np.array(sarsa_model['Q'])
    sarsa_visits = np.array(sarsa_model['visits'])

    print(f"Reference VI: gamma={ref_gamma}")
    print(f"Reference QL: reward_mode={ql_rm}, schedule={ql_sched}, seed={ql_seed}")
    print(f"Reference SARSA: lambda={sarsa_lam}, seed={sarsa_seed}")

    def agreement_report(name, states, Q, visits):
        disagreements = []
        all_agree = all_total = visited_agree = visited_total = 0
        near_pen_agree = near_pen_total = 0
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
                if env.danger_dist[s[0], s[1]] <= 1:
                    near_pen_total += 1
                    near_pen_agree += int(agree)
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
        near_pen_rate = near_pen_agree / near_pen_total if near_pen_total else float('nan')
        return all_rate, visited_rate, visited_total, near_pen_rate, near_pen_total, disagreements

    (ql_all, ql_vis, ql_n, ql_near_pen, ql_near_pen_n, ql_dis) = agreement_report(
        'q_learning', ql_states, ql_Q, ql_visits)
    (sarsa_all, sarsa_vis, sarsa_n, sarsa_near_pen, sarsa_near_pen_n, sarsa_dis) = agreement_report(
        'sarsa_lambda', sarsa_states, sarsa_Q, sarsa_visits)

    print(f"QL    agreement: all={ql_all:.3f}  visited-only={ql_vis:.3f}  (n={ql_n})  "
          f"near-penalty={ql_near_pen:.3f} (n={ql_near_pen_n})")
    print(f"SARSA agreement: all={sarsa_all:.3f}  visited-only={sarsa_vis:.3f}  (n={sarsa_n})  "
          f"near-penalty={sarsa_near_pen:.3f} (n={sarsa_near_pen_n})")

    return {
        'vi': {'model': vi_model, 'policy': vi_policy, 'V': vi_V, 'vi_obj': vi},
        'ql': {'reward_mode': ql_rm, 'schedule': ql_sched, 'seed': ql_seed,
               'model': ql_model, 'states': ql_states, 'Q': ql_Q, 'visits': ql_visits,
               'all_agreement': ql_all, 'visited_agreement': ql_vis, 'n_visited': ql_n,
               'near_penalty_agreement': ql_near_pen, 'near_penalty_n': ql_near_pen_n,
               'disagreements': ql_dis},
        'sarsa': {'lam': sarsa_lam, 'seed': sarsa_seed,
                  'model': sarsa_model, 'states': sarsa_states,
                  'Q': sarsa_Q, 'visits': sarsa_visits,
                  'all_agreement': sarsa_all, 'visited_agreement': sarsa_vis, 'n_visited': sarsa_n,
                  'near_penalty_agreement': sarsa_near_pen, 'near_penalty_n': sarsa_near_pen_n,
                  'disagreements': sarsa_dis},
        'env': env,
    }


def select_sample_states(disagreements, env, min_visits=10):
    if not disagreements:
        return {}
    well_visited = [d for d in disagreements if d['visits'] >= min_visits]
    pool = well_visited if well_visited else disagreements
    by_gap = sorted(pool, key=lambda d: d['action_gap'], reverse=True)
    nonzero = [d for d in by_gap if d['action_gap'] > 1e-6]
    nearest_penalty = min(
        disagreements,
        key=lambda d: (min(abs(d['state'][0] - pr) + abs(d['state'][1] - pc)
                           for pr, pc in env.penalty_cells), -d['visits']),
    )
    return {
        'largest_action_gap': by_gap[0],
        'smallest_nonzero_gap': nonzero[-1] if nonzero else None,
        'nearest_penalty_cell': nearest_penalty,
    }


def _representative_energy(result):
    totals = {}
    for key in ('ql', 'sarsa'):
        for s, v in zip(result[key]['states'], result[key]['visits']):
            r, c, hk, en = s
            totals[(r, c, hk, en)] = totals.get((r, c, hk, en), 0) + int(v)
    rep = {}
    for (r, c, hk, en), v in totals.items():
        cur = rep.get((r, c, hk))
        if cur is None or v > cur[1]:
            rep[(r, c, hk)] = (en, v)
    return {k: v[0] for k, v in rep.items()}


def plot_policy_grid(result, env, out_path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 19), facecolor='white')
    fig.suptitle('Learned policy per algorithm \u2014 arrows = greedy action, '
                 'shade = own V / max-Q at most-visited energy', fontsize=11)

    vi_model = result['vi']['model']
    vi_states = [tuple(s) for s in vi_model['states']]
    vi_V = {s: v for s, v in zip(vi_states, vi_model['V'])}
    vi_policy = result['vi']['policy']

    def ql_sarsa_lookup(key):
        states, Q, visits = result[key]['states'], result[key]['Q'], result[key]['visits']
        val, pol = {}, {}
        for s, q, v in zip(states, Q, visits):
            if v > 0:
                val[s] = float(np.max(q))
                pol[s] = int(np.argmax(q))
        return val, pol

    ql_val, ql_pol = ql_sarsa_lookup('ql')
    sarsa_val, sarsa_pol = ql_sarsa_lookup('sarsa')

    max_energy = env.MAX_ENERGY
    rep = _representative_energy(result)

    scale_vals = []
    for (r, c, hk), en in rep.items():
        for d in (vi_V, ql_val, sarsa_val):
            v = d.get((r, c, hk, en))
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                scale_vals.append(v)
    if not scale_vals:
        scale_vals = [v for s, v in vi_V.items() if s[3] == max_energy]
    global_vmax = max(scale_vals) if scale_vals else 1.0
    global_vmin = min(0.0, min(scale_vals)) if scale_vals else 0.0

    rows = [
        ('Value Iteration', lambda s: vi_V.get(s, np.nan), lambda s: vi_policy.get(s)),
        ('Q-Learning',      lambda s: ql_val.get(s, np.nan), lambda s: ql_pol.get(s)),
        ('SARSA(\u03bb)',     lambda s: sarsa_val.get(s, np.nan), lambda s: sarsa_pol.get(s)),
    ]

    for row_idx, (label, val_fn, pol_fn) in enumerate(rows):
        for col_idx, has_key in enumerate((0, 1)):
            ax = axes[row_idx][col_idx]
            _draw_maze_cells(ax, env)
            grid_val = np.full((env.maze_size, env.maze_size), np.nan)
            for r in range(env.maze_size):
                for c in range(env.maze_size):
                    if int(env.grid[r, c]) == CellType.WALL:
                        continue
                    en = rep.get((r, c, has_key), max_energy)
                    v = val_fn((r, c, has_key, en))
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        grid_val[r, c] = v
            ax.imshow(grid_val, cmap=VALUE_CMAP, vmin=global_vmin, vmax=global_vmax,
                      alpha=0.65, zorder=2, interpolation='nearest',
                      extent=(-0.5, env.maze_size - 0.5, env.maze_size - 0.5, -0.5))
            for r in range(env.maze_size):
                for c in range(env.maze_size):
                    en = rep.get((r, c, has_key), max_energy)
                    a = pol_fn((r, c, has_key, en))
                    if a is not None:
                        ax.text(c, r + 0.08, _ARROW_SYM[a], ha='center', va='center',
                                fontsize=11, color='#111111', zorder=6,
                                fontweight='bold')
            _overlay_cell_labels(ax, env)
            key_label = 'No key (k=0)' if has_key == 0 else 'Has key (k=1)'
            ax.set_title(f'{label} \u2014 {key_label}', fontsize=9)

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=VALUE_CMAP, norm=plt.Normalize(global_vmin, global_vmax))
    fig.colorbar(sm, cax=cax, label='V (VI) / max Q (QL, SARSA) at most-visited energy')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved {out_path}')


def plot_agreement_grid(vi_policy, result, env, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), facecolor='white')
    fig.suptitle('Agent policy vs. VI agreement (share of visited energy levels)',
                 fontsize=11)
    algo_items = [('ql', 'Q-Learning'), ('sarsa', 'SARSA(\u03bb)')]
    n_walkable = int(np.sum(env.grid != CellType.WALL))
    im = None
    for row_idx, (key, label) in enumerate(algo_items):
        states, Q, visits = result[key]['states'], result[key]['Q'], result[key]['visits']
        Q_by_state = {s: int(np.argmax(q)) for s, q in zip(states, Q)}
        visits_by_state = {s: int(v) for s, v in zip(states, visits)}
        for col_idx, has_key in enumerate((0, 1)):
            ax = axes[row_idx][col_idx]
            _draw_maze_cells(ax, env)
            agree = np.zeros((env.maze_size, env.maze_size))
            total = np.zeros((env.maze_size, env.maze_size))
            for s in states:
                r, c, hk, en = s
                if hk != has_key or s not in vi_policy or visits_by_state.get(s, 0) == 0:
                    continue
                agent_a = Q_by_state.get(s)
                if agent_a is None:
                    continue
                total[r, c] += 1
                if agent_a == vi_policy[s]:
                    agree[r, c] += 1
            frac = np.divide(agree, total, out=np.full_like(agree, np.nan), where=total > 0)
            im = ax.imshow(frac, cmap=AGREE_CMAP, vmin=0, vmax=1, alpha=0.70, zorder=2,
                           interpolation='nearest',
                           extent=(-0.5, env.maze_size - 0.5, env.maze_size - 0.5, -0.5))
            _overlay_cell_labels(ax, env)
            visited_cells = int(np.sum(total > 0))
            key_label = 'No key (k=0)' if has_key == 0 else 'Has key (k=1)'
            ax.set_title(f'{label} \u2014 {key_label}\nVisited: {visited_cells} / '
                         f'{n_walkable}', fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02,
                 label='Agreement fraction (1.0 = always matches VI)')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved {out_path}')


def main():
    cfg = load_config()
    _check_prereqs(cfg)
    result = build_comparison(cfg)
    env = result['env']

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

    ledger_runtime = _load_ledger_runtime()
    ql_gaps = [d['action_gap'] for d in result['ql']['disagreements']]
    sarsa_gaps = [d['action_gap'] for d in result['sarsa']['disagreements']]

    summary_rows = [
        {'algorithm': 'value_iteration', 'type': 'model-based',
         'progress_unit': 'bellman_iteration', 'main_output': 'V,policy',
         'runtime_seconds': vi_row['runtime_seconds'], 'convergence': vi_row['converged'],
         'samples_needed': vi_row['iterations'], 'stability': None,
         'stability_definition': 'n/a (deterministic algorithm)',
         'eval_success_rate': vi_row['eval_success_rate'], 'eval_mean_steps': vi_row['eval_mean_steps'],
         'vi_agreement_visited': 1.0, 'near_penalty_agreement': 1.0, 'median_action_gap': 0.0},
        {'algorithm': 'q_learning', 'type': 'model-free/off-policy',
         'progress_unit': 'episode', 'main_output': 'Q,policy',
         'runtime_seconds': ledger_runtime.get('q_learning'), 'convergence': None,
         'samples_needed': ql_subset['episodes_to_90pct_success'].mean(),
         'stability': ql_subset['eval_success_rate'].std(),
         'stability_definition': 'cross-seed std of eval_success_rate',
         'eval_success_rate': ql_subset['eval_success_rate'].mean(),
         'eval_mean_steps': ql_subset['eval_mean_steps'].mean(),
         'vi_agreement_visited': result['ql']['visited_agreement'],
         'near_penalty_agreement': result['ql']['near_penalty_agreement'],
         'median_action_gap': float(np.median(ql_gaps)) if ql_gaps else 0.0},
        {'algorithm': 'sarsa_lambda', 'type': 'model-free/on-policy',
         'progress_unit': 'episode', 'main_output': 'Q,E,policy',
         'runtime_seconds': ledger_runtime.get('sarsa_lambda'), 'convergence': None,
         'samples_needed': sarsa_subset['episodes_to_90pct_success'].mean(),
         'stability': sarsa_subset['late_return_std'].mean(),
         'stability_definition': 'mean within-run std of return, last 1000 episodes',
         'eval_success_rate': sarsa_subset['eval_success_rate'].mean(),
         'eval_mean_steps': sarsa_subset['eval_mean_steps'].mean(),
         'vi_agreement_visited': result['sarsa']['visited_agreement'],
         'near_penalty_agreement': result['sarsa']['near_penalty_agreement'],
         'median_action_gap': float(np.median(sarsa_gaps)) if sarsa_gaps else 0.0},
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

    plot_policy_grid(result, env, 'results/figures/comparison/policy_grid.png')
    plot_agreement_grid(result['vi']['policy'], result, env,
                        'results/figures/comparison/agreement_grid.png')


if __name__ == '__main__':
    main()
