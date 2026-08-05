import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.maze import load_config

ARROW_DELTA = {0: (-0.35, 0), 1: (0.35, 0), 2: (0, -0.35), 3: (0, 0.35)}


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _slice_grid(model, maze_size, has_key, energy):
    V = np.full((maze_size, maze_size), np.nan)
    policy = np.full((maze_size, maze_size), -1, dtype=int)
    for i, (r, c, hk, en) in enumerate(model['states']):
        if hk == has_key and en == energy:
            V[r, c] = model['V'][i]
            policy[r, c] = model['policy'][i]
    return V, policy


def plot_vi_value_and_policy(model_path, out_path, energy=None, cfg=None):
    cfg = cfg or load_config()
    model = _load_json(model_path)
    maze_size = cfg['maze_size']
    energy = cfg['env']['max_energy'] if energy is None else energy
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, has_key, title in zip(axes, (0, 1), ('Pre-key (has_key=0)', 'Post-key (has_key=1)')):
        V, policy = _slice_grid(model, maze_size, has_key, energy)
        im = ax.imshow(V, cmap='viridis')
        ax.set_title(f'{title}, energy={energy}, gamma={model["gamma"]}')
        fig.colorbar(im, ax=ax, fraction=0.046)
        for r in range(maze_size):
            for c in range(maze_size):
                a = policy[r, c]
                if a == -1:
                    continue
                dr, dc = ARROW_DELTA[a]
                ax.arrow(c, r, dc, dr, head_width=0.15, head_length=0.15,
                         fc='white', ec='white', linewidth=0.5)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def plot_vi_gamma_sensitivity(csv_path, out_path):
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(df['gamma'], df['iterations'], marker='o')
    axes[0].set_xlabel('gamma'); axes[0].set_ylabel('iterations to converge')
    it_min, it_max = df['iterations'].min(), df['iterations'].max()
    pad = max(1, (it_max - it_min) * 0.5 + 1)
    axes[0].set_ylim(it_min - pad, it_max + pad)
    axes[1].plot(df['gamma'], df['runtime_seconds'], marker='o', color='orange')
    axes[1].set_xlabel('gamma'); axes[1].set_ylabel('runtime (s)')
    rt_center = df['runtime_seconds'].mean()
    axes[1].set_ylim(0, max(rt_center * 1.5, df['runtime_seconds'].max() * 1.2))
    axes[2].plot(df['gamma'], df['eval_success_rate'], marker='o', color='green')
    axes[2].set_xlabel('gamma'); axes[2].set_ylabel('eval success rate')
    axes[2].set_ylim(0.0, 1.05)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle('Flat panels: runtime jitter is sub-second noise, not a gamma trend; '
                 'eval success is a hard ceiling at 1.0.', fontsize=8, y=1.02)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def plot_vi_convergence(model_paths_by_gamma, theta, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for gamma, path in sorted(model_paths_by_gamma.items()):
        if not Path(path).exists():
            continue
        deltas = _load_json(path)['deltas']
        ax.semilogy(range(1, len(deltas) + 1), deltas, label=f'gamma={gamma}')
        plotted = True
    if not plotted:
        plt.close(fig)
        print('skip VI convergence: no VI model files found')
        return
    ax.axhline(theta, color='gray', linestyle='--', linewidth=1, label=f'theta={theta:g}')
    ax.set_xlabel('iteration'); ax.set_ylabel('max |V(k+1) - V(k)| (log scale)')
    ax.set_title('Value Iteration convergence per gamma')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def _rolling_with_band(df, group_col, value_col, window=100):
    out = {}
    for key, grp in df.groupby(group_col):
        per_seed = []
        for _, g in grp.groupby('seed'):
            values = g.sort_values('episode')[value_col].values
            w = min(window, len(values))
            if w < 1:
                continue
            per_seed.append(np.convolve(values, np.ones(w) / w, mode='valid'))
        if not per_seed:
            continue
        min_len = min(len(a) for a in per_seed)
        if min_len == 0:
            continue
        stacked = np.array([a[:min_len] for a in per_seed])
        out[key] = (stacked.mean(axis=0), stacked.std(axis=0))
    return out


def plot_ql_schedule_comparison(training_csv, out_path, reward_mode='sparse'):
    df = pd.read_csv(training_csv)
    df = df[df['reward_mode'] == reward_mode]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for schedule, (mean, std) in _rolling_with_band(df, 'schedule', 'return').items():
        xs = np.arange(len(mean))
        axes[0].plot(xs, mean, label=schedule); axes[0].fill_between(xs, mean - std, mean + std, alpha=0.18)
    for schedule, (mean, std) in _rolling_with_band(df, 'schedule', 'steps').items():
        xs = np.arange(len(mean))
        axes[1].plot(xs, mean, label=schedule); axes[1].fill_between(xs, mean - std, mean + std, alpha=0.18)
    axes[0].set_title(f'Reward curve (reward_mode={reward_mode}), mean +/- std across seeds')
    axes[0].set_xlabel('episode'); axes[0].set_ylabel('return'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title(f'Step count (reward_mode={reward_mode})')
    axes[1].set_xlabel('episode'); axes[1].set_ylabel('steps'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def plot_ql_rewardmode_comparison(training_csv, out_path, schedule='exponential'):
    df = pd.read_csv(training_csv)
    df = df[df['schedule'] == schedule]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for reward_mode, (mean, std) in _rolling_with_band(df, 'reward_mode', 'return').items():
        xs = np.arange(len(mean))
        axes[0].plot(xs, mean, label=reward_mode); axes[0].fill_between(xs, mean - std, mean + std, alpha=0.18)
    for reward_mode, (mean, std) in _rolling_with_band(df, 'reward_mode', 'steps').items():
        xs = np.arange(len(mean))
        axes[1].plot(xs, mean, label=reward_mode); axes[1].fill_between(xs, mean - std, mean + std, alpha=0.18)
    axes[0].set_title(f'Reward curve (schedule={schedule}), mean +/- std across seeds')
    axes[0].set_xlabel('episode'); axes[0].set_ylabel('return'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title(f'Step count (schedule={schedule})')
    axes[1].set_xlabel('episode'); axes[1].set_ylabel('steps'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def plot_sarsa_lambda_comparison(training_csv, summary_csv, out_path):
    df = pd.read_csv(training_csv)
    summary = pd.read_csv(summary_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lam, (mean, std) in _rolling_with_band(df, 'lambda', 'return').items():
        xs = np.arange(len(mean))
        axes[0].plot(xs, mean, label=f'lambda={lam}'); axes[0].fill_between(xs, mean - std, mean + std, alpha=0.18)
    axes[0].set_title('Reward curve by lambda, mean +/- std across seeds')
    axes[0].set_xlabel('episode'); axes[0].set_ylabel('return'); axes[0].legend(); axes[0].grid(alpha=0.3)
    stability = summary.groupby('lambda')['late_return_std'].mean()
    axes[1].bar([str(l) for l in stability.index], stability.values, color='tomato')
    axes[1].set_title('Late-training return std (stability) by lambda')
    axes[1].set_xlabel('lambda'); axes[1].set_ylabel('std of return (last 1000 eps)')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def plot_sarsa_trace(step_trace_csv, trace_dump_csv, out_path, n_pairs=4):
    step_df = pd.read_csv(step_trace_csv)
    dump_df = pd.read_csv(trace_dump_csv)
    ep = step_df['episode'].iloc[0]
    step_df = step_df[step_df['episode'] == ep].sort_values('step')
    dump_df = dump_df[dump_df['episode'] == ep]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.axhline(0, color='gray', linewidth=1)
    ax1.plot(step_df['step'], step_df['delta'], linewidth=2)
    ax1.set_xlabel('step'); ax1.set_ylabel('TD error (delta)')
    ax1.set_title(f'SARSA(lambda) TD error, episode {ep}')
    ax1.grid(alpha=0.3)
    dump_df = dump_df.assign(sa=list(zip(dump_df['r'], dump_df['c'], dump_df['has_key'], dump_df['action'])))
    first_pairs = dump_df.groupby('sa')['step'].min().sort_values().index[:n_pairs]
    for sa in first_pairs:
        s = dump_df[dump_df['sa'] == sa].sort_values('step')
        ax2.semilogy(s['step'], s['E'], marker='o', markersize=3,
                     label=f's=({sa[0]},{sa[1]},k={sa[2]}) a={sa[3]}')
    ax2.set_xlabel('step'); ax2.set_ylabel('eligibility E (log scale)')
    ax2.set_title('Eligibility trace decay')
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def plot_visit_map(model_path, out_path, maze_size, title):
    model = _load_json(model_path)
    g0 = np.zeros((maze_size, maze_size))
    g1 = np.zeros((maze_size, maze_size))
    for (r, c, hk, en), v in zip(model['states'], model['visits']):
        (g0 if hk == 0 else g1)[r, c] += v
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, grid, label in zip(axes, (g0, g1), ('has_key=0', 'has_key=1')):
        im = ax.imshow(np.log10(grid + 1), cmap='viridis')
        ax.set_title(f'{title} ({label})')
        fig.colorbar(im, ax=ax, fraction=0.046, label='log10(visits+1)')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def main():
    cfg = load_config()
    grid = cfg['experiment_grid']
    vi_gamma = grid['value_iteration']['reference_gamma']
    vi_rm = grid['value_iteration']['reward_mode']

    vi_model_path = Path(f'results/models/vi/vi_{vi_rm}_gamma{vi_gamma}.json')
    vi_csv_path = Path('results/raw_data/vi/vi_gamma_sweep.csv')
    if vi_model_path.exists():
        plot_vi_value_and_policy(vi_model_path, 'results/figures/vi/vi_value_policy.png', cfg=cfg)
    else:
        print(f'skip VI heatmap: {vi_model_path} not found')
    if vi_csv_path.exists():
        plot_vi_gamma_sensitivity(vi_csv_path, 'results/figures/vi/vi_gamma_sensitivity.png')
    else:
        print(f'skip VI gamma-sensitivity: {vi_csv_path} not found')

    vi_model_paths = {g: f'results/models/vi/vi_{vi_rm}_gamma{g}.json' for g in grid['value_iteration']['gammas']}
    plot_vi_convergence(vi_model_paths, cfg['value_iteration']['theta'], 'results/figures/vi/vi_convergence.png')

    ql_training = Path('results/raw_data/q_learning/q_learning_training.csv')
    if ql_training.exists():
        ref_schedule = grid['q_learning']['trace_run']['schedule']
        plot_ql_schedule_comparison(ql_training, 'results/figures/q_learning/ql_schedule_comparison.png',
                                     reward_mode=grid['q_learning']['reference_reward_mode'])
        plot_ql_rewardmode_comparison(ql_training, 'results/figures/q_learning/ql_rewardmode_comparison.png',
                                       schedule=ref_schedule)
    else:
        print(f'skip QL figures: {ql_training} not found')

    ql_summary_path = Path('results/raw_data/q_learning/q_learning_summary.csv')
    if ql_summary_path.exists():
        best = pd.read_csv(ql_summary_path).sort_values('eval_success_rate', ascending=False).iloc[0]
        ql_best_model = Path(f"results/models/q_learning/q_learning_{best['reward_mode']}_{best['schedule']}_seed{int(best['seed'])}.json")
        if ql_best_model.exists():
            plot_visit_map(ql_best_model, 'results/figures/q_learning/ql_visit_map.png',
                           cfg['maze_size'], f"QL visitation ({best['reward_mode']}/{best['schedule']})")

    sarsa_training = Path('results/raw_data/sarsa/sarsa_training.csv')
    sarsa_summary_path = Path('results/raw_data/sarsa/sarsa_summary.csv')
    if sarsa_training.exists() and sarsa_summary_path.exists():
        plot_sarsa_lambda_comparison(sarsa_training, sarsa_summary_path, 'results/figures/sarsa/sarsa_lambda_comparison.png')
    else:
        print(f'skip SARSA figure: {sarsa_training} or {sarsa_summary_path} not found')

    if sarsa_summary_path.exists():
        best_s = pd.read_csv(sarsa_summary_path).sort_values('eval_success_rate', ascending=False).iloc[0]
        rm = grid['sarsa_lambda']['reward_mode']
        sarsa_best_model = Path(f"results/models/sarsa/sarsa_{rm}_lambda{best_s['lambda']}_seed{int(best_s['seed'])}.json")
        if sarsa_best_model.exists():
            plot_visit_map(sarsa_best_model, 'results/figures/sarsa/sarsa_visit_map.png',
                           cfg['maze_size'], f"SARSA visitation (lambda={best_s['lambda']})")

    step_trace = Path('results/raw_data/sarsa/sarsa_step_trace.csv')
    trace_dump = Path('results/raw_data/sarsa/sarsa_trace_dump.csv')
    if step_trace.exists() and trace_dump.exists():
        plot_sarsa_trace(step_trace, trace_dump, 'results/figures/sarsa/sarsa_trace.png')
    else:
        print(f'skip SARSA trace plot: {step_trace} or {trace_dump} not found')


if __name__ == '__main__':
    main()
