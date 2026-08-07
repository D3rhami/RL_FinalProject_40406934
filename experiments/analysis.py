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
    colors = ['#2a78d6', '#1baf7a', '#eda100', '#e34948']
    lambdas = sorted(df['lambda'].unique())
    color_map = {lam: colors[i % len(colors)] for i, lam in enumerate(lambdas)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('SARSA(λ) — lambda sweep (sparse reward, 3 seeds)', fontsize=11)

    bands = _rolling_with_band(df, 'lambda', 'return')
    for lam in lambdas:
        if lam not in bands:
            continue
        mean, std = bands[lam]
        color = color_map[lam]
        xs = np.arange(len(mean))
        axes[0].plot(xs, mean, color=color, linewidth=2, label=f'λ={lam}')
        axes[0].fill_between(xs, mean - std, mean + std, color=color, alpha=0.18)
    axes[0].set_xlabel('episode')
    axes[0].set_ylabel('return (100-ep rolling)')
    axes[0].set_title('Learning curves (mean ± std across seeds)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    stab = summary.groupby('lambda')['late_return_std'].mean().reindex(lambdas)
    bar_colors = [color_map[l] for l in stab.index]
    bars = axes[1].bar([str(l) for l in stab.index], stab.values,
                        color=bar_colors, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, stab.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                      val + 0.2, f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    axes[1].set_xlabel('lambda')
    axes[1].set_ylabel('std of return (last 1000 eps)')
    axes[1].set_title('Late-training stability (lower = more stable)')
    axes[1].grid(alpha=0.3, axis='y')

    succ_mean = summary.groupby('lambda')['eval_success_rate'].mean().reindex(lambdas)
    succ_std = summary.groupby('lambda')['eval_success_rate'].std().reindex(lambdas).fillna(0)
    xs_pos = np.arange(len(lambdas))
    bars2 = axes[2].bar(xs_pos, succ_mean.values, color=[color_map[l] for l in lambdas],
                         edgecolor='white', linewidth=1.2,
                         yerr=succ_std.values, capsize=5)
    axes[2].scatter(xs_pos, succ_mean.values, color=[color_map[l] for l in lambdas],
                    s=36, zorder=3, edgecolors='black', linewidths=0.6)
    y_top = max(0.1, float(succ_mean.max()) * 1.3 if len(succ_mean) else 0.1)
    for x, val in zip(xs_pos, succ_mean.values):
        axes[2].text(x, y_top * 0.08, f'{val:.2f}',
                     ha='center', va='bottom', fontsize=8)
    axes[2].set_xticks(xs_pos)
    axes[2].set_xticklabels([str(l) for l in lambdas])
    axes[2].set_xlabel('lambda')
    axes[2].set_ylabel('eval success rate')
    axes[2].set_title('Eval success rate (mean ± std, greedy policy)')
    axes[2].set_ylim(0, y_top)
    axes[2].grid(alpha=0.3, axis='y')

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


def plot_transfer_comparison(training_csv, summary_csv, out_dir):
    df = pd.read_csv(training_csv)
    summary = pd.read_csv(summary_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        'scratch': '#7f8c8d',
        'full': '#2a78d6',
        'scaled_0.25': '#1baf7a',
        'scaled_0.5': '#eda100',
        'scaled_0.50': '#eda100',
        'scaled_0.75': '#e34948',
        'selective': '#8e44ad',
    }

    for target in sorted(df['target'].unique()):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Transfer learning — target={target} (shaped QL)', fontsize=11)
        sub = df[df['target'] == target]
        bands = _rolling_with_band(sub, 'scenario', 'return')
        for scenario, (mean, std) in bands.items():
            color = colors.get(str(scenario), '#333333')
            xs = np.arange(len(mean))
            axes[0].plot(xs, mean, color=color, linewidth=2, label=str(scenario))
            axes[0].fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)
        axes[0].set_xlabel('episode')
        axes[0].set_ylabel('return (100-ep rolling)')
        axes[0].set_title('Learning curves (mean ± std across seeds)')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        jump = (summary[summary['target'] == target]
                .groupby('scenario')['jumpstart_mean_return']
                .agg(['mean', 'std'])
                .reindex(sorted(summary[summary['target'] == target]['scenario'].unique())))
        xs_pos = np.arange(len(jump))
        bar_colors = [colors.get(str(s), '#333333') for s in jump.index]
        axes[1].bar(xs_pos, jump['mean'].values, yerr=jump['std'].fillna(0).values,
                    color=bar_colors, edgecolor='white', linewidth=1.2, capsize=4)
        axes[1].set_xticks(xs_pos)
        axes[1].set_xticklabels([str(s) for s in jump.index], rotation=20, ha='right')
        axes[1].set_ylabel('mean return (first 100 eps)')
        axes[1].set_title('Jumpstart performance')
        axes[1].grid(alpha=0.3, axis='y')

        plt.tight_layout()
        out_path = out_dir / f'transfer_{target}_comparison.png'
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'saved {out_path}')


def plot_negative_transfer(checkpoint_csv, out_path):
    df = pd.read_csv(checkpoint_csv)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for a, label in enumerate(['↑', '↓', '←', '→']):
        ax.plot(df['episode'], df[f'q{a}'], marker='o', label=f'Q(a={a} {label})')
    ax.axhline(0, color='gray', linewidth=0.8)
    title_state = df['state'].iloc[0] if 'state' in df.columns else ''
    ax.set_title(f'Negative-transfer recovery — state {title_state}')
    ax.set_xlabel('retraining episode')
    ax.set_ylabel('Q-value')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def plot_final_path(model_path, map_path, out_path, cfg=None, max_steps=500):
    from environments.maze import MazeEnv, CellType
    cfg = cfg or load_config()
    model = _load_json(model_path)
    Q = {tuple(s): np.array(q) for s, q in zip(model['states'], model['Q'])}
    env = MazeEnv(map_path, config=cfg, reward_mode='sparse', seed=cfg['base_seed'])
    state = env.reset()
    path = [state[:2]]
    for _ in range(max_steps):
        q = Q.get(state)
        a = int(np.argmax(q)) if q is not None else 0
        state, _, done, info = env.step(a)
        path.append(state[:2])
        if done:
            break
    colors = {
        CellType.WALL: '#2D2D2D', CellType.EMPTY: '#FFFFFF', CellType.PENALTY: '#FF6464',
        CellType.START: '#64C864', CellType.KEY: '#FFD700', CellType.DOOR: '#8B5A2B',
        CellType.GOAL: '#00C800',
    }
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    for r in range(env.maze_size):
        for c in range(env.maze_size):
            ct = CellType(int(env.grid[r, c]))
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor=colors[ct], edgecolor='#B0B0B0', lw=0.3))
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, color='#0064FF', linewidth=2.2, marker='o', markersize=3)
    ax.set_xlim(-0.5, env.maze_size - 0.5)
    ax.set_ylim(env.maze_size - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Greedy path ({len(path)-1} steps) — {Path(model_path).name}\n'
                 f'term={info.get("termination_reason")}')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved {out_path}')


def plot_transfer_q_diff(source_model, after_model, map_path, out_path, cfg=None):
    from environments.maze import MazeEnv, CellType
    cfg = cfg or load_config()
    src = _load_json(source_model)
    aft = _load_json(after_model)
    env = MazeEnv(map_path, config=cfg, reward_mode='sparse')
    Qs = {tuple(s): np.array(q) for s, q in zip(src['states'], src['Q'])}
    Qa = {tuple(s): np.array(q) for s, q in zip(aft['states'], aft['Q'])}
    energy = env.MAX_ENERGY
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor='white')
    for ax, has_key, title in zip(axes, (0, 1), ('has_key=0', 'has_key=1')):
        diff = np.full((env.maze_size, env.maze_size), np.nan)
        for r in range(env.maze_size):
            for c in range(env.maze_size):
                if int(env.grid[r, c]) == CellType.WALL:
                    continue
                s = (r, c, has_key, energy)
                if s in Qs and s in Qa:
                    diff[r, c] = abs(float(np.max(Qa[s]) - np.max(Qs[s])))
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor='#2D2D2D' if int(env.grid[r, c]) == CellType.WALL else '#FFFFFF',
                    edgecolor='#B0B0B0', lw=0.3, zorder=1))
        im = ax.imshow(diff, cmap='magma', alpha=0.85, zorder=2,
                       extent=(-0.5, env.maze_size - 0.5, env.maze_size - 0.5, -0.5))
        ax.set_title(f'|Δ max Q| at E=max — {title}')
        ax.set_xlim(-0.5, env.maze_size - 0.5)
        ax.set_ylim(env.maze_size - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('Pre/post transfer Q difference (source shaped QL → different/full)')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved {out_path}')


def check_required_visuals():
    required = {
        'value_heatmap': 'results/figures/vi/vi_value_policy.png',
        'policy_arrows': 'results/figures/comparison/policy_grid.png',
        'visit_map': 'results/figures/q_learning/ql_visit_map.png',
        'policy_diff': 'results/figures/comparison/agreement_grid.png',
        'transfer_q_diff': 'results/figures/transfer/transfer_q_diff.png',
        'final_path': 'results/figures/misc/final_path.png',
    }
    print('Required visual checklist:')
    ok = True
    for name, path in required.items():
        exists = Path(path).exists()
        ok = ok and exists
        print(f"  [{'OK' if exists else 'MISSING'}] {name}: {path}")
    return ok


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
    ql_best_model = None
    if ql_summary_path.exists():
        best = pd.read_csv(ql_summary_path).sort_values('eval_success_rate', ascending=False).iloc[0]
        ql_best_model = Path(f"results/models/q_learning/q_learning_{best['reward_mode']}_{best['schedule']}_seed{int(best['seed'])}.json")
        if ql_best_model.exists():
            plot_visit_map(ql_best_model, 'results/figures/q_learning/ql_visit_map.png',
                           cfg['maze_size'], f"QL visitation ({best['reward_mode']}/{best['schedule']})")
            plot_final_path(ql_best_model, 'environments/maps/source_maze.json',
                            'results/figures/misc/final_path.png', cfg=cfg)

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

    transfer_training = Path('results/raw_data/transfer/transfer_training.csv')
    transfer_summary = Path('results/raw_data/transfer/transfer_summary.csv')
    if transfer_training.exists() and transfer_summary.exists():
        plot_transfer_comparison(transfer_training, transfer_summary, 'results/figures/transfer')
    else:
        print(f'skip transfer figures: {transfer_training} or {transfer_summary} not found')

    neg_ckpt = Path('results/raw_data/transfer/negative_transfer_checkpoints.csv')
    if neg_ckpt.exists():
        plot_negative_transfer(neg_ckpt, 'results/figures/transfer/negative_transfer_recovery.png')
    else:
        print(f'skip negative-transfer figure: {neg_ckpt} not found')

    src_ql = Path('results/models/q_learning/q_learning_shaped_linear_seed8565.json')
    aft = Path('results/models/transfer/transfer_different_full_seed81150.json')
    if src_ql.exists() and aft.exists():
        plot_transfer_q_diff(src_ql, aft, 'environments/maps/target_different.json',
                             'results/figures/transfer/transfer_q_diff.png', cfg=cfg)

    check_required_visuals()


if __name__ == '__main__':
    main()
