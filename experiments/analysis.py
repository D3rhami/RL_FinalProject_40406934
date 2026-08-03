import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.maze import load_config, CellType

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
    axes[1].plot(df['gamma'], df['runtime_seconds'], marker='o', color='orange')
    axes[1].set_xlabel('gamma'); axes[1].set_ylabel('runtime (s)')
    axes[2].plot(df['gamma'], df['eval_success_rate'], marker='o', color='green')
    axes[2].set_xlabel('gamma'); axes[2].set_ylabel('eval success rate')
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def _rolling(df, group_col, value_col, window=100):
    out = {}
    for key, grp in df.groupby(group_col):
        agg = grp.groupby('episode')[value_col].mean()
        out[key] = agg.rolling(window, min_periods=1).mean()
    return out


def plot_ql_schedule_comparison(training_csv, out_path, reward_mode='sparse'):
    df = pd.read_csv(training_csv)
    df = df[df['reward_mode'] == reward_mode]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for schedule, series in _rolling(df, 'schedule', 'return').items():
        axes[0].plot(series.index, series.values, label=schedule)
    for schedule, series in _rolling(df, 'schedule', 'steps').items():
        axes[1].plot(series.index, series.values, label=schedule)
    axes[0].set_title(f'Reward curve (reward_mode={reward_mode}), 100-ep rolling mean over seeds')
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
    for reward_mode, series in _rolling(df, 'reward_mode', 'return').items():
        axes[0].plot(series.index, series.values, label=reward_mode)
    for reward_mode, series in _rolling(df, 'reward_mode', 'steps').items():
        axes[1].plot(series.index, series.values, label=reward_mode)
    axes[0].set_title(f'Reward curve (schedule={schedule}), 100-ep rolling mean over seeds')
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
    for lam, series in _rolling(df, 'lambda', 'return').items():
        axes[0].plot(series.index, series.values, label=f'lambda={lam}')
    axes[0].set_title('Reward curve by lambda, 100-ep rolling mean over seeds')
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


def main():
    cfg = load_config()
    grid = cfg['experiment_grid']

    vi_gamma = grid['value_iteration']['reference_gamma']
    vi_model_path = Path(f'results/models/vi/vi_sparse_gamma{vi_gamma}.json')
    vi_csv_path = Path('results/raw_data/vi/vi_gamma_sweep.csv')
    if vi_model_path.exists():
        plot_vi_value_and_policy(vi_model_path, 'results/figures/vi/vi_value_policy.png', cfg=cfg)
    else:
        print(f'skip VI heatmap: {vi_model_path} not found')
    if vi_csv_path.exists():
        plot_vi_gamma_sensitivity(vi_csv_path, 'results/figures/vi/vi_gamma_sensitivity.png')
    else:
        print(f'skip VI gamma-sensitivity: {vi_csv_path} not found')

    ql_training = Path('results/raw_data/q_learning/q_learning_training.csv')
    if ql_training.exists():
        ref_schedule = grid['q_learning']['trace_run']['schedule']
        plot_ql_schedule_comparison(ql_training, 'results/figures/q_learning/ql_schedule_comparison.png',
                                     reward_mode=grid['q_learning']['reference_reward_mode'])
        plot_ql_rewardmode_comparison(ql_training, 'results/figures/q_learning/ql_rewardmode_comparison.png',
                                       schedule=ref_schedule)
    else:
        print(f'skip QL figures: {ql_training} not found')

    sarsa_training = Path('results/raw_data/sarsa/sarsa_training.csv')
    sarsa_summary = Path('results/raw_data/sarsa/sarsa_summary.csv')
    if sarsa_training.exists() and sarsa_summary.exists():
        plot_sarsa_lambda_comparison(sarsa_training, sarsa_summary,
                                      'results/figures/sarsa/sarsa_lambda_comparison.png')
    else:
        print(f'skip SARSA figure: missing {sarsa_training} or {sarsa_summary}')


if __name__ == '__main__':
    main()
