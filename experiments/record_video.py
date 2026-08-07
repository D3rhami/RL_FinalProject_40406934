
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.maze import MazeEnv, load_config
from agents.value_iteration import ValueIteration
from experiments.compare import _draw_maze_cells, _overlay_cell_labels

MAP_PATH = 'environments/maps/source_maze.json'
DEFAULT_OUT = Path('results/videos/vi_single_episode.gif')
AGENT_COLOR = '#FF3B30'
GOAL_COLOR = '#00C800'


def _fig_to_image(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(buf[:, :, :3])


def _load_or_run_vi(cfg, env, gamma, reward_mode):
    model_path = Path(f'results/models/vi/vi_{reward_mode}_gamma{gamma}.json')
    if model_path.exists():
        print(f'[record_video] loading saved VI policy from {model_path}')
        with open(model_path) as f:
            model = json.load(f)
        policy = {
            tuple(s): a for s, a in zip(model['states'], model['policy']) if a != -1
        }
        return policy, model.get('converged'), model.get('iterations')

    print(f'[record_video] no saved VI model at {model_path}, running VI fresh...')
    vi = ValueIteration(env, gamma=gamma, theta=cfg['value_iteration']['theta'],
                         reward_mode=reward_mode)
    V, n_iter, converged, _ = vi.run(max_iterations=2000, progress_bar=True,
                                      progress_desc='VI (for video)')
    return vi.extract_policy(V), converged, n_iter


def record_vi_episode(cfg=None, gamma=None, reward_mode=None,
                       out_path=None, fps=4, max_frames=300, seed=None):
    cfg = cfg or load_config()
    grid = cfg['experiment_grid']['value_iteration']
    gamma = gamma or grid['reference_gamma']
    reward_mode = reward_mode or grid['reward_mode']
    out_path = Path(out_path) if out_path else DEFAULT_OUT

    env = MazeEnv(MAP_PATH, config=cfg, reward_mode=reward_mode)
    policy, converged, n_iter = _load_or_run_vi(cfg, env, gamma, reward_mode)
    print(f'[record_video] policy ready (converged={converged}, iterations={n_iter})')

    state = env.reset(seed=seed)
    frames = []
    done, info, step = False, {}, 0

    print('[record_video] rendering rollout frames...')
    while not done and step < max_frames:
        fig, ax = plt.subplots(figsize=(6, 6))
        _draw_maze_cells(ax, env)
        _overlay_cell_labels(ax, env)
        r, c, has_key, energy = state
        ax.add_patch(plt.Circle((c, r), 0.32, color=AGENT_COLOR, zorder=6))
        ax.set_title(
            f'VI greedy rollout (gamma={gamma}) - step {step}, '
            f'key={"yes" if has_key else "no"}, energy={energy}',
            fontsize=10)
        frames.append(_fig_to_image(fig))
        plt.close(fig)

        a = policy.get(state, 0)
        state, reward, done, info = env.step(a)
        step += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_maze_cells(ax, env)
    _overlay_cell_labels(ax, env)
    r, c, has_key, energy = state
    reason = info.get('termination_reason', 'max_frames')
    final_color = GOAL_COLOR if reason == 'goal_reached' else AGENT_COLOR
    ax.add_patch(plt.Circle((c, r), 0.32, color=final_color, zorder=6))
    ax.set_title(f'VI greedy rollout - done: {reason} ({step} steps)', fontsize=10)
    frames.append(_fig_to_image(fig))
    plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    print(f'[record_video] saved {out_path} ({len(frames)} frames, '
          f'{duration_ms}ms/frame, termination={reason})')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Record one VI rollout as a GIF')
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--reward-mode', dest='reward_mode', default=None,
                         choices=['sparse', 'shaped'])
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--max-frames', dest='max_frames', type=int, default=300)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    record_vi_episode(gamma=args.gamma, reward_mode=args.reward_mode,
                       out_path=args.out, fps=args.fps,
                       max_frames=args.max_frames, seed=args.seed)


if __name__ == '__main__':
    main()
