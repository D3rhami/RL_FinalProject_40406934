import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


class EpisodeLogger:
    def __init__(self, summary_path: str, detail_dir: Optional[str] = None):
        self.summary_path = Path(summary_path)
        self.detail_dir = Path(detail_dir) if detail_dir else None
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        if self.detail_dir:
            self.detail_dir.mkdir(parents=True, exist_ok=True)

        self._summary_rows = []
        self._episode_idx = -1
        self._reset()

    def _reset(self):
        self._steps = []
        self._total_reward = 0.0
        self._step_count = 0
        self._wall_hits = 0
        self._penalty_entries = 0
        self._key_pickup_step = None
        self._goal_reached_step = None
        self._termination_reason = None
        self._keep_detail = False
        self._epsilon = None
        self._lam = None

    def start_episode(self, episode_idx: int, keep_detail: bool = False,
                       epsilon: Optional[float] = None, lam: Optional[float] = None):
        self._episode_idx = episode_idx
        self._reset()
        self._keep_detail = keep_detail
        self._epsilon = epsilon
        self._lam = lam

    def log_step(self, state, action, reward, info: Dict[str, Any], done: bool = False):
        self._step_count += 1
        self._total_reward += reward
        event = info.get('event', 'move')

        if event == 'wall_hit':
            self._wall_hits += 1
        elif event == 'penalty_entry':
            self._penalty_entries += 1
        elif event == 'key_pickup' and self._key_pickup_step is None:
            self._key_pickup_step = self._step_count
        elif event == 'goal_reached' and self._goal_reached_step is None:
            self._goal_reached_step = self._step_count

        term = info.get('termination_reason')
        if term:
            self._termination_reason = term

        if self._keep_detail:
            self._steps.append({
                'step': self._step_count,
                'state': state,
                'action': action,
                'reward': reward,
                'event': event,
                'direction': info.get('direction'),
                'done': done,
            })

    def end_episode(self) -> Dict[str, Any]:
        row = {
            'episode_idx': self._episode_idx,
            'total_reward': self._total_reward,
            'steps': self._step_count,
            'success': self._termination_reason == 'goal_reached',
            'wall_hits': self._wall_hits,
            'penalty_entries': self._penalty_entries,
            'key_pickup_step': self._key_pickup_step,
            'goal_reached_step': self._goal_reached_step,
            'termination_reason': self._termination_reason,
            'epsilon_final': self._epsilon,
            'lambda': self._lam,
        }
        self._summary_rows.append(row)

        if self._keep_detail and self.detail_dir:
            path = self.detail_dir / f'episode_{self._episode_idx}.json'
            with open(path, 'w') as f:
                json.dump(self._steps, f, indent=2, default=str)

        return row

    def flush_summary(self):
        if not self._summary_rows:
            return
        with open(self.summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self._summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._summary_rows)

    @property
    def summary_rows(self):
        return self._summary_rows
