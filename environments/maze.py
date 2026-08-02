from collections import deque
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple
import json
import numpy as np


class CellType(IntEnum):
    EMPTY   = 0
    WALL    = 1
    PENALTY = 2
    START   = 3
    KEY     = 4
    DOOR    = 5
    GOAL    = 6


class Actions(IntEnum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3


N_ACTIONS = 4

DELTAS = {
    Actions.UP:    (-1,  0),
    Actions.DOWN:  ( 1,  0),
    Actions.LEFT:  ( 0, -1),
    Actions.RIGHT: ( 0,  1),
}

# perpendicular slip directions per action
PERP = {
    Actions.UP:    (Actions.LEFT,  Actions.RIGHT),
    Actions.DOWN:  (Actions.LEFT,  Actions.RIGHT),
    Actions.LEFT:  (Actions.UP,    Actions.DOWN),
    Actions.RIGHT: (Actions.UP,    Actions.DOWN),
}

CELL_CHARS = {
    CellType.EMPTY:   '.',
    CellType.WALL:    '#',
    CellType.PENALTY: 'P',
    CellType.START:   'S',
    CellType.KEY:     'K',
    CellType.DOOR:    'D',
    CellType.GOAL:    'G',
}


def load_config(config_path: str = 'experiments/configs/default_config.json') -> dict:
    with open(config_path) as f:
        return json.load(f)


class MazeEnv:
    """
    State: (row, col, has_key, energy)
      - energy counts down by 1 each step; episode ends at 0
      - has_key: 0 until KEY cell is visited, then 1
    Transition: 0.8 intended, 0.1 each perpendicular direction.
    Wall/boundary collision: agent stays, pays wall_hit reward.
    """

    def __init__(self, map_path: str, config: Optional[dict] = None,
                 reward_mode: str = 'sparse', seed: Optional[int] = None):
        cfg = config or load_config()
        self.reward_mode = reward_mode
        self.env_seed    = seed

        r = cfg['rewards']
        self.R_STEP         = r['step']
        self.R_WALL         = r['wall_hit']
        self.R_PENALTY      = r['penalty_cell']
        self.R_KEY          = r['key']
        self.R_DOOR_ATTEMPT = r['door_attempt']
        self.R_GOAL         = r['goal']

        sh = cfg.get('shaping', {})
        self.SHAPE_GAMMA    = sh.get('gamma_shaping', 0.95)
        self.DIST_SCALE     = sh.get('distance_shaping_scale', 0.02)
        self.SAFE_DIST_CAP  = sh.get('safe_distance_cap', 3)
        self.SAFE_SCALE     = sh.get('safe_passage_scale', 0.05)
        self.WASTED_PENALTY = sh.get('wasted_move_penalty', -0.2)

        e = cfg['env']
        self.MAX_ENERGY  = e['max_energy']
        self.P_INTENDED  = e['p_intended']
        self.P_PERP      = e['p_perp']

        self._load_map(map_path)

        walkable       = int(np.sum(self.grid != CellType.WALL))
        self.max_steps = e['max_steps_multiplier'] * walkable

        self.rng   = np.random.default_rng(seed)
        self.state: Optional[Tuple] = None
        self.steps = 0

    def _load_map(self, map_path: str):
        with open(map_path) as f:
            data = json.load(f)
        self.grid          = np.array(data['grid'], dtype=np.int32)
        self.maze_size     = int(self.grid.shape[0])
        self.start_pos     = tuple(data['start_pos'])
        self.key_pos       = tuple(data['key_pos'])
        self.door_pos      = tuple(data['door_pos'])
        self.goal_pos      = tuple(data['goal_pos'])
        self.penalty_cells = [tuple(p) for p in data['penalty_cells']]
        self._compute_danger_distances()

    def _compute_danger_distances(self):
        size = self.maze_size
        dist = np.full((size, size), size * 2, dtype=np.int32)
        queue = deque()
        for (r, c) in self.penalty_cells:
            dist[r, c] = 0
            queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < size and 0 <= nc < size and
                        int(self.grid[nr, nc]) != CellType.WALL and
                        dist[nr, nc] > dist[r, c] + 1):
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
        self.danger_dist = dist

    def reset(self, seed: Optional[int] = None):
        self.rng   = np.random.default_rng(seed if seed is not None else self.env_seed)
        r, c       = self.start_pos
        self.state = (r, c, 0, self.MAX_ENERGY)
        self.steps = 0
        return self.state

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        row, col, has_key, energy = self.state
        info: Dict[str, Any] = {'event': 'move', 'termination_reason': None}

        rv = self.rng.random()
        if rv < self.P_INTENDED:
            direction = action
        elif rv < self.P_INTENDED + self.P_PERP:
            direction = PERP[action][0]
        else:
            direction = PERP[action][1]
        info['direction'] = int(direction)

        dr, dc       = DELTAS[direction]
        new_r, new_c = row + dr, col + dc

        out_of_bounds = (new_r < 0 or new_r >= self.maze_size or
                         new_c < 0 or new_c >= self.maze_size)
        is_wall = (not out_of_bounds and
                   int(self.grid[new_r, new_c]) == CellType.WALL)

        if out_of_bounds or is_wall:
            new_r, new_c  = row, col
            reward        = self.R_WALL
            info['event'] = 'wall_hit'

        elif int(self.grid[new_r, new_c]) == CellType.DOOR and has_key == 0:
            new_r, new_c  = row, col
            reward        = self.R_DOOR_ATTEMPT
            info['event'] = 'door_attempt'

        else:
            reward = self.R_STEP
            cell   = int(self.grid[new_r, new_c])

            if cell == CellType.PENALTY:
                reward       += self.R_PENALTY
                info['event'] = 'penalty_entry'
            elif cell == CellType.DOOR:
                info['event'] = 'door_open'
            elif cell == CellType.KEY and has_key == 0:
                has_key       = 1
                reward       += self.R_KEY
                info['event'] = 'key_pickup'

        if self.reward_mode == 'shaped':
            wasted  = (new_r, new_c) == (row, col)
            reward += self._shaping_bonus(row, col, new_r, new_c, has_key, wasted)

        new_energy = max(0, energy - 1)
        done = False

        if int(self.grid[new_r, new_c]) == CellType.GOAL and has_key == 1:
            reward                    += self.R_GOAL
            info['event']              = 'goal_reached'
            info['termination_reason'] = 'goal_reached'
            done                       = True

        self.steps += 1

        if not done and new_energy <= 0:
            info['termination_reason'] = 'energy_depleted'
            done                       = True

        if not done and self.steps >= self.max_steps:
            info['termination_reason'] = 'step_cap'
            done                       = True

        self.state = (new_r, new_c, has_key, new_energy)
        return self.state, reward, done, info

    def _safe_potential(self, r, c) -> float:
        d = int(self.danger_dist[r, c])
        if d == 0 or d > self.SAFE_DIST_CAP:
            return 0.0
        return float(self.SAFE_DIST_CAP + 1 - d)

    def _shaping_bonus(self, old_r, old_c, new_r, new_c, has_key, wasted) -> float:
        if wasted:
            return self.WASTED_PENALTY

        gamma  = self.SHAPE_GAMMA
        tr, tc = self.key_pos if has_key == 0 else self.goal_pos

        phi_old = -(abs(old_r - tr) + abs(old_c - tc))
        phi_new = -(abs(new_r - tr) + abs(new_c - tc))
        dist_bonus = self.DIST_SCALE * (gamma * phi_new - phi_old)

        safe_bonus = self.SAFE_SCALE * (
            gamma * self._safe_potential(new_r, new_c)
            - self._safe_potential(old_r, old_c)
        )

        return dist_bonus + safe_bonus

    def transition_model(self, state, action):
        if self.is_terminal(state):
            return []

        row, col, has_key, energy = state
        results = []

        for prob, direction in (
            (self.P_INTENDED, action),
            (self.P_PERP,     PERP[action][0]),
            (self.P_PERP,     PERP[action][1]),
        ):
            dr, dc       = DELTAS[direction]
            new_r, new_c = row + dr, col + dc

            out  = (new_r < 0 or new_r >= self.maze_size or
                    new_c < 0 or new_c >= self.maze_size)
            wall = not out and int(self.grid[new_r, new_c]) == CellType.WALL

            if out or wall:
                new_r, new_c = row, col
                new_has_key  = has_key
                new_energy   = max(0, energy - 1)
                reward       = self.R_WALL

            elif int(self.grid[new_r, new_c]) == CellType.DOOR and has_key == 0:
                new_r, new_c = row, col
                new_has_key  = has_key
                new_energy   = max(0, energy - 1)
                reward       = self.R_DOOR_ATTEMPT

            else:
                reward      = self.R_STEP
                new_has_key = has_key
                new_energy  = max(0, energy - 1)
                cell        = int(self.grid[new_r, new_c])

                if cell == CellType.PENALTY:
                    reward += self.R_PENALTY
                if cell == CellType.KEY and has_key == 0:
                    new_has_key  = 1
                    reward      += self.R_KEY
                if cell == CellType.GOAL and new_has_key == 1:
                    reward += self.R_GOAL

            if self.reward_mode == 'shaped':
                wasted  = (new_r, new_c) == (row, col)
                reward += self._shaping_bonus(row, col, new_r, new_c, has_key, wasted)

            results.append((prob, (new_r, new_c, new_has_key, new_energy), reward))

        return results

    def is_terminal(self, state):
        row, col, has_key, energy = state
        if energy <= 0:
            return True
        if int(self.grid[row, col]) == CellType.GOAL and has_key == 1:
            return True
        return False

    @property
    def n_energy_levels(self):
        return self.MAX_ENERGY + 1

    @property
    def n_states(self):
        return self.maze_size * self.maze_size * 2 * self.n_energy_levels

    def encode_state(self, state):
        row, col, has_key, energy = state
        E = self.n_energy_levels
        return ((row * self.maze_size + col) * 2 + has_key) * E + energy

    def decode_state(self, idx):
        E       = self.n_energy_levels
        energy  = idx % E;  idx //= E
        has_key = idx % 2;  idx //= 2
        col     = idx % self.maze_size
        row     = idx // self.maze_size
        return (row, col, has_key, energy)

    def all_states(self):
        states = []
        for row in range(self.maze_size):
            for col in range(self.maze_size):
                if int(self.grid[row, col]) != CellType.WALL:
                    for has_key in (0, 1):
                        for energy in range(self.n_energy_levels):
                            states.append((row, col, has_key, energy))
        return states

    def render_ascii(self):
        lines = []
        for r in range(self.maze_size):
            row_chars = []
            for c in range(self.maze_size):
                if self.state and (r, c) == (self.state[0], self.state[1]):
                    row_chars.append('A')
                else:
                    row_chars.append(CELL_CHARS.get(CellType(self.grid[r, c]), '?'))
            lines.append(' '.join(row_chars))
        return '\n'.join(lines)
