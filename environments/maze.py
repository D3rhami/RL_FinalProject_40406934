from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
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

    def __init__(self, map_path: str, config: dict = None,
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
        info: Dict[str, Any] = {'event': 'move'}

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
            reward += self._shaping_bonus(row, col, new_r, new_c, has_key)

        new_energy = max(0, energy - 1)
        done = False

        if int(self.grid[new_r, new_c]) == CellType.GOAL and has_key == 1:
            reward       += self.R_GOAL
            info['event'] = 'goal_reached'
            done          = True

        self.steps += 1

        if not done and new_energy <= 0:
            info['event'] = 'energy_depleted'
            done          = True

        if not done and self.steps >= self.max_steps:
            info['event'] = 'step_cap'
            done          = True

        self.state = (new_r, new_c, has_key, new_energy)
        return self.state, reward, done, info

    def _shaping_bonus(self, old_r, old_c, new_r, new_c, has_key):
        gamma  = 0.95
        tr, tc = self.key_pos if has_key == 0 else self.goal_pos
        return gamma * -(abs(new_r - tr) + abs(new_c - tc)) \
                     + (abs(old_r - tr) + abs(old_c - tc))

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
                reward += self._shaping_bonus(row, col, new_r, new_c, has_key)

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
