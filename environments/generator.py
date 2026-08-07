import json
import sys
from collections import deque
from pathlib import Path
from typing import FrozenSet, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environments.maze import CellType, CELL_CHARS, load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'experiments' / 'configs' / 'default_config.json'
MAP_FILE    = Path(__file__).parent / 'maps' / 'source_maze.json'

_PRE_KEY_PASS: FrozenSet = frozenset({
    CellType.EMPTY, CellType.PENALTY, CellType.START,
    CellType.KEY, CellType.GOAL,
})
_POST_KEY_PASS: FrozenSet = frozenset({
    CellType.EMPTY, CellType.PENALTY, CellType.START,
    CellType.KEY, CellType.DOOR, CellType.GOAL,
})


def _fixed_positions(size):
    return (
        (1, 1),
        (size // 3,     size // 3),
        (2 * size // 3, 2 * size // 3),
        (size - 2,      size - 2),
    )


def generate_maze(seed: int, cfg: dict) -> dict:
    rng  = np.random.default_rng(seed)
    size = cfg['maze_size']
    env  = cfg['env']

    start_pos, key_pos, door_pos, goal_pos = _fixed_positions(size)
    reserved = {start_pos, key_pos, door_pos, goal_pos}

    grid    = np.zeros((size, size), dtype=np.int32)
    n_walls = int(env['wall_fraction'] * size * size)
    all_free = [(r, c) for r in range(size) for c in range(size)
                if (r, c) not in reserved]
    for i in rng.choice(len(all_free), size=n_walls, replace=False):
        grid[all_free[i]] = CellType.WALL

    candidates = [(r, c) for r in range(size) for c in range(size)
                  if grid[r, c] == CellType.EMPTY and (r, c) not in reserved]
    penalty_cells: List = []
    n_pen = min(env['n_penalty_cells'], len(candidates))
    for i in rng.choice(len(candidates), size=n_pen, replace=False):
        pos = candidates[i]
        grid[pos] = CellType.PENALTY
        penalty_cells.append(pos)

    grid[start_pos] = CellType.START
    grid[key_pos]   = CellType.KEY
    grid[door_pos]  = CellType.DOOR
    grid[goal_pos]  = CellType.GOAL

    return {
        'student_id': cfg['student_id'], 'base_seed': cfg['base_seed'],
        'actual_seed': seed, 'maze_size': size,
        'grid': grid,
        'start_pos': start_pos, 'key_pos': key_pos,
        'door_pos': door_pos, 'goal_pos': goal_pos,
        'penalty_cells': penalty_cells,
    }


def bfs_reachable(grid, start, target, passable: FrozenSet) -> bool:
    size = grid.shape[0]
    visited, queue = {start}, deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == target:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < size and 0 <= nc < size
                    and (nr, nc) not in visited
                    and CellType(grid[nr, nc]) in passable):
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def _bfs_distance(grid, start, target, passable: FrozenSet) -> int:
    size = grid.shape[0]
    visited = {start: 0}
    queue   = deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == target:
            return dist
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < size and 0 <= nc < size
                    and (nr, nc) not in visited
                    and CellType(grid[nr, nc]) in passable):
                visited[(nr, nc)] = dist + 1
                queue.append(((nr, nc), dist + 1))
    return -1


def validate_map(data: dict, max_energy: int) -> bool:
    g = data['grid']
    ok1 = bfs_reachable(g, data['start_pos'], data['key_pos'],  _PRE_KEY_PASS)
    ok2 = bfs_reachable(g, data['key_pos'],   data['goal_pos'], _POST_KEY_PASS)
    if not (ok1 and ok2):
        return False
    d1 = _bfs_distance(g, data['start_pos'], data['key_pos'],  _PRE_KEY_PASS)
    d2 = _bfs_distance(g, data['key_pos'],   data['goal_pos'], _POST_KEY_PASS)
    return (d1 + d2) <= max_energy


def generate_valid_maze(max_attempts: int = 500) -> dict:
    cfg        = load_config(str(CONFIG_PATH))
    max_energy = cfg['env']['max_energy']
    base_seed  = cfg['base_seed']

    for attempt in range(max_attempts):
        seed = base_seed + attempt
        data = generate_maze(seed, cfg)
        if validate_map(data, max_energy):
            print(f"Valid maze: seed={seed}, attempt={attempt + 1}")
            return data
    raise RuntimeError(f"No valid maze in {max_attempts} attempts.")


def _to_json(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, list) and v and isinstance(v[0], tuple):
            out[k] = [list(p) for p in v]
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def save_map(data: dict, path: Optional[Path] = None):
    path = path or MAP_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_to_json(data), f, indent=2)
    print(f"Map saved -> {path}")


def load_map(path: Optional[Path] = None) -> dict:
    path = path or MAP_FILE
    with open(path) as f:
        data = json.load(f)
    data['grid'] = np.array(data['grid'], dtype=np.int32)
    for k in ('start_pos', 'key_pos', 'door_pos', 'goal_pos'):
        data[k] = tuple(data[k])
    data['penalty_cells'] = [tuple(p) for p in data['penalty_cells']]
    return data


def count_walkable(data: dict) -> int:
    return int(np.sum(np.array(data['grid']) != CellType.WALL))


def print_maze(data: dict):
    grid = data['grid']
    size = grid.shape[0]
    print('=' * (2 * size - 1))
    for r in range(size):
        print(' '.join(CELL_CHARS.get(CellType(grid[r, c]), '?') for c in range(size)))
    print('=' * (2 * size - 1))
    n_walls    = int(np.sum(grid == CellType.WALL))
    n_walkable = int(np.sum(grid != CellType.WALL))
    print(f"start={data['start_pos']} key={data['key_pos']} "
          f"door={data['door_pos']} goal={data['goal_pos']}")
    print(f"walls={n_walls} ({100 * n_walls / size**2:.1f}%) "
          f"walkable={n_walkable} max_steps={3 * n_walkable}")
    print(f"penalty={data['penalty_cells']}")


def create_target_map(source_data, obstacle_change_pct, relocate_key_goal,
                       n_extra_penalty, max_energy, rng, max_attempts=200):
    size = source_data['maze_size']
    start_pos = tuple(source_data['start_pos'])

    for attempt in range(max_attempts):
        grid = np.array(source_data['grid'], dtype=np.int32).copy()
        key_pos = tuple(source_data['key_pos'])
        door_pos = tuple(source_data['door_pos'])
        goal_pos = tuple(source_data['goal_pos'])
        penalty_cells = [tuple(p) for p in source_data['penalty_cells']]

        walls = [(r, c) for r in range(size) for c in range(size)
                 if grid[r, c] == CellType.WALL]
        empties = [(r, c) for r in range(size) for c in range(size)
                   if grid[r, c] == CellType.EMPTY]
        n_move = max(1, int(len(walls) * obstacle_change_pct))
        rng.shuffle(walls)
        rng.shuffle(empties)
        for w, e in zip(walls[:n_move], empties[:n_move]):
            grid[w], grid[e] = CellType.EMPTY, CellType.WALL

        if relocate_key_goal:
            for pos in (key_pos, door_pos, goal_pos):
                grid[pos] = CellType.EMPTY
            free = [(r, c) for r in range(size) for c in range(size)
                    if grid[r, c] == CellType.EMPTY and (r, c) != start_pos]
            if len(free) < 3 + n_extra_penalty:
                continue
            rng.shuffle(free)
            key_pos, door_pos, goal_pos = free[0], free[1], free[2]
            grid[key_pos] = CellType.KEY
            grid[door_pos] = CellType.DOOR
            grid[goal_pos] = CellType.GOAL
            for pos in free[3:3 + n_extra_penalty]:
                grid[pos] = CellType.PENALTY
                penalty_cells.append(pos)

        candidate = {
            **{k: v for k, v in source_data.items()
               if k not in ('grid', 'key_pos', 'door_pos', 'goal_pos',
                            'penalty_cells', 'actual_seed')},
            'grid': grid,
            'start_pos': start_pos,
            'key_pos': key_pos,
            'door_pos': door_pos,
            'goal_pos': goal_pos,
            'penalty_cells': penalty_cells,
            'obstacle_change_pct': obstacle_change_pct,
            'relocate_key_goal': relocate_key_goal,
            'n_extra_penalty': n_extra_penalty if relocate_key_goal else 0,
            'generation_attempt': attempt,
        }
        if validate_map(candidate, max_energy):
            return candidate
    raise RuntimeError(
        f'no valid target map found (pct={obstacle_change_pct}, '
        f'relocate={relocate_key_goal}) after {max_attempts} attempts')


def generate_target_maps(cfg=None):
    cfg = cfg or load_config(str(CONFIG_PATH))
    source = load_map()
    max_energy = cfg['env']['max_energy']
    tcfg = cfg['transfer']
    maps_dir = Path(__file__).parent / 'maps'

    similar_rng = np.random.default_rng(cfg['base_seed'] + 1000)
    similar = create_target_map(
        source, tcfg['similar_obstacle_change_pct'], False, 0,
        max_energy, similar_rng)
    similar['map_kind'] = 'similar'
    similar_path = maps_dir / 'target_similar.json'
    save_map(similar, similar_path)

    different_rng = np.random.default_rng(cfg['base_seed'] + 2000)
    different = create_target_map(
        source, tcfg['different_obstacle_change_pct'], True, 3,
        max_energy, different_rng)
    different['map_kind'] = 'different'
    different_path = maps_dir / 'target_different.json'
    save_map(different, different_path)

    print('Similar map validated:')
    print_maze(similar)
    print('Different map validated:')
    print_maze(different)
    return similar_path, different_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', action='store_true',
                        help='Generate similar/different transfer target maps')
    args = parser.parse_args()
    if args.targets:
        generate_target_maps()
    else:
        data = generate_valid_maze()
        print_maze(data)
        save_map(data)
