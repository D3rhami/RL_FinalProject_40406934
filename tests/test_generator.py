import numpy as np
import pytest

from environments.generator import (
    bfs_reachable, generate_maze, load_map, validate_map,
)
from environments.maze import CellType, load_config

SOURCE_MAP = 'environments/maps/source_maze.json'
SIMILAR_MAP = 'environments/maps/target_similar.json'
DIFFERENT_MAP = 'environments/maps/target_different.json'


@pytest.fixture(scope='module')
def cfg():
    return load_config()


def test_generate_maze_deterministic_for_same_seed(cfg):
    a = generate_maze(123, cfg)
    b = generate_maze(123, cfg)
    assert np.array_equal(a['grid'], b['grid'])
    assert a['start_pos'] == b['start_pos']
    assert a['key_pos'] == b['key_pos']
    assert a['penalty_cells'] == b['penalty_cells']


def test_generate_maze_different_seeds_usually_differ(cfg):
    a = generate_maze(1, cfg)
    b = generate_maze(2, cfg)
    assert not np.array_equal(a['grid'], b['grid'])


def test_bfs_reachable_trivial_self_path():
    grid = np.zeros((3, 3), dtype=np.int32)
    assert bfs_reachable(grid, (0, 0), (0, 0), {CellType.EMPTY, CellType.START})


def test_bfs_reachable_blocked_by_wall():
    grid = np.zeros((3, 3), dtype=np.int32)
    grid[:, 1] = CellType.WALL
    assert not bfs_reachable(grid, (0, 0), (0, 2), {CellType.EMPTY, CellType.START})


@pytest.mark.parametrize('map_path', [SOURCE_MAP, SIMILAR_MAP, DIFFERENT_MAP])
def test_committed_maps_are_valid(map_path, cfg):
    """Every map shipped in the repo must already satisfy validate_map,
    otherwise VI/QL/SARSA can never reach the goal."""
    data = load_map(map_path)
    assert validate_map(data, cfg['env']['max_energy'])


@pytest.mark.parametrize('map_path', [SOURCE_MAP, SIMILAR_MAP, DIFFERENT_MAP])
def test_committed_maps_have_required_cells(map_path):
    data = load_map(map_path)
    grid = data['grid']
    assert int(grid[data['start_pos']]) == CellType.START
    assert int(grid[data['key_pos']]) == CellType.KEY
    assert int(grid[data['door_pos']]) == CellType.DOOR
    assert int(grid[data['goal_pos']]) == CellType.GOAL
