import numpy as np
import pytest

from environments.maze import MazeEnv, load_config
from transfer.transfer_learning import build_initial_q, unchanged_state_indices

SOURCE_MAP = 'environments/maps/source_maze.json'
SIMILAR_MAP = 'environments/maps/target_similar.json'
DIFFERENT_MAP = 'environments/maps/target_different.json'


@pytest.fixture(scope='module')
def cfg():
    return load_config()


@pytest.fixture
def source_Q():
    rng = np.random.default_rng(0)
    return rng.normal(size=(6, 4))


def test_build_initial_q_scratch_is_zero(source_Q):
    Q = build_initial_q(source_Q, 'scratch', n_states_target=6)
    assert Q.shape == (6, 4)
    assert np.all(Q == 0.0)


def test_build_initial_q_full_copies_source(source_Q):
    Q = build_initial_q(source_Q, 'full', n_states_target=6)
    assert np.array_equal(Q, source_Q)
    Q[0, 0] = 999.0
    assert source_Q[0, 0] != 999.0, "full transfer must copy, not alias, source_Q"


def test_build_initial_q_scaled_applies_beta(source_Q):
    Q = build_initial_q(source_Q, 'scaled', n_states_target=6, beta=0.5)
    assert np.allclose(Q, 0.5 * source_Q)


def test_build_initial_q_selective_only_fills_given_indices(source_Q):
    idx = np.array([1, 3])
    Q = build_initial_q(source_Q, 'selective', n_states_target=6,
                         unchanged_state_indices=idx)
    assert np.allclose(Q[idx], source_Q[idx])
    other = np.setdiff1d(np.arange(6), idx)
    assert np.all(Q[other] == 0.0)


def test_build_initial_q_unknown_scenario_raises(source_Q):
    with pytest.raises(ValueError):
        build_initial_q(source_Q, 'bogus', n_states_target=6)


@pytest.mark.parametrize('target_map', [SIMILAR_MAP, DIFFERENT_MAP])
def test_unchanged_state_indices_are_valid_and_sane(cfg, target_map):
    source_env = MazeEnv(SOURCE_MAP, config=cfg, seed=0)
    target_env = MazeEnv(target_map, config=cfg, seed=0)
    idx = unchanged_state_indices(source_env, target_env)

    assert idx.dtype.kind in ('i', 'u')
    assert idx.size > 0
    assert idx.min() >= 0
    assert idx.max() < source_env.n_states


def test_unchanged_state_indices_similar_has_more_overlap_than_different(cfg):
    """The 'similar' target map should share more unchanged neighbourhoods
    with the source than the 'different' one (sanity check on map design)."""
    source_env = MazeEnv(SOURCE_MAP, config=cfg, seed=0)
    similar_env = MazeEnv(SIMILAR_MAP, config=cfg, seed=0)
    different_env = MazeEnv(DIFFERENT_MAP, config=cfg, seed=0)

    idx_similar = unchanged_state_indices(source_env, similar_env)
    idx_different = unchanged_state_indices(source_env, different_env)
    assert idx_similar.size >= idx_different.size
