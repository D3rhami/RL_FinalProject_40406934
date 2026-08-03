import numpy as np
import pytest

from environments.maze import MazeEnv, load_config, CellType

MAP_PATH = 'environments/maps/source_maze.json'


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def sparse_env(cfg):
    return MazeEnv(MAP_PATH, config=cfg, reward_mode='sparse', seed=1)


@pytest.fixture
def shaped_env(cfg):
    return MazeEnv(MAP_PATH, config=cfg, reward_mode='shaped', seed=1)


# ── Phase 2: MDP correctness ─────────────────────────────────────────────────

def test_reset_state(sparse_env):
    s = sparse_env.reset()
    assert s == (sparse_env.start_pos[0], sparse_env.start_pos[1], 0, sparse_env.MAX_ENERGY)


def test_encode_decode_roundtrip(sparse_env):
    for st in [(1, 1, 0, 66), (16, 16, 1, 33), (6, 6, 1, 0), (0, 0, 0, 1)]:
        assert sparse_env.decode_state(sparse_env.encode_state(st)) == st


def test_transition_model_probabilities_sum_to_one(sparse_env):
    for st in sparse_env.all_states()[::3000]:
        if sparse_env.is_terminal(st):
            continue
        for a in range(4):
            outs = sparse_env.transition_model(st, a)
            assert abs(sum(p for p, _, _ in outs) - 1.0) < 1e-9


def test_step_matches_transition_model_support(sparse_env):
    rng = np.random.default_rng(0)
    for _ in range(300):
        s = sparse_env.reset(seed=int(rng.integers(0, 100000)))
        a = int(rng.integers(0, 4))
        possible = {ns for _, ns, _ in sparse_env.transition_model(s, a)}
        ns, _, _, _ = sparse_env.step(a)
        assert ns in possible


def test_wall_collision_stays_in_place(sparse_env):
    sparse_env.reset()
    sparse_env.state = (0, 0, 0, sparse_env.MAX_ENERGY)
    s, r, done, info = sparse_env.step(0)  # UP into top-left boundary
    assert (s[0], s[1]) == (0, 0)
    assert info['event'] == 'wall_hit'
    assert r == sparse_env.R_WALL


def test_locked_door_blocks_without_key(sparse_env):
    """Use transition_model to deterministically test the door-bounce outcome."""
    dr_pos, dc_pos = sparse_env.door_pos
    neighbor = action = None
    for (dr, dc, a) in ((-1, 0, 1), (1, 0, 0), (0, -1, 3), (0, 1, 2)):
        nr, nc = dr_pos + dr, dc_pos + dc
        if (0 <= nr < sparse_env.maze_size and 0 <= nc < sparse_env.maze_size
                and int(sparse_env.grid[nr, nc]) not in (CellType.WALL,)):
            neighbor, action = (nr, nc), a
            break
    assert neighbor is not None, "no passable neighbor of door found"
    state = (neighbor[0], neighbor[1], 0, sparse_env.MAX_ENERGY)
    prob, ns, reward = sparse_env.transition_model(state, action)[0]
    assert prob == pytest.approx(sparse_env.P_INTENDED)
    assert (ns[0], ns[1]) == neighbor     # bounced back
    assert reward == pytest.approx(sparse_env.R_DOOR_ATTEMPT)


def test_key_pickup_sets_flag_and_reward(sparse_env):
    """Use transition_model: intended move into the key cell."""
    kr, kc = sparse_env.key_pos
    state = (kr + 1, kc, 0, sparse_env.MAX_ENERGY)
    if int(sparse_env.grid[kr + 1, kc]) == CellType.WALL:
        pytest.skip("cell below key is a wall in this map")
    prob, ns, reward = sparse_env.transition_model(state, 0)[0]  # UP
    assert prob == pytest.approx(sparse_env.P_INTENDED)
    assert (ns[0], ns[1]) == (kr, kc)
    assert ns[2] == 1
    assert reward == pytest.approx(sparse_env.R_STEP + sparse_env.R_KEY)


def test_penalty_cell_reward(sparse_env):
    """Use transition_model: intended move into a penalty cell."""
    for pr, pc in sparse_env.penalty_cells:
        for dr, dc, a in ((-1, 0, 1), (1, 0, 0), (0, -1, 3), (0, 1, 2)):
            nr, nc = pr + dr, pc + dc
            if not (0 <= nr < sparse_env.maze_size and 0 <= nc < sparse_env.maze_size):
                continue
            if int(sparse_env.grid[nr, nc]) in (CellType.WALL, CellType.DOOR, CellType.PENALTY):
                continue
            state = (nr, nc, 0, sparse_env.MAX_ENERGY)
            prob, ns, reward = sparse_env.transition_model(state, a)[0]
            if (ns[0], ns[1]) == (pr, pc):
                assert reward == pytest.approx(sparse_env.R_STEP + sparse_env.R_PENALTY)
                return
    pytest.fail("No reachable neighbour of any penalty cell found")


def test_energy_depletes_and_terminates(sparse_env):
    sparse_env.reset()
    sparse_env.state = (sparse_env.start_pos[0], sparse_env.start_pos[1], 0, 1)
    s, r, done, info = sparse_env.step(0)
    assert done and s[3] == 0
    assert info['termination_reason'] == 'energy_depleted'


def test_step_cap_terminates(cfg):
    env = MazeEnv(MAP_PATH, config=cfg, reward_mode='sparse', seed=2)
    env.reset()
    env.max_steps = 3
    env.state = (env.start_pos[0], env.start_pos[1], 0, env.MAX_ENERGY)
    done = False
    for _ in range(3):
        _, _, done, info = env.step(2)
    assert done
    assert info['termination_reason'] == 'step_cap'


def test_goal_requires_key(sparse_env):
    gr, gc = sparse_env.goal_pos
    assert not sparse_env.is_terminal((gr, gc, 0, 50))
    assert sparse_env.is_terminal((gr, gc, 1, 50))


# ── Phase 3: reward correctness ───────────────────────────────────────────────

def test_shaped_minus_sparse_equals_shaping_formula(cfg):
    """For every sampled (state, action, outcome), shaped - sparse == _shaping_bonus."""
    sparse = MazeEnv(MAP_PATH, config=cfg, reward_mode='sparse', seed=0)
    shaped = MazeEnv(MAP_PATH, config=cfg, reward_mode='shaped', seed=0)
    for state in sparse.all_states()[::4000]:
        if sparse.is_terminal(state):
            continue
        for a in range(4):
            for (_, ns_sp, r_sp), (_, ns_sh, r_sh) in zip(
                sparse.transition_model(state, a),
                shaped.transition_model(state, a),
            ):
                assert ns_sp == ns_sh
                row, col, has_key, _ = state
                wasted = (ns_sp[0], ns_sp[1]) == (row, col)
                expected = shaped._shaping_bonus(row, col, ns_sp[0], ns_sp[1], has_key, wasted)
                assert abs((r_sh - r_sp) - expected) < 1e-9


def test_shaping_magnitude_bounded(shaped_env):
    """Shaping bonus should stay within a reasonable multiple of step cost."""
    max_abs = 0.0
    for state in shaped_env.all_states()[::3000]:
        if shaped_env.is_terminal(state):
            continue
        for a in range(4):
            for _, ns, _ in shaped_env.transition_model(state, a):
                row, col, has_key, _ = state
                wasted = (ns[0], ns[1]) == (row, col)
                b = shaped_env._shaping_bonus(row, col, ns[0], ns[1], has_key, wasted)
                max_abs = max(max_abs, abs(b))
    assert max_abs < abs(shaped_env.R_STEP) * 30


def test_shaped_wall_bump_never_positive(shaped_env):
    shaped_env.reset()
    shaped_env.state = (0, 5, 0, shaped_env.MAX_ENERGY)
    for _ in range(10):
        s, r, done, info = shaped_env.step(0)  # UP into boundary
        if info['event'] == 'wall_hit':
            assert r <= 0
        if done:
            break


def test_has_key_monotone(sparse_env):
    rng = np.random.default_rng(5)
    for _ in range(30):
        s = sparse_env.reset(seed=int(rng.integers(0, 100000)))
        prev_key = 0
        done = False
        while not done:
            a = int(sparse_env.rng.integers(0, 4))
            s, r, done, info = sparse_env.step(a)
            assert s[2] >= prev_key
            prev_key = s[2]


def test_termination_distribution_with_energy(cfg):
    """With default max_energy, random policy should often hit energy_depleted."""
    env = MazeEnv(MAP_PATH, config=cfg, reward_mode='sparse', seed=99)
    counts = {}
    for ep in range(100):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, done, info = env.step(int(env.rng.integers(0, 4)))
        reason = info.get('termination_reason', 'unknown')
        counts[reason] = counts.get(reason, 0) + 1
    assert counts.get('energy_depleted', 0) > 30, (
        f"Expected energy_depleted in >30/100 episodes, got: {counts}"
    )
    assert sum(counts.values()) == 100
