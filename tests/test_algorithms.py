import numpy as np
import pytest

from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import ValueIteration
from environments.maze import MazeEnv, load_config

MAP_PATH = 'environments/maps/source_maze.json'


@pytest.fixture(scope='module')
def cfg():
    return load_config()


@pytest.fixture
def env(cfg):
    return MazeEnv(MAP_PATH, config=cfg, reward_mode='shaped', seed=0)


def test_value_iteration_smoke_learns_on_real_env(env): 
    vi = ValueIteration(env, gamma=0.95, theta=1e-3)
    V, n_iter, converged, history = vi.run(max_iterations=2)

    assert n_iter > 0
    assert len(history) == n_iter
    assert np.any(V != 0.0), "V never moved away from all-zero init"

    policy = vi.extract_policy(V)
    assert len(policy) > 0
    assert set(policy.values()).issubset({0, 1, 2, 3})


def test_q_learning_smoke_learns_on_real_env(env):
    agent = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon_start=1.0,
                            epsilon_min=0.05, decay_type='linear', decay_param=20,
                            num_episodes=20, seed=0)
    Q_before = agent.Q.copy()
    agent.train()
    assert not np.array_equal(Q_before, agent.Q), "Q table never changed after training"
    assert np.isfinite(agent.Q).all()


def test_sarsa_lambda_smoke_learns_on_real_env(env):
    agent = SarsaLambdaAgent(env, alpha=0.1, gamma=0.95, lam=0.7, trace_type='replacing',
                              epsilon_start=1.0, epsilon_min=0.05, decay_type='linear',
                              decay_param=20, num_episodes=20, seed=0)
    Q_before = agent.Q.copy()
    agent.train()
    assert not np.array_equal(Q_before, agent.Q), "Q table never changed after training"
    assert np.isfinite(agent.Q).all()


def test_progress_bar_flag_does_not_break_training(env):
    """The tqdm progress_bar option must be a no-op on results, just cosmetic."""
    agent = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon_start=1.0,
                            epsilon_min=0.05, decay_type='linear', decay_param=5,
                            num_episodes=5, seed=0)
    agent.train(progress_bar=True, progress_desc='smoke-test')
    assert np.isfinite(agent.Q).all()


def test_all_three_algorithms_agree_reward_scale(env):
    """Cross-algorithm smoke check: none of the three blow up to absurd magnitudes
    on the same tiny budget, i.e. they are all wired to the same environment/reward scale."""
    vi = ValueIteration(env, gamma=0.95, theta=1e-3)
    V, *_ = vi.run(max_iterations=20)

    ql = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_min=0.05,
                         decay_type='linear', decay_param=10, num_episodes=10, seed=0)
    ql.train()

    sarsa = SarsaLambdaAgent(env, alpha=0.1, gamma=0.95, lam=0.5, trace_type='replacing',
                              epsilon_start=1.0, epsilon_min=0.05, decay_type='linear',
                              decay_param=10, num_episodes=10, seed=0)
    sarsa.train()

    for values in (V, ql.Q, sarsa.Q):
        assert np.isfinite(values).all()
        assert np.max(np.abs(values)) < 1e6
