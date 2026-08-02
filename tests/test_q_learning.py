import numpy as np
import pytest

from agents.q_learning import QLearningAgent
from environments.maze import MazeEnv, load_config
from experiments.logger import EpisodeLogger


class FakeEnv:
    @property
    def n_states(self):
        return 5

    def encode_state(self, s):
        return s


def test_q_update_matches_formula():
    agent = QLearningAgent(FakeEnv(), alpha=0.5, gamma=0.9, epsilon_start=1.0,
                            epsilon_min=0.01, decay_type='linear', decay_param=100,
                            num_episodes=1, seed=0)
    agent.Q[0, 1] = 2.0
    agent.Q[2] = np.array([1.0, 5.0, 3.0, 0.0])

    agent.update(s=0, a=1, r=3.0, s_next=2, done=False)
    expected = 2.0 + 0.5 * (3.0 + 0.9 * 5.0 - 2.0)
    assert agent.Q[0, 1] == pytest.approx(expected)


def test_q_update_terminal_ignores_bootstrap():
    agent = QLearningAgent(FakeEnv(), alpha=0.5, gamma=0.9, epsilon_start=1.0,
                            epsilon_min=0.01, decay_type='linear', decay_param=100,
                            num_episodes=1, seed=0)
    agent.Q[0, 1] = 2.0
    agent.Q[2] = np.array([1.0, 5.0, 3.0, 0.0])
    agent.update(s=0, a=1, r=3.0, s_next=2, done=True)
    expected = 2.0 + 0.5 * (3.0 - 2.0)
    assert agent.Q[0, 1] == pytest.approx(expected)


def test_linear_epsilon_decay():
    agent = QLearningAgent(FakeEnv(), alpha=0.1, gamma=0.95, epsilon_start=1.0,
                            epsilon_min=0.05, decay_type='linear', decay_param=1000,
                            num_episodes=1)
    assert agent.epsilon_at(0) == pytest.approx(1.0)
    assert agent.epsilon_at(1000) == pytest.approx(0.05)
    assert agent.epsilon_at(500) == pytest.approx(1.0 - (1.0 - 0.05) * 0.5)
    prev = agent.epsilon_at(0)
    for ep in (100, 300, 600, 900, 1000):
        cur = agent.epsilon_at(ep)
        assert cur <= prev
        prev = cur


def test_exponential_epsilon_decay():
    agent = QLearningAgent(FakeEnv(), alpha=0.1, gamma=0.95, epsilon_start=1.0,
                            epsilon_min=0.01, decay_type='exponential', decay_param=0.99,
                            num_episodes=1)
    assert agent.epsilon_at(0) == pytest.approx(1.0)
    assert agent.epsilon_at(10) == pytest.approx(max(0.01, 1.0 * 0.99 ** 10))
    assert agent.epsilon_at(2000) == pytest.approx(0.01, abs=1e-9)


def test_train_tiny_run_logs_all_fields(tmp_path):
    cfg = load_config()
    env = MazeEnv('environments/maps/source_maze.json', config=cfg, seed=0)
    agent = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon_start=1.0,
                            epsilon_min=0.05, decay_type='linear', decay_param=5,
                            num_episodes=5, seed=0)
    logger = EpisodeLogger(str(tmp_path / 'summary.csv'))
    agent.train(logger=logger)

    assert len(logger.summary_rows) == 5
    required_fields = ('episode_idx', 'total_reward', 'steps', 'success',
                        'wall_hits', 'penalty_entries', 'termination_reason', 'epsilon_final')
    for row in logger.summary_rows:
        for field in required_fields:
            assert field in row
        assert row['termination_reason'] is not None
        assert row['steps'] > 0


def test_initial_q_table_is_used_not_overwritten():
    preset = np.full((5, 4), 7.0)
    agent = QLearningAgent(FakeEnv(), alpha=0.1, gamma=0.9, epsilon_start=1.0,
                            epsilon_min=0.01, decay_type='linear', decay_param=10,
                            num_episodes=1, initial_Q=preset)
    assert np.all(agent.Q == 7.0)
    preset[0, 0] = 999.0
    assert agent.Q[0, 0] == 7.0
