import numpy as np
import pytest

from agents.sarsa_lambda import SarsaLambdaAgent
from environments.maze import MazeEnv, load_config
from experiments.logger import EpisodeLogger


class FakeEnv:
    @property
    def n_states(self):
        return 3

    def encode_state(self, s):
        return s


def test_toy_chain_matches_independent_hand_derivation():
    """Independently re-derive delta/E/Q for a 3-step toy chain, compare against the agent."""
    alpha, gamma, lam = 0.5, 0.9, 0.5
    agent = SarsaLambdaAgent(FakeEnv(), alpha=alpha, gamma=gamma, lam=lam,
                              trace_type='accumulating', epsilon_start=1.0,
                              epsilon_min=1.0, decay_type='linear', decay_param=1,
                              num_episodes=1, trace_threshold=1e-9)

    Q = {(0, 0): 0.0, (1, 0): 0.0}
    E = {}

    def manual_step(s, a, r, s_next, a_next, done):
        delta = (r - Q[(s, a)]) if done else (r + gamma * Q[(s_next, a_next)] - Q[(s, a)])
        E[(s, a)] = E.get((s, a), 0.0) + 1.0
        for k in list(E.keys()):
            Q[k] += alpha * delta * E[k]
            E[k] *= gamma * lam
        return delta

    transitions = [
        (0, 0, -1.0, 1, 0, False),
        (1, 0, -1.0, 0, 0, False),
        (0, 0, 10.0, 2, 0, True),
    ]

    E_agent = {}
    for (s, a, r, s_next, a_next, done) in transitions:
        manual_step(s, a, r, s_next, a_next, done)
        agent.sarsa_lambda_update(E_agent, s, a, r, s_next, a_next, done)

    assert agent.Q[0, 0] == pytest.approx(Q[(0, 0)], abs=1e-9)
    assert agent.Q[1, 0] == pytest.approx(Q[(1, 0)], abs=1e-9)


def test_lambda_zero_updates_current_pair_then_prunes_all():
    """lambda=0 collapses to 1-step SARSA: current pair updates with E=1, then trace dies."""
    agent = SarsaLambdaAgent(FakeEnv(), alpha=0.5, gamma=0.9, lam=0.0,
                              trace_type='accumulating', epsilon_start=1.0,
                              epsilon_min=1.0, decay_type='linear', decay_param=1,
                              num_episodes=1, trace_threshold=1e-9)
    E = {}
    delta = agent.sarsa_lambda_update(E, s=0, a=0, r=-1.0, s_next=1, a_next=0, done=False)
    assert delta == pytest.approx(-1.0)
    assert agent.Q[0, 0] == pytest.approx(0.5 * (-1.0) * 1.0)
    assert E == {}, "gamma*lambda=0 must decay every entry (including current) to zero"


def test_lambda_zero_does_not_propagate_stale_traces():
    agent = SarsaLambdaAgent(FakeEnv(), alpha=0.5, gamma=0.9, lam=0.0,
                              trace_type='accumulating', epsilon_start=1.0,
                              epsilon_min=1.0, decay_type='linear', decay_param=1,
                              num_episodes=1, trace_threshold=1e-9)
    E = {(2, 3): 0.8}
    agent.sarsa_lambda_update(E, s=0, a=0, r=-1.0, s_next=1, a_next=0, done=False)
    assert (2, 3) not in E


def test_replacing_vs_accumulating_on_repeated_visit():
    """Revisit the same (s,a) twice, one step apart, without letting it decay to zero.
    Replacing must reset the trace to exactly 1.0 each visit; accumulating must exceed 1.0."""
    def build(trace_type):
        return SarsaLambdaAgent(FakeEnv(), alpha=0.1, gamma=0.9, lam=0.9,
                                 trace_type=trace_type, epsilon_start=1.0,
                                 epsilon_min=1.0, decay_type='linear', decay_param=1,
                                 num_episodes=1, trace_threshold=1e-9)

    for trace_type, expect_capped in (('replacing', True), ('accumulating', False)):
        agent = build(trace_type)
        E = {}
        agent.sarsa_lambda_update(E, s=0, a=0, r=-1.0, s_next=1, a_next=0, done=False)
        decayed = E[(0, 0)]                     # value right after first visit, post-decay
        agent.sarsa_lambda_update(E, s=0, a=0, r=-1.0, s_next=1, a_next=0, done=False)
        # value used for the second update, before its own decay:
        pre_decay_second = E[(0, 0)] / (agent.gamma * agent.lam)
        if expect_capped:
            assert pre_decay_second == pytest.approx(1.0)
        else:
            assert pre_decay_second == pytest.approx(decayed + 1.0)
            assert pre_decay_second > 1.0


def test_train_tiny_run_logs_all_fields(tmp_path):
    cfg = load_config()
    env = MazeEnv('environments/maps/source_maze.json', config=cfg, seed=0)
    agent = SarsaLambdaAgent(env, alpha=0.1, gamma=0.95, lam=0.3, trace_type='replacing',
                              epsilon_start=1.0, epsilon_min=0.05, decay_type='linear',
                              decay_param=5, num_episodes=5, seed=0)
    logger = EpisodeLogger(str(tmp_path / 'summary.csv'))
    agent.train(logger=logger)

    assert len(logger.summary_rows) == 5
    for row in logger.summary_rows:
        assert row['termination_reason'] is not None
        assert row['lambda'] == pytest.approx(0.3)
        assert row['steps'] > 0
