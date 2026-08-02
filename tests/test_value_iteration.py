import numpy as np
import pytest

from agents.value_iteration import ValueIteration


class ToyChainEnv:
    """A -> B -> G(terminal). FORWARD=0 is the only productive action."""

    A, B, G = 0, 1, 2
    FORWARD = 0

    @property
    def n_states(self):
        return 3

    def all_states(self):
        return [self.A, self.B, self.G]

    def is_terminal(self, s):
        return s == self.G

    def encode_state(self, s):
        return s

    def transition_model(self, s, a):
        if self.is_terminal(s):
            return []
        if s == self.A:
            return [(1.0, self.B, -1.0)] if a == self.FORWARD else [(1.0, self.A, -1.0)]
        if s == self.B:
            return [(1.0, self.G, 10.0)] if a == self.FORWARD else [(1.0, self.B, -1.0)]
        return []


def test_bellman_backup_matches_hand_computed_values():
    env = ToyChainEnv()
    gamma = 0.9
    vi = ValueIteration(env, gamma=gamma, theta=1e-8)
    V, n_iter, converged, history = vi.run(max_iterations=1000)

    assert converged
    assert V[env.encode_state(env.G)] == pytest.approx(0.0, abs=1e-4)
    assert V[env.encode_state(env.B)] == pytest.approx(10.0, abs=1e-4)
    assert V[env.encode_state(env.A)] == pytest.approx(-1 + gamma * 10.0, abs=1e-4)


def test_compute_delta():
    V_old = np.array([1.0, 2.0, 3.0])
    V_new = np.array([1.5, 2.0, -1.0])
    assert ValueIteration.compute_delta(V_old, V_new) == pytest.approx(4.0)


def test_extract_policy_picks_best_action():
    env = ToyChainEnv()
    vi = ValueIteration(env, gamma=0.9, theta=1e-8)
    V, *_ = vi.run(max_iterations=1000)
    policy = vi.extract_policy(V)
    assert policy[env.A] == env.FORWARD
    assert policy[env.B] == env.FORWARD
