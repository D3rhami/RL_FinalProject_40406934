import time
import numpy as np
from tqdm import tqdm


class ValueIteration:
    def __init__(self, env, gamma: float, theta: float = 1e-6, reward_mode: str = 'sparse'):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.reward_mode = reward_mode
        self.states = env.all_states()

    @staticmethod
    def compute_delta(V_old, V_new) -> float:
        return float(np.max(np.abs(np.asarray(V_new) - np.asarray(V_old))))

    def _q_value(self, V, s, a) -> float:
        q = 0.0
        for p, ns, r in self.env.transition_model(s, a):
            q += p * (r + self.gamma * V[self.env.encode_state(ns)])
        return q

    def run(self, max_iterations: int = 1000, progress_bar: bool = False,
            progress_desc: str = None):
        n = self.env.n_states
        V = np.zeros(n)
        history = []
        start = time.time()
        converged = False
        n_iter = 0

        iteration_iter = tqdm(
            range(max_iterations), disable=not progress_bar,
            desc=progress_desc or f'VI (gamma={self.gamma})', unit='iter', leave=False,
        )
        for it in iteration_iter:
            V_new = V.copy()
            for s in self.states:
                idx = self.env.encode_state(s)
                if self.env.is_terminal(s):
                    V_new[idx] = 0.0
                    continue
                V_new[idx] = max(self._q_value(V, s, a) for a in range(4))
            delta = self.compute_delta(V, V_new)
            V = V_new
            history.append(delta)
            n_iter = it + 1
            if progress_bar:
                iteration_iter.set_postfix(delta=f'{delta:.2e}', refresh=False)
            if delta < self.theta:
                converged = True
                iteration_iter.close()
                break

        self.V = V
        self.history = history
        self.n_iterations = n_iter
        self.converged = converged
        self.runtime = time.time() - start
        return V, n_iter, converged, history

    def extract_policy(self, V) -> dict:
        policy = {}
        for s in self.states:
            if self.env.is_terminal(s):
                continue
            best_a, best_q = None, float('-inf')
            for a in range(4):
                q = self._q_value(V, s, a)
                if q > best_q:
                    best_q, best_a = q, a
            policy[s] = best_a
        return policy
