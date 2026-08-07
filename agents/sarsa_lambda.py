import numpy as np
from tqdm import tqdm


class SarsaLambdaAgent:
    def __init__(self, env, alpha, gamma, lam, trace_type,
                 epsilon_start, epsilon_min, decay_type, decay_param,
                 num_episodes, reward_mode='sparse', seed=None, trace_threshold=1e-6):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.trace_type = trace_type
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_type = decay_type
        self.decay_param = decay_param
        self.num_episodes = num_episodes
        self.reward_mode = reward_mode
        self.trace_threshold = trace_threshold
        self.rng = np.random.default_rng(seed)

        n = env.n_states
        self.Q = np.zeros((n, 4))
        self.visits = np.zeros(n, dtype=np.int64)

    def epsilon_at(self, episode: int) -> float:
        if self.decay_type == 'linear':
            frac = min(1.0, episode / self.decay_param)
            eps = self.epsilon_start - (self.epsilon_start - self.epsilon_min) * frac
        elif self.decay_type == 'exponential':
            eps = self.epsilon_start * (self.decay_param ** episode)
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")
        return max(self.epsilon_min, eps)

    def select_action(self, state, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, 4))
        idx = self.env.encode_state(state)
        return int(np.argmax(self.Q[idx]))

    def _apply_trace(self, E: dict, s_idx, a, delta):
        if self.trace_type == 'replacing':
            E[(s_idx, a)] = 1.0
        else:
            E[(s_idx, a)] = E.get((s_idx, a), 0.0) + 1.0

        to_delete = []
        for (si, ai), e in E.items():
            self.Q[si, ai] += self.alpha * delta * e
            new_e = self.gamma * self.lam * e
            if new_e < self.trace_threshold:
                to_delete.append((si, ai))
            else:
                E[(si, ai)] = new_e
        for k in to_delete:
            del E[k]

    def sarsa_lambda_update(self, E: dict, s, a, r, s_next, a_next, done) -> float:
        s_idx = self.env.encode_state(s)
        self.visits[s_idx] += 1
        if done:
            delta = r - self.Q[s_idx, a]
        else:
            s_next_idx = self.env.encode_state(s_next)
            delta = r + self.gamma * self.Q[s_next_idx, a_next] - self.Q[s_idx, a]
        self._apply_trace(E, s_idx, a, delta)
        return delta

    def train(self, logger=None, keep_detail_episodes=None,
              step_callback=None, trace_episodes=None, trace_dump_callback=None,
              progress_bar=False, progress_desc=None):
        keep_detail_episodes = keep_detail_episodes or set()
        trace_episodes = trace_episodes or set()
        episode_iter = tqdm(
            range(self.num_episodes), disable=not progress_bar,
            desc=progress_desc or f'SARSA(lambda={self.lam})', unit='ep', leave=False,
        )
        for ep in episode_iter:
            epsilon = self.epsilon_at(ep)
            E = {}
            state = self.env.reset()
            action = self.select_action(state, epsilon)
            done = False
            do_trace = ep in trace_episodes
            if logger:
                logger.start_episode(ep, keep_detail=(ep in keep_detail_episodes),
                                      epsilon=epsilon, lam=self.lam)
            step_idx = 0
            while not done:
                s_next, r, done, info = self.env.step(action)
                a_next = None if done else self.select_action(s_next, epsilon)

                delta = self.sarsa_lambda_update(E, state, action, r, s_next, a_next, done)

                if logger:
                    logger.log_step(state, action, r, info, done=done)

                if do_trace and step_callback:
                    step_callback({
                        'episode': ep, 'step': step_idx, 'state': state, 'action': action,
                        'next_state': s_next, 'next_action': a_next, 'reward': r,
                        'event': info.get('event'), 'delta': delta,
                        'active_traces': len(E), 'alpha': self.alpha,
                        'gamma': self.gamma, 'lam': self.lam, 'epsilon': epsilon,
                    })
                if do_trace and trace_dump_callback:
                    for (s_idx, a_idx), e_val in E.items():
                        trace_dump_callback(ep, step_idx, s_idx, a_idx, e_val)

                if done:
                    break
                state, action = s_next, a_next
                step_idx += 1
            row = logger.end_episode() if logger else None
            if progress_bar and row is not None:
                episode_iter.set_postfix(
                    eps=f'{epsilon:.3f}', success=int(row['success']),
                    steps=row['steps'], refresh=False,
                )
        return self.Q
