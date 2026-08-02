import numpy as np


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
        if done:
            delta = r - self.Q[s_idx, a]
        else:
            s_next_idx = self.env.encode_state(s_next)
            delta = r + self.gamma * self.Q[s_next_idx, a_next] - self.Q[s_idx, a]
        self._apply_trace(E, s_idx, a, delta)
        return delta

    def train(self, logger=None, keep_detail_episodes=None):
        keep_detail_episodes = keep_detail_episodes or set()
        for ep in range(self.num_episodes):
            epsilon = self.epsilon_at(ep)
            E = {}
            state = self.env.reset()
            action = self.select_action(state, epsilon)
            done = False
            if logger:
                logger.start_episode(ep, keep_detail=(ep in keep_detail_episodes),
                                      epsilon=epsilon, lam=self.lam)
            while not done:
                s_next, r, done, info = self.env.step(action)
                if done:
                    self.sarsa_lambda_update(E, state, action, r, s_next, None, done)
                    if logger:
                        logger.log_step(state, action, r, info, done=done)
                    break
                a_next = self.select_action(s_next, epsilon)
                self.sarsa_lambda_update(E, state, action, r, s_next, a_next, done)
                if logger:
                    logger.log_step(state, action, r, info, done=done)
                state, action = s_next, a_next
            if logger:
                logger.end_episode()
        return self.Q
