import numpy as np


class QLearningAgent:
    def __init__(self, env, alpha, gamma, epsilon_start, epsilon_min,
                 decay_type, decay_param, num_episodes,
                 reward_mode='sparse', initial_Q=None, seed=None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_type = decay_type
        self.decay_param = decay_param
        self.num_episodes = num_episodes
        self.reward_mode = reward_mode
        self.rng = np.random.default_rng(seed)

        n = env.n_states
        if initial_Q is not None:
            self.Q = np.array(initial_Q, dtype=np.float64, copy=True)
        else:
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

    def update(self, s, a, r, s_next, done):
        idx = self.env.encode_state(s)
        self.visits[idx] += 1
        target = r if done else r + self.gamma * np.max(self.Q[self.env.encode_state(s_next)])
        self.Q[idx, a] += self.alpha * (target - self.Q[idx, a])

    def train(self, logger=None, keep_detail_episodes=None,
              step_callback=None, trace_episodes=None):
        keep_detail_episodes = keep_detail_episodes or set()
        trace_episodes = trace_episodes or set()
        for ep in range(self.num_episodes):
            epsilon = self.epsilon_at(ep)
            state = self.env.reset()
            done = False
            do_trace = ep in trace_episodes
            if logger:
                logger.start_episode(ep, keep_detail=(ep in keep_detail_episodes), epsilon=epsilon)
            step_idx = 0
            while not done:
                a = self.select_action(state, epsilon)
                s_next, r, done, info = self.env.step(a)

                if do_trace and step_callback:
                    idx = self.env.encode_state(state)
                    idx_next = self.env.encode_state(s_next)
                    q_before = float(self.Q[idx, a])
                    max_next_q = 0.0 if done else float(np.max(self.Q[idx_next]))
                    td_target = r if done else r + self.gamma * max_next_q

                self.update(state, a, r, s_next, done)

                if logger:
                    logger.log_step(state, a, r, info, done=done)

                if do_trace and step_callback:
                    q_after = float(self.Q[self.env.encode_state(state), a])
                    step_callback({
                        'episode': ep, 'step': step_idx, 'state': state, 'action': a,
                        'reward': r, 'next_state': s_next, 'event': info.get('event'),
                        'q_before': q_before, 'max_next_q': max_next_q,
                        'td_target': td_target, 'td_error': td_target - q_before,
                        'q_after': q_after, 'alpha': self.alpha, 'gamma': self.gamma,
                        'epsilon': epsilon,
                    })

                state = s_next
                step_idx += 1
            if logger:
                logger.end_episode()
        return self.Q
