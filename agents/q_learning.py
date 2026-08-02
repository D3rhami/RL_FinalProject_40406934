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
        target = r if done else r + self.gamma * np.max(self.Q[self.env.encode_state(s_next)])
        self.Q[idx, a] += self.alpha * (target - self.Q[idx, a])

    def train(self, logger=None, keep_detail_episodes=None):
        keep_detail_episodes = keep_detail_episodes or set()
        for ep in range(self.num_episodes):
            epsilon = self.epsilon_at(ep)
            state = self.env.reset()
            done = False
            if logger:
                logger.start_episode(ep, keep_detail=(ep in keep_detail_episodes), epsilon=epsilon)
            while not done:
                a = self.select_action(state, epsilon)
                s_next, r, done, info = self.env.step(a)
                self.update(state, a, r, s_next, done)
                if logger:
                    logger.log_step(state, a, r, info, done=done)
                state = s_next
            if logger:
                logger.end_episode()
        return self.Q
