"""
RL Maze GUI — premium dark chrome + tk.Canvas maze.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tkinter as tk
import customtkinter as ctk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environments.maze import MazeEnv, CellType, load_config
from gui.renderer import MazeRenderer

ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('dark-blue')

BG = '#0B0E14'
PANEL = '#151A22'
PANEL2 = '#1B2130'
BORDER = '#2A3344'
TEXT = '#E8EDF5'
MUTED = '#8B95A8'
ACCENT = '#3B82F6'
ACCENT2 = '#22C55E'
WARN = '#F59E0B'
DANGER = '#EF4444'

MAPS = {
    'source': 'environments/maps/source_maze.json',
    'similar': 'environments/maps/target_similar.json',
    'different': 'environments/maps/target_different.json',
}

TERMINATION_STYLE = {
    'goal_reached': dict(icon='\U0001F3C1', title='Goal Reached!',
                          subtitle='The agent found the goal.',
                          bg='#14532D', border='#86EFAC'),
    'energy_depleted': dict(icon='\u26A1', title='Energy Depleted',
                             subtitle='The agent ran out of energy before reaching the goal.',
                             bg='#7F1D1D', border='#FECACA'),
    'step_cap': dict(icon='\u23F1', title='Step Limit Reached',
                      subtitle='Episode ended after the maximum step budget.',
                      bg='#78350F', border='#FDE68A'),
}

MODE_HELP = {
    'eval': 'Eval mode: the agent acts greedily (\u03B5=0) from the loaded model \u2014 no learning happens.',
    'train': 'Train mode: the agent explores (\u03B5=0.05) and updates its Q-values live, for the episode count below.',
}


def load_episode(path):
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and 'steps' in data:
        return data['steps']
    return data


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('RL Maze Studio')
        self.geometry('1360x920')
        self.minsize(1100, 760)
        self.configure(fg_color=BG)
        self.cfg = load_config()

        self.env = None
        self.agent = None
        self.agent_kind = None
        self.agent_state = None
        self.renderer = None
        self.playing = False
        self.paused = False
        self.mode = 'eval'
        self.algo = 'Q-Learning'
        self.map_kind = 'source'
        self.delay_ms = 100
        self.episode_steps = []
        self.frame_i = 0
        self.current_episode = 0
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.recent_success = []
        self.policy_on = False
        self._tick_job = None
        self._run_id = 0
        self._redrawing = False
        self._model_cache = {}
        self._sarsa_E = None
        self._sarsa_action = None
        self._last_policy_grid = None
        self._wall_clock0 = time.time()
        self.event_log = []
        self.start_energy = int(self.cfg['env']['max_energy'])
        self._target_episodes = 1
        self._episodes_done = 0

        self._build_layout()
        self._build_info()
        self._build_controls()
        self._build_table()
        self.renderer = MazeRenderer(self.maze_canvas)
        self.maze_canvas.bind('<Configure>', self._on_canvas_resize)
        self.after(100, self._bootstrap)

    def _card(self, parent, **kw):
        defaults = dict(fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER)
        defaults.update(kw)
        return ctk.CTkFrame(parent, **defaults)

    def _label(self, parent, text, muted=False, size=13, bold=False, **kw):
        return ctk.CTkLabel(
            parent, text=text,
            text_color=MUTED if muted else TEXT,
            font=ctk.CTkFont(family='Segoe UI', size=size,
                             weight='bold' if bold else 'normal'),
            **kw)

    def _build_layout(self):
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=8)
        self.grid_rowconfigure(2, weight=2)
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color=BG, height=52)
        header.grid(row=0, column=0, sticky='ew', padx=14, pady=(12, 4))
        header.grid_columnconfigure(1, weight=1)
        self.header_title = self._label(header, 'Idle', size=20, bold=True)
        self.header_title.grid(row=0, column=0, sticky='w')
        self.status_pill = ctk.CTkLabel(
            header, text='  idle  ', corner_radius=999,
            fg_color='#1F2937', text_color=MUTED,
            font=ctk.CTkFont(size=12, weight='bold'))
        self.status_pill.grid(row=0, column=2, sticky='e', padx=6)
        self.map_pill = ctk.CTkLabel(
            header, text='  source  ', corner_radius=999,
            fg_color='#1E3A5F', text_color='#93C5FD',
            font=ctk.CTkFont(size=12, weight='bold'))
        self.map_pill.grid(row=0, column=1, sticky='e', padx=6)

        top = ctk.CTkFrame(self, fg_color='transparent')
        top.grid(row=1, column=0, sticky='nsew', padx=14, pady=6)
        top.grid_columnconfigure(0, weight=7)
        top.grid_columnconfigure(1, weight=3)
        top.grid_rowconfigure(0, weight=1)

        self.maze_frame = self._card(top)
        self.maze_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        self.maze_canvas = tk.Canvas(
            self.maze_frame, bg='#0F1218', highlightthickness=0, bd=0)
        self.maze_canvas.pack(fill='both', expand=True, padx=10, pady=10)

        self.info_frame = self._card(top)
        self.info_frame.grid(row=0, column=1, sticky='nsew')

        bottom = ctk.CTkFrame(self, fg_color='transparent')
        bottom.grid(row=2, column=0, sticky='nsew', padx=14, pady=(4, 14))
        bottom.grid_columnconfigure(0, weight=6)
        bottom.grid_columnconfigure(1, weight=4)
        bottom.grid_rowconfigure(0, weight=1)

        self.controls_frame = self._card(bottom)
        self.controls_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        self.table_frame = self._card(bottom)
        self.table_frame.grid(row=0, column=1, sticky='nsew')

    def _metric_card(self, parent, row, col, title, var):
        card = ctk.CTkFrame(
            parent, fg_color=PANEL2, corner_radius=8,
            border_width=1, border_color=BORDER)
        card.grid(row=row, column=col, sticky='nsew', padx=4, pady=3)
        card.grid_columnconfigure(0, weight=1)
        self._label(card, title, muted=True, size=10).grid(
            row=0, column=0, sticky='w', padx=8, pady=(6, 0))
        ctk.CTkLabel(
            card, textvariable=var, anchor='w', text_color=TEXT,
            font=ctk.CTkFont(family='Segoe UI', size=14, weight='bold')
        ).grid(row=1, column=0, sticky='w', padx=8, pady=(0, 6))

    def _build_info(self):
        f = self.info_frame
        f.grid_columnconfigure(0, weight=1)
        f.grid_columnconfigure(1, weight=1)
        self._label(f, 'LIVE STATUS', muted=True, size=11, bold=True).grid(
            row=0, column=0, columnspan=2, sticky='w', padx=12, pady=(10, 2))

        self.energy_label = self._label(f, 'Energy 60/60', muted=True, size=11)
        self.energy_label.grid(row=1, column=0, columnspan=2, sticky='w', padx=12, pady=(4, 0))
        self.energy_bar = ctk.CTkProgressBar(
            f, progress_color=ACCENT2, fg_color='#111827', height=14, corner_radius=8)
        self.energy_bar.grid(row=2, column=0, columnspan=2, sticky='ew', padx=12, pady=(4, 8))
        self.energy_bar.set(1.0)

        self.episode_var = tk.StringVar(value='0')
        self.step_var = tk.StringVar(value='0')
        self.reward_var = tk.StringVar(value='0.00')
        self.total_r_var = tk.StringVar(value='0.0')
        self.gamma_var = tk.StringVar(value=str(self.cfg['q_learning']['gamma']))
        self.eps_var = tk.StringVar(value='-')
        self.key_var = tk.StringVar(value='no')
        self.door_var = tk.StringVar(value='closed')
        self.rm_var = tk.StringVar(value='sparse')
        self.algo_var = tk.StringVar(value='Q-Learning')
        self.success_var = tk.StringVar(value='-')
        self.time_var = tk.StringVar(value='0s')

        metrics = [
            ('Episode', self.episode_var),
            ('Step', self.step_var),
            ('Last reward', self.reward_var),
            ('Episode return', self.total_r_var),
            ('gamma', self.gamma_var),
            ('epsilon', self.eps_var),
            ('Key', self.key_var),
            ('Door', self.door_var),
            ('Reward mode', self.rm_var),
            ('Algorithm', self.algo_var),
            ('Recent success', self.success_var),
            ('Elapsed', self.time_var),
        ]
        for i, (title, var) in enumerate(metrics):
            self._metric_card(f, 3 + i // 2, i % 2, title, var)

    def _seg(self, parent, values, command, selected):
        w = ctk.CTkSegmentedButton(
            parent, values=values, command=command,
            fg_color='#0F141D', selected_color=ACCENT,
            selected_hover_color='#2563EB', unselected_color='#1F2937',
            unselected_hover_color='#273449', text_color=TEXT,
            font=ctk.CTkFont(size=12, weight='bold'), height=32)
        w.set(selected)
        return w

    def _build_controls(self):
        f = self.controls_frame
        for c in range(8):
            f.grid_columnconfigure(c, weight=1)

        self._label(f, 'CONTROLS', muted=True, size=11, bold=True).grid(
            row=0, column=0, columnspan=8, sticky='w', padx=12, pady=(10, 2))

        self._label(f, 'Algorithm', muted=True, size=11).grid(row=1, column=0, sticky='w', padx=12)
        self.algo_seg = self._seg(f, ['VI', 'Q-Learning', 'SARSA(L)'], self._on_algo, 'Q-Learning')
        self.algo_seg.grid(row=1, column=1, columnspan=2, sticky='ew', padx=4, pady=4)

        self._label(f, 'Map', muted=True, size=11).grid(row=1, column=3, sticky='w', padx=8)
        self.env_seg = self._seg(f, ['source', 'similar', 'different'], self._on_env, 'source')
        self.env_seg.grid(row=1, column=4, columnspan=2, sticky='ew', padx=4, pady=4)

        self.mode_seg = self._seg(f, ['train', 'eval'], self._on_mode, 'eval')
        self.mode_seg.grid(row=1, column=6, columnspan=2, sticky='ew', padx=4, pady=4)

        btn_specs = [
            ('Start', self.start_run, ACCENT2),
            ('Stop', self.stop_run, DANGER),
            ('Continue', self.continue_run, ACCENT),
            ('Reset', self.reset_run, '#374151'),
            ('Rerun', self.rerun, '#4B5563'),
        ]
        for i, (label, cmd, color) in enumerate(btn_specs):
            ctk.CTkButton(
                f, text=label, command=cmd, width=88, height=34,
                fg_color=color, hover_color=color, corner_radius=10,
                font=ctk.CTkFont(size=13, weight='bold')
            ).grid(row=2, column=i, padx=6, pady=10, sticky='ew')

        self._label(f, 'Speed', muted=True, size=11).grid(row=3, column=0, sticky='w', padx=12)
        self.speed_slider = ctk.CTkSlider(
            f, from_=20, to=400, number_of_steps=38, command=self._on_speed,
            progress_color=ACCENT, button_color='#93C5FD', fg_color='#111827')
        self.speed_slider.set(100)
        self.speed_slider.grid(row=3, column=1, columnspan=4, sticky='ew', padx=8, pady=8)
        self.speed_label = self._label(f, '100 ms', muted=True, size=12)
        self.speed_label.grid(row=3, column=5, sticky='w')

        self.policy_switch = ctk.CTkSwitch(
            f, text='Policy overlay', command=self._on_policy_toggle,
            progress_color=ACCENT, button_color='#E5E7EB',
            font=ctk.CTkFont(size=12, weight='bold'))
        self.policy_switch.grid(row=3, column=6, columnspan=2, sticky='e', padx=12)

        self._label(f, 'Start energy', muted=True, size=11).grid(
            row=4, column=0, sticky='w', padx=12, pady=(6, 0))
        energy_box = ctk.CTkFrame(f, fg_color='transparent')
        energy_box.grid(row=4, column=1, columnspan=2, sticky='ew', padx=4, pady=(4, 0))
        energy_box.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            energy_box, text='\u2212', width=32, height=28, corner_radius=8,
            fg_color='#374151', hover_color='#4B5563',
            font=ctk.CTkFont(size=14, weight='bold'),
            command=lambda: self._on_energy_delta(-5)
        ).grid(row=0, column=0, padx=2)
        self.energy_start_var = tk.StringVar(value=str(self.start_energy))
        ctk.CTkLabel(
            energy_box, textvariable=self.energy_start_var, text_color=TEXT,
            font=ctk.CTkFont(size=13, weight='bold')
        ).grid(row=0, column=1)
        ctk.CTkButton(
            energy_box, text='+', width=32, height=28, corner_radius=8,
            fg_color='#374151', hover_color='#4B5563',
            font=ctk.CTkFont(size=14, weight='bold'),
            command=lambda: self._on_energy_delta(5)
        ).grid(row=0, column=2, padx=2)

        self.episodes_label = self._label(f, 'Episodes (train)', muted=True, size=11)
        self.episodes_label.grid(row=4, column=3, sticky='w', padx=8, pady=(6, 0))
        self.train_episodes_var = tk.StringVar(value='50')
        self.episodes_entry = ctk.CTkEntry(
            f, textvariable=self.train_episodes_var, width=70, height=28,
            fg_color='#0F141D', border_color=BORDER, text_color=TEXT,
            justify='center')
        self.episodes_entry.grid(row=4, column=4, sticky='w', padx=4, pady=(4, 0))
        self.episodes_label.grid_remove()
        self.episodes_entry.grid_remove()

        self.mode_help_var = tk.StringVar(value=MODE_HELP['eval'])
        ctk.CTkLabel(
            f, textvariable=self.mode_help_var, anchor='w', wraplength=520,
            text_color=MUTED, font=ctk.CTkFont(size=11), justify='left'
        ).grid(row=5, column=0, columnspan=8, sticky='w', padx=12, pady=(8, 8))

    def _build_table(self):
        f = self.table_frame
        f.grid_rowconfigure(1, weight=1)
        f.grid_columnconfigure(0, weight=1)
        self._label(f, 'EVENT FEED', muted=True, size=11, bold=True).grid(
            row=0, column=0, sticky='w', padx=12, pady=(10, 2))
        self.table_box = ctk.CTkTextbox(
            f, font=ctk.CTkFont(family='Consolas', size=12),
            fg_color='#0F141D', text_color='#CBD5E1', border_color=BORDER, border_width=1)
        self.table_box.grid(row=1, column=0, sticky='nsew', padx=10, pady=(4, 10))
        self.table_box.insert('1.0', 'Ready. Press Start for a greedy rollout.\n')

    def _log_event(self, msg):
        self.event_log.append(msg)
        self.event_log = self.event_log[-40:]
        self.table_box.delete('1.0', 'end')
        self.table_box.insert('1.0', '\n'.join(self.event_log))
        self.table_box.see('end')

    def _set_status(self, text, color='#1F2937', fg=MUTED):
        self.status_pill.configure(text=f'  {text}  ', fg_color=color, text_color=fg)

    def _set_header(self, text, color=TEXT):
        self.header_title.configure(text=text, text_color=color)

    def _bootstrap(self):
        self.load_environment('source')
        self.update_episode(0)
        self._tick_clock()
        self._set_status('ready', '#1E3A5F', '#93C5FD')
        self._set_header('Idle')

    def _on_canvas_resize(self, _event=None):
        if self.env is None or self.playing or self._redrawing:
            return
        self._redraw_board()

    def _redraw_board(self):
        if self.env is None or self._redrawing:
            return
        self._redrawing = True
        try:
            self.renderer.draw_maze(self.env.grid, door_pos=self.env.door_pos)
            if self.door_var.get() == 'open':
                self.renderer.set_door_state(True)
            state = getattr(self, 'agent_state', None)
            if state is not None:
                self.renderer.draw_agent(state[0], state[1], state[2])
            if self.policy_on:
                self._apply_policy_overlay()
        finally:
            self._redrawing = False

    def load_environment(self, kind):
        self.stop_run()
        self.map_kind = kind
        self.map_pill.configure(text=f'  {kind}  ')
        path = MAPS[kind]
        rm = 'sparse'
        self.env = MazeEnv(path, config=self.cfg, reward_mode=rm, seed=self.cfg['base_seed'])
        self.agent = None
        self.agent_kind = None
        self._sarsa_E = None
        self._sarsa_action = None
        self.env.reset()
        self._apply_start_energy()
        state = self.env.state
        self.agent_state = state
        self.renderer.clear_trail()
        self.renderer.clear_overlay()
        self._redraw_board()
        self._sync_info_from_state(state, epsilon=None, reward=0.0, done=False, info={})
        self.rm_var.set(rm)
        self.door_var.set('closed')
        self._log_event(f'Loaded map={kind}  walkable={int(np.sum(self.env.grid != CellType.WALL))}')

    def update_episode(self, n):
        self.current_episode = int(n)
        self.episode_var.set(str(self.current_episode))

    def _sync_info_from_state(self, state, epsilon, reward, done, info):
        r, c, hk, en = state
        self.key_var.set('YES' if hk else 'no')
        self.energy_bar.set(en / max(1, self.env.MAX_ENERGY))
        self.energy_label.configure(text=f'Energy {en}/{self.env.MAX_ENERGY}')
        if epsilon is not None:
            self.eps_var.set(f'{epsilon:.3f}')
        if reward is not None:
            self.last_reward = float(reward)
            self.reward_var.set(f'{self.last_reward:.2f}')
        term = info.get('termination_reason')
        if term:
            self._log_event(f'END: {term}')
        event = info.get('event')
        if event == 'key_pickup':
            self.renderer.set_door_state(True)
            self.door_var.set('open')
            self.renderer.show_toast('\U0001F511 Key found!', bg='#7C5E10', fg='#FFF3C4', app=self)
            self._set_header('Key Found!', '#FDE68A')
            self._log_event(f'key pickup at ({r},{c})')
        if event == 'wall_hit':
            self.renderer.flash_cell(r, c, color='#F87171', app=self)
            self._log_event(f'wall hit at ({r},{c})')
        if event == 'penalty_entry':
            self.renderer.flash_cell(r, c, color='#FB7185', app=self)
            self._log_event(f'penalty at ({r},{c})')
        if event == 'goal_reached':
            self._log_event(f'GOAL at ({r},{c})')

    def _on_algo(self, value):
        self.stop_run()
        self.algo = value
        self.algo_var.set(value)
        if self.map_kind != 'source' and value != 'Q-Learning':
            self.algo_seg.set('Q-Learning')
            self.algo = 'Q-Learning'
            self.algo_var.set('Q-Learning')
            self._log_event('Target maps: Q-Learning only')
        self.agent = None
        self.agent_kind = None
        if self.policy_on:
            self._apply_policy_overlay()

    def _on_env(self, value):
        if value != 'source' and self.algo != 'Q-Learning':
            self.algo_seg.set('Q-Learning')
            self.algo = 'Q-Learning'
            self.algo_var.set('Q-Learning')
        self.load_environment(value)

    def _on_mode(self, value):
        self.mode = value
        self.mode_help_var.set(MODE_HELP.get(value, ''))
        if value == 'train':
            self.episodes_label.grid()
            self.episodes_entry.grid()
        else:
            self.episodes_label.grid_remove()
            self.episodes_entry.grid_remove()
        self._log_event(f'mode -> {value}')

    def _on_energy_delta(self, delta):
        if self.env is None:
            return
        new_val = max(0, min(int(self.env.MAX_ENERGY), self.start_energy + delta))
        self.start_energy = new_val
        self.energy_start_var.set(str(new_val))
        if not self.playing:
            self._apply_start_energy()
            self.agent_state = self.env.state
            self._sync_info_from_state(self.agent_state, epsilon=None, reward=0.0, done=False, info={})
            self._redraw_board()
        self._log_event(f'start energy -> {new_val}')

    def _apply_start_energy(self):
        if self.env is None or self.env.state is None:
            return
        r, c, hk, _ = self.env.state
        self.env.state = (r, c, hk, self.start_energy)

    def _on_speed(self, value):
        self.delay_ms = int(float(value))
        self.speed_label.configure(text=f'{self.delay_ms} ms')

    def _on_policy_toggle(self):
        self.policy_on = bool(self.policy_switch.get())
        if self.policy_on:
            self._apply_policy_overlay()
            n = int(np.sum(self._last_policy_grid >= 0)) if self._last_policy_grid is not None else 0
            self._log_event(f'policy overlay ON ({n} cells)')
        else:
            self.renderer.clear_policy_overlay()
            self._log_event('policy overlay OFF')

    def _read_model(self, path):
        path = str(path)
        if path in self._model_cache:
            return self._model_cache[path]
        with open(path) as f:
            model = json.load(f)
        self._model_cache[path] = model
        return model

    def _apply_policy_overlay(self):
        actions = self._load_policy_grid()
        self._last_policy_grid = actions
        self.renderer.draw_policy_overlay(actions)

    def _slice_has_key_energy(self):
        state = getattr(self, 'agent_state', None)
        if state is not None:
            return int(state[2]), int(state[3])
        return 0, int(self.env.MAX_ENERGY)

    def _load_policy_grid(self):
        has_key, energy = self._slice_has_key_energy()

        if self.agent_kind == 'mf' and self.agent is not None and hasattr(self.agent, 'Q'):
            return self._policy_from_qtable(self.agent.Q, has_key, energy)

        if self.agent_kind == 'vi' and isinstance(self.agent, dict):
            return self._policy_from_vi_dict(self.agent, has_key, energy)

        path = self._default_model_path()
        if path is None or not Path(path).exists():
            return np.full((self.env.maze_size, self.env.maze_size), -1, dtype=int)
        model = self._read_model(path)
        if 'Q' in model:
            return self._policy_from_saved_q(model, has_key, energy)
        return self._policy_from_saved_vi(model, has_key, energy)

    def _policy_from_qtable(self, Q, has_key, energy):
        size = self.env.maze_size
        grid = np.full((size, size), -1, dtype=int)
        for r in range(size):
            for c in range(size):
                if int(self.env.grid[r, c]) == CellType.WALL:
                    continue
                idx = self.env.encode_state((r, c, has_key, energy))
                q = Q[idx]
                if float(np.max(np.abs(q))) < 1e-12:
                    continue
                grid[r, c] = int(np.argmax(q))
        return grid

    def _policy_from_vi_dict(self, policy, has_key, energy):
        size = self.env.maze_size
        grid = np.full((size, size), -1, dtype=int)
        for r in range(size):
            for c in range(size):
                if int(self.env.grid[r, c]) == CellType.WALL:
                    continue
                a = policy.get((r, c, has_key, energy), -1)
                if a is None or int(a) < 0:
                    a = self._best_vi_action(policy, r, c, has_key)
                if a is not None and int(a) >= 0:
                    grid[r, c] = int(a)
        return grid

    def _best_vi_action(self, policy, r, c, has_key):
        best_a, best_en = -1, -1
        for (rr, cc, hk, en), a in policy.items():
            if rr == r and cc == c and hk == has_key and int(a) >= 0 and en >= best_en:
                best_en, best_a = en, int(a)
        return best_a

    def _policy_from_saved_vi(self, model, has_key, energy):
        policy = {tuple(s): int(a) for s, a in zip(model['states'], model['policy'])}
        return self._policy_from_vi_dict(policy, has_key, energy)

    def _policy_from_saved_q(self, model, has_key, energy):
        size = self.env.maze_size
        grid = np.full((size, size), -1, dtype=int)
        visits = model.get('visits')
        by_cell = {}
        for i, s in enumerate(model['states']):
            r, c, hk, en = s
            if hk != has_key:
                continue
            q = model['Q'][i]
            if visits is not None:
                score = int(visits[i])
            else:
                score = float(np.max(np.abs(q)))
            if score <= 0:
                continue
            key = (r, c)
            cur = by_cell.get(key)
            take = False
            if cur is None:
                take = True
            elif en == energy and (cur['en'] != energy or score > cur['score']):
                take = True
            elif cur['en'] != energy and en != energy and score > cur['score']:
                take = True
            if take:
                by_cell[key] = {'en': en, 'score': score, 'a': int(np.argmax(q))}
        for (r, c), info in by_cell.items():
            if int(self.env.grid[r, c]) == CellType.WALL:
                continue
            grid[r, c] = info['a']
        return grid

    def _default_model_path(self):
        if self.map_kind != 'source':
            return ('results/models/transfer/transfer_different_full_seed81150.json'
                    if self.map_kind == 'different'
                    else 'results/models/transfer/transfer_similar_full_seed81150.json')
        if self.algo == 'VI':
            g = self.cfg['experiment_grid']['value_iteration']['reference_gamma']
            return f'results/models/vi/vi_sparse_gamma{g}.json'
        if self.algo.startswith('SARSA'):
            return 'results/models/sarsa/sarsa_sparse_lambda0.3_seed8565.json'
        return 'results/models/q_learning/q_learning_shaped_linear_seed8565.json'

    def _make_agent(self):
        seed = self.cfg['base_seed']
        if self.algo == 'VI':
            path = self._default_model_path()
            model = self._read_model(path)
            policy = {tuple(s): int(a) for s, a in zip(model['states'], model['policy'])}
            self._log_event(f'loaded VI policy {Path(path).name}')
            return ('vi', policy)
        if self.algo.startswith('SARSA'):
            from agents.sarsa_lambda import SarsaLambdaAgent
            scfg = self.cfg['sarsa_lambda']
            agent = SarsaLambdaAgent(
                self.env, alpha=scfg['alpha'], gamma=scfg['gamma'], lam=0.3,
                trace_type=scfg['trace_type'], epsilon_start=scfg['epsilon_start'],
                epsilon_min=scfg['epsilon_min'], decay_type='exponential',
                decay_param=scfg['exponential_decay_rate'],
                num_episodes=1, reward_mode='sparse', seed=seed)
            self._load_q_into_agent(agent, self._default_model_path())
            return ('mf', agent)
        from agents.q_learning import QLearningAgent
        qcfg = self.cfg['q_learning']
        agent = QLearningAgent(
            self.env, alpha=qcfg['alpha'], gamma=qcfg['gamma'],
            epsilon_start=0.05 if self.mode == 'train' else 0.0,
            epsilon_min=0.05 if self.mode == 'train' else 0.0,
            decay_type='exponential', decay_param=qcfg['exponential_decay_rate'],
            num_episodes=1, reward_mode='sparse', seed=seed)
        self._load_q_into_agent(agent, self._default_model_path())
        return ('mf', agent)

    def _load_q_into_agent(self, agent, path):
        if path is None or not Path(path).exists():
            self._log_event(f'model missing: {path}')
            return
        model = self._read_model(path)
        if 'Q' not in model:
            return
        for s, q in zip(model['states'], model['Q']):
            idx = agent.env.encode_state(tuple(s))
            agent.Q[idx] = q
        if 'visits' in model:
            for s, v in zip(model['states'], model['visits']):
                idx = agent.env.encode_state(tuple(s))
                agent.visits[idx] = int(v)
        self._log_event(f'loaded {Path(path).name}')

    def start_run(self):
        self.stop_run()
        self._run_id += 1
        run_id = self._run_id
        self.playing = True
        self.paused = False
        self.episode_return = 0.0
        self.frame_i = 0
        self._episodes_done = 0
        if self.mode == 'train':
            try:
                self._target_episodes = max(1, min(5000, int(self.train_episodes_var.get())))
            except ValueError:
                self._target_episodes = 50
                self.train_episodes_var.set('50')
        else:
            self._target_episodes = 1
        self.renderer.clear_trail()
        self.renderer.clear_overlay()
        self.update_episode(self.current_episode + 1)
        self._set_status('running', '#14532D', '#86EFAC')
        self._set_header('Started', ACCENT2 if self.mode == 'train' else '#93C5FD')
        self.agent_kind, self.agent = self._make_agent()
        self._sarsa_E = {} if self.algo.startswith('SARSA') else None
        self.env.reset()
        self._apply_start_energy()
        self.agent_state = self.env.state
        self.renderer.set_door_state(False)
        self.door_var.set('closed')
        self.renderer.draw_agent(self.agent_state[0], self.agent_state[1], self.agent_state[2])
        self.n_steps = 0
        if self.agent_kind == 'mf':
            eps = 0.05 if self.mode == 'train' else 0.0
            self._sarsa_action = self.agent.select_action(self.agent_state, eps)
        else:
            self._sarsa_action = None
        if self.policy_on:
            self._apply_policy_overlay()
        if self.mode == 'train':
            self._log_event(f'Start {self.algo} / train on {self.map_kind} '
                             f'({self._target_episodes} episodes)')
        else:
            self._log_event(f'Start {self.algo} / eval on {self.map_kind}')
        self._tick(run_id)

    def stop_run(self):
        was = self.playing
        self.playing = False
        self.paused = False
        self._run_id += 1
        if self._tick_job is not None:
            try:
                self.after_cancel(self._tick_job)
            except Exception:
                pass
            self._tick_job = None
        if was:
            self._set_status('stopped', '#7F1D1D', '#FECACA')
            self._set_header('Stopped', '#FECACA')

    def continue_run(self):
        if not self.playing:
            self.start_run()
            return
        self.paused = False
        self._set_status('running', '#14532D', '#86EFAC')
        self._tick(self._run_id)

    def reset_run(self):
        self.stop_run()
        self.agent = None
        self.agent_kind = None
        self._sarsa_E = None
        self._sarsa_action = None
        self.load_environment(self.map_kind)
        self._set_status('ready', '#1E3A5F', '#93C5FD')
        self._set_header('Idle')
        self._log_event('reset')

    def rerun(self):
        self.reset_run()
        self.start_run()

    def _act(self, state):
        if self.agent_kind == 'vi':
            a = self.agent.get(state, -1)
            if a is None or int(a) < 0:
                a = self._best_vi_action(self.agent, state[0], state[1], state[2])
            return 0 if a is None or int(a) < 0 else int(a)
        eps = 0.05 if self.mode == 'train' else 0.0
        self.eps_var.set(f'{eps:.3f}')
        if self._sarsa_E is not None and self._sarsa_action is not None:
            return int(self._sarsa_action)
        return self.agent.select_action(state, eps)

    def _tick(self, run_id=None):
        if run_id is None:
            run_id = self._run_id
        if not self.playing or self.paused or run_id != self._run_id:
            return
        state = self.agent_state
        a = self._act(state)
        s_next, r, done, info = self.env.step(a)
        if self.mode == 'train' and self.agent_kind == 'mf':
            if self._sarsa_E is not None:
                eps = 0.05
                a_next = 0 if done else self.agent.select_action(s_next, eps)
                self.agent.sarsa_lambda_update(self._sarsa_E, state, a, r, s_next, a_next, done)
                self._sarsa_action = None if done else a_next
            else:
                self.agent.update(state, a, r, s_next, done)
        self.n_steps += 1
        self.episode_return += r
        self.step_var.set(str(self.n_steps))
        self.total_r_var.set(f'{self.episode_return:.1f}')
        self.renderer.draw_agent(s_next[0], s_next[1], s_next[2])
        self._sync_info_from_state(s_next, None, r, done, info)
        self.agent_state = s_next
        if self.policy_on:
            self._apply_policy_overlay()
        if done:
            term = info.get('termination_reason')
            success = term == 'goal_reached'
            self.recent_success.append(int(success))
            self.recent_success = self.recent_success[-20:]
            rate = float(np.mean(self.recent_success)) if self.recent_success else 0.0
            self.success_var.set(f'{rate:.0%} (n={len(self.recent_success)})')
            self._episodes_done += 1
            is_final = self.mode != 'train' or self._episodes_done >= self._target_episodes

            if is_final:
                self.playing = False
                self._set_status('goal' if success else 'ended',
                                 '#14532D' if success else '#7F1D1D',
                                 '#86EFAC' if success else '#FECACA')
                style = TERMINATION_STYLE.get(term)
                if style:
                    self.renderer.show_overlay(**style)
                    self._set_header(style['title'],
                                     style['border'] if success else '#FECACA')
                return

            self._log_event(f'ep {self._episodes_done}/{self._target_episodes} '
                             f'ended: {term}')
            self._set_header(f'Training \u2014 episode {self._episodes_done + 1}'
                             f'/{self._target_episodes}', ACCENT2)
            self.update_episode(self.current_episode + 1)
            self.n_steps = 0
            self.episode_return = 0.0
            self.renderer.clear_trail()
            self.env.reset()
            self._apply_start_energy()
            self.agent_state = self.env.state
            self.renderer.set_door_state(False)
            self.door_var.set('closed')
            self.renderer.draw_agent(self.agent_state[0], self.agent_state[1], self.agent_state[2])
            if self.agent_kind == 'mf':
                self._sarsa_action = self.agent.select_action(self.agent_state, 0.05)
            if run_id != self._run_id:
                return
            self._tick_job = self.after(self.delay_ms, lambda: self._tick(run_id))
            return
        if run_id != self._run_id:
            return
        self._tick_job = self.after(self.delay_ms, lambda: self._tick(run_id))

    def _tick_clock(self):
        elapsed = int(time.time() - self._wall_clock0)
        self.time_var.set(f'{elapsed}s')
        self.after(1000, self._tick_clock)

    def play_frame(self, i):
        if not self.episode_steps:
            return
        i = max(0, min(i, len(self.episode_steps) - 1))
        self.frame_i = i
        step = self.episode_steps[i]
        state = step['state']
        if isinstance(state, list):
            state = tuple(state)
        self.renderer.draw_agent(state[0], state[1], state[2])
        self.step_var.set(str(step.get('step', i)))
        self.reward_var.set(f'{step.get("reward", 0):.2f}')


if __name__ == '__main__':
    App().mainloop()
