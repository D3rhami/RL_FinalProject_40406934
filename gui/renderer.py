"""
Premium maze canvas renderer (plain tk.Canvas — no CTkCanvas).
"""
import tkinter as tk

from environments.maze import CellType

# Assignment hex table on a dark board frame
CELL_FILL = {
    CellType.EMPTY:   '#FFFFFF',
    CellType.WALL:    '#2D2D2D',
    CellType.PENALTY: '#FF6464',
    CellType.START:   '#64C864',
    CellType.KEY:     '#FFD700',
    CellType.DOOR:    '#8B5A2B',
    CellType.GOAL:    '#00C800',
}

CELL_LETTER = {
    CellType.START: ('S', '#0B3D14'),
    CellType.KEY: ('K', '#1A1200'),
    CellType.DOOR: ('D', '#FFF8F0'),
    CellType.GOAL: ('G', '#06381A'),
    CellType.PENALTY: ('P', '#FFFFFF'),
}
ARROW = {0: '↑', 1: '↓', 2: '←', 3: '→'}
DOOR_CLOSED = '#8B5A2B'
DOOR_OPEN = '#C4A574'
AGENT_NO_KEY = '#2563EB'
AGENT_HAS_KEY = '#D946EF'
BOARD_BG = '#0B0E14'
FRAME_FILL = '#151A22'
FRAME_EDGE = '#3B82F6'
GRID_LINE = '#1A1A1A'
EMPTY_LINE = '#E5E7EB'
TRAIL = '#93C5FD'


class MazeRenderer:
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.cell_ids = {}
        self.letter_ids = {}
        self.arrow_ids = {}
        self.trail_ids = []
        self.agent_ids = []
        self.banner_ids = []
        self.grid = None
        self.size = 0
        self.cell_px = 24
        self.pad_x = 8
        self.pad_y = 8
        self.door_pos = None
        self.door_open = False
        self.trail = []

    def clear(self):
        self.canvas.delete('all')
        self.cell_ids.clear()
        self.letter_ids.clear()
        self.arrow_ids.clear()
        self.trail_ids.clear()
        self.agent_ids.clear()
        self.banner_ids.clear()
        self.trail.clear()

    def _cell_box(self, r, c, inset=0):
        x1 = self.pad_x + c * self.cell_px + inset
        y1 = self.pad_y + r * self.cell_px + inset
        x2 = self.pad_x + (c + 1) * self.cell_px - inset
        y2 = self.pad_y + (r + 1) * self.cell_px - inset
        return x1, y1, x2, y2

    def draw_maze(self, grid, door_pos=None):
        self.clear()
        self.grid = grid
        self.size = int(grid.shape[0])
        self.door_pos = tuple(door_pos) if door_pos is not None else None
        self.door_open = False
        self.canvas.update_idletasks()
        w = max(self.canvas.winfo_width(), 200)
        h = max(self.canvas.winfo_height(), 200)
        usable = min(w, h) - 24
        self.cell_px = max(14, usable // self.size)
        board = self.cell_px * self.size
        self.pad_x = max(12, (w - board) // 2)
        self.pad_y = max(12, (h - board) // 2)

        self.canvas.configure(bg=BOARD_BG)
        # outer glow frame
        self.canvas.create_rectangle(
            self.pad_x - 10, self.pad_y - 10,
            self.pad_x + board + 10, self.pad_y + board + 10,
            fill=FRAME_FILL, outline=FRAME_EDGE, width=2)
        self.canvas.create_rectangle(
            self.pad_x - 4, self.pad_y - 4,
            self.pad_x + board + 4, self.pad_y + board + 4,
            fill='#0F1218', outline='#2A3344', width=1)

        for r in range(self.size):
            for c in range(self.size):
                ct = CellType(int(grid[r, c]))
                fill = CELL_FILL.get(ct, '#FFFFFF')
                if self.door_pos == (r, c) and self.door_open:
                    fill = DOOR_OPEN
                x1, y1, x2, y2 = self._cell_box(r, c)
                outline = GRID_LINE if ct == CellType.WALL else EMPTY_LINE
                rid = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=fill, outline=outline, width=1)
                self.cell_ids[(r, c)] = rid
                if ct == CellType.WALL:
                    self.canvas.create_line(
                        x1 + 1, y2 - 1, x2 - 1, y2 - 1, fill='#111111', width=1)
                    self.canvas.create_line(
                        x2 - 1, y1 + 1, x2 - 1, y2 - 1, fill='#111111', width=1)
                    self.canvas.create_line(
                        x1 + 1, y1 + 1, x2 - 2, y1 + 1, fill='#4A4A4A', width=1)
                if ct in CELL_LETTER:
                    txt, tc = CELL_LETTER[ct]
                    fs = max(9, self.cell_px // 2)
                    tid = self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text=txt, fill=tc,
                        font=('Segoe UI', fs, 'bold'))
                    self.letter_ids[(r, c)] = tid

    def set_door_state(self, open_: bool):
        self.door_open = bool(open_)
        if self.door_pos is None or self.door_pos not in self.cell_ids:
            return
        fill = DOOR_OPEN if self.door_open else DOOR_CLOSED
        self.canvas.itemconfig(self.cell_ids[self.door_pos], fill=fill)

    def add_trail(self, row, col):
        if self.trail and self.trail[-1] == (row, col):
            return
        self.trail.append((row, col))
        inset = max(3, self.cell_px // 3)
        x1, y1, x2, y2 = self._cell_box(row, col, inset=inset)
        tid = self.canvas.create_oval(
            x1, y1, x2, y2, fill=TRAIL, outline='', tags='trail')
        self.canvas.tag_lower(tid)
        self.trail_ids.append(tid)
        if len(self.trail_ids) > 70:
            old_id = self.trail_ids.pop(0)
            try:
                self.canvas.delete(old_id)
            except tk.TclError:
                pass

    def clear_trail(self):
        for tid in self.trail_ids:
            try:
                self.canvas.delete(tid)
            except tk.TclError:
                pass
        self.trail_ids.clear()
        self.trail.clear()

    def draw_agent(self, row, col, has_key=0):
        for iid in self.agent_ids:
            try:
                self.canvas.delete(iid)
            except tk.TclError:
                pass
        self.agent_ids.clear()
        self.add_trail(row, col)
        x1, y1, x2, y2 = self._cell_box(row, col, inset=max(2, self.cell_px // 6))
        fill = AGENT_HAS_KEY if has_key else AGENT_NO_KEY
        glow = self.canvas.create_oval(
            x1 - 3, y1 - 3, x2 + 3, y2 + 3, fill='', outline=fill, width=2)
        body = self.canvas.create_oval(
            x1, y1, x2, y2, fill=fill, outline='#FFFFFF', width=2)
        hx1 = x1 + (x2 - x1) * 0.22
        hy1 = y1 + (y2 - y1) * 0.18
        hx2 = x1 + (x2 - x1) * 0.52
        hy2 = y1 + (y2 - y1) * 0.45
        hi = self.canvas.create_oval(hx1, hy1, hx2, hy2, fill='#FFFFFF', outline='')
        self.agent_ids = [glow, body, hi]
        for iid in self.agent_ids:
            self.canvas.tag_raise(iid)

    def flash_cell(self, row, col, color='#FF4D4D', ms=180, app=None):
        if (row, col) not in self.cell_ids:
            return
        rid = self.cell_ids[(row, col)]
        old_fill = self.canvas.itemcget(rid, 'fill')
        old_outline = self.canvas.itemcget(rid, 'outline')
        self.canvas.itemconfig(rid, fill=color, outline='#FFFFFF', width=2)

        def restore():
            try:
                self.canvas.itemconfig(rid, fill=old_fill, outline=old_outline, width=1)
            except tk.TclError:
                pass

        if app is not None:
            app.after(ms, restore)

    def clear_policy_overlay(self):
        for iid in self.arrow_ids.values():
            try:
                self.canvas.delete(iid)
            except tk.TclError:
                pass
        self.arrow_ids.clear()

    def draw_policy_overlay(self, action_grid):
        self.clear_policy_overlay()
        if action_grid is None:
            return
        fs = max(11, int(self.cell_px * 0.55))
        for r in range(self.size):
            for c in range(self.size):
                a = int(action_grid[r, c])
                if a < 0:
                    continue
                x1, y1, x2, y2 = self._cell_box(r, c)
                tid = self.canvas.create_text(
                    (x1 + x2) / 2, (y1 + y2) / 2 + 1,
                    text=ARROW.get(a, ''), fill='#1E3A8A',
                    font=('Segoe UI', fs, 'bold'))
                self.arrow_ids[(r, c)] = tid
                self.canvas.tag_raise(tid)
        for iid in self.agent_ids:
            self.canvas.tag_raise(iid)

    def show_banner(self, text, tone='ok'):
        for iid in self.banner_ids:
            try:
                self.canvas.delete(iid)
            except tk.TclError:
                pass
        self.banner_ids.clear()
        fill = '#14532D' if tone == 'ok' else '#7F1D1D'
        accent = '#86EFAC' if tone == 'ok' else '#FECACA'
        x1, y1 = self.pad_x + 10, self.pad_y + 10
        bg = self.canvas.create_rectangle(
            x1, y1, x1 + 240, y1 + 36, fill=fill, outline=accent, width=2)
        tx = self.canvas.create_text(
            x1 + 14, y1 + 18, anchor='w', fill='#FFFFFF',
            text=text.upper(), font=('Segoe UI', 12, 'bold'))
        self.banner_ids = [bg, tx]
        for iid in self.banner_ids:
            self.canvas.tag_raise(iid)
