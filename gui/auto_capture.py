import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui.app import App
from gui import gui_verify

OUT_DIR = Path('results/figures/gui_screenshots')


def _pump(app, seconds):
    """Process tkinter events for `seconds` of real wall-clock time so
    the app's own self.after(...)-scheduled animation ticks get a chance
    to fire."""
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.03)


def main():
    gui_verify.OUT_DIR = OUT_DIR
    app = App()
    app.geometry('1320x900')
    app.update()
    _pump(app, 0.5)

    try:
        app._on_algo('VI')
        app._on_mode('eval')
        app.start_run()
        _pump(app, 1.5)
        gui_verify.snapshot(app, '01_vi_eval_running')

        app.policy_switch.select()
        app._on_policy_toggle()
        _pump(app, 0.5)
        gui_verify.snapshot(app, '02_vi_policy_overlay')

        app._on_algo('Q-Learning')
        app.start_run()
        _pump(app, 1.5)
        gui_verify.snapshot(app, '03_q_learning_eval_running')

        app._on_algo('SARSA(L)')
        app.start_run()
        _pump(app, 1.5)
        gui_verify.snapshot(app, '04_sarsa_eval_running')

        print(f'[auto_capture] 4 screenshots saved under {OUT_DIR}/')
    finally:
        app.stop_run()
        app.destroy()


if __name__ == '__main__':
    main()
