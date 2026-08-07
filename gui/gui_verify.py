"""
gui_verify.py — snapshot / widget manifest / GIF helpers (Windows + Pillow).
"""
import time
from pathlib import Path

from PIL import ImageGrab

OUT_DIR = Path('results/figures/gui_checks')


def _bbox(window):
    window.update_idletasks()
    x = window.winfo_rootx()
    y = window.winfo_rooty()
    w = window.winfo_width()
    h = window.winfo_height()
    return (x, y, x + w, y + h)


def snapshot(window, name='snapshot'):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'{name}.png'
    img = ImageGrab.grab(bbox=_bbox(window))
    img.save(out_path)
    print(f'[gui_verify] {out_path}  {img.size[0]}x{img.size[1]}')
    return out_path


def _widget_label(w):
    try:
        txt = w.cget('text')
        if txt:
            return str(txt)
    except Exception:
        pass
    try:
        var_name = w.cget('textvariable')
        if var_name:
            return str(w.tk.globalgetvar(var_name))
    except Exception:
        pass
    return ''


def widget_manifest(window, _indent=0, _lines=None):
    top = _lines is None
    lines = [] if _lines is None else _lines
    for child in window.winfo_children():
        note = ''
        if child.winfo_class() == 'Canvas':
            note = f'  [{len(child.find_all())} canvas items]'
        lines.append(f"{'  ' * _indent}{child.winfo_class()} {_widget_label(child)!r}{note}")
        widget_manifest(child, _indent + 1, lines)
    if top:
        text = '\n'.join(lines)
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', 'replace').decode('ascii'))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / 'widget_manifest.txt').write_text(text, encoding='utf-8')
        return text


def record_run(window, name='run', n_frames=60, interval_ms=150):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, bbox = [], _bbox(window)
    for _ in range(n_frames):
        frames.append(ImageGrab.grab(bbox=bbox))
        window.update()
        time.sleep(interval_ms / 1000)
    out_path = OUT_DIR / f'{name}.gif'
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                    duration=interval_ms, loop=0)
    print(f'[gui_verify] {out_path}  {len(frames)} frames @ {interval_ms}ms')
    return out_path
