"""Растущее поле ввода для ноутбука (работает и в VS Code, и в браузерном Jupyter).

`install()` подменяет builtin `input()` на многострочное поле, которое растёт вниз по мере
текста. Enter — отправить, Ctrl+Enter (или Shift+Enter) — перенос строки. Кнопки нет.

Реализация БЕЗ кастомного JS (VS Code не исполняет JS в выводах ячеек) — только на
виджет-событиях: ipyevents ловит нажатия клавиш, jupyter-ui-poll синхронно прокачивает
события ядра, пока `input()` блокирует kernel. Если библиотек/виджетов нет и доустановить
не удалось — тихо остаётся обычный input().
"""
from __future__ import annotations

import builtins
import subprocess
import sys

_MAX_ROWS = 25
_MIN_ROWS = 3
_WRAP = 90            # прибл. символов в строке при width=99% — для оценки высоты без DOM


def _ensure_deps() -> bool:
    """Доустанавливает нужные пакеты в текущий kernel при первом запуске."""
    import importlib
    ok = True
    for mod, pkg in (("ipywidgets", "ipywidgets"), ("ipyevents", "ipyevents"),
                     ("jupyter_ui_poll", "jupyter-ui-poll")):
        try:
            importlib.import_module(mod)
        except ImportError:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
                importlib.invalidate_caches()
                importlib.import_module(mod)
            except Exception:
                ok = False
    return ok


def _rows_for(text: str) -> int:
    """Оценка числа строк под текст (учитывает и переносы, и мягкий перенос длинных строк)."""
    n = sum(max(1, (len(line) // _WRAP) + 1) for line in text.split("\n"))
    return min(_MAX_ROWS, max(_MIN_ROWS, n))


def install() -> None:
    """Включить растущее поле ввода. Идемпотентно; при недоступности — no-op."""
    if getattr(builtins, "_t2s_growing_input", False):
        return
    if not _ensure_deps():
        return
    try:
        import time

        import ipywidgets as W
        from ipyevents import Event
        from IPython.display import display
        from jupyter_ui_poll import ui_events
    except Exception:
        return  # не Jupyter / нет виджетов

    _orig = builtins.input

    def growing_input(prompt: str = "") -> str:
        try:
            ta = W.Textarea(layout=W.Layout(width="99%"))
            ta.rows = _MIN_ROWS
            ta.observe(lambda c: setattr(ta, "rows", _rows_for(c["new"])), "value")
            head = W.HTML(f"<b>{_escape(prompt)}</b>") if str(prompt).strip() else W.HTML("")
            box = W.VBox([head, ta])

            st = {"done": False, "val": ""}

            def on_key(e):
                if e.get("key") != "Enter":
                    return
                if e.get("ctrlKey") or e.get("metaKey") or e.get("shiftKey"):
                    ta.value = ta.value + "\n"          # Ctrl/Shift+Enter → перенос
                else:
                    st["val"] = ta.value.rstrip("\n")   # Enter → отправить
                    st["done"] = True

            Event(source=ta, watched_events=["keydown"]).on_dom_event(on_key)

            display(box)
            with ui_events() as poll:                   # прокачиваем события, пока ждём Enter
                while not st["done"]:
                    poll(10)
                    time.sleep(0.03)
            box.close()
            print(f"{prompt}{st['val']}")               # эхо в транскрипт (виджет закрылся)
            return st["val"]
        except Exception:
            return _orig(prompt)

    builtins.input = growing_input
    builtins._t2s_growing_input = True


def _escape(s) -> str:
    import html
    return html.escape(str(s))
