"""Авто-растущее поле ввода для ноутбука.

`install()` подменяет builtin `input()` на `Textarea`, которая растёт вниз по мере текста —
удобно для длинных промтов. Работает для ВСЕХ полей (главный `🟢 >` и вложенные под-промпты),
логику CLI трогать не нужно. Если это не Jupyter или нет виджетов — тихо остаётся обычный input().
"""
from __future__ import annotations

import builtins
import subprocess
import sys

_MAX_ROWS = 25


def _ensure_deps() -> bool:
    """Доустанавливает ipywidgets/nest_asyncio в текущий kernel при первом запуске."""
    import importlib
    for pkg in ("ipywidgets", "nest_asyncio"):
        try:
            importlib.import_module(pkg)
        except ImportError:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
            except Exception:
                return False
    return True


def _enable_ctrl_enter() -> None:
    """Ctrl+Enter = отправить (best-effort; кнопка «Отправить» работает всегда)."""
    from IPython.display import Javascript, display
    try:
        display(Javascript(
            "if(!window.__t2s_kd){window.__t2s_kd=1;"
            "document.addEventListener('keydown',function(e){"
            "if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){var el=document.activeElement;"
            "if(el&&el.tagName==='TEXTAREA'){var r=el.closest('.widget-vbox');"
            "var b=r&&r.querySelector('button');if(b){e.preventDefault();b.click();}}}});}"))
    except Exception:
        pass


def install() -> None:
    """Включить растущее поле ввода. Идемпотентно; при недоступности — no-op."""
    if getattr(builtins, "_t2s_growing_input", False):
        return
    if not _ensure_deps():
        return
    try:
        import asyncio
        import html

        import ipywidgets as W
        import nest_asyncio
        from IPython.display import display
        nest_asyncio.apply()
    except Exception:
        return  # не Jupyter / нет виджетов

    _enable_ctrl_enter()
    _orig = builtins.input

    def growing_input(prompt: str = "") -> str:
        try:
            ta = W.Textarea(layout=W.Layout(width="99%"))
            ta.rows = 1
            ta.observe(lambda c: setattr(ta, "rows", min(_MAX_ROWS, max(1, c["new"].count("\n") + 1))), "value")
            send = W.Button(description="Отправить  (Ctrl+Enter)", button_style="primary",
                            layout=W.Layout(width="auto"))
            head = W.HTML(f"<b>{html.escape(str(prompt))}</b>") if str(prompt).strip() else W.HTML("")
            box = W.VBox([head, ta, send])
            loop = asyncio.get_event_loop()
            fut = loop.create_future()
            send.on_click(lambda _: None if fut.done() else fut.set_result(ta.value))
            display(box)
            try:
                val = loop.run_until_complete(fut)
            finally:
                box.close()
            print(f"{prompt}{val}")               # эхо в транскрипт (виджет закрылся)
            return val
        except Exception:
            return _orig(prompt)

    builtins.input = growing_input
    builtins._t2s_growing_input = True
