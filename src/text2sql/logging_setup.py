"""Файловое логирование для разбора инцидентов (особенно с ПРОМа).

Пишет всё в workspace/agent.log (DEBUG), без консольного шума (в Jupyter мешает).
Лог-файл удобно прислать для отладки: в нём поток узлов графа, исполняемый SQL,
ответы/ошибки LLM и трейсбэки."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import PATHS

_CONFIGURED = False


def setup_logging(*, level: int = logging.DEBUG) -> Path:
    """Настроить корневой логгер на файл workspace/agent.log. Идемпотентно."""
    global _CONFIGURED
    PATHS.workspace_dir.mkdir(parents=True, exist_ok=True)
    log_file = PATHS.workspace_dir / "agent.log"
    if _CONFIGURED:
        return log_file

    root = logging.getLogger()
    root.setLevel(level)
    # Снимаем чужие хендлеры (Jupyter ставит свои), чтобы не дублировать/не шуметь.
    root.handlers.clear()

    handler = logging.FileHandler(str(log_file), encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root.addHandler(handler)
    # Глушим болтливые сторонние логгеры (иначе лог тонет в DEBUG транспорта).
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "sqlalchemy.engine",
                  "pydot", "pydot.core", "pydot.dot_parser", "langchain",
                  "langchain_gigachat", "gigachat", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).info("Логирование инициализировано → %s", log_file)
    return log_file
