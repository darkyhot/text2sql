"""Трассировка (observability с дня 1). Каждый узел графа и каждый вызов
инструмента/LLM пишет событие. Без трейсов многошаговый агент неулучшаем."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import PATHS


class Tracer:
    """Накопитель событий сессии + запись в jsonl."""

    def __init__(self, session_id: str | None = None, *, trace_dir: Path | None = None):
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S")
        self.events: list[dict[str, Any]] = []
        self._dir = trace_dir or PATHS.trace_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{self.session_id}.jsonl"

    def __call__(self, event: dict[str, Any]) -> None:
        event = {"ts": time.time(), **event}
        self.events.append(event)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def by_kind(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("kind") == kind]
