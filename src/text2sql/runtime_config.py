"""Персистентные runtime-настройки агента (config_runtime.json).

Сейчас хранит выбор LLM (provider+model) — чтобы /model сохранялся между
перезапусками и не откатывался к дефолту из .env."""

from __future__ import annotations

import json
from pathlib import Path

from .config import PROJECT_ROOT

RUNTIME_PATH = PROJECT_ROOT / "config_runtime.json"


def load_runtime(path: Path = RUNTIME_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_runtime(values: dict, path: Path = RUNTIME_PATH) -> None:
    current = load_runtime(path)
    current.update(values)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def load_llm_override() -> tuple[str, str] | None:
    """(provider, model) из конфига, либо None если не сохранён."""
    data = load_runtime()
    provider = str(data.get("llm_provider") or "").strip()
    model = str(data.get("llm_model") or "").strip()
    if provider and model:
        return provider, model
    return None


def save_llm_override(provider: str, model: str) -> None:
    save_runtime({"llm_provider": provider, "llm_model": model})
