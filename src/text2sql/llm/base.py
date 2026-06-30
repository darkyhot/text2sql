"""Базовые типы LLM-слоя и контракт бэкенда.

Узлы графа работают только с LLMClient (complete / complete_json) и не знают,
какой провайдер под капотом: DeepSeek (OpenAI-совместимый) сейчас, GigaChat
на проде. Бэкенд реализует единственный метод chat()."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResult:
    text: str
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class LLMBackend(ABC):
    """Транспорт к конкретному провайдеру. Без JSON-логики (она в LLMClient)."""

    model: str

    @abstractmethod
    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        ...
