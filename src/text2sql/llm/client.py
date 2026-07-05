"""LLMClient — то, чем пользуются узлы графа. Оборачивает любой бэкенд
(DeepSeek/GigaChat) и добавляет строгий JSON-вывод с одним раундом ремонта.
Каждый вызов логируется в trace, если передан."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable

from ..config import LLM, LLMConfig
from .base import LLMBackend, LLMResult

logger = logging.getLogger(__name__)
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def make_backend(cfg: LLMConfig | None = None) -> LLMBackend:
    cfg = cfg or LLM
    if cfg.provider == "gigachat":
        from .gigachat import GigaChatBackend
        return GigaChatBackend(cfg)
    from .openai_compat import OpenAICompatBackend
    return OpenAICompatBackend(cfg)


class LLMClient:
    # Ретраи: сначала N_QUICK быстрых попыток (короткая пауза), затем N_SLOW «медленных»
    # с длинной паузой — на случай сбоев/пустых ответов (reasoning съел бюджет) и
    # временной недоступности/лимитов провайдера. Настраивается через env.
    N_QUICK = int(os.getenv("LLM_RETRY_QUICK", "4"))
    QUICK_DELAY = float(os.getenv("LLM_RETRY_QUICK_DELAY", "3"))
    N_SLOW = int(os.getenv("LLM_RETRY_SLOW", "3"))
    SLOW_DELAY = float(os.getenv("LLM_RETRY_SLOW_DELAY", "30"))

    def __init__(
        self,
        backend: LLMBackend | None = None,
        *,
        cfg: LLMConfig | None = None,
        tracer: Callable[[dict], None] | None = None,
    ):
        self.cfg = cfg or LLM
        self.backend = backend or make_backend(self.cfg)
        self._tracer = tracer

    def _delays(self) -> list[float]:
        # паузы ПЕРЕД повторной попыткой (первая попытка — без паузы)
        return [0.0] + [self.QUICK_DELAY] * self.N_QUICK + [self.SLOW_DELAY] * self.N_SLOW

    def complete(self, system: str, user: str, *, max_tokens: int | None = None,
                 temperature: float | None = None, node: str = "") -> LLMResult:
        last_exc: Exception | None = None
        delays = self._delays()
        for attempt, delay in enumerate(delays, 1):
            if delay:
                logger.warning("LLM[%s]: повтор %d/%d через %.0fс (%s)",
                               node or "-", attempt - 1, len(delays) - 1, delay, last_exc)
                time.sleep(delay)
            try:
                res = self.backend.chat(system, user, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:  # noqa: BLE001  (сеть/лимит/провайдер — повторяем)
                last_exc = exc
                continue
            if not (res.text and res.text.strip()):     # пустой ответ (напр. reasoning съел бюджет)
                last_exc = ValueError("пустой ответ LLM")
                continue
            if self._tracer:
                self._tracer({
                    "kind": "llm", "node": node, "model": self.backend.model,
                    "system": system, "user": user, "text": res.text,
                    "finish_reason": res.finish_reason, "usage": res.usage,
                    "reasoning_len": len(res.reasoning),
                })
            return res
        raise last_exc or RuntimeError("LLM: все попытки исчерпаны")

    def complete_json(self, system: str, user: str, *, max_tokens: int | None = None,
                      node: str = "") -> dict[str, Any]:
        sys_json = system.rstrip() + "\n\nОтвечай ТОЛЬКО валидным JSON-объектом, без markdown и пояснений."
        res = self.complete(sys_json, user, max_tokens=max_tokens, node=node)
        parsed = self._try_parse(res.text)
        if parsed is not None:
            return parsed
        repair = self.complete(
            sys_json, "Преобразуй в один валидный JSON-объект без markdown и текста:\n\n" + res.text,
            max_tokens=max_tokens, node=node + ":repair",
        )
        parsed = self._try_parse(repair.text)
        if parsed is None:
            raise ValueError(f"LLM не вернул валидный JSON после ремонта. Сырой ответ:\n{res.text[:500]}")
        return parsed

    @staticmethod
    def _try_parse(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^json\s*", "", candidate.strip("`"), flags=re.IGNORECASE).strip()
        for chunk in (candidate, _extract(candidate)):
            if not chunk:
                continue
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        return None


def _extract(text: str) -> str:
    m = _JSON_BLOCK.search(text)
    return m.group(0) if m else ""
