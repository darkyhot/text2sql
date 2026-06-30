"""GigaChat-бэкенд для прода (model=Gigachat-3-ultra).

Переносит семантику боевого RateLimitedLLM: глобальный (на уровень класса)
rate-limit, retry, повтор при finish_reason=blacklist (контент-фильтр GigaChat
флакает). langchain_gigachat импортируется лениво — локально может быть не
установлен. ENV: GIGACHAT_API_URL, JPY_API_TOKEN, GIGACHAT_MODEL.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict

from ..config import LLM, LLMConfig
from .base import LLMBackend, LLMResult

logger = logging.getLogger(__name__)


class GigaChatBackend(LLMBackend):
    MIN_INTERVAL: float = 5.0
    MAX_RETRIES: int = 5
    MAX_CACHED_TEMPERATURES: int = 3
    _RETRYABLE_FINISH_REASONS = frozenset({"blacklist"})

    _global_last_call_time: float = 0.0
    _global_lock = threading.Lock()

    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLM
        self.model = os.getenv("GIGACHAT_MODEL", self.cfg.model or "Gigachat-3-ultra")
        self._base_url = os.getenv("GIGACHAT_API_URL")
        self._access_token = os.getenv("JPY_API_TOKEN")
        self._default = self._make(None)
        self._cache: "OrderedDict[float, object]" = OrderedDict()

    def _make(self, temperature: float | None):
        from langchain_gigachat.chat_models import GigaChat  # ленивый импорт

        kwargs = dict(base_url=self._base_url, access_token=self._access_token,
                      model=self.model, timeout=120)
        if temperature is not None:
            kwargs["temperature"] = temperature
        return GigaChat(**kwargs)

    def _get(self, temperature: float | None):
        if temperature is None:
            return self._default
        if temperature in self._cache:
            self._cache.move_to_end(temperature)
            return self._cache[temperature]
        llm = self._make(temperature)
        self._cache[temperature] = llm
        while len(self._cache) > self.MAX_CACHED_TEMPERATURES:
            self._cache.popitem(last=False)
        return llm

    def _wait(self) -> None:
        with GigaChatBackend._global_lock:
            elapsed = time.time() - GigaChatBackend._global_last_call_time
            if elapsed < self.MIN_INTERVAL:
                time.sleep(self.MIN_INTERVAL - elapsed)

    def chat(self, system, user, *, max_tokens=None, temperature=None) -> LLMResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        llm = self._get(temperature)
        for attempt in range(1, self.MAX_RETRIES + 1):
            self._wait()
            try:
                GigaChatBackend._global_last_call_time = time.time()
                resp = llm.invoke(messages)
                meta = getattr(resp, "response_metadata", None) or {}
                finish = str(meta.get("finish_reason", "")).lower()
                if finish in self._RETRYABLE_FINISH_REASONS and attempt < self.MAX_RETRIES:
                    logger.warning("GigaChat blacklist, попытка %d/%d", attempt, self.MAX_RETRIES)
                    time.sleep(self.MIN_INTERVAL)
                    continue
                return LLMResult(text=str(resp.content).strip(), finish_reason=finish)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GigaChat ошибка (%d/%d): %s", attempt, self.MAX_RETRIES, exc)
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.MIN_INTERVAL)
        raise RuntimeError(f"GigaChat не ответил после {self.MAX_RETRIES} попыток.")
