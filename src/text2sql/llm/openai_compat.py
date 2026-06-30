"""OpenAI-совместимый бэкенд (DeepSeek сейчас; подходит любому OpenAI-API).

Осведомлён о reasoning-моделях: ход мысли уходит в reasoning_content, ответ —
в content; пустой content при finish_reason=length означает, что бюджет токенов
съело рассуждение — это ошибка, не молчим."""

from __future__ import annotations

from ..config import LLM, LLMConfig
from .base import LLMBackend, LLMResult


class OpenAICompatBackend(LLMBackend):
    def __init__(self, cfg: LLMConfig | None = None):
        from openai import OpenAI

        self.cfg = cfg or LLM
        self.model = self.cfg.model
        self._client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url)

    def chat(self, system, user, *, max_tokens=None, temperature=None) -> LLMResult:
        budget = max_tokens or self.cfg.max_tokens
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=budget,
            temperature=self.cfg.temperature if temperature is None else temperature,
        )
        choice = resp.choices[0]
        msg = choice.message
        reasoning = getattr(msg, "reasoning_content", "") or ""
        text = (msg.content or "").strip()
        finish = choice.finish_reason or ""
        if not text and finish == "length":
            raise RuntimeError(
                f"LLM пустой ответ: reasoning съел бюджет {budget} токенов. Увеличьте max_tokens."
            )
        return LLMResult(
            text=text, finish_reason=finish,
            usage=resp.usage.model_dump() if resp.usage else {}, reasoning=reasoning,
        )
