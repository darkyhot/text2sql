"""Человеческие подписи колонок и форматирование значений разрезов.

Заголовки/подписи в отчёте должны быть на бизнес-языке: не «unrealized_deal_potential
по src_task_as_code», а «Нереализованный потенциал (unrealized_deal_potential) по
источнику задач (src_task_as_code)». Короткие подписи делает LLM из описаний колонок
(фокусный вызов), fallback — обрезанное описание, потом имя.

fmt_val — форматирование ЗНАЧЕНИЙ разрезов/сущностей: целое как целое (9038, не 9038.0),
NaN → «—», длинные строки обрезаются. Числовые id больше не «плывут» в float."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


_LABELS_SYS = (
    "Ты делаешь КОРОТКИЕ бизнес-подписи для колонок таблицы (для заголовков графиков "
    "отчёта). По имени и описанию каждой колонки дай подпись 2-5 слов на русском, "
    "по-деловому и без техжаргона (напр. unrealized_deal_potential → «нереализованный "
    "потенциал», src_task_as_code → «источник задач», gosb_name → «ГОСБ»). "
    "Не повторяй техническое имя, не пиши целые предложения. "
    'Верни JSON: {"labels": {"имя_колонки": "короткая подпись", ...}} для ВСЕХ колонок.'
)


class Labels:
    """Отображение колонка → короткая бизнес-подпись + сборка заголовков «Подпись (имя)»."""

    def __init__(self, mapping: dict[str, str] | None = None):
        self._m = {k: v.strip() for k, v in (mapping or {}).items() if v and v.strip()}

    def of(self, col: str) -> str:
        """Короткая подпись колонки (или само имя, если подписи нет)."""
        return self._m.get(col, col)

    def col_title(self, col: str) -> str:
        """«Подпись (имя_колонки)» — если подпись есть и отличается; иначе просто имя."""
        lab = self._m.get(col)
        return f"{lab} ({col})" if lab and lab.lower() != col.lower() else col

    def has(self, col: str) -> bool:
        return col in self._m


def build_labels_llm(llm, table_desc: str, meta: dict[str, dict], columns: list[str]) -> Labels:
    """Короткие бизнес-подписи колонок (фокусный LLM-вызов). Fallback — обрезка описания."""
    lines = [f"- {c}: {meta.get(c, {}).get('desc', '')}" for c in columns]
    user = f"Таблица: {table_desc}\nКолонки (имя: описание):\n" + "\n".join(lines)
    mapping: dict[str, str] = {}
    try:
        out = llm.complete_json(_LABELS_SYS, user, max_tokens=4000, node="report_labels")
        mapping = {k: str(v) for k, v in (out.get("labels") or {}).items()
                   if isinstance(v, str) and v.strip()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: подписи колонок не сгенерированы (%s) — fallback описания", exc)
    # fallback: обрезанное описание для колонок без подписи
    for c in columns:
        if c not in mapping:
            desc = str(meta.get(c, {}).get("desc", "")).strip()
            if desc:
                short = desc.split(",")[0].split(".")[0].split("(")[0].strip()
                if 0 < len(short) <= 40:
                    mapping[c] = short
    logger.info("report: подписей колонок: %d/%d", len(mapping), len(columns))
    return Labels(mapping)


def fmt_val(v) -> str:
    """Значение разреза/сущности для показа: целое как целое, NaN → «—», обрезка длинных."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (int,)):
        return str(v)
    # float-целое, пришедшее строкой/Decimal
    try:
        f = float(v)
        if f.is_integer() and not isinstance(v, str):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    s = str(v)
    return s if len(s) <= 42 else s[:40] + "…"
