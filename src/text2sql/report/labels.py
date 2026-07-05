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
    "Ты делаешь КОРОТКИЕ бизнес-подписи колонок И определяешь ЕДИНИЦУ ИЗМЕРЕНИЯ числовых.\n"
    "Для каждой колонки верни label и unit.\n"
    "label — 2-5 слов по-русски, по-деловому, без техжаргона (unrealized_deal_potential → "
    "«нереализованный потенциал», src_task_as_code → «источник задач», gosb_name → «ГОСБ»). "
    "Не повторяй тех.имя, не пиши предложения.\n"
    "unit — вид величины:\n"
    " • money — ДЕНЬГИ (рубли/сумма/ФОТ/оборот/ЗП/стоимость/выручка). Ставь ТОЛЬКО когда УВЕРЕН;\n"
    " • people — количество ЛЮДЕЙ (физлица/клиенты/сотрудники/человек);\n"
    " • count — прочее количество/штуки/единицы;\n"
    " • percent — доля/процент;\n"
    " • other — не число (id, дата, текст, флаг).\n"
    "ВАЖНО: деньги и количество из имени/описания часто НЕОТЛИЧИМЫ. Если не уверен, что это "
    "именно ДЕНЬГИ — ставь count/people, НЕ money. «Физлица…», «Потенциал (кол-во)», "
    "«Количество…» — это НЕ деньги. Копейки в примерах — лишь слабый намёк.\n"
    'Верни JSON: {"cols": {"имя": {"label":"...","unit":"money|people|count|percent|other"}}}'
)

_UNIT_SIGN = {"money": "₽", "percent": "%", "people": "чел.", "count": "", "other": ""}


class Labels:
    """Колонка → короткая подпись + ЕДИНИЦА (money/people/count/percent/other)."""

    def __init__(self, mapping: dict[str, str] | None = None, units: dict[str, str] | None = None):
        self._m = {k: v.strip() for k, v in (mapping or {}).items() if v and v.strip()}
        self._u = {k: v for k, v in (units or {}).items() if v}

    def of(self, col: str) -> str:
        return self._m.get(col, col)

    def col_title(self, col: str) -> str:
        lab = self._m.get(col)
        return f"{lab} ({col})" if lab and lab.lower() != col.lower() else col

    def has(self, col: str) -> bool:
        return col in self._m

    def unit_kind(self, col: str) -> str | None:
        """Вид величины по мнению LLM (money/people/count/percent/other) или None."""
        return self._u.get(col)


def build_labels_llm(llm, table_desc: str, meta: dict[str, dict], columns: list[str],
                     df=None) -> Labels:
    """Короткие подписи + классификация единицы измерения (фокусный LLM-вызов).
    Примеры значений (копейки/целые) даём как подсказку. Fallback — обрезка описания."""
    lines = []
    for c in columns:
        desc = meta.get(c, {}).get("desc", "")
        ex = ""
        if df is not None and c in df.columns:
            vals = [str(v) for v in df[c].dropna().unique()[:3]]
            ex = f" | примеры: {', '.join(vals)}" if vals else ""
        lines.append(f"- {c}: {desc}{ex}")
    user = f"Таблица: {table_desc}\nКолонки (имя: описание | примеры):\n" + "\n".join(lines)
    mapping: dict[str, str] = {}
    units: dict[str, str] = {}
    try:
        out = llm.complete_json(_LABELS_SYS, user, max_tokens=6000, node="report_labels")
        for k, v in (out.get("cols") or {}).items():
            if isinstance(v, dict):
                if str(v.get("label", "")).strip():
                    mapping[k] = str(v["label"]).strip()
                if str(v.get("unit", "")).strip():
                    units[k] = str(v["unit"]).strip().lower()
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: подписи/единицы не сгенерированы (%s) — fallback описания", exc)
    for c in columns:                      # fallback подписи из описания
        if c not in mapping:
            desc = str(meta.get(c, {}).get("desc", "")).strip()
            if desc:
                short = desc.split(",")[0].split(".")[0].split("(")[0].strip()
                if 0 < len(short) <= 40:
                    mapping[c] = short
    logger.info("report: подписей=%d, единиц=%d из %d колонок", len(mapping), len(units), len(columns))
    return Labels(mapping, units)


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
