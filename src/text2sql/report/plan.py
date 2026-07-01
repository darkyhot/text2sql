"""Планирование отчёта (какие разрезы строить) и нарратив (бизнес-выводы).

Код генерирует ПУЛ валидных кандидатов-разрезов из ролей колонок; LLM только
ВЫБИРАЕТ самые интересные для бизнеса (с учётом фокуса пользователя) и пишет
выводы простым языком. Так надёжнее слабых моделей и не строится лишнего."""

from __future__ import annotations

import logging
from pathlib import Path

from . import core
from .core import AnalysisResult, Roles

logger = logging.getLogger(__name__)


def candidate_specs(roles: Roles) -> list[dict]:
    dims = roles.dimensions[:4]
    mets = roles.metrics[:4]
    date = roles.dates[0] if roles.dates else None
    specs: list[dict] = []
    for m in mets[:2]:
        for d in dims[:3]:
            specs.append({"kind": "top_n", "dim": d, "metric": m})
        if dims:
            specs.append({"kind": "concentration", "dim": dims[0], "metric": m})
        if date:
            specs.append({"kind": "trend", "metric": m, "date": date})
            if dims:
                specs.append({"kind": "period_compare", "dim": dims[0], "metric": m, "date": date})
    for f in roles.flags[:3]:
        if dims:
            specs.append({"kind": "flag_breakdown", "flag": f, "dim": dims[0]})
    # сущности (ФИО/ИНН/компания) — только рейтинги ТОП-N
    for e in roles.entities[:3]:
        specs.append({"kind": "top_n_count", "dim": e})
        if mets:
            specs.append({"kind": "top_n", "dim": e, "metric": mets[0]})
        if roles.flags:
            specs.append({"kind": "top_n", "dim": e, "metric": roles.flags[0]})
    # дедуп
    seen, out = set(), []
    for s in specs:
        key = (s["kind"], s.get("dim"), s.get("metric"), s.get("flag"), s.get("date"))
        if key not in seen:
            seen.add(key); out.append(s)
    return out


def _spec_label(s: dict) -> str:
    if s["kind"] == "top_n":
        return f"ТОП по «{s['dim']}» (метрика {s['metric']})"
    if s["kind"] == "top_n_count":
        return f"ТОП по «{s['dim']}» (по количеству записей)"
    if s["kind"] == "concentration":
        return f"Концентрация {s['metric']} по «{s['dim']}» (Парето)"
    if s["kind"] == "trend":
        return f"Динамика {s['metric']} во времени"
    if s["kind"] == "period_compare":
        return f"Что выросло/упало по «{s['dim']}» ({s['metric']})"
    if s["kind"] == "flag_breakdown":
        return f"Доля «{s['flag']}» по «{s['dim']}»"
    return s["kind"]


_SELECT_SYS = (
    "Ты бизнес-аналитик. Из списка кандидатов-разрезов выбери 4-6 САМЫХ ИНТЕРЕСНЫХ "
    "для бизнес-отчёта, который отвечает на вопрос «что интересного и полезного в этих "
    "данных» (а не «что это за таблица»). Учитывай фокус пользователя, если он задан. "
    "Не выбирай похожие дубли. Верни JSON: "
    '{"chosen":[номера], "angle":"о чём отчёт одной фразой"}'
)


def select_specs(llm, table_desc: str, focus: str, specs: list[dict], *, k: int = 6) -> tuple[list[dict], str]:
    if not specs:
        return [], ""
    listing = "\n".join(f"{i}. {_spec_label(s)}" for i, s in enumerate(specs))
    user = (f"Таблица: {table_desc}\n"
            f"Фокус пользователя: {focus or '(не задан — общий обзор)'}\n\n"
            f"Кандидаты:\n{listing}\n\nВыбери 4-6 самых полезных.")
    try:
        out = llm.complete_json(_SELECT_SYS, user, max_tokens=1500, node="report_plan")
        idx = [i for i in out.get("chosen", []) if isinstance(i, int) and 0 <= i < len(specs)]
        angle = str(out.get("angle", "")).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: LLM-выбор разрезов не удался (%s), беру дефолт", exc)
        idx, angle = [], ""
    if not idx:
        idx = _default_pick(specs, k)
    chosen = [specs[i] for i in idx[:k]]
    return chosen, angle


def _default_pick(specs: list[dict], k: int) -> list[int]:
    """Разнообразный дефолт: по одному от каждого вида, метрика — приоритетная."""
    order = ["trend", "top_n", "concentration", "period_compare", "flag_breakdown"]
    picked, used_kind = [], set()
    for kind in order:
        for i, s in enumerate(specs):
            if s["kind"] == kind and i not in picked:
                picked.append(i); used_kind.add(kind); break
    for i in range(len(specs)):
        if len(picked) >= k:
            break
        if i not in picked:
            picked.append(i)
    return picked[:k]


def run_spec(spec: dict, df, assets: Path) -> AnalysisResult | None:
    k = spec["kind"]
    try:
        if k == "top_n":
            return core.top_n(df, spec["dim"], spec["metric"], assets)
        if k == "top_n_count":
            return core.top_n_count(df, spec["dim"], assets)
        if k == "concentration":
            return core.concentration(df, spec["dim"], spec["metric"], assets)
        if k == "trend":
            return core.trend(df, spec["date"], spec["metric"], assets)
        if k == "period_compare":
            return core.period_compare(df, spec["date"], spec["dim"], spec["metric"], assets)
        if k == "flag_breakdown":
            return core.flag_breakdown(df, spec["flag"], spec["dim"], assets)
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: разрез %s не построен: %s", spec, exc)
    return None


_NARRATE_SYS = (
    "Ты пишешь бизнес-отчёт простым языком для руководителя (не аналитика). "
    "По цифрам каждой секции дай КОРОТКИЙ вывод: 1-2 предложения, по-русски, "
    "конкретно и полезно. ЗАПРЕЩЕНО: математический жаргон (дисперсия, z-score, "
    "корреляция, стандартное отклонение), вода и пересказ чисел без смысла. "
    "Сделай акцент на том, что ИНТЕРЕСНО и что с этим ДЕЛАТЬ. "
    "Верни JSON: {\"summary\":[3-5 пунктов главного], "
    "\"insights\":{\"<ключ секции>\":\"вывод\"}, \"attention\":[1-3 пункта на что обратить внимание]}"
)


def narrate(llm, table_desc: str, focus: str, results: list[AnalysisResult]) -> tuple[list[str], list[str]]:
    sections = [{"key": r.key, "title": r.title, "facts": r.facts} for r in results if r.facts]
    user = (f"Таблица: {table_desc}\nФокус: {focus or '(общий обзор)'}\n\n"
            f"Секции с посчитанными цифрами:\n{sections}")
    try:
        out = llm.complete_json(_NARRATE_SYS, user, max_tokens=3000, node="report_narrate")
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: нарратив не сгенерирован: %s", exc)
        return [], []
    insights = out.get("insights", {}) or {}
    for r in results:
        r.insight = str(insights.get(r.key, "")).strip()
    summary = [str(x).strip() for x in (out.get("summary") or []) if str(x).strip()]
    attention = [str(x).strip() for x in (out.get("attention") or []) if str(x).strip()]
    return summary, attention
