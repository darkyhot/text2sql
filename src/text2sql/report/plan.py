"""Планирование отчёта (какие разрезы строить) и нарратив (бизнес-выводы).

Код генерирует ПУЛ валидных кандидатов-разрезов из ролей колонок; LLM только
ВЫБИРАЕТ самые интересные для бизнеса (с учётом фокуса пользователя) и пишет
выводы простым языком. Так надёжнее слабых моделей и не строится лишнего."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from . import core
from .core import AnalysisResult, Roles

logger = logging.getLogger(__name__)


_METRIC_SYS = (
    "Ты бизнес-аналитик. По ОПИСАНИЮ таблицы выбери ГЛАВНЫЕ количественные метрики "
    "для бизнес-отчёта — те, что отражают суть таблицы. Пример: таблица про отток → "
    "главная метрика это кол-во/сумма оттока, а НЕ технические/расчётные/промежуточные "
    "поля. Если таблица — витрина разных метрик, верни несколько (2-4) самых значимых, "
    "по важности. Верни JSON: {\"metrics\":[имена_колонок по важности]}"
)


def select_primary_metrics(llm, table_desc: str, roles: Roles) -> list[str]:
    """LLM ранжирует метрики по смыслу таблицы (описания колонок). Возвращает
    переупорядоченный список: главные впереди."""
    if len(roles.metrics) <= 1:
        return roles.metrics
    lst = "\n".join(f"- {m}: {roles.meta.get(m, {}).get('desc', '')}" for m in roles.metrics)
    user = f"Таблица: {table_desc}\nМетрики (колонка: описание):\n{lst}\n\nВыбери главные по важности."
    try:
        out = llm.complete_json(_METRIC_SYS, user, max_tokens=800, node="report_metrics")
        chosen = [m for m in out.get("metrics", []) if m in roles.metrics]
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: выбор главной метрики не удался: %s", exc)
        chosen = []
    if not chosen:
        return roles.metrics
    return chosen + [m for m in roles.metrics if m not in chosen]


_PROFILE_SYS = (
    "Ты профилируешь таблицу для БИЗНЕС-отчёта. Каждой колонке назначь РОЛЬ:\n"
    "- metric — числовая мера для агрегации (суммы, количества, деньги, проценты, потенциал);\n"
    "- dimension — категориальный разрез с НЕБОЛЬШИМ числом значений (сегмент, тип, статус, "
    "код источника, регион, категория);\n"
    "- entity — бизнес-сущность для рейтингов ТОП (клиент, ИНН, компания, ФИО сотрудника/"
    "руководителя; обычно высокая кардинальность);\n"
    "- date — дата для анализа динамики;\n"
    "- flag — булев признак да/нет;\n"
    "- ignore — тех.поля и id БЕЗ бизнес-смысла, системные метки времени (*_dttm, inserted/"
    "modified/updated), логины, служебные коды, большие свободные тексты (комментарии/анкеты) — "
    "их НЕ анализируем.\n"
    "Правила: если в таблице есть И id, И НАЗВАНИЕ ОДНОГО объекта (gosb_id и gosb_name) — "
    "выбери НАЗВАНИЕ (dimension/entity), а парный id → ignore (не дублируем). НО если у "
    "территориального id (tb_id, gosb_id, region_id) НЕТ парной колонки-названия в этой "
    "таблице — это ВАЖНЫЙ разрез (идентификатор территории/ГОСБ/ТБ), помечай его dimension. "
    "Отчётную дату (report_dt) укажи как primary_date, а не дату создания/открытия счёта. "
    "primary_metrics — 1-4 ГЛАВНЫЕ метрики по сути таблицы.\n"
    'Верни JSON: {"roles":{"колонка":"роль"}, "primary_date":"колонка"|null, '
    '"primary_metrics":["по важности"]}'
)


def build_roles_llm(llm, table_desc: str, df, meta: dict[str, dict]):
    """LLM-first профилирование колонок отчёта (роли + главная дата + главные метрики).
    Возвращает Roles или None (тогда builder падёт на regex-fallback core.profile)."""
    facts = core.column_facts(df, meta)
    lines = []
    for f in facts:
        ex = ", ".join(f["samples"][:4])
        lines.append(f"- {f['col']} | {f['dtype']} | уник={f['card']} | пусто={f['null_pct']}% | "
                     f"{f['desc']} | примеры: {ex}")
    user = (f"Таблица: {table_desc}\nКолонки [имя | тип | уник | пусто | описание | примеры]:\n"
            + "\n".join(lines) + "\n\nНазначь роль каждой колонке.")
    try:
        out = llm.complete_json(_PROFILE_SYS, user, max_tokens=8000, node="report_profile")
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: LLM-профайлер не сработал (%s) — regex-fallback", exc)
        return None
    assignment = {k: str(v).strip().lower() for k, v in (out.get("roles") or {}).items()}
    if not assignment:
        return None
    primary_date = out.get("primary_date") or None
    primary_metrics = [m for m in (out.get("primary_metrics") or []) if isinstance(m, str)]
    roles = core.roles_from_assignment(df, meta, assignment, primary_date, primary_metrics)
    logger.info("report LLM-профиль: metrics=%s dims=%s entities=%s dates=%s flags=%s",
                roles.metrics[:4], roles.dimensions[:5], roles.entities, roles.dates[:2], roles.flags[:4])
    return roles if roles.metrics else None


def candidate_specs(roles: Roles) -> list[dict]:
    dims = roles.dimensions[:4]
    mets = roles.metrics[:4]                # уже переупорядочены select_primary_metrics
    date = roles.dates[0] if roles.dates else None
    specs: list[dict] = []
    for m in mets[:3]:                      # динамика — по топ-3 метрикам (для витрин метрик)
        if date:
            specs.append({"kind": "trend", "metric": m, "date": date})
    for m in mets[:2]:                      # тяжёлые разрезы — по топ-2 главным метрикам
        for d in dims[:3]:
            specs.append({"kind": "top_n", "dim": d, "metric": m})
        # концентрация (Парето) — ТОЛЬКО для многокатегорийных разрезов (>15),
        # иначе она дублирует ТОП-N (как было для segment_name)
        conc_dim = next((d for d in dims if roles.card.get(d, 0) > 15), None)
        if conc_dim:
            specs.append({"kind": "concentration", "dim": conc_dim, "metric": m})
        if date and dims:
            specs.append({"kind": "period_compare", "dim": dims[0], "metric": m, "date": date})
    for f in roles.flags[:3]:
        for d in dims[:2]:                  # доля флага по 1-2 разрезам (в т.ч. по коду источника)
            specs.append({"kind": "flag_breakdown", "flag": f, "dim": d})
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
    "конкретно и полезно. Отклонения объясняй в РАЗАХ и ДОЛЯХ («вдвое хуже среднего», "
    "«даёт 40% всех потерь»), а НЕ в статжаргоне. Где есть и деньги, и количество — "
    "подчёркивай ДЕНЬГИ как главный эффект, но упоминай и людей/штуки. Для аномальных "
    "срезов и перекосов деньги/количество делай акцент, что с этим ДЕЛАТЬ (куда смотреть). "
    "ЗАПРЕЩЕНО: математический жаргон (дисперсия, z-score, корреляция, стандартное "
    "отклонение, квантиль), вода и пересказ чисел без смысла. "
    "Ключи секций (facts._kind): rate_dev — срез отклоняется от среднего; interaction — "
    "аномальное СОЧЕТАНИЕ двух разрезов (сильнее, чем ждали); money_conc — концентрация "
    "значимости; value_mismatch — мал по количеству, но крупный по деньгам/объёму. "
    "ВАЖНО: называй показатель «деньгами»/«рублями»/«₽» ТОЛЬКО если facts.is_money=true. "
    "Если is_money отсутствует/false — это НЕ деньги (люди, штуки, потенциал): пиши в "
    "единицах показателя (weight_name), без знака рубля. "
    "Верни JSON: {\"summary\":[3-5 пунктов главного], "
    "\"insights\":{\"<ключ секции>\":\"вывод\"}, \"attention\":[1-3 пункта на что обратить внимание]}"
)


_FOCUS_SYS = (
    "Пользователь заказал бизнес-отчёт с КОНКРЕТНЫМ запросом. Разложи запрос на набор "
    "разбивок, которые ПРЯМО на него отвечают. Каждая разбивка = показатель × разрез × агрегат:\n"
    "- measure: имя показателя из списка ИЛИ \"__count__\" — количество САМИХ записей/сущностей "
    "(бери его, когда спрашивают «сколько», «какие», распределение/состав по разрезу);\n"
    "- dim: имя разреза из списка;\n"
    "- agg: \"count\" (сколько записей), \"avg\" (среднее значение показателя), \"sum\" (сумма/итого).\n"
    "Пример: «средний потенциал по территории» → {measure: потенциал, dim: территория, agg: avg}; "
    "«источник и тип задач» → две разбивки {__count__, источник, count} и {__count__, тип, count}.\n"
    "Бери 3-8 самых точных под запрос. Только имена из списков.\n"
    'Верни JSON: {"question":"суть запроса одной фразой","breakdowns":[{"measure":..,"dim":..,"agg":..}]}'
)


def focus_plan_llm(llm, focus: str, measures, dims: list[str], labels) -> dict:
    """Свободный текст запроса → конкретные разбивки (показатель×разрез×агрегат), которыми
    отчёт прямо ответит. Валидируется по спискам мер/разрезов."""
    m_list = "\n".join(f"- {m.name}: {m.label} [{m.kind}]" for m in measures)
    m_list += "\n- __count__: количество самих записей/сущностей"
    d_list = "\n".join(f"- {d}: {labels.of(d)}" for d in dims)
    user = f"Запрос пользователя: {focus}\n\nПоказатели:\n{m_list}\n\nРазрезы:\n{d_list}"
    try:
        out = llm.complete_json(_FOCUS_SYS, user, max_tokens=2500, node="report_focus")
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: фокус-план не построен: %s", exc)
        return {"question": focus, "breakdowns": []}
    m_names = {m.name for m in measures} | {"__count__"}
    dset = set(dims)
    bd = []
    for b in (out.get("breakdowns") or []):
        if not isinstance(b, dict):
            continue
        mn, dn = b.get("measure"), b.get("dim")
        ag = str(b.get("agg") or "count").lower()
        if mn in m_names and dn in dset and ag in ("count", "avg", "sum"):
            bd.append({"measure": mn, "dim": dn, "agg": ag})
    return {"question": str(out.get("question") or focus).strip(), "breakdowns": bd}


# Детерминированная страховка: даже с прямым запретом в промпте reasoning-модель
# изредка срывается в матжаргон («максимальная дисперсия»). Заменяем на простой русский —
# инвариант «без матжаргона» держится независимо от капризов модели.
_DEJARGON: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bz[-\s]?score\w*", re.I), "отклонение"),
    (re.compile(r"\bp[-\s]?value\w*", re.I), "значимость"),
    (re.compile(r"стандартн\w*\s+отклонени\w*", re.I), "разброс"),
    (re.compile(r"дисперси\w*", re.I), "вариативность"),   # жен. род — согласуется («максимальная …»)
    (re.compile(r"корреляци\w*", re.I), "взаимосвязь"),
    (re.compile(r"коррелир\w*", re.I), "связан"),
    (re.compile(r"(кванти|перценти)л\w*", re.I), "уровень"),
]


def _dejargon(text: str) -> str:
    for rx, repl in _DEJARGON:
        text = rx.sub(repl, text)
    return text


def narrate(llm, table_desc: str, focus: str, results: list[AnalysisResult]) -> tuple[list[str], list[str]]:
    # короткие id (s0, s1…) — надёжнее длинных ключей для слабых моделей.
    # Ограничиваем число секций в промпте: у больших отчётов reasoning-модель иначе
    # съедает бюджет и возвращает пустой ответ (нет summary → нет «Главного»).
    rich = [r for r in results if r.facts][:20]
    ids = {f"s{i}": r for i, r in enumerate(rich)}
    # компактные факты: только осмысленные для вывода поля (без служебных _kind/_line)
    def _slim(facts: dict) -> dict:
        return {k: v for k, v in facts.items()
                if k not in ("_line", "_kind", "_section") and v not in (None, "")}
    sections = [{"id": sid, "title": r.title, "section": r.facts.get("_section", ""),
                 "facts": _slim(r.facts)} for sid, r in ids.items()]
    user = (f"Таблица: {table_desc}\nФокус: {focus or '(общий обзор)'}\n\n"
            f"Секции (id, раздел отчёта, что посчитано):\n{sections}\n\n"
            "В insights ключами используй id секций (s0, s1, …). В каждом пункте summary, "
            "где это уместно, СОШЛИСЬ на раздел-источник факта по его названию в скобках "
            "(поле section), напр. «…(раздел ⚖️ Ценность важнее количества)». Так руководитель "
            "видит, куда смотреть в отчёте.")
    try:
        # reasoning-модели (DeepSeek) тратят бюджет на размышление — даём с запасом
        out = llm.complete_json(_NARRATE_SYS, user, max_tokens=8000, node="report_narrate")
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: нарратив не сгенерирован: %s", exc)
        return [], []
    insights = out.get("insights", {}) or {}
    for sid, r in ids.items():
        r.insight = _dejargon(str(insights.get(sid, "")).strip())
    summary = [_dejargon(str(x).strip()) for x in (out.get("summary") or []) if str(x).strip()]
    attention = [_dejargon(str(x).strip()) for x in (out.get("attention") or []) if str(x).strip()]
    return summary, attention
