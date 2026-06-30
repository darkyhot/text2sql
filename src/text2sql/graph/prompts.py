"""Промпты узлов. Принцип под слабые модели: маленькие, одна задача, явная
схема ответа, компактный контекст. Всё на русском (язык вопросов)."""

from __future__ import annotations

from typing import Any

from ..catalog.catalog import TableMeta

# ---------- SCOPE ----------

SCOPE_SYS = (
    "Ты выбираешь таблицы БД для ответа на вопрос аналитика.\n"
    "Рассуждай по шагам:\n"
    "1) Что именно считаем/измеряем (главное существительное) и какие у него уточнения.\n"
    "2) Для КАЖДОГО кандидата проверь: может ли он ответить на вопрос?\n"
    "3) Если ответить могут ДВЕ И БОЛЕЕ таблицы, но с РАЗНЫМ смыслом результата — "
    "это неоднозначность: верни её опции, НЕ выбирай молча.\n"
    "Явный признак неоднозначности: понятие из вопроса встречается и как отдельная "
    "таблица-сущность, и как атрибут/значение/флаг/подтип в ДРУГОЙ таблице — тогда "
    "и подсчёт по таблице-сущности, и фильтр по атрибуту равноправны, покажи оба.\n"
    "Если неоднозначности нет — верни минимальный набор таблиц и ambiguity=null.\n"
    'Схема ответа: {"chosen_tables":[fqn,...],'
    '"ambiguity":null | {"question":str,"options":[{"label":str,"tables":[fqn],"rationale":str}]}}'
)


def scope_user(question: str, corrections: list[str], candidates: list[dict]) -> str:
    lines = [f"Вопрос: {question}"]
    if corrections:
        lines.append("Правки пользователя: " + " | ".join(corrections))
    lines.append("\nКандидаты (таблица — роль/грануляция — описание):")
    for c in candidates:
        lines.append(f"- {c['fqn']} — {c.get('role','')}/{c.get('grain','')} — {c.get('description','')}")
    lines.append("\nВыбери таблицы и при наличии — опиши неоднозначность.")
    return "\n".join(lines)


# ---------- PLAN ----------

PLAN_SYS = (
    "Ты строишь СТРУКТУРИРОВАННЫЙ план SQL-запроса (форма select-join-aggregate). "
    "Используй только перечисленные таблицы и колонки. Правила:\n"
    "- Аналитическая БД: НЕЛЬЗЯ размножать строки. При join к справочнику соединяй "
    "по его ПОЛНОМУ составному ключу (PK-гипотеза). Маппинг колонок бери из "
    "join-кандидатов по максимальному overlap значений.\n"
    "- Для агрегатов укажи metrics и group_by; неагрегированные колонки в SELECT "
    "обязаны быть в group_by.\n"
    "- Период по дате задавай ПОЛУОТКРЫТЫМ интервалом двумя фильтрами: "
    "col op '>=' value 'YYYY-MM-01' И col op '<' value '<первое-число-следующего-месяца>'. "
    "НЕ используй BETWEEN с последним днём месяца (ошибки вида 29 февраля, потеря "
    "последнего дня для timestamp). Пример «февраль 2026»: >= '2026-02-01' и < '2026-03-01'.\n"
    "- Булевы фильтры: op 'IS TRUE'/'IS FALSE'.\n"
    "Схема ответа (StructuredPlan):\n"
    '{"intent":str,'
    '"tables":[{"ref":fqn,"alias":str}],'
    '"joins":[{"left_alias":str,"right_alias":str,"on":[[lcol,rcol]]}],'
    '"projections":[{"column":"alias.col"}],'
    '"metrics":[{"agg":"count|count_distinct|sum|avg|min|max|none","column":"alias.col|*","alias":str}],'
    '"filters":[{"column":"alias.col","op":str,"value":any,"value2":any}],'
    '"group_by":["alias.col"],"order_by":["alias.col"],"limit":int|null,'
    '"grain_note":str,"assumptions":[str],"ambiguities":[str]}'
)


def plan_user(question: str, corrections: list[str], tables: list[TableMeta],
              join_candidates: list[dict]) -> str:
    lines = [f"Вопрос: {question}"]
    if corrections:
        lines.append("Правки пользователя (учти их): " + " | ".join(corrections))
    for t in tables:
        lines.append(f"\nТаблица {t.fqn} (alias предложи сам). PK-гипотеза: {t.pk_hypothesis or '—'}")
        lines.append("Колонки [имя | тип | класс | pk | описание]:")
        for c in t.columns:
            pk = "PK" if c.is_pk_hypothesis else ""
            lines.append(f"  {c.name} | {c.dtype} | {c.semantic_class} | {pk} | {c.description}")
    if join_candidates:
        lines.append("\nJoin-кандидаты (overlap значений — чем больше, тем вернее маппинг):")
        for jc in join_candidates:
            pairs = ", ".join(f"{p['left_col']}↔{p['right_col']}(ov={p['overlap']})" for p in jc["pairs"])
            lines.append(f"- {jc['left']} ↔ {jc['right']}: {pairs}")
    lines.append("\nПострой план. Помни про запрет размножения строк.")
    return "\n".join(lines)


# ---------- JOIN FIX ----------

JOIN_FIX_SYS = (
    "Join в плане размножает строки (N:M): ключ не уникален ни на одной стороне. "
    "Исправь ключ join — добавь колонки, чтобы он покрывал полный составной ключ "
    "справочной стороны (см. PK-гипотезу и join-кандидаты). Верни ТОЛЬКО обновлённый "
    'JSON joins: {"joins":[{"left_alias":str,"right_alias":str,"on":[[lcol,rcol]]}]}'
)


def join_fix_user(plan: dict, verdict_note: str, join_candidates: list[dict],
                  pk_by_alias: dict[str, list[str]]) -> str:
    lines = [f"Текущие joins: {plan.get('joins')}", f"Вердикт: {verdict_note}",
             f"PK справочников по alias: {pk_by_alias}"]
    if join_candidates:
        lines.append("Join-кандидаты:")
        for jc in join_candidates:
            pairs = ", ".join(f"{p['left_col']}↔{p['right_col']}(ov={p['overlap']})" for p in jc["pairs"])
            lines.append(f"- {jc['left']} ↔ {jc['right']}: {pairs}")
    return "\n".join(lines)
