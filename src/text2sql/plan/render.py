"""Рендер StructuredPlan → SQL (детерминированно) и → NL (для показа юзеру),
плюс структурная валидация. SQL-сборка generic, без доменной логики."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from ..db.adapter import DbAdapter
from .model import Filter, Metric, Projection, StructuredPlan

_AGG_SQL = {
    "count": "COUNT", "sum": "SUM", "avg": "AVG", "min": "MIN", "max": "MAX",
}

# Признак ВЫРАЖЕНИЯ (а не простого идентификатора alias.col): скобки, операторы,
# cast, пробел, CASE/ключевые слова. Такое рендерим ВЕРБАТИМ (не квотим по точкам),
# т.к. это date_trunc(...), a/b, CASE WHEN…, SUM(...)/… и т.п. Идентификаторы в
# схеме — lowercase snake_case, кавычки им не нужны. Безопасность — read-only guard
# адаптера (только SELECT, без ;/DML) + EXPLAIN + апрув человека.
_EXPR_RE = re.compile(r"[()+*/]|\s|::|\bcase\b|\bwhen\b|\bdistinct\b|\bover\b|\bfilter\b", re.I)


def _is_expr(s: str) -> bool:
    return bool(_EXPR_RE.search(s or ""))


# Признак АГРЕГАТ-выражения (SUM(...), COUNT(...), AVG(...)…) — но НЕ оконного
# (…OVER(…)): агрегаты живут только в metrics. Такая запись в GROUP BY невалидна,
# в SELECT — дублирует метрику. Оконные функции идут через raw-SQL, здесь их нет.
_AGG_EXPR_RE = re.compile(r"\b(sum|avg|min|max|count)\s*\(", re.I)


def _is_agg_expr(s: str) -> bool:
    s = s or ""
    return bool(_AGG_EXPR_RE.search(s)) and not re.search(r"\bover\b", s, re.I)


def _q(db: DbAdapter, dotted: str) -> str:
    """'alias.col' → закавыченный идентификатор; ВЫРАЖЕНИЕ → как есть."""
    dotted = (dotted or "").strip()
    if dotted == "*":
        return "*"
    if _is_expr(dotted):
        return dotted
    return ".".join(db.quote_ident(p) for p in dotted.split("."))


def _lit(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (date, datetime)):
        return "'" + value.isoformat() + "'"
    return "'" + str(value).replace("'", "''") + "'"


def _metric_sql(db: DbAdapter, m: Metric) -> str:
    col = m.column or "*"
    is_star = col == "*" or col.endswith(".*")   # 'alias.*' тоже COUNT(*)
    if m.agg == "none":
        expr = _q(db, col)
    elif m.agg == "count_distinct":
        expr = "COUNT(*)" if is_star else f"COUNT(DISTINCT {_q(db, col)})"
    elif m.agg == "count":
        expr = "COUNT(*)" if is_star else f"COUNT({_q(db, col)})"
    else:
        expr = f"{_AGG_SQL[m.agg]}({_q(db, col)})"
    return f"{expr} AS {db.quote_ident(m.alias)}" if m.alias else expr


def _proj_sql(db: DbAdapter, p: Projection) -> str:
    expr = _q(db, p.column)
    return f"{expr} AS {db.quote_ident(p.alias)}" if p.alias else expr


def _filter_sql(db: DbAdapter, f: Filter) -> str:
    col = _q(db, f.column)
    if f.op in ("IS TRUE", "IS FALSE", "IS NULL"):
        return f"{col} {f.op}"
    if f.op == "BETWEEN":
        return f"{col} BETWEEN {_lit(f.value)} AND {_lit(f.value2)}"
    if f.op == "IN":
        vals = f.value if isinstance(f.value, (list, tuple)) else [f.value]
        return f"{col} IN ({', '.join(_lit(v) for v in vals)})"
    if f.op == "ILIKE":
        return f"{col} {db.ilike_op()} {_lit(f.value)}"
    return f"{col} {f.op} {_lit(f.value)}"


def _source_sql(db: DbAdapter, tref) -> str:
    """FROM/JOIN-источник: обычная таблица или ДЕДУПЛИЦИРОВАННАЯ (DISTINCT ON) —
    свежая строка на ключ, для join к справочнику с историей."""
    tbl = db.qualified(*tref.ref.split(".", 1))
    a = db.quote_ident(tref.alias)
    d = getattr(tref, "dedup", None)
    if d and d.by and d.order_by:
        by = ", ".join(db.quote_ident(c) for c in d.by)
        order = f"{by}, {db.quote_ident(d.order_by)} {'DESC' if d.desc else 'ASC'}"
        return f"(SELECT DISTINCT ON ({by}) * FROM {tbl} ORDER BY {order}) AS {a}"
    return f"{tbl} AS {a}"


def render_sql(plan: StructuredPlan, db: DbAdapter) -> str:
    if not plan.tables:
        raise ValueError("План без таблиц — нечего рендерить.")

    select_items: list[str] = []
    select_items += [_proj_sql(db, p) for p in plan.projections]
    select_items += [_metric_sql(db, m) for m in plan.metrics]
    if not select_items:
        select_items = ["*"]

    base = plan.tables[0]
    lines = [f"SELECT {', '.join(select_items)}", f"FROM {_source_sql(db, base)}"]

    for j in plan.joins:
        rt = plan.table_by_alias(j.right_alias)
        if rt is None:
            raise ValueError(f"Join ссылается на неизвестный alias {j.right_alias}")
        conds = " AND ".join(
            f"{db.quote_ident(j.left_alias)}.{db.quote_ident(lc)} = "
            f"{db.quote_ident(j.right_alias)}.{db.quote_ident(rc)}"
            for lc, rc in j.on
        )
        kw = "LEFT JOIN" if getattr(j, "join_type", "inner") == "left" else "INNER JOIN"
        lines.append(f"{kw} {_source_sql(db, rt)} ON {conds}")

    if plan.filters:
        lines.append("WHERE " + " AND ".join(_filter_sql(db, f) for f in plan.filters))
    if plan.group_by:
        lines.append("GROUP BY " + ", ".join(_q(db, c) for c in plan.group_by))
    if plan.having:
        lines.append("HAVING " + " AND ".join(_filter_sql(db, f) for f in plan.having))
    if plan.order_by:
        lines.append("ORDER BY " + ", ".join(_orderby_sql(db, c) for c in plan.order_by))
    if plan.limit is not None:
        lines.append(f"LIMIT {int(plan.limit)}")
    return "\n".join(lines)


def _orderby_sql(db: DbAdapter, item: str) -> str:
    parts = item.strip().rsplit(" ", 1)
    if len(parts) == 2 and parts[1].upper() in ("ASC", "DESC"):
        return f"{_q(db, parts[0])} {parts[1].upper()}"
    return _q(db, item)


def normalize_plan(plan: StructuredPlan) -> StructuredPlan:
    """Generic-нормализация SELECT↔GROUP BY (валидный SQL + видимые разрезы).

    1. При наличии агрегата неагрегированные проекции обязаны быть в GROUP BY —
       иначе СУБД падает с GroupingError. Добавляем их в group_by.
    2. Измерения из GROUP BY должны быть видимы в SELECT — добавляем как проекции,
       чтобы пользователь видел разрезы, а не голые агрегаты."""
    # 0. Убрать вырождение: (а) projection/group_by, совпадающие с агрегируемой колонкой
    #    (SELECT task_code, COUNT(DISTINCT task_code) GROUP BY task_code → просто COUNT);
    #    (б) записи, САМИ являющиеся агрегатом (SUM(...), COUNT(...)): в GROUP BY они
    #    невалидны (СУБД падает), в SELECT — дублируют метрику. Агрегаты живут в metrics.
    metric_cols = {m.column for m in plan.metrics if m.agg != "none" and m.column and m.column != "*"}
    plan.projections = [p for p in plan.projections
                        if p.column not in metric_cols and not _is_agg_expr(p.column)]
    plan.group_by = [c for c in plan.group_by
                     if c not in metric_cols and not _is_agg_expr(c)]

    has_agg = any(m.agg != "none" for m in plan.metrics)
    if has_agg:
        for p in plan.projections:
            if p.column not in plan.group_by:
                plan.group_by.append(p.column)

    projected = {p.column for p in plan.projections}
    metric_cols = {m.column for m in plan.metrics}
    missing = [c for c in plan.group_by if c not in projected and c not in metric_cols]
    if missing:
        plan.projections = [Projection(column=c) for c in missing] + plan.projections

    # дедуп проекций (по column+alias) и group_by — на случай повторов от LLM
    seen: set = set()
    dedup: list[Projection] = []
    for p in plan.projections:
        key = (p.column, p.alias)
        if key not in seen:
            seen.add(key); dedup.append(p)
    plan.projections = dedup
    plan.group_by = list(dict.fromkeys(plan.group_by))
    return plan


class _PlainSQL:
    """Псевдо-адаптер для читаемого SQL без кавычек идентификаторов (для показа
    пользователю). Исполняется всегда квотированный render_sql(plan, db)."""
    def quote_ident(self, name: str) -> str:
        return name

    def qualified(self, schema: str, table: str) -> str:
        return f"{schema}.{table}"

    def ilike_op(self) -> str:
        return "ILIKE"


def render_sql_plain(plan: StructuredPlan) -> str:
    """Читаемый SQL без кавычек — для показа в плане и результате."""
    try:
        return render_sql(plan, _PlainSQL())
    except Exception:  # noqa: BLE001
        return ""


def render_nl(plan: StructuredPlan) -> str:
    """Человекочитаемый план для показа пользователю (то, что он окает)."""
    out: list[str] = []
    if plan.intent:
        out.append(f"Намерение: {plan.intent}")
    tbls = ", ".join(f"{t.ref} ({t.alias})" for t in plan.tables)
    out.append(f"Таблицы: {tbls}")
    for t in plan.tables:
        if getattr(t, "dedup", None) and t.dedup.by:
            out.append(f"Дедупликация {t.alias}: свежая строка на {', '.join(t.dedup.by)} "
                       f"(по {t.dedup.order_by} ↓) — чтобы join не размножил строки")
    for j in plan.joins:
        on = " и ".join(f"{j.left_alias}.{lc}={j.right_alias}.{rc}" for lc, rc in j.on)
        safe = "" if j.fanout_safe is None else (
            f" [{j.classification}, без размножения строк]" if j.fanout_safe
            else f" [{j.classification}, ВНИМАНИЕ: размножение строк!]"
        )
        out.append(f"Join: {j.left_alias} ⨝ {j.right_alias} по {on}{safe}")
    if plan.filters:
        fl = "; ".join(_filter_human(f) for f in plan.filters)
        out.append(f"Фильтры: {fl}")
    if plan.group_by:
        out.append(f"Группировка по: {', '.join(plan.group_by)}")
    if plan.metrics:
        ms = ", ".join(_metric_human(m) for m in plan.metrics)
        out.append(f"Метрики: {ms}")
    if plan.projections and not plan.metrics:
        out.append(f"Колонки: {', '.join(p.column for p in plan.projections)}")
    if plan.order_by:
        out.append(f"Сортировка: {', '.join(plan.order_by)}")
    if plan.limit is not None:
        out.append(f"Лимит: {plan.limit}")
    if plan.grain_note:
        out.append(f"Гранула результата: {plan.grain_note}")
    if plan.assumptions:
        out.append("Предположения: " + "; ".join(plan.assumptions))
    if plan.ambiguities:
        out.append("⚠ Неоднозначности: " + "; ".join(plan.ambiguities))
    return "\n".join(out)


def _metric_human(m: Metric) -> str:
    names = {"count": "количество", "count_distinct": "уникальных", "sum": "сумма",
             "avg": "среднее", "min": "минимум", "max": "максимум", "none": ""}
    col = m.column or "*"
    label = names.get(m.agg, m.agg)
    return f"{label}({col})" if m.agg != "none" else col


def _filter_human(f: Filter) -> str:
    if f.op in ("IS TRUE", "IS FALSE", "IS NULL"):
        return f"{f.column} {f.op}"
    if f.op == "BETWEEN":
        return f"{f.column} от {f.value} до {f.value2}"
    return f"{f.column} {f.op} {f.value}"


def validate_plan(plan: StructuredPlan) -> list[str]:
    """Структурная валидация: алиасы, group-by согласованность, join-ссылки."""
    issues: list[str] = []
    aliases = {t.alias for t in plan.tables}
    if not plan.tables:
        issues.append("Нет ни одной таблицы.")
    for j in plan.joins:
        if j.left_alias not in aliases:
            issues.append(f"Join: неизвестный left alias {j.left_alias}")
        if j.right_alias not in aliases:
            issues.append(f"Join: неизвестный right alias {j.right_alias}")
        if j.fanout_safe is False:
            issues.append(f"Join {j.left_alias}⨝{j.right_alias} помечен как размножающий строки (N:M).")

    def _alias_of(dotted: str) -> str | None:
        return dotted.split(".")[0] if "." in dotted else None

    for ref in [*(p.column for p in plan.projections),
                *(m.column for m in plan.metrics if m.column and m.column != "*"),
                *(f.column for f in plan.filters), *plan.group_by]:
        a = _alias_of(ref or "")
        if a and a not in aliases:
            issues.append(f"Ссылка {ref}: alias {a} не объявлен в таблицах.")

    # GROUP BY согласованность: при наличии агрегатов все необёрнутые проекции
    # должны быть в group_by.
    has_agg = any(m.agg != "none" for m in plan.metrics)
    if has_agg or plan.group_by:
        gb = set(plan.group_by)
        for p in plan.projections:
            if p.column not in gb:
                issues.append(f"Колонка {p.column} не агрегирована и не в GROUP BY.")
    return issues
