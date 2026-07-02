"""Контракт плана: СТРУКТУРИРОВАННЫЙ объект — единственный источник истины.

Пользователь видит NL-рендер этого объекта и окает именно его. SQL собирается
из той же структуры детерминированно (это сериализация, а не доменное решение —
generic на любые таблицы), поэтому SQL не может «уехать» от одобренного плана.
Все РЕШЕНИЯ (какие таблицы/колонки/джойны/фильтры) принимает LLM и кладёт сюда.

Покрывает аналитическую форму SPJA (select-project-join-aggregate). Сложные
случаи (оконные функции, подзапросы) — отдельный путь raw-SQL позже.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

AggT = Literal["count", "count_distinct", "sum", "avg", "min", "max", "none"]
OpT = Literal["=", "!=", ">", "<", ">=", "<=", "ILIKE", "BETWEEN", "IS TRUE", "IS FALSE", "IS NULL", "IN"]


class Dedup(BaseModel):
    """Дедупликация источника: оставить ОДНУ (свежую) строку на комбинацию `by`.
    Нужна при join к справочной/атрибутивной таблице с историей (несколько строк
    на ключ) — иначе join размножит строки. Рендерится как DISTINCT ON."""
    by: list[str] = Field(default_factory=list)      # ключ дедупликации (= колонки join)
    order_by: str = ""                                # колонка актуальности (дата карточки)
    desc: bool = True                                 # свежая запись первой


class TableRef(BaseModel):
    ref: str           # fully-qualified schema.table
    alias: str
    dedup: Dedup | None = None


class JoinSpec(BaseModel):
    left_alias: str
    right_alias: str
    on: list[tuple[str, str]]            # пары (колонка_слева, колонка_справа), без алиасов
    join_type: str = "inner"             # inner | left (LEFT JOIN — включить строки без пары)
    classification: str = ""             # заполняется check_join: 1:1/N:1/1:N/N:M
    fanout_safe: bool | None = None


class Metric(BaseModel):
    agg: AggT = "none"
    column: str | None = None            # "alias.col" или "*"
    alias: str | None = None


class Filter(BaseModel):
    column: str                          # "alias.col"
    op: OpT
    value: Any = None
    value2: Any = None                   # для BETWEEN
    resolved_via: str = ""               # probe | metadata | user | direct


class Projection(BaseModel):
    """Колонка в SELECT без агрегации (для group-by измерений или сырых выборок)."""
    column: str                          # "alias.col"
    alias: str | None = None


class StructuredPlan(BaseModel):
    intent: str = ""                     # краткая формулировка намерения (для NL)
    tables: list[TableRef] = Field(default_factory=list)
    joins: list[JoinSpec] = Field(default_factory=list)
    projections: list[Projection] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    filters: list[Filter] = Field(default_factory=list)
    having: list[Filter] = Field(default_factory=list)  # фильтры по агрегатам (HAVING)
    group_by: list[str] = Field(default_factory=list)   # "alias.col" или выражение
    order_by: list[str] = Field(default_factory=list)
    limit: int | None = None
    grain_note: str = ""
    assumptions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        """Слабые модели вольно обращаются со схемой: null вместо []/'',
        строки вместо объектов, разные формы join.on. Нормализуем всё здесь."""
        if not isinstance(data, dict):
            return data
        list_fields = ("tables", "joins", "projections", "metrics", "filters", "having",
                       "group_by", "order_by", "assumptions", "ambiguities")
        for f in list_fields:
            if data.get(f) is None:
                data[f] = []
        for f in ("intent", "grain_note"):
            if data.get(f) is None:
                data[f] = ""
        # projections: "alias.col" -> {"column": "alias.col"}
        data["projections"] = [
            {"column": p} if isinstance(p, str) else p for p in data.get("projections", [])
        ]
        # metrics: "alias.col" -> {"agg":"none","column":...}
        data["metrics"] = [
            {"agg": "none", "column": m} if isinstance(m, str) else m for m in data.get("metrics", [])
        ]
        # group_by/order_by: вытащить колонку, если пришли объекты
        for f in ("group_by", "order_by"):
            data[f] = [x.get("column", "") if isinstance(x, dict) else x for x in data.get(f, [])]
        # joins.on: привести к списку пар [lcol, rcol]
        for j in data.get("joins", []):
            if isinstance(j, dict) and "on" in j:
                j["on"] = _normalize_on(j["on"])
        # tables.dedup: пустой/битый dedup → None
        for t in data.get("tables", []):
            if isinstance(t, dict):
                d = t.get("dedup")
                if isinstance(d, dict):
                    by = d.get("by") or []
                    if isinstance(by, str):
                        by = [by]
                    d["by"] = [str(c).split(".")[-1] for c in by]
                    if not d["by"] or not d.get("order_by"):
                        t["dedup"] = None
                elif d is not None and not isinstance(d, dict):
                    t["dedup"] = None
        return data

    def table_by_alias(self, alias: str) -> TableRef | None:
        return next((t for t in self.tables if t.alias == alias), None)


def _normalize_on(on: Any) -> list[list[str]]:
    """join.on в разных формах -> [[lcol, rcol], ...]."""
    out: list[list[str]] = []
    if not isinstance(on, list):
        return out
    def bare(c: Any) -> str:
        return str(c).strip().split(".")[-1]   # снять alias-префикс: f.gosb_id -> gosb_id

    for item in on:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append([bare(item[0]), bare(item[1])])
        elif isinstance(item, dict):
            lc = item.get("left") or item.get("left_col") or item.get("lcol")
            rc = item.get("right") or item.get("right_col") or item.get("rcol")
            if lc and rc:
                out.append([bare(lc), bare(rc)])
        elif isinstance(item, str) and "=" in item:
            lc, rc = item.split("=", 1)
            out.append([bare(lc), bare(rc)])
    return out
