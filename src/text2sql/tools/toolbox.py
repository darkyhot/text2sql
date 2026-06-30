"""Инструменты агента — узкие, generic, за диалект-адаптером.

Это субстрат, который дёргают узлы графа по решению LLM. Возвращают компактные,
дружелюбные к слабой модели структуры (Tier-0/1/2 контекста). Доменной логики
нет — только доступ к каталогу и read-only БД + проверки кардинальности.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..catalog.catalog import Catalog, TableMeta
from ..db.adapter import DbAdapter, QueryResult


@dataclass
class JoinVerdict:
    on: list[tuple[str, str]]
    left_unique: bool
    right_unique: bool
    classification: str          # "1:1" | "N:1" | "1:N" | "N:M"
    fanout_safe: bool            # ключ уникален хотя бы на одной стороне
    note: str                    # человекочитаемое объяснение
    dup_example: dict | None = None


class Toolbox:
    def __init__(self, catalog: Catalog, db: DbAdapter, *, tracer: Callable[[dict], None] | None = None):
        self.catalog = catalog
        self.db = db
        self._tracer = tracer

    def _trace(self, event: dict) -> None:
        if self._tracer:
            self._tracer({"kind": "tool", **event})

    # --- Tier-0: retrieval-first выбор таблиц ---
    def search_tables(self, query: str, k: int = 5) -> list[dict[str, str]]:
        hits = self.catalog.search(query, k=k)
        out = [{"fqn": t.fqn, "description": t.description, "grain": t.grain, "role": t.role} for t in hits]
        self._trace({"tool": "search_tables", "query": query, "result": [h["fqn"] for h in out]})
        return out

    # --- Tier-1: колонки выбранной таблицы ---
    def describe_table(self, fqn: str) -> dict[str, Any]:
        t = self.catalog.get(fqn)
        if t is None:
            return {"error": f"таблица {fqn} не найдена в каталоге"}
        cols = [{
            "name": c.name, "dtype": c.dtype, "class": c.semantic_class,
            "pk_hypothesis": c.is_pk_hypothesis, "description": c.description,
            "samples": c.sample_values[:8],
        } for c in t.columns]
        self._trace({"tool": "describe_table", "fqn": fqn, "ncols": len(cols)})
        return {
            "fqn": fqn, "description": t.description, "grain": t.grain, "role": t.role,
            "pk_hypothesis": t.pk_hypothesis, "columns": cols,
        }

    # --- Tier-2: живые значения колонки ---
    def distinct_values(self, fqn: str, column: str, *, like: str | None = None, n: int = 20,
                        enforce_cost: bool = True) -> list[str]:
        """Живые значения колонки (для резолва фильтров). enforce_cost=False —
        не упираться в cost-потолок (для value-резолва: LIMIT + statement_timeout
        и так ограничивают; на больших таблицах ILIKE c LIMIT часто завершается рано).
        При любой ошибке/таймауте возвращает [] (резолв деградирует мягко, не падает)."""
        t = self.catalog.get(fqn)
        if t is None:
            return []
        col = self.db.quote_ident(column)
        where = ""
        params_note = ""
        if like:
            safe = like.replace("'", "''")
            where = f"WHERE {col} {self.db.ilike_op()} '%{safe}%'"
            params_note = like
        sql = f"SELECT DISTINCT {col} AS v FROM {self.db.qualified(t.schema, t.table)} {where} LIMIT {int(n)}"
        try:
            res = self.db.run_select(sql, limit=n, enforce_cost=enforce_cost)
        except Exception:  # noqa: BLE001  (cost/timeout/ошибка — мягкая деградация)
            self._trace({"tool": "distinct_values", "fqn": fqn, "column": column, "like": params_note, "error": True})
            return []
        vals = [str(r["v"]) for r in res.rows if r["v"] is not None]
        self._trace({"tool": "distinct_values", "fqn": fqn, "column": column, "like": params_note, "n_found": len(vals)})
        return vals

    # --- произвольная read-only проба ---
    def probe_sql(self, sql: str, *, limit: int | None = None) -> QueryResult:
        return self.db.run_select(sql, limit=limit)

    # --- ядро: проверка кардинальности join (анти-fan-out) ---
    def key_unique(self, fqn: str, cols: list[str]) -> tuple[bool, dict | None]:
        """Уникален ли набор колонок на таблице (живая проба). Возвращает (uniq, пример_дубля)."""
        t = self.catalog.get(fqn)
        if t is None or not cols:
            return False, None
        qcols = ", ".join(self.db.quote_ident(c) for c in cols)
        sql = (
            f"SELECT {qcols}, count(*) AS _cnt FROM {self.db.qualified(t.schema, t.table)} "
            f"GROUP BY {qcols} HAVING count(*) > 1 LIMIT 1"
        )
        res = self.db.run_select(sql, limit=1)
        uniq = res.rowcount == 0
        return uniq, (res.rows[0] if not uniq else None)

    def check_join(self, left_fqn: str, right_fqn: str, on: list[tuple[str, str]]) -> JoinVerdict:
        """Классифицировать кардинальность join по живым данным.

        Безопасность от размножения строк: join не множит сторону X, если
        ключ уникален на ПРОТИВОПОЛОЖНОЙ стороне. Если ключ не уникален ни на
        одной стороне (N:M) — гарантированный fan-out.
        """
        left_cols = [l for l, _ in on]
        right_cols = [r for _, r in on]
        left_uniq, left_dup = self.key_unique(left_fqn, left_cols)
        right_uniq, right_dup = self.key_unique(right_fqn, right_cols)

        if left_uniq and right_uniq:
            cls, safe = "1:1", True
        elif right_uniq:
            cls, safe = "N:1", True
        elif left_uniq:
            cls, safe = "1:N", True
        else:
            cls, safe = "N:M", False

        if safe:
            note = (
                f"Join {cls}: ключ уникален на "
                f"{'обеих сторонах' if cls=='1:1' else (right_fqn if cls=='N:1' else left_fqn)}. "
                "Размножения строк не будет."
            )
            dup = None
        else:
            note = (
                f"Join N:M: ключ {on} НЕ уникален ни на {left_fqn}, ни на {right_fqn}. "
                "Будет размножение строк — добавьте колонки в ключ или агрегируйте."
            )
            dup = left_dup or right_dup
        verdict = JoinVerdict(on, left_uniq, right_uniq, cls, safe, note, dup)
        self._trace({"tool": "check_join", "left": left_fqn, "right": right_fqn,
                     "on": on, "classification": cls, "fanout_safe": safe})
        return verdict
