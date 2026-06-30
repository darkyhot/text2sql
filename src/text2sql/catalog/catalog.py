"""Каталог метаданных — минимальный предвычисленный слой.

Загружает то, что дорого/нестабильно добывать вживую на каждый вопрос:
каталог таблиц, колонки (dtype/класс/описание/sample), PK-гипотезы. Строит
BM25-индекс для retrieval-first выбора таблиц (масштабируется на тысячи таблиц).

ВАЖНО: PK здесь — ГИПОТЕЗА (выведена по сэмплу оффлайн). Агент обязан
подтверждать её живой пробой перед тем, как полагаться на неё в join.
Источник сейчас — каталог data_for_agent; на любой БД его можно пересобрать
оффлайн-джобом, структура классов от источника не зависит.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

from ..config import PATHS

_TOKEN = re.compile(r"[a-zа-я0-9_]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


@dataclass
class ColumnMeta:
    name: str
    dtype: str
    description: str
    semantic_class: str = ""
    is_pk_hypothesis: bool = False
    not_null_perc: float = 0.0
    unique_perc: float = 0.0
    sample_values: list[str] = field(default_factory=list)


@dataclass
class TableMeta:
    schema: str
    table: str
    description: str
    grain: str = ""
    role: str = ""
    columns: list[ColumnMeta] = field(default_factory=list)

    @property
    def fqn(self) -> str:
        return f"{self.schema}.{self.table}"

    @property
    def pk_hypothesis(self) -> list[str]:
        return [c.name for c in self.columns if c.is_pk_hypothesis]

    def search_doc(self) -> str:
        """Текст для BM25: описание таблицы + имена и описания колонок + sample."""
        parts = [self.table, self.description, self.grain, self.role]
        for c in self.columns:
            parts.append(c.name)
            parts.append(c.description)
            parts.extend(c.sample_values[:6])
        return " ".join(p for p in parts if p)


class Catalog:
    def __init__(self, tables: dict[str, TableMeta], *, join_candidates: list[dict] | None = None):
        self.tables = tables
        self.join_candidates = join_candidates or []
        self._fqns = list(tables.keys())
        self._bm25 = BM25Okapi([_tokenize(tables[f].search_doc()) for f in self._fqns]) if self._fqns else None

    # --- доступ ---
    def get(self, fqn: str) -> TableMeta | None:
        return self.tables.get(fqn)

    def join_candidates_for(self, fqns: list[str]) -> list[dict]:
        """Кандидаты join, относящиеся к указанным таблицам (для join_advisor)."""
        s = set(fqns)
        return [c for c in self.join_candidates if c["left"] in s and c["right"] in s]

    def all_tables(self) -> list[TableMeta]:
        return [self.tables[f] for f in self._fqns]

    def search(self, query: str, k: int = 5) -> list[TableMeta]:
        """Retrieval-first выбор таблиц-кандидатов по BM25."""
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._fqns, scores), key=lambda x: x[1], reverse=True)
        return [self.tables[f] for f, s in ranked[:k]]

    # --- загрузка из data_for_agent ---
    @classmethod
    def load(cls, data_dir: Path | None = None) -> "Catalog":
        data_dir = data_dir or PATHS.data_dir
        tables_df = pd.read_csv(data_dir / "tables_list.csv").fillna("")
        attrs_df = pd.read_csv(data_dir / "attr_list.csv").fillna("")
        # semantic_class и role теперь живут прямо в CSV (регенерируются рефрешем).
        # Fallback на старые JSON-файлы — для совместимости со старой метадатой.
        col_sem = {} if "semantic_class" in attrs_df.columns else _load_json(data_dir / "column_semantics.json")
        tbl_sem = {} if "role" in tables_df.columns else _load_json(data_dir / "table_semantics.json")
        join_cand = _load_list(data_dir / "join_candidates.json")

        tables: dict[str, TableMeta] = {}
        for row in tables_df.itertuples(index=False):
            fqn = f"{row.schema_name}.{row.table_name}"
            role = str(getattr(row, "role", "")) or tbl_sem.get(fqn, {}).get("table_role", "")
            tables[fqn] = TableMeta(
                schema=str(row.schema_name),
                table=str(row.table_name),
                description=str(row.description),
                grain=str(getattr(row, "grain", "")),
                role=role,
            )

        for row in attrs_df.itertuples(index=False):
            fqn = f"{row.schema_name}.{row.table_name}"
            tm = tables.get(fqn)
            if tm is None:
                continue
            sclass = str(getattr(row, "semantic_class", "")) or \
                col_sem.get(f"{fqn}.{row.column_name}", {}).get("semantic_class", "")
            samples = [s for s in str(row.sample_values).split("|") if s] if row.sample_values else []
            tm.columns.append(ColumnMeta(
                name=str(row.column_name),
                dtype=str(row.dType),
                description=str(row.description),
                semantic_class=sclass,
                is_pk_hypothesis=_truthy(row.is_primary_key),
                not_null_perc=_num(row.not_null_perc),
                unique_perc=_num(row.unique_perc),
                sample_values=samples,
            ))
        return cls(tables, join_candidates=join_cand)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_list(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _truthy(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
