"""Семантическая модель таблицы (этап D) — декларативное описание того, что аналитика
«поняла» о таблице: показатели (мера, единица, агрегат), разрезы, сущности, даты — плюс
посчитанные движком `AggSpec` headline-факты (итог по мере, ведущий разрез, концентрация).

Раньше это знание было размазано по объектам roles/measures/labels и жило только в памяти
одного прогона. Теперь оно собирается в один сериализуемый объект и выгружается в
`workspace/<table>_semantic_model.json` — его можно ревьюить, диффать между прогонами и
переиспользовать. Все числа — через `aggregate`/`FrameStore` (детерминированно, без LLM).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .aggspec import AggSpec
from .labels import Labels, fmt_val
from .metrics import ROW_COL, Measure
from .store import FrameStore

logger = logging.getLogger(__name__)


@dataclass
class MeasureSpec:
    name: str
    label: str
    column: str
    kind: str
    unit: str
    agg: str
    total: float | None = None          # итог по всей таблице (движком)
    top_dim: str | None = None          # разрез с максимальной концентрацией
    top_value: str | None = None        # лидер этого разреза
    top_share_pct: float | None = None  # доля лидера, %


@dataclass
class DimSpec:
    column: str
    label: str
    cardinality: int


@dataclass
class SemanticModel:
    table: str
    fqn: str
    description: str
    row_count: int
    measures: list[MeasureSpec] = field(default_factory=list)
    dimensions: list[DimSpec] = field(default_factory=list)
    entities: list[DimSpec] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_schema"] = "text2sql.semantic_model/1"
        return d

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def dump(self, path: Path) -> Path:
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def _round(v, n=2):
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


def build_semantic_model(fqn: str, table_desc: str, df: pd.DataFrame, roles, measures: list[Measure],
                         lbls: Labels, *, store: FrameStore | None = None) -> SemanticModel:
    """Собрать семантическую модель. Headline-факты по мерам считает движок `AggSpec`
    (через переданный `FrameStore` — с мемоизацией, либо создаётся локальный)."""
    table = fqn.split(".", 1)[1] if "." in fqn else fqn
    st = store or FrameStore(df)
    dims = [d for d in roles.dimensions if d in df.columns]

    mspecs: list[MeasureSpec] = []
    for m in measures:
        if m.col not in df.columns and m.col != ROW_COL:
            continue
        agg = "count" if (m.kind == "count" and m.col == ROW_COL) else m.agg
        meas_col = None if agg == "count" and m.col == ROW_COL else m.col
        ms = MeasureSpec(name=m.name, label=m.label, column=m.tech or m.col,
                         kind=m.kind, unit=m.unit, agg=agg)
        try:
            ms.total = _round(st.aggregate(AggSpec(measure=meas_col, agg=agg)).frame["value"].iloc[0])
        except Exception as exc:  # noqa: BLE001
            logger.debug("semantic: итог по %s не посчитан: %s", m.name, exc)
        # аддитивные меры: находим разрез максимальной концентрации (лидер даёт больше всего)
        if agg in ("sum", "count"):
            best = None
            for d in dims:
                if df[d].nunique(dropna=True) < 2:
                    continue
                res = st.aggregate(AggSpec(measure=meas_col, by=(d,), agg=agg, nonzero=True, top=1))
                if res.empty or res.total <= 0:
                    continue
                share = float(res.frame["value"].iloc[0]) / res.total
                if best is None or share > best[0]:
                    best = (share, d, res.frame.iloc[0])
            if best is not None:
                share, d, row = best
                ms.top_dim = lbls.of(d)
                ms.top_value = fmt_val(row[d])          # int-как-int (без «40000000.0»)
                ms.top_share_pct = _round(share * 100, 1)
        mspecs.append(ms)

    dspecs = [DimSpec(column=d, label=lbls.of(d), cardinality=int(roles.card.get(d, df[d].nunique(dropna=True))))
              for d in dims]
    especs = [DimSpec(column=e, label=lbls.of(e), cardinality=int(roles.card.get(e, df[e].nunique(dropna=True))))
              for e in roles.entities if e in df.columns]

    return SemanticModel(table=table, fqn=fqn, description=table_desc, row_count=int(len(df)),
                         measures=mspecs, dimensions=dspecs, entities=especs,
                         dates=[d for d in roles.dates if d in df.columns])
