"""AggregateRequest — декларативный запрос агрегации + детерминированный pandas-исполнитель.

Примитив аналитического ядра (этап D). Раньше агрегации были россыпью `groupby(...).agg(...)`
по детекторам `mining.py`/`investigate.py`; теперь — один типизированный запрос
«мера × разрезы × агрегат × фильтр × сортировка/топ» и единственный исполнитель `aggregate`.
Плюсы: одно место для правил (числовое приведение, dropna, «не-ноль»-скоуп, доли), общий
кэш (см. `store.py`) и основа для playbook'ов/примитивов, собираемых из таких запросов.

Исполнитель НЕ зовёт LLM — только pandas. Возвращает tidy-DataFrame `[*by, value]`
плюс `total` для долей.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_NUMERIC_AGGS = {"sum", "mean", "median", "min", "max"}
_AGGS = _NUMERIC_AGGS | {"count", "nunique"}


@dataclass(frozen=True)
class AggSpec:
    """Декларативный запрос агрегации. Хэшируемый (frozen) — годится ключом кэша."""
    measure: str | None = None          # колонка меры; None + agg='count' → счёт строк
    by: tuple[str, ...] = ()            # разрезы (0..2 колонки)
    agg: str = "sum"                    # sum|mean|median|min|max|count|nunique
    where: str | None = None           # pandas-query() фильтр (детерминированный, без LLM)
    nonzero: bool = False              # выбросить строки с нулевым/пустым агрегатом (бизнес-скоуп)
    sort_desc: bool | None = True      # True/False — сортировать по значению; None — не трогать
    top: int | None = None             # оставить топ-N после сортировки

    def __post_init__(self) -> None:
        if self.agg not in _AGGS:
            raise ValueError(f"agg должен быть из {sorted(_AGGS)}, дано {self.agg!r}")
        if len(self.by) > 2:
            raise ValueError("AggSpec.by поддерживает максимум 2 разреза")
        if self.agg != "count" and self.measure is None:
            raise ValueError(f"agg={self.agg!r} требует measure")


@dataclass
class AggResult:
    spec: AggSpec
    frame: pd.DataFrame                 # колонки: [*spec.by, 'value']
    total: float                        # агрегат по всему срезу (знаменатель долей)

    @property
    def empty(self) -> bool:
        return self.frame.empty

    @property
    def leader(self) -> pd.Series | None:
        return None if self.frame.empty else self.frame.iloc[0]

    def shares(self) -> pd.Series:
        """Доля каждого разреза от суммы значений (для концентрации/Парето)."""
        v = self.frame["value"].astype(float)
        tot = float(v.sum())
        return v / tot if tot else v * 0.0

    def series(self) -> pd.Series:
        """value, проиндексированный разрезом (1 by) — для совместимости с groupby-кодом."""
        if len(self.spec.by) == 1:
            return self.frame.set_index(self.spec.by[0])["value"]
        return self.frame["value"]


def _apply_where(df: pd.DataFrame, where: str | None) -> pd.DataFrame:
    if not where:
        return df
    try:
        return df.query(where)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"AggSpec.where не применился ({where!r}): {exc}") from exc


def aggregate(df: pd.DataFrame, spec: AggSpec) -> AggResult:
    """Исполнить AggSpec над df. Чистый pandas, детерминированно."""
    d = _apply_where(df, spec.where)
    by = list(spec.by)

    # --- значения ---
    if spec.agg == "count" and spec.measure is None:
        grouped = d.groupby(by).size() if by else pd.Series([len(d)], index=[""])
        total = float(len(d))
    else:
        col = spec.measure
        src = pd.to_numeric(d[col], errors="coerce") if spec.agg in _NUMERIC_AGGS else d[col]
        if by:
            grouped = src.groupby([d[b] for b in by]).agg(spec.agg)
        else:
            grouped = pd.Series([getattr(src, spec.agg)()], index=[""])
        # total: тот же агрегат по всему срезу (для sum это сумма; для mean/count — по всему)
        total = float(getattr(src, spec.agg)()) if spec.agg != "count" else float(src.count())

    grouped = grouped.dropna()
    if spec.nonzero:
        grouped = grouped[grouped != 0]

    # --- сортировка / топ ---
    if spec.sort_desc is not None and not grouped.empty:
        grouped = grouped.sort_values(ascending=not spec.sort_desc)
    if spec.top is not None:
        grouped = grouped.head(spec.top)

    # --- tidy DataFrame [*by, value] ---
    if by:
        frame = grouped.rename("value").reset_index()
        # groupby по нескольким ключам даёт MultiIndex → reset даёт колонки by
        frame.columns = by + ["value"]
    else:
        frame = pd.DataFrame({"value": [float(grouped.iloc[0]) if len(grouped) else 0.0]})
    return AggResult(spec=spec, frame=frame.reset_index(drop=True), total=total)


# ------- лёгкий self-test (запуск: python -m text2sql.report.aggspec) -------
if __name__ == "__main__":  # pragma: no cover
    import numpy as np
    dfx = pd.DataFrame({
        "seg": ["A", "A", "B", "B", "B", "C"],
        "reg": ["x", "y", "x", "x", "y", "y"],
        "amt": [10, 20, 5, 5, 40, 3],
        "flag": [1, 0, 1, 1, 0, 1],
    })
    r = aggregate(dfx, AggSpec(measure="amt", by=("seg",), agg="sum"))
    assert list(r.frame["seg"]) == ["B", "A", "C"], r.frame
    assert r.total == 83.0
    assert abs(r.shares().iloc[0] - 50 / 83) < 1e-9
    r2 = aggregate(dfx, AggSpec(agg="count", by=("seg",)))
    assert dict(zip(r2.frame["seg"], r2.frame["value"])) == {"A": 2, "B": 3, "C": 1}
    r3 = aggregate(dfx, AggSpec(measure="amt", by=("seg",), agg="sum", where="flag == 1", top=2))
    assert set(r3.frame["seg"]) == {"A", "B"} and len(r3.frame) == 2, r3.frame  # A=10, B=10 (ничья)
    r4 = aggregate(dfx, AggSpec(measure="amt", by=("seg", "reg"), agg="sum"))
    assert set(r4.frame.columns) == {"seg", "reg", "value"}
    r5 = aggregate(dfx, AggSpec(measure="amt", agg="mean"))
    assert abs(float(r5.frame["value"].iloc[0]) - float(np.mean([10, 20, 5, 5, 40, 3]))) < 1e-9
    print("aggspec self-test OK")
