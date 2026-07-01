"""Ядро бизнес-отчёта: профилирование колонок в роли и pandas-примитивы анализа
(бизнес-разрезы + seaborn-графики). Никакой «математики ради математики» —
только то, что интересно бизнесу: ТОП-ы, динамика, концентрация, сравнение
периодов, доли, флаги. Все вычисления — pandas."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.titlesize"] = 12

_METRIC_RE = re.compile(r"(qty|amount|amt|sum|cnt|count|val|value|potential|salary|fot|perc|percent|share|rate)", re.I)
_MEAN_RE = re.compile(r"(perc|percent|share|rate|avg|average|доля|средн)", re.I)
_ID_RE = re.compile(r"(^|_)(id|code|key|inn|kpp|ogrn|okato|oktmo|saphr|epk|num)($|_)", re.I)


@dataclass
class Roles:
    dimensions: list[str] = field(default_factory=list)   # читаемые разрезы (*_name, сегменты)
    metrics: list[str] = field(default_factory=list)      # числовые меры
    dates: list[str] = field(default_factory=list)        # даты для динамики
    flags: list[str] = field(default_factory=list)        # булевы признаки
    entities: list[str] = field(default_factory=list)     # высококард. сущности (ФИО/ИНН/компания) — только ТОП-N
    meta: dict[str, dict] = field(default_factory=dict)   # {col: {desc, unique_perc}}


# Сущности для рейтингов (высокая кардинальность — только ТОП, без Парето/долей)
_ENTITY_RE = re.compile(r"(_fio$|^inn$|_inn$|company_name|client|manager|employee|holder_name)", re.I)


@dataclass
class AnalysisResult:
    key: str
    title: str
    kind: str
    facts: dict[str, Any]
    table_md: str = ""
    chart: str | None = None
    insight: str = ""


def agg_for(metric: str) -> str:
    return "mean" if _MEAN_RE.search(metric) else "sum"


def profile(df: pd.DataFrame, meta: dict[str, dict]) -> Roles:
    """Определить роли колонок для бизнес-анализа (метадата + pandas)."""
    r = Roles(meta=meta)
    n = len(df)
    for col in df.columns:
        s = df[col]
        m = meta.get(col, {})
        sclass = m.get("semantic_class", "")
        dtype = str(s.dtype)
        nun = s.nunique(dropna=True)
        uniq_ratio = (nun / n) if n else 0
        if pd.api.types.is_bool_dtype(s) or sclass == "flag":
            r.flags.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s) or sclass == "date" or col.endswith("_dt"):
            if not col.endswith("dttm"):        # системные timestamp не берём в динамику
                r.dates.append(col)
        elif _ENTITY_RE.search(col) and nun > 10 and "author" not in col.lower():
            # сущность (ФИО/ИНН/компания) — для рейтингов ТОП-N
            r.entities.append(col)
        elif pd.api.types.is_numeric_dtype(s) and not _ID_RE.search(col) and _METRIC_RE.search(col):
            r.metrics.append(col)
        elif (sclass in ("enum_like", "label") or dtype == "object") and not _ID_RE.search(col):
            # читаемый разрез: невысокая кардинальность, желательно *_name
            if 1 < nun <= max(60, n * 0.2) and uniq_ratio < 0.5:
                r.dimensions.append(col)
    # приоритет разрезов: *_name / *_type / сегмент — впереди
    r.dimensions.sort(key=lambda c: (0 if re.search(r"(name|type|segment|сегмент|отрасл|регион|region|категор|category)", c, re.I) else 1, c))
    r.metrics.sort(key=lambda c: (0 if re.search(r"(outflow|отток|amt|amount|fot|qty)", c, re.I) else 1, c))
    return r


# ---------- графики ----------
def _save(fig, assets: Path, name: str) -> str:
    assets.mkdir(parents=True, exist_ok=True)
    path = assets / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _fmt(v: float) -> str:
    v = float(v)
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f} млн"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f} тыс"
    return f"{v:.0f}" if v == int(v) else f"{v:.2f}"


def _md_table(df: pd.DataFrame, value_cols: list[str]) -> str:
    d = df.copy()
    for c in value_cols:
        if c in d.columns:
            d[c] = d[c].map(_fmt)
    return d.to_markdown(index=False)


# ---------- бизнес-примитивы ----------
def top_n(df, dim, metric, assets, n=10) -> AnalysisResult:
    d = df[[dim, metric]].copy()
    if pd.api.types.is_bool_dtype(d[metric]):
        d[metric] = d[metric].astype(int)     # sum(bool) = количество «да» (напр. закрытых задач)
    agg = agg_for(metric)
    g = d.groupby(dim)[metric].agg(agg).sort_values(ascending=False)
    total = g.sum()
    head = g.head(n).reset_index()
    head[dim] = head[dim].astype(str)          # категориальная ось (важно для ИНН/числовых)
    top3_share = (g.head(3).sum() / total * 100) if total else 0
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(head))))
    sns.barplot(data=head, y=dim, x=metric, ax=ax, color="#3B7DD8")
    ax.set_title(f"ТОП-{len(head)}: {dim} по {metric}")
    ax.set_xlabel(""); ax.set_ylabel("")
    chart = _save(fig, assets, f"top_{dim}_{metric}")
    facts = {"leader": str(head.iloc[0][dim]), "leader_value": _fmt(head.iloc[0][metric]),
             "top3_share_pct": round(float(top3_share), 1), "agg": agg,
             "n_categories": int(g.shape[0])}
    return AnalysisResult(f"top_{dim}_{metric}", f"ТОП по «{dim}» (метрика {metric})", "top_n", facts,
                          _md_table(head, [metric]), chart)


def top_n_count(df, dim, assets, n=15) -> AnalysisResult:
    g = df.groupby(dim).size().sort_values(ascending=False)
    head = g.head(n).reset_index()
    head.columns = [dim, "количество"]
    head[dim] = head[dim].astype(str)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(head))))
    sns.barplot(data=head, y=dim, x="количество", ax=ax, color="#4F81BD")
    ax.set_title(f"ТОП-{len(head)}: {dim} по количеству записей")
    ax.set_xlabel(""); ax.set_ylabel("")
    chart = _save(fig, assets, f"topcnt_{dim}")
    facts = {"leader": str(head.iloc[0][dim]), "leader_count": int(head.iloc[0]["количество"]),
             "n_entities": int(g.shape[0])}
    return AnalysisResult(f"topcnt_{dim}", f"ТОП по «{dim}» (по количеству)", "top_n_count", facts,
                          _md_table(head, []), chart)


def trend(df, date, metric, assets, freq="MS") -> AnalysisResult:
    agg = agg_for(metric)
    d = df.dropna(subset=[date]).copy()
    d[date] = pd.to_datetime(d[date])
    ts = d.set_index(date).resample(freq)[metric].agg(agg).dropna()
    if len(ts) < 2:
        return AnalysisResult(f"trend_{metric}", f"Динамика «{metric}»", "trend",
                              {"note": "недостаточно периодов"}, "", None)
    first, last = ts.iloc[0], ts.iloc[-1]
    growth = ((last - first) / abs(first) * 100) if first else 0
    peak_period = ts.idxmax()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(x=ts.index, y=ts.values, marker="o", ax=ax, color="#2E8B57")
    ax.set_title(f"Динамика: {metric} ({agg}) по месяцам")
    ax.set_xlabel(""); ax.set_ylabel("")
    chart = _save(fig, assets, f"trend_{metric}")
    tbl = ts.reset_index(); tbl.columns = [date, metric]
    tbl[date] = tbl[date].dt.strftime("%Y-%m")
    facts = {"direction": "рост" if growth > 5 else ("снижение" if growth < -5 else "стабильно"),
             "growth_pct": round(float(growth), 1), "first": _fmt(first), "last": _fmt(last),
             "peak_period": peak_period.strftime("%Y-%m"), "peak_value": _fmt(ts.max())}
    return AnalysisResult(f"trend_{metric}", f"Динамика «{metric}»", "trend", facts,
                          _md_table(tbl.tail(12), [metric]), chart)


def concentration(df, dim, metric, assets) -> AnalysisResult:
    agg = agg_for(metric)
    g = df.groupby(dim)[metric].agg(agg).sort_values(ascending=False)
    total = g.sum()
    if not total:
        return AnalysisResult(f"conc_{dim}", f"Концентрация по «{dim}»", "concentration", {"note": "нет данных"})
    cum = g.cumsum() / total
    n_for_80 = int((cum <= 0.8).sum()) + 1
    pct_for_80 = round(n_for_80 / len(g) * 100, 1)
    head = g.head(10).reset_index()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(head))))
    sns.barplot(data=head, y=dim, x=metric, ax=ax, color="#C0504D")
    ax.set_title(f"Концентрация: {metric} по {dim}")
    ax.set_xlabel(""); ax.set_ylabel("")
    chart = _save(fig, assets, f"conc_{dim}_{metric}")
    facts = {"n_for_80pct": n_for_80, "pct_categories_for_80": pct_for_80,
             "total_categories": int(len(g)), "top_share_pct": round(float(g.iloc[0] / total * 100), 1),
             "top_name": str(g.index[0])}
    return AnalysisResult(f"conc_{dim}_{metric}", f"Концентрация по «{dim}»", "concentration", facts,
                          _md_table(head, [metric]), chart)


def period_compare(df, date, dim, metric, assets) -> AnalysisResult:
    agg = agg_for(metric)
    d = df.dropna(subset=[date]).copy()
    d[date] = pd.to_datetime(d[date])
    d["_p"] = d[date].dt.to_period("M")
    periods = sorted(d["_p"].unique())
    if len(periods) < 2:
        return AnalysisResult(f"cmp_{dim}_{metric}", f"Что изменилось: «{dim}»", "period_compare",
                              {"note": "нужно ≥2 месяцев"})
    prev, cur = periods[-2], periods[-1]
    gp = d[d["_p"] == prev].groupby(dim)[metric].agg(agg)
    gc = d[d["_p"] == cur].groupby(dim)[metric].agg(agg)
    delta = (gc - gp).dropna().sort_values(ascending=False)
    if delta.empty:
        return AnalysisResult(f"cmp_{dim}_{metric}", f"Что изменилось: «{dim}»", "period_compare", {"note": "нет пересечения"})
    movers = pd.concat([delta.head(5), delta.tail(5)]).drop_duplicates()
    tbl = movers.reset_index(); tbl.columns = [dim, "Δ"]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(tbl))))
    colors = ["#2E8B57" if v >= 0 else "#C0504D" for v in tbl["Δ"]]
    sns.barplot(data=tbl, y=dim, x="Δ", ax=ax, palette=colors)
    ax.set_title(f"Изменение {metric}: {cur} vs {prev}")
    ax.set_xlabel(""); ax.set_ylabel("")
    chart = _save(fig, assets, f"cmp_{dim}_{metric}")
    facts = {"period_cur": str(cur), "period_prev": str(prev),
             "top_riser": str(delta.index[0]), "riser_delta": _fmt(delta.iloc[0]),
             "top_faller": str(delta.index[-1]), "faller_delta": _fmt(delta.iloc[-1])}
    return AnalysisResult(f"cmp_{dim}_{metric}", f"Что изменилось: «{dim}»", "period_compare", facts,
                          _md_table(tbl, ["Δ"]), chart)


def flag_breakdown(df, flag, dim, assets) -> AnalysisResult:
    d = df[[flag, dim]].dropna()
    if d.empty:
        return AnalysisResult(f"flag_{flag}_{dim}", f"Доля «{flag}» по «{dim}»", "flag", {"note": "нет данных"})
    rate = d.groupby(dim)[flag].mean().sort_values(ascending=False) * 100
    overall = float(d[flag].mean() * 100)
    head = rate.head(10).reset_index(); head.columns = [dim, "доля_%"]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(head))))
    sns.barplot(data=head, y=dim, x="доля_%", ax=ax, color="#8064A2")
    ax.axvline(overall, color="gray", ls="--", lw=1)
    ax.set_title(f"Доля «{flag}»=да по {dim} (среднее {overall:.0f}%)")
    ax.set_xlabel("%"); ax.set_ylabel("")
    chart = _save(fig, assets, f"flag_{flag}_{dim}")
    facts = {"overall_pct": round(overall, 1), "leader": str(rate.index[0]),
             "leader_pct": round(float(rate.iloc[0]), 1)}
    return AnalysisResult(f"flag_{flag}_{dim}", f"Доля «{flag}» по «{dim}»", "flag", facts,
                          head.round(1).to_markdown(index=False), chart)


def kpi(df, roles: Roles) -> AnalysisResult:
    rows = [("Строк", _fmt(len(df)))]
    for m in roles.metrics[:5]:
        agg = agg_for(m)
        val = df[m].agg(agg)
        rows.append((f"{'Σ' if agg=='sum' else 'среднее'} {m}", _fmt(val)))
    if roles.dates:
        d = pd.to_datetime(df[roles.dates[0]].dropna())
        if len(d):
            rows.append(("Период", f"{d.min():%Y-%m-%d} … {d.max():%Y-%m-%d}"))
    tbl = pd.DataFrame(rows, columns=["Показатель", "Значение"])
    facts = {r[0]: r[1] for r in rows}
    return AnalysisResult("kpi", "Ключевые цифры", "kpi", facts, tbl.to_markdown(index=False))
