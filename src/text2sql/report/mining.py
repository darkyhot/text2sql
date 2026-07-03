"""Майнинг интересных срезов: крутим показатели по множеству разрезов и находим
НЕМНОГО действительно значимого — то, что полезно бизнесу, а не «стену математики».

Принципы (лечат провал прошлого подхода):
  1. МАТЕРИАЛЬНОСТЬ — первым фильтром. Срез игнорируется, если мал по объёму
     (мало строк И мало денег). Убивает шум мелких ячеек.
  2. Отклонение — в РАЗАХ и ДОЛЯХ, не в статжаргоне («вдвое хуже среднего»,
     «даёт 40% всех потерь»).
  3. Ранжируем по ДЕНЬГАМ (ключевая мера), но количество тоже показываем.
  4. Отбор и обрезка до топ-находок, дедуп пересечений.

Что ищем:
  • rate-отклонения (1D): срез, где доля (просрочка/закрытие) заметно хуже/лучше
    среднего и существенен по объёму;
  • 2D-взаимодействия: пара разрезов, где эффект СИЛЬНЕЕ, чем предсказывают
    одиночные разрезы (ММБ × Москва × отток) — настоящая аномалия, не сумма частей;
  • концентрация денег (Парето): где сосредоточены потери/потенциал;
  • перекос ДЕНЬГИ vs КОЛИЧЕСТВО: срез мал по людям, но крупный по деньгам
    (10 человек с ЗП ×30 весомее 100 «дешёвых»);
  • drill по сущностям: кто (сотрудники/ИНН) создаёт проблему внутри горячего среза.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .core import AnalysisResult, _fmt, _save
from .labels import Labels, fmt_val
from .metrics import Measure, ROW_COL

logger = logging.getLogger(__name__)


def _dt(labels: Labels | None, d: str) -> str:      # «подпись (col)» для заголовка
    return labels.col_title(d) if labels else d


def _ds(labels: Labels | None, d: str) -> str:      # короткая подпись разреза
    return labels.of(d) if labels else d


@dataclass
class Finding:
    kind: str                       # rate_dev | interaction | money_conc | value_mismatch
    measure: str
    unit: str
    dims: list[str]                 # РЕАЛЬНЫЕ имена колонок-разрезов (для маски среза)
    title: str
    facts: dict
    score: float
    n_rows: int = 0
    money: float | None = None
    chart: str | None = None
    drill_md: str = ""
    key: str = ""
    slice_vals: list = field(default_factory=list)   # СЫРЫЕ значения среза (для маски)


def _money_measure(measures: list[Measure]) -> Measure | None:
    # для концентрации/веса нужны АДДИТИВНЫЕ деньги (sum), не средняя ЗП (mean)
    money = [m for m in measures if m.kind == "money" and m.agg == "sum"]
    return money[0] if money else None


def _fmt_unit(v: float, unit: str) -> str:
    s = _fmt(v)
    if unit == "%":
        return f"{v*100:.0f}%" if v <= 1.5 else f"{v:.0f}%"
    return f"{s} {unit}".strip()


def mine(df: pd.DataFrame, measures: list[Measure], dims: list[str], entities: list[str],
         assets: Path, *, focus_dims: list[str] | None = None,
         labels: Labels | None = None) -> list[Finding]:
    """Полный майнинг. Возвращает отобранные находки (уже отранжированы, дедуп)."""
    n = len(df)
    if n == 0 or not dims:
        return []
    df[ROW_COL] = 1
    min_rows = max(30, int(n * 0.005))
    fd = focus_dims or []
    dims = fd + [d for d in dims if d not in fd]      # фокус-разрезы вперёд
    dims = [d for d in dims if d in df.columns and df[d].nunique(dropna=True) >= 2][:10]
    rate_measures = [m for m in measures if m.kind in ("rate", "duration")]
    # концентрация/перекос/вес — только по АДДИТИВНЫМ мерам (sum); средние/доли не суммируем
    val_measures = [m for m in measures if m.kind in ("money", "count", "value") and m.agg == "sum"]

    # «мера значимости»: деньги, если есть; иначе главный количественный показатель.
    # ₽ рисуем ТОЛЬКО когда это настоящие деньги (weight.kind == 'money').
    weight = _money_measure(measures) or (val_measures[0] if val_measures else None)
    total_w = float(df[weight.col].sum()) if weight is not None else 0.0

    findings: list[Finding] = []
    # 1) rate/duration-отклонения по одиночным разрезам
    for m in rate_measures:
        findings += _rate_deviations(df, m, dims, min_rows, weight, total_w, labels)
    # 2) концентрация денег/объёма (по подмножеству, где мера заполнена — бизнес-скоуп)
    for m in val_measures[:3]:
        findings += _money_concentration(_pop(df, m), m, dims, labels)
    # 3) перекос значимость vs количество (мал по строкам, крупный по деньгам/объёму)
    if weight is not None:
        findings += _value_mismatch(_pop(df, weight), weight, dims, min_rows, labels)
    # 4) 2D-взаимодействия — только по rate/duration («хуже, чем ждали» осмысленно
    #    для долей/сроков; для сумм это шум — их закрывают концентрация и перекос)
    top_dims = (focus_dims or []) + [d for d in dims if d not in (focus_dims or [])]
    for m in rate_measures[:3]:
        findings += _interactions_2d(df, m, top_dims[:5], min_rows, weight, total_w, labels)

    findings = _select(findings)
    # 5) drill по сущностям внутри горячих срезов (только для отобранных топ-4)
    for f in findings[:4]:
        f.drill_md = _drill(df, f, entities, weight, measures, labels)
    for f in findings:
        f.chart = _chart(df, f, measures, assets, labels)
    return findings


def _has_value(df: pd.DataFrame, m: Measure) -> pd.Series:
    """Строки, где числовая мера НЕСЁТ значение = не-null И не-ноль. Ноль для деловых
    метрик (кол-во сделок/потенциал) означает «не применимо» (напр. task_category='Задача'
    не несёт сделку), поэтому его тоже считаем «нет значения»."""
    vals = pd.to_numeric(df[m.col], errors="coerce").fillna(0)
    return vals != 0


def _pop(df: pd.DataFrame, m: Measure) -> pd.DataFrame:
    """Подмножество строк, где мера ОСМЫСЛЕННА (бизнес-скоуп): если у меры значимая доля
    строк без значения (null ИЛИ 0), анализируем только там, где она есть — напр. потенциал/
    кол-во сделок только для 'Предложение', а не 'Задача'. Доли/сроки не режем (0 осмыслен)."""
    if m is None or m.col not in df.columns or m.kind in ("rate", "duration"):
        return df
    has = _has_value(df, m)
    return df[has] if (~has).mean() > 0.1 else df


def scope_notes(df: pd.DataFrame, measures: list[Measure], dims: list[str],
                labels: Labels | None = None) -> list[str]:
    """Бизнес-скоуп показателей: если метрика ОСМЫСЛЕННА (не-null и не-ноль) только для
    части значений некой категории (сделочные числа — только для 'Предложение', не 'Задача'),
    фиксируем заметкой. Детерминированно, без хардкода: категория делит метрику на есть/нет."""
    notes: list[str] = []
    low_dims = [d for d in dims if d in df.columns and 2 <= df[d].nunique(dropna=True) <= 15]
    for m in measures:
        if m.kind not in ("money", "count", "value") or m.col not in df.columns:
            continue
        has = _has_value(df, m)
        if has.mean() > 0.9:                  # значение почти везде — не условная
            continue
        for d in low_dims:
            frac = has.groupby(df[d]).mean()
            populated = frac[frac >= 0.5].index.tolist()
            empty = frac[frac <= 0.05].index.tolist()
            if populated and empty:           # категория делит на есть/нет значения
                vals = ", ".join(fmt_val(v) for v in populated[:4])
                notes.append(f"«{m.label}» осмыслен только для «{_ds(labels, d)}» = {vals} "
                             f"(в остальных — нет значения) — анализирую только по ним.")
                break
    return notes


# ---------- детекторы ----------
def _rate_deviations(df, m: Measure, dims, min_rows, weight, total_w, labels=None) -> list[Finding]:
    overall = float(df[m.col].mean())
    if not np.isfinite(overall):
        return []
    is_money = weight is not None and weight.kind == "money"
    out: list[Finding] = []
    for d in dims:
        g = df.groupby(d).agg(val=(m.col, "mean"), n=(ROW_COL, "sum"))
        if weight is not None:
            g["w"] = df.groupby(d)[weight.col].sum()
        g = g[g["n"] >= min_rows]
        if g.empty:
            continue
        for name, row in g.iterrows():
            val, nrows = float(row["val"]), int(row["n"])
            if not np.isfinite(val) or overall == 0:
                continue
            lift = val / overall if overall else 1.0
            gap = val - overall
            # rate: существенное относительное И абсолютное отклонение; duration: относительное
            if m.kind == "rate" and (abs(gap) < 0.05 or 0.77 <= lift <= 1.3):
                continue
            if m.kind == "duration" and 0.7 <= lift <= 1.43:
                continue
            slice_w = float(row.get("w", 0.0)) if weight is not None else 0.0
            material = (slice_w / total_w) if total_w else (nrows / len(df))
            score = abs(lift - 1) * np.log1p(material * 100) * (2 if is_money and slice_w else 1)
            direction = "выше" if gap > 0 else "ниже"
            out.append(Finding(
                kind="rate_dev", measure=m.name, unit=m.unit, dims=[d],
                title=f"{m.title()} по «{_dt(labels, d)}»: отклонение от среднего",
                facts={"dim": _dt(labels, d), "slice": fmt_val(name), "value": _fmt_unit(val, m.unit),
                       "overall": _fmt_unit(overall, m.unit), "lift": round(lift, 2),
                       "direction": direction, "n_rows": nrows,
                       "weight": _fmt_unit(slice_w, weight.unit) if weight is not None else None,
                       "weight_name": weight.label if weight is not None else None,
                       "is_money": is_money},
                score=float(score), n_rows=nrows, slice_vals=[name],
                money=slice_w if is_money else None,
                key=f"rate|{m.name}|{d}|{name}"))
    return out


def _money_concentration(df, m: Measure, dims, labels=None) -> list[Finding]:
    out: list[Finding] = []
    for d in dims:
        g = df.groupby(d)[m.col].sum().sort_values(ascending=False)
        total = float(g.sum())
        if total <= 0 or len(g) < 3:
            continue
        top_share = float(g.iloc[0]) / total
        cum = g.cumsum() / total
        n80 = int((cum <= 0.8).sum()) + 1
        # интересно, если концентрация выражена: 1 срез ≥25% ИЛИ ≤20% категорий дают 80%
        if top_share < 0.25 and (n80 / len(g)) > 0.35:
            continue
        score = top_share * 3 + (1 - n80 / len(g))
        out.append(Finding(
            kind="money_conc", measure=m.name, unit=m.unit, dims=[d],
            title=f"Концентрация «{m.title()}» по «{_dt(labels, d)}»",
            facts={"dim": _dt(labels, d), "leader": fmt_val(g.index[0]),
                   "leader_val": _fmt_unit(float(g.iloc[0]), m.unit),
                   "leader_share": round(top_share * 100, 1), "n_for_80": n80,
                   "total_categories": len(g), "unit": m.unit},
            score=float(score), money=float(g.iloc[0]) if m.kind == "money" else None,
            slice_vals=[g.index[0]], key=f"conc|{m.name}|{d}"))
    return out


def _value_mismatch(df, weight: Measure, dims, min_rows, labels=None) -> list[Finding]:
    """Срезы, где доля ЗНАЧИМОСТИ (деньги/объём) существенно выше доли КОЛИЧЕСТВА строк —
    мал по людям/строкам, но крупный по деньгам/объёму (высокая ценность на единицу)."""
    total_w = float(df[weight.col].sum())
    if total_w <= 0:
        return []
    is_money = weight.kind == "money"
    noun = "деньгам" if is_money else f"«{weight.label}»"
    total_rows = len(df)
    out: list[Finding] = []
    for d in dims:
        g = df.groupby(d).agg(w=(weight.col, "sum"), n=(ROW_COL, "sum"))
        g = g[g["n"] >= min_rows]
        if g.empty:
            continue
        g["w_share"] = g["w"] / total_w
        g["cnt_share"] = g["n"] / total_rows
        g["ratio"] = g["w_share"] / g["cnt_share"].replace(0, np.nan)
        cand = g[(g["w_share"] >= 0.08) & (g["ratio"] >= 1.8)].sort_values("w", ascending=False)
        for name, row in cand.head(3).iterrows():
            out.append(Finding(
                kind="value_mismatch", measure=weight.name, unit=weight.unit, dims=[d],
                title=f"«{fmt_val(name)}» ({_ds(labels, d)}) — мал по количеству, крупный по {noun}",
                facts={"dim": _dt(labels, d), "slice": fmt_val(name), "is_money": is_money,
                       "weight_name": weight.label,
                       "weight": _fmt_unit(float(row["w"]), weight.unit),
                       "w_share": round(float(row["w_share"]) * 100, 1),
                       "cnt_share": round(float(row["cnt_share"]) * 100, 1),
                       "ratio": round(float(row["ratio"]), 1), "n_rows": int(row["n"])},
                score=float(row["ratio"]) * float(row["w_share"]) * 4,
                n_rows=int(row["n"]), money=float(row["w"]) if is_money else None,
                slice_vals=[name], key=f"mismatch|{weight.name}|{d}|{name}"))
    return out


def _interactions_2d(df, m: Measure, dims, min_rows, weight, total_w, labels=None) -> list[Finding]:
    """Пара разрезов, где показатель СИЛЬНЕЕ, чем предсказывают одиночные эффекты
    (мультипликативная модель). Это настоящая аномалия сочетания."""
    if len(dims) < 2:
        return []
    is_money = weight is not None and weight.kind == "money"
    agg = "mean" if m.kind in ("rate", "duration") else "sum"
    overall = float(df[m.col].agg(agg))
    if not np.isfinite(overall) or overall == 0:
        return []
    out: list[Finding] = []
    for da, db in combinations(dims, 2):
        ma = df.groupby(da)[m.col].agg(agg)          # средние по одиночным разрезам
        mb = df.groupby(db)[m.col].agg(agg)
        grp = df.groupby([da, db]).agg(val=(m.col, agg), n=(ROW_COL, "sum"))
        if weight is not None:
            grp["w"] = df.groupby([da, db])[weight.col].sum()
        grp = grp[grp["n"] >= min_rows]
        if grp.empty:
            continue
        for (va, vb), row in grp.iterrows():
            val, nrows = float(row["val"]), int(row["n"])
            sa, sb = float(ma.get(va, overall)), float(mb.get(vb, overall))
            pred = overall * (sa / overall) * (sb / overall) if overall else 0
            if pred <= 0 or not np.isfinite(val):
                continue
            inter = val / pred                       # >1 сильнее, чем ждали от суммы эффектов
            if m.kind == "rate" and abs(val - overall) < 0.05:
                continue
            if not (inter >= 1.4 or inter <= 0.6):
                continue
            # НАСТОЯЩАЯ синергия: комбо ХУЖЕ (или ЛУЧШЕ) каждого разреза по отдельности,
            # иначе один разрез уже всё объясняет (не interaction, а его 1D-эффект)
            if inter >= 1.4 and not (val > sa * 1.1 and val > sb * 1.1):
                continue
            if inter <= 0.6 and not (val < sa * 0.9 and val < sb * 0.9):
                continue
            slice_w = float(row.get("w", 0.0)) if weight is not None else 0.0
            material = (slice_w / total_w) if total_w else (nrows / len(df))
            if material < 0.01 and nrows < min_rows * 2:
                continue
            score = abs(inter - 1) * np.log1p(material * 100) * 2.5
            out.append(Finding(
                kind="interaction", measure=m.name, unit=m.unit, dims=[da, db],
                title=f"Аномальное сочетание: «{fmt_val(va)}» × «{fmt_val(vb)}» — {m.title()}",
                facts={"dim_a": _dt(labels, da), "val_a": fmt_val(va),
                       "dim_b": _dt(labels, db), "val_b": fmt_val(vb),
                       "value": _fmt_unit(val, m.unit), "overall": _fmt_unit(overall, m.unit),
                       "vs_expected": round(inter, 2), "n_rows": nrows,
                       "weight": _fmt_unit(slice_w, weight.unit) if weight is not None else None,
                       "weight_name": weight.label if weight is not None else None},
                score=float(score), n_rows=nrows, slice_vals=[va, vb],
                money=slice_w if is_money else None,
                key=f"inter|{m.name}|{da}|{va}|{db}|{vb}"))
    return out


# ---------- отбор ----------
_KIND_CAP = {"rate_dev": 6, "interaction": 3, "money_conc": 4, "value_mismatch": 3}


def _select(findings: list[Finding], *, total=14) -> list[Finding]:
    findings.sort(key=lambda f: f.score, reverse=True)
    seen_vals: set[tuple] = set()
    by_kind: dict[str, int] = {}
    out: list[Finding] = []
    for f in findings:
        vals = [f.facts.get(k) for k in ("slice", "val_a", "val_b", "leader") if f.facts.get(k)]
        # дедуп: если ВСЕ значения среза уже показаны для этой меры — пропускаем
        # (не повторяем один и тот же ГОСБ/сегмент в разных сочетаниях)
        if vals and all((f.measure, v) in seen_vals for v in vals):
            continue
        if by_kind.get(f.kind, 0) >= _KIND_CAP.get(f.kind, 5):
            continue
        by_kind[f.kind] = by_kind.get(f.kind, 0) + 1
        for v in vals:
            seen_vals.add((f.measure, v))
        out.append(f)
        if len(out) >= total:
            break
    return out


# ---------- drill по сущностям ----------
def _drill(df, f: Finding, entities: list[str], weight: Measure | None,
           measures: list[Measure], labels=None) -> str:
    """Кто внутри горячего среза создаёт эффект. Для доли/срока ранжируем по ЧИСЛУ
    «проблемных» случаев (кто даёт больше всего просрочек), иначе — по мере значимости
    (деньгам/объёму). Из сущностей берём самую концентрированную (ТОП-5 объясняет больше)."""
    if not entities:
        return ""
    sub = df[_slice_mask(df, f)]
    if len(sub) < 5:
        return ""
    meas = _measure_by_name(measures, f.measure)
    is_rate = meas is not None and meas.kind in ("rate", "duration")
    best = None
    for ent in entities[:5]:
        if ent not in sub.columns or sub[ent].nunique() < 3:
            continue
        if is_rate and meas.kind == "rate":
            g = sub.groupby(ent)[meas.col].sum().sort_values(ascending=False)   # число случаев
            val_name, unit = f"{meas.label}, случаев", "шт"
        elif weight is not None:
            g = sub.groupby(ent)[weight.col].sum().sort_values(ascending=False)
            val_name, unit = weight.label, weight.unit
        else:
            g = sub.groupby(ent).size().sort_values(ascending=False)
            val_name, unit = "количество", "шт"
        total = float(g.sum())
        if total <= 0:
            continue
        topk = g.head(5)
        share = float(topk.sum()) / total * 100
        if best is None or share > best[0]:
            best = (share, ent, topk, val_name, unit)
    if best is None:
        return ""
    share, ent, topk, val_name, unit = best
    ent_lbl = _ds(labels, ent)
    lines = [f"Внутри этого среза ТОП-{len(topk)} по «{ent_lbl}» дают {share:.0f}% ({val_name}):", "",
             f"| {ent_lbl} | {val_name} |", "|---|---|"]
    for name, v in topk.items():
        lines.append(f"| {fmt_val(name)} | {_fmt_unit(float(v), unit)} |")
    return "\n".join(lines)


def _slice_mask(df, f: Finding) -> pd.Series:
    """Маска строк среза — по РЕАЛЬНЫМ колонкам (f.dims) и СЫРЫМ значениям (f.slice_vals),
    а не по отформатированным подписям."""
    m = pd.Series(True, index=df.index)
    for col, val in zip(f.dims, f.slice_vals):
        if col in df.columns:
            m &= df[col].astype(str) == str(val)
    return m


# ---------- рендер ----------
_SECTION = {
    "money_conc": ("💰 Где сосредоточены деньги и объёмы", "money_conc"),
    "value_mismatch": ("⚖️ Ценность важнее количества", "value_mismatch"),
    "rate_dev": ("🚨 Аномальные срезы", "deviation"),
    "interaction": ("🚨 Аномальные срезы", "deviation"),
}


def _finding_line(f: Finding) -> str:
    """Детерминированная строка с ЧИСЛАМИ находки (всегда видна, даже без нарратора)."""
    a = f.facts
    wname = a.get("weight_name") or "объём"
    wtail = f" · {wname}: {a['weight']}" if a.get("weight") else ""
    if f.kind == "rate_dev":
        return (f"**{a['slice']}**: {a['value']} — {a['direction']} среднего {a['overall']} "
                f"(×{a['lift']}). Записей: {a['n_rows']}{wtail}.")
    if f.kind == "interaction":
        return (f"Сочетание **{a['val_a']}** × **{a['val_b']}**: {a['value']} против "
                f"{a['overall']} в среднем — в {a['vs_expected']}× сильнее ожидаемого. "
                f"Записей: {a['n_rows']}{wtail}.")
    if f.kind == "money_conc":
        return (f"Лидер **{a['leader']}**: {a['leader_val']} ({a['leader_share']}% всего). "
                f"80% даёт {a['n_for_80']} из {a['total_categories']} категорий.")
    if f.kind == "value_mismatch":
        whole = "всех денег" if a.get("is_money") else f"всего объёма «{a.get('weight_name', '')}»"
        return (f"**{a['slice']}**: {a['weight']} — {a['w_share']}% {whole}, "
                f"но лишь {a['cnt_share']}% записей (ценность на единицу ×{a['ratio']}). "
                f"Записей: {a['n_rows']}.")
    return ""


def finding_to_result(f: Finding) -> AnalysisResult:
    """Находка → секция отчёта (для нарратора и рендера)."""
    facts = dict(f.facts)
    facts["_kind"] = f.kind
    facts["_line"] = _finding_line(f)
    key = "".join(ch if ch.isalnum() else "_" for ch in f.key)[:60]
    return AnalysisResult(key=f"mine_{key}", title=f.title, kind="mined", facts=facts,
                          table_md=f.drill_md, chart=f.chart)


def section_for(f: Finding) -> str:
    return _SECTION.get(f.kind, ("🚨 Аномальные срезы", "deviation"))[0]


def overview(df: pd.DataFrame, measures: list[Measure], dims: list[str], date: str | None,
             assets: Path, labels: Labels | None = None) -> list[AnalysisResult]:
    """Обзорные разрезы: главные показатели по 1-2 основным разрезам (не аномалии, а
    картина в целом) + динамика главной денежной меры."""
    df[ROW_COL] = 1 if ROW_COL not in df.columns else df[ROW_COL]
    dims = [d for d in dims if df[d].nunique(dropna=True) >= 2][:2]
    count = next((m for m in measures if m.kind == "count" and m.col == ROW_COL), None)
    money = _money_measure(measures)
    rates = [m for m in measures if m.kind == "rate"][:2]
    picks = ([count] if count else []) + ([money] if money else []) + rates   # кол-во сущностей — вперёд
    out: list[AnalysisResult] = []
    for m in picks:
        dfm = _pop(df, m)                          # бизнес-скоуп: только где мера заполнена
        for d in dims:
            if dfm[d].nunique(dropna=True) < 2:    # после скоупа разрез мог схлопнуться
                continue
            title = f"{m.title()} по «{_dt(labels, d)}»"
            f = Finding(kind="rate_dev" if m.kind == "rate" else "money_conc", measure=m.name,
                        unit=m.unit, dims=[d], title=title, facts={"dim": d}, score=0)
            chart = _bar_vs_baseline(dfm, f, measures, assets, labels)
            if chart:
                out.append(AnalysisResult(f"ovw_{m.col}_{d}", title, "overview",
                                          {"measure": m.label, "dim": _dt(labels, d), "unit": m.unit},
                                          "", chart))
    if date and money is not None:
        r = _trend(_pop(df, money), date, money, assets)
        if r:
            out.append(r)
    return out


def focus_answer(df: pd.DataFrame, reqs: dict, measures: list[Measure], assets: Path,
                 labels: Labels | None = None) -> list[AnalysisResult]:
    """Раздел «Ответ на ваш запрос»: строит ровно те разбивки (показатель×разрез×агрегат),
    что LLM извлёк из запроса пользователя — включая КОЛИЧЕСТВО задач по разрезам и СРЕДНИЕ."""
    df[ROW_COL] = 1 if ROW_COL not in df.columns else df[ROW_COL]
    out: list[AnalysisResult] = []
    seen: set = set()
    count_meas = next((m for m in measures if m.kind == "count" and m.col == ROW_COL),
                      Measure("Количество записей", ROW_COL, "sum", "count", ""))
    for b in reqs.get("breakdowns", [])[:8]:
        mn, dim, agg = b["measure"], b["dim"], b["agg"]
        if (mn, dim, agg) in seen or dim not in df.columns:
            continue
        seen.add((mn, dim, agg))
        meas = count_meas if mn == "__count__" else _measure_by_name(measures, mn)
        if meas is None:
            continue
        r = _bar_by_dim(df, meas, dim, agg, assets, labels)
        if r:
            out.append(r)
    return out


def _bar_by_dim(df, m: Measure, dim: str, agg: str, assets: Path, labels=None) -> AnalysisResult | None:
    """Бар «агрегат(показатель) по разрезу». agg: count|avg|sum. Для count/sum пустые/нулевые
    срезы отбрасываются (нет информации). Возвращает секцию фокус-ответа."""
    if dim not in df.columns:
        return None
    dfx = _pop(df, m) if agg != "count" else df
    is_rate = m.kind == "rate"
    if agg == "count":
        s = dfx.groupby(dim)[m.col].sum() if m.col == ROW_COL else dfx.groupby(dim)[m.col].count()
        unit, agg_lbl, pct = "шт", "Количество", False
    elif agg == "avg":
        s = dfx.groupby(dim)[m.col].mean()
        s = s * 100 if is_rate else s
        unit, agg_lbl, pct = ("%" if is_rate else m.unit), "Среднее", is_rate
    else:
        s = dfx.groupby(dim)[m.col].sum()
        unit, agg_lbl, pct = m.unit, "Всего", False
    s = s.dropna()
    if agg != "avg":
        s = s[s != 0]
    if s.empty:
        return None
    s = s.sort_values(ascending=False)
    total = float(s.sum())
    head = s.head(12)
    tbl = head.reset_index(); tbl.columns = [dim, "v"]; tbl[dim] = tbl[dim].map(_tick)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(tbl) + 0.6)))
    sns.barplot(data=tbl, y=dim, x="v", ax=ax, color="#3B7DD8")
    ax.margins(x=0.16)
    _annot(ax, tbl["v"].tolist(), horizontal=True, pct=pct)
    mlabel = m.label if not (agg == "count" and m.col == ROW_COL) else m.label
    title = f"{agg_lbl}: {mlabel} по «{_ds(labels, dim)}»"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(unit); ax.set_ylabel(""); ax.grid(axis="x", alpha=.3)
    chart = _save(fig, assets, f"focus_{agg}_{m.col}_{dim}")
    facts = {"agg": agg_lbl, "measure": m.label, "dim": _dt(labels, dim),
             "leader": fmt_val(s.index[0]), "leader_val": _fmt_unit(float(s.iloc[0]), unit),
             "leader_share": round(float(s.iloc[0]) / total * 100, 1) if agg != "avg" and total else None,
             "n_categories": int(s.shape[0])}
    return AnalysisResult(f"focus_{agg}_{m.col}_{dim}", title, "focus", facts, "", chart)


def _trend(df, date, m: Measure, assets: Path) -> AnalysisResult | None:
    d = pd.DataFrame({"_d": pd.to_datetime(df[date], errors="coerce"), "v": df[m.col]}).dropna()
    if d.empty:
        return None
    ts = d.set_index("_d").resample("MS")["v"].agg(m.agg).dropna()
    if len(ts) < 3:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(x=ts.index, y=ts.values, marker="o", ax=ax, color="#2E8B57")
    ax.set_title(f"Динамика: {m.label} по месяцам")
    ax.set_xlabel(""); ax.set_ylabel(m.unit)
    ax.grid(True, alpha=.3)
    chart = _save(fig, assets, f"trend_{m.col}")
    first, last = float(ts.iloc[0]), float(ts.iloc[-1])
    growth = ((last - first) / abs(first) * 100) if first else 0
    facts = {"direction": "рост" if growth > 5 else ("снижение" if growth < -5 else "стабильно"),
             "growth_pct": round(growth, 1), "first": _fmt(first), "last": _fmt(last),
             "peak_period": ts.idxmax().strftime("%Y-%m"), "measure": m.name}
    return AnalysisResult(f"trend_{m.col}", f"Динамика «{m.name}»", "overview", facts, "", chart)


def entity_ratings(df: pd.DataFrame, entities: list[str], measures: list[Measure],
                   assets: Path, labels: Labels | None = None) -> list[AnalysisResult]:
    """Рейтинги по КАЖДОЙ ключевой сущности (сотрудник/менеджер, ИНН/клиент, компания):
    ТОП по деньгам/объёму + концентрация. Так ИНН и клиенты точно получают аналитику,
    а не только «первая сущность»."""
    df[ROW_COL] = 1 if ROW_COL not in df.columns else df[ROW_COL]
    weight = _money_measure(measures) or next(
        (m for m in measures if m.kind in ("count", "value") and m.agg == "sum"), None)
    out: list[AnalysisResult] = []
    for ent in entities[:4]:
        if ent not in df.columns or df[ent].nunique(dropna=True) < 5:
            continue
        r = _entity_top(df, ent, weight, assets, labels)
        if r:
            out.append(r)
    return out


def _entity_top(df, ent: str, weight: Measure | None, assets: Path,
                labels=None) -> AnalysisResult | None:
    dfx = _pop(df, weight) if weight is not None else df
    if weight is not None:
        g = dfx.groupby(ent)[weight.col].sum().sort_values(ascending=False)
        val_name, unit, kind = weight.label, weight.unit, weight.kind
    else:
        g = dfx.groupby(ent).size().sort_values(ascending=False)
        val_name, unit, kind = "количество", "шт", "count"
    g = g[g > 0]
    if len(g) < 5:
        return None
    total = float(g.sum())
    cum = g.cumsum() / total
    n80 = int((cum <= 0.8).sum()) + 1
    head = g.head(12)
    tbl = head.reset_index(); tbl.columns = [ent, "v"]; tbl[ent] = tbl[ent].map(_tick)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(tbl) + 0.6)))
    sns.barplot(data=tbl, y=ent, x="v", ax=ax, color="#2E8B57")
    ax.margins(x=0.16)
    _annot(ax, tbl["v"].tolist(), horizontal=True, pct=(kind == "rate"))
    ent_lbl = _ds(labels, ent)
    ax.set_title(f"ТОП по «{ent_lbl}» ({val_name})", fontsize=11)
    ax.set_xlabel(unit); ax.set_ylabel(""); ax.grid(axis="x", alpha=.3)
    chart = _save(fig, assets, f"ent_{ent}_{weight.col if weight else 'cnt'}")
    facts = {"entity": ent_lbl, "leader": fmt_val(g.index[0]),
             "leader_val": _fmt_unit(float(g.iloc[0]), unit),
             "leader_share": round(float(g.iloc[0]) / total * 100, 1),
             "n_for_80": n80, "total_entities": int(len(g)), "value_name": val_name}
    return AnalysisResult(f"ent_{ent}", f"ТОП и концентрация по «{_dt(labels, ent)}»",
                          "entity", facts, "", chart)


def headline_kpi(df: pd.DataFrame, measures: list[Measure]) -> AnalysisResult:
    """Расширенные ключевые цифры: строки + каждый показатель в целом по таблице
    (доля %, деньги Σ, срок среднее, объём)."""
    rows = [("Всего записей", _fmt(len(df)))]
    for m in measures:
        try:
            if m.kind == "count" and m.col == ROW_COL:   # «Количество задач» — без Σ, просто число
                rows.append((m.label, _fmt(float(df[m.col].sum()))))
            elif m.kind == "rate":
                rows.append((m.label, f"{df[m.col].mean()*100:.0f}%"))
            elif m.kind == "duration":
                v = df[m.col].mean()
                rows.append((m.label, f"{v:.0f} {m.unit}" if pd.notna(v) else "—"))
            elif m.agg == "mean":                      # средняя ЗП и т.п. — среднее, не сумма
                rows.append((f"среднее {m.label}", _fmt_unit(float(df[m.col].mean()), m.unit)))
            elif m.kind == "money":
                rows.append((f"Σ {m.label}", _fmt_unit(float(df[m.col].sum()), m.unit)))
            else:
                rows.append((f"Σ {m.label}", _fmt(float(df[m.col].sum()))))
        except Exception:  # noqa: BLE001
            continue
    tbl = pd.DataFrame(rows, columns=["Показатель", "Значение"])
    return AnalysisResult("kpi", "Ключевые цифры", "kpi", {r[0]: r[1] for r in rows},
                          tbl.to_markdown(index=False))


# ---------- графики ----------
def _chart(df, f: Finding, measures: list[Measure], assets: Path, labels=None) -> str | None:
    try:
        m = _measure_by_name(measures, f.measure)
        if f.kind == "interaction":
            return _heatmap(_pop(df, m) if m else df, f, measures, assets, labels)
        if f.kind in ("rate_dev", "value_mismatch", "money_conc"):
            return _bar_vs_baseline(_pop(df, m) if m else df, f, measures, assets, labels)
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: график находки не построен (%s): %s", f.key, exc)
    return None


def _measure_by_name(measures, name) -> Measure | None:
    return next((m for m in measures if m.name == name), None)


def _tick(v) -> str:
    """Подпись категории на оси: целое как целое, обрезка длинного (без наезда)."""
    s = fmt_val(v)
    return s if len(s) <= 22 else s[:20] + "…"


def _annot(ax, values, horizontal=True, pct=False):
    """Числа на концах баров, компактно, без наложения."""
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        txt = f"{v:.0f}%" if pct else _fmt(v)
        if horizontal:
            ax.text(v, i, f" {txt}", va="center", ha="left", fontsize=8, color="#334")
        else:
            ax.text(i, v, txt, va="bottom", ha="center", fontsize=8, color="#334")


def _bar_vs_baseline(df, f: Finding, measures, assets, labels=None) -> str | None:
    m = _measure_by_name(measures, f.measure)
    if m is None:
        return None
    d = f.dims[0]
    agg = "mean" if m.kind in ("rate", "duration") else "sum"
    g = df.groupby(d)[m.col].agg(agg)
    nrows = df.groupby(d)[ROW_COL].sum()
    g = g[nrows >= max(30, int(len(df) * 0.005))]
    if g.empty:
        return None
    g = g.sort_values(ascending=False).head(12)
    vals = (g * 100) if m.kind == "rate" else g
    tbl = vals.reset_index(); tbl.columns = [d, "v"]
    tbl[d] = tbl[d].map(_tick)                       # int-как-int + обрезка длинных
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(tbl) + 0.6)))
    sns.barplot(data=tbl, y=d, x="v", ax=ax, color="#C0504D" if m.kind == "rate" else "#3B7DD8")
    if m.kind == "rate":
        ax.axvline(df[m.col].mean() * 100, color="gray", ls="--", lw=1)
    ax.margins(x=0.16)                               # место под числа справа
    _annot(ax, tbl["v"].tolist(), horizontal=True, pct=(m.kind == "rate"))
    ttl = f"{m.label} по «{_ds(labels, d)}»" + (" (пунктир — среднее)" if m.kind == "rate" else "")
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel(m.unit); ax.set_ylabel(""); ax.grid(axis="x", alpha=.3)
    return _save(fig, assets, f"mine_{f.kind}_{m.col}_{d}")


def _heatmap(df, f: Finding, measures, assets, labels=None) -> str | None:
    m = _measure_by_name(measures, f.measure)
    if m is None:
        return None
    da, db = f.dims
    agg = "mean" if m.kind in ("rate", "duration") else "sum"
    piv = df.pivot_table(index=da, columns=db, values=m.col, aggfunc=agg)
    # ограничим до топ-значений по объёму, иначе хитмап нечитаем
    top_a = df[da].value_counts().head(8).index
    top_b = df[db].value_counts().head(8).index
    piv = piv.reindex(index=top_a, columns=top_b)
    if piv.empty:
        return None
    if m.kind == "rate":
        piv = piv * 100
    piv.index = [_tick(x) for x in piv.index]        # int-как-int + обрезка
    piv.columns = [_tick(x) for x in piv.columns]
    fig, ax = plt.subplots(figsize=(min(12, 1.3 * len(piv.columns) + 3), min(9, 0.7 * len(piv.index) + 2)))
    sns.heatmap(piv, annot=True, fmt=".0f", cmap="Reds" if m.kind == "rate" else "Blues",
                ax=ax, cbar_kws={"label": m.unit}, linewidths=.5, annot_kws={"fontsize": 8})
    ax.set_title(f"{m.label}: {_ds(labels, da)} × {_ds(labels, db)}", fontsize=11)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    return _save(fig, assets, f"mine_heat_{m.col}_{da}_{db}")
