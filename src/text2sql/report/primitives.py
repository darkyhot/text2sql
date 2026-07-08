"""Библиотека аналитических примитивов (этап D, §8.1 дизайна).

Конечный набор ПРОВЕРЕННЫХ детерминированных операций над DataFrame Store. Каждый примитив:
вход — `FrameStore` + параметры, выход — `PrimitiveResult` (tidy-frame + `facts`). Никакого LLM.
Новая математика в системе появляется ТОЛЬКО как новый примитив с юнит-тестом на эталонных
числах; LLM примитивы не пишет и не меняет — только вызывает их через план (`plans.py`).

ЧЕСТНЫЕ УПРОЩЕНИЯ против §8.1 дизайна:
- Пакетная раскладка `insight/`/`engine/`/`semantics/` (§3.1) НЕ вводилась — примитивы живут
  рядом с движком в `report/`, поверх лёгкого `AggSpec`/`FrameStore`, а не полного pydantic
  `SemanticTable`/`QueryResultFrame` (§4/§5) — те не реализованы.
- Реализовано ядро, покрывающее плейбуки `loss_attribution` и `impact`: aggregate, decompose,
  pareto, trend_break, lift_profile, compare_groups, did. Причинные примитивы
  stratified_compare/selection_check/placebo_check/extrapolate/match_groups из §8.1 пока НЕ
  реализованы (точки расширения — тем же паттерном `@primitive`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .aggspec import AggSpec
from .labels import fmt_val
from .store import FrameStore


@dataclass
class PrimitiveResult:
    name: str
    frame: pd.DataFrame
    facts: dict = field(default_factory=dict)


_REGISTRY: dict[str, Callable[..., PrimitiveResult]] = {}


def primitive(fn: Callable[..., PrimitiveResult]) -> Callable[..., PrimitiveResult]:
    """Регистрирует примитив по имени функции (для вызова из плана как данных)."""
    _REGISTRY[fn.__name__] = fn
    return fn


def get(name: str) -> Callable[..., PrimitiveResult]:
    if name not in _REGISTRY:
        raise KeyError(f"неизвестный примитив {name!r}; есть: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


# ---------------- примитивы ----------------
@primitive
def aggregate(store: FrameStore, *, measure: str | None = None, by=(), agg: str = "sum",
              where: str | None = None, top: int | None = None, nonzero: bool = False) -> PrimitiveResult:
    """Обёртка AggregateRequest: мера × разрезы × агрегат × фильтр (§5.1)."""
    by = tuple(by) if not isinstance(by, str) else (by,)
    res = store.aggregate(AggSpec(measure=measure, by=by, agg=agg, where=where, top=top, nonzero=nonzero))
    facts = {"total": round(res.total, 2)}
    if by and not res.empty:
        facts["leader"] = fmt_val(res.frame.iloc[0][by[0]])
        facts["leader_value"] = round(float(res.frame.iloc[0]["value"]), 2)
    return PrimitiveResult("aggregate", res.frame, facts)


@primitive
def decompose(store: FrameStore, *, measure: str, dim: str, side: str = "loss") -> PrimitiveResult:
    """Вклад значений разреза в итог/потери со знаком (текущая механика расследования).
    side: loss (отрицательные) | gain (положительные) | all (по модулю)."""
    res = store.aggregate(AggSpec(measure=measure, by=(dim,), agg="sum", sort_desc=None))
    s = res.frame.set_index(dim)["value"].astype(float)
    if side == "loss":
        picked = s[s < 0].sort_values()
    elif side == "gain":
        picked = s[s > 0].sort_values(ascending=False)
    else:
        picked = s.reindex(s.abs().sort_values(ascending=False).index)
    denom = abs(float(picked.sum())) or 1.0
    tbl = picked.rename("contrib").reset_index()
    tbl["share"] = (tbl["contrib"].abs() / denom).round(4)
    facts = {"dim": dim, "n": int(len(tbl))}
    if len(tbl):
        facts["top_value"] = fmt_val(tbl[dim].iloc[0])
        facts["top_contrib"] = round(float(tbl["contrib"].iloc[0]), 2)
        facts["top_share_pct"] = round(float(tbl["share"].iloc[0]) * 100, 1)
    return PrimitiveResult("decompose", tbl, facts)


@primitive
def pareto(store: FrameStore, *, measure: str, entity: str, side: str = "loss") -> PrimitiveResult:
    """Концентрация по сущностям: сколько сущностей дают 80% (топ-N = X%)."""
    res = store.aggregate(AggSpec(measure=measure, by=(entity,), agg="sum", sort_desc=None))
    s = res.frame.set_index(entity)["value"].astype(float)
    s = (-s[s < 0]) if side == "loss" else (s[s > 0] if side == "gain" else s.abs())
    s = s.sort_values(ascending=False)
    total = float(s.sum()) or 1.0
    cum = (s.cumsum() / total)
    n80 = int((cum <= 0.8).sum()) + 1 if len(s) else 0
    tbl = s.rename("value").reset_index()
    tbl["cum_share"] = cum.values.round(4) if len(s) else []
    facts = {"entity": entity, "total_entities": int(len(s)), "n_for_80": n80,
             "top1_share_pct": round(float(s.iloc[0]) / total * 100, 1) if len(s) else 0.0}
    return PrimitiveResult("pareto", tbl, facts)


@primitive
def trend_break(store: FrameStore, *, measure: str, date: str, freq: str = "M",
                side: str = "loss") -> PrimitiveResult:
    """Динамика меры по периодам + детекция самого резкого сдвига (когда началось)."""
    d = pd.to_datetime(store.base[date], errors="coerce")
    v = _num(store.base, measure)
    ok = d.notna() & v.notna()
    ts = v[ok].groupby(d[ok].dt.to_period(freq)).sum().sort_index()
    tbl = ts.rename("value").reset_index()
    tbl[date] = tbl[date].astype(str)
    facts: dict = {"periods": int(len(ts))}
    if len(ts) >= 2:
        diff = ts.diff().dropna()
        cand = diff[diff < 0] if side == "loss" else (diff[diff > 0] if side == "gain" else diff)
        cand = cand if not cand.empty else diff
        brk = cand.abs().idxmax()
        facts["break_period"] = str(brk)
        facts["break_delta"] = round(float(diff.get(brk, 0.0)), 2)
    return PrimitiveResult("trend_break", tbl, facts)


@primitive
def lift_profile(store: FrameStore, *, measure: str, dim: str, side: str = "loss") -> PrimitiveResult:
    """«Почему»: доля значения в ПОТЕРЯХ против его доли в СТРОКАХ (lift к базе).
    Отделяет драйвер («причина X даёт 45% потерь при 18% строк — ×2.5») от фона."""
    df = store.base
    s = _num(df, measure)
    g = s.groupby(df[dim]).sum()
    g = g[g < 0] if side == "loss" else (g[g > 0] if side == "gain" else g)
    total_loss = abs(float(g.sum())) or 1.0
    rows = df.groupby(dim).size()
    total_rows = int(len(df)) or 1
    recs = []
    for val, contrib in g.items():
        ls = abs(float(contrib)) / total_loss
        bs = float(rows.get(val, 0)) / total_rows
        recs.append({"value": fmt_val(val), "loss_share": round(ls * 100, 1),
                     "base_share": round(bs * 100, 1), "lift": round((ls / bs) if bs > 0 else 0.0, 1)})
    tbl = pd.DataFrame(recs)
    facts: dict = {"dim": dim}
    if len(tbl):
        material = tbl[tbl["loss_share"] >= 5.0]
        best = (material if len(material) else tbl).sort_values("lift", ascending=False).iloc[0]
        facts.update({"driver": best["value"], "loss_share_pct": float(best["loss_share"]),
                      "base_share_pct": float(best["base_share"]), "lift": float(best["lift"])})
    return PrimitiveResult("lift_profile", tbl, facts)


@primitive
def compare_groups(store: FrameStore, *, measure: str, flag: str, agg: str = "mean") -> PrimitiveResult:
    """Наивное сравнение исходов группы с признаком (flag=1) и без (flag=0).
    ЯВНО помечается как наивное: без поправки на состав может завышать/занижать эффект."""
    df = store.base
    tr = _num(df, flag) > 0
    m = _num(df, measure)
    def _agg(mask):
        x = m[mask].dropna()
        return float(getattr(x, agg)()) if len(x) else float("nan")
    v1, v0 = _agg(tr), _agg(~tr)
    tbl = pd.DataFrame({"group": ["с признаком", "без признака"], "value": [v1, v0],
                        "n": [int(tr.sum()), int((~tr).sum())]})
    facts = {"measure": measure, "flag": flag, "with": round(v1, 4), "without": round(v0, 4),
             "diff": round(v1 - v0, 4), "note": "наивное сравнение — без поправки на состав групп"}
    return PrimitiveResult("compare_groups", tbl, facts)


@primitive
def did(store: FrameStore, *, measure: str, flag: str, time_col: str, cutoff: str,
        agg: str = "mean") -> PrimitiveResult:
    """Difference-in-differences: Δ(пилот, после−до) − Δ(контроль, после−до).
    Убирает и разницу составов, и общий тренд времени — честная оценка эффекта."""
    df = store.base
    t = pd.to_datetime(df[time_col], errors="coerce")
    post = t >= pd.to_datetime(cutoff)
    treat = _num(df, flag) > 0
    m = _num(df, measure)
    def _cell(a, b):
        x = m[a & b].dropna()
        return float(getattr(x, agg)()) if len(x) else float("nan")
    tp, tpre = _cell(treat, post), _cell(treat, ~post)
    cp, cpre = _cell(~treat, post), _cell(~treat, ~post)
    did_val = (tp - tpre) - (cp - cpre)
    tbl = pd.DataFrame({"group": ["пилот", "контроль"], "before": [tpre, cpre], "after": [tp, cp],
                        "delta": [tp - tpre, cp - cpre]})
    facts = {"did": round(did_val, 4), "treat_delta": round(tp - tpre, 4),
             "control_delta": round(cp - cpre, 4), "cutoff": str(cutoff)}
    return PrimitiveResult("did", tbl, facts)


@primitive
def stratified_compare(store: FrameStore, *, measure: str, flag: str, strata,
                       agg: str = "mean") -> PrimitiveResult:
    """Сравнение с/без ВНУТРИ однородных ячеек (стратификация), эффект = взвешенное по
    размеру страты среднее пострат-разниц. Убирает смещение состава + устойчивость:
    в какой доле страт эффект в ту же сторону (§8.3, шаг 3a/4)."""
    df = store.base
    strata = [strata] if isinstance(strata, str) else list(strata)
    tr = _num(df, flag) > 0
    m = _num(df, measure)
    rows = []
    for keys, sub in df.groupby(strata):
        idx = sub.index
        a = m.loc[idx][tr.loc[idx]].dropna()
        b = m.loc[idx][~tr.loc[idx]].dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        rows.append({"stratum": keys if isinstance(keys, tuple) else (keys,),
                     "effect": float(getattr(a, agg)()) - float(getattr(b, agg)()),
                     "weight": int(len(sub)), "n_treat": int(len(a)), "n_ctrl": int(len(b))})
    tbl = pd.DataFrame(rows)
    if len(tbl):
        weff = float((tbl["effect"] * tbl["weight"]).sum() / tbl["weight"].sum())
        pos = float((tbl["effect"] > 0).mean())
    else:
        weff, pos = float("nan"), 0.0
    facts = {"weighted_effect": round(weff, 4), "n_strata": int(len(tbl)),
             "share_positive": round(pos, 3)}
    return PrimitiveResult("stratified_compare", tbl, facts)


@primitive
def selection_check(store: FrameStore, *, measure: str, flag: str, time_col: str, cutoff: str,
                    agg: str = "mean") -> PrimitiveResult:
    """Проверка самоотбора: исходы будущих пилотов vs контроль В ДО-ПЕРИОДЕ. Если пилоты
    были лучше ещё до внедрения — наивную разницу нельзя приписывать инструменту (§8.3, шаг 4)."""
    df = store.base
    pre = pd.to_datetime(df[time_col], errors="coerce") < pd.to_datetime(cutoff)
    tr = _num(df, flag) > 0
    m = _num(df, measure)
    def _cell(mask):
        x = m[mask].dropna()
        return float(getattr(x, agg)()) if len(x) else float("nan")
    tpre, cpre = _cell(tr & pre), _cell(~tr & pre)
    gap = tpre - cpre
    facts = {"pre_gap": round(gap, 4), "treated_pre": round(tpre, 4), "control_pre": round(cpre, 4),
             "self_selection": bool(abs(gap) >= 0.02)}
    return PrimitiveResult("selection_check", pd.DataFrame([facts]), facts)


@primitive
def placebo_check(store: FrameStore, *, placebo_measure: str, flag: str, time_col: str,
                  cutoff: str, agg: str = "mean") -> PrimitiveResult:
    """Плацебо: DiD на мере, на которую инструмент влиять НЕ должен. Эффект там ≈ 0 = честно."""
    r = did(store, measure=placebo_measure, flag=flag, time_col=time_col, cutoff=cutoff, agg=agg)
    val = float(r.facts["did"])
    facts = {"placebo_did": round(val, 4), "clean": bool(abs(val) < 0.02)}
    return PrimitiveResult("placebo_check", r.frame, facts)


@primitive
def extrapolate(store: FrameStore, *, uplift: float, adoption: float,
                volume_measure: str | None = None, where: str | None = None) -> PrimitiveResult:
    """Перенос эффекта на НЕОХВАЧЕННЫЙ объём: прирост = uplift × (1−adoption) × объём.
    Консервативно, с явными допущениями (состав не меняется). uplift/adoption — доли."""
    df = store.base if not where else store.base.query(where)
    total = float(len(df)) if volume_measure is None else float(_num(df, volume_measure).sum())
    uncovered = max(0.0, 1.0 - float(adoption)) * total
    projected = float(uplift) * uncovered
    facts = {"uplift": round(float(uplift), 4), "adoption": round(float(adoption), 4),
             "uncovered_volume": round(uncovered, 1), "projected": round(projected, 1)}
    return PrimitiveResult("extrapolate", pd.DataFrame([facts]), facts)


# ---------------- self-test на эталонных числах ----------------
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    n = 400
    seg = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    # B — «потери»: diff отрицательный и крупный; A/C — около нуля/плюс
    diff = np.where(seg == "B", -rng.integers(50, 150, n), rng.integers(-5, 30, n)).astype(float)
    reason = np.where(seg == "B", "закрытие", rng.choice(["прочее", "перевод"], n))
    ent = rng.choice([f"inn{i}" for i in range(20)], size=n)
    treat = (seg == "A").astype(int)                       # пилот = сегмент A
    dt = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    # исход: закрытие; у пилота после cutoff — заметно выше
    cutoff = pd.Timestamp("2025-07-01")
    base = 0.5 + 0.2 * (seg == "A")
    uplift = 0.15 * ((dt >= cutoff) & (treat == 1))
    outcome = (base + uplift).clip(0, 1)
    df = pd.DataFrame({"seg": seg, "diff": diff, "reason": reason, "inn": ent,
                       "treat": treat, "dt": dt, "closed": outcome})
    st = FrameStore(df)

    r = decompose(st, measure="diff", dim="seg", side="loss")
    assert r.facts["top_value"] == "B", r.facts
    assert r.facts["top_share_pct"] > 80, r.facts       # B доминирует в потерях

    lp = lift_profile(st, measure="diff", dim="reason", side="loss")
    assert lp.facts["driver"] == "закрытие" and lp.facts["lift"] > 1.5, lp.facts

    pr = pareto(st, measure="diff", entity="inn", side="loss")
    assert pr.facts["total_entities"] >= 10 and pr.facts["n_for_80"] >= 1, pr.facts

    tb = trend_break(st, measure="diff", date="dt", side="loss")
    assert tb.facts["periods"] >= 6 and "break_period" in tb.facts, tb.facts

    cg = compare_groups(st, measure="closed", flag="treat", agg="mean")
    assert cg.facts["diff"] > 0, cg.facts                # наивно пилот выше

    dd = did(st, measure="closed", flag="treat", time_col="dt", cutoff="2025-07-01")
    assert 0.08 < dd.facts["did"] < 0.22, dd.facts       # DiD ≈ заложенный uplift 0.15

    # --- причинные примитивы на отдельной чистой фикстуре ---
    n2 = 600
    region = rng.choice(["R1", "R2", "R3"], size=n2)
    tr2 = rng.random(n2) < 0.35                          # пилот есть во всех регионах
    post2 = rng.random(n2) < 0.5
    reg_base = pd.Series(region).map({"R1": 0.5, "R2": 0.55, "R3": 0.6}).to_numpy()
    y = reg_base + 0.10 * (tr2 & post2)                  # эффект +0.10 только у пилота после
    placebo = reg_base + rng.normal(0, 0.01, n2)         # плацебо: зависит от региона, НЕ от пилота/периода
    dt2 = np.where(post2, pd.Timestamp("2025-09-01").value, pd.Timestamp("2025-03-01").value)
    df2 = pd.DataFrame({"region": region, "treat": tr2.astype(int),
                        "y": y, "placebo": placebo, "dt": pd.to_datetime(dt2)})
    st2 = FrameStore(df2)

    sc = stratified_compare(st2, measure="y", flag="treat", strata="region", agg="mean")
    assert sc.facts["n_strata"] == 3, sc.facts           # эффект «размазан» по до/после → величина мягче

    sel = selection_check(st2, measure="y", flag="treat", time_col="dt", cutoff="2025-06-01")
    assert abs(sel.facts["pre_gap"]) < 0.05, sel.facts   # до внедрения пилот≈контроль (нет самоотбора)

    pl = placebo_check(st2, placebo_measure="placebo", flag="treat", time_col="dt", cutoff="2025-06-01")
    assert pl.facts["clean"], pl.facts                   # плацебо-эффект ≈ 0

    ex = extrapolate(st2, uplift=0.10, adoption=0.2)
    assert abs(ex.facts["projected"] - 0.10 * 0.8 * n2) < 1e-6, ex.facts   # 0.1×0.8×600=48

    print("primitives self-test OK:", names())
