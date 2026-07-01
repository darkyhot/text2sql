"""Детекторы бизнес-закономерностей и аномалий (pandas, бизнес-формулировки).

В отличие от «математического» поиска отклонений, тут ищем то, что реально
интересно бизнесу и легко объяснимо:
  • Сезонность — сущность (ИНН/компания) повторяется в одни и те же месяцы
    разных лет (ушёл в июне-25 и июне-26 → похоже на сезонный отток);
  • Повторный/хронический отток — сущность появляется в нескольких периодах;
  • Новые и ушедшие — кто впервые появился в последнем периоде, кто исчез;
  • Всплески/провалы — периоды, заметно выше/ниже обычного уровня;
  • Аномальный лидер — сущность с показателем в разы выше типичного.

Каждый детектор возвращает AnalysisResult (kind='pattern') с посчитанными фактами
или None, если закономерность не выражена (чтобы не перегружать отчёт). Текст-вывод
пишет нарратор по фактам (как для остальных секций)."""

from __future__ import annotations

import pandas as pd

from .core import AnalysisResult, _fmt, _md_table, _save, agg_for, Roles

_MONTHS = {1: "январь", 2: "февраль", 3: "март", 4: "апрель", 5: "май", 6: "июнь",
           7: "июль", 8: "август", 9: "сентябрь", 10: "октябрь", 11: "ноябрь", 12: "декабрь"}


def detect_all(df: pd.DataFrame, roles: Roles, assets) -> list[AnalysisResult]:
    out: list[AnalysisResult] = []
    date = roles.dates[0] if roles.dates else None
    entity = roles.entities[0] if roles.entities else None
    metric = roles.metrics[0] if roles.metrics else None
    if date and entity:
        for det in (seasonality, repeat_churn):   # new_and_gone убран: для event-таблиц
            try:                                   # («ушедшие клиенты») «новые» вводят в заблуждение
                r = det(df, date, entity, assets)
                if r:
                    out.append(r)
            except Exception:  # noqa: BLE001
                pass
    if date and metric:
        try:
            r = spikes(df, date, metric, assets)
            if r:
                out.append(r)
        except Exception:  # noqa: BLE001
            pass
    if entity and metric:
        try:
            r = anomalous_leader(df, entity, metric)
            if r:
                out.append(r)
        except Exception:  # noqa: BLE001
            pass
    return out


def seasonality(df, date, entity, assets) -> AnalysisResult | None:
    d = pd.DataFrame({entity: df[entity], "_d": pd.to_datetime(df[date], errors="coerce")}).dropna()
    d["y"], d["m"] = d["_d"].dt.year, d["_d"].dt.month
    if d["y"].nunique() < 2:
        return None  # для сезонности нужно ≥2 лет
    # (сущность, месяц), встречающиеся в ≥2 разных годах
    yrs = d.groupby([entity, "m"])["y"].nunique()
    seasonal = yrs[yrs >= 2]
    if seasonal.empty:
        return None
    ent_seasonal = seasonal.index.get_level_values(0).nunique()
    total = d[entity].nunique()
    by_month = seasonal.reset_index().groupby("m")[entity].nunique().sort_values(ascending=False)
    top_month = int(by_month.index[0])
    examples = [str(x) for x in seasonal.reset_index()[entity].drop_duplicates().head(6)]
    tbl = by_month.rename("сезонных_сущностей").reset_index()
    tbl["m"] = tbl["m"].map(_MONTHS)
    tbl.columns = ["месяц", "сезонных_сущностей"]
    fig = None
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(tbl))))
        sns.barplot(data=tbl.head(12), y="месяц", x="сезонных_сущностей", ax=ax, color="#E8A33D")
        ax.set_title("Сезонность: сколько сущностей повторяется по месяцам")
        ax.set_xlabel(""); ax.set_ylabel("")
        chart = _save(fig, assets, f"season_{entity}")
    except Exception:  # noqa: BLE001
        chart = None
    facts = {"seasonal_count": int(ent_seasonal), "total_entities": int(total),
             "share_pct": round(ent_seasonal / total * 100, 1), "peak_month": _MONTHS[top_month],
             "peak_month_count": int(by_month.iloc[0]), "examples": examples,
             "entity_col": entity}
    return AnalysisResult(f"season_{entity}", f"🔁 Сезонность по «{entity}»", "pattern", facts,
                          _md_table(tbl.head(8), []), chart)


def repeat_churn(df, date, entity, assets) -> AnalysisResult | None:
    d = pd.DataFrame({entity: df[entity], "_p": pd.to_datetime(df[date], errors="coerce").dt.to_period("M")}).dropna()
    per_ent = d.groupby(entity)["_p"].nunique()
    total = len(per_ent)
    if total == 0:
        return None
    repeat = int((per_ent > 1).sum())
    share = round(repeat / total * 100, 1)
    if repeat < 3 or share < 5:
        return None
    top = per_ent.sort_values(ascending=False).head(8)
    tbl = top.rename("периодов").reset_index()
    tbl[entity] = tbl[entity].astype(str)
    facts = {"repeat_count": repeat, "total_entities": total, "repeat_share_pct": share,
             "max_periods": int(per_ent.max()), "top_repeater": str(per_ent.idxmax()),
             "entity_col": entity}
    return AnalysisResult(f"repeat_{entity}", f"🔂 Повторяемость по «{entity}»", "pattern", facts,
                          _md_table(tbl, []), None)


def new_and_gone(df, date, entity, assets) -> AnalysisResult | None:
    d = pd.DataFrame({entity: df[entity], "_p": pd.to_datetime(df[date], errors="coerce").dt.to_period("M")}).dropna()
    periods = sorted(d["_p"].unique())
    if len(periods) < 2:
        return None
    last, prev_set = periods[-1], set(periods[:-1])
    first_seen = d.groupby(entity)["_p"].min()
    last_seen = d.groupby(entity)["_p"].max()
    new_ent = int((first_seen == last).sum())
    gone_ent = int((last_seen < last).sum())
    if new_ent == 0 and gone_ent == 0:
        return None
    facts = {"last_period": str(last), "new_count": new_ent, "gone_count": gone_ent,
             "total_entities": int(d[entity].nunique()), "entity_col": entity}
    return AnalysisResult(f"newgone_{entity}", f"🆕 Новые и ушедшие «{entity}»", "pattern", facts, "", None)


def spikes(df, date, metric, assets) -> AnalysisResult | None:
    agg = agg_for(metric)
    d = pd.DataFrame({"_d": pd.to_datetime(df[date], errors="coerce"), metric: df[metric]}).dropna()
    ts = d.set_index("_d").resample("MS")[metric].agg(agg).dropna()
    if len(ts) < 4:
        return None
    med = ts.median()
    if not med:
        return None
    ratio = ts / med
    up = ratio[ratio >= 1.5]
    down = ratio[ratio <= 0.5]
    if up.empty and down.empty:
        return None
    facts = {"median": _fmt(med), "metric": metric}
    if not up.empty:
        peak = up.idxmax()
        facts.update({"spike_period": peak.strftime("%Y-%m"), "spike_ratio": round(float(up.max()), 1),
                      "spike_value": _fmt(ts.loc[peak])})
    if not down.empty:
        low = down.idxmin()
        facts.update({"dip_period": low.strftime("%Y-%m"), "dip_ratio": round(float(down.min()), 2),
                      "dip_value": _fmt(ts.loc[low])})
    return AnalysisResult(f"spikes_{metric}", f"📈 Всплески и провалы «{metric}»", "pattern", facts, "", None)


def anomalous_leader(df, entity, metric) -> AnalysisResult | None:
    agg = agg_for(metric)
    g = df.groupby(entity)[metric].agg(agg).sort_values(ascending=False)
    if len(g) < 5:
        return None
    med = g.median()
    if not med:
        return None
    ratio = g.iloc[0] / med
    if ratio < 4:
        return None  # лидер не «аномальный», а обычный
    facts = {"leader": str(g.index[0]), "leader_value": _fmt(g.iloc[0]),
             "median_value": _fmt(med), "times_above_median": round(float(ratio), 1),
             "entity_col": entity, "metric": metric}
    return AnalysisResult(f"anomlead_{entity}_{metric}", f"🚩 Аномальный лидер по «{entity}»", "pattern", facts, "", None)
