"""Режим «Расследование»: не витрина слайсов, а ДИАГНОСТИКА конкретного вопроса
(«где мы потеряли 500к человек», «откуда взялся отток», «почему упал X»).

Как работает аналитик — так и здесь:
  1. РАМКА: LLM по вопросу выбирает целевую величину (target) и режим — «изменение»
     (со знаком: потери=отрицательные, net/gross) или «величина» (отток/количество).
  2. АТРИБУЦИЯ: детерминированно раскладываем target по разрезам — сколько КАЖДЫЙ
     внёс в потери (вклад, доля), где концентрация.
  3. СПУСК по дереву (где-внутри-где): идём в самый крупный вклад, раскладываем внутри,
     до концентрации/сущностей. Пороги материальности, лимит глубины, честный вывод
     «размазано», если концентрации нет.
  4. ПОЧЕМУ: разрезы-причины (причина оттока/статус) + СЛАГАЕМЫЕ target (напр. самозанятые
     vs ГПХ) — структурный сдвиг.
  5. КТО: конкретные сущности (компании/ИНН), составляющие потери, + их ЦЕННОСТЬ (потенциал)
     для приоритета возврата.
  6. СИНТЕЗ: LLM собирает причинную цепочку + Парето действий.

Всё числовое — pandas; LLM только рамка и синтез (надёжно, дёшево)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import PATHS
from . import core, labels as labels_mod, metrics, plan
from .builder import (_HTML_CSS, _img_b64, _load, _load_columns, _md_block_to_html,
                      _md_inline_to_html, _meta_for, _rel_chart)
from .core import _fmt, _save
from .labels import Labels, fmt_val
from .metrics import Measure, ROW_COL

logger = logging.getLogger(__name__)

_C_LOSS, _C_GAIN, _C_NEUT = "#C0504D", "#2E8B57", "#3B7DD8"


# ---------- подготовка данных (загрузка, роли, меры) ----------
@dataclass
class Prep:
    df: pd.DataFrame
    table_desc: str
    meta: dict
    lbls: Labels
    roles: object
    measures: list[Measure]


def _prepare(db, catalog, llm, fqn, where, progress) -> Prep:
    schema, table = fqn.split(".", 1)
    progress("загружаю данные…")
    df = _load(db, schema, table, where, _load_columns(catalog, fqn))
    if df.empty:
        raise ValueError("По заданному фильтру нет данных.")
    table_desc, meta = _meta_for(catalog, fqn)
    progress("подписи и единицы измерения…")
    lbls = labels_mod.build_labels_llm(llm, table_desc, meta, list(df.columns), df)
    roles = plan.build_roles_llm(llm, table_desc, df, meta) or core.profile(df, meta)
    for c in df.columns:
        if (c not in roles.entities and c not in roles.dimensions
                and core._ENTITY_RE.search(c) and df[c].nunique(dropna=True) > 10):
            roles.entities.append(c)
    core.normalize_roles(roles, list(df.columns))
    progress("показатели…")
    behav, rec = metrics.build_behaviors_llm(llm, table_desc, df, meta)
    measures = metrics.build_derived(df, behav, meta)
    metrics.money_from_metrics(df, meta, measures, roles.metrics, lbls)
    cm = metrics.record_count_measure(df, rec)
    if cm:
        measures.insert(0, cm)
    return Prep(df, table_desc, meta, lbls, roles, measures)


# ---------- рамка расследования (LLM) ----------
@dataclass
class Frame:
    target: str
    mode: str                # change | magnitude
    direction: str           # loss | gain | top
    drill_dims: list[str] = field(default_factory=list)
    why_dims: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    entity: str | None = None
    value: str | None = None
    restated: str = ""


_FRAME_SYS = (
    "Ты аналитик-исследователь. По ВОПРОСУ пользователя определи, ЧТО объяснять и как. "
    "Верни JSON:\n"
    "{\"target\": имя числовой колонки — величина, которую объясняем (изменение/убыль/отток/"
    "количество); \"mode\": \"change\" если это ИЗМЕНЕНИЕ со знаком (потери=отрицательные) | "
    "\"magnitude\" если положительная величина (отток/кол-во); \"direction\": \"loss\" (ищем "
    "где падение/потери) | \"gain\" | \"top\" (где сосредоточено); \"drill_dims\": [разрезы по "
    "убыванию важности для дерева «где внутри где»: сегмент, территория, холдинг…]; \"why_dims\": "
    "[разрезы-ПРИЧИНЫ: причина оттока, статус…]; \"components\": [колонки-СЛАГАЕМЫЕ target, если "
    "он раскладывается на части, напр. самозанятые+ГПХ]|[]; \"entity\": колонка-сущность "
    "(компания/ИНН — конкретные виновники)|null; \"value\": колонка-ЦЕННОСТЬ для приоритета "
    "(потенциал/сумма)|null; \"restated\": суть вопроса одной фразой}\n"
    "Бери ТОЛЬКО имена из предложенных списков."
)


def _frame(llm, question, prep: Prep) -> Frame:
    df = prep.df
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    lines = []
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        neg = "со знаком(есть отриц.)" if (s.min() is not None and s.min() < 0) else "положит."
        lines.append(f"- {c}: {prep.lbls.of(c)} | {neg} | {prep.meta.get(c, {}).get('desc', '')[:60]}")
    dims = "\n".join(f"- {d}: {prep.lbls.of(d)}" for d in prep.roles.dimensions)
    ents = "\n".join(f"- {e}: {prep.lbls.of(e)}" for e in prep.roles.entities)
    user = (f"Вопрос: {question}\n\nЧисловые колонки:\n" + "\n".join(lines)
            + f"\n\nРазрезы:\n{dims}\n\nСущности:\n{ents}")
    try:
        out = llm.complete_json(_FRAME_SYS, user, max_tokens=2000, node="investigate_frame")
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: рамка не построена (%s)", exc)
        out = {}
    numset, dimset = set(num_cols), set(prep.roles.dimensions)
    entset = set(prep.roles.entities)
    target = out.get("target") if out.get("target") in numset else (num_cols[0] if num_cols else None)
    if target is None:
        raise ValueError("Не найдено числовой величины для расследования.")
    s = pd.to_numeric(df[target], errors="coerce")
    mode = out.get("mode") if out.get("mode") in ("change", "magnitude") else \
        ("change" if (s.min() is not None and s.min() < 0) else "magnitude")
    return Frame(
        target=target, mode=mode,
        direction=out.get("direction") if out.get("direction") in ("loss", "gain", "top")
        else ("loss" if mode == "change" else "top"),
        drill_dims=[d for d in (out.get("drill_dims") or []) if d in dimset] or list(prep.roles.dimensions),
        why_dims=[d for d in (out.get("why_dims") or []) if d in dimset],
        components=[c for c in (out.get("components") or []) if c in numset and c != target],
        entity=(out.get("entity") if out.get("entity") in entset else (prep.roles.entities[0] if prep.roles.entities else None)),
        value=(out.get("value") if out.get("value") in numset else None),
        restated=str(out.get("restated") or question).strip(),
    )


# ---------- декомпозиция вклада ----------
@dataclass
class Contrib:
    dim: str
    table: pd.DataFrame        # index=value, cols: contrib, n, share
    total: float               # знаковый итог узла (net) или величина
    loss: float                # сумма отрицательных вкладов (для change), иначе = total
    top_share: float           # доля крупнейшего вклада в |loss|
    n_for_half: int            # сколько значений даёт 50% |loss|


def _decompose(df: pd.DataFrame, target: str, dim: str, mode: str,
               ref: float | None = None) -> Contrib:
    """ref — знаменатель для ДОЛИ (обычно общие потери всей таблицы), чтобы «доля лидера»
    была сопоставима между уровнями («49% ВСЕХ потерь»), а не 100% локально."""
    s = pd.to_numeric(df[target], errors="coerce")
    g = s.groupby(df[dim]).sum()
    n = df.groupby(dim)[target].size()
    total = float(g.sum())
    if mode == "change":
        neg, pos = float(g[g < 0].sum()), float(g[g > 0].sum())
        if abs(neg) >= abs(pos) and neg < 0:
            loss, asc = neg, True                    # потери: самые отрицательные вперёд
        else:
            loss, asc = pos, False                   # прирост: самые крупные вперёд
        contrib = g.sort_values(ascending=asc)
    else:
        loss, asc = total, False
        contrib = g.sort_values(ascending=False)
    ref_denom = abs(ref) if ref else (abs(loss) or 1.0)   # доля — от общей стороны
    local_denom = abs(loss) or 1.0                        # концентрация — внутри узла
    share = (contrib / (ref if ref else loss) * 100).clip(lower=0)
    tbl = pd.DataFrame({"contrib": contrib, "n": n.reindex(contrib.index), "share": share})
    cum = tbl["contrib"].abs().cumsum() / local_denom
    n_half = int((cum <= 0.5).sum()) + 1
    top_share = min(100.0, float(abs(tbl["contrib"].iloc[0]) / ref_denom * 100))
    return Contrib(dim, tbl, total, loss, top_share, n_half)


# ---------- спуск по дереву ----------
@dataclass
class Level:
    dim: str
    contrib: Contrib
    followed: object
    chart: str | None


def _drill(df, frame: Frame, assets, lbls, ref, *, max_depth=3, min_rows=30,
           min_top_share=15.0) -> list[Level]:
    # разрезы для дерева: не сущности (они — «кто»), достаточно значений и объёма
    ent_concepts = {core._concept(e) for e in [frame.entity] if frame.entity}
    cand = [d for d in frame.drill_dims
            if d in df.columns and core._concept(d) not in ent_concepts
            and 2 <= df[d].nunique(dropna=True) <= 60]
    levels: list[Level] = []
    sub = df
    for _ in range(max_depth):
        cand = [d for d in cand if sub[d].nunique(dropna=True) >= 2]
        if not cand or len(sub) < min_rows:
            break
        # выбираем разрез с максимальной концентрацией вклада (детерминированно)
        decs = [_decompose(sub, frame.target, d, frame.mode, ref) for d in cand]
        decs = [c for c in decs if np.isfinite(c.top_share)]
        if not decs:
            break
        best = max(decs, key=lambda c: c.top_share)
        chart = _contrib_chart(best, frame, assets, lbls, path_len=len(levels))
        follow = best.table.index[0]                 # крупнейший вклад
        levels.append(Level(best.dim, best, follow, chart))
        # стоп, если потери «размазаны» (нет доминирующего) — глубже не идём
        if best.top_share < min_top_share:
            break
        sub = sub[sub[best.dim].astype(str) == str(follow)]
        cand = [d for d in cand if d != best.dim]
    return levels


# ---------- «почему» и «кто» ----------
def _why(df, frame: Frame, assets, lbls, ref) -> list[tuple[str, Contrib, str | None]]:
    out = []
    for d in frame.why_dims[:2]:
        if d in df.columns and 2 <= df[d].nunique(dropna=True) <= 40:
            c = _decompose(df, frame.target, d, frame.mode, ref)
            out.append((d, c, _contrib_chart(c, frame, assets, lbls, tag="why")))
    return out


def _components(df, frame: Frame, lbls) -> list[tuple[str, float]]:
    """Структурный разбор target на слагаемые (напр. самозанятые vs ГПХ)."""
    res = []
    for c in frame.components:
        v = float(pd.to_numeric(df[c], errors="coerce").sum())
        res.append((lbls.of(c), v))
    return res


def _who(df, frame: Frame, assets, lbls) -> tuple[pd.DataFrame | None, str | None]:
    ent = frame.entity
    if not ent or ent not in df.columns:
        return None, None
    s = pd.to_numeric(df[frame.target], errors="coerce")
    g = s.groupby(df[ent]).sum()
    g = g.sort_values() if frame.mode == "change" else g.sort_values(ascending=False)
    top = g.head(12)
    if frame.mode == "change":
        top = top[top < 0]
    if top.empty:
        return None, None
    val = None
    if frame.value and frame.value in df.columns:
        vv = pd.to_numeric(df[frame.value], errors="coerce").groupby(df[ent]).sum()
        val = vv.reindex(top.index)
    tbl = pd.DataFrame({"contrib": top})
    if val is not None:
        tbl["value"] = val
    chart = _entity_chart(top, ent, frame, assets, lbls)
    return tbl, chart


# ---------- графики ----------
def _tick(v):
    s = fmt_val(v)
    return s if len(s) <= 24 else s[:22] + "…"


def _contrib_chart(c: Contrib, frame: Frame, assets, lbls, *, path_len=0, tag="lvl") -> str | None:
    try:
        t = c.table.head(12).iloc[::-1]              # крупнейшие сверху
        vals = t["contrib"].astype(float).values
        names = [_tick(x) for x in t.index]
        colors = [_C_LOSS if v < 0 else _C_GAIN for v in vals] if frame.mode == "change" else \
            [_C_NEUT] * len(vals)
        fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(t) + 0.6)))
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=9)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {_fmt(abs(v))}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color="#334")
        ax.axvline(0, color="#888", lw=1)
        ax.margins(x=0.16)
        side = "потери" if c.loss < 0 else "прирост"
        ttl = f"Вклад в {side}: {lbls.of(frame.target)} по «{lbls.of(c.dim)}»"
        ax.set_title(ttl, fontsize=11)
        ax.grid(axis="x", alpha=.3)
        return _save(fig, assets, f"inv_{tag}{path_len}_{c.dim}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: график вклада не построен: %s", exc)
        return None


def _entity_chart(top: pd.Series, ent: str, frame: Frame, assets, lbls) -> str | None:
    try:
        t = top.iloc[::-1]
        vals = t.astype(float).values
        fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(t) + 0.6)))
        ax.barh(range(len(vals)), vals, color=_C_LOSS if frame.mode == "change" else _C_NEUT)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels([_tick(x) for x in t.index], fontsize=9)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {_fmt(abs(v))}", va="center", ha="right" if v < 0 else "left",
                    fontsize=8, color="#334")
        ax.axvline(0, color="#888", lw=1); ax.margins(x=0.16)
        ax.set_title(f"Кто: ТОП по «{lbls.of(ent)}» ({lbls.of(frame.target)})", fontsize=11)
        ax.grid(axis="x", alpha=.3)
        return _save(fig, assets, f"inv_who_{ent}")
    except Exception:  # noqa: BLE001
        return None


# ---------- синтез (LLM) ----------
_SYNTH_SYS = (
    "Ты пишешь ВЫВОД расследования простым языком для руководителя. По фактам дай:\n"
    "answer — прямой ответ на вопрос (1-2 предложения, где именно сосредоточено);\n"
    "chain — причинная ЦЕПОЧКА: total → главный разрез → внутри него → причина → кто "
    "(3-5 коротких пунктов с числами);\n"
    "actions — 2-4 конкретных действия (куда смотреть/что делать, приоритет по ценности).\n"
    "СТРОГО опирайся на ФАКТЫ, НИЧЕГО не додумывай:\n"
    "• «Причину» бери как ЗНАЧЕНИЕ разреза-причины (facts.почему[].разрез = такая-то колонка, "
    "лидер = такое-то значение). Формулируй «по <разрез> преобладает значение «X» (N%)», а НЕ "
    "как доказанный механизм.\n"
    "• Сущности из facts.кто называй их ТИПОМ (facts.кто.сущность, напр. «компании по ИНН», "
    "«клиенты»), и НЕ путай с целевой величиной (facts.целевая) — ИНН это компания, а не физлицо.\n"
    "• Числа и названия — только из фактов.\n"
    "ЗАПРЕЩЕНО: матжаргон. Если потери РАЗМАЗАНЫ (нет доминирующего разреза) — честно скажи "
    "это в answer. Верни JSON: {\"answer\":\"...\",\"chain\":[...],\"actions\":[...]}"
)


def _synthesize(llm, question, facts: dict) -> dict:
    try:
        out = llm.complete_json(_SYNTH_SYS, f"Вопрос: {question}\n\nФакты:\n{facts}",
                                max_tokens=3000, node="investigate_synth")
        return {"answer": str(out.get("answer", "")).strip(),
                "chain": [str(x).strip() for x in (out.get("chain") or []) if str(x).strip()],
                "actions": [str(x).strip() for x in (out.get("actions") or []) if str(x).strip()]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: синтез не удался (%s)", exc)
        return {"answer": "", "chain": [], "actions": []}


# ---------- оркестратор ----------
def investigate(db, catalog, llm, fqn: str, question: str, *, where: str | None = None,
                out_dir: Path | None = None, progress=lambda m: None) -> dict:
    schema, table = fqn.split(".", 1)
    out_dir = out_dir or PATHS.workspace_dir
    assets = out_dir / "report_assets" / f"{table}_investigate"
    if assets.exists():
        import shutil
        shutil.rmtree(assets, ignore_errors=True)

    prep = _prepare(db, catalog, llm, fqn, where, progress)
    progress("формулирую цель расследования…")
    frame = _frame(llm, question, prep)
    df = prep.df
    s = pd.to_numeric(df[frame.target], errors="coerce")
    net = float(s.sum())
    gross_loss = float(s[s < 0].sum()) if frame.mode == "change" else None
    gross_gain = float(s[s > 0].sum()) if frame.mode == "change" else None
    unit = _target_unit(prep, frame.target)
    # знаменатель для «доли» — доминирующая сторона всей таблицы (все потери/весь прирост)
    if frame.mode == "change":
        ref = gross_loss if abs(gross_loss) >= abs(gross_gain) else gross_gain
    else:
        ref = net
    ref = ref or 1.0

    progress("раскладываю потери по разрезам (спуск по дереву)…")
    levels = _drill(df, frame, assets, prep.lbls, ref)
    progress("ищу причины…")
    why = _why(df, frame, assets, prep.lbls, ref)
    comps = _components(df, frame, prep.lbls)
    progress("нахожу конкретных «виновников»…")
    who_tbl, who_chart = _who(df, frame, assets, prep.lbls)

    facts = _facts_for_synth(frame, prep.lbls, net, gross_loss, gross_gain, unit, levels, why, comps, who_tbl)
    progress("собираю причинную цепочку и действия…")
    synth = _synthesize(llm, question, facts)

    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = dict(prep=prep, frame=frame, table=table, fqn=fqn, question=question, where=where,
               net=net, gross_loss=gross_loss, gross_gain=gross_gain, unit=unit,
               levels=levels, why=why, comps=comps, who_tbl=who_tbl, who_chart=who_chart, synth=synth)
    md_path = out_dir / f"{table}_investigation.md"
    html_path = out_dir / f"{table}_investigation.html"
    md_path.write_text(_assemble_md(**ctx), encoding="utf-8")
    html_path.write_text(_assemble_html(**ctx), encoding="utf-8")
    return {"md_path": str(md_path), "html_path": str(html_path),
            "levels": len(levels), "rows": len(df)}


def _target_unit(prep: Prep, col: str) -> str:
    m = next((x for x in prep.measures if x.tech == col or x.col == col), None)
    if m:
        return m.unit
    uk = prep.lbls.unit_kind(col)
    return {"money": "₽", "people": "чел.", "percent": "%"}.get(uk, "")


def _fmt_u(v, unit):
    return f"{_fmt(v)} {unit}".strip()


def _side(loss) -> str:
    return "потерь" if loss < 0 else "прироста"


def _facts_for_synth(frame, lbls, net, gloss, ggain, unit, levels, why, comps, who_tbl) -> dict:
    f = {"целевая": f"{lbls.of(frame.target)} (единица: {unit or 'шт'})", "режим": frame.mode,
         "итог_net": _fmt_u(net, unit)}
    if gloss is not None:
        f["всего_потеряно"] = _fmt_u(gloss, unit)
        f["всего_приросло"] = _fmt_u(ggain, unit)
    f["дерево"] = [{
        "разрез": lbls.of(l.dim), "лидер": fmt_val(l.followed),
        "вклад_лидера": _fmt_u(float(l.contrib.table["contrib"].iloc[0]), unit),
        "доля_лидера_%": round(l.contrib.top_share, 1),
        "N_на_половину": l.contrib.n_for_half,
    } for l in levels]
    f["почему"] = [{"разрез": lbls.of(d), "преобладает_значение": fmt_val(c.table.index[0]),
                    "доля_%": round(c.top_share, 1)} for d, c, _ in why]
    if comps:
        f["слагаемые"] = [{"часть": name, "значение": _fmt_u(v, unit)} for name, v in comps]
    if who_tbl is not None:
        f["кто"] = {"сущность": lbls.of(frame.entity) if frame.entity else "сущность",
                    "топ_значения": [fmt_val(i) for i in who_tbl.index[:6]]}
    return f


# ---------- рендер ----------
def _assemble_md(prep, frame, table, fqn, question, where, net, gross_loss, gross_gain, unit,
                 levels, why, comps, who_tbl, who_chart, synth) -> str:
    L = [f"# 🔎 Расследование: {prep.table_desc}", f"**Вопрос:** {question}\n"]
    meta = f"**Таблица:** `{fqn}` · **строк:** {len(prep.df):,}".replace(",", " ")
    if where:
        meta += f" · **фильтр:** `{where}`"
    L.append(meta + "\n")
    if synth.get("answer"):
        L.append("## 🎯 Ответ\n" + synth["answer"] + "\n")
    # KPI
    L.append("## 📊 Итог")
    L.append(f"- **{prep.lbls.of(frame.target)} (net):** {_fmt_u(net, unit)}")
    if gross_loss is not None:
        L.append(f"- **Всего потеряно:** {_fmt_u(gross_loss, unit)} · **приросло:** {_fmt_u(gross_gain, unit)}")
    L.append("")
    if comps:
        L.append("## 🧩 Из чего складывается")
        for name, v in comps:
            L.append(f"- {name}: {_fmt_u(v, unit)}")
        L.append("")
    if levels:
        L.append("## 📉 Где сосредоточено (спуск по дереву)")
        for i, lv in enumerate(levels, 1):
            c = lv.contrib
            L.append(f"### Уровень {i}: по «{prep.lbls.col_title(lv.dim)}»")
            L.append(f"Крупнейший: **{fmt_val(lv.followed)}** — {_fmt_u(float(c.table['contrib'].iloc[0]), unit)} "
                     f"({c.top_share:.0f}% {_side(c.loss)}); {c.n_for_half} знач. дают половину.\n")
            if lv.chart:
                L.append(f"![{lv.dim}]({_rel_chart(table + '_investigate', lv.chart)})\n")
    if why:
        L.append("## ❓ Почему")
        for d, c, ch in why:
            L.append(f"### По «{prep.lbls.col_title(d)}»")
            L.append(f"Главное: **{fmt_val(c.table.index[0])}** ({c.top_share:.0f}%).\n")
            if ch:
                L.append(f"![{d}]({_rel_chart(table + '_investigate', ch)})\n")
    if who_tbl is not None:
        L.append(f"## 👤 Кто (конкретные {prep.lbls.of(frame.entity)})")
        has_val = "value" in who_tbl.columns
        L.append(f"| {prep.lbls.of(frame.entity)} | {prep.lbls.of(frame.target)} |"
                 + (f" {prep.lbls.of(frame.value)} |" if has_val else ""))
        L.append("|---|---|" + ("---|" if has_val else ""))
        for idx, row in who_tbl.head(8).iterrows():
            line = f"| {fmt_val(idx)} | {_fmt_u(float(row['contrib']), unit)} |"
            if has_val:
                line += f" {_fmt(float(row['value']))} |" if pd.notna(row['value']) else " — |"
            L.append(line)
        L.append("")
        if who_chart:
            L.append(f"![who]({_rel_chart(table + '_investigate', who_chart)})\n")
    if synth.get("chain"):
        L.append("## 🧭 Причинная цепочка")
        L += [f"{i}. {x}" for i, x in enumerate(synth["chain"], 1)]
        L.append("")
    if synth.get("actions"):
        L.append("## ✅ Что делать")
        L += [f"- {a}" for a in synth["actions"]]
        L.append("")
    L.append("---\n_Расследование: pandas-декомпозиция + LLM-синтез._")
    return "\n".join(L)


def _assemble_html(prep, frame, table, fqn, question, where, net, gross_loss, gross_gain, unit,
                   levels, why, comps, who_tbl, who_chart, synth) -> str:
    import html as _h
    H = ["<!doctype html><html lang='ru'><head><meta charset='utf-8'>",
         f"<title>Расследование: {_h.escape(table)}</title><style>{_HTML_CSS}</style></head><body>",
         f"<h1>🔎 Расследование: {_h.escape(prep.table_desc)}</h1>",
         f"<p class='angle'>{_h.escape(question)}</p>"]
    meta = f"Таблица: <code>{_h.escape(fqn)}</code> · строк: {len(prep.df):,}".replace(",", " ")
    if where:
        meta += f" · фильтр: <code>{_h.escape(where)}</code>"
    H.append(f"<p class='meta'>{meta}</p>")
    if synth.get("answer"):
        H.append(f"<div class='card summary'><h2 style='border:none;margin-top:0'>🎯 Ответ</h2>"
                 f"<p class='insight'>{_h.escape(synth['answer'])}</p></div>")
    # итог
    kpi = [f"<b>{_h.escape(prep.lbls.of(frame.target))} (net):</b> {_h.escape(_fmt_u(net, unit))}"]
    if gross_loss is not None:
        kpi.append(f"<b>Потеряно:</b> {_h.escape(_fmt_u(gross_loss, unit))} · "
                   f"<b>приросло:</b> {_h.escape(_fmt_u(gross_gain, unit))}")
    H.append("<h2>📊 Итог</h2><div class='card'><p>" + " &nbsp;·&nbsp; ".join(kpi) + "</p></div>")
    if comps:
        H.append("<h2>🧩 Из чего складывается</h2><div class='card'><ul>"
                 + "".join(f"<li>{_h.escape(n)}: {_h.escape(_fmt_u(v, unit))}</li>" for n, v in comps)
                 + "</ul></div>")
    if levels:
        H.append("<h2>📉 Где сосредоточено (спуск по дереву)</h2>")
        for i, lv in enumerate(levels, 1):
            c = lv.contrib
            H.append(f"<h3>Уровень {i}: по «{_h.escape(prep.lbls.col_title(lv.dim))}»</h3><div class='card'>")
            H.append(f"<p class='factline'>Крупнейший: <strong>{_h.escape(fmt_val(lv.followed))}</strong> — "
                     f"{_h.escape(_fmt_u(float(c.table['contrib'].iloc[0]), unit))} "
                     f"({c.top_share:.0f}% {_side(c.loss)}); {c.n_for_half} знач. дают половину.</p>")
            img = _img_b64(lv.chart)
            if img:
                H.append(f"<img src='{img}'>")
            H.append("</div>")
    if why:
        H.append("<h2>❓ Почему</h2>")
        for d, c, ch in why:
            H.append(f"<h3>По «{_h.escape(prep.lbls.col_title(d))}»</h3><div class='card'>")
            H.append(f"<p class='factline'>Главное: <strong>{_h.escape(fmt_val(c.table.index[0]))}</strong> "
                     f"({c.top_share:.0f}%).</p>")
            img = _img_b64(ch)
            if img:
                H.append(f"<img src='{img}'>")
            H.append("</div>")
    if who_tbl is not None:
        has_val = "value" in who_tbl.columns
        H.append(f"<h2>👤 Кто (конкретные {_h.escape(prep.lbls.of(frame.entity))})</h2><div class='card pattern-card'>")
        rows = "".join(
            f"<tr><td>{_h.escape(fmt_val(idx))}</td><td>{_h.escape(_fmt_u(float(r['contrib']), unit))}</td>"
            + (f"<td>{_h.escape(_fmt(float(r['value'])) if pd.notna(r['value']) else '—')}</td>" if has_val else "")
            + "</tr>" for idx, r in who_tbl.head(8).iterrows())
        head = (f"<th>{_h.escape(prep.lbls.of(frame.entity))}</th><th>{_h.escape(prep.lbls.of(frame.target))}</th>"
                + (f"<th>{_h.escape(prep.lbls.of(frame.value))}</th>" if has_val else ""))
        H.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
        img = _img_b64(who_chart)
        if img:
            H.append(f"<img src='{img}'>")
        H.append("</div>")
    if synth.get("chain"):
        H.append("<h2>🧭 Причинная цепочка</h2><div class='card'><ol>"
                 + "".join(f"<li>{_h.escape(x)}</li>" for x in synth["chain"]) + "</ol></div>")
    if synth.get("actions"):
        H.append("<div class='card attention'><h2 style='border:none;margin-top:0'>✅ Что делать</h2><ul>"
                 + "".join(f"<li>{_h.escape(a)}</li>" for a in synth["actions"]) + "</ul></div>")
    H.append("<p class='meta'>Расследование: pandas-декомпозиция + LLM-синтез.</p></body></html>")
    return "\n".join(H)
