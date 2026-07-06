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
import re
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
# сущность-ИДЕНТИФИКАТОР vs ЧИТАЕМОЕ имя; разрезы-ПРИЧИНЫ/статусы (детекция по смыслу)
_ID_ENT = re.compile(r"(^inn$|_inn$|_id$|_code$|saphr|epk|ogrn|kpp)", re.I)
_NAME_ENT = re.compile(r"(name|fio|компан|client|клиент|организ|holder|наимен)", re.I)
_REASON_RE = re.compile(r"(причин|reason|повод|cause|статус|status|основан|infopovod|признак|тип)", re.I)


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
    id_col: str | None = None            # id-партнёр сущности (ИНН) — показываем в скобках
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
    # СУЩНОСТЬ: предпочитаем ЧИТАЕМОЕ имя организации/клиента идентификатору (ИНН/коду)
    ents = list(prep.roles.entities)
    chosen = out.get("entity") if out.get("entity") in entset else None
    name_ents = [e for e in ents if _NAME_ENT.search(e) and not _ID_ENT.search(e)]
    if (chosen is None or _ID_ENT.search(chosen)) and name_ents:
        entity = name_ents[0]
    else:
        entity = chosen or (name_ents[0] if name_ents else (ents[0] if ents else None))
    # id-партнёр (ИНН/код) той же сущности — для показа в скобках после имени
    id_col = None
    if entity and _NAME_ENT.search(entity):
        cands = [c for c in df.columns if _ID_ENT.search(c) and df[c].nunique(dropna=True) >= df[entity].nunique(dropna=True)]
        id_col = next((c for c in cands if c.lower() in ("inn",) or "inn" in c.lower()), cands[0] if cands else None)

    # ПОЧЕМУ: к разрезам от LLM добавляем очевидные причины/статусы из ЛЮБЫХ колонок
    # (детерминированно, чтобы блок «почему» не пропадал, если LLM/профайлер их не отметил)
    why = [d for d in (out.get("why_dims") or []) if d in dimset]
    for c in df.columns:
        if c in why or pd.api.types.is_numeric_dtype(df[c]):
            continue
        desc = prep.meta.get(c, {}).get("desc", "")
        if (_REASON_RE.search(c) or _REASON_RE.search(desc)) and 2 <= df[c].nunique(dropna=True) <= 40:
            why.append(c)

    return Frame(
        target=target, mode=mode,
        direction=out.get("direction") if out.get("direction") in ("loss", "gain", "top")
        else ("loss" if mode == "change" else "top"),
        drill_dims=[d for d in (out.get("drill_dims") or []) if d in dimset] or list(prep.roles.dimensions),
        why_dims=why[:3],
        components=[c for c in (out.get("components") or []) if c in numset and c != target],
        entity=entity, id_col=id_col,
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
               ref: float | None = None, side: str = "auto") -> Contrib:
    """ref — знаменатель для ДОЛИ (обычно общие потери всей таблицы), чтобы «доля лидера»
    была сопоставима между уровнями («49% ВСЕХ потерь»), а не 100% локально.
    side — какую сторону атрибутировать: 'loss' (отрицательные) | 'gain' | 'auto' (доминирующую).
    Важно: при вопросе про ПОТЕРИ берём потери, даже если net положительный."""
    s = pd.to_numeric(df[target], errors="coerce")
    g = s.groupby(df[dim]).sum()
    n = df.groupby(dim)[target].size()
    total = float(g.sum())
    if mode == "change":
        neg, pos = float(g[g < 0].sum()), float(g[g > 0].sum())
        want_loss = side == "loss" or (side == "auto" and abs(neg) >= abs(pos) and neg < 0)
        if want_loss:
            loss, asc = (neg if neg < 0 else -1e-9), True    # потери: самые отрицательные вперёд
        else:
            loss, asc = (pos if pos > 0 else 1e-9), False    # прирост: самые крупные вперёд
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
def _why(df, frame: Frame, assets, lbls, ref, side="auto") -> list[tuple[str, Contrib, str | None]]:
    out = []
    for d in frame.why_dims[:2]:
        if d in df.columns and 2 <= df[d].nunique(dropna=True) <= 40:
            c = _decompose(df, frame.target, d, frame.mode, ref, side)
            out.append((d, c, _contrib_chart(c, frame, assets, lbls, tag="why")))
    return out


def _components(df, frame: Frame, lbls) -> list[tuple[str, float]]:
    """Структурный разбор target на слагаемые (напр. самозанятые vs ГПХ)."""
    res = []
    for c in frame.components:
        v = float(pd.to_numeric(df[c], errors="coerce").sum())
        res.append((lbls.of(c), v))
    return res


def _who(df, frame: Frame, assets, lbls, elab=fmt_val) -> tuple[pd.DataFrame | None, str | None]:
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
    disp = top.copy(); disp.index = [elab(i) for i in top.index]     # «Имя (ИНН …)»
    tbl = pd.DataFrame({"contrib": disp})
    if val is not None:
        tbl["value"] = val.values
    chart = _entity_chart(disp, ent, frame, assets, lbls)
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


# ---------- ГЛУБОКАЯ ДЕКОМПОЗИЦИЯ ----------
def _side_of(frame: Frame) -> str:
    return frame.direction if frame.direction in ("loss", "gain") else "auto"


def _make_elab(df, frame: Frame, lbls):
    """Подпись значения сущности: «Название (ИНН 123…)», если у имени есть id-партнёр."""
    ent, idc = frame.entity, frame.id_col
    if not (ent and idc and idc in df.columns and ent in df.columns):
        return lambda v: fmt_val(v)
    # показываем id в скобках только при связи имя↔id ≈ 1:1 (иначе id вводит в заблуждение)
    if df.groupby(ent)[idc].nunique().median() > 1:
        return lambda v: fmt_val(v)
    m = df.groupby(ent)[idc].agg(lambda s: (s.dropna().mode().iloc[0]
                                            if not s.dropna().mode().empty else None))
    idlabel = lbls.of(idc)

    def elab(v):
        idv = m.get(v)
        return f"{fmt_val(v)} ({idlabel} {fmt_val(idv)})" if idv is not None else fmt_val(v)
    return elab


def _side_series(g: pd.Series, side: str) -> pd.Series:
    if side == "loss":
        return g[g < 0].sort_values()
    if side == "gain":
        return g[g > 0].sort_values(ascending=False)
    return g.reindex(g.abs().sort_values(ascending=False).index)


@dataclass
class Child:
    label: str
    contrib: float
    share_in_parent: float


@dataclass
class Node:
    label: str
    contrib: float
    share_of_total: float
    children: list          # list[Child]
    tail_contrib: float     # «прочие» внутри узла
    tail_count: int


def _build_tree(df, frame: Frame, ref: float, top_dim: str, entity: str, side: str,
                elab=fmt_val, *, max_seg=6, max_comp=8) -> list[Node]:
    """Дерево: top_dim (сегмент) → топ-entity (компании) внутри + «прочие»."""
    # СОГЛАСОВАННАЯ loss-декомпозиция: узел (сегмент) = сумма ПРОИГРЫВАЮЩИХ сущностей внутри,
    # чтобы «топ + прочие» = узлу, а Σ узлов = общим потерям (без взаимозачёта с приростом).
    seg_side: dict = {}
    for segname, sub in df.groupby(top_dim):
        cs = _side_series(pd.to_numeric(sub[frame.target], errors="coerce").groupby(sub[entity]).sum(), side)
        val = float(cs.sum())
        if abs(val) > 1e-9:
            seg_side[segname] = (val, cs)
    nodes: list[Node] = []
    for segname, (val, cs) in sorted(seg_side.items(), key=lambda kv: abs(kv[1][0]), reverse=True)[:max_seg]:
        segabs = abs(val) or 1.0
        top = cs.head(max_comp)
        tail = float(cs.iloc[max_comp:].sum()) if len(cs) > max_comp else 0.0
        children = [Child(elab(i), float(v), abs(v) / segabs * 100) for i, v in top.items()]
        nodes.append(Node(fmt_val(segname), val, abs(val) / abs(ref) * 100,
                          children, tail, max(0, len(cs) - max_comp)))
    return nodes


def _treemap_chart(tree: list[Node], frame: Frame, assets, lbls, top_dim, entity) -> str | None:
    """Вложенный treemap: сегмент-прямоугольник, внутри — компании (размер = вклад)."""
    try:
        import squarify
        segs = [(n.label, abs(n.contrib), n) for n in tree if abs(n.contrib) > 0]
        if not segs:
            return None
        W, H = 100.0, 100.0
        seg_sizes = squarify.normalize_sizes([x[1] for x in segs], W, H)
        seg_rects = squarify.squarify(seg_sizes, 0, 0, W, H)
        leaf_pal = sns.color_palette("tab20", 20)                   # цвет — по КОМПАНИИ, не сегменту
        ci = 0
        fig, ax = plt.subplots(figsize=(12, 7.5))
        for (name, sz, node), rect in zip(segs, seg_rects):
            ax.add_patch(plt.Rectangle((rect["x"], rect["y"]), rect["dx"], rect["dy"],
                                       facecolor="none", edgecolor="#111", lw=3, zorder=5))
            parts = [(c.label, abs(c.contrib)) for c in node.children if abs(c.contrib) > 0]
            if node.tail_count:
                parts.append((f"прочие ({node.tail_count})", abs(node.tail_contrib)))
            if not parts:
                continue
            csz = squarify.normalize_sizes([p[1] for p in parts], rect["dx"], rect["dy"])
            crects = squarify.squarify(csz, rect["x"], rect["y"], rect["dx"], rect["dy"])
            for (clabel, cval), cr in zip(parts, crects):
                is_other = clabel.startswith("прочие")
                fc = "#c9d2de" if is_other else leaf_pal[ci % 20]
                if not is_other:
                    ci += 1
                ax.add_patch(plt.Rectangle((cr["x"], cr["y"]), cr["dx"], cr["dy"],
                                           facecolor=fc, edgecolor="white", lw=1.0, alpha=.92))
                if cr["dx"] > 7 and cr["dy"] > 5:
                    ax.text(cr["x"] + cr["dx"] / 2, cr["y"] + cr["dy"] / 2,
                            f"{clabel[:16]}\n{_fmt(cval)}", ha="center", va="center", fontsize=7)
            # подпись сегмента
            ax.text(rect["x"] + rect["dx"] / 2, rect["y"] + rect["dy"] - 2.5,
                    f"{name} · {_fmt(abs(node.contrib))}", ha="center", va="top",
                    fontsize=9, fontweight="bold", color="#111")
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
        ax.set_title(f"Дерево потерь: «{lbls.of(top_dim)}» → «{lbls.of(entity)}» (площадь = вклад)", fontsize=12)
        return _save(fig, assets, "inv_treemap")
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: treemap не построен: %s", exc)
        return None


def _waterfall_chart(df, frame: Frame, dim, side, assets, lbls) -> str | None:
    """Водопад: как складывается итог из вкладов по разрезу (потери ↓, прирост ↑ → net)."""
    try:
        s = pd.to_numeric(df[frame.target], errors="coerce")
        g = s.groupby(df[dim]).sum().sort_values()          # от потерь к приросту
        items = list(g.items())
        if len(items) > 12:                                 # схлопнуть хвост
            items = items[:6] + [("прочие", float(g.iloc[6:-3].sum()))] + items[-3:] if len(items) > 9 else items
        labels_ = [_tick(k) for k, _ in items] + ["ИТОГ (net)"]
        vals = [float(v) for _, v in items]
        fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(labels_) + 2), 5))
        run = 0.0
        for i, v in enumerate(vals):
            ax.bar(i, v, bottom=run, color=_C_LOSS if v < 0 else _C_GAIN, edgecolor="white")
            ax.text(i, run + v + (max(map(abs, vals)) * 0.01 if v >= 0 else -max(map(abs, vals)) * 0.01),
                    _fmt(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
            run += v
        ax.bar(len(vals), run, color=_C_NEUT, edgecolor="white")
        ax.text(len(vals), run, _fmt(run), ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.axhline(0, color="#888", lw=1)
        ax.set_xticks(range(len(labels_))); ax.set_xticklabels(labels_, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"Как складывается {lbls.of(frame.target)} по «{lbls.of(dim)}»", fontsize=11)
        ax.grid(axis="y", alpha=.3)
        return _save(fig, assets, f"inv_waterfall_{dim}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: водопад не построен: %s", exc)
        return None


def _pareto_curve(df, frame: Frame, entity, side, assets, lbls) -> tuple[str | None, dict]:
    """Кривая Парето: накопленная доля потерь по сущностям (топ-N = X%)."""
    try:
        s = pd.to_numeric(df[frame.target], errors="coerce")
        g = _side_series(s.groupby(df[entity]).sum(), side)
        vals = g.abs().values
        total = vals.sum()
        if total <= 0 or len(vals) < 5:
            return None, {}
        cum = np.cumsum(vals) / total * 100
        n80 = int((cum < 80).sum()) + 1
        top10 = float(cum[min(9, len(cum) - 1)])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(range(1, len(cum) + 1), cum, color=_C_LOSS, lw=2)
        ax.axhline(80, color="gray", ls="--", lw=1); ax.axvline(n80, color="gray", ls=":", lw=1)
        ax.set_ylim(0, 105); ax.set_xlabel(f"число «{lbls.of(entity)}» (по убыванию вклада)")
        ax.set_ylabel("накопленная доля потерь, %")
        ax.set_title(f"Концентрация: топ-{n80} «{lbls.of(entity)}» = 80% потерь", fontsize=11)
        ax.grid(alpha=.3)
        chart = _save(fig, assets, f"inv_pareto_{entity}")
        return chart, {"n_for_80": n80, "total": int(len(vals)), "top10_share": round(top10, 1)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: Парето-кривая не построена: %s", exc)
        return None, {}


def _crosstab_heatmap(df, frame: Frame, da, db, side, assets, lbls) -> str | None:
    try:
        s = pd.to_numeric(df[frame.target], errors="coerce")
        piv = s.groupby([df[da], df[db]]).sum().unstack()
        # оставляем сторону потерь/прироста, топ по объёму
        keep_a = _side_series(s.groupby(df[da]).sum(), side).head(8).index
        keep_b = _side_series(s.groupby(df[db]).sum(), side).head(8).index
        piv = piv.reindex(index=keep_a, columns=keep_b)
        if piv.empty:
            return None
        piv.index = [_tick(x) for x in piv.index]; piv.columns = [_tick(x) for x in piv.columns]
        fig, ax = plt.subplots(figsize=(min(12, 1.3 * len(piv.columns) + 3), min(8, 0.6 * len(piv.index) + 2)))
        sns.heatmap(piv / 1000, annot=True, fmt=".0f", cmap="RdYlGn", center=0, ax=ax,
                    cbar_kws={"label": f"{lbls.of(frame.target)}, тыс"}, linewidths=.5, annot_kws={"fontsize": 8})
        ax.set_title(f"{lbls.of(frame.target)}: «{lbls.of(da)}» × «{lbls.of(db)}»", fontsize=11)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        return _save(fig, assets, f"inv_heat_{da}_{db}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: кросс-таб не построен: %s", exc)
        return None


def _quadrant_chart(df, frame: Frame, entity, value, side, assets, lbls) -> str | None:
    """Квадрант: убыль (X) vs ценность/потенциал (Y) по сущностям — приоритет возврата."""
    try:
        if not value or value not in df.columns:
            return None
        s = pd.to_numeric(df[frame.target], errors="coerce")
        loss = _side_series(s.groupby(df[entity]).sum(), side).abs()
        val = pd.to_numeric(df[value], errors="coerce").groupby(df[entity]).sum().reindex(loss.index)
        d = pd.DataFrame({"loss": loss, "val": val}).dropna().head(60)
        if len(d) < 4:
            return None
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(d["loss"], d["val"], s=40, color=_C_LOSS, alpha=.6)
        ax.axvline(d["loss"].median(), color="gray", ls="--", lw=1)
        ax.axhline(d["val"].median(), color="gray", ls="--", lw=1)
        # подписать топ-6 по (убыль×ценность)
        pri = (d["loss"] * d["val"]).sort_values(ascending=False).head(6).index
        for i in pri:
            ax.annotate(_tick(i), (d.loc[i, "loss"], d.loc[i, "val"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"убыль ({lbls.of(frame.target)})"); ax.set_ylabel(lbls.of(value))
        ax.set_title(f"Приоритет возврата: убыль × «{lbls.of(value)}» по «{lbls.of(entity)}»", fontsize=11)
        ax.grid(alpha=.3)
        return _save(fig, assets, f"inv_quadrant_{entity}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: квадрант не построен: %s", exc)
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
    side = _side_of(frame)                                # какую сторону расследуем
    # знаменатель «доли»: сторона потерь (если вопрос про потери), иначе доминирующая
    if frame.mode == "change":
        ref = gross_loss if side == "loss" else (gross_gain if side == "gain"
              else (gross_loss if abs(gross_loss) >= abs(gross_gain) else gross_gain))
    else:
        ref = net
    ref = ref or 1.0

    # главный разрез для дерева/водопада/кросс-таба — с макс. концентрацией вклада
    ent_con = {core._concept(e) for e in [frame.entity] if frame.entity}
    dcand = [d for d in frame.drill_dims if d in df.columns and core._concept(d) not in ent_con
             and 2 <= df[d].nunique(dropna=True) <= 60]
    primary = max(dcand, key=lambda d: _decompose(df, frame.target, d, frame.mode, ref, side).top_share,
                  default=(dcand[0] if dcand else None))

    progress("строю водопад и кривую концентрации…")
    waterfall = _waterfall_chart(df, frame, primary, side, assets, prep.lbls) if primary else None
    pareto_chart, pareto_facts = (_pareto_curve(df, frame, frame.entity, side, assets, prep.lbls)
                                  if frame.entity else (None, {}))
    progress("раскладываю потери по дереву (сегмент → компании)…")
    elab = _make_elab(df, frame, prep.lbls)                # подпись «Имя (ИНН …)»
    tree = _build_tree(df, frame, ref, primary, frame.entity, side, elab) if (primary and frame.entity) else []
    treemap = _treemap_chart(tree, frame, assets, prep.lbls, primary, frame.entity) if tree else None
    progress("ищу причины и структуру…")
    why = _why(df, frame, assets, prep.lbls, ref, side)
    heat = (_crosstab_heatmap(df, frame, primary, frame.why_dims[0], side, assets, prep.lbls)
            if (primary and frame.why_dims) else None)
    comps = _components(df, frame, prep.lbls)
    progress("нахожу «виновников» и приоритет возврата…")
    who_tbl, who_chart = _who(df, frame, assets, prep.lbls, elab)
    quadrant = _quadrant_chart(df, frame, frame.entity, frame.value, side, assets, prep.lbls) if frame.entity else None

    facts = _facts_for_synth(frame, prep.lbls, net, gross_loss, gross_gain, unit, tree, why,
                             comps, who_tbl, pareto_facts, primary)
    progress("собираю причинную цепочку и действия…")
    synth = _synthesize(llm, question, facts)

    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = dict(prep=prep, frame=frame, table=table, fqn=fqn, question=question, where=where,
               net=net, gross_loss=gross_loss, gross_gain=gross_gain, unit=unit, primary=primary,
               waterfall=waterfall, pareto_chart=pareto_chart, pareto_facts=pareto_facts,
               tree=tree, treemap=treemap, why=why, heat=heat, comps=comps,
               who_tbl=who_tbl, who_chart=who_chart, quadrant=quadrant, synth=synth)
    md_path = out_dir / f"{table}_investigation.md"
    html_path = out_dir / f"{table}_investigation.html"
    md_path.write_text(_assemble_md(**ctx), encoding="utf-8")
    html_path.write_text(_assemble_html(**ctx), encoding="utf-8")
    return {"md_path": str(md_path), "html_path": str(html_path),
            "segments": len(tree), "rows": len(df)}


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


def _facts_for_synth(frame, lbls, net, gloss, ggain, unit, tree, why, comps, who_tbl,
                     pareto_facts, primary) -> dict:
    f = {"целевая": f"{lbls.of(frame.target)} (единица: {unit or 'шт'})", "режим": frame.mode,
         "итог_net": _fmt_u(net, unit)}
    if gloss is not None:
        f["всего_потеряно"] = _fmt_u(gloss, unit)
        f["всего_приросло"] = _fmt_u(ggain, unit)
    f["главный_разрез"] = lbls.of(primary) if primary else None
    f["дерево"] = [{
        "узел": n.label, "вклад": _fmt_u(n.contrib, unit), "доля_от_общих_%": round(n.share_of_total, 1),
        "топ_внутри": [{"кто": c.label, "вклад": _fmt_u(c.contrib, unit),
                        "доля_в_узле_%": round(c.share_in_parent, 1)} for c in n.children[:4]],
    } for n in tree[:5]]
    if pareto_facts:
        f["концентрация"] = (f"топ-{pareto_facts.get('n_for_80')} «{lbls.of(frame.entity)}» = 80% потерь "
                             f"(всего {pareto_facts.get('total')}); топ-10 = {pareto_facts.get('top10_share')}%")
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
                 primary, waterfall, pareto_chart, pareto_facts, tree, treemap, why, heat,
                 comps, who_tbl, who_chart, quadrant, synth) -> str:
    T = table + "_investigate"
    lb = prep.lbls
    def im(ch, alt=""):
        return f"![{alt}]({_rel_chart(T, ch)})\n" if ch else ""
    L = [f"# 🔎 Расследование: {prep.table_desc}", f"**Вопрос:** {question}\n"]
    meta = f"**Таблица:** `{fqn}` · **строк:** {len(prep.df):,}".replace(",", " ")
    if where:
        meta += f" · **фильтр:** `{where}`"
    L.append(meta + "\n")
    if synth.get("answer"):
        L.append("## 🎯 Ответ\n" + synth["answer"] + "\n")
    # масштаб: net vs gross
    L.append("## 📊 Масштаб")
    if gross_loss is not None:
        L.append(f"- **Валовые потери:** {_fmt_u(gross_loss, unit)} · **приток:** {_fmt_u(gross_gain, unit)} "
                 f"· **net:** {_fmt_u(net, unit)}")
    else:
        L.append(f"- **{lb.of(frame.target)}:** {_fmt_u(net, unit)}")
    if pareto_facts:
        L.append(f"- **Концентрация:** топ-{pareto_facts['n_for_80']} «{lb.of(frame.entity)}» = 80% потерь "
                 f"(из {pareto_facts['total']}); топ-10 = {pareto_facts['top10_share']}%.")
    L.append("")
    L.append(im(waterfall, "водопад")); L.append(im(pareto_chart, "Парето"))
    if comps:
        L.append("## 🧩 Из чего складывается")
        L += [f"- {name}: {_fmt_u(v, unit)}" for name, v in comps]
        L.append("")
    if tree:
        L.append(f"## 🌳 Дерево потерь: «{lb.col_title(primary)}» → «{lb.col_title(frame.entity)}»")
        L.append(im(treemap, "treemap"))
        for n in tree:
            L.append(f"### {n.label} — {_fmt_u(n.contrib, unit)} ({n.share_of_total:.0f}% всех потерь)")
            for c in n.children[:5]:
                L.append(f"- {c.label}: {_fmt_u(c.contrib, unit)} ({c.share_in_parent:.0f}% узла)")
            if n.tail_count:
                L.append(f"- прочие ({n.tail_count}): {_fmt_u(n.tail_contrib, unit)}")
            L.append("")
    if why or heat:
        L.append("## ❓ Почему")
        for d, c, ch in why:
            L.append(f"### По «{lb.col_title(d)}»: преобладает **{fmt_val(c.table.index[0])}** ({c.top_share:.0f}%)")
            L.append(im(ch, d))
        if heat:
            L.append(f"### Где какая причина: «{lb.of(primary)}» × «{lb.of(frame.why_dims[0])}»")
            L.append(im(heat, "heat"))
    if who_tbl is not None:
        L.append(f"## 👤 Кто + приоритет возврата ({lb.of(frame.entity)})")
        has_val = "value" in who_tbl.columns
        L.append(f"| {lb.of(frame.entity)} | {lb.of(frame.target)} |"
                 + (f" {lb.of(frame.value)} |" if has_val else ""))
        L.append("|---|---|" + ("---|" if has_val else ""))
        for idx, row in who_tbl.head(8).iterrows():
            line = f"| {fmt_val(idx)} | {_fmt_u(float(row['contrib']), unit)} |"
            if has_val:
                line += (f" {_fmt(float(row['value']))} |" if pd.notna(row['value']) else " — |")
            L.append(line)
        L.append("")
        L.append(im(quadrant, "квадрант")); L.append(im(who_chart, "кто"))
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
                   primary, waterfall, pareto_chart, pareto_facts, tree, treemap, why, heat,
                   comps, who_tbl, who_chart, quadrant, synth) -> str:
    import html as _h
    lb = prep.lbls
    def im(ch):
        b = _img_b64(ch)
        return f"<img src='{b}'>" if b else ""
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
    # масштаб
    H.append("<h2>📊 Масштаб</h2><div class='card'>")
    if gross_loss is not None:
        H.append(f"<p><b>Валовые потери:</b> {_h.escape(_fmt_u(gross_loss, unit))} &nbsp;·&nbsp; "
                 f"<b>приток:</b> {_h.escape(_fmt_u(gross_gain, unit))} &nbsp;·&nbsp; "
                 f"<b>net:</b> {_h.escape(_fmt_u(net, unit))}</p>")
    if pareto_facts:
        H.append(f"<p class='factline'>Концентрация: топ-{pareto_facts['n_for_80']} «{_h.escape(lb.of(frame.entity))}» "
                 f"= 80% потерь (из {pareto_facts['total']}); топ-10 = {pareto_facts['top10_share']}%.</p>")
    H.append(im(waterfall) + im(pareto_chart) + "</div>")
    if comps:
        H.append("<h2>🧩 Из чего складывается</h2><div class='card'><ul>"
                 + "".join(f"<li>{_h.escape(n)}: {_h.escape(_fmt_u(v, unit))}</li>" for n, v in comps) + "</ul></div>")
    if tree:
        H.append(f"<h2>🌳 Дерево потерь: «{_h.escape(lb.col_title(primary))}» → «{_h.escape(lb.col_title(frame.entity))}»</h2>")
        H.append("<div class='card'>" + im(treemap) + "</div>")
        for n in tree:
            H.append(f"<h3>{_h.escape(n.label)} — {_h.escape(_fmt_u(n.contrib, unit))} "
                     f"({n.share_of_total:.0f}% всех потерь)</h3><div class='card'><ul>")
            for c in n.children[:5]:
                H.append(f"<li>{_h.escape(c.label)}: {_h.escape(_fmt_u(c.contrib, unit))} "
                         f"({c.share_in_parent:.0f}% узла)</li>")
            if n.tail_count:
                H.append(f"<li>прочие ({n.tail_count}): {_h.escape(_fmt_u(n.tail_contrib, unit))}</li>")
            H.append("</ul></div>")
    if why or heat:
        H.append("<h2>❓ Почему</h2>")
        for d, c, ch in why:
            H.append(f"<h3>По «{_h.escape(lb.col_title(d))}»</h3><div class='card'>"
                     f"<p class='factline'>Преобладает <strong>{_h.escape(fmt_val(c.table.index[0]))}</strong> "
                     f"({c.top_share:.0f}%).</p>" + im(ch) + "</div>")
        if heat:
            H.append(f"<h3>Где какая причина: «{_h.escape(lb.of(primary))}» × «{_h.escape(lb.of(frame.why_dims[0]))}»</h3>"
                     f"<div class='card'>" + im(heat) + "</div>")
    if who_tbl is not None:
        has_val = "value" in who_tbl.columns
        H.append(f"<h2>👤 Кто + приоритет возврата ({_h.escape(lb.of(frame.entity))})</h2><div class='card pattern-card'>")
        rows = "".join(
            f"<tr><td>{_h.escape(fmt_val(idx))}</td><td>{_h.escape(_fmt_u(float(r['contrib']), unit))}</td>"
            + (f"<td>{_h.escape(_fmt(float(r['value'])) if pd.notna(r['value']) else '—')}</td>" if has_val else "")
            + "</tr>" for idx, r in who_tbl.head(8).iterrows())
        head = (f"<th>{_h.escape(lb.of(frame.entity))}</th><th>{_h.escape(lb.of(frame.target))}</th>"
                + (f"<th>{_h.escape(lb.of(frame.value))}</th>" if has_val else ""))
        H.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
        H.append(im(quadrant) + im(who_chart) + "</div>")
    if synth.get("chain"):
        H.append("<h2>🧭 Причинная цепочка</h2><div class='card'><ol>"
                 + "".join(f"<li>{_h.escape(x)}</li>" for x in synth["chain"]) + "</ol></div>")
    if synth.get("actions"):
        H.append("<div class='card attention'><h2 style='border:none;margin-top:0'>✅ Что делать</h2><ul>"
                 + "".join(f"<li>{_h.escape(a)}</li>" for a in synth["actions"]) + "</ul></div>")
    H.append("<p class='meta'>Расследование: pandas-декомпозиция + LLM-синтез.</p></body></html>")
    return "\n".join(H)
