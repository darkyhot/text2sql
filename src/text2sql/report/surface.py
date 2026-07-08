"""Поверхность плейбуков (§8.1 routing + §5.5–5.6 render) — доводит семмодель + плейбуки
до РАБОТЫ: вопрос → LLM-FRAME (тип + параметры) → executor примитивов → интерактивный
офлайн-HTML (ECharts-графики + Tabulator-таблицы, вшитые в файл).

FRAME теперь через LLM (`frame_llm`): классифицирует расследование и привязывает параметры
к колонкам семмодели; эвристика `_bindings` — детерминированный фолбэк, если LLM недоступен
или что-то не заполнил.
"""

from __future__ import annotations

import html as _html
import logging
import re
from pathlib import Path

import pandas as pd

from ..config import PATHS
from . import plans, render, semantics
from .core import _fmt
from .labels import fmt_val
from .store import FrameStore

logger = logging.getLogger(__name__)
_ID_RE = re.compile(r"(^inn$|_inn$|_id$|_code$|saphr|epk|ogrn|kpp)", re.I)
_OUTCOME_FLAG_RE = re.compile(r"(success|closed|закрыт|успех|выполн|resolve)", re.I)
_COL_KEYS = {"flag", "outcome", "time_col", "strata", "placebo",
             "target", "dim", "entity", "reason", "date"}


def _pick_target(df: pd.DataFrame) -> str | None:
    """Целевая величина потерь: числовая не-id колонка с самой большой ВАЛОВОЙ убылью."""
    gross_loss, mag = {}, {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not _ID_RE.search(c):
            v = pd.to_numeric(df[c], errors="coerce")
            neg = float(v[v < 0].sum())
            if neg < 0:
                gross_loss[c] = neg
            mag[c] = abs(float(v.sum()))
    if gross_loss:
        return min(gross_loss, key=gross_loss.get)
    return max(mag, key=mag.get) if mag else None


def _median_date(df: pd.DataFrame, col: str) -> str | None:
    d = pd.to_datetime(df[col], errors="coerce").dropna()
    return str(d.quantile(0.5).date()) if len(d) else None


def _num01_ok(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return len(v) > 0 and set(v.unique()).issubset({0, 1})


def _bindings(pb: plans.Playbook, model: semantics.SemanticTable, df: pd.DataFrame) -> dict:
    """Детерминированный fallback FRAME: параметры плейбука из семмодели."""
    if pb.name == "impact":
        tcol = next((t for t in model.time if t.is_reporting), model.time[0] if model.time else None)
        cutoff = _median_date(df, tcol.name) if tcol else None
        b = semantics.impact_bindings(model, cutoff=cutoff) or {}
        succ = (next((c for c in df.columns if re.search(r"(success|успех)", c, re.I) and _num01_ok(df[c])), None)
                or next((c for c in df.columns if _OUTCOME_FLAG_RE.search(c) and _num01_ok(df[c])), None))
        out = b.get("outcome")
        bad = out and (out not in df.columns or pd.to_numeric(df[out], errors="coerce").notna().sum() == 0)
        if succ and (bad or str(out).startswith("__")):
            b["outcome"] = succ
        fl = b.get("flag")
        if fl and (fl not in df.columns or not _num01_ok(df[fl])):
            alt = next((c for c in model.tool_flags if c in df.columns and _num01_ok(df[c])), None)
            if alt:
                b["flag"] = alt
        # страта для stratified_compare: интерпретируемый разрез малой кардинальности
        if not b.get("strata"):
            strat = next((d.name for d in model.dimensions if d.name in df.columns
                          and d.kind in ("geo", "category", "status")
                          and 2 <= df[d.name].nunique(dropna=True) <= 30), None)
            b["strata"] = strat
        return b
    dims = [d for d in model.dimensions if d.name in df.columns]
    primary = next((d.name for d in dims if d.kind in ("geo", "category", "status")
                    and 2 <= df[d.name].nunique(dropna=True) <= 60), dims[0].name if dims else None)
    reason = next((d.name for d in dims if d.kind == "reason"), None)
    entity = model.entities[0].key_col if model.entities else None
    date = next((t.name for t in model.time if t.is_reporting), None)
    return {"target": _pick_target(df), "dim": primary, "entity": entity,
            "reason": reason, "date": date, "side": "loss"}


# ---------------- LLM-FRAME (§7.1) ----------------
_FRAME_SYS = (
    "Ты — модуль FRAME аналитического ядра. По ВОПРОСУ и СЕММОДЕЛИ таблицы определи тип "
    "расследования и параметры. Значения параметров — СТРОГО имена колонок из семмодели.\n"
    "playbook:\n"
    " • impact — вопрос про ЭФФЕКТИВНОСТЬ / влияние / пользу / пилот инструмента, И у таблицы "
    "есть булев признак-лечение (tool_flag) и мера-исход;\n"
    " • loss_attribution — вопрос про то, ГДЕ и ПОЧЕМУ потеряли / упал показатель / отток.\n"
    "Для impact заполни: flag (булев tool_flag-лечение), outcome (мера-исход, предпочти бинарную "
    "success/closed), time_col (отчётная дата), cutoff (дата внедрения из ВОПРОСА в формате "
    "YYYY-MM-DD, иначе null), strata (разрез для стратификации: территория/тип/сегмент), "
    "placebo (мера, на которую инструмент влиять НЕ должен, иначе null).\n"
    "Для loss_attribution заполни: target (числовая мера убыли), dim (главный разрез), "
    "entity (сущность-ключ), reason (разрез-причина), date (отчётная дата), side ('loss'|'gain').\n"
    "Верни JSON со ВСЕМИ ключами: playbook, flag, outcome, time_col, cutoff, strata, placebo, "
    "target, dim, entity, reason, side. Ненужные для выбранного playbook — null."
)


def _model_brief(model: semantics.SemanticTable, df: pd.DataFrame) -> str:
    def col(c):
        return c if c in df.columns else f"{c}(нет в данных)"
    meas = "; ".join(f"{m.name}[{m.kind}/{m.unit}]" for m in model.measures[:20])
    dims = "; ".join(f"{d.name}({d.kind},card={getattr(d, 'card', '?')})" for d in model.dimensions[:20])
    ents = "; ".join(f"{e.key_col}" + (f"←{e.name_col}" if e.name_col else "") for e in model.entities[:6])
    times = "; ".join(f"{t.name}({'отчётная' if t.is_reporting else 'событийная'})" for t in model.time[:6])
    return (f"Таблица: {model.label or model.fqn}\n"
            f"Меры: {meas}\nРазрезы: {dims}\nСущности: {ents}\nДаты: {times}\n"
            f"tool_flags(лечения): {', '.join(model.tool_flags) or '—'}\n"
            f"outcome_measures(исходы): {', '.join(model.outcome_measures) or '—'}\n"
            f"Все колонки: {', '.join(map(str, df.columns))}")


def frame_llm(llm, question: str, model: semantics.SemanticTable, df: pd.DataFrame) -> dict | None:
    """LLM-FRAME: тип расследования + привязка параметров к колонкам. Валидирует по данным."""
    if llm is None:
        return None
    try:
        out = llm.complete_json(_FRAME_SYS, f"Вопрос: {question}\n\nСеммодель:\n{_model_brief(model, df)}",
                                max_tokens=1400, node="playbook_frame")
    except Exception as exc:  # noqa: BLE001
        logger.warning("FRAME LLM не удался (%s) — фолбэк на эвристику", exc)
        return None
    cols = set(df.columns)
    res: dict = {}
    pb = out.get("playbook")
    res["playbook"] = pb if pb in ("impact", "loss_attribution") else None
    for k in _COL_KEYS:
        v = out.get(k)
        if isinstance(v, str) and v in cols:
            res[k] = v
    cutoff = out.get("cutoff")
    if isinstance(cutoff, str) and re.match(r"\d{4}-\d{2}", cutoff):
        res["cutoff"] = cutoff
    if out.get("side") in ("loss", "gain"):
        res["side"] = out["side"]
    return res


def run_playbook(df: pd.DataFrame, model: semantics.SemanticTable, pb: plans.Playbook, question: str,
                 *, table_desc: str, fqn: str, out_dir: Path | None = None,
                 progress=lambda m: None, frame: dict | None = None) -> dict:
    """Исполнить плейбук: эвристические bindings, поверх — валидные значения LLM-FRAME."""
    out_dir = out_dir or PATHS.workspace_dir
    binds = _bindings(pb, model, df)
    if frame:                                        # LLM-FRAME перекрывает эвристику
        for k in list(binds):
            fv = frame.get(k)
            if not fv:
                continue
            if k in _COL_KEYS and fv not in df.columns:
                continue
            binds[k] = fv
    progress(f"плейбук «{pb.name}»: исполняю шаги…")
    run = plans.run_plan(FrameStore(df), pb.plan, binds)
    verdict = plans.impact_verdict(run) if pb.name == "impact" else None

    table = fqn.split(".", 1)[1] if "." in fqn else fqn
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{table}_playbook.md"
    html_path = out_dir / f"{table}_playbook.html"
    ctx = dict(model=model, pb=pb, binds=binds, run=run, verdict=verdict,
               question=question, table_desc=table_desc, fqn=fqn, nrows=len(df))
    md_path.write_text(_assemble_md(**ctx), encoding="utf-8")
    html_path.write_text(_assemble_html(**ctx), encoding="utf-8")
    return {"md_path": str(md_path), "html_path": str(html_path), "playbook": pb.name,
            "verdict": (verdict[0] if verdict else None), "skipped": run.skipped, "rows": len(df)}


# ---------------- рендер ----------------
_STEP_TITLES = {
    "scale": "📊 Масштаб", "where": "📍 Где сосредоточено", "who": "👤 Кто",
    "why": "❓ Почему (lift к базе)", "when": "🕒 Когда",
    "adoption": "📈 Охват инструмента", "naive": "⚖️ Наивное сравнение",
    "strat": "🧪 Стратифицированное сравнение", "did": "🎯 Difference-in-Differences",
    "selection": "🔍 Проверка самоотбора", "placebo": "🧫 Плацебо",
    "extrapolate": "📐 Экстраполяция эффекта",
}


def _facts_lines(facts: dict) -> list[str]:
    out = []
    for k, v in facts.items():
        if v is None or k.startswith("_"):
            continue
        vv = _fmt(v) if isinstance(v, (int, float)) else str(v)
        out.append(f"- **{k}**: {vv}")
    return out


def _frame_md(frame: pd.DataFrame, limit: int = 8) -> str:
    if frame is None or frame.empty:
        return ""
    show = frame.head(limit).copy()
    for c in show.columns:
        show[c] = show[c].map(lambda x: fmt_val(x) if not isinstance(x, float) else _fmt(x))
    return show.to_markdown(index=False)


def _assemble_md(*, model, pb, binds, run, verdict, question, table_desc, fqn, nrows) -> str:
    L = [f"# 🎛️ Плейбук «{pb.name}»: {table_desc}", f"**Вопрос:** {question}\n"]
    L.append(f"**Таблица:** `{fqn}` · **строк:** {nrows:,}".replace(",", " "))
    L.append(f"**Параметры (FRAME):** " + ", ".join(f"{k}={v}" for k, v in binds.items() if v) + "\n")
    if verdict:
        L.append(f"## ✅ Вердикт: {verdict[0]}\n{verdict[1]}\n")
    for sid, res in run.results.items():
        L.append(f"## {_STEP_TITLES.get(sid, sid)}")
        L += _facts_lines(res.facts)
        tbl = _frame_md(res.frame)
        if tbl:
            L.append("\n" + tbl)
        L.append("")
    if run.skipped:
        L.append(f"_Шаги пропущены (нет данных/колонки): {', '.join(run.skipped)}._")
    L.append("\n---\n_Плейбук: примитивы на pandas + семантическая модель. Числа детерминированы._")
    return "\n".join(L)


# базовые токены/компоненты приходят из render.THEME_CSS (page() их добавляет); здесь — только
# специфика плейбук-страницы
_CSS = ".verdict h2{color:var(--good)}"


def _pp(v):
    return round(float(v) * 100, 1)


def _step_widget(sid: str, res, binds: dict) -> str:
    """ECharts-виджет под конкретный шаг (интерактив в браузере)."""
    f = res.facts
    cid = f"plot_{sid}"
    try:
        if sid == "adoption":
            return render.indicator(cid, _pp(f.get("total", 0)), title="Охват инструмента, %", suffix=" %")
        if sid == "naive":
            return render.bar(cid, ["с инструментом", "без инструмента"],
                              [round(f.get("with", 0), 3), round(f.get("without", 0), 3)],
                              title="Наивное сравнение исхода", horizontal=False, color="neut")
        if sid == "strat":
            return render.bar(cid, [f"взвеш. эффект ({f.get('n_strata', 0)} страт)"],
                              [round(f.get("weighted_effect", 0), 4)],
                              title="Стратифицированный эффект", horizontal=False, color="gain")
        if sid == "did":
            fr = res.frame
            if fr is not None and {"group", "before", "after"}.issubset(fr.columns):
                byg = {str(r["group"]): r for _, r in fr.iterrows()}
                series = {g: [float(byg[g]["before"]), float(byg[g]["after"])] for g in byg}
                return render.grouped_bar(cid, ["до внедрения", "после"], series,
                                          title="DiD: пилот vs контроль (до/после)")
        if sid == "selection":
            return render.indicator(cid, _pp(f.get("pre_gap", 0)),
                                    title="Разрыв пилота ДО внедрения (самоотбор)", suffix=" п.п.")
        if sid == "placebo":
            return render.indicator(cid, _pp(f.get("placebo_did", 0)),
                                    title="Плацебо-эффект (норма ≈ 0)", suffix=" п.п.")
        if sid == "extrapolate":
            return render.indicator(cid, round(f.get("projected", 0), 0),
                                    title="Доп. исходов при полном охвате")
        # loss_attribution
        if sid == "where" and res.frame is not None and "contrib" in getattr(res.frame, "columns", []):
            fr = res.frame.head(12)
            return render.bar(cid, list(fr.iloc[:, 0].map(fmt_val)), list(fr["contrib"].astype(float)),
                              title="Вклад в потери", color="loss")
        if sid == "when" and res.frame is not None and "value" in getattr(res.frame, "columns", []):
            fr = res.frame
            return render.line(cid, list(fr.iloc[:, 0]), {"target": list(fr["value"].astype(float))},
                               title="Динамика по периодам", marker_x=f.get("break_period"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("echarts-виджет %s не построен: %s", sid, exc)
    return ""


def _facts_html(facts: dict) -> str:
    items = "".join(f"<li><b>{_html.escape(k)}</b>: "
                    f"{_html.escape(_fmt(v) if isinstance(v, (int, float)) else str(v))}</li>"
                    for k, v in facts.items() if v is not None and not k.startswith("_"))
    return f"<ul>{items}</ul>" if items else ""


def _assemble_html(*, model, pb, binds, run, verdict, question, table_desc, fqn, nrows) -> str:
    B = [f"<h1>🎛️ Плейбук «{_html.escape(pb.name)}»: {_html.escape(table_desc)}</h1>",
         f"<p class='meta'>Вопрос: {_html.escape(question)}</p>",
         f"<p class='meta'>Таблица: <code>{_html.escape(fqn)}</code> · строк: {nrows:,}".replace(",", " ") + "</p>",
         f"<p class='meta'>Параметры FRAME: "
         + _html.escape(", ".join(f"{k}={v}" for k, v in binds.items() if v)) + "</p>"]
    if verdict:
        B.append(f"<div class='card verdict'><h2 style='border:none;margin-top:0'>✅ Вердикт: "
                 f"{_html.escape(verdict[0])}</h2><p>{_html.escape(verdict[1])}</p></div>")
    for sid, res in run.results.items():
        B.append(f"<h2>{_html.escape(_STEP_TITLES.get(sid, sid))}</h2><div class='card'>")
        B.append(_facts_html(res.facts))
        B.append(_step_widget(sid, res, binds))
        B.append(render.df_table(res.frame,
                                 fmt=lambda c, v: _fmt(v) if isinstance(v, float) else fmt_val(v)))
        B.append("</div>")
    if run.skipped:
        B.append(f"<p class='meta'>Шаги пропущены (нет данных/колонки): "
                 f"{_html.escape(', '.join(run.skipped))}.</p>")
    B.append("<p class='meta'>Плейбук: примитивы на pandas + семантическая модель; "
             "графики ECharts, таблицы Tabulator — вшиты в файл.</p>")
    return render.page(f"Плейбук {pb.name}", "".join(B), css=_CSS, charts=True, tabulator=True)
