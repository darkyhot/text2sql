"""Поверхность `/playbook` (этап D, §8.1 routing) — доводит семмодель + плейбуки до РАБОТЫ:
вопрос → загрузка (1 запрос) → SemanticTable → выбор плейбука → executor примитивов → HTML+MD.

Роутинг (§8.1): impact-плейбук, если вопрос про эффективность И у таблицы есть tool_flag;
иначе дефолтный `loss_attribution` (где/кто/почему/когда). Параметры плейбука (bindings)
выводятся ИЗ СЕММОДЕЛИ детерминированно (target/разрез/сущность/причина/дата или flag/outcome/
время/cutoff) — это и есть «FRAME» без отдельного LLM-вызова (честное упрощение §7.1 FRAME:
классификацию/привязку делаем эвристикой по модели, а не LLM).
"""

from __future__ import annotations

import html as _html
import logging
import re
from pathlib import Path

import pandas as pd

from ..config import PATHS
from . import plans, semantics
from .core import _fmt
from .interactive import enhance as _enhance
from .labels import fmt_val
from .store import FrameStore

logger = logging.getLogger(__name__)
_ID_RE = re.compile(r"(^inn$|_inn$|_id$|_code$|saphr|epk|ogrn|kpp)", re.I)
_OUTCOME_FLAG_RE = re.compile(r"(success|closed|закрыт|успех|выполн|resolve)", re.I)


def _pick_target(df: pd.DataFrame) -> str | None:
    """Целевая величина потерь: числовая не-id колонка с самой большой ВАЛОВОЙ убылью
    (сумма отрицательных значений). Валовая, а не net — иначе метрика с net-приростом,
    но реальными потерями (diff с положительным итогом) не выбирается. Если убыли нет —
    самая материальная по |сумме|."""
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
    if pb.name == "impact":
        tcol = next((t for t in model.time if t.is_reporting), model.time[0] if model.time else None)
        cutoff = _median_date(df, tcol.name) if tcol else None
        b = semantics.impact_bindings(model, cutoff=cutoff) or {}
        # outcome: ПРЕДПОЧИТАЕМ интерпретируемый бинарный исход (is_*success/closed) анонимной
        # производной ставке (__rate_N) — его видно в вердикте и он однозначен для DiD.
        succ = (next((c for c in df.columns if re.search(r"(success|успех)", c, re.I) and _num01_ok(df[c])), None)
                or next((c for c in df.columns if _OUTCOME_FLAG_RE.search(c) and _num01_ok(df[c])), None))
        out = b.get("outcome")
        bad = out and (out not in df.columns or pd.to_numeric(df[out], errors="coerce").notna().sum() == 0)
        if succ and (bad or str(out).startswith("__")):
            b["outcome"] = succ
        # treatment-флаг: в df и бинарный
        fl = b.get("flag")
        if fl and (fl not in df.columns or not _num01_ok(df[fl])):
            alt = next((c for c in model.tool_flags if c in df.columns and _num01_ok(df[c])), None)
            if alt:
                b["flag"] = alt
        return b
    # loss_attribution (и дефолт)
    dims = [d for d in model.dimensions if d.name in df.columns]
    primary = next((d.name for d in dims if d.kind in ("geo", "category", "status")
                    and 2 <= df[d.name].nunique(dropna=True) <= 60), dims[0].name if dims else None)
    reason = next((d.name for d in dims if d.kind == "reason"), None)
    entity = model.entities[0].key_col if model.entities else None
    date = next((t.name for t in model.time if t.is_reporting), None)
    return {"target": _pick_target(df), "dim": primary, "entity": entity,
            "reason": reason, "date": date, "side": "loss"}


def run_playbook(df: pd.DataFrame, model: semantics.SemanticTable, pb: plans.Playbook, question: str,
                 *, table_desc: str, fqn: str, out_dir: Path | None = None,
                 progress=lambda m: None) -> dict:
    """Исполнить ДАННЫЙ плейбук на уже подготовленных df+model → HTML+MD. Вызывается
    роутером `/investigate` (§8.1): загрузку/профилирование/выбор плейбука делает он."""
    out_dir = out_dir or PATHS.workspace_dir
    binds = _bindings(pb, model, df)
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
    html_path.write_text(_enhance(_assemble_html(**ctx)), encoding="utf-8")
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
    L.append(f"**Параметры (из семмодели):** " + ", ".join(f"{k}={v}" for k, v in binds.items() if v) + "\n")
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


_CSS = ("body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:960px;margin:0 auto;"
        "padding:28px 22px;color:#1f2933;background:#f7f9fc;line-height:1.55}"
        "h1{font-size:24px;color:#12263f}h2{font-size:19px;margin:26px 0 8px;border-bottom:2px solid #e3e8ef;"
        "padding-bottom:5px;color:#12263f}.card{background:#fff;border:1px solid #e3e8ef;border-radius:12px;"
        "padding:14px 18px;margin:12px 0}.verdict{background:#eef7ee;border-color:#bfe3bf}.meta{color:#616e7c;"
        "font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}"
        "th,td{border:1px solid #e3e8ef;padding:6px 10px;text-align:left}th{background:#f0f4f8}"
        "ul{margin:6px 0 0 18px}")


def _facts_html(facts: dict) -> str:
    items = "".join(f"<li><b>{_html.escape(k)}</b>: "
                    f"{_html.escape(_fmt(v) if isinstance(v, (int, float)) else str(v))}</li>"
                    for k, v in facts.items() if v is not None and not k.startswith("_"))
    return f"<ul>{items}</ul>" if items else ""


def _frame_html(frame: pd.DataFrame, limit: int = 8) -> str:
    md = _frame_md(frame, limit)
    if not md:
        return ""
    rows = [r for r in md.splitlines() if r.strip().startswith("|")]
    if len(rows) < 2:
        return ""
    cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
    th = "".join(f"<th>{_html.escape(c)}</th>" for c in cells(rows[0]))
    body = "".join("<tr>" + "".join(f"<td>{_html.escape(c)}</td>" for c in cells(r)) + "</tr>"
                   for r in rows[2:])
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _assemble_html(*, model, pb, binds, run, verdict, question, table_desc, fqn, nrows) -> str:
    H = ["<!doctype html><html lang='ru'><head><meta charset='utf-8'>",
         f"<title>Плейбук {_html.escape(pb.name)}</title><style>{_CSS}</style></head><body>",
         f"<h1>🎛️ Плейбук «{_html.escape(pb.name)}»: {_html.escape(table_desc)}</h1>",
         f"<p class='meta'>Вопрос: {_html.escape(question)}</p>",
         f"<p class='meta'>Таблица: <code>{_html.escape(fqn)}</code> · строк: {nrows:,}".replace(",", " ") + "</p>",
         f"<p class='meta'>Параметры из семмодели: "
         + _html.escape(", ".join(f"{k}={v}" for k, v in binds.items() if v)) + "</p>"]
    if verdict:
        H.append(f"<div class='card verdict'><h2 style='border:none;margin-top:0'>✅ Вердикт: "
                 f"{_html.escape(verdict[0])}</h2><p>{_html.escape(verdict[1])}</p></div>")
    for sid, res in run.results.items():
        H.append(f"<h2>{_html.escape(_STEP_TITLES.get(sid, sid))}</h2><div class='card'>"
                 + _facts_html(res.facts) + _frame_html(res.frame) + "</div>")
    if run.skipped:
        H.append(f"<p class='meta'>Шаги пропущены (нет данных/колонки): "
                 f"{_html.escape(', '.join(run.skipped))}.</p>")
    H.append("<p class='meta'>Плейбук: примитивы на pandas + семантическая модель.</p></body></html>")
    return "\n".join(H)
