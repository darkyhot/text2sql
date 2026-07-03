"""Оркестратор бизнес-отчёта: загрузка (1 запрос) → профиль → выбор разрезов (LLM)
→ pandas-аналитика + графики → бизнес-нарратив (LLM) → business_report.md."""

from __future__ import annotations

import base64
import html as _html
import logging
import re
from pathlib import Path

import pandas as pd

from ..config import PATHS
from . import core, labels, metrics, mining, patterns, plan

logger = logging.getLogger(__name__)


# большие свободные тексты (varchar 4000) — не грузим: тяжёлые и не нужны для аналитики
_BIGTEXT_RE = re.compile(r"(_text$|_comment$|questionnaire|infopovod|anketa|_note$|_notes$|old_epk_id)", re.I)


def _load_columns(catalog, fqn: str) -> list[str] | None:
    t = catalog.get(fqn)
    if not t:
        return None
    cols = [c.name for c in t.columns
            if c.semantic_class != "free_text" and not _BIGTEXT_RE.search(c.name)]
    return cols or None


def _load(db, schema: str, table: str, where: str | None, columns: list[str] | None) -> pd.DataFrame:
    proj = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    sql = f'SELECT {proj} FROM "{schema}"."{table}"'
    if where:
        sql += f" WHERE {where}"
    # все строки из запроса (ограничение — через --where), дальше только pandas
    res = db.run_export(sql, enforce_limit=False)
    df = pd.DataFrame(res.rows, columns=res.columns)
    # numeric приходит Decimal (object) → приводим к числам, текст не трогаем
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            nn = df[c].notna().sum()
            if nn and conv.notna().sum() >= nn * 0.9:
                df[c] = conv
        # целочисленные колонки держим как Int64 (иначе id «плывёт» в 9038.0 из-за NaN)
        if pd.api.types.is_float_dtype(df[c]):
            s = df[c].dropna()
            if len(s) and (s % 1 == 0).all():
                df[c] = df[c].astype("Int64")
    return df


def _meta_for(catalog, fqn: str) -> tuple[str, dict[str, dict]]:
    t = catalog.get(fqn)
    if not t:
        return fqn, {}
    meta = {c.name: {"semantic_class": c.semantic_class, "unique_perc": c.unique_perc,
                     "desc": c.description} for c in t.columns}
    return (t.description or fqn), meta


def build_business_report(db, catalog, llm, fqn: str, *, where: str | None = None,
                          focus: str = "", out_dir: Path | None = None,
                          progress=lambda m: None) -> dict:
    """Собрать бизнес-отчёт по таблице. Возвращает {md_path, sections, rows}."""
    schema, table = fqn.split(".", 1)
    out_dir = out_dir or PATHS.workspace_dir
    assets = out_dir / "report_assets" / table   # своя папка на таблицу (не перетирает)
    if assets.exists():                            # чистим старые PNG (не мешаем новым)
        import shutil
        shutil.rmtree(assets, ignore_errors=True)

    progress("загружаю данные…")
    df = _load(db, schema, table, where, _load_columns(catalog, fqn))
    if df.empty:
        raise ValueError("По заданному фильтру нет данных.")
    table_desc, meta = _meta_for(catalog, fqn)

    progress("готовлю бизнес-подписи колонок…")
    lbls = labels.build_labels_llm(llm, table_desc, meta, list(df.columns))

    progress("профилирую колонки (роли, главные метрики)…")
    # LLM-first профилирование (роли колонок по смыслу); regex-эвристика — фолбэк
    roles = plan.build_roles_llm(llm, table_desc, df, meta) or core.profile(df, meta)
    logger.info("report %s: строк=%d dims=%s metrics=%s dates=%s flags=%s entities=%s",
                fqn, len(df), roles.dimensions[:5], roles.metrics[:5], roles.dates[:2],
                roles.flags[:4], roles.entities[:3])
    if not roles.metrics:
        raise ValueError("В таблице не найдено числовых метрик для бизнес-аналитики.")
    # страховка: явные сущности-люди/клиенты (*_fio, inn, company) — в drill-кандидаты,
    # даже если LLM-профайлер их не отметил (для дрилла «кто внутри горячего среза»)
    for c in df.columns:
        if (c not in roles.entities and c not in roles.dimensions
                and core._ENTITY_RE.search(c) and df[c].nunique(dropna=True) > 10):
            roles.entities.append(c); roles.card[c] = int(df[c].nunique(dropna=True))
    core.normalize_roles(roles)     # свернуть id↔name (оставить name) в dims и entities

    # явный фокус: колонки, названные пользователем, поднимаем в начало
    focus_dims: list[str] = []
    if focus:
        fl = focus.lower()
        key = lambda c: 0 if c.lower() in fl else 1
        roles.dimensions.sort(key=key)
        roles.entities.sort(key=key)
        roles.metrics.sort(key=key)
        focus_dims = [d for d in roles.dimensions if d.lower() in fl]
    logger.info("report %s: главные метрики=%s", fqn, roles.metrics[:4])

    progress("вывожу производные показатели (закрытие, просрочка, срок, деньги)…")
    behav_defs = metrics.build_behaviors_llm(llm, table_desc, df, meta)
    measures = metrics.build_derived(df, behav_defs, meta)
    metrics.money_from_metrics(df, meta, measures, roles.metrics, lbls)   # деньги — только уверенно
    if not measures:
        raise ValueError("Не удалось определить показатели для аналитики.")
    date = roles.dates[0] if roles.dates else None
    scope = mining.scope_notes(df, measures, roles.dimensions, lbls)      # условная заполненность

    progress("считаю обзорные разрезы и динамику…")
    results = [mining.headline_kpi(df, measures)]
    results += mining.overview(df, measures, roles.dimensions, date, assets, lbls)

    progress("кручу разрезы: ищу аномалии, концентрацию, перекос деньги/количество…")
    findings = mining.mine(df, measures, roles.dimensions, roles.entities, assets,
                           focus_dims=focus_dims or None, labels=lbls)
    mined_results = [mining.finding_to_result(f) for f in findings]
    section_of = {mining.finding_to_result(f).key: mining.section_for(f) for f in findings}
    results += mined_results

    progress("считаю рейтинги по сущностям (сотрудники, клиенты, ИНН)…")
    results += mining.entity_ratings(df, roles.entities, measures, assets, lbls)

    progress("ищу закономерности во времени…")
    results += patterns.detect_all(df, roles, assets, lbls)

    progress("пишу выводы бизнес-языком…")
    summary, attention = plan.narrate(llm, table_desc, focus, [r for r in results if r.key != "kpi"])
    attention = scope + attention          # скоуп-заметки — впереди «на что обратить внимание»
    angle = ""

    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = dict(table_desc=table_desc, fqn=fqn, table=table, where=where, focus=focus,
               angle=angle, nrows=len(df), summary=summary, attention=attention,
               results=results, section_of=section_of)
    # имена с префиксом таблицы — отчёты не перетирают друг друга
    md_path = out_dir / f"{table}_business_report.md"
    html_path = out_dir / f"{table}_business_report.html"
    md_path.write_text(_assemble_md(**ctx), encoding="utf-8")
    html_path.write_text(_assemble_html(**ctx), encoding="utf-8")
    return {"md_path": str(md_path), "html_path": str(html_path),
            "sections": len(results), "rows": len(df),
            "charts": sum(1 for r in results if r.chart)}


# порядок и заголовки тематических разделов
_SECTION_ORDER = [
    "📈 Обзор по показателям",
    "💰 Где сосредоточены деньги и объёмы",
    "⚖️ Ценность важнее количества",
    "🚨 Аномальные срезы",
    "👥 Ключевые игроки: сотрудники, клиенты, ИНН",
    "🔍 Закономерности во времени",
]


def _section_of_result(r, section_of: dict) -> str:
    if r.kind == "overview":
        return "📈 Обзор по показателям"
    if r.kind == "entity":
        return "👥 Ключевые игроки: сотрудники, клиенты, ИНН"
    if r.kind == "pattern":
        return "🔍 Закономерности во времени"
    if r.kind == "mined":
        return section_of.get(r.key, "🚨 Аномальные срезы")
    return "📈 Обзор по показателям"


def _grouped(results, section_of: dict) -> list[tuple[str, list]]:
    buckets: dict[str, list] = {}
    for r in results:
        if r.key == "kpi" or r.facts.get("note"):
            continue
        buckets.setdefault(_section_of_result(r, section_of), []).append(r)
    ordered = [(h, buckets[h]) for h in _SECTION_ORDER if buckets.get(h)]
    for h, rs in buckets.items():          # неучтённые (на всякий) — в конец
        if h not in _SECTION_ORDER:
            ordered.append((h, rs))
    return ordered


def _rel_chart(table: str, chart: str | None) -> str:
    return f"report_assets/{table}/{Path(chart).name}" if chart else ""


def _img_b64(chart: str | None) -> str:
    if not chart or not Path(chart).exists():
        return ""
    data = base64.b64encode(Path(chart).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _assemble_md(table_desc, fqn, table, where, focus, angle, nrows, summary, attention,
                 results, section_of) -> str:
    L: list[str] = [f"# Бизнес-отчёт: {table_desc}"]
    if angle:
        L.append(f"_{angle}_\n")
    meta_line = f"**Таблица:** `{fqn}` · **строк:** {nrows:,}".replace(",", " ")
    if where:
        meta_line += f" · **фильтр:** `{where}`"
    if focus:
        meta_line += f" · **фокус:** {focus}"
    L.append(meta_line + "\n")

    if summary:
        L.append("## 🎯 Главное")
        L += [f"- {s}" for s in summary]
        L.append("")
    for r in results:
        if r.key == "kpi":
            L.append("## 📊 Ключевые цифры\n" + r.table_md + "\n")
            break
    for header, rs in _grouped(results, section_of):
        L.append(f"## {header}\n")
        for r in rs:
            L.append(f"### {r.title}")
            if r.facts.get("_line"):
                L.append(r.facts["_line"] + "\n")
            if r.insight:
                L.append("💡 " + r.insight + "\n")
            if r.chart:
                L.append(f"![{r.title}]({_rel_chart(table, r.chart)})\n")
            if r.table_md:
                L.append(r.table_md + "\n")
    if attention:
        L.append("## ⚠️ На что обратить внимание")
        L += [f"- {a}" for a in attention]
        L.append("")
    L.append("---\n_Все расчёты — pandas, графики — seaborn._")
    return "\n".join(L)


_HTML_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1000px;margin:0 auto;
padding:32px 24px;color:#1f2933;background:#f7f9fc;line-height:1.55}
h1{font-size:26px;margin:0 0 4px;color:#12263f}
h2{font-size:20px;margin:34px 0 10px;padding-bottom:6px;border-bottom:2px solid #e3e8ef;color:#12263f}
h3{font-size:16px;margin:18px 0 6px;color:#12263f}
.pattern-card{background:#f3f0fb;border-color:#d9cff0}
.angle{color:#52606d;font-style:italic;margin:0 0 10px}
.meta{color:#616e7c;font-size:13px;margin-bottom:8px}
.card{background:#fff;border:1px solid #e3e8ef;border-radius:12px;padding:18px 22px;margin:16px 0;
box-shadow:0 1px 3px rgba(16,42,67,.05)}
.summary{background:#eef4ff;border-color:#c9dbff}
.attention{background:#fff7ed;border-color:#fdd9a8}
.insight{font-size:15px;margin:0 0 12px}
.factline{font-size:14px;margin:0 0 8px;color:#243b53}
img{max-width:100%;border-radius:8px;margin:6px 0}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th,td{border:1px solid #e3e8ef;padding:6px 10px;text-align:left}
th{background:#f0f4f8}
ul{margin:6px 0 0 18px}
details summary{cursor:pointer;color:#3b7dd8;font-size:13px}
.kpi td:first-child{color:#52606d}.kpi td:last-child{font-weight:600;text-align:right}
"""


def _md_table_to_html(md: str) -> str:
    rows = [r for r in md.strip().splitlines() if r.strip().startswith("|")]
    if len(rows) < 2:
        return ""
    def cells(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]
    head = cells(rows[0])
    body = [cells(r) for r in rows[2:]]
    th = "".join(f"<th>{_html.escape(c)}</th>" for c in head)
    trs = "".join("<tr>" + "".join(f"<td>{_html.escape(c)}</td>" for c in row) + "</tr>" for row in body)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def _md_inline_to_html(text: str) -> str:
    """Экранирует текст и превращает **жирный** в <strong> (для строк с числами)."""
    out, i, bold = [], 0, False
    esc = _html.escape(text)
    for part in esc.split("**"):
        out.append(part if not bold else f"<strong>{part}</strong>")
        bold = not bold
    return "".join(out)


def _md_block_to_html(md: str) -> str:
    """Блок: ведущие текстовые строки → <p>, таблица (|…|) → <table>."""
    text = [ln for ln in md.strip().splitlines() if ln.strip() and not ln.strip().startswith("|")]
    parts = []
    if text:
        parts.append("<p>" + _html.escape(" ".join(text)) + "</p>")
    tbl = _md_table_to_html(md)
    if tbl:
        parts.append(tbl)
    return "".join(parts)


def _assemble_html(table_desc, fqn, table, where, focus, angle, nrows, summary, attention,
                   results, section_of) -> str:
    H = [f"<!doctype html><html lang='ru'><head><meta charset='utf-8'>",
         f"<title>Бизнес-отчёт: {_html.escape(table)}</title><style>{_HTML_CSS}</style></head><body>"]
    H.append(f"<h1>📈 Бизнес-отчёт: {_html.escape(table_desc)}</h1>")
    if angle:
        H.append(f"<p class='angle'>{_html.escape(angle)}</p>")
    meta = f"Таблица: <code>{_html.escape(fqn)}</code> · строк: {nrows:,}".replace(",", " ")
    if where:
        meta += f" · фильтр: <code>{_html.escape(where)}</code>"
    if focus:
        meta += f" · фокус: {_html.escape(focus)}"
    H.append(f"<p class='meta'>{meta}</p>")

    if summary:
        H.append("<div class='card summary'><h2 style='border:none;margin-top:0'>🎯 Главное</h2><ul>")
        H += [f"<li>{_html.escape(s)}</li>" for s in summary]
        H.append("</ul></div>")
    for r in results:
        if r.key == "kpi":
            H.append("<h2>📊 Ключевые цифры</h2><div class='card kpi'>"
                     + _md_table_to_html(r.table_md) + "</div>")
            break
    for header, rs in _grouped(results, section_of):
        anomaly = header.startswith("🚨") or header.startswith("⚖️")
        H.append(f"<h2>{_html.escape(header)}</h2>")
        for r in rs:
            cls = "card pattern-card" if anomaly else "card"
            H.append(f"<h3>{_html.escape(r.title)}</h3><div class='{cls}'>")
            if r.facts.get("_line"):
                H.append(f"<p class='factline'>{_md_inline_to_html(r.facts['_line'])}</p>")
            if r.insight:
                H.append(f"<p class='insight'>💡 {_html.escape(r.insight)}</p>")
            img = _img_b64(r.chart)
            if img:
                H.append(f"<img src='{img}' alt='{_html.escape(r.title)}'>")
            if r.table_md:
                H.append(_md_block_to_html(r.table_md))
            H.append("</div>")
    if attention:
        H.append("<div class='card attention'><h2 style='border:none;margin-top:0'>⚠️ На что обратить внимание</h2><ul>")
        H += [f"<li>{_html.escape(a)}</li>" for a in attention]
        H.append("</ul></div>")
    H.append("<p class='meta'>Все расчёты — pandas, графики — seaborn. Сгенерировано автоматически.</p>")
    H.append("</body></html>")
    return "\n".join(H)
