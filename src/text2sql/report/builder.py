"""Оркестратор бизнес-отчёта: загрузка (1 запрос) → профиль → выбор разрезов (LLM)
→ pandas-аналитика + графики → бизнес-нарратив (LLM) → business_report.md."""

from __future__ import annotations

import base64
import html as _html
import logging
from pathlib import Path

import pandas as pd

from ..config import PATHS
from . import core, patterns, plan

logger = logging.getLogger(__name__)


def _load(db, schema: str, table: str, where: str | None) -> pd.DataFrame:
    sql = f'SELECT * FROM "{schema}"."{table}"'
    if where:
        sql += f" WHERE {where}"
    res = db.run_export(sql)          # один запрос к БД, дальше только pandas
    df = pd.DataFrame(res.rows, columns=res.columns)
    # numeric приходит Decimal (object) → приводим к числам, текст не трогаем
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            nn = df[c].notna().sum()
            if nn and conv.notna().sum() >= nn * 0.9:
                df[c] = conv
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

    progress("загружаю данные…")
    df = _load(db, schema, table, where)
    if df.empty:
        raise ValueError("По заданному фильтру нет данных.")
    table_desc, meta = _meta_for(catalog, fqn)
    roles = core.profile(df, meta)
    logger.info("report %s: строк=%d dims=%s metrics=%s dates=%s flags=%s",
                fqn, len(df), roles.dimensions[:5], roles.metrics[:5], roles.dates, roles.flags[:5])
    if not roles.metrics:
        raise ValueError("В таблице не найдено числовых метрик для бизнес-аналитики.")

    progress("выбираю интересные разрезы…")
    specs = plan.candidate_specs(roles)
    chosen, angle = plan.select_specs(llm, table_desc, focus, specs)

    progress("считаю аналитику и строю графики…")
    results = [core.kpi(df, roles)]
    for s in chosen:
        r = plan.run_spec(s, df, assets)
        if r and (r.chart or r.facts.get("note") is None):
            results.append(r)

    progress("ищу закономерности и аномалии…")
    pattern_results = patterns.detect_all(df, roles, assets)
    results += pattern_results

    progress("пишу выводы бизнес-языком…")
    summary, attention = plan.narrate(llm, table_desc, focus, [r for r in results if r.key != "kpi"])

    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = dict(table_desc=table_desc, fqn=fqn, table=table, where=where, focus=focus,
               angle=angle, nrows=len(df), summary=summary, attention=attention, results=results)
    # имена с префиксом таблицы — отчёты не перетирают друг друга
    md_path = out_dir / f"{table}_business_report.md"
    html_path = out_dir / f"{table}_business_report.html"
    md_path.write_text(_assemble_md(**ctx), encoding="utf-8")
    html_path.write_text(_assemble_html(**ctx), encoding="utf-8")
    return {"md_path": str(md_path), "html_path": str(html_path),
            "sections": len(results), "rows": len(df),
            "charts": sum(1 for r in results if r.chart)}


def _rel_chart(table: str, chart: str | None) -> str:
    return f"report_assets/{table}/{Path(chart).name}" if chart else ""


def _img_b64(chart: str | None) -> str:
    if not chart or not Path(chart).exists():
        return ""
    data = base64.b64encode(Path(chart).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _assemble_md(table_desc, fqn, table, where, focus, angle, nrows, summary, attention, results) -> str:
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
    pattern_hdr = False
    for r in results:
        if r.key == "kpi" or r.facts.get("note"):
            continue
        if r.kind == "pattern":
            if not pattern_hdr:
                L.append("## 🔍 Закономерности и аномалии\n")
                pattern_hdr = True
            L.append(f"### {r.title}")
        else:
            L.append(f"## {r.title}")
        if r.insight:
            L.append(r.insight + "\n")
        if r.chart:
            L.append(f"![{r.title}]({_rel_chart(table, r.chart)})\n")
        if r.table_md:
            L.append("<details><summary>Таблица</summary>\n\n" + r.table_md + "\n\n</details>\n")
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


def _assemble_html(table_desc, fqn, table, where, focus, angle, nrows, summary, attention, results) -> str:
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
    pattern_hdr = False
    for r in results:
        if r.key == "kpi" or r.facts.get("note"):
            continue
        if r.kind == "pattern" and not pattern_hdr:
            H.append("<h2>🔍 Закономерности и аномалии</h2>")
            pattern_hdr = True
        tag = "h3" if r.kind == "pattern" else "h2"
        cls = "card pattern-card" if r.kind == "pattern" else "card"
        H.append(f"<{tag}>{_html.escape(r.title)}</{tag}><div class='{cls}'>")
        if r.insight:
            H.append(f"<p class='insight'>{_html.escape(r.insight)}</p>")
        img = _img_b64(r.chart)
        if img:
            H.append(f"<img src='{img}' alt='{_html.escape(r.title)}'>")
        if r.table_md:
            H.append("<details><summary>Таблица с числами</summary>" + _md_table_to_html(r.table_md) + "</details>")
        H.append("</div>")
    if attention:
        H.append("<div class='card attention'><h2 style='border:none;margin-top:0'>⚠️ На что обратить внимание</h2><ul>")
        H += [f"<li>{_html.escape(a)}</li>" for a in attention]
        H.append("</ul></div>")
    H.append("<p class='meta'>Все расчёты — pandas, графики — seaborn. Сгенерировано автоматически.</p>")
    H.append("</body></html>")
    return "\n".join(H)
