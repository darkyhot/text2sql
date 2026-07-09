"""Оркестратор бизнес-отчёта: загрузка (1 запрос) → профиль → выбор разрезов (LLM)
→ pandas-аналитика + графики → бизнес-нарратив (LLM) → business_report.md."""

from __future__ import annotations

import base64
import html as _html
import logging
import os
import re
from pathlib import Path

import pandas as pd

from ..config import PATHS
from . import core, interactive, labels, metrics, mining, patterns, plan, render, scoring, semantic
from .store import FrameStore

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
    # ГЕЙТ ПО ПАМЯТИ: оцениваем размер (EXPLAIN, без выполнения); если таблица не влезает
    # в бюджет RAM — грузим СОГЛАСОВАННЫЙ сэмпл (WHERE random()<p) на уровне SQL + плашка.
    budget_gb = float(os.getenv("PANDAS_MEM_BUDGET_GB", "80"))
    ncols = len(columns) if columns else 30
    frac, sample_note = None, None
    try:
        est_rows = db.estimate_row_count(schema, table)
    except Exception:  # noqa: BLE001
        est_rows = None
    if est_rows:
        est_gb = est_rows * ncols * 60 / 1e9          # ~60 байт/ячейка (грубая оценка)
        if est_gb > budget_gb:
            frac = max(0.001, min(0.9, budget_gb / est_gb * 0.85))
    conds = ([f"({where})"] if where else []) + ([f"random() < {frac:.6f}"] if frac is not None else [])
    sql = f'SELECT {proj} FROM "{schema}"."{table}"' + (" WHERE " + " AND ".join(conds) if conds else "")
    res = db.run_export(sql, enforce_limit=False)
    df = pd.DataFrame(res.rows, columns=res.columns)
    if frac is not None:
        rows_h = f"{est_rows:,}".replace(",", " ")
        sample_note = (f"Таблица (~{rows_h} строк) не влезает в бюджет {budget_gb:.0f} ГБ RAM — "
                       f"загружен согласованный сэмпл ~{frac*100:.1f}% строк; абсолютные числа "
                       f"масштабированы по сэмплу, доли/средние корректны.")
        df.attrs["sample_note"] = sample_note
        logger.warning("report _load %s.%s: сэмпл %.2f%% (est_rows=%s, est_gb=%.1f > budget=%.0f)",
                        schema, table, frac * 100, est_rows, est_gb, budget_gb)
    # numeric приходит Decimal/строкой (object ИЛИ str-dtype) → приводим к числам
    for c in df.columns:
        if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c])
                or pd.api.types.is_bool_dtype(df[c])):
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
                          progress=lambda m: None, money_confirm=None) -> dict:
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

    progress("готовлю бизнес-подписи и единицы измерения колонок…")
    lbls = labels.build_labels_llm(llm, table_desc, meta, list(df.columns), df)

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
    core.normalize_roles(roles, list(df.columns))   # id → название (по всем колонкам таблицы)

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
    behav_defs, rec = metrics.build_behaviors_llm(llm, table_desc, df, meta)
    measures = metrics.build_derived(df, behav_defs, meta)
    metrics.money_from_metrics(df, meta, measures, roles.metrics, lbls)   # деньги — только уверенно
    count_m = metrics.record_count_measure(df, rec)     # «Количество задач» — если строка=сущность
    if count_m:
        measures.insert(0, count_m)                     # основная сущность — вперёд
        logger.info("report %s: считаем записи как сущность: %s", fqn, count_m.label)
    if not measures:
        raise ValueError("Не удалось определить показатели для аналитики.")
    # переспрос про деньги (интерактив): пользователь подтверждает/правит денежные поля
    if money_confirm is not None:
        cand = [m for m in measures if m.kind == "money" and m.tech]
        if cand:
            keep = money_confirm([(m.tech, m.label) for m in cand]) or set()
            for m in cand:
                if m.tech not in keep:                  # не деньги → количество, без ₽
                    m.kind, m.unit = "count", ""
                    logger.info("report: %s помечено как НЕ деньги (по правке пользователя)", m.tech)
    date = roles.dates[0] if roles.dates else None
    scope = mining.scope_notes(df, measures, roles.dimensions, lbls)      # условная заполненность
    if df.attrs.get("sample_note"):                                      # плашка о сэмпле (гейт памяти)
        scope = [df.attrs["sample_note"]] + scope

    # D: семантическая модель — декларативное описание таблицы (меры/разрезы/сущности +
    # headline-факты движком AggSpec) → сериализуемый JSON-артефакт для ревью/диффа.
    store = FrameStore(df)
    sem = semantic.build_semantic_model(fqn, table_desc, df, roles, measures, lbls, store=store)

    with core.report_style():                # локальный стиль графиков (не мутируем ноутбук)
        progress("считаю обзорные разрезы и динамику…")
        results = [mining.headline_kpi(df, measures)]
        # фокус-раздел: разложить запрос пользователя в конкретные разбивки и ответить ими
        if focus:
            reqs = plan.focus_plan_llm(llm, focus, measures, roles.dimensions, lbls)
            results += mining.focus_answer(df, reqs, measures, assets, lbls)
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
    for r in results:                        # C3: размечаем раздел-источник для кросс-ссылок
        r.facts["_section"] = _section_of_result(r, section_of)
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
    sem_path = sem.dump(out_dir / f"{table}_semantic_model.json")
    logger.info("report %s: семантическая модель → %s (кэш агрегаций: %s)", fqn, sem_path, store.stats)
    return {"md_path": str(md_path), "html_path": str(html_path),
            "semantic_path": str(sem_path), "sections": len(results), "rows": len(df),
            "charts": sum(1 for r in results if r.chart)}


def _section_of_result(r, section_of: dict) -> str:
    if r.kind == "focus":
        return "🎯 Ответ на ваш запрос"
    if r.kind == "overview":
        return "📈 Обзор по показателям"
    if r.kind == "entity":
        return "👥 Ключевые игроки: сотрудники, клиенты"
    if r.kind == "pattern":
        return "🔍 Закономерности во времени"
    if r.kind == "mined":
        return section_of.get(r.key, "🚨 Аномальные срезы")
    return "📈 Обзор по показателям"


def _grouped(results, section_of: dict) -> list[tuple[str, list]]:
    """Группируем находки по разделам и упорядочиваем по ЗНАЧИМОСТИ (C2): базовый ранг
    раздела, а внутри discovery-разделов (💰/⚖️/🚨, один ранг) — по агрегатному score
    находок, чтобы самый острый разрез всплывал выше. Внутри раздела — тоже по score."""
    buckets: dict[str, list] = {}
    for r in results:
        if r.key == "kpi" or r.facts.get("note"):
            continue
        buckets.setdefault(_section_of_result(r, section_of), []).append(r)
    for rs in buckets.values():
        rs.sort(key=lambda r: r.score, reverse=True)
    def _key(item):
        header, rs = item
        agg = max((r.score for r in rs), default=0.0)
        return scoring.section_sort_key(header, agg)
    return sorted(buckets.items(), key=_key, reverse=True)


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


# единая дизайн-система (токены свет/тёмная) + мелкие отчётные добавки
_HTML_CSS = render.THEME_CSS + """
details summary{cursor:pointer;color:var(--accent);font-size:13px}
.kpi td:first-child{color:var(--ink-2)}.kpi td:last-child{font-weight:600;text-align:right}
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
         f"<title>Бизнес-отчёт: {_html.escape(table)}</title><style>{_HTML_CSS}</style>",
         render.charts_head(), "</head><body>"]
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
            H.append(render.embed(r.chart))          # ECharts (сайдкар) или base64-PNG
            if r.table_md:
                H.append(_md_block_to_html(r.table_md))
            H.append("</div>")
    if attention:
        H.append("<div class='card attention'><h2 style='border:none;margin-top:0'>⚠️ На что обратить внимание</h2><ul>")
        H += [f"<li>{_html.escape(a)}</li>" for a in attention]
        H.append("</ul></div>")
    H.append("<p class='meta'>Все расчёты — pandas, графики — seaborn. Сгенерировано автоматически.</p>")
    H.append("</body></html>")
    return interactive.enhance("\n".join(H))     # сортировка + фильтр таблиц (self-contained JS)
