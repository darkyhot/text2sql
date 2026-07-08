"""Render Engine (§5.5–5.6 дизайна): интерактивный ОФЛАЙН-HTML на вшитых Plotly.js + Tabulator.

Библиотеки лежат в `assets/` (vendored) и ИНЛАЙНЯТСЯ в HTML — файл полностью автономен,
нулевые внешние запросы (банковский офлайн-контур). Здесь — только «голова» (вшитые CSS/JS)
и генераторы виджетов: Plotly-графики (bar/grouped/line/waterfall/indicator) и Tabulator-таблицы
(сортировка, поиск по колонкам, пагинация, экспорт CSV).
"""

from __future__ import annotations

import base64
import html as _html
import json
import re
from pathlib import Path

_ASSETS = Path(__file__).parent / "assets"
_cache: dict[str, str] = {}


def _read(name: str) -> str:
    if name not in _cache:
        _cache[name] = (_ASSETS / name).read_text(encoding="utf-8")
    return _cache[name]


# Дизайн-токены (свет/тёмная авто) + базовые компоненты. Один источник для всех поверхностей —
# отчёт/расследование/плейбук выглядят как одна система. Tabulator-тема ссылается на эти токены.
THEME_CSS = """
:root{--plane:#f9f9f7;--surface-1:#fcfcfb;--surface-2:#f2f1ee;--ink-1:#0b0b0b;--ink-2:#52514e;
--muted:#898781;--hair:rgba(11,11,11,.10);--hair-soft:rgba(11,11,11,.06);--wash:rgba(42,120,214,.06);
--accent:#2a78d6;--good:#006300;--bad:#d03b3b;--warn:#b57400;--radius:14px;
--shadow:0 1px 2px rgba(16,42,67,.06),0 1px 3px rgba(16,42,67,.05)}
@media (prefers-color-scheme:dark){:root{--plane:#0d0d0d;--surface-1:#1a1a19;--surface-2:#232322;
--ink-1:#fff;--ink-2:#c3c2b7;--muted:#898781;--hair:rgba(255,255,255,.12);--hair-soft:rgba(255,255,255,.07);
--wash:rgba(57,135,229,.12);--accent:#3987e5;--good:#0ca30c;--bad:#e66767;--warn:#fab219;
--shadow:0 1px 2px rgba(0,0,0,.45)}}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;max-width:1080px;margin:0 auto;
padding:34px 24px;color:var(--ink-1);background:var(--plane);line-height:1.55;font-variant-numeric:tabular-nums}
h1{font-size:26px;letter-spacing:-.015em;margin:0 0 6px}
h2{font-size:19px;letter-spacing:-.01em;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--hair)}
h3{font-size:15px;margin:16px 0 6px}
p{margin:0 0 10px}a{color:var(--accent)}
code{background:var(--surface-2);padding:1px 6px;border-radius:6px;font-size:.9em}
ul{margin:6px 0 0 18px}
.card{background:var(--surface-1);border:1px solid var(--hair);border-radius:var(--radius);
padding:18px 22px;margin:14px 0;box-shadow:var(--shadow)}
.pattern-card{background:var(--surface-2)}
.summary{border-color:var(--accent)}.attention{border-color:var(--bad)}.verdict{border-color:var(--good)}
.meta{color:var(--muted);font-size:13px}.angle{color:var(--ink-2);font-style:italic;font-size:14px}
.insight{font-size:15px;color:var(--ink-1)}.factline{font-size:14px;color:var(--ink-2)}
img{max-width:100%;border-radius:10px;margin:6px 0;border:1px solid var(--hair)}
.t2s-plot{margin:6px 0;min-height:60px}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:12px 0}
.kpi-tile{background:var(--surface-1);border:1px solid var(--hair);border-radius:var(--radius);
padding:14px 16px;box-shadow:var(--shadow)}
.kpi-tile .k-label{font-size:12px;color:var(--muted)}
.kpi-tile .k-value{font-size:23px;font-weight:650;margin-top:3px}
"""


# ---------------- дизайн-система (валидированная палитра dataviz-скилла) ----------------
# Категориальная (фикс. порядок — механизм CVD-безопасности), свет/тёмная. Валидатор: PASS,
# worst adjacent ΔE 24.2. Свет: 3 слота <3:1 → relief через прямые подписи на барах (есть).
_CAT_LIGHT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
_CAT_DARK = ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"]
_SEQ = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]           # sequential blue
_SEQ_RED = [[0.0, "#fbe0da"], [0.5, "#e34948"], [1.0, "#8f1f1e"]]
_DIV = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#d03b3b"]]          # diverging blue↔gray↔red
_SCALE = {"Blues": _SEQ, "Reds": _SEQ_RED, "RdYlGn": _DIV, "seq": _SEQ, "div": _DIV}


def _tpl(mode: str) -> dict:
    light = mode == "light"
    ink = "#0b0b0b" if light else "#ffffff"
    sec = "#52514e" if light else "#c3c2b7"
    mut = "#898781"
    grid = "#e1e0d9" if light else "#2c2c2a"
    base = "#c3c2b7" if light else "#383835"
    surf = "#fcfcfb" if light else "#1a1a19"
    ax = {"gridcolor": grid, "zerolinecolor": base, "linecolor": "rgba(0,0,0,0)",
          "tickfont": {"color": mut, "size": 11}, "title": {"font": {"color": sec}},
          "automargin": True}
    return {"layout": {
        "font": {"family": "system-ui,-apple-system,Segoe UI,sans-serif", "size": 12, "color": sec},
        "title": {"font": {"color": ink, "size": 14}, "x": 0.02, "xanchor": "left"},
        "colorway": _CAT_LIGHT if light else _CAT_DARK,
        "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "xaxis": {**ax, "showgrid": True}, "yaxis": {**ax, "showgrid": True},
        "hoverlabel": {"bgcolor": surf, "bordercolor": grid,
                       "font": {"family": "system-ui,-apple-system,sans-serif", "color": ink}},
        "bargap": 0.3, "legend": {"orientation": "h", "y": -0.2, "font": {"color": sec}},
        "margin": {"t": 42, "r": 16, "b": 46, "l": 62}}}


def plotly_head() -> str:
    tpl = json.dumps({"light": _tpl("light"), "dark": _tpl("dark")}, ensure_ascii=False)
    js = ("window.T2S=window.T2S||{};T2S.tpl=%s;T2S.charts=[];"
          "T2S.mode=function(){return matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';};"
          "T2S.cfg={responsive:true,displaylogo:false,"
          "modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d','zoom2d']};"
          "T2S.draw=function(id,tr,lay){Plotly.react(id,tr,"
          "Object.assign({template:T2S.tpl[T2S.mode()]},lay||{}),T2S.cfg);};"
          "T2S.plot=function(id,tr,lay){T2S.charts.push([id,tr,lay]);T2S.draw(id,tr,lay);};"
          "T2S.redraw=function(){T2S.charts.forEach(function(c){T2S.draw(c[0],c[1],c[2]);});};"
          "matchMedia('(prefers-color-scheme: dark)').addEventListener('change',T2S.redraw);") % tpl
    return f"<script>{_read('plotly.min.js')}</script><script>{js}</script>"


def tabulator_head() -> str:
    return f"<style>{_read('tabulator.min.css')}</style>\n<script>{_read('tabulator.min.js')}</script>"


# Тема Tabulator (токены дизайн-системы, свет/тёмная): без зебры, тонкие горизонтальные
# разделители, липкая приглушённая шапка, tabular-nums, мягкий hover-wash. Числовые колонки —
# вправо; колонки-доли (%/доля) получают прогресс-бар в ячейке.
_TAB_CSS = """
<style>
.t2s-tab-wrap{margin:12px 0}
.t2s-tools{display:flex;justify-content:flex-end;margin:0 0 6px}
.t2s-tools button{cursor:pointer;border:1px solid var(--hair);background:var(--surface-2);
  color:var(--ink-2);border-radius:8px;padding:4px 11px;font-size:12px}
.t2s-tools button:hover{background:var(--surface-1)}
.tabulator{font-size:13px;border:1px solid var(--hair);border-radius:12px;overflow:hidden;
  background:var(--surface-1);font-variant-numeric:tabular-nums}
.tabulator .tabulator-header{background:var(--surface-2);border-bottom:1px solid var(--hair)}
.tabulator .tabulator-header .tabulator-col{background:transparent;border-right:none;
  color:var(--ink-2);font-weight:600}
.tabulator .tabulator-header .tabulator-col .tabulator-col-title{padding:8px 10px}
.tabulator .tabulator-header .tabulator-header-filter input{border:1px solid var(--hair);
  border-radius:6px;background:var(--surface-1);color:var(--ink-1);padding:3px 6px}
.tabulator-row{background:transparent;border-bottom:1px solid var(--hair-soft)}
.tabulator-row.tabulator-row-even{background:transparent}
.tabulator-row:hover{background:var(--wash)!important}
.tabulator-row .tabulator-cell{border-right:none;padding:7px 10px;color:var(--ink-1)}
.tabulator .tabulator-footer{background:var(--surface-2);border-top:1px solid var(--hair);color:var(--ink-2)}
.tabulator .tabulator-footer .tabulator-page{background:var(--surface-1);border:1px solid var(--hair);
  border-radius:6px;color:var(--ink-2)}
</style>
"""

_TAB_INIT = _TAB_CSS + """
<script>
document.addEventListener("DOMContentLoaded",function(){
  if(typeof Tabulator==="undefined")return;
  var NUM=/^[\\s+\\-]?[\\d \\u00a0.,]+%?\\s*(тыс|млн|млрд|₽|руб|чел\\.?|шт|дн|%|п\\.п\\.)?\\s*$/;
  var SHARE=/(%|доля|share|концентрац)/i;
  Array.prototype.slice.call(document.querySelectorAll("table")).forEach(function(t){
    try{
      var rows=t.tBodies[0]?t.tBodies[0].rows.length:0, big=rows>15;
      var wrap=document.createElement("div"); wrap.className="t2s-tab-wrap";
      t.parentNode.insertBefore(wrap,t);
      var tools=document.createElement("div"); tools.className="t2s-tools"; wrap.appendChild(tools);
      wrap.appendChild(t);
      var tab=new Tabulator(t,{layout:"fitColumns",movableColumns:true,resizableColumns:true,
        pagination:big?"local":false,paginationSize:15,
        columnDefaults:{headerFilter:"input",resizable:true,tooltip:true,headerHozAlign:"left"}});
      tab.on("tableBuilt",function(){
        try{
          var body=t.tBodies[0];
          tab.getColumns().forEach(function(col,ci){
            var title=(col.getDefinition().title||"");
            // числовая колонка? смотрим первые ячейки
            var numeric=true, seen=0;
            for(var r=0; r<Math.min(rows,8); r++){
              var td=body&&body.rows[r]&&body.rows[r].cells[ci];
              var v=td?td.textContent.trim():""; if(!v||v==="—")continue; seen++;
              if(!NUM.test(v)){numeric=false;break;}
            }
            if(numeric&&seen>0){col.updateDefinition({hozAlign:"right"});}
          });
        }catch(e){}
      });
      var bc=document.createElement("button"); bc.textContent="✕ Сбросить фильтры";
      bc.onclick=function(){try{tab.clearHeaderFilter();tab.clearFilter(true);}catch(e){}};
      tools.appendChild(bc);
      var b=document.createElement("button"); b.textContent="⬇ CSV";
      b.onclick=function(){tab.download("csv","data.csv");}; tools.appendChild(b);
    }catch(e){console.warn("tabulator init failed",e);}
  });
});
</script>
"""


def enhance_tables(html: str) -> str:
    """Вшить Tabulator и сделать все таблицы интерактивными. Идемпотентно."""
    if "t2s-tab-wrap" in html:
        return html
    assets = tabulator_head() + _TAB_INIT
    return html.replace("</body>", assets + "</body>", 1) if "</body>" in html else html + assets


# ---------------- Plotly-виджеты ----------------
# Семантические цвета (из status-палитры dataviz — фиксированы, читаются на свете и в тёмной):
# потери=critical, прирост=good, нейтраль=blue slot1.
_C = {"loss": "#d03b3b", "gain": "#0ca30c", "neut": "#2a78d6", "muted": "#898781"}


def chart(cid: str, traces: list[dict], layout: dict | None = None, *, height: int = 340) -> str:
    """Div + Plotly через T2S (theme-aware: подхватывает свет/тёмную и перерисовывается при
    смене темы). Данные — inline JSON, интерактив в браузере. Требует plotly_head в <head>."""
    lay = {"height": height}
    lay.update(layout or {})
    return (f"<div id='{cid}' class='t2s-plot'></div><script>"
            f"T2S.plot('{cid}',{json.dumps(traces, ensure_ascii=False)},"
            f"{json.dumps(lay, ensure_ascii=False)});</script>")


def bar(cid: str, x: list, y: list, *, title: str = "", color: str = "neut",
        horizontal: bool = True, unit: str = "", height: int = 340) -> str:
    col = _C.get(color, color)
    tr = {"type": "bar", "orientation": "h" if horizontal else "v",
          "marker": {"color": col, "cornerradius": 4},
          "text": [f"{v:,.0f}".replace(",", " ") for v in y], "textposition": "auto"}
    if horizontal:
        tr["x"], tr["y"] = y, [str(v) for v in x]
    else:
        tr["x"], tr["y"] = [str(v) for v in x], y
    lay = {"title": {"text": title, "font": {"size": 13}}}
    if unit:
        (lay.setdefault("xaxis", {}) if horizontal else lay.setdefault("yaxis", {}))["title"] = unit
    return chart(cid, [tr], lay, height=height)


def grouped_bar(cid: str, categories: list, series: dict[str, list], *, title: str = "",
                unit: str = "", height: int = 340) -> str:
    traces = [{"type": "bar", "name": str(name), "x": [str(c) for c in categories], "y": vals,
               "marker": {"cornerradius": 4},          # цвет — из colorway шаблона (theme-aware)
               "text": [f"{v:.2f}" for v in vals], "textposition": "auto"}
              for name, vals in series.items()]
    lay = {"title": {"text": title, "font": {"size": 13}}, "barmode": "group",
           "yaxis": {"title": unit}}
    return chart(cid, traces, lay, height=height)


def line(cid: str, x: list, series: dict[str, list], *, title: str = "", unit: str = "",
         marker_x=None, height: int = 340) -> str:
    traces = [{"type": "scatter", "mode": "lines+markers", "name": str(name),
               "x": [str(v) for v in x], "y": vals, "line": {"width": 2}, "marker": {"size": 8}}
              for name, vals in series.items()]
    lay = {"title": {"text": title, "font": {"size": 13}}, "yaxis": {"title": unit}}
    if marker_x is not None:
        lay["shapes"] = [{"type": "line", "x0": str(marker_x), "x1": str(marker_x),
                          "yref": "paper", "y0": 0, "y1": 1,
                          "line": {"color": _C["muted"], "dash": "dash"}}]
    return chart(cid, traces, lay, height=height)


def waterfall(cid: str, labels: list, values: list, *, title: str = "", unit: str = "",
              height: int = 340) -> str:
    tr = {"type": "waterfall", "orientation": "v", "x": [str(l) for l in labels], "y": values,
          "measure": ["absolute"] + ["relative"] * (len(values) - 2) + ["total"],
          "text": [f"{v:+,.0f}".replace(",", " ") for v in values], "textposition": "outside",
          "connector": {"line": {"color": _C["muted"]}}}
    lay = {"title": {"text": title, "font": {"size": 13}}, "yaxis": {"title": unit}}
    return chart(cid, [tr], lay, height=height)


def indicator(cid: str, value: float, *, title: str = "", suffix: str = "", height: int = 170) -> str:
    tr = {"type": "indicator", "mode": "number", "value": value,
          "number": {"suffix": suffix, "font": {"size": 40}},
          "title": {"text": title, "font": {"size": 13}}}
    return chart(cid, [tr], {"margin": {"t": 40, "b": 10, "l": 10, "r": 10}}, height=height)


def df_table(frame, *, limit: int = 12, fmt=None) -> str:
    """HTML-таблица из DataFrame (её усилит Tabulator). fmt(col,val)->str опционально."""
    if frame is None or getattr(frame, "empty", True):
        return ""
    show = frame.head(limit)
    cols = list(show.columns)
    th = "".join(f"<th>{_html.escape(str(c))}</th>" for c in cols)
    body = []
    for _, row in show.iterrows():
        tds = "".join(f"<td>{_html.escape(fmt(c, row[c]) if fmt else str(row[c]))}</td>" for c in cols)
        body.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


# ---------------- спек-билдеры (данные графика → Plotly-фигура) ----------------
# Возвращают dict {"traces":[...], "layout":{...}, "height":N}. Чарт-функции строят их
# рядом с matplotlib и кладут в сайдкар `<name>.plotly.json` (см. core._save), а HTML-рендер
# отдаёт интерактивный Plotly; MD и фолбэк — прежний PNG.
def _base_layout(title: str, **extra) -> dict:
    lay = {"title": {"text": title, "font": {"size": 13}},
           "margin": {"t": 40, "r": 14, "b": 46, "l": 60}}
    lay.update(extra)
    return lay


def spec_barh(labels, values, *, title="", unit="", color="neut", colors=None,
              baseline=None, pct=False) -> dict:
    mcolor = [_C.get(c, c) for c in colors] if colors is not None else _C.get(color, color)
    txt = [(f"{v:.0f}%" if pct else f"{v:,.0f}".replace(",", " ")) for v in values]
    tr = {"type": "bar", "orientation": "h", "x": list(values), "y": [str(l) for l in labels],
          "marker": {"color": mcolor, "cornerradius": 4}, "text": txt, "textposition": "auto",
          "hovertemplate": "%{y}: %{x}<extra></extra>"}
    lay = _base_layout(title, xaxis={"title": unit}, yaxis={"automargin": True})
    if baseline is not None:
        lay["shapes"] = [{"type": "line", "x0": baseline, "x1": baseline, "yref": "paper",
                          "y0": 0, "y1": 1, "line": {"color": "#888", "dash": "dash", "width": 1}}]
    return {"traces": [tr], "layout": lay, "height": max(300, 26 * len(labels) + 90)}


def spec_bar_v(labels, values, *, title="", unit="", colors=None, color="neut") -> dict:
    mcolor = [_C.get(c, c) for c in colors] if colors is not None else _C.get(color, color)
    tr = {"type": "bar", "x": [str(l) for l in labels], "y": list(values),
          "marker": {"color": mcolor, "cornerradius": 4},
          "text": [f"{v:,.0f}".replace(",", " ") for v in values], "textposition": "auto"}
    return {"traces": [tr], "layout": _base_layout(title, yaxis={"title": unit}), "height": 340}


def spec_line(x, series: dict, *, title="", unit="", marker_x=None) -> dict:
    traces = [{"type": "scatter", "mode": "lines+markers", "name": str(n),
               "x": [str(v) for v in x], "y": list(vals),
               "line": {"width": 2}, "marker": {"size": 8}}   # цвет — из colorway (theme-aware)
              for n, vals in series.items()]
    lay = _base_layout(title, yaxis={"title": unit})
    if marker_x is not None:
        lay["shapes"] = [{"type": "line", "x0": str(marker_x), "x1": str(marker_x), "yref": "paper",
                          "y0": 0, "y1": 1, "line": {"color": "#888", "dash": "dash"}}]
    return {"traces": traces, "layout": lay, "height": 360}


def spec_heatmap(z, x, y, *, title="", unit="", scale="Blues") -> dict:
    diverging = scale in ("RdYlGn", "div")
    tr = {"type": "heatmap", "z": z, "x": [str(c) for c in x], "y": [str(r) for r in y],
          "colorscale": _SCALE.get(scale, _SEQ), "colorbar": {"title": unit},
          "text": [[f"{v:.0f}" if v is not None else "" for v in row] for row in z],
          "texttemplate": "%{text}", "hoverongaps": False,
          "xgap": 2, "ygap": 2}                        # 2px surface gap между ячейками
    if diverging:
        tr["zmid"] = 0
    lay = _base_layout(title, xaxis={"tickangle": -35, "showgrid": False},
                       yaxis={"automargin": True, "showgrid": False})
    return {"traces": [tr], "layout": lay, "height": max(300, 40 * len(y) + 120)}


def spec_waterfall(labels, values, *, title="", unit="") -> dict:
    tr = {"type": "waterfall", "orientation": "v", "x": [str(l) for l in labels], "y": list(values),
          "measure": ["relative"] * (len(values) - 1) + ["total"],
          "text": [f"{v:+,.0f}".replace(",", " ") for v in values], "textposition": "outside",
          "connector": {"line": {"color": "#888"}}}
    return {"traces": [tr], "layout": _base_layout(title, yaxis={"title": unit}), "height": 380}


def spec_scatter(x, y, text, *, title="", xlab="", ylab="", vline=None, hline=None) -> dict:
    tr = {"type": "scatter", "mode": "markers", "x": list(x), "y": list(y),
          "text": [str(t) for t in text], "marker": {"size": 11, "color": _C["loss"], "opacity": 0.7},
          "hovertemplate": "%{text}<br>%{x}, %{y}<extra></extra>"}
    lay = _base_layout(title, xaxis={"title": xlab}, yaxis={"title": ylab})
    shapes = []
    if vline is not None:
        shapes.append({"type": "line", "x0": vline, "x1": vline, "yref": "paper", "y0": 0, "y1": 1,
                       "line": {"color": "#aaa", "dash": "dot"}})
    if hline is not None:
        shapes.append({"type": "line", "y0": hline, "y1": hline, "xref": "paper", "x0": 0, "x1": 1,
                       "line": {"color": "#aaa", "dash": "dot"}})
    if shapes:
        lay["shapes"] = shapes
    return {"traces": [tr], "layout": lay, "height": 420}


def spec_pareto(labels, bars, cum, *, title="", unit="") -> dict:
    traces = [{"type": "bar", "x": [str(l) for l in labels], "y": list(bars), "name": unit or "вклад",
               "marker": {"color": _C["loss"]}},
              {"type": "scatter", "x": [str(l) for l in labels], "y": list(cum), "name": "накопл. %",
               "yaxis": "y2", "mode": "lines+markers", "line": {"color": "#333"}}]
    lay = _base_layout(title, xaxis={"tickangle": -35}, yaxis={"title": unit},
                       yaxis2={"title": "%", "overlaying": "y", "side": "right", "range": [0, 100]})
    return {"traces": traces, "layout": lay, "height": 380}


def spec_treemap(labels, parents, values, *, ids=None, title="") -> dict:
    tr = {"type": "treemap", "labels": [str(l) for l in labels], "parents": [str(p) for p in parents],
          "values": list(values), "branchvalues": "total", "textinfo": "label+value+percent parent"}
    if ids is not None:
        tr["ids"] = [str(i) for i in ids]
    return {"traces": [tr], "layout": _base_layout(title), "height": 460}


def embed(chart_path: str | None) -> str:
    """HTML-встраивание графика: интерактивный Plotly из сайдкара `<name>.plotly.json`,
    иначе base64-PNG (фолбэк/для графиков без спека). Требует plotly в <head> страницы."""
    if not chart_path:
        return ""
    p = Path(chart_path)
    sidecar = p.parent / (p.stem + ".plotly.json")
    if sidecar.exists():
        try:
            spec = json.loads(sidecar.read_text(encoding="utf-8"))
            cid = "pl_" + re.sub(r"\W", "_", p.stem)
            return chart(cid, spec["traces"], spec.get("layout"), height=spec.get("height", 340))
        except Exception:  # noqa: BLE001  (битый спек → PNG-фолбэк)
            pass
    if p.exists():
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'>"
    return ""


def page(title: str, body: str, *, css: str = "", plotly: bool = True, tabulator: bool = True) -> str:
    """Собрать автономную HTML-страницу с вшитыми библиотеками (одна на файл)."""
    head = [f"<!doctype html><html lang='ru'><head><meta charset='utf-8'>",
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>",
            f"<title>{_html.escape(title)}</title><style>{THEME_CSS}{css}</style>"]
    if plotly:
        head.append(plotly_head())
    if tabulator:
        head.append(tabulator_head())
    doc = "".join(head) + "</head><body>" + body
    if tabulator:
        doc += _TAB_INIT
    return doc + "</body></html>"
