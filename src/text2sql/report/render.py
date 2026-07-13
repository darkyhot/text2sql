"""Render Engine (§5.5–5.6 дизайна): интерактивный ОФЛАЙН-HTML на вшитых Apache ECharts + Tabulator.

Библиотеки лежат в `assets/` (vendored) и ИНЛАЙНЯТСЯ в HTML — файл полностью автономен,
нулевые внешние запросы (банковский офлайн-контур). Графики — ECharts (легче Plotly ~в 4 раза,
современнее из коробки, нативные treemap/heatmap/waterfall) через тема-обёртку T2S (свет/тёмная,
ресайз, форматтеры чисел). Таблицы — Tabulator (сортировка/поиск/пагинация/сброс фильтров/CSV).
Спек-билдеры отдают ECharts `option`; чарт-функции кладут его в сайдкар `<name>.chart.json`,
HTML-рендер (`embed`) отдаёт интерактив, MD/фолбэк — прежний PNG.
"""

from __future__ import annotations

import base64
import html as _html
import json
import re
from pathlib import Path

_ASSETS = Path(__file__).parent / "assets"
_cache: dict[str, str] = {}


def _safe_js(js: str) -> str:
    """Экранировать `</script>` внутри инлайнимого JS — иначе минифицированная библиотека
    (в ECharts есть строка с `</script>`) преждевременно закрывает <script>-тег, и остаток
    вываливается в страницу как текст. `<\\/script>` для JS эквивалентно `</script>`."""
    return re.sub(r"</(script)", r"<\\/\1", js, flags=re.I)


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
_SEQ_HEX = ["#cde2fb", "#3987e5", "#0d366b"]                     # sequential blue
_SEQ_RED_HEX = ["#fbe0da", "#e34948", "#8f1f1e"]
_DIV_HEX = ["#2a78d6", "#f0efec", "#d03b3b"]                     # diverging blue↔gray↔red
_SCALE = {"Blues": _SEQ_HEX, "Reds": _SEQ_RED_HEX, "RdYlGn": _DIV_HEX, "seq": _SEQ_HEX, "div": _DIV_HEX}

# Тема-обёртка T2S для Apache ECharts (theme-aware: базовая тема свет/тёмная мержится в каждый
# график, перерисовка при смене темы ОС; ресайз). Форматтеры чисел (пробел-разделители, %,
# знак) подставляются на клиенте через сентинелы «__int__»/«__pct__»/… (в JSON функции нельзя).
_T2S_JS = """
window.T2S=window.T2S||{};T2S.charts=[];T2S.insts={};T2S.PAL=__PAL__;
T2S.dark=function(){return !!(window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches);};
T2S.FMT={
 __int__:function(p){var v=(p&&p.value!=null)?p.value:p;return v==null||isNaN(v)?'':Number(v).toLocaleString('ru-RU',{maximumFractionDigits:0});},
 __pct__:function(p){var v=(p&&p.value!=null)?p.value:p;return v==null||isNaN(v)?'':Math.round(v)+'%';},
 __signed__:function(p){var v=(p&&p.value!=null)?p.value:p;if(v==null||isNaN(v))return '';return (v>0?'+':'')+Number(v).toLocaleString('ru-RU',{maximumFractionDigits:0});},
 __scatter__:function(p){return (p.data&&p.data[2]?p.data[2]+'<br>':'')+Number(p.data[0]).toLocaleString('ru-RU')+' · '+Number(p.data[1]).toLocaleString('ru-RU');}
};
T2S.hydrate=function(o){if(!o||typeof o!=='object')return;for(var k in o){var v=o[k];
 if(k==='formatter'&&typeof v==='string'&&T2S.FMT[v]){o[k]=T2S.FMT[v];}else if(v&&typeof v==='object'){T2S.hydrate(v);}}};
T2S.base=function(){var d=T2S.dark();
 var ink=d?'#fff':'#0b0b0b',sec=d?'#c3c2b7':'#52514e',surf=d?'#1a1a19':'#fcfcfb',grid=d?'#2c2c2a':'#e1e0d9';
 return {color:d?T2S.PAL.dark:T2S.PAL.light,
  textStyle:{fontFamily:'system-ui,-apple-system,Segoe UI,sans-serif',color:sec},
  title:{left:2,top:4,textStyle:{color:ink,fontSize:14,fontWeight:600}},
  grid:{left:8,right:22,top:46,bottom:8,containLabel:true},
  tooltip:{backgroundColor:surf,borderColor:grid,borderWidth:1,textStyle:{color:ink},confine:true,
   extraCssText:'box-shadow:0 6px 18px rgba(0,0,0,.14);border-radius:10px;padding:8px 10px;'},
  legend:{bottom:0,textStyle:{color:sec},icon:'roundRect',itemWidth:12,itemHeight:8}};};
T2S.ax=function(){var d=T2S.dark();var mut='#898781',grid=d?'#2c2c2a':'#e1e0d9',axl=d?'#383835':'#c3c2b7';
 return {cat:{axisLine:{lineStyle:{color:axl}},axisTick:{show:false},axisLabel:{color:mut},splitLine:{show:false}},
  val:{axisLine:{show:false},axisTick:{show:false},axisLabel:{color:mut},splitLine:{lineStyle:{color:grid,type:'dashed'}}}};};
T2S.axisStyle=function(opt){var A=T2S.ax();['xAxis','yAxis'].forEach(function(k){if(!opt[k])return;
 var arr=Array.isArray(opt[k])?opt[k]:[opt[k]];arr.forEach(function(a){var st=(a.type==='value')?A.val:A.cat;
  for(var p in st){a[p]=Object.assign({},st[p],a[p]||{});}});});};
T2S.draw=function(cid,option){var el=document.getElementById(cid);if(!el||!window.echarts)return;
 var inst=T2S.insts[cid];if(inst){inst.dispose();}inst=echarts.init(el,null,{renderer:'canvas'});T2S.insts[cid]=inst;
 var opt=JSON.parse(JSON.stringify(option));T2S.hydrate(opt);T2S.axisStyle(opt);
 inst.setOption(T2S.base());inst.setOption(opt);};
T2S.plot=function(cid,option){T2S.charts.push([cid,option]);T2S.draw(cid,option);};
if(window.matchMedia){matchMedia('(prefers-color-scheme: dark)').addEventListener('change',function(){
 T2S.charts.forEach(function(c){T2S.draw(c[0],c[1]);});});}
window.addEventListener('resize',function(){for(var k in T2S.insts){if(T2S.insts[k])T2S.insts[k].resize();}});
"""


def charts_head() -> str:
    """Вшитый ECharts + тема-обёртка T2S. Одна на HTML-файл (в <head>)."""
    js = _T2S_JS.replace("__PAL__", json.dumps({"light": _CAT_LIGHT, "dark": _CAT_DARK}, ensure_ascii=False))
    return (f"<script>{_safe_js(_read('echarts.min.js'))}</script>"
            f"<script>{_safe_js(js)}</script>")


def tabulator_head() -> str:
    return (f"<style>{_read('tabulator.min.css')}</style>\n"
            f"<script>{_safe_js(_read('tabulator.min.js'))}</script>")


# Тема Tabulator (токены дизайн-системы, свет/тёмная): без зебры, тонкие горизонтальные
# разделители, липкая приглушённая шапка, tabular-nums, мягкий hover-wash. Числовые колонки —
# вправо; колонки-доли (%/доля) получают прогресс-бар в ячейке.
_TAB_CSS = """
<style>.t2s-gsum{color:var(--ink-2);font-weight:600;margin-left:8px}
.t2s-gcnt{color:var(--muted);font-size:.85em;margin-left:6px}</style>
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

_TAB_INIT = _TAB_CSS + r"""
<script>
document.addEventListener("DOMContentLoaded",function(){
  if(typeof Tabulator==="undefined")return;
  var NUM=/^[\\s+\\-]?[\\d \\u00a0.,]+%?\\s*(тыс|млн|млрд|₽|руб|чел\\.?|шт|дн|%|п\\.п\\.)?\\s*$/;
  var SHARE=/(%|доля|share|концентрац)/i;
  function fmtNum(v){                      // подытоги групп: 1 234 567 → «1.2 млн»
    var a=Math.abs(v);
    if(a>=1e6) return (v/1e6).toFixed(1).replace(/\.0$/,"")+" млн";
    if(a>=1e3) return (v/1e3).toFixed(1).replace(/\.0$/,"")+" тыс";
    return String(Math.round(v));
  }
  Array.prototype.slice.call(document.querySelectorAll("table")).forEach(function(t){
    try{
      var thead=t.tHead, tbody=t.tBodies[0];
      if(!thead||!thead.rows.length||!tbody)return;
      var isPivot=t.hasAttribute("data-pivot");
      var ths=Array.prototype.slice.call(thead.rows[0].cells);
      var data=Array.prototype.slice.call(tbody.rows).map(function(tr){
        var o={};
        Array.prototype.slice.call(tr.cells).forEach(function(td,i){
          o["c"+i]=td.textContent.trim();
          if(td.hasAttribute("data-v")) o["v"+i]=parseFloat(td.getAttribute("data-v"));
        });
        return o;
      });
      // Выравнивание считаем ДО создания таблицы и задаём в определении колонок.
      // updateDefinition() на HTML-импорте пересоздаёт колонку и теряет привязку к полю —
      // числовые колонки («250.1 тыс») из-за этого рендерились ПУСТЫМИ.
      var cols=ths.map(function(th,i){
        var numeric=true, seen=0;
        for(var r=0; r<Math.min(data.length,8); r++){
          var v=data[r]["c"+i]||""; if(!v||v==="—")continue; seen++;
          if(!NUM.test(v)){numeric=false;break;}
        }
        return {title:th.textContent.trim(), field:"c"+i,
                hozAlign:(numeric&&seen>0)?"right":"left"};
      });
      var big=data.length>15;
      var wrap=document.createElement("div"); wrap.className="t2s-tab-wrap";
      t.parentNode.insertBefore(wrap,t);
      var tools=document.createElement("div"); tools.className="t2s-tools"; wrap.appendChild(tools);
      var host=document.createElement("div"); wrap.appendChild(host);
      t.parentNode.removeChild(t);              // исходную HTML-таблицу заменяем на Tabulator
      var opt={data:data,columns:cols,layout:"fitColumns",
        movableColumns:true,resizableColumns:true,
        pagination:big?"local":false,paginationSize:15,
        columnDefaults:{headerFilter:"input",resizable:true,tooltip:true,headerHozAlign:"left"}};
      // СВОДНАЯ: разрезы — колонки, иерархию и ПОДЫТОГИ считает Tabulator (groupBy).
      // Суммы честные: усечённые ветки пришли строкой «прочие», она входит в группу.
      var nd=isPivot?parseInt(t.getAttribute("data-pivot"),10):0;
      if(isPivot){
        var vIdx=[]; for(var q=nd;q<ths.length;q++) vIdx.push(q);
        opt.groupBy=cols.slice(0,nd).map(function(c){return c.field;});
        opt.groupStartOpen=[true,false,false,false,false];
        opt.pagination=false; opt.movableColumns=false;
        opt.groupHeader=function(value,count,rws){
          var sums=vIdx.map(function(i){
            return rws.reduce(function(a,r){var d=r.getData?r.getData():r; return a+(d["v"+i]||0);},0);
          });
          return "<b>"+(value||"—")+"</b> <span class='t2s-gsum'>"+sums.map(fmtNum).join(" · ")
                 +"</span> <span class='t2s-gcnt'>("+count+")</span>";
        };
      }
      var tab=new Tabulator(host,opt);
      if(isPivot){
        var eall=false;
        var be=document.createElement("button"); be.textContent="⊞ Развернуть всё";
        be.onclick=function(){
          eall=!eall;
          try{
            (function walk(gs){gs.forEach(function(g){
              eall?g.show():g.hide();
              var sg=g.getSubGroups(); if(sg&&sg.length) walk(sg);
            });})(tab.getGroups());
          }catch(e){console.warn("expand all",e);}
          be.textContent=eall?"⊟ Свернуть всё":"⊞ Развернуть всё";
        };
        tools.appendChild(be);
      }
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
    """Вшить Tabulator и сделать все таблицы интерактивными. Идемпотентно.
    Вставляем перед ПОСЛЕДНИМ </body> (не первым!): вшитый ECharts содержит `</body>` внутри
    строкового литерала (saveAsImage), и `.replace('</body>', …)` по первому вхождению разрезал
    бы библиотеку пополам."""
    if "t2s-tab-wrap" in html:
        return html
    assets = tabulator_head() + _TAB_INIT
    idx = html.rfind("</body>")
    return (html[:idx] + assets + html[idx:]) if idx != -1 else html + assets


# ---------------- ECharts-виджеты ----------------
# Семантические цвета (status-палитра dataviz — фиксированы, читаются на свете и в тёмной):
# потери=critical, прирост=good, нейтраль=blue slot1.
_C = {"loss": "#d03b3b", "gain": "#0ca30c", "neut": "#2a78d6", "muted": "#898781"}


def _hex(c: str) -> str:
    return _C.get(c, c)


def chart(cid: str, option: dict, *, height: int = 340) -> str:
    """Div + ECharts через T2S (theme-aware, ресайз). option — ECharts-схема (inline JSON).
    Требует charts_head в <head> страницы."""
    return (f"<div id='{cid}' class='t2s-plot' style='width:100%;height:{height}px'></div>"
            f"<script>T2S.plot('{cid}',{json.dumps(option, ensure_ascii=False)});</script>")


def bar(cid: str, x: list, y: list, *, title: str = "", color: str = "neut",
        horizontal: bool = True, unit: str = "", height: int = 340) -> str:
    spec = (spec_barh(x, y, title=title, unit=unit, color=color) if horizontal
            else spec_bar_v(x, y, title=title, unit=unit, color=color))
    return chart(cid, spec["option"], height=spec.get("height", height))


def grouped_bar(cid: str, categories: list, series: dict[str, list], *, title: str = "",
                unit: str = "", height: int = 340) -> str:
    opt = {"title": {"text": title},
           "xAxis": {"type": "category", "data": [str(c) for c in categories]},
           "yAxis": {"type": "value", "name": unit},
           "legend": {"data": [str(n) for n in series]}, "tooltip": {"trigger": "axis"},
           "series": [{"type": "bar", "name": str(n), "data": list(v), "barMaxWidth": 34,
                       "itemStyle": {"borderRadius": [3, 3, 0, 0]}} for n, v in series.items()]}
    return chart(cid, opt, height=height)


def line(cid: str, x: list, series: dict[str, list], *, title: str = "", unit: str = "",
         marker_x=None, height: int = 340) -> str:
    return chart(cid, spec_line(x, series, title=title, unit=unit, marker_x=marker_x)["option"],
                 height=height)


def waterfall(cid: str, labels: list, values: list, *, title: str = "", unit: str = "",
              height: int = 340) -> str:
    return chart(cid, spec_waterfall(labels, values, title=title, unit=unit)["option"], height=height)


def indicator(cid: str, value: float, *, title: str = "", suffix: str = "", height: int = 170) -> str:
    """KPI-плитка (HTML, не график) — крупное число + подпись. Легче и «плиточнее» гейджа."""
    try:
        fv = float(value)
        txt = f"{fv:,.0f}".replace(",", " ") if abs(fv) >= 100 else f"{fv:g}"
    except (TypeError, ValueError):
        txt = str(value)
    return (f"<div class='kpi-tile'><div class='k-label'>{_html.escape(title)}</div>"
            f"<div class='k-value'>{txt}{_html.escape(suffix)}</div></div>")


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


# ---------------- спек-билдеры (данные графика → ECharts option) ----------------
# Возвращают dict {"option": {...}, "height": N}. Чарт-функции строят их рядом с matplotlib и
# кладут в сайдкар `<name>.chart.json` (core._save), а HTML-рендер отдаёт интерактивный ECharts;
# MD и фолбэк — прежний PNG. Форматтеры чисел — сентинелы «__int__»/«__pct__» (см. _T2S_JS).
def spec_barh(labels, values, *, title="", unit="", color="neut", colors=None,
              baseline=None, pct=False) -> dict:
    labels = [str(l) for l in labels]
    values = [float(v) for v in values]
    if colors is not None:
        data = [{"value": v, "itemStyle": {"color": _hex(c)}} for v, c in zip(values, colors)]
    else:
        data = [{"value": v} for v in values]
    ser = {"type": "bar", "data": data, "barMaxWidth": 22,
           "itemStyle": {"color": _hex(color), "borderRadius": [0, 4, 4, 0]},
           "label": {"show": True, "position": "right", "color": "inherit",
                     "formatter": "__pct__" if pct else "__int__"}}
    if baseline is not None:
        ser["markLine"] = {"symbol": "none", "silent": True, "data": [{"xAxis": float(baseline)}],
                           "lineStyle": {"type": "dashed", "color": "#888"}, "label": {"show": False}}
    opt = {"title": {"text": title}, "grid": {"left": 8, "right": 66, "top": 46, "bottom": 8, "containLabel": True},
           "xAxis": {"type": "value", "name": unit}, "yAxis": {"type": "category", "data": labels},
           "tooltip": {"trigger": "item"}, "series": [ser]}
    return {"option": opt, "height": max(300, 26 * len(labels) + 96)}


def spec_bar_v(labels, values, *, title="", unit="", colors=None, color="neut") -> dict:
    labels = [str(l) for l in labels]
    values = [float(v) for v in values]
    if colors is not None:
        data = [{"value": v, "itemStyle": {"color": _hex(c)}} for v, c in zip(values, colors)]
    else:
        data = [{"value": v} for v in values]
    ser = {"type": "bar", "data": data, "barMaxWidth": 40,
           "itemStyle": {"color": _hex(color), "borderRadius": [4, 4, 0, 0]},
           "label": {"show": True, "position": "top", "color": "inherit", "formatter": "__int__"}}
    opt = {"title": {"text": title}, "xAxis": {"type": "category", "data": labels},
           "yAxis": {"type": "value", "name": unit}, "tooltip": {"trigger": "item"}, "series": [ser]}
    return {"option": opt, "height": 340}


def spec_line(x, series: dict, *, title="", unit="", marker_x=None) -> dict:
    ser = []
    for i, (n, vals) in enumerate(series.items()):
        s = {"type": "line", "name": str(n), "data": [float(v) for v in vals], "smooth": False,
             "symbolSize": 8, "lineStyle": {"width": 2}, "showSymbol": True}
        if i == 0 and marker_x is not None:
            s["markLine"] = {"symbol": "none", "silent": True, "data": [{"xAxis": str(marker_x)}],
                             "lineStyle": {"type": "dashed", "color": "#888"}, "label": {"show": False}}
        ser.append(s)
    opt = {"title": {"text": title}, "tooltip": {"trigger": "axis"},
           "xAxis": {"type": "category", "data": [str(v) for v in x], "boundaryGap": False},
           "yAxis": {"type": "value", "name": unit}, "series": ser}
    if len(ser) > 1:
        opt["legend"] = {"data": [str(n) for n in series]}
    return {"option": opt, "height": 360}


def spec_heatmap(z, x, y, *, title="", unit="", scale="Blues") -> dict:
    diverging = scale in ("RdYlGn", "div")
    data, vals = [], []
    for yi, row in enumerate(z):
        for xi, v in enumerate(row):
            if v is None:
                continue
            data.append([xi, yi, float(v)])
            vals.append(float(v))
    m = max((abs(v) for v in vals), default=1.0)
    vmin, vmax = (-m, m) if diverging else (min(vals, default=0.0), max(vals, default=1.0))
    opt = {"title": {"text": title},
           "grid": {"top": 46, "bottom": 64, "left": 8, "right": 16, "containLabel": True},
           "xAxis": {"type": "category", "data": [str(c) for c in x], "axisLabel": {"rotate": 35}},
           "yAxis": {"type": "category", "data": [str(r) for r in y]},
           "visualMap": {"min": vmin, "max": vmax, "calculable": True, "orient": "horizontal",
                         "left": "center", "bottom": 0, "inRange": {"color": _SCALE.get(scale, _SEQ_HEX)},
                         "text": [unit, ""]},
           "tooltip": {"trigger": "item"},
           "series": [{"type": "heatmap", "data": data,
                       "label": {"show": True, "formatter": "__int__", "fontSize": 10},
                       "itemStyle": {"borderColor": "#fff", "borderWidth": 2}}]}
    return {"option": opt, "height": max(300, 40 * len(y) + 130)}


def spec_waterfall(labels, values, *, title="", unit="") -> dict:
    labels = [str(l) for l in labels]
    vals = [float(v) for v in values]
    placeholder, bars, cols = [], [], []
    run = 0.0
    for i in range(len(vals) - 1):
        d = vals[i]
        if d >= 0:
            placeholder.append(run); bars.append(d); cols.append(_C["gain"])
        else:
            placeholder.append(run + d); bars.append(-d); cols.append(_C["loss"])
        run += d
    placeholder.append(0.0); bars.append(vals[-1]); cols.append(_C["neut"])   # итог от нуля
    bardata = [{"value": b, "itemStyle": {"color": c}} for b, c in zip(bars, cols)]
    opt = {"title": {"text": title}, "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
           "xAxis": {"type": "category", "data": labels, "axisLabel": {"rotate": 30}},
           "yAxis": {"type": "value", "name": unit},
           "series": [
               {"type": "bar", "stack": "wf", "silent": True, "itemStyle": {"color": "transparent"},
                "data": placeholder, "tooltip": {"show": False}},
               {"type": "bar", "stack": "wf", "data": bardata, "barMaxWidth": 42,
                "itemStyle": {"borderRadius": [3, 3, 0, 0]},
                "label": {"show": True, "position": "top", "color": "inherit", "formatter": "__int__"}}]}
    return {"option": opt, "height": 380}


def spec_scatter(x, y, text, *, title="", xlab="", ylab="", vline=None, hline=None) -> dict:
    data = [[float(a), float(b), str(t)] for a, b, t in zip(x, y, text)]
    ser = {"type": "scatter", "data": data, "symbolSize": 12,
           "itemStyle": {"color": _C["loss"], "opacity": 0.7}}
    ml = []
    if vline is not None:
        ml.append({"xAxis": float(vline)})
    if hline is not None:
        ml.append({"yAxis": float(hline)})
    if ml:
        ser["markLine"] = {"symbol": "none", "silent": True, "data": ml,
                           "lineStyle": {"type": "dotted", "color": "#aaa"}, "label": {"show": False}}
    opt = {"title": {"text": title}, "tooltip": {"trigger": "item", "formatter": "__scatter__"},
           "xAxis": {"type": "value", "name": xlab}, "yAxis": {"type": "value", "name": ylab},
           "series": [ser]}
    return {"option": opt, "height": 420}


def spec_treemap(labels, parents, values, *, ids=None, title="") -> dict:
    ids = list(ids) if ids is not None else [str(i) for i in range(len(labels))]
    nodes = {i: {"name": str(l), "value": float(v), "children": []}
             for i, l, v in zip(ids, labels, values)}
    roots = []
    for i, par in zip(ids, parents):
        if par and par in nodes:
            nodes[par]["children"].append(nodes[i])
        else:
            roots.append(nodes[i])
    def _clean(nd):
        if nd.get("children"):
            for ch in nd["children"]:
                _clean(ch)
        else:
            nd.pop("children", None)
    for r in roots:
        _clean(r)
    opt = {"title": {"text": title},
           "series": [{"type": "treemap", "data": roots, "roam": False, "nodeClick": "zoomToNode",
                       "breadcrumb": {"show": True, "bottom": 2},
                       "label": {"show": True, "formatter": "{b}"},
                       "upperLabel": {"show": True, "height": 22},
                       "levels": [{"itemStyle": {"gapWidth": 2, "borderWidth": 0}},
                                  {"itemStyle": {"gapWidth": 2, "borderColorSaturation": 0.3}}]}]}
    return {"option": opt, "height": 460}


def embed(chart_path: str | None) -> str:
    """HTML-встраивание графика: интерактивный ECharts из сайдкара `<name>.chart.json`,
    иначе base64-PNG (фолбэк/для графиков без спека). Требует charts_head в <head> страницы."""
    if not chart_path:
        return ""
    p = Path(chart_path)
    sidecar = p.parent / (p.stem + ".chart.json")
    if sidecar.exists():
        try:
            spec = json.loads(sidecar.read_text(encoding="utf-8"))
            cid = "ec_" + re.sub(r"\W", "_", p.stem)
            return chart(cid, spec["option"], height=spec.get("height", 340))
        except Exception:  # noqa: BLE001  (битый спек → PNG-фолбэк)
            pass
    if p.exists():
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'>"
    return ""


def page(title: str, body: str, *, css: str = "", charts: bool = True, tabulator: bool = True) -> str:
    """Собрать автономную HTML-страницу с вшитыми библиотеками (одна на файл)."""
    head = [f"<!doctype html><html lang='ru'><head><meta charset='utf-8'>",
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>",
            f"<title>{_html.escape(title)}</title><style>{THEME_CSS}{css}</style>"]
    if charts:
        head.append(charts_head())
    if tabulator:
        head.append(tabulator_head())
    doc = "".join(head) + "</head><body>" + body
    if tabulator:
        doc += _TAB_INIT
    return doc + "</body></html>"
