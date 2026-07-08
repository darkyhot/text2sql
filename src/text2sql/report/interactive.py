"""Интерактивность HTML-отчётов (этап D, слой рендера) — БЕЗ внешних библиотек.

ЧЕСТНОЕ УПРОЩЕНИЕ: вместо Tabulator/Plotly (мегабайты JS или CDN — ломает автономность
отчёта) — маленький self-contained vanilla-JS: любую таблицу делаем СОРТИРУЕМОЙ по клику
на заголовок (числа сортируются как числа), а к большим таблицам добавляем строку ПОИСКА.
Графики остаются встроенными PNG (matplotlib) — инвариант «HTML со встроенными картинками»
сохраняется, внешних запросов ноль.

Применение — один пост-процессор `enhance(html)`: вставляет <style>+<script> перед </body>,
скрипт на DOMContentLoaded сам находит все <table> и навешивает поведение. Точки рендера
таблиц трогать не нужно.
"""

from __future__ import annotations

# порог: таблицы длиннее — получают поле поиска
_FILTER_MIN_ROWS = 8

_ASSETS = """
<style>
.t2s-tablewrap{margin:8px 0}
.t2s-filter{width:100%;box-sizing:border-box;padding:6px 10px;margin:0 0 6px;
  border:1px solid #cdd6e2;border-radius:8px;font-size:13px}
table th[data-t2s]{cursor:pointer;user-select:none;position:relative}
table th[data-t2s]:hover{background:#e6edf6}
table th[data-t2s].t2s-asc::after{content:" \\25B2";font-size:10px;color:#5b6b7f}
table th[data-t2s].t2s-desc::after{content:" \\25BC";font-size:10px;color:#5b6b7f}
.t2s-hidden{display:none}
</style>
<script>
(function(){
  function num(s){
    // «-1.5 тыс», «52%», «1 234,5 ₽» → число, если это по сути число
    var t=(s||"").replace(/\\u00a0/g," ").replace(/[%₽]/g,"").trim();
    var mult=1;
    if(/тыс/.test(t)) mult=1e3; else if(/млн/.test(t)) mult=1e6; else if(/млрд/.test(t)) mult=1e9;
    t=t.replace(/(тыс|млн|млрд|чел\\.?|шт|дн)/g,"").replace(/\\s+/g,"").replace(",",".");
    if(t===""||isNaN(Number(t))) return null;
    return Number(t)*mult;
  }
  function sortBy(table, idx, dir){
    var tb=table.tBodies[0]; if(!tb) return;
    var rows=Array.prototype.slice.call(tb.rows);
    rows.sort(function(a,b){
      var x=(a.cells[idx]||{}).textContent||"", y=(b.cells[idx]||{}).textContent||"";
      var nx=num(x), ny=num(y), r;
      if(nx!==null&&ny!==null) r=nx-ny; else r=x.localeCompare(y,"ru");
      return dir==="asc"?r:-r;
    });
    rows.forEach(function(r){tb.appendChild(r);});
  }
  function enhance(table){
    var head=table.tHead||(table.rows[0]&&table.rows[0].parentNode.tagName==="THEAD"?table.rows[0].parentNode:null);
    var hrow=table.tHead?table.tHead.rows[0]:table.rows[0];
    if(!hrow) return;
    Array.prototype.forEach.call(hrow.cells,function(th,i){
      th.setAttribute("data-t2s","");
      th.addEventListener("click",function(){
        var dir=th.classList.contains("t2s-asc")?"desc":"asc";
        Array.prototype.forEach.call(hrow.cells,function(o){o.classList.remove("t2s-asc","t2s-desc");});
        th.classList.add(dir==="asc"?"t2s-asc":"t2s-desc");
        sortBy(table,i,dir);
      });
    });
    var body=table.tBodies[0];
    if(body&&body.rows.length>=%FILTER_MIN%){
      var wrap=document.createElement("div"); wrap.className="t2s-tablewrap";
      table.parentNode.insertBefore(wrap,table);
      var inp=document.createElement("input");
      inp.className="t2s-filter"; inp.type="search"; inp.placeholder="Фильтр по таблице…";
      wrap.appendChild(inp); wrap.appendChild(table);
      inp.addEventListener("input",function(){
        var q=inp.value.toLowerCase();
        Array.prototype.forEach.call(body.rows,function(r){
          r.classList.toggle("t2s-hidden", q && r.textContent.toLowerCase().indexOf(q)<0);
        });
      });
    }
  }
  document.addEventListener("DOMContentLoaded",function(){
    Array.prototype.forEach.call(document.querySelectorAll("table"),enhance);
  });
})();
</script>
""".replace("%FILTER_MIN%", str(_FILTER_MIN_ROWS))


def enhance(html: str) -> str:
    """Вставить интерактивность (сортировка + фильтр таблиц) перед </body>. Идемпотентно."""
    if "t2s-filter" in html:                       # уже усилено — не дублируем
        return html
    if "</body>" in html:
        return html.replace("</body>", _ASSETS + "</body>", 1)
    return html + _ASSETS
