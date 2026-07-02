"""Промпты узлов. Принцип под слабые модели: маленькие, одна задача, явная
схема ответа, компактный контекст. Всё на русском (язык вопросов)."""

from __future__ import annotations

from typing import Any

from ..catalog.catalog import TableMeta

# ---------- SCOPE ----------

SCOPE_SYS = (
    "Ты выбираешь таблицы БД для ответа на вопрос аналитика.\n"
    "Рассуждай по шагам:\n"
    "1) Что именно считаем/измеряем (главное существительное) и какие у него уточнения.\n"
    "2) Для КАЖДОГО кандидата проверь: может ли он ответить на вопрос?\n"
    "3) Если ответить могут ДВЕ И БОЛЕЕ таблицы, но с РАЗНЫМ смыслом результата — "
    "это неоднозначность: верни её опции, НЕ выбирай молча.\n"
    "Явный признак неоднозначности: понятие из вопроса встречается и как отдельная "
    "таблица-сущность, и как атрибут/значение/флаг/подтип в ДРУГОЙ таблице — тогда "
    "и подсчёт по таблице-сущности, и фильтр по атрибуту равноправны, покажи оба.\n"
    "Если неоднозначности нет — верни минимальный набор таблиц и ambiguity=null.\n"
    'Схема ответа: {"chosen_tables":[fqn,...],'
    '"ambiguity":null | {"question":str,"options":[{"label":str,"tables":[fqn],"rationale":str}]}}'
)


def scope_user(question: str, corrections: list[str], candidates: list[dict]) -> str:
    lines = [f"Вопрос: {question}"]
    if corrections:
        lines.append("Правки пользователя: " + " | ".join(corrections))
    lines.append("\nКандидаты (таблица — роль/грануляция — описание):")
    for c in candidates:
        lines.append(f"- {c['fqn']} — {c.get('role','')}/{c.get('grain','')} — {c.get('description','')}")
    lines.append("\nВыбери таблицы и при наличии — опиши неоднозначность.")
    return "\n".join(lines)


# ---------- PLAN ----------

PLAN_SYS = (
    "Ты строишь СТРУКТУРИРОВАННЫЙ план SQL-запроса (форма select-join-aggregate). "
    "Используй только перечисленные таблицы и колонки. Правила:\n"
    "- Аналитическая БД: НЕЛЬЗЯ размножать строки. При join к справочнику соединяй "
    "по его ПОЛНОМУ составному ключу (PK-гипотеза справочника). Маппинг FK-колонок "
    "факта на колонки справочника бери ТОЛЬКО из join-кандидатов — по максимальному "
    "overlap значений. НЕ выбирай колонку по «новизне/актуальности» (для join важен "
    "overlap, а не тип идентификатора: gosb_id факта чаще ложится на old_gosb_id, "
    "если у него больший overlap). НЕ соединяй колонки разных типов (дата ≠ число) "
    "и НЕ выдумывай несуществующие колонки.\n"
    "- Если join-ключ НЕ уникален в справочной/атрибутивной таблице (в ней несколько "
    "строк на ключ — история, напр. epk по inn), НЕ делай прямой join (размножит строки). "
    "Вместо этого пометь этот источник как dedup: by = колонки join на стороне этого "
    "источника, order_by = колонка АКТУАЛЬНОСТИ (дата создания/обновления карточки, "
    "напр. epk_create_dttm), desc=true — возьмём СВЕЖУЮ строку на ключ. Тогда join к "
    "дедуплицированному источнику не множит строки.\n"
    "- Для агрегатов укажи metrics и group_by; неагрегированные колонки в SELECT "
    "обязаны быть в group_by.\n"
    "- Период по дате задавай ПОЛУОТКРЫТЫМ интервалом двумя фильтрами: "
    "col op '>=' value 'YYYY-MM-01' И col op '<' value '<первое-число-следующего-месяца>'. "
    "НЕ используй BETWEEN с последним днём месяца (ошибки вида 29 февраля, потеря "
    "последнего дня для timestamp). Пример «февраль 2026»: >= '2026-02-01' и < '2026-03-01'.\n"
    "- Булевы фильтры: op 'IS TRUE'/'IS FALSE'.\n"
    "- ТОЛЬКО при подсчёте количества сущностей в ОДНОЙ таблице (COUNT DISTINCT, без "
    "join) бери идентификатор АКТУАЛЬНЫЙ и наиболее уникальный (большой уник%, напр. "
    "'новый номер'), а не старый. Это правило НЕ относится к выбору колонок для join.\n"
    "- Для «сколько …» (одно число) НЕ добавляй сам идентификатор в projections/group_by: "
    "используй одну метрику count/count_distinct без разрезов, иначе получишь строку на "
    "каждую сущность вместо итога.\n"
    "- Текстовый фильтр по категории/подтипу: НАЙДИ колонку, в чьих «примерах значений» "
    "понятие из вопроса совпадает НАИБОЛЕЕ ПОЛНО, и фильтруй по ней (op 'ILIKE'). "
    "Предпочитай точное/полное совпадение фразы частичному: для «фактический отток» бери "
    "колонку со значением 'Фактический отток' (value '%фактический отток%'), а НЕ колонку, "
    "где есть лишь общее 'Отток'. Значение бери ПОЛНЫМ из формулировки вопроса, не усекай "
    "('%фактический отток%', а не '%отток%'). НЕ выдумывай значения и не переводи на английский. "
    "Если подходящего значения нет ни в одной колонке — добавь в ambiguities, не выдумывай условие.\n"
    "- ВЫРАЖЕНИЯ разрешены в projections.column, group_by и metrics.column: "
    "date_trunc('month', alias.col) — группировка по месяцам/кварталам; арифметика "
    "alias.a / NULLIF(alias.b,0) — доли/средние на единицу; CASE WHEN … THEN … — "
    "условные значения. Для МЕТРИКИ-выражения (напр. доля/ratio) ставь agg='none', а в "
    "column — полное выражение с агрегатами: 'AVG(CASE WHEN alias.flag THEN 1 ELSE 0 END)' "
    "или 'SUM(alias.a)/NULLIF(SUM(alias.b),0)'. Группируя по выражению, положи ТО ЖЕ "
    "выражение и в group_by.\n"
    "- HAVING: фильтр по АГРЕГАТУ (напр. sum>5000) клади в having (НЕ в filters/WHERE): "
    "having.column = агрегатное выражение 'SUM(alias.col)'.\n"
    "- LEFT JOIN: join_type='left' если нужно включить строки без пары (напр. все ГОСБ, "
    "даже без оттока). По умолчанию 'inner'.\n"
    "Схема ответа (StructuredPlan):\n"
    '{"intent":str,'
    '"tables":[{"ref":fqn,"alias":str,"dedup":null|{"by":[col],"order_by":col,"desc":true}}],'
    '"joins":[{"left_alias":str,"right_alias":str,"on":[[lcol,rcol]],"join_type":"inner|left"}],'
    '"projections":[{"column":"alias.col ИЛИ выражение","alias":str}],'
    '"metrics":[{"agg":"count|count_distinct|sum|avg|min|max|none","column":"alias.col|*|выражение","alias":str}],'
    '"filters":[{"column":"alias.col","op":str,"value":any,"value2":any}],'
    '"having":[{"column":"SUM(alias.col)","op":str,"value":any}],'
    '"group_by":["alias.col или выражение"],"order_by":["alias.col или alias метрики"],"limit":int|null,'
    '"grain_note":str,"assumptions":[str],"ambiguities":[str]}\n'
    "ЕСЛИ запрос требует ОКОННЫХ функций (SUM() OVER, RANK, LAG — доля от общего, ранг, "
    "изменение к прошлому периоду), коррелированных ПОДЗАПРОСОВ, полу-/анти-join "
    "(есть/нет в другой таблице через IN/NOT IN (SELECT…) или EXCEPT), UNION — структурой "
    "это НЕ выразить. Тогда верни ВМЕСТО плана прямой SQL: "
    '{"raw_sql":"<полный read-only SELECT или WITH…>","intent":str,"note":"почему прямой SQL"}. '
    "В raw_sql используй ТОЧНЫЕ полные имена таблиц и колонок как перечислены ниже "
    "(schema.table целиком, напр. s_grnplm_..._sn_uzp.uzp_dwh_sale_funnel_task — НЕ "
    "сокращай до sale_funnel_task). Соблюдай: не размножай строки (дедуп справочников "
    "через DISTINCT ON), полуоткрытые интервалы дат, ТОЛЬКО SELECT (без ; и DML). "
    "Предпочитай СТРУКТУРИРОВАННЫЙ план; raw_sql — только когда иначе никак."
)


def plan_user(question: str, corrections: list[str], tables: list[TableMeta],
              join_candidates: list[dict]) -> str:
    lines = [f"Вопрос: {question}"]
    if corrections:
        lines.append("Правки пользователя (учти их): " + " | ".join(corrections))
    for t in tables:
        lines.append(f"\nТаблица {t.fqn} (alias предложи сам). PK-гипотеза: {t.pk_hypothesis or '—'}")
        lines.append("Колонки [имя | тип | класс | pk | уник% | описание | примеры значений]:")
        for c in t.columns:
            pk = "PK" if c.is_pk_hypothesis else ""
            # Примеры значений критичны для текстовых фильтров: по ним видно, в какой
            # колонке лежит нужное понятие (напр. подтип «фактический отток»).
            samples = f" | примеры: {', '.join(c.sample_values[:25])}" if c.sample_values else ""
            lines.append(f"  {c.name} | {c.dtype} | {c.semantic_class} | {pk} | "
                         f"{c.unique_perc:g}% | {c.description}{samples}")
    if join_candidates:
        lines.append("\nJoin-кандидаты (overlap значений — чем больше, тем вернее маппинг):")
        for jc in join_candidates:
            pairs = ", ".join(f"{p['left_col']}↔{p['right_col']}(ov={p['overlap']})" for p in jc["pairs"])
            lines.append(f"- {jc['left']} ↔ {jc['right']}: {pairs}")
    lines.append("\nПострой план. Помни про запрет размножения строк.")
    return "\n".join(lines)


# ---------- COUNT-ID CORRECTOR (какой идентификатор считать) ----------

COUNT_ID_SYS = (
    "Ты выбираешь ПРАВИЛЬНЫЙ идентификатор для подсчёта количества уникальных "
    "сущностей (COUNT DISTINCT). Правила приоритета:\n"
    "1) Если в ВОПРОСЕ пользователь ЯВНО назвал конкретный идентификатор/колонку "
    "(напр. «по old_gosb_id», «по старому номеру») — используй ИМЕННО его.\n"
    "2) Иначе бери АКТУАЛЬНЫЙ/ТЕКУЩИЙ идентификатор сущности (напр. «новый номер»), "
    "а НЕ исторический/устаревший/технический/производный.\n"
    'Верни JSON {"column":"имя_колонки"} — ровно одну из предложенных колонок.'
)


def count_id_user(question: str, current: str, candidates: list) -> str:
    lines = "\n".join(
        f"- {c.name} | {c.description} | уникальность={c.unique_perc:g}%" for c in candidates
    )
    return (f"Вопрос пользователя: {question}\n"
            f"Сейчас в плане считаем COUNT(DISTINCT {current}).\n"
            f"Колонки-идентификаторы этой сущности:\n{lines}\n\n"
            "Какую колонку использовать для подсчёта?")


# ---------- JOIN FIX ----------

JOIN_FIX_SYS = (
    "Join в плане размножает строки (N:M): ключ не уникален ни на одной стороне. "
    "Исправь ключ join — добавь колонки, чтобы он покрывал полный составной ключ "
    "справочной стороны (см. PK-гипотезу и join-кандидаты). Верни ТОЛЬКО обновлённый "
    'JSON joins: {"joins":[{"left_alias":str,"right_alias":str,"on":[[lcol,rcol]]}]}'
)


def join_fix_user(plan: dict, verdict_note: str, join_candidates: list[dict],
                  pk_by_alias: dict[str, list[str]]) -> str:
    lines = [f"Текущие joins: {plan.get('joins')}", f"Вердикт: {verdict_note}",
             f"PK справочников по alias: {pk_by_alias}"]
    if join_candidates:
        lines.append("Join-кандидаты:")
        for jc in join_candidates:
            pairs = ", ".join(f"{p['left_col']}↔{p['right_col']}(ov={p['overlap']})" for p in jc["pairs"])
            lines.append(f"- {jc['left']} ↔ {jc['right']}: {pairs}")
    return "\n".join(lines)
