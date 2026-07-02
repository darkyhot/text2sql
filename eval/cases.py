"""Эталонные кейсы + проверки. Грейдинг по свойствам результата/плана, а не по
тексту SQL (несколько SQL валидны). Каждый кейс: вопрос, выбор при неоднозначности
и список проверок check(turn, db)->(ok, detail)."""

from __future__ import annotations

S = "s_grnplm_ld_salesntwrk_pcap_sn_uzp"


def _values(turn):
    rows = _result_rows(turn)
    return {v for r in rows for v in r.values()}


def _result_rows(turn):
    import csv
    res = turn.result or {}
    path = res.get("csv")
    if not path:
        return []
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _plan(turn):
    return turn.state.get("plan") or {}


def _join_keys(turn):
    keys = set()
    for j in _plan(turn).get("joins", []):
        for pair in j.get("on", []):
            keys.update(pair)
    return keys


def _no_fanout(turn):
    joins = _plan(turn).get("joins", [])
    return all(j.get("fanout_safe") is not False for j in joins)


# ---- проверки ----
def chk_q1_distinct_counts(turn, db):
    """Результат содержит реальные distinct tb_id и new_gosb_id (считаем из БД)."""
    ref = db.run_select(
        f"SELECT count(DISTINCT tb_id) a, count(DISTINCT new_gosb_id) b FROM {S}.uzp_dim_gosb")
    exp = {str(ref.rows[0]["a"]), str(ref.rows[0]["b"])}
    got = {str(v) for v in _values(turn)}
    return exp.issubset(got), f"expected⊆got? exp={exp} got={got}"


def chk_single_row(turn, db):
    return (turn.result or {}).get("rowcount") == 1, f"rowcount={(turn.result or {}).get('rowcount')}"


def chk_join_no_fanout(turn, db):
    return _no_fanout(turn), f"joins={_plan(turn).get('joins')}"


def chk_join_keys_full_pk(turn, db):
    keys = _join_keys(turn)
    ok = {"tb_id", "old_gosb_id"}.issubset(keys)
    return ok, f"join_keys={keys}"


def chk_has_sum_metric(turn, db):
    metrics = _plan(turn).get("metrics", [])
    ok = any(m.get("agg") == "sum" for m in metrics)
    return ok, f"metrics={[(m.get('agg'),m.get('column')) for m in metrics]}"


def chk_q2_has_rows(turn, db):
    """Join вернул непустой результат (точный rowcount зависит от данных — не хардкодим)."""
    rc = (turn.result or {}).get("rowcount") or 0
    return rc > 0, f"rowcount={rc}"


def _filters(turn):
    return _plan(turn).get("filters", [])


def _has_filter(turn, col_sub, val_sub=None):
    for f in _filters(turn):
        c = str(f.get("column", "")).lower()
        v = str(f.get("value", "")).lower()
        if col_sub in c and (val_sub is None or val_sub in v):
            return True
    return False


def chk_q3_correct_interpretation(turn, db):
    """Q3: верная трактовка 'фактический отток' — ЛИБО fact_outflow.is_task,
    ЛИБО sale_funnel_task.task_subtype ~ 'отток'. task_type='Отток' (широкий) — НЕВЕРНО."""
    tables = {t["ref"] for t in _plan(turn).get("tables", [])}
    variant_fact = (f"{S}.uzp_dwh_fact_outflow" in tables) and _has_filter(turn, "is_task")
    variant_task = (f"{S}.uzp_dwh_sale_funnel_task" in tables) and _has_filter(turn, "task_subtype", "отток")
    ok = variant_fact or variant_task
    detail = (f"tables={[t.split('.')[-1] for t in tables]} "
              f"fact.is_task={variant_fact} task.subtype~отток={variant_task} "
              f"filters={[(f.get('column'), f.get('op'), f.get('value')) for f in _filters(turn)]}")
    return ok, detail


def chk_q3_feb_date(turn, db):
    """Должен быть фильтр по дате создания/отчёта в феврале 2026."""
    ok = _has_filter(turn, "dt", "2026-02") or _has_filter(turn, "report_dt", "2026-02") \
        or _has_filter(turn, "create", "2026-02")
    return ok, f"filters={[(f.get('column'), f.get('value')) for f in _filters(turn)]}"


def chk_dedup_join(turn, db):
    """Q4: join к неуникальному справочнику epk по inn должен идти через ДЕДУПЛИКАЦИЮ
    (свежая карточка), а не размножать строки."""
    tables = _plan(turn).get("tables", [])
    deduped = any(t.get("dedup") and t["dedup"].get("by") for t in tables)
    return deduped and _no_fanout(turn), \
        f"dedup={[t.get('dedup') for t in tables if t.get('dedup')]} no_fanout={_no_fanout(turn)}"


def _sql(turn):
    return (turn.state.get("sql") or "").lower()


def chk_month_bucket(turn, db):
    s = _sql(turn)
    return "date_trunc" in s, f"date_trunc in sql? sql={s[:120]}"


def chk_having(turn, db):
    return "having" in _sql(turn), f"having in sql? sql={_sql(turn)[:120]}"


def chk_ratio_expr(turn, db):
    s = _sql(turn)
    return ("case when" in s or "/" in s), f"выражение-доля? sql={s[:140]}"


def chk_window_or_raw(turn, db):
    s = _sql(turn)
    return bool(turn.state.get("raw_sql")) or "over" in s, f"raw={bool(turn.state.get('raw_sql'))} over={'over' in s}"


CASES = [
    {
        "id": "Q1_count_tb_gosb",
        "question": "Сколько всего есть тб и госб?",
        "ambiguity_pick": 0,
        "checks": [chk_q1_distinct_counts, chk_single_row],
    },
    {
        "id": "Q2_outflow_by_date_gosb",
        "question": "Посчитай сумму оттоков по дате и названию ГОСБ",
        "ambiguity_pick": 0,
        "checks": [chk_join_no_fanout, chk_join_keys_full_pk, chk_has_sum_metric, chk_q2_has_rows],
    },
    {
        "id": "Q3_tasks_outflow_feb2026",
        "question": "Сколько задач по фактическому оттоку поставили в феврале 2026 года?",
        # при неоднозначности выбираем вариант с sale_funnel_task (task_subtype);
        # _ambiguity_pick_for ниже выберет по содержимому, индекс — запасной.
        "ambiguity_pick": 0,
        "checks": [chk_q3_correct_interpretation, chk_q3_feb_date, chk_single_row],
    },
    {
        "id": "Q4_outflow_by_segment_from_epk",
        "question": ("Посчитай сумму оттока outflow_qty по дате report_dt и сегменту. "
                     "Сегмент возьми из таблицы uzp_data_epk_consolidation по inn"),
        "ambiguity_pick": 0,
        "checks": [chk_dedup_join, chk_has_sum_metric],
    },
    # --- сложный SQL ---
    {
        "id": "Q5_outflow_by_month",
        "question": "Сумма оттока outflow_qty по месяцам",
        "ambiguity_pick": 0,
        "checks": [chk_month_bucket],
    },
    {
        "id": "Q6_gosb_having",
        "question": "Покажи gosb_id где суммарный отток outflow_qty больше 5000",
        "ambiguity_pick": 0,
        "checks": [chk_having, chk_has_sum_metric],
    },
    {
        "id": "Q7_share_of_total_window",
        "question": "ТОП-5 gosb_id по сумме оттока outflow_qty и доля каждого от общего оттока",
        "ambiguity_pick": 0,
        "checks": [chk_window_or_raw],
    },
]
