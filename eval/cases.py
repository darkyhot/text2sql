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
def chk_values_9_57(turn, db):
    vals = {str(v) for v in _values(turn)}
    ok = {"9", "57"}.issubset(vals)
    return ok, f"values={vals}"


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


def chk_q2_rowcount(turn, db):
    rc = (turn.result or {}).get("rowcount")
    return rc == 92, f"rowcount={rc} (эталон 92)"


def chk_ambiguity_or_valid_table(turn, db):
    """Q3: либо показана неоднозначность с обеими таблицами, либо выбрана одна из валидных."""
    valid = {f"{S}.uzp_dwh_fact_outflow", f"{S}.uzp_dwh_sale_funnel_task"}
    chosen = set(turn.state.get("chosen_tables", []))
    surfaced = turn.state.get("ambiguity_surfaced", False)
    ok = bool(chosen & valid) or surfaced
    return ok, f"chosen={[c.split('.')[-1] for c in chosen]} surfaced={surfaced}"


CASES = [
    {
        "id": "Q1_count_tb_gosb",
        "question": "Сколько всего есть тб и госб?",
        "ambiguity_pick": 0,
        "checks": [chk_values_9_57, chk_single_row],
    },
    {
        "id": "Q2_outflow_by_date_gosb",
        "question": "Посчитай сумму оттоков по дате и названию ГОСБ",
        "ambiguity_pick": 0,
        "checks": [chk_join_no_fanout, chk_join_keys_full_pk, chk_has_sum_metric, chk_q2_rowcount],
    },
    {
        "id": "Q3_tasks_outflow_feb2026",
        "question": "Сколько задач по фактическому оттоку поставили в феврале 2026 года?",
        "ambiguity_pick": 0,
        "checks": [chk_ambiguity_or_valid_table],
    },
]
