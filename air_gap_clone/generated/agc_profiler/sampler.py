"""Адаптивный сэмплер: тянет случайную выборку строк таблицы в pandas.

Вся статистика профиля считается в pandas НА ЭТОМ СЭМПЛЕ (как в исходном проекте),
а не отдельными GROUP BY к БД. На контуре с большим RAM это дёшево и быстро.

Оценка числа строк est берётся из планировщика (EXPLAIN, без выполнения) или из
reltuples. Стратегия:
  - 0 < est <= порога → ORDER BY random() LIMIT :n  (точная выборка, дёшево на малых);
  - est > порога      → WHERE random() < p LIMIT :n, p=min(1, oversample*n/est) (без сортировки);
  - est <= 0 / None    → считаем объект большим (assumed rows), берём p-стратегию.

ОГОВОРКА (смещение): `WHERE random() < p LIMIT n` обрывает скан по достижении n
строк — это «начало физического скана», а не равномерная выборка. Для GPDB
(партиции, append-optimized, порядок вставки) это даёт смещение по содержимому;
редкие категории в сэмпле могут потеряться (это допустимо по договорённости).
"""
from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from agc_common import get_logger, validate_identifier

log = get_logger("profiler.sampler")

SAMPLE_SORT_THRESHOLD = 2_000_000
SAMPLE_OVERSAMPLE = 2.0
SAMPLE_ASSUMED_ROWS_UNKNOWN = 10_000_000


def estimate_row_count(engine: Engine, schema: str, table: str) -> int | None:
    """Оценка числа строк через планировщик (EXPLAIN FORMAT JSON, без скана).
    Работает и для вью. None — оценить не удалось."""
    validate_identifier(schema, "schema")
    validate_identifier(table, "table")
    sql = f'EXPLAIN (FORMAT JSON) SELECT * FROM "{schema}"."{table}"'
    try:
        with engine.connect() as conn:
            raw = conn.execute(text(sql)).scalar()
        data = json.loads(raw) if isinstance(raw, str) else raw
        rows = int(data[0]["Plan"]["Plan Rows"])
        return rows if rows > 0 else None
    except Exception as exc:  # noqa: BLE001
        log.info("estimate_row_count(%s.%s) не удалась: %s", schema, table, exc)
        return None


def build_sample_sql(schema: str, table: str, n: int, est: int | None) -> str:
    """SQL адаптивного сэмпла `SELECT *` по оценке est (см. оговорку в docstring модуля)."""
    validate_identifier(schema, "schema")
    validate_identifier(table, "table")
    base = f'SELECT * FROM "{schema}"."{table}"'
    if est is not None and 0 < est <= SAMPLE_SORT_THRESHOLD:
        return base + " ORDER BY random() LIMIT :n"
    denom = est if (est and est > 0) else SAMPLE_ASSUMED_ROWS_UNKNOWN
    p = min(1.0, SAMPLE_OVERSAMPLE * float(n) / float(denom))
    return base + f" WHERE random() < {p:.10g} LIMIT :n"


def sample_dataframe(engine: Engine, schema: str, table: str, n: int = 1_000_000,
                     *, est: int | None = None, timeout_ms: int | None = None) -> pd.DataFrame:
    """Случайный сэмпл до n строк таблицы целиком → pandas.DataFrame."""
    if est is None:
        est = estimate_row_count(engine, schema, table)
    sql = build_sample_sql(schema, table, n, est)
    strategy = "sort_random" if "ORDER BY random()" in sql else "random_filter"
    log.info("sample %s.%s: strategy=%s est=%s n=%d", schema, table, strategy, est, n)
    with engine.connect() as conn:
        if timeout_ms:
            conn.execute(text(f"SET statement_timeout = {int(timeout_ms)}"))
        res = conn.execute(text(sql), {"n": int(n)})
        cols = list(res.keys())
        rows = res.fetchall()
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    log.info("sample %s.%s: получено %d строк, %d колонок", schema, table, len(df), len(cols))
    return df
