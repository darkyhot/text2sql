"""Адаптивный сэмплер — ТОЛЬКО для инспекции форматов/длин/доли NULL
чувствительных колонок. НЕ для кардинальности и НЕ для распределения значений
(их берём из pg_stats/каталога — см. profile_builder).

Оценка числа строк est берётся из планировщика (EXPLAIN, без выполнения)
или из reltuples. Стратегия:
  - 0 < est <= порога → ORDER BY random() LIMIT :n  (точная выборка, дёшево на малых);
  - est > порога      → WHERE random() < p LIMIT :n, p=min(1, oversample*n/est) (без сортировки);
  - est <= 0 / None    → считаем объект большим (assumed rows), берём p-стратегию.
"""
from __future__ import annotations

import json

from sqlalchemy import text
from sqlalchemy.engine import Engine

from agc_common import get_logger, quote_ident, validate_identifier

log = get_logger("profiler.sampler")

SAMPLE_SORT_THRESHOLD = 2_000_000
SAMPLE_OVERSAMPLE = 2.0
SAMPLE_ASSUMED_ROWS_UNKNOWN = 10_000_000


def estimate_row_count(engine: Engine, schema: str, table: str) -> int | None:
    """Оценка числа строк через планировщик (EXPLAIN FORMAT JSON, без скана).
    Работает и для вью (планировщик их раскрывает). None — оценить не удалось."""
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


def build_sample_sql(schema: str, table: str, cols_expr: str, n: int, est: int | None) -> str:
    """SQL адаптивного сэмпла по оценке est.

    ВАЖНАЯ ОГОВОРКА (смещение):
    `WHERE random() < p LIMIT n` обрывает скан по достижении n строк, то есть
    выбирает «начало физического скана», а не равномерную выборку по всей таблице.
    Для GPDB (партиции, append-optimized, порядок по времени вставки) это даёт
    смещение по СОДЕРЖИМОМУ. Поэтому из такого сэмпла берём ТОЛЬКО то, что от
    порядка не зависит: долю NULL, длины, форматы/маски. Распределение значений и
    кардинальность — из pg_stats/каталога, где смещения нет.

    Для длинных текстовых колонок в cols_expr передавайте length(col), а не сам
    col — не тащим реальные длинные значения без нужды.
    """
    validate_identifier(schema, "schema")
    validate_identifier(table, "table")
    base = f'SELECT {cols_expr} FROM "{schema}"."{table}"'
    if est is not None and 0 < est <= SAMPLE_SORT_THRESHOLD:
        return base + " ORDER BY random() LIMIT :n"
    denom = est if (est and est > 0) else SAMPLE_ASSUMED_ROWS_UNKNOWN
    p = min(1.0, SAMPLE_OVERSAMPLE * float(n) / float(denom))
    return base + f" WHERE random() < {p:.10g} LIMIT :n"


# Типы, для которых берём length(col) вместо самого значения (не тащим длинный текст).
_LONG_TEXT = ("text", "character varying", "varchar", "character", "char", "bytea", "json", "jsonb")


def _is_long_text(pg_type: str) -> bool:
    t = (pg_type or "").lower()
    return any(t.startswith(p) for p in _LONG_TEXT)


def sample_columns(
    engine: Engine,
    schema: str,
    table: str,
    columns: list[dict],
    n: int = 20_000,
    *,
    est: int | None = None,
    timeout_ms: int | None = None,
) -> dict[str, list]:
    """Возвращает {col: [значения-или-длины]} для указанных колонок.

    columns — список dict с ключами name/pg_type. Для длинного текста берём
    length(col) (число), иначе само значение (для инспекции формата коротких
    полей: телефон/email/счёт). Только порядко-независимые признаки пригодны.
    """
    if not columns:
        return {}
    if est is None:
        est = estimate_row_count(engine, schema, table)
    select_parts, out_names = [], []
    for col in columns:
        name = validate_identifier(col["name"], "column")
        if _is_long_text(col.get("pg_type", "")):
            select_parts.append(f'length({quote_ident(name)}) AS {quote_ident(name)}')
        else:
            select_parts.append(quote_ident(name))
        out_names.append(name)
    sql = build_sample_sql(schema, table, ", ".join(select_parts), n, est)
    strategy = "sort_random" if "ORDER BY random()" in sql else "random_filter"
    log.info("sample %s.%s: strategy=%s est=%s n=%d cols=%d",
             schema, table, strategy, est, n, len(out_names))
    result: dict[str, list] = {name: [] for name in out_names}
    with engine.connect() as conn:
        if timeout_ms:
            conn.execute(text(f"SET statement_timeout = {int(timeout_ms)}"))
        for row in conn.execute(text(sql), {"n": int(n)}).mappings():
            for name in out_names:
                result[name].append(row[name])
    return result
