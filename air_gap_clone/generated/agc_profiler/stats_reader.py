"""Чтение статистики из pg_stats — это чтение каталога, а НЕ скан данных.

Один запрос на схему по всем нужным таблицам. Читаем:
null_frac, n_distinct, avg_width, most_common_vals, most_common_freqs, histogram_bounds.

Про n_distinct: положительное = абсолютное число distinct; отрицательное = доля
от числа строк. Знак сохраняем как есть — относительная форма удобно масштабируется
генератором.

Если по таблице pg_stats пуст/устарел — вызывающий код логирует предупреждение и
может точечно досчитать null_frac/n_distinct ТОЛЬКО по нужным колонкам через
лёгкие агрегаты (recompute_missing). По всем колонкам подряд так не делаем.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from agc_common import get_logger, quote_ident, validate_identifier

log = get_logger("profiler.stats")


def read_pg_stats(engine: Engine, schema: str, tables: list[str]) -> dict[str, dict[str, dict]]:
    """{table: {column: {null_frac, n_distinct, avg_width, mcv, mcf, histogram}}}.

    most_common_vals / histogram_bounds приводим к тексту (::text) и парсим как
    литерал массива — иначе anyarray не десериализуется драйвером надёжно.
    """
    validate_identifier(schema, "schema")
    for t in tables:
        validate_identifier(t, "table")
    sql = text(
        "SELECT tablename, attname, null_frac, n_distinct, avg_width, "
        "       most_common_vals::text AS mcv, most_common_freqs, "
        "       histogram_bounds::text AS histogram "
        "FROM pg_stats WHERE schemaname = :s AND tablename = ANY(:tables)"
    )
    out: dict[str, dict[str, dict]] = {t: {} for t in tables}
    with engine.connect() as conn:
        for row in conn.execute(sql, {"s": schema, "tables": list(tables)}).mappings():
            out.setdefault(row["tablename"], {})[row["attname"]] = {
                "null_frac": float(row["null_frac"] or 0.0),
                "n_distinct": float(row["n_distinct"]) if row["n_distinct"] is not None else None,
                "avg_width": int(row["avg_width"]) if row["avg_width"] is not None else None,
                "mcv": parse_pg_array(row["mcv"]),
                "mcf": [float(x) for x in (row["most_common_freqs"] or [])],
                "histogram": parse_pg_array(row["histogram"]),
            }
    return out


def recompute_missing(engine: Engine, schema: str, table: str, columns: list[str]) -> dict[str, dict]:
    """Точечный досчёт null_frac и n_distinct для указанных колонок через лёгкие
    агрегаты. Внимание: это СКАН таблицы — вызывать только для колонок без
    pg_stats и только по явному флагу. Логируем предупреждение."""
    validate_identifier(schema, "schema")
    validate_identifier(table, "table")
    if not columns:
        return {}
    log.warning("pg_stats пуст по %s.%s — досчитываю %d колонок агрегатами (скан!): %s",
                schema, table, len(columns), ", ".join(columns))
    parts = ["COUNT(*) AS _total"]
    for i, col in enumerate(columns):
        validate_identifier(col, "column")
        q = quote_ident(col)
        parts.append(f"COUNT(DISTINCT {q}) AS d{i}")
        parts.append(f"COUNT(*) FILTER (WHERE {q} IS NULL) AS nn{i}")
    sql = f'SELECT {", ".join(parts)} FROM "{schema}"."{table}"'
    with engine.connect() as conn:
        row = conn.execute(text(sql)).mappings().first()
    total = int(row["_total"]) or 1
    out: dict[str, dict] = {}
    for i, col in enumerate(columns):
        distinct = int(row[f"d{i}"])
        nulls = int(row[f"nn{i}"])
        out[col] = {
            "null_frac": nulls / total,
            "n_distinct": float(distinct),  # абсолютное; знак положительный
            "avg_width": None, "mcv": [], "mcf": [], "histogram": [],
        }
    return out


def parse_pg_array(literal: str | None) -> list[str]:
    """Парсер текстового литерала PG-массива: {a,b,"c,d",NULL} -> ['a','b','c,d', None].

    Достаточен для most_common_vals/histogram_bounds (строки/числа/даты).
    NULL внутри массива -> None.
    """
    if not literal:
        return []
    s = literal.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return []
    s = s[1:-1]
    out: list = []
    i, n = 0, len(s)
    while i < n:
        if s[i] == '"':
            i += 1
            buf = []
            while i < n:
                c = s[i]
                if c == "\\" and i + 1 < n:
                    buf.append(s[i + 1]); i += 2; continue
                if c == '"':
                    i += 1; break
                buf.append(c); i += 1
            out.append("".join(buf))
            if i < n and s[i] == ",":
                i += 1
        else:
            j = s.find(",", i)
            if j == -1:
                j = n
            token = s[i:j].strip()
            out.append(None if token == "NULL" else token)
            i = j + 1
    return out
