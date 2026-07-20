"""Чтение СТРУКТУРЫ из системного каталога — это ground truth, дёшево и без скана.

Читаем: типы/nullability/default, PK/UNIQUE/NOT NULL/FK/CHECK, ключ распределения
Greenplum (gp_distribution_policy), партиционирование, тип хранения (heap /
append-optimized / column-oriented), relkind (таблица/вью).

Версия GPDB заранее неизвестна — GPDB-специфику (gp_distribution_policy,
pg_partitions, pg_appendonly) оборачиваем в try/except и мягко деградируем.
"""
from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.engine import Engine

from agc_common import get_logger, validate_identifier

log = get_logger("profiler.catalog")

_NUMTYPE_RE = re.compile(r"^(?:numeric|decimal)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_CHARLEN_RE = re.compile(r"\(\s*(\d+)\s*\)")

_RELKIND = {
    "r": "table", "v": "view", "m": "matview",
    "f": "foreign_table", "p": "partitioned_table", "t": "toast",
}


def read_table_meta(engine: Engine, schema: str, table: str) -> dict:
    """relkind, оценка reltuples, oid, тип хранения."""
    validate_identifier(schema, "schema")
    validate_identifier(table, "table")
    sql = text(
        "SELECT c.oid AS oid, c.relkind AS relkind, c.reltuples::bigint AS reltuples "
        "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = :s AND c.relname = :t"
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"s": schema, "t": table}).mappings().first()
    if row is None:
        raise LookupError(f"Объект {schema}.{table} не найден в pg_class")
    oid = int(row["oid"])
    reltuples = int(row["reltuples"] or 0)
    meta = {
        "relkind": _RELKIND.get(row["relkind"], row["relkind"]),
        "is_view": row["relkind"] in ("v", "m"),
        "reltuples": reltuples,
        # reltuples всегда лишь оценка (обновляется ANALYZE); 0 — почти наверняка stale.
        "row_count_estimated": True,
        "storage": _read_storage(engine, oid),
        "oid": oid,
    }
    return meta


def _read_storage(engine: Engine, oid: int) -> str:
    """heap / append_optimized_row / append_optimized_column. GPDB-специфика — guarded."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT columnstore FROM pg_appendonly WHERE relid = :oid"),
                {"oid": oid},
            ).mappings().first()
        if row is None:
            return "heap"
        return "append_optimized_column" if row["columnstore"] else "append_optimized_row"
    except Exception:  # noqa: BLE001 — не GPDB / нет pg_appendonly
        return "heap"


def read_columns(engine: Engine, schema: str, table: str) -> list[dict]:
    """Колонки: имя, точный pg_type (format_type), nullability, default, precision/scale."""
    validate_identifier(schema, "schema")
    validate_identifier(table, "table")
    sql = text(
        "SELECT a.attnum AS attnum, a.attname AS name, "
        "       format_type(a.atttypid, a.atttypmod) AS pg_type, "
        "       a.attnotnull AS notnull, "
        "       pg_get_expr(ad.adbin, ad.adrelid) AS default_expr "
        "FROM pg_attribute a "
        "JOIN pg_class c ON c.oid = a.attrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "LEFT JOIN pg_attrdef ad ON ad.adrelid = a.attrelid AND ad.adnum = a.attnum "
        "WHERE n.nspname = :s AND c.relname = :t AND a.attnum > 0 AND NOT a.attisdropped "
        "ORDER BY a.attnum"
    )
    cols: list[dict] = []
    with engine.connect() as conn:
        for row in conn.execute(sql, {"s": schema, "t": table}).mappings():
            pg_type = str(row["pg_type"])
            precision, scale = _parse_numeric(pg_type)
            cols.append({
                "name": str(row["name"]),
                "pg_type": pg_type,
                "nullable": not bool(row["notnull"]),
                "default": row["default_expr"],
                "precision": precision,
                "scale": scale,
                "char_len": _parse_charlen(pg_type),
                "ordinal": int(row["attnum"]),
            })
    return cols


def _parse_numeric(pg_type: str) -> tuple[int | None, int | None]:
    m = _NUMTYPE_RE.match(pg_type)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _parse_charlen(pg_type: str) -> int | None:
    if pg_type.lower().startswith(("character", "varchar", "char")):
        m = _CHARLEN_RE.search(pg_type)
        return int(m.group(1)) if m else None
    return None


# NOTE: PK/FK из DDL не читаем — в целевых таблицах они не объявлены. PK выводим
# гипотезой по сэмплу (analyze.find_pk, как в исходном проекте); FK не выводим
# вовсе (ключи джойнов подбираются позже на синтетике). CHECK/UNIQUE тоже опускаем.


def read_distribution(engine: Engine, schema: str, table: str) -> list[str]:
    """Ключ распределения Greenplum из gp_distribution_policy. [] если не GPDB
    или распределение случайное/реплицированное."""
    validate_identifier(schema, "schema")
    validate_identifier(table, "table")
    # GPDB 6+: колонка distkey (int2vector). GPDB 5: attrnums (smallint[]).
    variants = (
        "SELECT att.attname AS col "
        "FROM gp_distribution_policy p "
        "JOIN pg_class c ON c.oid = p.localoid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN unnest(p.distkey) WITH ORDINALITY k(a, o) ON true "
        "JOIN pg_attribute att ON att.attrelid = c.oid AND att.attnum = k.a "
        "WHERE n.nspname = :s AND c.relname = :t ORDER BY k.o",
        "SELECT att.attname AS col "
        "FROM gp_distribution_policy p "
        "JOIN pg_class c ON c.oid = p.localoid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN unnest(p.attrnums) WITH ORDINALITY k(a, o) ON true "
        "JOIN pg_attribute att ON att.attrelid = c.oid AND att.attnum = k.a "
        "WHERE n.nspname = :s AND c.relname = :t ORDER BY k.o",
    )
    for sql in variants:
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(sql), {"s": schema, "t": table}).scalars().all()
            return [str(r) for r in rows]
        except Exception:  # noqa: BLE001 — пробуем следующий вариант / не GPDB
            continue
    return []


def read_partition_keys(engine: Engine, schema: str, table: str) -> list[str]:
    """Колонки партиционирования. GPDB classic (pg_partition_columns) → PG native
    (pg_partitioned_table). [] если не партиционировано / представление недоступно."""
    validate_identifier(schema, "schema")
    validate_identifier(table, "table")
    # GPDB classic partitioning.
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT columnname FROM pg_partition_columns "
                "WHERE schemaname = :s AND tablename = :t "
                "ORDER BY position_in_partition_key"
            ), {"s": schema, "t": table}).scalars().all()
        if rows:
            return [str(r) for r in rows]
    except Exception:  # noqa: BLE001
        pass
    # PG native declarative partitioning (GPDB 7 / Postgres).
    try:
        with engine.connect() as conn:
            cols = conn.execute(text(
                "SELECT a.attname "
                "FROM pg_partitioned_table pt "
                "JOIN pg_class c ON c.oid = pt.partrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "JOIN unnest(pt.partattrs) WITH ORDINALITY k(a, o) ON true "
                "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = k.a "
                "WHERE n.nspname = :s AND c.relname = :t ORDER BY k.o"
            ), {"s": schema, "t": table}).scalars().all()
        return [str(c) for c in cols]
    except Exception:  # noqa: BLE001
        return []
