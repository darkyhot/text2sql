"""DDL-билдер: CREATE TABLE из профиля (не завися от pg_dump).

Воспроизводим типы, NOT NULL, DEFAULT и PK (гипотеза по сэмплу — удобно агенту,
хотя на реальных данных ключ не объявлен). FK НЕ создаём: связей в профиле нет,
ключи джойнов подбираются позже на синтетике. По умолчанию тестовая БД — heap,
одиночный сегмент; флаг keep_gpdb сохраняет DISTRIBUTED BY и заметку о партициях.
Вью материализуем как обычные таблицы-заглушки.
"""
from __future__ import annotations

from agc_common import get_logger, quote_ident
from agc_generator.profile_parser import Profile, Table

log = get_logger("generator.ddl")


def _column_ddl(table: Table, name: str) -> str:
    col = table.columns[name]
    parts = [f"  {quote_ident(name)} {col.pg_type}"]
    if name in table.not_null and name not in table.pk:
        parts.append("NOT NULL")
    default = table.defaults.get(name)
    # nextval(...) отбрасываем: секвенций в тестовой БД нет, PK генерируем сами.
    if default and "nextval(" not in str(default).lower():
        parts.append(f"DEFAULT {default}")
    return " ".join(parts)


def build_table_ddl(table: Table, keep_gpdb: bool) -> str:
    lines = [_column_ddl(table, name) for name in table.columns]
    if table.pk:
        cols = ", ".join(quote_ident(c) for c in table.pk)
        lines.append(f"  PRIMARY KEY ({cols})  -- гипотеза по сэмплу")

    ddl = (
        f"CREATE TABLE {quote_ident(table.schema)}.{quote_ident(table.table)} (\n"
        + ",\n".join(lines) + "\n)"
    )
    if keep_gpdb and table.distributed_by:
        cols = ", ".join(quote_ident(c) for c in table.distributed_by)
        ddl += f"\nDISTRIBUTED BY ({cols})"
    elif keep_gpdb:
        ddl += "\nDISTRIBUTED RANDOMLY"
    ddl += ";"
    if keep_gpdb and table.partitioned_by:
        ddl += (f"\n-- NOTE: исходная таблица партиционирована по "
                f"{', '.join(table.partitioned_by)}; для теста партиции упрощены.")
    return ddl


def build_ddl(profile: Profile, *, keep_gpdb: bool = False) -> str:
    tables = list(profile.tables)
    views = [t for t in tables if t.is_view]
    if views:
        log.info("Вью в профиле (%d) материализуем как таблицы-заглушки: %s",
                 len(views), ", ".join(v.fqn for v in views))
    schemas = sorted({t.schema for t in tables})
    header = [
        "-- Автосгенерированный DDL тестовой БД (air-gap clone).",
        "-- Реальные данные не воспроизводятся; структура — из profile.json.",
        "-- PK — гипотеза по сэмплу; FK не создаются (подбираются на синтетике).",
        "",
    ]
    for s in schemas:
        header.append(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(s)};")
    header.append("")
    body = [build_table_ddl(t, keep_gpdb) for t in tables]
    return "\n".join(header) + "\n\n".join(body) + "\n"
