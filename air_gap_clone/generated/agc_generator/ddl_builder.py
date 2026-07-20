"""DDL-билдер: CREATE TABLE из профиля (не завися от pg_dump).

Воспроизводим типы, PK/UNIQUE/NOT NULL/FK/CHECK, DEFAULT. По умолчанию тестовая
БД — всё heap, одиночный сегмент (DISTRIBUTED BY / партиции опускаем). Флаг
keep_gpdb сохраняет GPDB-специфику (DISTRIBUTED BY, комментарий о партициях).
Внешние таблицы (gpfdist/PXF) → обычные таблицы-заглушки.

Таблицы упорядочиваем топологически по FK, чтобы родители создавались раньше детей.
FK на таблицы вне профиля опускаем (не на что ссылаться), но значения всё равно
резолвим из синтетического пула (см. key_linker).
"""
from __future__ import annotations

from agc_common import get_logger, quote_ident
from agc_generator.profile_parser import Profile, Table

log = get_logger("generator.ddl")


def topo_sort(tables: list[Table]) -> list[Table]:
    """Порядок создания: родитель раньше ребёнка. Циклы разрываем стабильно."""
    fqns = {t.fqn for t in tables}
    by_fqn = {t.fqn: t for t in tables}
    deps: dict[str, set] = {t.fqn: set() for t in tables}
    for t in tables:
        for fk in t.fks:
            ref = f"{fk.get('ref_schema')}.{fk.get('ref_table')}"
            if ref in fqns and ref != t.fqn:
                deps[t.fqn].add(ref)
    ordered, visited, temp = [], set(), set()

    def visit(fqn: str):
        if fqn in visited:
            return
        if fqn in temp:  # цикл — не углубляемся дальше
            return
        temp.add(fqn)
        for d in sorted(deps[fqn]):
            visit(d)
        temp.discard(fqn)
        visited.add(fqn)
        ordered.append(by_fqn[fqn])

    for fqn in sorted(by_fqn):
        visit(fqn)
    return ordered


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


def build_table_ddl(table: Table, profile_fqns: set, keep_gpdb: bool) -> str:
    lines = [_column_ddl(table, name) for name in table.columns]

    if table.pk:
        cols = ", ".join(quote_ident(c) for c in table.pk)
        lines.append(f"  PRIMARY KEY ({cols})")
    for uni in table.constraints.get("uniques") or []:
        cols = ", ".join(quote_ident(c) for c in uni)
        lines.append(f"  UNIQUE ({cols})")
    for fk in table.fks:
        ref = f"{fk.get('ref_schema')}.{fk.get('ref_table')}"
        if ref not in profile_fqns:
            log.info("FK %s.%s -> %s опущен (цель вне профиля)", table.schema, table.table, ref)
            continue
        cols = ", ".join(quote_ident(c) for c in fk.get("columns") or [])
        ref_cols = ", ".join(quote_ident(c) for c in fk.get("ref_columns") or [])
        lines.append(
            f"  FOREIGN KEY ({cols}) REFERENCES "
            f"{quote_ident(fk['ref_schema'])}.{quote_ident(fk['ref_table'])} ({ref_cols})"
        )
    for check in table.constraints.get("checks") or []:
        lines.append(f"  {check}")

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
    tables = [t for t in profile.tables if not t.is_view]
    views = [t for t in profile.tables if t.is_view]
    if views:
        log.info("Вью в профиле (%d) материализуем как обычные таблицы-заглушки: %s",
                 len(views), ", ".join(v.fqn for v in views))
    all_tables = tables + views  # вью тоже как таблицы
    profile_fqns = {t.fqn for t in all_tables}
    ordered = topo_sort(all_tables)

    schemas = sorted({t.schema for t in ordered})
    header = [
        "-- Автосгенерированный DDL тестовой БД (air-gap clone).",
        "-- Реальные данные не воспроизводятся; структура — из profile.json.",
        "",
    ]
    for s in schemas:
        header.append(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(s)};")
    header.append("")
    body = [build_table_ddl(t, profile_fqns, keep_gpdb) for t in ordered]
    return "\n".join(header) + "\n\n".join(body) + "\n"
