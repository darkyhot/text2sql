"""CLI программы 1 (profiler). Закрытый контур.

Пример:
    python -m agc_profiler.cli --tables "public.tasks,public.clients" \
        --policy policy.yaml --out profile.json --sample

    python -m agc_profiler.cli --tables-csv tables.csv --out profile.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from agc_common import get_logger
from agc_profiler.db import make_engine
from agc_profiler.linter import check_profile
from agc_profiler.policy import Policy
from agc_profiler.profile_builder import build_profile

log = get_logger("profiler.cli")


def parse_tables_arg(value: str) -> list[tuple[str, str]]:
    """'schema.table,schema.table2' -> [(schema, table), ...]."""
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "." not in item:
            raise ValueError(f"Ожидался формат schema.table, получено: {item!r}")
        schema, table = item.split(".", 1)
        out.append((schema.strip(), table.strip()))
    return out


def parse_tables_csv(path: str | Path) -> list[tuple[str, str]]:
    """CSV со списком таблиц. Дефолт-колонки: schema,table. Поддерживаем также
    schema_name,table_name (как tables_list.csv проекта)."""
    rows: list[tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        sc = cols.get("schema") or cols.get("schema_name")
        tc = cols.get("table") or cols.get("table_name")
        if not sc or not tc:
            raise ValueError("CSV должен содержать колонки schema,table (или schema_name,table_name)")
        for r in reader:
            schema, table = (r[sc] or "").strip(), (r[tc] or "").strip()
            if schema and table:
                rows.append((schema, table))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agc_profiler", description="Profiler (закрытый контур)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tables", help="schema.table через запятую")
    src.add_argument("--tables-csv", help="CSV со списком таблиц (колонки schema,table)")
    p.add_argument("--policy", help="YAML policy-файл (whitelist чувствительности)")
    p.add_argument("--out", default="profile.json", help="куда писать профиль")
    p.add_argument("--dsn", help="DSN БД (иначе AGC_DB_DSN/DB_DSN/db_config.json)")
    p.add_argument("--db-config", help="db_config.json с параметрами подключения")
    p.add_argument("--sample-n", type=int, default=1_000_000,
                   help="сколько строк тянуть в сэмпл на таблицу (pandas в памяти)")
    p.add_argument("--statement-timeout-ms", type=int, default=600_000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tables = parse_tables_csv(args.tables_csv) if args.tables_csv else parse_tables_arg(args.tables)
    if not tables:
        log.error("Список таблиц пуст."); return 2
    log.info("К обработке %d таблиц: %s", len(tables),
             ", ".join(f"{s}.{t}" for s, t in tables[:10]))

    engine = make_engine(args.dsn, args.db_config, args.statement_timeout_ms)
    policy = Policy.load(args.policy)
    profile = build_profile(engine, tables, policy, sample_n=args.sample_n,
                            timeout_ms=args.statement_timeout_ms)

    # Линтер — строгая проверка перед записью (главная точка аудита утечек).
    check_profile(profile)
    log.info("Линтер пройден: реальные значения только в categorical_keep.")

    Path(args.out).write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Профиль записан: %s (%d таблиц)", args.out, len(profile["tables"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
