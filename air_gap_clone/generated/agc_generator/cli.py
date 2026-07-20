"""CLI программы 2 (generator). Открытый контур.

Пример:
    python -m agc_generator.cli --profile profile.json --scale 0.001 \
        --seed 42 --format csv --out out/

    python -m agc_generator.cli --profile profile.json --format sql --keep-gpdb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agc_common import get_logger
from agc_generator.ddl_builder import build_ddl
from agc_generator.key_linker import generate
from agc_generator.profile_parser import load_profile
from agc_generator.writer import write_csv, write_sql

log = get_logger("generator.cli")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agc_generator", description="Generator (открытый контур)")
    p.add_argument("--profile", required=True, help="profile.json от profiler")
    p.add_argument("--scale", type=float, default=0.001,
                   help="scale_factor: доля от исходного числа строк (1.2M * 0.001 ~ 1200)")
    p.add_argument("--seed", type=int, default=42, help="seed для детерминизма")
    p.add_argument("--format", choices=("csv", "sql"), default="csv",
                   help="csv (файл на таблицу) или sql (батч INSERT-ов)")
    p.add_argument("--out", default="out", help="каталог/файл вывода")
    p.add_argument("--keep-gpdb", action="store_true",
                   help="сохранить GPDB-специфику (DISTRIBUTED BY, заметки о партициях)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    profile = load_profile(args.profile)
    log.info("Профиль загружен: %d таблиц. scale=%s seed=%s format=%s",
             len(profile.tables), args.scale, args.seed, args.format)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) DDL.
    ddl = build_ddl(profile, keep_gpdb=args.keep_gpdb)
    ddl_path = outdir / "schema.sql"
    ddl_path.write_text(ddl, encoding="utf-8")
    log.info("DDL записан: %s", ddl_path)

    # 2) Данные.
    data = generate(profile, args.scale, args.seed)
    if args.format == "csv":
        write_csv(profile, data, outdir / "data")
    else:
        write_sql(profile, data, outdir / "data.sql")

    total = sum(len(v) for v in data.values())
    log.info("Готово: %d таблиц, %d строк суммарно в %s", len(data), total, outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
