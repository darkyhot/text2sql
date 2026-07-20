"""Запись синтетических данных: CSV на таблицу или батч INSERT-ов."""
from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from agc_common import get_logger, quote_ident
from agc_generator.ddl_builder import topo_sort
from agc_generator.profile_parser import Profile

log = get_logger("generator.writer")


def _cell_csv(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (datetime, date)):
        return v.isoformat(sep=" ") if isinstance(v, datetime) else v.isoformat()
    return str(v)


def write_csv(profile: Profile, data: dict[str, list[dict]], outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for table in profile.tables:
        rows = data.get(table.fqn, [])
        path = outdir / f"{table.schema}.{table.table}.csv"
        cols = list(table.columns.keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for row in rows:
                w.writerow([_cell_csv(row.get(c)) for c in cols])
        written.append(path)
        log.info("CSV %s: %d строк", path.name, len(rows))
    return written


def _cell_sql(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, datetime):
        return "'" + v.isoformat(sep=" ") + "'"
    if isinstance(v, date):
        return "'" + v.isoformat() + "'"
    return "'" + str(v).replace("'", "''") + "'"


def write_sql(profile: Profile, data: dict[str, list[dict]], out_path: Path,
              batch: int = 500) -> Path:
    ordered = topo_sort(list(profile.tables))  # родители раньше — FK не нарушаем при вставке
    lines = ["-- Синтетические данные (air-gap clone). INSERT-ы в порядке FK-зависимостей.", ""]
    for table in ordered:
        rows = data.get(table.fqn, [])
        if not rows:
            continue
        cols = list(table.columns.keys())
        collist = ", ".join(quote_ident(c) for c in cols)
        target = f"{quote_ident(table.schema)}.{quote_ident(table.table)}"
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            values = ",\n  ".join(
                "(" + ", ".join(_cell_sql(r.get(c)) for c in cols) + ")" for r in chunk)
            lines.append(f"INSERT INTO {target} ({collist}) VALUES\n  {values};")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("SQL данные: %s", out_path.name)
    return out_path
