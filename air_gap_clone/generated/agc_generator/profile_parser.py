"""Парсер profile.json в удобные структуры для генератора."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Column:
    name: str
    pg_type: str
    policy: str
    raw: dict = field(default_factory=dict)

    def get(self, key, default=None):
        return self.raw.get(key, default)


@dataclass
class Table:
    schema: str
    table: str
    is_view: bool
    storage: str
    distributed_by: list
    partitioned_by: list
    row_count: int
    row_count_estimated: bool
    constraints: dict
    defaults: dict
    not_null: list
    columns: dict  # name -> Column

    @property
    def fqn(self) -> str:
        return f"{self.schema}.{self.table}"

    @property
    def pk(self) -> list:
        return self.constraints.get("pk") or []

    @property
    def fks(self) -> list:
        return self.constraints.get("fks") or []


@dataclass
class Profile:
    version: int
    tables: list  # list[Table]

    def by_fqn(self) -> dict:
        return {t.fqn: t for t in self.tables}


def load_profile(path: str | Path) -> Profile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tables = []
    for t in data.get("tables", []):
        columns = {
            name: Column(name=name, pg_type=c.get("pg_type", "text"),
                         policy=c.get("policy", "sensitive"), raw=c)
            for name, c in (t.get("columns") or {}).items()
        }
        rc = t.get("row_count") or {}
        tables.append(Table(
            schema=t["schema"], table=t["table"],
            is_view=bool(t.get("is_view")), storage=t.get("storage", "heap"),
            distributed_by=t.get("distributed_by") or [],
            partitioned_by=t.get("partitioned_by") or [],
            row_count=int(rc.get("value") or 0),
            row_count_estimated=bool(rc.get("estimated", True)),
            constraints=t.get("constraints") or {},
            defaults=t.get("defaults") or {},
            not_null=t.get("not_null") or [],
            columns=columns,
        ))
    return Profile(version=int(data.get("profile_version", 1)), tables=tables)
