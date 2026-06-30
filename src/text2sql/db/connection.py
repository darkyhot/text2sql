"""Конфигурация подключения к БД (прод-контур Greenplum).

Хранится в config_db.json (как боевой config.json). Подключение — SQLAlchemy
по URL postgresql://user@host:port/db (без пароля; Kerberos/GSSAPI на проде).
Если файла нет — БД считается не настроенной, и агент просит /config_db_conn.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import PROJECT_ROOT

CONN_PATH = PROJECT_ROOT / "config_db.json"
_FIELDS = ("user_id", "host", "port", "database", "dialect")


@dataclass
class ConnectionConfig:
    user_id: str = ""
    host: str = ""
    port: int = 5432
    database: str = "prom"
    dialect: str = "greenplum"

    def is_complete(self) -> bool:
        return bool(self.user_id and self.host and self.port and self.database)

    def url(self) -> str:
        # default-драйвер psycopg2 (как на проде: create_engine("postgresql://...")).
        return f"postgresql://{self.user_id}@{self.host}:{self.port}/{self.database}"

    def summary(self) -> str:
        return f"{self.user_id}@{self.host}:{self.port}/{self.database} [{self.dialect}]"


def load_connection(path: Path = CONN_PATH) -> ConnectionConfig | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return ConnectionConfig(**{k: data[k] for k in _FIELDS if k in data})


def save_connection(cfg: ConnectionConfig, path: Path = CONN_PATH) -> None:
    path.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
