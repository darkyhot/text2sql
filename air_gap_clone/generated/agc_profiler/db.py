"""Подключение к БД закрытого контура (единственный сетевой вызов инструмента).

САМОСТОЯТЕЛЬНЫЙ модуль — ничего из внешних проектов не импортируем. Параметры
подключения задаются явно (в ячейке ноутбука, через --dsn или db_config.json).

Порядок разрешения DSN:
  1) явный dsn (аргумент / --dsn / ячейка ноутбука);
  2) переменные окружения AGC_DB_DSN, затем DB_DSN;
  3) db_config.json рядом (ключ "dsn" ИЛИ поля host/port/database/user[/password]).

Движок — read-only: default_transaction_read_only=on + statement_timeout,
чтобы профайлер физически не мог ничего записать в реальную БД.

Greenplum на проде часто ходит по Kerberos/GSSAPI (без пароля) — тогда просто не
указывайте password, драйвер возьмёт тикет из окружения.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from agc_common import get_logger

log = get_logger("profiler.db")


def dsn_from_parts(host: str, database: str, *, port: int = 5432,
                   user: str = "", password: str = "", driver: str = "postgresql") -> str:
    """Собрать DSN из компонентов. password пустой → Kerberos/ident (как на проде GPDB)."""
    auth = user
    if user and password:
        auth = f"{user}:{password}"
    at = f"{auth}@" if auth else ""
    return f"{driver}://{at}{host}:{port}/{database}"


def resolve_dsn(dsn: str | None = None, config_path: str | Path | None = None) -> str:
    if dsn:
        return dsn
    for env in ("AGC_DB_DSN", "DB_DSN"):
        if os.getenv(env):
            return os.environ[env]
    if config_path and Path(config_path).exists():
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if data.get("dsn"):
            return str(data["dsn"])
        return dsn_from_parts(
            data["host"], data["database"],
            port=int(data.get("port", 5432)),
            user=data.get("user") or data.get("user_id") or "",
            password=data.get("password", ""),
            driver=data.get("driver", "postgresql"),
        )
    raise RuntimeError(
        "Не задан DSN БД. Укажите dsn/--dsn, переменную AGC_DB_DSN/DB_DSN или db_config.json "
        "(host/port/database/user[/password])."
    )


def make_engine(
    dsn: str | None = None,
    config_path: str | Path | None = None,
    statement_timeout_ms: int = 600_000,
) -> Engine:
    """Read-only SQLAlchemy-движок к Greenplum/Postgres (драйвер psycopg2)."""
    url = resolve_dsn(dsn, config_path)
    safe = url.split("@", 1)[-1] if "@" in url else url  # креды в лог не пишем
    log.info("Подключение к БД (read-only): ...@%s", safe)
    opts = (
        f"-c statement_timeout={int(statement_timeout_ms)} "
        f"-c default_transaction_read_only=on"
    )
    return create_engine(url, pool_pre_ping=True, connect_args={"options": opts})
