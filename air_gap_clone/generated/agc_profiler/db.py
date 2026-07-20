"""Подключение к БД закрытого контура (единственный сетевой вызов инструмента).

Порядок разрешения DSN:
  1) явный аргумент dsn / --dsn;
  2) переменные окружения AGC_DB_DSN, затем DB_DSN;
  3) db_config.json рядом с инструментом (ключ "dsn" или user_id/host/port/database);
  4) подключение проекта text2sql (src/text2sql/db/connection.py), если пакет
     доступен на PYTHONPATH — так инструмент переиспользует уже настроенный
     коннект проекта (config_db.json / .env), как просили.

Движок — read-only: default_transaction_read_only=on + statement_timeout,
чтобы профайлер физически не мог ничего записать в реальную БД.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from agc_common import get_logger

log = get_logger("profiler.db")


def _from_project() -> str | None:
    """DSN из проекта text2sql, если он импортируется (запуск внутри репозитория)."""
    try:
        from text2sql.config import DB  # type: ignore
        from text2sql.db.connection import load_connection  # type: ignore

        conn = load_connection()
        if conn and conn.is_complete():
            return conn.url()
        if DB.dsn:
            return DB.dsn
    except Exception:  # noqa: BLE001 — проект недоступен, это не ошибка
        return None
    return None


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
        user = data.get("user_id") or data.get("user") or ""
        host = data["host"]
        port = data.get("port", 5432)
        database = data["database"]
        return f"postgresql://{user}@{host}:{port}/{database}"
    proj = _from_project()
    if proj:
        return proj
    raise RuntimeError(
        "Не задан DSN БД. Укажите --dsn, переменную AGC_DB_DSN/DB_DSN, "
        "db_config.json или запустите внутри проекта text2sql (config_db.json/.env)."
    )


def make_engine(
    dsn: str | None = None,
    config_path: str | Path | None = None,
    statement_timeout_ms: int = 600_000,
) -> Engine:
    """Read-only SQLAlchemy-движок к Greenplum/Postgres (драйвер psycopg2)."""
    url = resolve_dsn(dsn, config_path)
    # Прячем всё до '@' в логе (host/db показываем, user/креды — нет).
    safe = url.split("@", 1)[-1] if "@" in url else url
    log.info("Подключение к БД (read-only): ...@%s", safe)
    opts = (
        f"-c statement_timeout={int(statement_timeout_ms)} "
        f"-c default_transaction_read_only=on"
    )
    return create_engine(url, pool_pre_ping=True, connect_args={"options": opts})
