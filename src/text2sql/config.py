"""Конфигурация из окружения/.env. Без бизнес-логики — только параметры."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _path(env: str, default: str) -> Path:
    raw = os.getenv(env, default)
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass(frozen=True)
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "deepseek")
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    api_key: str = os.getenv("LLM_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "deepseek-v4-flash")
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))


@dataclass(frozen=True)
class DBConfig:
    dialect: str = os.getenv("DB_DIALECT", "postgres")
    dsn: str = os.getenv("DB_DSN", "postgresql://test@localhost:55432/agent_test")
    probe_row_limit: int = int(os.getenv("PROBE_ROW_LIMIT", "1000"))
    probe_timeout_ms: int = int(os.getenv("PROBE_TIMEOUT_MS", "15000"))
    probe_max_cost: float = float(os.getenv("PROBE_MAX_COST", "5000000"))
    # Финальная выгрузка результата: полный результат в CSV (без probe-LIMIT и
    # без cost-потолка), ограничен только этим потолком строк + statement_timeout.
    export_max_rows: int = int(os.getenv("EXPORT_MAX_ROWS", "1000000"))
    export_timeout_ms: int = int(os.getenv("EXPORT_TIMEOUT_MS", "600000"))


@dataclass(frozen=True)
class Paths:
    data_dir: Path = _path("DATA_DIR", "data_for_agent")
    workspace_dir: Path = _path("WORKSPACE_DIR", "workspace")
    trace_dir: Path = _path("TRACE_DIR", "traces")


LLM = LLMConfig()
DB = DBConfig()
PATHS = Paths()
