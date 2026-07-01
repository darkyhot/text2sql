"""Адаптер БД на SQLAlchemy с диалект-границей и предохранителями.

Подключение — как на проде (core/database.py): create_engine("postgresql://...")
с драйвером psycopg2, statement_timeout и read-only через connect_args. Работает
и с тестовым Postgres, и с прод-Greenplum (один протокол).

Доступ агента — только read-only. Пробы агента: guard (SELECT/EXPLAIN), авто-LIMIT,
cost-потолок. Финальная выгрузка (run_export): полный результат без probe-LIMIT и
без cost-потолка, ограничена export_max_rows + statement_timeout.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ..config import DB, DBConfig
from .connection import ConnectionConfig, load_connection

logger = logging.getLogger(__name__)

# Адаптивный сэмплинг метаданных (перенос из боевого core/database.py).
# Для небольших объектов (≤ порога) — точная случайная выборка ORDER BY random();
# для больших — бессортировочный SELECT ... WHERE random() < p LIMIT n, чтобы не
# упираться в statement_timeout на полной сортировке огромной таблицы/вью.
SAMPLE_SORT_THRESHOLD = 2_000_000
SAMPLE_OVERSAMPLE = 2.0
SAMPLE_ASSUMED_ROWS_UNKNOWN = 10_000_000

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_ALLOWED_START = re.compile(r"^\s*(select|with|explain)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|"
    r"copy|merge|call|do|vacuum|reindex|comment|set|begin|commit|rollback)\b",
    re.IGNORECASE,
)


def _validate_identifier(name: str, kind: str = "identifier") -> str:
    if not _IDENTIFIER_RE.match(name or ""):
        raise ValueError(f"Недопустимый {kind}: '{name}'.")
    return name


def build_metadata_sample_sql(schema: str, table: str, n: int, est_rows: int | None) -> str:
    """SQL адаптивного сэмпла по оценке числа строк est_rows (перенос с прода).

    - 0 < est ≤ порога → ORDER BY random() LIMIT :n (точная выборка, дёшево на малых);
    - est > порога     → WHERE random() < p LIMIT :n, p=min(1, oversample*n/est) (без сортировки);
    - est ≤ 0 / None    → считаем объект большим (assumed rows), берём p-стратегию.
    """
    _validate_identifier(schema, "schema")
    _validate_identifier(table, "table")
    base = f'SELECT * FROM "{schema}"."{table}"'
    if est_rows is not None and 0 < est_rows <= SAMPLE_SORT_THRESHOLD:
        return base + " ORDER BY random() LIMIT :n"
    denom = est_rows if (est_rows and est_rows > 0) else SAMPLE_ASSUMED_ROWS_UNKNOWN
    p = min(1.0, SAMPLE_OVERSAMPLE * float(n) / float(denom))
    return base + f" WHERE random() < {p:.10g} LIMIT :n"


KERBEROS_MESSAGE = (
    "Ошибка Kerberos: тикет истёк или недоступен (GSSAPI). "
    "Перевыпустите kerberos-тикет (kinit) и повторите запрос."
)
_KERBEROS_MARKERS = (
    "gssapi", "kerberos", "ticket expired", "gss failure", "gss encryption",
    "credentials cache", "no credentials", "server_lost",
)


def is_kerberos_auth_error(exc: BaseException | None) -> bool:
    """True, если в цепочке исключений есть признак истёкшего/отсутствующего
    kerberos-тикета (GSSAPI). Проверяет __cause__/__context__ рекурсивно."""
    seen: set[int] = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if any(m in str(cur).lower() for m in _KERBEROS_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


class GuardError(Exception):
    """Запрос отклонён предохранителем (не read-only, дорогой, многооператорный)."""


class NotConfiguredError(Exception):
    """Нет данных для подключения к БД — нужен /config_db_conn."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    truncated: bool = False
    cost: float | None = None

    @property
    def rowcount(self) -> int:
        return len(self.rows)


class DbAdapter:
    """Адаптер на SQLAlchemy. Диалект-специфику переопределяют наследники."""

    def __init__(self, cfg: DBConfig | None = None, *, tracer: Callable[[dict], None] | None = None):
        self.cfg = cfg or DB
        self._tracer = tracer
        self._engine: Engine | None = None
        self._conn_cfg: ConnectionConfig | None = load_connection()

    # --- состояние подключения ---
    @property
    def is_configured(self) -> bool:
        """Есть ли куда подключаться: либо config_db.json, либо .env DB_DSN."""
        if self._conn_cfg and self._conn_cfg.is_complete():
            return True
        return bool(self.cfg.dsn)

    @property
    def dialect(self) -> str:
        if self._conn_cfg and self._conn_cfg.is_complete():
            return self._conn_cfg.dialect
        return self.cfg.dialect

    def connection_summary(self) -> str:
        if self._conn_cfg and self._conn_cfg.is_complete():
            return self._conn_cfg.summary()
        return f"{self.cfg.dsn} [{self.cfg.dialect}]" if self.cfg.dsn else "не настроено"

    def reload(self) -> None:
        """Перечитать config_db.json и пересоздать engine (после /config_db_conn)."""
        if self._engine is not None:
            self._engine.dispose()
        self._engine = None
        self._conn_cfg = load_connection()

    def _url(self) -> str:
        if self._conn_cfg and self._conn_cfg.is_complete():
            return self._conn_cfg.url()
        if self.cfg.dsn:
            return self.cfg.dsn
        raise NotConfiguredError("Не настроен коннект к БД, выполните /config_db_conn")

    def get_engine(self) -> Engine:
        if self._engine is None:
            opts = (
                f"-c statement_timeout={int(self.cfg.export_timeout_ms)} "
                f"-c default_transaction_read_only=on"
            )
            self._engine = create_engine(
                self._url(), pool_pre_ping=True, connect_args={"options": opts}
            )
        return self._engine

    def test_connection(self) -> None:
        with self.get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))

    # --- диалект-специфика (переопределяется для Greenplum) ---
    def quote_ident(self, name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def qualified(self, schema: str, table: str) -> str:
        return f"{self.quote_ident(schema)}.{self.quote_ident(table)}"

    def ilike_op(self) -> str:
        return "ILIKE"

    def sample_clause(self, limit: int) -> str:
        return f"ORDER BY random() LIMIT {int(limit)}"

    # --- guard ---
    def _validate(self, sql: str) -> None:
        stripped = sql.strip().rstrip(";")
        if ";" in stripped:
            raise GuardError("Несколько операторов в одном запросе запрещены.")
        if not _ALLOWED_START.match(stripped):
            raise GuardError("Разрешены только SELECT / WITH / EXPLAIN.")
        if _FORBIDDEN.search(stripped):
            raise GuardError("Обнаружено запрещённое ключевое слово (DML/DDL/SET).")

    def _rows(self, conn, sql: str) -> tuple[list[str], list[dict]]:
        res = conn.execute(text(sql))
        cols = list(res.keys())
        rows = [dict(m) for m in res.mappings()]
        return cols, rows

    # --- read-only операции ---
    def explain_cost(self, sql: str) -> float:
        self._validate(sql)
        with self.get_engine().connect() as conn:
            raw = conn.execute(text(f"EXPLAIN (FORMAT JSON) {sql}")).scalar()
        data = json.loads(raw) if isinstance(raw, str) else raw
        return float(data[0]["Plan"]["Total Cost"])

    def run_select(self, sql: str, *, limit: int | None = None, enforce_cost: bool = True) -> QueryResult:
        """Проба агента: авто-LIMIT (probe_row_limit) + cost-потолок."""
        self._validate(sql)
        is_explain = sql.strip().lower().startswith("explain")
        cost: float | None = None
        if enforce_cost and not is_explain:
            cost = self.explain_cost(sql)
            if cost > self.cfg.probe_max_cost:
                raise GuardError(
                    f"Запрос слишком дорогой (EXPLAIN cost={cost:.0f} > "
                    f"потолок {self.cfg.probe_max_cost:.0f}). Сузьте выборку."
                )
        eff_limit = self.cfg.probe_row_limit if limit is None else min(int(limit), self.cfg.probe_row_limit)
        wrapped = sql if is_explain else self._apply_limit(sql, eff_limit)
        with self.get_engine().connect() as conn:
            columns, rows = self._rows(conn, wrapped)
        truncated = (not is_explain) and len(rows) >= eff_limit
        result = QueryResult(columns=columns, rows=rows, truncated=truncated, cost=cost)
        if self._tracer:
            self._tracer({"kind": "probe", "sql": sql, "cost": cost,
                          "rowcount": result.rowcount, "truncated": truncated})
        return result

    def run_export(self, sql: str) -> QueryResult:
        """Финальная выгрузка: полный результат (без probe-LIMIT, без cost-потолка),
        ограничен export_max_rows + statement_timeout."""
        self._validate(sql)
        cap = int(self.cfg.export_max_rows)
        wrapped = self._apply_limit(sql, cap)
        with self.get_engine().connect() as conn:
            columns, rows = self._rows(conn, wrapped)
        truncated = len(rows) >= cap
        result = QueryResult(columns=columns, rows=rows, truncated=truncated)
        if self._tracer:
            self._tracer({"kind": "export", "sql": sql, "rowcount": result.rowcount, "truncated": truncated})
        return result

    @staticmethod
    def _apply_limit(sql: str, limit: int) -> str:
        if re.search(r"\blimit\b\s+\d+\s*$", sql.strip(), re.IGNORECASE):
            return sql
        return f"SELECT * FROM (\n{sql}\n) AS _t LIMIT {int(limit)}"

    def estimate_row_count(self, schema: str, table: str) -> int | None:
        """Быстрая оценка числа строк через планировщик (EXPLAIN, без выполнения).
        Работает и для вью (планировщик раскрывает их). None — оценить не удалось."""
        _validate_identifier(schema, "schema")
        _validate_identifier(table, "table")
        sql = f'EXPLAIN (FORMAT JSON) SELECT * FROM "{schema}"."{table}"'
        try:
            with self.get_engine().connect() as conn:
                raw = conn.execute(text(sql)).scalar()
            data = json.loads(raw) if isinstance(raw, str) else raw
            rows = int(data[0]["Plan"]["Plan Rows"])
            return rows if rows > 0 else None
        except Exception as exc:  # noqa: BLE001
            logger.info("estimate_row_count(%s.%s) не удалась: %s", schema, table, exc)
            return None

    def metadata_sample(self, schema: str, table: str, n: int = 100_000,
                        *, timeout_ms: int | None = None) -> QueryResult:
        """Адаптивный сэмпл для метаданных, безопасный для больших таблиц.
        timeout_ms ограничивает время на ОДНУ таблицу (по умолчанию из export_timeout)."""
        est = self.estimate_row_count(schema, table)
        sql = build_metadata_sample_sql(schema, table, n, est)
        strategy = "sort_random" if "ORDER BY random()" in sql else "random_filter"
        logger.info("metadata_sample %s.%s: strategy=%s est=%s n=%d timeout_ms=%s",
                    schema, table, strategy, est, n, timeout_ms)
        with self.get_engine().connect() as conn:
            if timeout_ms:
                conn.execute(text(f"SET statement_timeout = {int(timeout_ms)}"))
            res = conn.execute(text(sql), {"n": int(n)})
            columns = list(res.keys())
            rows = [dict(m) for m in res.mappings()]
        return QueryResult(columns=columns, rows=rows)

    def table_exists(self, schema: str, table: str) -> bool:
        sql = (
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = :schema AND table_name = :table LIMIT 1"
        )
        with self.get_engine().connect() as conn:
            return conn.execute(text(sql), {"schema": schema, "table": table}).scalar() is not None

    def introspect_columns(self, schema: str, table: str) -> list[dict[str, Any]]:
        sql = (
            "SELECT column_name, data_type, is_nullable, ordinal_position "
            "FROM information_schema.columns WHERE table_schema = :schema AND table_name = :table "
            "ORDER BY ordinal_position"
        )
        with self.get_engine().connect() as conn:
            res = conn.execute(text(sql), {"schema": schema, "table": table})
            return [dict(m) for m in res.mappings()]

    def read_comments(self, schema: str, table: str) -> tuple[str, dict[str, str]]:
        """Комментарии таблицы и колонок из pg-каталога (pg_description). Читается
        из каталога напрямую — видно даже без GRANT на сам объект (нужно для
        view-схемы с описаниями на проде). Возвращает (comment_таблицы, {колонка: comment})."""
        tbl_sql = (
            "SELECT obj_description(c.oid) AS d FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = :schema AND c.relname = :table"
        )
        col_sql = (
            "SELECT a.attname AS name, d.description AS d FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped "
            "LEFT JOIN pg_description d ON d.objoid = c.oid AND d.objsubid = a.attnum "
            "WHERE n.nspname = :schema AND c.relname = :table"
        )
        params = {"schema": schema, "table": table}
        try:
            with self.get_engine().connect() as conn:
                tc = conn.execute(text(tbl_sql), params).scalar()
                cols = {r["name"]: (r["d"] or "") for r in conn.execute(text(col_sql), params).mappings()}
        except Exception as exc:  # noqa: BLE001
            logger.info("read_comments(%s.%s) не удалось: %s", schema, table, exc)
            return "", {}
        return (str(tc or "").strip()), {k: str(v or "").strip() for k, v in cols.items()}

    def list_user_tables(self, schema_like: str = "%") -> list[tuple[str, str]]:
        sql = (
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_type='BASE TABLE' AND table_schema LIKE :schema "
            "ORDER BY table_schema, table_name"
        )
        with self.get_engine().connect() as conn:
            res = conn.execute(text(sql), {"schema": schema_like})
            return [(r["table_schema"], r["table_name"]) for r in res.mappings()]


class PostgresAdapter(DbAdapter):
    """Тестовый Postgres. Дефолтная логика подходит."""


class GreenplumAdapter(DbAdapter):
    """Прод Greenplum. PG-совместим; при необходимости переопределить адаптивный
    сэмплинг больших таблиц (ORDER BY random() дорог) — см. core/database.py на проде."""


def make_adapter(cfg: DBConfig | None = None, *, tracer: Callable[[dict], None] | None = None) -> DbAdapter:
    cfg = cfg or DB
    # Диалект берём из config_db.json, если он есть (прод), иначе из .env.
    conn = load_connection()
    dialect = conn.dialect if (conn and conn.is_complete()) else cfg.dialect
    if dialect == "greenplum":
        return GreenplumAdapter(cfg, tracer=tracer)
    return PostgresAdapter(cfg, tracer=tracer)
