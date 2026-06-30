"""Управление целевым списком таблиц и синхронизация каталога метаданных."""

from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sqlalchemy import inspect

from core.enrichment_pipeline import EnrichmentPipeline
from core.exceptions import KERBEROS_USER_MESSAGE, KerberosAuthError, is_kerberos_auth_error
from core.schema_loader import DATA_DIR, SchemaLoader

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGETS_PATH = DATA_DIR / "metadata_targets.yaml"
TABLE_FEW_SHOTS_PATH = DATA_DIR / "table_description_few_shots.yaml"
COLUMN_FEW_SHOTS_PATH = DATA_DIR / "column_description_few_shots.yaml"

SN_UZP_SCHEMA = "s_grnplm_ld_salesntwrk_pcap_sn_uzp"
SN_UZP_VIEW_SCHEMA = "s_grnplm_as_salesntwrk_pcap_sn_view"
SN_T_UZP_SCHEMA = "s_grnplm_ld_salesntwrk_pcap_sn_t_uzp"
ALLOWED_SCHEMAS = {
    SN_UZP_SCHEMA,
    SN_UZP_VIEW_SCHEMA,
    SN_T_UZP_SCHEMA,
}

TARGET_COLUMNS = ["schema_name", "table_name"]
TABLE_COLUMNS = ["schema_name", "table_name", "description", "grain"]
ATTR_COLUMNS = [
    "schema_name", "table_name", "column_name", "dType",
    "is_not_null", "description", "is_primary_key",
    "not_null_perc", "unique_perc",
    "foreign_key_target", "sample_values", "partition_key",
]
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _is_identifier(value: str) -> bool:
    return bool(_IDENTIFIER_RE.match(value or ""))


def _raise_if_kerberos_auth_error(exc: BaseException) -> None:
    if isinstance(exc, KerberosAuthError):
        raise exc
    if is_kerberos_auth_error(exc):
        raise KerberosAuthError(KERBEROS_USER_MESSAGE) from exc


def _normalize_table_refs(refs: list[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in refs:
        item = (raw or "").strip().lower()
        if not item:
            continue
        if "." not in item:
            raise ValueError(
                f"Недопустимый формат таблицы '{raw}'. Используйте schema.table"
            )
        schema, table = item.split(".", 1)
        if not (_is_identifier(schema) and _is_identifier(table)):
            raise ValueError(
                f"Недопустимое имя таблицы '{raw}'. Допустимы только schema.table"
            )
        key = (schema, table)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def parse_table_refs(text: str) -> list[tuple[str, str]]:
    """Разобрать список schema.table, разделённый запятыми или пробелами."""
    chunks = [part.strip() for part in re.split(r"[\s,]+", text or "") if part.strip()]
    return _normalize_table_refs(chunks)


def _humanize_name(name: str) -> str:
    words = [part for part in str(name or "").strip().split("_") if part]
    if not words:
        return ""
    return " ".join(words)


def _read_yaml_examples(path: Path, key: str) -> list[dict[str, str]]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(loaded, dict):
        return []
    items = loaded.get(key) or []
    if not isinstance(items, list):
        return []
    result: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        result.append({str(k): str(v) for k, v in item.items() if v is not None})
    return result


def _format_column_examples_for_prompt(examples: list[dict[str, str]]) -> str:
    if not examples:
        return ""
    lines: list[str] = []
    for idx, item in enumerate(examples, 1):
        lines.append(f"Пример {idx}:")
        lines.append(f"Атрибут: {item.get('column_name', '')}")
        lines.append(f"Описание: {item.get('description', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_table_examples_for_prompt(examples: list[dict[str, str]]) -> str:
    if not examples:
        return ""
    lines: list[str] = []
    for idx, item in enumerate(examples, 1):
        lines.append(f"Пример {idx}:")
        lines.append(f"Таблица: {item.get('table_name', '')}")
        lines.append(f"Описание: {item.get('description', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def _normalize_name(value: str) -> str:
    return str(value or "").strip().lower()


_METRIC_NAME_PATTERNS = (
    re.compile(r"(^|_)(qty|quantity|amt|amount|sum|total|cnt|count|avg|rate|ratio|pct|perc|percent|value|val)($|_)", re.I),
    re.compile(r"_(qty|amt|cnt|sum|avg|pct|perc|amount|value|val)$", re.I),
)


def _is_metric_like_column(column_name: str) -> bool:
    name = str(column_name or "").strip().lower()
    if not name:
        return False
    return any(pattern.search(name) for pattern in _METRIC_NAME_PATTERNS)


def _find_candidate_primary_key(
    df: pd.DataFrame,
    max_columns: int = 5,
    progress_callback: Any | None = None,
) -> list[str]:
    """Найти минимальную уникальную комбинацию колонок на sample.

    Сначала пробуем комбинации без явных measure/fact-полей (`qty`, `amt`, `cnt` и т.п.),
    и только потом подключаем их как fallback. Это уменьшает шанс пометить
    составной ключ через метрики вместо бизнес-идентификаторов.
    """
    if df.empty:
        return []

    cols = [col for col in df.columns if df[col].nunique(dropna=False) > 1]
    cols = [col for col in cols if not df[col].isnull().any()]
    if not cols:
        return []

    preferred_cols = [col for col in cols if not _is_metric_like_column(col)]
    deferred_cols = [col for col in cols if _is_metric_like_column(col)]

    search_groups: list[list[str]] = []
    if preferred_cols:
        search_groups.append(preferred_cols)
    if deferred_cols:
        search_groups.append(preferred_cols + deferred_cols)

    total_combinations = 0
    for candidates in search_groups:
        max_columns_local = min(max_columns, len(candidates))
        total_combinations += sum(math.comb(len(candidates), size) for size in range(1, max_columns_local + 1))

    checked = 0

    def _emit_progress() -> None:
        if progress_callback is None or total_combinations <= 0:
            return
        if checked == 1 or checked == total_combinations or checked % 100 == 0:
            try:
                progress_callback(checked, total_combinations)
            except Exception:  # noqa: BLE001
                logger.debug("Primary key progress callback failed")

    for candidates in search_groups:
        max_columns_local = min(max_columns, len(candidates))
        for size in range(1, max_columns_local + 1):
            for combo in combinations(candidates, size):
                checked += 1
                _emit_progress()
                if not df.duplicated(subset=list(combo)).any():
                    if progress_callback is not None and checked != total_combinations:
                        try:
                            progress_callback(checked, total_combinations)
                        except Exception:  # noqa: BLE001
                            logger.debug("Primary key progress callback failed on finish")
                    return list(combo)
    return []


class MetadataRefreshService:
    """Сервис управления manifest-списком таблиц и каталогом метаданных."""

    def __init__(
        self,
        schema_loader: SchemaLoader,
        db_manager: Any,
        llm: Any | None = None,
        *,
        targets_path: Path | None = None,
        table_few_shots_path: Path | None = None,
        column_few_shots_path: Path | None = None,
        sample_limit: int = 100_000,
    ) -> None:
        self.schema_loader = schema_loader
        self.db = db_manager
        self.llm = llm
        default_data_dir = getattr(self.schema_loader, "_data_dir", DATA_DIR)
        self.targets_path = targets_path or TARGETS_PATH
        self.table_few_shots_path = table_few_shots_path or (default_data_dir / TABLE_FEW_SHOTS_PATH.name)
        self.column_few_shots_path = column_few_shots_path or (default_data_dir / COLUMN_FEW_SHOTS_PATH.name)
        self.sample_limit = int(sample_limit)
        self.targets_path.parent.mkdir(parents=True, exist_ok=True)
        self.table_few_shots_path.parent.mkdir(parents=True, exist_ok=True)
        self.column_few_shots_path.parent.mkdir(parents=True, exist_ok=True)

    def _bootstrap_targets_from_catalog(self) -> pd.DataFrame:
        tables_df = self.schema_loader.tables_df
        if tables_df.empty:
            return pd.DataFrame(columns=TARGET_COLUMNS)
        result = (
            tables_df[["schema_name", "table_name"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["schema_name", "table_name"])
            .reset_index(drop=True)
        )
        return result

    @staticmethod
    def _normalize_targets_df(df: pd.DataFrame) -> pd.DataFrame:
        for col in TARGET_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return (
            df[TARGET_COLUMNS]
            .fillna("")
            .astype(str)
            .drop_duplicates()
            .sort_values(TARGET_COLUMNS)
            .reset_index(drop=True)
        )

    def load_targets_df(self) -> pd.DataFrame:
        """Загрузить manifest таблиц или создать его из текущего каталога."""
        if self.targets_path.exists():
            try:
                loaded = yaml.safe_load(self.targets_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                loaded = None
            if isinstance(loaded, dict):
                items = loaded.get("tables") or []
            else:
                items = []
            rows: list[dict[str, str]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                rows.append({
                    "schema_name": str(item.get("schema_name", "") or ""),
                    "table_name": str(item.get("table_name", "") or ""),
                })
            df = pd.DataFrame(rows, columns=TARGET_COLUMNS)
            return self._normalize_targets_df(df)

        df = self._bootstrap_targets_from_catalog()
        self.save_targets_df(df)
        return df

    def save_targets_df(self, df: pd.DataFrame) -> None:
        """Сохранить manifest таблиц."""
        persisted = self._normalize_targets_df(df)
        self.targets_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tables": [
                {
                    "schema_name": str(row.schema_name),
                    "table_name": str(row.table_name),
                }
                for row in persisted.itertuples(index=False)
            ]
        }
        self.targets_path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def list_targets(self) -> list[str]:
        """Вернуть текущий список таблиц из manifest."""
        df = self.load_targets_df()
        return [f"{row.schema_name}.{row.table_name}" for row in df.itertuples()]

    def _table_exists(self, schema: str, table: str) -> bool:
        return bool(self.db.table_exists(schema, table))

    @staticmethod
    def _is_allowed_schema(schema: str) -> bool:
        return schema in ALLOWED_SCHEMAS

    def _get_sample_df(self, schema: str, table: str) -> pd.DataFrame:
        # Адаптивный сэмпл: на больших таблицах/вью ORDER BY random() требует
        # полной сортировки и упирается в statement_timeout — get_metadata_sample
        # выбирает бессортировочную стратегию по EXPLAIN-оценке размера.
        if hasattr(self.db, "get_metadata_sample"):
            return self.db.get_metadata_sample(schema, table, n=self.sample_limit)
        if hasattr(self.db, "get_random_sample"):
            return self.db.get_random_sample(schema, table, n=self.sample_limit)
        sql = f'SELECT * FROM "{schema}"."{table}" ORDER BY random() LIMIT {self.sample_limit}'
        return self.db.execute_query(sql, limit=self.sample_limit)

    def _get_comment_bundle(
        self,
        inspector: Any,
        schema: str,
        table: str,
    ) -> tuple[str, dict[str, str]]:
        """Получить комментарии таблицы и колонок с учётом policy по схеме."""
        table_comment = ""
        column_comments: dict[str, str] = {}

        def _read_comments(src_schema: str) -> tuple[str, dict[str, str]]:
            try:
                src_columns = inspector.get_columns(table, schema=src_schema)
            except Exception as exc:  # noqa: BLE001
                _raise_if_kerberos_auth_error(exc)
                # diagnostic: раньше ошибка рефлексии глоталась молча и таблица
                # без видимой причины уходила в humanize-фолбэк.
                logger.warning(
                    "MetadataRefresh: рефлексия колонок упала для %s.%s — комментарии не прочитаны: %s",
                    src_schema, table, exc,
                )
                return ("", {})
            try:
                src_table_comment = inspector.get_table_comment(table, schema=src_schema)
                src_comment = str(src_table_comment.get("text") or "").strip()
            except Exception as exc:  # noqa: BLE001
                _raise_if_kerberos_auth_error(exc)
                logger.warning(
                    "MetadataRefresh: get_table_comment упал для %s.%s: %s",
                    src_schema, table, exc,
                )
                src_comment = ""
            src_columns_map = {
                str(col.get("name") or ""): str(col.get("comment") or "").strip()
                for col in src_columns
            }
            non_empty = sum(1 for v in src_columns_map.values() if v)
            logger.info(
                "MetadataRefresh: _read_comments %s.%s — колонок=%d, непустых комментариев=%d, table_comment=%s",
                src_schema, table, len(src_columns_map), non_empty, bool(src_comment),
            )
            return (src_comment, src_columns_map)

        # Существование вью-источника комментариев проверяем по каталогу
        # (pg_class), а не через information_schema: у агента может не быть GRANT
        # на объекты sn_view, но комментарии читаемы из каталога. Fallback на
        # _table_exists — для стабов/совместимости.
        _catalog_exists = getattr(self.db, "relation_exists_in_catalog", None)
        view_exists = (
            _catalog_exists(SN_UZP_VIEW_SCHEMA, table)
            if callable(_catalog_exists)
            else self._table_exists(SN_UZP_VIEW_SCHEMA, table)
        )
        redirect = schema == SN_UZP_SCHEMA and view_exists
        logger.info(
            "MetadataRefresh: _get_comment_bundle %s.%s — view-redirect=%s (целевая view-схема %s)",
            schema, table, redirect, SN_UZP_VIEW_SCHEMA,
        )
        if redirect:
            table_comment, column_comments = _read_comments(SN_UZP_VIEW_SCHEMA)
            if table_comment or any(column_comments.values()):
                logger.info(
                    "MetadataRefresh: %s.%s — комментарии взяты из view-redirect (%s)",
                    schema, table, SN_UZP_VIEW_SCHEMA,
                )
                return (table_comment, column_comments)
            logger.info(
                "MetadataRefresh: %s.%s — в view-схеме комментариев нет, читаю собственную схему",
                schema, table,
            )

        table_comment, column_comments = _read_comments(schema)
        return (table_comment, column_comments)

    def _build_column_prompt(
        self,
        schema: str,
        table: str,
        columns: list[str],
    ) -> tuple[str, str]:
        examples = _format_column_examples_for_prompt(
            _read_yaml_examples(self.column_few_shots_path, "columns")
        )
        system_prompt = (
            "Ты senior аналитик DWH. Кратко расшифруй имена атрибутов БД.\n"
            "Не добавляй нумерацию, пояснения и знаки препинания.\n"
            "Сохраняй порядок атрибутов.\n"
            "Если видишь аббревиатуру, не пытайся её разворачивать."
        )
        examples_block = f"\nПримеры:\n{examples}\n" if examples else ""
        logger.info(
            "MetadataRefresh: _build_column_prompt %s.%s — блок «Примеры» в промпте=%s",
            schema, table, bool(examples),
        )
        user_prompt = (
            f"Схема: {schema}\n"
            f"Таблица: {table}\n"
            f"{examples_block}"
            "Расшифруй список атрибутов, один на строку:\n"
            + "\n".join(f"- {column}" for column in columns)
        )
        return (system_prompt, user_prompt)

    def _build_table_prompt(
        self,
        schema: str,
        table: str,
        columns_df: pd.DataFrame,
        sample_df: pd.DataFrame,
    ) -> tuple[str, str]:
        examples = _format_table_examples_for_prompt(
            _read_yaml_examples(self.table_few_shots_path, "tables")
        )
        logger.info(
            "MetadataRefresh: _build_table_prompt %s.%s — блок «Примеры» в промпте=%s",
            schema, table, bool(examples),
        )
        attrs = "\n".join(
            f"- {row.column_name}: {row.description or row.column_name}"
            for row in columns_df.itertuples()
        )
        sample_lines: list[str] = []
        if not sample_df.empty:
            for column in sample_df.columns:
                values = [
                    str(value).strip()
                    for value in sample_df[column].dropna().astype(str).head(10).tolist()
                    if str(value).strip()
                ]
                if values:
                    sample_lines.append(f"- {column}: {', '.join(values[:10])}")

        system_prompt = (
            "Ты senior аналитик DWH. Сформируй компактное описание таблицы.\n"
            "Нужны 4 пункта:\n"
            "1. Общее назначение таблицы\n"
            "2. Применение таблицы\n"
            "3. Ограничения и особенности данных\n"
            "4. Ключевые атрибуты\n"
            "Пиши узко и предметно, без воды."
        )
        examples_block = f"\nПримеры:\n{examples}\n" if examples else ""
        sample_block = "\n".join(sample_lines) if sample_lines else "Нет sample-значений"
        user_prompt = (
            f"Схема: {schema}\n"
            f"Таблица: {table}\n"
            f"{examples_block}"
            f"Атрибуты:\n{attrs}\n\n"
            f"Примеры значений:\n{sample_block}"
        )
        return (system_prompt, user_prompt)

    def _generate_column_descriptions(
        self,
        schema: str,
        table: str,
        missing_columns: list[str],
    ) -> dict[str, str]:
        if not missing_columns:
            return {}

        examples = _read_yaml_examples(self.column_few_shots_path, "columns")
        example_map = {
            _normalize_name(item.get("column_name", "")): str(item.get("description", "")).strip()
            for item in examples
            if str(item.get("column_name", "")).strip() and str(item.get("description", "")).strip()
        }
        logger.info(
            "MetadataRefresh: генерация колонок %s.%s — примеров в справочнике=%d (файл %s), без описания=%d",
            schema, table, len(example_map), self.column_few_shots_path.name, len(missing_columns),
        )

        resolved: dict[str, str] = {}
        unresolved: list[str] = []
        for column in missing_columns:
            matched = example_map.get(_normalize_name(column))
            if matched:
                resolved[column] = matched
            else:
                unresolved.append(column)

        logger.info(
            "MetadataRefresh: %s.%s — из справочника разрешено=%d, в LLM уйдёт=%d",
            schema, table, len(resolved), len(unresolved),
        )
        if not unresolved:
            return resolved
        if self.llm is None:
            fallback = {column: _humanize_name(column) for column in unresolved}
            return {**resolved, **fallback}

        system_prompt, user_prompt = self._build_column_prompt(schema, table, unresolved)
        try:
            response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.2)
        except Exception:  # noqa: BLE001
            logger.warning("LLM не смог сгенерировать описания колонок для %s.%s", schema, table)
            fallback = {column: _humanize_name(column) for column in unresolved}
            return {**resolved, **fallback}

        lines = [line.strip("- ").strip() for line in str(response).splitlines() if line.strip()]
        generated: dict[str, str] = {}
        for idx, column in enumerate(unresolved):
            generated[column] = lines[idx] if idx < len(lines) and lines[idx] else _humanize_name(column)
        # diagnostic: видно язык/качество сгенерированных описаний (пример пары).
        sample_pair = next(iter(generated.items()), None)
        logger.info(
            "MetadataRefresh: %s.%s — LLM сгенерировал %d описаний колонок, пример: %s",
            schema, table, len(generated), sample_pair,
        )
        return {**resolved, **generated}

    def _generate_table_description(
        self,
        schema: str,
        table: str,
        columns_df: pd.DataFrame,
        sample_df: pd.DataFrame,
    ) -> str:
        table_examples = _read_yaml_examples(self.table_few_shots_path, "tables")
        table_example_map = {
            _normalize_name(item.get("table_name", "")): str(item.get("description", "")).strip()
            for item in table_examples
            if str(item.get("table_name", "")).strip() and str(item.get("description", "")).strip()
        }
        matched = table_example_map.get(_normalize_name(table))
        logger.info(
            "MetadataRefresh: генерация описания таблицы %s.%s — примеров=%d, найдено в справочнике=%s",
            schema, table, len(table_example_map), bool(matched),
        )
        if matched:
            return matched
        if self.llm is None:
            return _humanize_name(table)
        system_prompt, user_prompt = self._build_table_prompt(schema, table, columns_df, sample_df)
        try:
            response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.2)
        except Exception:  # noqa: BLE001
            logger.warning("LLM не смог сгенерировать описание таблицы %s.%s", schema, table)
            return _humanize_name(table)
        return str(response).strip() or _humanize_name(table)

    @staticmethod
    def _build_sample_values(series: pd.Series) -> str:
        non_null = series.dropna()
        if non_null.empty:
            return ""
        unique_values = [str(value).strip() for value in non_null.astype(str).unique().tolist()]
        unique_values = [value for value in unique_values if value]
        if not unique_values:
            return ""
        if len(unique_values) > 12:
            return ""
        return "|".join(unique_values[:12])

    def _collect_table_metadata(
        self,
        inspector: Any,
        schema: str,
        table: str,
        progress_callback: Any | None = None,
        allow_generation: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
        columns = inspector.get_columns(table, schema=schema)
        pk = inspector.get_pk_constraint(table, schema=schema) or {}
        pk_columns = set(pk.get("constrained_columns") or [])
        table_comment, comment_map = self._get_comment_bundle(inspector, schema, table)
        sample_df = self._get_sample_df(schema, table)

        if sample_df.empty:
            sample_df = pd.DataFrame(columns=[str(col.get("name") or "") for col in columns])

        if not pk_columns:
            def _pk_progress(checked: int, total: int) -> None:
                if progress_callback is None:
                    return
                pct = int((checked / total) * 100) if total > 0 else 0
                progress_callback(
                    f"Обновляю метаданные для таблицы {schema}.{table} | "
                    f"ищу первичный ключ: {pct}% ({checked}/{total} комбинаций)"
                )

            pk_columns = set(
                _find_candidate_primary_key(
                    sample_df,
                    progress_callback=_pk_progress,
                )
            )

        missing_comments = [
            str(col.get("name") or "")
            for col in columns
            if not str(comment_map.get(str(col.get("name") or ""), "") or "").strip()
        ]
        generation_gate = schema == SN_T_UZP_SCHEMA and allow_generation
        generated_comments = {}
        if generation_gate:
            generated_comments = self._generate_column_descriptions(schema, table, missing_comments)

        # diagnostic-счётчики: откуда взялось описание каждой колонки.
        src_counts = {"comment": 0, "generated": 0, "humanize": 0}
        rows: list[dict[str, Any]] = []
        for column in columns:
            name = str(column.get("name") or "")
            dtype = str(column.get("type") or "").strip().lower()
            comment_desc = str(comment_map.get(name, "") or "").strip()
            generated_desc = generated_comments.get(name, "")
            if comment_desc:
                description = comment_desc
                src_counts["comment"] += 1
            elif generated_desc:
                description = generated_desc
                src_counts["generated"] += 1
            else:
                description = _humanize_name(name)
                src_counts["humanize"] += 1
            not_null_perc = 0.0
            unique_perc = 0.0
            sample_values = ""
            if name in sample_df.columns and len(sample_df) > 0:
                not_null_perc = round(float(sample_df[name].notnull().mean() * 100), 2)
                unique_perc = round(float(sample_df[name].dropna().nunique() / len(sample_df) * 100), 2)
                sample_values = self._build_sample_values(sample_df[name])
            rows.append({
                "schema_name": schema,
                "table_name": table,
                "column_name": name,
                "dType": dtype,
                "is_not_null": bool(not column.get("nullable", True)),
                "description": description,
                "is_primary_key": bool(name in pk_columns),
                "not_null_perc": not_null_perc,
                "unique_perc": unique_perc,
                "foreign_key_target": "",
                "sample_values": sample_values,
                "partition_key": False,
            })

        table_description = str(table_comment or "").strip()
        table_desc_source = "comment"
        if schema == SN_T_UZP_SCHEMA and not table_description and allow_generation:
            columns_df = pd.DataFrame(rows)
            table_description = self._generate_table_description(schema, table, columns_df, sample_df)
            table_desc_source = "generated"
        elif not table_description:
            table_description = _humanize_name(table)
            table_desc_source = "humanize"

        logger.info(
            "MetadataRefresh: %s.%s — описания колонок: comment=%d, generated=%d, humanize=%d "
            "(всего %d); генерация_включена=%s; описание_таблицы=%s",
            schema, table, src_counts["comment"], src_counts["generated"],
            src_counts["humanize"], len(rows), generation_gate, table_desc_source,
        )

        table_row = {
            "schema_name": schema,
            "table_name": table,
            "description": table_description,
            "grain": "",
        }
        return (table_row, rows, sample_df)

    @staticmethod
    def _sort_catalog(tables_df: pd.DataFrame, attrs_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        tables_df = (
            tables_df[TABLE_COLUMNS]
            .drop_duplicates(subset=["schema_name", "table_name"], keep="last")
            .sort_values(["schema_name", "table_name"])
            .reset_index(drop=True)
        )
        attrs_df = (
            attrs_df[ATTR_COLUMNS]
            .drop_duplicates(subset=["schema_name", "table_name", "column_name"], keep="last")
            .sort_values(["schema_name", "table_name", "column_name"])
            .reset_index(drop=True)
        )
        return (tables_df, attrs_df)

    def _persist_few_shot_files(
        self,
        table_descriptions: "OrderedDict[str, str]",
        column_descriptions: "OrderedDict[str, str]",
    ) -> None:
        """Записать накопленные справочники few-shot (table_name/column_name → description)."""
        self.table_few_shots_path.parent.mkdir(parents=True, exist_ok=True)
        self.column_few_shots_path.parent.mkdir(parents=True, exist_ok=True)
        table_payload = {
            "tables": [
                {"table_name": table, "description": description}
                for table, description in table_descriptions.items()
            ]
        }
        column_payload = {
            "columns": [
                {"column_name": column_name, "description": description}
                for column_name, description in column_descriptions.items()
            ]
        }
        self.table_few_shots_path.write_text(
            yaml.safe_dump(table_payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.column_few_shots_path.write_text(
            yaml.safe_dump(column_payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _rebuild_few_shot_files(
        self,
        tables_df: pd.DataFrame | None = None,
        attrs_df: pd.DataFrame | None = None,
    ) -> None:
        """Накопить справочники few-shot: существующие записи сохраняются, новые
        уникальные ключи дописываются (merge, не перезапись). Так справочник
        копит описания между рефрешами и не теряет те, чьих объектов уже нет в
        текущем каталоге."""
        if tables_df is None:
            tables_df = self.schema_loader.tables_df.copy()
        if attrs_df is None:
            attrs_df = self.schema_loader.attrs_df.copy()
        if tables_df.empty:
            tables_df = pd.DataFrame(columns=TABLE_COLUMNS)
        if attrs_df.empty:
            attrs_df = pd.DataFrame(columns=ATTR_COLUMNS)

        tables_df, attrs_df = self._sort_catalog(tables_df, attrs_df)

        # Существующие накопленные записи (сохраняем как есть, в исходном порядке).
        table_descriptions: "OrderedDict[str, str]" = OrderedDict()
        for item in _read_yaml_examples(self.table_few_shots_path, "tables"):
            name = str(item.get("table_name", "") or "").strip()
            description = str(item.get("description", "") or "").strip()
            if name and description:
                table_descriptions.setdefault(name, description)

        column_descriptions: "OrderedDict[str, str]" = OrderedDict()
        for item in _read_yaml_examples(self.column_few_shots_path, "columns"):
            name = str(item.get("column_name", "") or "").strip().lower()
            description = str(item.get("description", "") or "").strip()
            if name and description:
                column_descriptions.setdefault(name, description)

        # Новые осмысленные описания из текущего каталога — добавляем только
        # отсутствующие ключи (keep-existing: накопленное не перетираем).
        for row in tables_df.itertuples(index=False):
            table = str(row.table_name or "").strip()
            description = str(row.description or "").strip()
            if not table or not description:
                continue
            if description == _humanize_name(table):
                continue
            table_descriptions.setdefault(table, description)

        for row in attrs_df.itertuples(index=False):
            column_name = str(row.column_name or "").strip().lower()
            description = str(row.description or "").strip()
            if not column_name or not description:
                continue
            if description == _humanize_name(column_name):
                continue
            column_descriptions.setdefault(column_name, description)

        self._persist_few_shot_files(table_descriptions, column_descriptions)

    def refresh_tables(
        self,
        tables: list[tuple[str, str]],
        *,
        prune_to_manifest: bool = False,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Пересобрать метаданные для конкретного списка таблиц."""
        if not tables:
            return {"refreshed": [], "failed": []}

        try:
            inspector = inspect(self.db.get_engine())
        except Exception as exc:  # noqa: BLE001
            _raise_if_kerberos_auth_error(exc)
            logger.warning("Не удалось создать SQLAlchemy inspector: %s", exc)
            return {
                "refreshed": [],
                "failed": [f"{schema}.{table}" for schema, table in tables],
            }
        current_tables = self.schema_loader.tables_df.copy()
        current_attrs = self.schema_loader.attrs_df.copy()
        if current_tables.empty:
            current_tables = pd.DataFrame(columns=TABLE_COLUMNS)
        if current_attrs.empty:
            current_attrs = pd.DataFrame(columns=ATTR_COLUMNS)

        refreshed: list[str] = []
        failed: list[str] = []
        sample_cache: dict[tuple[str, str], pd.DataFrame] = {}
        tables_with_comments: list[tuple[str, str]] = []
        tables_without_comments: list[tuple[str, str]] = []

        for schema, table in tables:
            if not self._table_exists(schema, table):
                tables_without_comments.append((schema, table))
                continue
            table_comment, comment_map = self._get_comment_bundle(inspector, schema, table)
            has_comments = bool(str(table_comment or "").strip()) or any(
                str(value or "").strip() for value in comment_map.values()
            )
            if has_comments:
                tables_with_comments.append((schema, table))
            else:
                tables_without_comments.append((schema, table))

        def _apply_result(
            schema: str,
            table: str,
            table_row: dict[str, Any],
            attr_rows: list[dict[str, Any]],
            sample_df: pd.DataFrame,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            nonlocal current_tables, current_attrs
            current_tables = current_tables[
                ~(
                    (current_tables["schema_name"] == schema)
                    & (current_tables["table_name"] == table)
                )
            ]
            current_attrs = current_attrs[
                ~(
                    (current_attrs["schema_name"] == schema)
                    & (current_attrs["table_name"] == table)
                )
            ]
            table_df = pd.DataFrame([table_row], columns=TABLE_COLUMNS)
            attrs_df = pd.DataFrame(attr_rows, columns=ATTR_COLUMNS)
            if current_tables.empty:
                current_tables = table_df
            else:
                current_tables = pd.concat([current_tables, table_df], ignore_index=True)
            if current_attrs.empty:
                current_attrs = attrs_df
            else:
                current_attrs = pd.concat([current_attrs, attrs_df], ignore_index=True)
            sample_cache[(schema, table)] = sample_df
            refreshed.append(f"{schema}.{table}")
            return current_tables, current_attrs

        def _process_pass(pass_tables: list[tuple[str, str]], *, allow_generation: bool) -> None:
            for schema, table in pass_tables:
                full_name = f"{schema}.{table}"
                if progress_callback is not None:
                    try:
                        progress_callback(f"Обновляю метаданные для таблицы {full_name}")
                    except Exception:  # noqa: BLE001
                        logger.debug("Metadata progress callback failed for %s", full_name)
                try:
                    if not self._table_exists(schema, table):
                        raise ValueError(f"Таблица {full_name} не найдена в БД")
                    table_row, attr_rows, sample_df = self._collect_table_metadata(
                        inspector,
                        schema,
                        table,
                        progress_callback=progress_callback,
                        allow_generation=allow_generation,
                    )
                except Exception as exc:  # noqa: BLE001
                    _raise_if_kerberos_auth_error(exc)
                    logger.warning("Metadata refresh skipped for %s: %s", full_name, exc)
                    failed.append(full_name)
                    continue
                _apply_result(schema, table, table_row, attr_rows, sample_df)

        _process_pass(tables_with_comments, allow_generation=False)
        self._rebuild_few_shot_files(current_tables, current_attrs)
        _process_pass(tables_without_comments, allow_generation=True)

        for schema, table in tables:
            full_name = f"{schema}.{table}"
            if full_name not in refreshed and full_name not in failed and not self._table_exists(schema, table):
                failed.append(full_name)

        if prune_to_manifest:
            manifest = set(tuple(row) for row in self.load_targets_df()[TARGET_COLUMNS].itertuples(index=False, name=None))
            current_tables = current_tables[
                current_tables.apply(lambda row: (row["schema_name"], row["table_name"]) in manifest, axis=1)
            ]
            current_attrs = current_attrs[
                current_attrs.apply(lambda row: (row["schema_name"], row["table_name"]) in manifest, axis=1)
            ]

        current_tables, current_attrs = self._sort_catalog(current_tables, current_attrs)
        self.schema_loader.replace_catalog(current_tables, current_attrs)
        self._rebuild_few_shot_files(current_tables, current_attrs)
        EnrichmentPipeline(self.schema_loader, llm=self.llm, db_manager=self.db).run(sample_cache=sample_cache)
        return {"refreshed": refreshed, "failed": failed}

    def refresh_all(self, *, progress_callback: Any | None = None) -> dict[str, Any]:
        """Пересобрать каталог по всему manifest."""
        targets = list(self.load_targets_df().itertuples(index=False, name=None))
        return self.refresh_tables(
            targets,
            prune_to_manifest=True,
            progress_callback=progress_callback,
        )

    def add_targets(
        self,
        refs: list[str],
        *,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Добавить таблицы в manifest и сразу собрать их метаданные."""
        targets = _normalize_table_refs(refs)
        current = self.load_targets_df()
        existing = set(tuple(row) for row in current[TARGET_COLUMNS].itertuples(index=False, name=None))
        added: list[tuple[str, str]] = []
        already_present: list[str] = []
        invalid_schemas: list[str] = []
        missing_tables: list[str] = []
        for schema, table in targets:
            full_name = f"{schema}.{table}"
            if not self._is_allowed_schema(schema):
                invalid_schemas.append(full_name)
                continue
            if not self._table_exists(schema, table):
                missing_tables.append(full_name)
                continue
            if (schema, table) in existing:
                already_present.append(full_name)
                continue
            added.append((schema, table))
            existing.add((schema, table))

        if added:
            updated = pd.concat([
                current,
                pd.DataFrame(added, columns=TARGET_COLUMNS),
            ], ignore_index=True)
            self.save_targets_df(updated)
            try:
                refresh_result = self.refresh_tables(
                    added,
                    progress_callback=progress_callback,
                )
            except Exception:
                failed_set = set(added)
                rolled_back = current[
                    ~current.apply(lambda row: (row["schema_name"], row["table_name"]) in failed_set, axis=1)
                ].reset_index(drop=True)
                self.save_targets_df(rolled_back)
                raise

            failed_set = {
                tuple(item.split(".", 1))
                for item in (refresh_result.get("failed") or [])
                if "." in item
            }
            if failed_set:
                persisted = self.load_targets_df()
                rolled_back = persisted[
                    ~persisted.apply(lambda row: (row["schema_name"], row["table_name"]) in failed_set, axis=1)
                ].reset_index(drop=True)
                self.save_targets_df(rolled_back)
            added = [item for item in added if item not in failed_set]
        else:
            refresh_result = {"refreshed": [], "failed": []}

        return {
            "added": [f"{schema}.{table}" for schema, table in added],
            "already_present": already_present,
            "invalid_schemas": invalid_schemas,
            "missing_tables": missing_tables,
            "refresh": refresh_result,
        }

    def remove_targets(self, refs: list[str]) -> dict[str, Any]:
        """Удалить таблицы из manifest и стереть их из CSV-каталога."""
        targets = _normalize_table_refs(refs)
        current = self.load_targets_df()
        target_set = set(targets)
        existing = set(tuple(row) for row in current[TARGET_COLUMNS].itertuples(index=False, name=None))
        removed = [f"{schema}.{table}" for schema, table in targets if (schema, table) in existing]
        missing = [f"{schema}.{table}" for schema, table in targets if (schema, table) not in existing]

        updated = current[
            ~current.apply(lambda row: (row["schema_name"], row["table_name"]) in target_set, axis=1)
        ].reset_index(drop=True)
        self.save_targets_df(updated)

        if removed:
            keep = set(tuple(row) for row in updated[TARGET_COLUMNS].itertuples(index=False, name=None))
            tables_df = self.schema_loader.tables_df.copy()
            attrs_df = self.schema_loader.attrs_df.copy()
            if not tables_df.empty:
                tables_df = tables_df[
                    tables_df.apply(lambda row: (row["schema_name"], row["table_name"]) in keep, axis=1)
                ]
            if not attrs_df.empty:
                attrs_df = attrs_df[
                    attrs_df.apply(lambda row: (row["schema_name"], row["table_name"]) in keep, axis=1)
                ]
            tables_df, attrs_df = self._sort_catalog(
                tables_df if not tables_df.empty else pd.DataFrame(columns=TABLE_COLUMNS),
                attrs_df if not attrs_df.empty else pd.DataFrame(columns=ATTR_COLUMNS),
            )
            self.schema_loader.replace_catalog(tables_df, attrs_df)
            self._rebuild_few_shot_files()
            EnrichmentPipeline(self.schema_loader, llm=self.llm, db_manager=self.db).run()

        return {"removed": removed, "missing": missing}
