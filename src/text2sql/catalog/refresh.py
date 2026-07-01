"""Generic-рефреш метаданных из ЛЮБОЙ БД.

Строит минимальный предвычисленный слой, консистентный с реальными данными той
БД, которую агент опрашивает: структура+типы (интроспекция), статистики и
sample_values (по сэмплу), составной PK-гипотеза (минимальная уникальная
комбинация на сэмпле), semantic_class (generic-эвристика), join-кандидаты.

Описания/grain СИДИРУЮТСЯ из кураторских few-shots (привязаны к именам и
стабильны при той же схеме); fallback — LLM, затем humanize. Никакой доменной
логики под конкретные таблицы — всё выводится из данных и имён.

Выход: tables_list.csv, attr_list.csv, join_candidates.json.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config import PATHS
from ..db.adapter import DbAdapter

logger = logging.getLogger(__name__)

_METRIC_RE = re.compile(r"(^|_)(qty|quantity|amt|amount|sum|total|cnt|count|avg|rate|ratio|pct|perc|percent|val|value)($|_)", re.I)
_ID_RE = re.compile(r"(^|_)(id|code|key|inn|kpp|ogrn|okato|oktmo)($|_)", re.I)
# Системные таймстемпы — почти всегда уникальны, но НЕ бизнес-ключи. Отличаем от
# бизнес-дат (report_dt, task_create_dt): системные — *_dttm / inserted/modified/updated.
_SYS_TS_RE = re.compile(r"(dttm$|timestamp|inserted|modified|updated|_update_|^update_|load_)", re.I)
_DATE_TYPES = ("date", "timestamp", "time")
_NUM_TYPES = ("int", "numeric", "decimal", "double", "real", "float", "smallint", "bigint")
_BOOL_TYPES = ("bool",)
_TEXT_TYPES = ("char", "text")


def _humanize(name: str) -> str:
    return " ".join(p for p in str(name or "").split("_") if p)


def _is_metric_name(name: str) -> bool:
    return bool(_METRIC_RE.search(name or ""))


def _classify(name: str, dtype: str, unique_pct: float, n_distinct: int) -> str:
    """Generic semantic_class по типу+имени+кардинальности. Не доменная логика."""
    d = dtype.lower()
    if any(b in d for b in _BOOL_TYPES):
        return "flag"
    if any(t in d for t in _DATE_TYPES):
        return "date"
    if any(t in d for t in _NUM_TYPES):
        if _is_metric_name(name):
            return "metric"
        if _ID_RE.search(name):
            return "join_key"
        return "metric" if unique_pct > 50 else "join_key"
    if any(t in d for t in _TEXT_TYPES):
        if _ID_RE.search(name):
            return "join_key"
        if n_distinct <= 50 and unique_pct < 20:
            return "enum_like"
        if unique_pct > 80:
            return "free_text"
        return "label"
    return "attribute"


def _find_pk(df: pd.DataFrame, max_cols: int = 4) -> list[str]:
    """Минимальная уникальная комбинация на сэмпле. Бизнес-ключи в приоритете:
    метрики и СИСТЕМНЫЕ таймстемпы (inserted/modified/update _dttm) откладываем —
    иначе почти-уникальный load-таймстемп ложно становится PK (и на проде тоже)."""
    if df.empty:
        return []
    cols = [c for c in df.columns if df[c].notna().all() and df[c].nunique(dropna=False) > 1]
    if not cols:
        return []
    def _low_priority(c: str) -> bool:
        return _is_metric_name(c) or bool(_SYS_TS_RE.search(c))
    preferred = [c for c in cols if not _low_priority(c)]
    deferred = [c for c in cols if _low_priority(c)]
    for candidates in ([preferred] if preferred else []) + [preferred + deferred]:
        upper = min(max_cols, len(candidates))
        for size in range(1, upper + 1):
            for combo in combinations(candidates, size):
                if not df.duplicated(subset=list(combo)).any():
                    return list(combo)
    return []


def _sample_values(series: pd.Series, cap: int = 25) -> str:
    """Значения категориальной колонки (для резолва фильтров). Порог 25 —
    чтобы захватить enum/подтипы (напр. task_subtype). Высококардинальные → ''."""
    non_null = series.dropna()
    if non_null.empty:
        return ""
    uniq = [str(v).strip() for v in non_null.astype(str).unique().tolist() if str(v).strip()]
    if not uniq or len(uniq) > cap:
        return ""
    return "|".join(uniq[:cap])


def _read_seed(path: Path, key: str, name_field: str) -> dict[str, str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    out: dict[str, str] = {}
    for item in data.get(key, []) or []:
        if isinstance(item, dict):
            n = str(item.get(name_field, "")).strip().lower()
            d = str(item.get("description", "")).strip()
            if n and d:
                out[n] = d
    return out


class MetadataRefresh:
    def __init__(self, db: DbAdapter, *, llm=None, data_dir: Path | None = None,
                 sample_n: int = 100_000, per_table_timeout_ms: int = 300_000,
                 pk_source: str | None = None):
        self.db = db
        self.llm = llm
        self.data_dir = data_dir or PATHS.data_dir
        self.sample_n = sample_n
        self.per_table_timeout_ms = per_table_timeout_ms  # 5 мин на таблицу
        # Источник PK: "compute" — вычислять из сэмпла (прод, реальные данные);
        # "seed" — брать из кураторской метадаты (тест-контейнер с синтетикой, где
        # эвристика по сэмплу ненадёжна). По умолчанию из env PK_SOURCE, иначе compute.
        self.pk_source = (pk_source or os.getenv("PK_SOURCE", "compute")).strip().lower()
        self._col_seed = _read_seed(self.data_dir / "column_description_few_shots.yaml", "columns", "column_name")
        self._tbl_seed = _read_seed(self.data_dir / "table_description_few_shots.yaml", "tables", "table_name")
        self._grain_seed = self._load_grain_seed()
        self._pk_seed = self._load_pk_seed()

    def _load_grain_seed(self) -> dict[str, str]:
        path = self.data_dir / "tables_list.csv"
        if not path.exists():
            return {}
        try:
            df = pd.read_csv(path).fillna("")
        except Exception:  # noqa: BLE001
            return {}
        return {str(r.table_name).lower(): str(getattr(r, "grain", "")) for r in df.itertuples(index=False)}

    def _load_pk_seed(self) -> dict[str, list[str]]:
        """Кураторские составные PK из существующего attr_list.csv. На синтетических
        данных контейнера эвристика по сэмплу ненадёжна (случайная уникальность),
        поэтому известные ключи сидируем; эвристика — только fallback для новых таблиц."""
        path = self.data_dir / "attr_list.csv"
        if not path.exists():
            return {}
        try:
            df = pd.read_csv(path).fillna("")
        except Exception:  # noqa: BLE001
            return {}
        seed: dict[str, list[str]] = {}
        for r in df.itertuples(index=False):
            if str(getattr(r, "is_primary_key", "")).strip().lower() in {"true", "1", "yes"}:
                seed.setdefault(f"{r.schema_name}.{r.table_name}", []).append(str(r.column_name))
        return seed

    @staticmethod
    def _comments_schema(schema: str) -> str:
        """View-redirect: для *_ld_..._sn_uzp описания-комментарии живут в
        *_as_..._sn_view (как в исходном проекте). Иначе — та же схема."""
        if "_ld_" in schema and schema.endswith("_sn_uzp"):
            return schema.replace("_ld_", "_as_")[: -len("_sn_uzp")] + "_sn_view"
        return schema

    def _read_comments(self, schema: str, table: str) -> tuple[str, dict[str, str]]:
        """Комментарии с view-redirect: сначала пробуем view-схему, потом свою."""
        redirect = self._comments_schema(schema)
        if redirect != schema:
            tc, cc = self.db.read_comments(redirect, table)
            if tc or any(cc.values()):
                return tc, cc
        return self.db.read_comments(schema, table)

    def _describe_columns(self, schema: str, table: str, cols: list[str],
                          comments: dict[str, str]) -> dict[str, str]:
        """Описание каждой колонки: comment(view) → few-shot сид → LLM(рус) → humanize."""
        out: dict[str, str] = {}
        to_generate: list[str] = []
        for name in cols:
            desc = comments.get(name) or self._col_seed.get(name.lower())
            if desc:
                out[name] = desc
            else:
                to_generate.append(name)
        if to_generate and self.llm is not None:
            out.update(self._llm_column_descriptions(schema, table, to_generate))
            to_generate = [c for c in to_generate if c not in out]
        for name in to_generate:
            out[name] = _humanize(name)  # последний фолбэк
        return out

    def _llm_column_descriptions(self, schema: str, table: str, cols: list[str]) -> dict[str, str]:
        system = (
            "Ты senior аналитик DWH. Кратко опиши ПО-РУССКИ смысл каждого атрибута БД "
            "по его имени. Одна строка на атрибут, тот же порядок, без нумерации и лишних "
            "слов. Аббревиатуры не расшифровывай."
        )
        user = (f"Схема: {schema}\nТаблица: {table}\nАтрибуты (по одному на строку):\n"
                + "\n".join(f"- {c}" for c in cols))
        try:
            text_out = self.llm.complete(system, user, max_tokens=2000, node="meta_cols").text
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM-описания колонок %s.%s не сгенерированы: %s", schema, table, exc)
            return {}
        lines = [ln.strip(" -\t").strip() for ln in text_out.splitlines() if ln.strip()]
        out: dict[str, str] = {}
        for i, c in enumerate(cols):
            if i < len(lines) and lines[i]:
                out[c] = lines[i]
        return out

    def _describe_table(self, schema: str, table: str, table_comment: str) -> str:
        desc = table_comment or self._tbl_seed.get(table.lower())
        if desc:
            return desc
        if self.llm is not None:
            try:
                system = ("Ты senior аналитик DWH. Дай КОРОТКОЕ описание таблицы ПО-РУССКИ "
                          "(1-2 предложения) по её имени. Без лишних слов.")
                out = self.llm.complete(system, f"Схема: {schema}\nТаблица: {table}",
                                        max_tokens=300, node="meta_table").text.strip()
                if out:
                    return out
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM-описание таблицы %s.%s не сгенерировано: %s", schema, table, exc)
        return _humanize(table)

    def _role(self, grain: str, class_counts: dict[str, int]) -> str:
        if grain in ("organization", "reference") or class_counts.get("label", 0) >= class_counts.get("metric", 0) * 2 + 1 and class_counts.get("metric", 0) == 0:
            return "reference"
        if grain == "event":
            return "event"
        return "fact"

    def _collect_table(self, schema: str, table: str) -> tuple[dict, list[dict], list[str]]:
        """Собрать метаданные ОДНОЙ таблицы. Сэмпл — адаптивный (по размеру),
        с таймаутом per_table_timeout_ms. Может бросить исключение (таймаут/ошибка)."""
        fqn = f"{schema}.{table}"
        cols_meta = self.db.introspect_columns(schema, table)
        # Описания: комментарий из view-схемы → few-shot сид → LLM(рус) → humanize.
        table_comment, col_comments = self._read_comments(schema, table)
        col_desc = self._describe_columns(schema, table, [c["column_name"] for c in cols_meta], col_comments)
        sample = self.db.metadata_sample(schema, table, self.sample_n,
                                         timeout_ms=self.per_table_timeout_ms)
        df = pd.DataFrame(sample.rows, columns=sample.columns) if sample.rows else \
            pd.DataFrame(columns=sample.columns)
        n = len(df)
        # На проде PK вычисляется из реальных данных (как в исходном проекте);
        # сид — fallback. На синтетике (PK_SOURCE=seed) — наоборот. См. __init__.
        if self.pk_source == "seed":
            pk = self._pk_seed.get(fqn) or _find_pk(df)
        else:
            pk = _find_pk(df) or self._pk_seed.get(fqn)
        class_counts: dict[str, int] = {}
        attr_rows: list[dict] = []
        for cm in cols_meta:
            name, dtype = cm["column_name"], cm["data_type"]
            nn_perc = round(float(df[name].notna().mean() * 100), 2) if name in df.columns and n else 0.0
            n_distinct = int(df[name].dropna().nunique()) if name in df.columns and n else 0
            uniq_perc = round(n_distinct / n * 100, 2) if n else 0.0
            sclass = _classify(name, dtype, uniq_perc, n_distinct)
            class_counts[sclass] = class_counts.get(sclass, 0) + 1
            attr_rows.append({
                "schema_name": schema, "table_name": table, "column_name": name,
                "dType": dtype, "is_not_null": cm["is_nullable"] == "NO",
                "description": col_desc.get(name) or _humanize(name), "is_primary_key": name in pk,
                "not_null_perc": nn_perc, "unique_perc": uniq_perc,
                "sample_values": _sample_values(df[name]) if name in df.columns else "",
                "semantic_class": sclass,
            })
        grain = self._grain_seed.get(table.lower(), "")
        table_row = {"schema_name": schema, "table_name": table,
                     "description": self._describe_table(schema, table, table_comment), "grain": grain,
                     "role": self._role(grain, class_counts)}
        logger.info("metadata: собрана %s (строк сэмпла=%d, колонок=%d, pk=%s)", fqn, n, len(attr_rows), pk)
        return table_row, attr_rows, pk

    def manifest_tables(self) -> list[tuple[str, str]]:
        """Таблицы из манифеста tables_list.csv (источник истины для рефреша)."""
        path = self.data_dir / "tables_list.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path).fillna("")
        return [(str(r.schema_name), str(r.table_name)) for r in df.itertuples(index=False)]

    def refresh_manifest(self, *, progress_callback=None) -> dict[str, Any]:
        """Пересобрать метаданные ТОЛЬКО по таблицам манифеста. Таблица, не прошедшая
        по таймауту/ошибке, пропускается (старые метаданные сохраняются), рефреш
        продолжается со следующей."""
        manifest = self.manifest_tables()
        old_tables = {(r["schema_name"], r["table_name"]): r for r in self._read_csv("tables_list.csv")}
        old_attrs: dict[tuple, list[dict]] = {}
        for r in self._read_csv("attr_list.csv"):
            old_attrs.setdefault((r["schema_name"], r["table_name"]), []).append(r)

        table_rows: list[dict] = []
        attr_rows: list[dict] = []
        refreshed: list[str] = []
        failed: list[tuple[str, str]] = []

        for schema, table in manifest:
            fqn = f"{schema}.{table}"
            if progress_callback:
                progress_callback(f"Обновляю метаданные: {fqn}")
            try:
                tr, ar, _ = self._collect_table(schema, table)
                table_rows.append(tr)
                attr_rows.extend(ar)
                refreshed.append(fqn)
            except Exception as exc:  # noqa: BLE001  (таймаут/ошибка — не падаем)
                logger.warning("refresh: таблица %s пропущена: %s", fqn, exc)
                failed.append((fqn, str(exc)))
                if progress_callback:
                    progress_callback(f"⚠ {fqn}: пропущена ({exc}). Продолжаю со следующей.")
                # Сохраняем прежние метаданные пропущенной таблицы, если были.
                if (schema, table) in old_tables:
                    table_rows.append(old_tables[(schema, table)])
                    attr_rows.extend(old_attrs.get((schema, table), []))

        pk_map = self._pk_map(attr_rows)
        jc = self._build_join_candidates(attr_rows, pk_map)
        self._persist(table_rows, attr_rows, jc)
        return {"refreshed": refreshed, "failed": failed, "tables": len(table_rows),
                "columns": len(attr_rows), "join_candidates": len(jc)}

    def add_table(self, schema: str, table: str) -> dict[str, Any]:
        """Добавить таблицу в манифест и собрать её метаданные (инкрементально)."""
        fqn = f"{schema}.{table}"
        if not self.db.table_exists(schema, table):
            return {"status": "missing", "fqn": fqn}
        try:
            tr, ar, pk = self._collect_table(schema, table)
        except Exception as exc:  # noqa: BLE001
            logger.warning("add_table %s: ошибка сбора: %s", fqn, exc)
            return {"status": "error", "fqn": fqn, "error": str(exc)}

        tables = [r for r in self._read_csv("tables_list.csv")
                  if not (r["schema_name"] == schema and r["table_name"] == table)] + [tr]
        attrs = [r for r in self._read_csv("attr_list.csv")
                 if not (r["schema_name"] == schema and r["table_name"] == table)] + ar
        jc = self._build_join_candidates(attrs, self._pk_map(attrs))
        self._persist(tables, attrs, jc)
        return {"status": "added", "fqn": fqn, "columns": len(ar), "pk": pk}

    def remove_table(self, schema: str, table: str) -> dict[str, Any]:
        """Удалить таблицу из манифеста и метаданных (tables_list, attr_list, join-граф)."""
        fqn = f"{schema}.{table}"
        tables_all = self._read_csv("tables_list.csv")
        present = any(r["schema_name"] == schema and r["table_name"] == table for r in tables_all)
        if not present:
            return {"status": "absent", "fqn": fqn}
        tables = [r for r in tables_all if not (r["schema_name"] == schema and r["table_name"] == table)]
        attrs = [r for r in self._read_csv("attr_list.csv")
                 if not (r["schema_name"] == schema and r["table_name"] == table)]
        jc = self._build_join_candidates(attrs, self._pk_map(attrs))
        self._persist(tables, attrs, jc)
        logger.info("metadata: таблица %s удалена из манифеста", fqn)
        return {"status": "removed", "fqn": fqn}

    def _read_csv(self, name: str) -> list[dict]:
        path = self.data_dir / name
        if not path.exists():
            return []
        return pd.read_csv(path).fillna("").to_dict("records")

    @staticmethod
    def _pk_map(attr_rows: list[dict]) -> dict[str, list[str]]:
        pk_map: dict[str, list[str]] = {}
        for r in attr_rows:
            if str(r.get("is_primary_key")).strip().lower() in {"true", "1", "yes"}:
                pk_map.setdefault(f"{r['schema_name']}.{r['table_name']}", []).append(str(r["column_name"]))
        return pk_map

    def _build_join_candidates(self, attr_rows: list[dict], pk_map: dict[str, list[str]]) -> list[dict]:
        """Кандидаты join: пары key-like колонок с совместимым типом и
        пересечением значений (живая проба). Реколл-подсказка для LLM, не истина."""
        by_table: dict[str, list[dict]] = {}
        for r in attr_rows:
            if r["semantic_class"] in ("join_key", "identifier") or r["is_primary_key"]:
                by_table.setdefault(f"{r['schema_name']}.{r['table_name']}", []).append(r)
        out: list[dict] = []
        fqns = list(by_table.keys())
        for i in range(len(fqns)):
            for j in range(i + 1, len(fqns)):
                la, ra = fqns[i], fqns[j]
                pairs = []
                for lc in by_table[la]:
                    for rc in by_table[ra]:
                        if not _type_compatible(lc["dType"], rc["dType"]):
                            continue
                        if lc["column_name"] != rc["column_name"] and not (
                            rc["column_name"] in pk_map.get(ra, []) or lc["column_name"] in pk_map.get(la, [])
                        ):
                            continue
                        ov = self._overlap(la, lc["column_name"], ra, rc["column_name"])
                        if ov > 0:
                            pairs.append({"left_col": lc["column_name"], "right_col": rc["column_name"], "overlap": ov})
                if pairs:
                    out.append({"left": la, "right": ra, "pairs": pairs})
        return out

    def _overlap(self, la: str, lc: str, ra: str, rc: str) -> int:
        ls, lt = la.split(".", 1)
        rs, rt = ra.split(".", 1)
        sql = (
            f"SELECT count(*) AS c FROM (SELECT DISTINCT {self.db.quote_ident(lc)} AS v "
            f"FROM {self.db.qualified(ls, lt)} WHERE {self.db.quote_ident(lc)} IS NOT NULL) a "
            f"JOIN (SELECT DISTINCT {self.db.quote_ident(rc)} AS v "
            f"FROM {self.db.qualified(rs, rt)} WHERE {self.db.quote_ident(rc)} IS NOT NULL) b USING (v)"
        )
        try:
            res = self.db.run_select(sql, limit=1)
            return int(res.rows[0]["c"]) if res.rows else 0
        except Exception:  # noqa: BLE001
            return 0

    def _persist(self, table_rows, attr_rows, join_candidates) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(table_rows, columns=["schema_name", "table_name", "description", "grain", "role"]) \
            .to_csv(self.data_dir / "tables_list.csv", index=False)
        pd.DataFrame(attr_rows, columns=[
            "schema_name", "table_name", "column_name", "dType", "is_not_null", "description",
            "is_primary_key", "not_null_perc", "unique_perc", "sample_values", "semantic_class",
        ]).to_csv(self.data_dir / "attr_list.csv", index=False)
        (self.data_dir / "join_candidates.json").write_text(
            json.dumps(join_candidates, ensure_ascii=False, indent=2), encoding="utf-8")


def _type_compatible(a: str, b: str) -> bool:
    def fam(t: str) -> str:
        t = t.lower()
        if any(x in t for x in _NUM_TYPES):
            return "num"
        if any(x in t for x in _DATE_TYPES):
            return "date"
        if any(x in t for x in _TEXT_TYPES):
            return "text"
        return t
    return fam(a) == fam(b)
