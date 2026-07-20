"""Сборка profile.json из каталога (типы/хранение) + анализа СЭМПЛА в pandas.

Ключевые решения (по договорённости):
- PK не объявлен в DDL → выводим гипотезу по сэмплу (analyze.find_pk). FK не выводим.
- Вся статистика/категории/зависимости считаются в pandas на случайном сэмпле
  (до ~1M строк), а не отдельными GROUP BY к БД. Редкие категории могут потеряться.
- Функциональные зависимости (task_subtype → task_questionary): для каждой категории
  берём одного представителя dependent (свежайшая строка, где dependent не NULL).

Для каждой колонки собираем ТОЛЬКО поля, разрешённые её классом (whitelist):
- categorical_keep : реальные категории и доли (values), либо представители FD (value_map);
- categorical_synth: n_distinct + доли (freqs), БЕЗ реальных значений;
- sensitive_numeric: тип, precision/scale, null_frac, форма распределения, порядок среднего;
- pii             : тип, null_frac, генератор, unique_like — синтетика на выходе;
- key             : роль pk (гипотеза), n_distinct, null_frac;
- datetime        : тип, null_frac, синтетический диапазон, order_group.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy.engine import Engine

from agc_common import get_logger
from agc_profiler import analyze
from agc_profiler import catalog_reader as cat
from agc_profiler import classifier as clf
from agc_profiler.policy import Policy
from agc_profiler.sampler import sample_dataframe

log = get_logger("profiler.build")

PROFILE_VERSION = 2


def _coarse_magnitude(value: float) -> str | None:
    """Огрубление до порядка величины: 48213.7 -> '~5e4'. Не реальное значение строки."""
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return None
    if v == 0:
        return "~0"
    exp = int(math.floor(math.log10(v)))
    lead = round(v / (10 ** exp))
    if lead == 10:
        lead, exp = 1, exp + 1
    return f"~{lead}e{exp}"


def _numeric_avg_hint(policy_entry: dict, series: "pd.Series | None") -> str | None:
    """avg_hint: из policy; иначе огрублённое среднее сэмпла (наружу — только порядок)."""
    if policy_entry.get("avg_hint"):
        return str(policy_entry["avg_hint"])
    if series is None:
        return None
    nums = pd.to_numeric(series, errors="coerce").dropna()
    if nums.empty:
        return None
    return _coarse_magnitude(float(nums.mean()))


def _rel_n_distinct(stats: dict, klass: str) -> float | int | None:
    """n_distinct для профиля. Категории: абсолютное число из сэмпла (не масштабируем).
    Прочее: относительная оценка (доля уникальных в сэмпле, со знаком минус)."""
    n_abs = stats.get("n_distinct")
    if klass in ("categorical_keep", "categorical_synth", "key"):
        return n_abs
    up = stats.get("unique_perc") or 0.0
    if up >= 99.0:
        return -1.0
    return -round(up / 100.0, 4) if up else n_abs


def _build_column(name: str, col_meta: dict, stats: dict, policy_entry: dict,
                  series: "pd.Series | None", is_pk: bool, order_group_cols: set,
                  dep_info: dict | None) -> dict:
    klass = policy_entry["class"]
    base = {"pg_type": col_meta["pg_type"], "policy": klass,
            "proposed_source": policy_entry.get("source"),
            "null_frac": round(stats.get("null_frac") or 0.0, 6),
            "not_null_perc": stats.get("not_null_perc"),
            "unique_perc": stats.get("unique_perc")}

    # Зависимая колонка (dependent) с сохранением представителя — особый случай.
    if dep_info and dep_info.get("value_map") is not None:
        base["policy"] = "categorical_keep"       # хранит реальные представители → whitelist
        base["depends_on"] = [dep_info["determinant"]]
        base["value_map"] = dep_info["value_map"]  # {категория: представитель}
        base["n_distinct"] = len(dep_info["value_map"])
        return base
    if dep_info and dep_info.get("value_map") is None:
        # Синтетический представитель на категорию (реальные значения не храним).
        base["policy"] = "categorical_synth"
        base["depends_on"] = [dep_info["determinant"]]
        base["dependent_synth"] = True
        base["n_distinct"] = dep_info.get("card")
        return base

    if klass == "categorical_keep":
        base["n_distinct"] = stats.get("n_distinct")
        base["values"] = stats.get("categories") or []
    elif klass == "categorical_synth":
        base["n_distinct"] = stats.get("n_distinct")
        freqs = [f for _, f in (stats.get("categories") or [])]
        if freqs:
            base["freqs"] = freqs
    elif klass == "sensitive_numeric":
        base["n_distinct"] = _rel_n_distinct(stats, klass)
        base["precision"] = col_meta.get("precision")
        base["scale"] = col_meta.get("scale")
        base["dist"] = policy_entry.get("dist", "lognormal")
        hint = _numeric_avg_hint(policy_entry, series)
        if hint:
            base["avg_hint"] = hint
    elif klass == "pii":
        up = stats.get("unique_perc") or 0.0
        base["generator"] = policy_entry.get("generator") or "text"
        base["unique_like"] = is_pk or up >= 90.0
        if col_meta.get("char_len"):
            base["char_len"] = col_meta["char_len"]
    elif klass == "key":
        base["role"] = "pk" if is_pk else "key"
        base["pk_inferred"] = bool(is_pk)
        base["n_distinct"] = _rel_n_distinct(stats, klass)
    elif klass == "datetime":
        base["range"] = policy_entry.get("range", {"start": "2018-01-01", "end": "2024-12-31"})
        if name in order_group_cols:
            base["order_group"] = True
    else:  # sensitive (generic)
        base["generator"] = policy_entry.get("generator") or "text"
    return base


def build_table_profile(engine: Engine, schema: str, table: str, policy: Policy,
                        *, sample_n: int = 1_000_000, timeout_ms: int | None = None) -> dict:
    meta = cat.read_table_meta(engine, schema, table)
    columns = cat.read_columns(engine, schema, table)
    dist_key = cat.read_distribution(engine, schema, table)
    part_keys = cat.read_partition_keys(engine, schema, table)

    df = sample_dataframe(engine, schema, table, sample_n,
                          est=meta.get("reltuples") or None, timeout_ms=timeout_ms)

    pk = analyze.find_pk(df) if not df.empty else []
    pk_set = set(pk)

    # Статистика + категории по каждой колонке из сэмпла.
    stats_by_col: dict[str, dict] = {}
    for col in columns:
        name = col["name"]
        s = analyze.column_stats(df, name)
        if 0 < s["n_distinct"] <= analyze.CATEGORICAL_MAX_DISTINCT:
            s["categories"] = analyze.top_categories(df, name)
        stats_by_col[name] = s

    # Функциональные зависимости (policy): представитель dependent на категорию.
    dep_by_col: dict[str, dict] = {}
    for dep in policy.dependencies(schema, table):
        det, dcol = dep["determinant"], dep["dependent"]
        rep = analyze.dependency_representatives(df, det, dcol, dep.get("order_by"))
        if not rep:
            log.warning("Зависимость %s→%s: представителей не найдено (пусто/нет колонок)", det, dcol)
            continue
        if dep["keep_representative"]:
            dep_by_col[dcol] = {"determinant": det, "value_map": rep}
        else:
            dep_by_col[dcol] = {"determinant": det, "value_map": None, "card": len(rep)}
        log.info("Зависимость %s→%s: %d категорий, keep_representative=%s",
                 det, dcol, len(rep), dep["keep_representative"])

    order_group_cols = {c for grp in policy.order_groups(schema, table) for c in grp}

    out_columns: dict[str, dict] = {}
    for col in columns:
        name = col["name"]
        stats = stats_by_col[name]
        sample_vals = (df[name].dropna().astype(str).head(30).tolist()
                       if (not df.empty and name in df.columns) else [])
        proposed, gen = clf.propose(name, col["pg_type"], is_pk=name in pk_set,
                                    n_distinct=stats.get("n_distinct"), sample_values=sample_vals)
        policy_entry = policy.resolve(schema, table, name, proposed, gen)
        series = df[name] if (not df.empty and name in df.columns) else None
        out_columns[name] = _build_column(name, col, stats, policy_entry, series,
                                           name in pk_set, order_group_cols, dep_by_col.get(name))

    profile = {
        "schema": schema, "table": table,
        "relkind": meta["relkind"], "is_view": meta["is_view"], "storage": meta["storage"],
        "distributed_by": dist_key, "partitioned_by": part_keys,
        "row_count": {"value": meta["reltuples"], "estimated": meta["row_count_estimated"],
                      "sample_rows": int(len(df))},
        "pk_hypothesis": pk,
        "column_dependencies": [{"determinant": d["determinant"], "dependent": dcol}
                                for dcol, d in dep_by_col.items()],
        "defaults": {c["name"]: c["default"] for c in columns if c["default"]},
        "not_null": [c["name"] for c in columns if not c["nullable"]],
        "columns": out_columns,
    }
    log.info("Профиль %s.%s: %d колонок, pk=%s, storage=%s, sample=%d строк",
             schema, table, len(out_columns), pk or "-", meta["storage"], len(df))
    return profile


def build_profile(engine: Engine, tables: list[tuple[str, str]], policy: Policy,
                  *, sample_n: int = 1_000_000, timeout_ms: int | None = None) -> dict:
    table_profiles = [build_table_profile(engine, s, t, policy,
                                          sample_n=sample_n, timeout_ms=timeout_ms)
                      for s, t in tables]
    return {
        "profile_version": PROFILE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dialect": "greenplum",
        "note": "Синтетический профиль: реальные значения только в categorical_keep. "
                "PK — гипотеза по сэмплу; FK не выводится.",
        "tables": table_profiles,
    }
