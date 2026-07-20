"""Сборка profile.json из каталога + pg_stats + (опц.) сэмпла форматов.

Для каждой колонки собираем ТОЛЬКО поля, разрешённые её классом:
- categorical_keep : реальные distinct-значения и частоты (единственный класс,
                     куда уходят most_common_vals);
- categorical_synth: n_distinct + null_frac + форма частот (числа), БЕЗ значений;
- sensitive_numeric: тип, precision/scale, null_frac, n_distinct, форма распределения,
                     грубый ориентир среднего (порядок величины), БЕЗ mcv/histogram/min/max;
- pii             : тип, null_frac, avg_width, генератор, unique_like, БЕЗ значений;
- key             : роль pk/fk, references, n_distinct, null_frac, fanout;
- datetime        : тип, null_frac, синтетический диапазон, order_group.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from sqlalchemy.engine import Engine

from agc_common import get_logger
from agc_profiler import catalog_reader as cat
from agc_profiler import classifier as clf
from agc_profiler import stats_reader as st
from agc_profiler.policy import Policy
from agc_profiler.sampler import estimate_row_count, sample_columns

log = get_logger("profiler.build")

PROFILE_VERSION = 1


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


def _numeric_avg_hint(policy_entry: dict, hist: list) -> str | None:
    """avg_hint: сначала из policy; иначе огрублённая медиана histogram_bounds
    (внутренняя величина, наружу уходит только порядок, не само значение)."""
    if policy_entry.get("avg_hint"):
        return str(policy_entry["avg_hint"])
    nums = []
    for x in hist or []:
        try:
            nums.append(float(x))
        except (TypeError, ValueError):
            continue
    if not nums:
        return None
    nums.sort()
    return _coarse_magnitude(nums[len(nums) // 2])


def _build_categorical_keep(stats: dict) -> dict:
    mcv, mcf = stats.get("mcv") or [], stats.get("mcf") or []
    values = [[v, round(float(f), 6)] for v, f in zip(mcv, mcf)]
    covered = sum(f for _, f in values)
    # Хвост распределения (всё, что вне most_common) — одной синтетической группой.
    if covered < 0.999 and (stats.get("null_frac") or 0) + covered < 0.999:
        values.append(["__other__", round(max(0.0, 1.0 - covered - (stats.get("null_frac") or 0)), 6)])
    return {
        "null_frac": round(stats.get("null_frac") or 0.0, 6),
        "n_distinct": stats.get("n_distinct"),
        "values": values,
    }


def _build_column(schema: str, table: str, col: dict, stats: dict, policy_entry: dict,
                  cons: dict, order_group_cols: set) -> dict:
    klass = policy_entry["class"]
    name = col["name"]
    base = {"pg_type": col["pg_type"], "policy": klass, "proposed_source": policy_entry.get("source")}
    null_frac = round(stats.get("null_frac") or 0.0, 6)
    n_distinct = stats.get("n_distinct")

    if klass == "categorical_keep":
        base.update(_build_categorical_keep(stats))
    elif klass == "categorical_synth":
        # Форму частот (числа) сохраняем, реальные метки — НЕТ.
        freqs = [round(float(f), 6) for f in (stats.get("mcf") or [])]
        base.update({"null_frac": null_frac, "n_distinct": n_distinct})
        if freqs:
            base["freqs"] = freqs
    elif klass == "sensitive_numeric":
        base.update({
            "null_frac": null_frac,
            "n_distinct": n_distinct,
            "precision": col.get("precision"),
            "scale": col.get("scale"),
            "dist": policy_entry.get("dist", "lognormal"),
        })
        hint = _numeric_avg_hint(policy_entry, stats.get("histogram"))
        if hint:
            base["avg_hint"] = hint
    elif klass == "pii":
        uniqueish = _is_unique_like(name, n_distinct, cons)
        base.update({
            "null_frac": null_frac,
            "avg_width": stats.get("avg_width"),
            "generator": policy_entry.get("generator") or "text",
            "unique_like": uniqueish,
        })
        if col.get("char_len"):
            base["char_len"] = col["char_len"]
    elif klass == "key":
        role = "pk" if name in (cons.get("pk") or []) else ("fk" if _is_fk(name, cons) else "key")
        base.update({"role": role, "null_frac": null_frac, "n_distinct": n_distinct})
        fk = _fk_for(name, cons)
        if fk:
            idx = fk["columns"].index(name)
            ref_col = fk["ref_columns"][idx] if idx < len(fk["ref_columns"]) else None
            base["references"] = {"schema": fk.get("ref_schema"),
                                  "table": fk.get("ref_table"), "column": ref_col}
            base["fanout"] = {"dist": "uniform"}  # форма fan-out по умолчанию равномерная
    elif klass == "datetime":
        base.update({"null_frac": null_frac})
        # Диапазон СИНТЕТИЧЕСКИЙ (не реальные min/max): дефолт — последние ~7 лет.
        base["range"] = policy_entry.get("range", {"start": "2018-01-01", "end": "2024-12-31"})
        if name in order_group_cols:
            base["order_group"] = True
    else:  # sensitive (generic)
        base.update({
            "null_frac": null_frac,
            "avg_width": stats.get("avg_width"),
            "generator": policy_entry.get("generator") or "text",
        })
    return base


def _is_fk(name: str, cons: dict) -> bool:
    return any(name in (fk.get("columns") or []) for fk in cons.get("fks") or [])


def _fk_for(name: str, cons: dict) -> dict | None:
    for fk in cons.get("fks") or []:
        if name in (fk.get("columns") or []):
            return fk
    return None


def _is_unique_like(name: str, n_distinct: float | None, cons: dict) -> bool:
    if name in (cons.get("pk") or []):
        return True
    if any(name in u for u in cons.get("uniques") or []):
        return True
    if n_distinct is not None and n_distinct < 0 and n_distinct <= -0.9:
        return True
    return False


def build_table_profile(engine: Engine, schema: str, table: str, policy: Policy,
                        table_stats: dict, *, sample: bool = False,
                        recompute_missing: bool = False) -> dict:
    meta = cat.read_table_meta(engine, schema, table)
    columns = cat.read_columns(engine, schema, table)
    cons = cat.read_constraints(engine, schema, table)
    dist_key = cat.read_distribution(engine, schema, table)
    part_keys = cat.read_partition_keys(engine, schema, table)

    pk_set = set(cons.get("pk") or [])
    fk_cols = {c for fk in cons.get("fks") or [] for c in (fk.get("columns") or [])}

    # Досчёт статистики по колонкам без pg_stats (по флагу — это скан).
    if recompute_missing:
        missing = [c["name"] for c in columns if c["name"] not in table_stats]
        if missing:
            table_stats = {**table_stats,
                           **st.recompute_missing(engine, schema, table, missing)}
    else:
        missing = [c["name"] for c in columns if c["name"] not in table_stats]
        if missing:
            log.warning("Нет pg_stats для %d колонок %s.%s (ANALYZE не делался?): %s. "
                        "Запустите с --recompute-missing для точечного досчёта.",
                        len(missing), schema, table, ", ".join(missing[:8]))

    # Сэмпл форматов — только для колонок-кандидатов в pii/sensitive (короткие поля).
    samples: dict[str, list] = {}
    if sample:
        need = [c for c in columns
                if c["name"] not in pk_set and c["name"] not in fk_cols
                and clf._base_type(c["pg_type"]) in
                ("text", "character varying", "varchar", "character", "char")]
        if need:
            try:
                samples = sample_columns(engine, schema, table, need,
                                         est=meta.get("reltuples") or None)
            except Exception as exc:  # noqa: BLE001
                log.warning("Сэмпл %s.%s не удался: %s", schema, table, exc)

    order_group_cols = {c for grp in policy.order_groups(schema, table) for c in grp}

    out_columns: dict[str, dict] = {}
    for col in columns:
        name = col["name"]
        stats = table_stats.get(name, {})
        proposed, gen = clf.propose(
            name, col["pg_type"],
            is_pk=name in pk_set, is_fk=name in fk_cols,
            n_distinct=stats.get("n_distinct"),
            sample_values=samples.get(name),
        )
        policy_entry = policy.resolve(schema, table, name, proposed, gen)
        out_columns[name] = _build_column(schema, table, col, stats, policy_entry,
                                           cons, order_group_cols)

    profile = {
        "schema": schema,
        "table": table,
        "relkind": meta["relkind"],
        "is_view": meta["is_view"],
        "storage": meta["storage"],
        "distributed_by": dist_key,
        "partitioned_by": part_keys,
        "row_count": {"value": meta["reltuples"], "estimated": meta["row_count_estimated"]},
        "constraints": {
            "pk": cons.get("pk") or [],
            "uniques": cons.get("uniques") or [],
            "fks": [{k: v for k, v in fk.items() if not k.startswith("_")}
                    for fk in cons.get("fks") or []],
            "checks": cons.get("checks") or [],
        },
        "defaults": {c["name"]: c["default"] for c in columns if c["default"]},
        "not_null": [c["name"] for c in columns if not c["nullable"]],
        "columns": out_columns,
    }
    log.info("Профиль %s.%s: %d колонок, storage=%s, dist=%s, part=%s, rows~%s",
             schema, table, len(out_columns), meta["storage"], dist_key or "-",
             part_keys or "-", meta["reltuples"])
    return profile


def build_profile(engine: Engine, tables: list[tuple[str, str]], policy: Policy,
                  *, sample: bool = False, recompute_missing: bool = False) -> dict:
    """Собирает полный профиль по списку (schema, table). pg_stats читаем пачкой на схему."""
    by_schema: dict[str, list[str]] = {}
    for schema, table in tables:
        by_schema.setdefault(schema, []).append(table)

    stats_cache: dict[str, dict[str, dict]] = {}
    for schema, tnames in by_schema.items():
        try:
            stats_cache[schema] = st.read_pg_stats(engine, schema, tnames)
        except Exception as exc:  # noqa: BLE001
            log.warning("pg_stats по схеме %s не прочитан: %s", schema, exc)
            stats_cache[schema] = {t: {} for t in tnames}

    table_profiles = []
    for schema, table in tables:
        table_stats = stats_cache.get(schema, {}).get(table, {})
        table_profiles.append(build_table_profile(
            engine, schema, table, policy, table_stats,
            sample=sample, recompute_missing=recompute_missing,
        ))

    return {
        "profile_version": PROFILE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dialect": "greenplum",
        "note": "Синтетический профиль: реальные значения только в categorical_keep.",
        "tables": table_profiles,
    }
