"""Генерация строк таблицы: суррогатный PK, категории со всеми значениями,
функциональные зависимости. FK НЕ используется — таблицы независимы (ключи
джойнов подбираются позже на синтетике).

Масштабирование числа строк: N = max(1, round(row_count * scale)). Кардинальность
категорий сохраняем (все категории домена), не масштабируем вниз.
"""
from __future__ import annotations

import random

from agc_common import get_logger, stable_hash
from agc_generator import value_generators as vg
from agc_generator.profile_parser import Profile, Table

log = get_logger("generator.keys")


def scaled_rowcount(row_count: int, scale: float) -> int:
    return max(1, int(round(row_count * scale)))


# Абсолютные n_distinct до этого — «категориальная» кардинальность домена
# (коды/статусы/флаги), её НЕ масштабируем вниз (иначе GROUP BY схлопнется).
CATEGORICAL_ABS_MAX = 200


def categorical_cardinality(n_distinct, freqs, n_rows: int) -> int:
    """Число distinct для categorical_synth: сохраняем кардинальность домена
    (зажато в [1, n_rows]); число корзин freqs — естественный пол."""
    floor = len(freqs) if freqs else 0
    if n_distinct is None:
        k = floor or n_rows
    elif n_distinct < 0:
        k = round(-float(n_distinct) * n_rows)
    else:
        k = int(round(n_distinct))
    return max(1, min(max(k, floor), n_rows))


def _col_seed(base_seed: int, table: Table, col: str) -> int:
    return (stable_hash(table.fqn, col) ^ base_seed) & 0x7FFFFFFF


def _gen_pk_pool(table: Table, pk_col: str, n_rows: int) -> list:
    """Суррогатный PK: 1..N для числовых, иначе синтетические уникальные токены."""
    col = table.columns.get(pk_col)
    base = (col.pg_type.split("(")[0].strip().lower() if col else "bigint")
    if base in ("smallint", "integer", "bigint", "numeric", "decimal"):
        return list(range(1, n_rows + 1))
    return [f"{table.table[:8]}_{i:06d}" for i in range(1, n_rows + 1)]


def _column_order(table: Table, dep_map: dict) -> list[str]:
    """Порядок генерации: детерминант раньше зависимой колонки."""
    names = list(table.columns.keys())
    placed, order = set(), []

    def place(name):
        if name in placed:
            return
        det = dep_map.get(name)
        if det and det in table.columns and det not in placed:
            place(det)
        placed.add(name)
        order.append(name)

    for n in names:
        place(n)
    return order


def generate(profile: Profile, scale: float, seed: int) -> dict[str, list[dict]]:
    """{fqn: [row_dict, ...]}. Таблицы независимы; зависимости внутри таблицы соблюдены."""
    data: dict[str, list[dict]] = {}
    for table in profile.tables:
        n_rows = scaled_rowcount(table.row_count, scale)
        dep_map = {d["dependent"]: d["determinant"] for d in table.column_dependencies}
        order = _column_order(table, dep_map)
        cols_out: dict[str, list] = {}

        for name in order:
            col = table.columns[name]
            spec = dict(col.raw)
            spec.setdefault("pg_type", col.pg_type)
            klass = col.policy
            crng = random.Random(_col_seed(seed, table, name))
            null_frac = float(spec.get("null_frac") or 0.0)

            # Зависимая колонка (task_questionary): значение определяется детерминантом.
            if name in dep_map:
                det = dep_map[name]
                det_vals = cols_out.get(det) or [None] * n_rows
                vmap = spec.get("value_map")
                det_is_keep = table.columns.get(det) and table.columns[det].policy == "categorical_keep"
                if vmap is not None and det_is_keep:
                    cols_out[name] = vg.gen_dependent_keep(
                        det_vals, {str(k): v for k, v in vmap.items()}, crng, null_frac)
                else:
                    cols_out[name] = vg.gen_dependent_synth(
                        det_vals, crng, prefix=name[:8], null_frac=null_frac)
                continue

            if name in table.pk and klass == "key":
                cols_out[name] = _gen_pk_pool(table, name, n_rows)
            elif klass == "key":
                cols_out[name] = list(range(1, n_rows + 1))
            elif klass == "categorical_keep":
                cols_out[name] = vg.gen_categorical_keep(
                    spec.get("values") or [], n_rows, null_frac, crng)
            elif klass == "categorical_synth":
                k = categorical_cardinality(spec.get("n_distinct"), spec.get("freqs"), n_rows)
                cols_out[name] = vg.gen_categorical_synth(
                    k, spec.get("freqs"), n_rows, null_frac, crng, prefix=name[:8])
            elif klass == "sensitive_numeric":
                cols_out[name] = vg.gen_sensitive_numeric(spec, n_rows, crng)
            elif klass == "pii":
                cols_out[name] = vg.gen_pii(
                    spec.get("generator") or "text", n_rows, null_frac,
                    bool(spec.get("unique_like")), crng)
            elif klass == "datetime":
                cols_out[name] = vg.gen_datetime(spec, n_rows, crng)
            else:
                cols_out[name] = vg.gen_generic(spec, n_rows, crng)

        rows = _to_rows(table, cols_out, n_rows)
        rows = _enforce_date_order(table, rows)
        data[table.fqn] = rows
        log.info("Данные %s: %d строк, %d колонок (scale=%s)",
                 table.fqn, n_rows, len(cols_out), scale)
    return data


def _to_rows(table: Table, cols_out: dict[str, list], n_rows: int) -> list[dict]:
    names = list(table.columns.keys())
    return [{name: cols_out.get(name, [None] * n_rows)[i] for name in names}
            for i in range(n_rows)]


def _enforce_date_order(table: Table, rows: list[dict]) -> list[dict]:
    """Внутри order_group даты по возрастанию (created <= updated <= ...)."""
    groups = [name for name, col in table.columns.items() if col.get("order_group")]
    if len(groups) < 2:
        return rows
    for row in rows:
        vals = sorted(row[name] for name in groups if row[name] is not None)
        j = 0
        for name in groups:
            if row[name] is not None:
                row[name] = vals[j]
                j += 1
    return rows
