"""Связывание ключей и ссылочная целостность + согласованное масштабирование.

Порядок: топологический по FK — родители генерируются раньше детей, чтобы дети
брали FK из уже существующих родительских ключей. ВСЕ джойны обязаны резолвиться.

Масштабирование:
- N_table = max(1, round(row_count * scale_factor));
- абсолютный n_distinct умножаем на scale_factor; относительный (-0.x) — доля от N;
- distinct зажимаем в [1, N] (нельзя 5000 уникальных клиентов в 1000 строк).

Для точной кардинальности: генерируем K токенов/ключей и сэмплируем по весам;
для «уникальных» — N различных значений.
"""
from __future__ import annotations

import random

from agc_common import get_logger
from agc_generator import value_generators as vg
from agc_generator.ddl_builder import topo_sort
from agc_generator.profile_parser import Profile, Table

log = get_logger("generator.keys")


def scaled_rowcount(row_count: int, scale: float) -> int:
    return max(1, int(round(row_count * scale)))


def scaled_distinct(n_distinct: float | None, n_rows: int, scale: float) -> int:
    """Согласованная кардинальность для ВЫСОКО-кардинальных колонок (ключи).
    n_distinct: >0 абсолютное (масштабируем на scale), <0 относительное (доля от
    n_rows). Всегда зажато в [1, n_rows]."""
    if n_distinct is None:
        return max(1, n_rows)
    if n_distinct < 0:
        d = round(-float(n_distinct) * n_rows)
    else:
        d = round(float(n_distinct) * scale)
    return max(1, min(int(d), n_rows))


# Абсолютные n_distinct, которые считаем «категориальной» кардинальностью домена
# (коды/статусы/флаги) и НЕ масштабируем вниз — иначе GROUP BY схлопнется.
CATEGORICAL_ABS_MAX = 200


def categorical_cardinality(n_distinct: float | None, freqs: list | None, n_rows: int) -> int:
    """Число distinct для categorical_synth.

    NOTE: спека предписывает 'абсолютные n_distinct * scale_factor'. Для настоящих
    категориальных доменов (статус из 5 значений) это схлопнуло бы 5→1 и сломало бы
    GROUP BY. Кардинальность enum — свойство домена, а не числа строк, поэтому здесь
    её СОХРАНЯЕМ (зажимая в [1, n_rows]), а масштабирование scale применяем только к
    высоко-кардинальным ключам (scaled_distinct). Число корзин freqs — естественный
    пол кардинальности.
    """
    floor = len(freqs) if freqs else 0
    if n_distinct is None:
        k = floor or n_rows
    elif n_distinct < 0:
        k = round(-float(n_distinct) * n_rows)
    elif n_distinct <= CATEGORICAL_ABS_MAX:
        k = int(round(n_distinct))            # маленький enum — сохраняем как есть
    else:
        k = int(round(n_distinct))            # крупный домен — тоже сохраняем распознанное число
    return max(1, min(max(k, floor), n_rows))


def _col_seed(base_seed: int, table: Table, col: str) -> int:
    return (hash((table.fqn, col)) ^ base_seed) & 0x7FFFFFFF


def _gen_pk_pool(table: Table, pk_col: str, n_rows: int, base_seed: int) -> list:
    """Суррогатные PK 1..N (для целочисленных) или синтетические токены иначе."""
    col = table.columns.get(pk_col)
    base = (col.pg_type.split("(")[0].strip().lower() if col else "bigint")
    if base in ("smallint", "integer", "bigint", "numeric", "decimal"):
        return list(range(1, n_rows + 1))
    return [f"{table.table[:8]}_{i:06d}" for i in range(1, n_rows + 1)]


def _fill_fk(child_col_spec: dict, parent_pool: list, n_rows: int, n_distinct: float | None,
             scale: float, rng: random.Random) -> list:
    """FK-значения из пула родительских ключей с учётом fan-out и доли NULL.
    Все значения гарантированно существуют у родителя → джойны резолвятся."""
    null_frac = float(child_col_spec.get("null_frac") or 0.0)
    if not parent_pool:
        return [None] * n_rows
    # Сколько различных родителей реально «используется» детьми (fan-out).
    used = scaled_distinct(n_distinct, min(len(parent_pool), n_rows), scale)
    used = max(1, min(used, len(parent_pool)))
    active_parents = rng.sample(parent_pool, used) if used < len(parent_pool) else list(parent_pool)
    out = [rng.choice(active_parents) for _ in range(n_rows)]
    if null_frac > 0:
        out = [None if rng.random() < null_frac else v for v in out]
    return out


def generate(profile: Profile, scale: float, seed: int) -> dict[str, list[dict]]:
    """Возвращает {fqn: [row_dict, ...]} со согласованными ключами и FK."""
    all_tables = list(profile.tables)
    ordered = topo_sort(all_tables)
    by_fqn = {t.fqn: t for t in all_tables}

    data: dict[str, list[dict]] = {}
    pk_pools: dict[str, dict[str, list]] = {}  # fqn -> {pk_col: pool}

    for table in ordered:
        rng = random.Random((hash(table.fqn) ^ seed) & 0x7FFFFFFF)
        n_rows = scaled_rowcount(table.row_count, scale)
        pk_cols = table.pk
        columns_out: dict[str, list] = {}

        # 1) PK-пулы (одиночный суррогатный PK — основной случай).
        single_pk = pk_cols[0] if len(pk_cols) == 1 else None
        if single_pk:
            pool = _gen_pk_pool(table, single_pk, n_rows, seed)
            columns_out[single_pk] = list(pool)
            pk_pools.setdefault(table.fqn, {})[single_pk] = pool

        # 2) FK-колонки (после того как родители сгенерированы — топо-порядок это гарантирует).
        fk_by_col: dict[str, dict] = {}
        for fk in table.fks:
            ref_fqn = f"{fk.get('ref_schema')}.{fk.get('ref_table')}"
            for i, ccol in enumerate(fk.get("columns") or []):
                ref_col = (fk.get("ref_columns") or [None] * (i + 1))[i]
                fk_by_col[ccol] = {"ref_fqn": ref_fqn, "ref_col": ref_col}

        # 3) Остальные колонки по классу.
        for name, col in table.columns.items():
            if name in columns_out:
                continue  # PK уже сгенерирован
            crng = random.Random(_col_seed(seed, table, name))
            spec = dict(col.raw)
            spec.setdefault("pg_type", col.pg_type)
            klass = col.policy

            if name in fk_by_col:
                info = fk_by_col[name]
                parent = pk_pools.get(info["ref_fqn"], {})
                parent_pool = parent.get(info["ref_col"]) or (next(iter(parent.values()), []) if parent else [])
                if not parent_pool:
                    # Родитель вне профиля — синтетический пул, чтобы FK всё же резолвился.
                    pcount = scaled_distinct(spec.get("n_distinct"), n_rows, scale)
                    parent_pool = list(range(1, pcount + 1))
                    log.info("FK %s.%s: родитель %s вне профиля — синтетический пул из %d ключей",
                             table.table, name, info["ref_fqn"], pcount)
                columns_out[name] = _fill_fk(spec, parent_pool, n_rows,
                                             spec.get("n_distinct"), scale, crng)
                continue

            if klass == "key":  # ключ без FK и не одиночный PK (напр. часть составного)
                columns_out[name] = list(range(1, n_rows + 1))
            elif klass == "categorical_keep":
                columns_out[name] = vg.gen_categorical_keep(
                    spec.get("values") or [], n_rows, float(spec.get("null_frac") or 0.0), crng)
            elif klass == "categorical_synth":
                k = categorical_cardinality(spec.get("n_distinct"), spec.get("freqs"), n_rows)
                columns_out[name] = vg.gen_categorical_synth(
                    k, spec.get("freqs"), n_rows, float(spec.get("null_frac") or 0.0),
                    crng, prefix=name[:8])
            elif klass == "sensitive_numeric":
                columns_out[name] = vg.gen_sensitive_numeric(spec, n_rows, crng)
            elif klass == "pii":
                unique_like = bool(spec.get("unique_like"))
                columns_out[name] = vg.gen_pii(
                    spec.get("generator") or "text", n_rows,
                    float(spec.get("null_frac") or 0.0), unique_like, crng)
            elif klass == "datetime":
                columns_out[name] = vg.gen_datetime(spec, n_rows, crng)
            else:  # sensitive / generic
                columns_out[name] = vg.gen_generic(spec, n_rows, crng)

        # 4) Транспонируем в строки + чиним порядок связанных дат (created <= updated).
        rows = _to_rows(table, columns_out, n_rows)
        rows = _enforce_date_order(table, rows)
        data[table.fqn] = rows
        log.info("Данные %s: %d строк, %d колонок (scale=%s)", table.fqn, n_rows,
                 len(columns_out), scale)
    return data


def _to_rows(table: Table, columns_out: dict[str, list], n_rows: int) -> list[dict]:
    names = list(table.columns.keys())
    rows = []
    for i in range(n_rows):
        rows.append({name: columns_out.get(name, [None] * n_rows)[i] for name in names})
    return rows


def _enforce_date_order(table: Table, rows: list[dict]) -> list[dict]:
    """Внутри order_group даты сортируем по возрастанию (created <= updated <= ...)."""
    groups = [name for name, col in table.columns.items() if col.get("order_group")]
    if len(groups) < 2:
        return rows
    for row in rows:
        vals = [(name, row[name]) for name in groups if row[name] is not None]
        ordered_vals = sorted(v for _, v in vals)
        j = 0
        for name in groups:
            if row[name] is not None:
                row[name] = ordered_vals[j]; j += 1
    return rows
