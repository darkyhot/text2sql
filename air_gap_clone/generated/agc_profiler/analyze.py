"""Анализ сэмпла в pandas: PK-гипотеза, статистика колонок, категории,
представители функциональных зависимостей.

PK нигде не объявлен в DDL — выводим гипотезу по сэмплу (минимальная уникальная
комбинация), как в исходном проекте. FK НЕ выводим: связи джойнов подбираются
позже на синтетике.
"""
from __future__ import annotations

import re
from itertools import combinations

import pandas as pd

from agc_common import get_logger

log = get_logger("profiler.analyze")

# Категория = меньше этого числа уникальных значений в сэмпле.
CATEGORICAL_MAX_DISTINCT = 200

_METRIC_RE = re.compile(
    r"(^|_)(qty|quantity|amt|amount|sum|total|cnt|count|avg|rate|ratio|pct|perc|percent|val|value)($|_)",
    re.I)
# Системные таймстемпы почти уникальны, но НЕ бизнес-ключи — откладываем в PK-поиске.
_SYS_TS_RE = re.compile(r"(dttm$|timestamp|inserted|modified|updated|_update_|^update_|load_)", re.I)


def _is_metric_name(name: str) -> bool:
    return bool(_METRIC_RE.search(name or ""))


def find_pk(df: pd.DataFrame, max_cols: int = 4) -> list[str]:
    """Минимальная уникальная комбинация на сэмпле (PK-гипотеза).

    Бизнес-ключи в приоритете: метрики и системные таймстемпы откладываем — иначе
    почти-уникальный load-таймстемп ложно становится PK. Возвращает [] если не нашли.
    ВНИМАНИЕ: это гипотеза по сэмплу — уникальность в сэмпле не доказывает PK.
    """
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


def column_stats(df: pd.DataFrame, col: str) -> dict:
    """not_null_perc, null_frac, n_distinct (в сэмпле), unique_perc."""
    n = len(df)
    if n == 0 or col not in df.columns:
        return {"not_null_perc": 0.0, "null_frac": 0.0, "n_distinct": 0, "unique_perc": 0.0}
    notna = int(df[col].notna().sum())
    n_distinct = int(df[col].dropna().nunique())
    return {
        "not_null_perc": round(notna / n * 100, 2),
        "null_frac": round(1 - notna / n, 6),
        "n_distinct": n_distinct,
        "unique_perc": round(n_distinct / n * 100, 2),
    }


def top_categories(df: pd.DataFrame, col: str, cap: int = CATEGORICAL_MAX_DISTINCT) -> list[list]:
    """Все категории в сэмпле c долями: [[value, freq], ...] по убыванию частоты.

    Доли считаются от НЕ-NULL значений (NULL-долю несёт отдельный null_frac).
    Редкие категории, не попавшие в сэмпл, теряются — это допустимо.
    """
    if col not in df.columns:
        return []
    vc = df[col].dropna().value_counts(normalize=True)
    out = [[_norm(v), round(float(f), 6)] for v, f in vc.head(cap).items()]
    return out


def dependency_representatives(df: pd.DataFrame, determinant: str, dependent: str,
                               order_by: str | None = None) -> dict[str, str]:
    """Для каждой категории determinant — ОДИН представитель dependent.

    Берём строку с самым большим order_by (если задан), где dependent не NULL.
    Так «опросник» остаётся привязан к своему подтипу задачи. Если order_by нет —
    берём первое встретившееся не-NULL значение.
    """
    if determinant not in df.columns or dependent not in df.columns:
        return {}
    sub = df[df[dependent].notna()]
    if sub.empty:
        return {}
    if order_by and order_by in sub.columns:
        sub = sub.sort_values(order_by, ascending=True, na_position="first")
        picked = sub.groupby(determinant, dropna=True)[dependent].last()
    else:
        picked = sub.groupby(determinant, dropna=True)[dependent].first()
    return {_norm(k): _norm(v) for k, v in picked.items()}


def _norm(v):
    """Значение → JSON-безопасный скаляр (строка/число/None)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float, bool)):
        return v
    return str(v)
