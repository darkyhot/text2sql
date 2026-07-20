"""Policy — whitelist чувствительности (граница безопасности).

По умолчанию КАЖДАЯ колонка считается чувствительной и будет синтезироваться.
Реальные значения (most_common_vals/histogram_bounds) попадают в профиль ТОЛЬКО
для колонок, явно помеченных в policy как categorical_keep. Никакая авто-эвристика
не может выдать categorical_keep — это делает только человек в YAML.

Формат policy-файла (YAML):

    version: 1
    columns:
      "schema.table.col":  categorical_keep          # короткая форма
      "schema.table.name": {class: pii, generator: full_name}
      "schema.table.amt":  {class: sensitive_numeric, dist: lognormal, avg_hint: "~5e4"}
    tables:
      "schema.table":
        order_groups: [["created_at", "updated_at"]]  # даты: created <= updated
        dependencies:                                  # функциональные зависимости
          - determinant: task_subtype                  # категория-детерминант
            dependent:   task_questionary               # зависимая колонка
            order_by:    task_date                      # (опц.) представитель — свежайший
            keep_representative: true                   # хранить реальный представитель (whitelist)

Разрешённые классы в policy: categorical_keep, sensitive_numeric, pii, key,
datetime, sensitive.

Зависимость: для каждой категории determinant берётся ОДИН представитель dependent
(строка с максимальным order_by, где dependent не NULL). В синтетике та же категория
получает тот же представитель — «опросник остаётся у своего подтипа». При
keep_representative=true реальный представитель (whitelist-opt-in) кладётся в профиль;
иначе генератор ставит стабильный синтетический токен на категорию.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from agc_common import get_logger

log = get_logger("profiler.policy")

VALID_CLASSES = {
    "categorical_keep", "sensitive_numeric", "pii", "key", "datetime", "sensitive",
}
# Классы, которые НЕЛЬЗЯ вывести автоматически — только явным whitelist.
WHITELIST_ONLY = {"categorical_keep"}


class Policy:
    def __init__(self, columns: dict, tables: dict):
        self._columns = columns          # "s.t.c" -> {"class":..., extras...}
        self._tables = tables            # "s.t"   -> {"order_groups": [...]}

    @classmethod
    def load(cls, path: str | Path | None) -> "Policy":
        if not path:
            log.info("Policy-файл не задан — все колонки трактуются как чувствительные.")
            return cls({}, {})
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        columns: dict[str, dict] = {}
        for key, val in (data.get("columns") or {}).items():
            if isinstance(val, str):
                entry = {"class": val}
            elif isinstance(val, dict):
                entry = dict(val)
            else:
                raise ValueError(f"policy.columns[{key!r}] должен быть строкой или dict")
            klass = entry.get("class")
            if klass not in VALID_CLASSES:
                raise ValueError(f"policy.columns[{key!r}].class={klass!r} не из {VALID_CLASSES}")
            columns[key] = entry
        tables = dict(data.get("tables") or {})
        log.info("Policy загружена: %d правил по колонкам, %d по таблицам.",
                 len(columns), len(tables))
        return cls(columns, tables)

    def resolve(self, schema: str, table: str, column: str, proposed: str,
                proposed_gen: str | None) -> dict:
        """Возвращает финальный dict политики для колонки: {'class':..., extras}.

        Правила:
        - явное правило в policy побеждает всегда;
        - иначе берём предложенный класс, НО categorical_candidate → categorical_synth
          (реальные значения не сохраняем, синтезируем токены с сохранением
          кардинальности/долей);
        - categorical_keep недостижим без явного whitelist.
        """
        key = f"{schema}.{table}.{column}"
        if key in self._columns:
            entry = dict(self._columns[key])
            entry.setdefault("generator", proposed_gen)
            entry["source"] = "policy"
            return entry
        # Дефолт (whitelist): ничего не «кипаем».
        if proposed == "categorical_candidate":
            klass = "categorical_synth"
        elif proposed in VALID_CLASSES:
            klass = proposed
        else:
            klass = "sensitive"
        return {"class": klass, "generator": proposed_gen, "source": "auto"}

    def order_groups(self, schema: str, table: str) -> list[list[str]]:
        return list((self._tables.get(f"{schema}.{table}") or {}).get("order_groups") or [])

    def dependencies(self, schema: str, table: str) -> list[dict]:
        """Функциональные зависимости таблицы: [{determinant, dependent, order_by?, keep_representative?}]."""
        deps = (self._tables.get(f"{schema}.{table}") or {}).get("dependencies") or []
        out = []
        for d in deps:
            if not d.get("determinant") or not d.get("dependent"):
                continue
            out.append({
                "determinant": d["determinant"],
                "dependent": d["dependent"],
                "order_by": d.get("order_by"),
                "keep_representative": bool(d.get("keep_representative", True)),
            })
        return out
