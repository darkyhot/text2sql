"""Профиль-линтер — главная точка аудита утечек. Делаем строгим.

Правило: если колонка НЕ класса categorical_keep, но в её профиле оказались
реальные значения (values / most_common_vals / histogram / histogram_bounds /
реальные min/max) — это ошибка, падаем с явным сообщением ДО записи profile.json.
"""
from __future__ import annotations

# Ключи, несущие реальные значения строк. Разрешены ТОЛЬКО в categorical_keep.
# value_map — представители функциональных зависимостей (реальные значения dependent);
# он тоже допустим лишь в categorical_keep (keep_representative — это whitelist-opt-in).
FORBIDDEN_REAL_VALUE_KEYS = (
    "values", "value_map", "most_common_vals", "mcv",
    "histogram", "histogram_bounds",
    "min_real", "max_real", "min", "max",
)


class ProfileLeakError(Exception):
    """Линтер нашёл реальные значения в неразрешённой колонке."""


def check_profile(profile: dict) -> None:
    """Бросает ProfileLeakError со списком нарушений. None — если чисто."""
    violations: list[str] = []
    for table in profile.get("tables", []):
        schema = table.get("schema")
        tname = table.get("table")
        for col_name, col in (table.get("columns") or {}).items():
            klass = col.get("policy")
            if klass == "categorical_keep":
                continue
            leaked = [k for k in FORBIDDEN_REAL_VALUE_KEYS
                      if k in col and col[k] not in (None, [], {}, "")]
            for k in leaked:
                violations.append(
                    f"{schema}.{tname}.{col_name}: класс '{klass}' содержит реальные "
                    f"значения в поле '{k}' (разрешено только для categorical_keep)"
                )
    if violations:
        raise ProfileLeakError(
            "Линтер профиля: обнаружены потенциальные утечки реальных значений:\n  - "
            + "\n  - ".join(violations)
        )
