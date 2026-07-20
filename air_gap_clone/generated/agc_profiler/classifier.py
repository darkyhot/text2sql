"""Авто-эвристики классификации колонок — они только ПРЕДЛАГАЮТ класс.

Финальное решение принимает policy (см. policy.py): всё, что не одобрено явно,
синтезируется. Классификатор не может пометить колонку categorical_keep — это
делает только whitelist в policy-файле (граница безопасности).

Классы: categorical_candidate, sensitive_numeric, pii, key, datetime, sensitive.
Для pii дополнительно предлагаем имя генератора (full_name/email/phone/...).
"""
from __future__ import annotations

import re

# Регэкспы форматов значений (проверяются по сэмплу коротких полей).
_RE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_RE_PHONE = re.compile(r"^[+()\-\s\d]{7,20}$")
_RE_INN = re.compile(r"^\d{10}$|^\d{12}$")
_RE_SNILS = re.compile(r"^\d{3}-\d{3}-\d{3} \d{2}$|^\d{11}$")
_RE_ACCOUNT = re.compile(r"^\d{20}$")

# Подсказки по ИМЕНИ колонки → (класс, генератор).
_NAME_HINTS = (
    (re.compile(r"e?mail", re.I), "pii", "email"),
    (re.compile(r"phone|tel|mobile|телефон", re.I), "pii", "phone"),
    (re.compile(r"\binn\b|_inn|инн", re.I), "pii", "inn"),
    (re.compile(r"snils|снилс", re.I), "pii", "snils"),
    (re.compile(r"account|schet|счет|счёт|\bacc\b|iban|card", re.I), "pii", "account"),
    (re.compile(r"passport|паспорт", re.I), "pii", "passport"),
    (re.compile(r"fio|full_?name|fam|surname|lastname|firstname|имя|фамилия|фио", re.I), "pii", "full_name"),
    (re.compile(r"\bname\b|_name$|наимен", re.I), "pii", "full_name"),
    (re.compile(r"address|addr|адрес", re.I), "pii", "address"),
    (re.compile(r"company|org|orgname|организац|компан", re.I), "pii", "company"),
    (re.compile(r"birth|dob|рожд", re.I), "datetime", None),
)

_NUMERIC_TYPES = ("smallint", "integer", "bigint", "numeric", "decimal",
                  "real", "double precision", "money")
_DATETIME_TYPES = ("date", "timestamp", "time")
# Категория = меньше этого числа уникальных значений в сэмпле.
CATEGORICAL_MAX_DISTINCT = 200


def _base_type(pg_type: str) -> str:
    return (pg_type or "").split("(")[0].strip().lower()


def _looks_like(values: list, regex: re.Pattern) -> bool:
    """>=80% непустых сэмпл-значений матчат regex."""
    vals = [str(v) for v in values if v is not None and str(v) != ""]
    if len(vals) < 3:
        return False
    hits = sum(1 for v in vals if regex.match(v))
    return hits / len(vals) >= 0.8


def propose(
    name: str,
    pg_type: str,
    *,
    is_pk: bool = False,
    n_distinct: float | None = None,
    sample_values: list | None = None,
) -> tuple[str, str | None]:
    """Возвращает (предложенный_класс, генератор|None). FK не выводим — только PK."""
    base = _base_type(pg_type)

    if is_pk:
        return "key", None

    # Формат значений из сэмпла (только короткие поля попадают сюда со значениями).
    sv = sample_values or []
    if _looks_like(sv, _RE_EMAIL):
        return "pii", "email"
    if _looks_like(sv, _RE_ACCOUNT):
        return "pii", "account"
    if _looks_like(sv, _RE_INN):
        return "pii", "inn"
    if _looks_like(sv, _RE_SNILS):
        return "pii", "snils"

    # Подсказки по имени.
    for regex, klass, gen in _NAME_HINTS:
        if regex.search(name):
            return klass, gen

    if base in _DATETIME_TYPES:
        return "datetime", None

    # Текст с низкой кардинальностью → КАНДИДАТ в категориальные (не keep!).
    if base in ("text", "character varying", "varchar", "character", "char"):
        if n_distinct is not None and 0 < n_distinct <= CATEGORICAL_MAX_DISTINCT:
            return "categorical_candidate", None
        return "sensitive", "text"

    if base in _NUMERIC_TYPES:
        # Маленькая кардинальность у числа — тоже кандидат в категориальные (коды/флаги).
        if n_distinct is not None and 0 < n_distinct <= CATEGORICAL_MAX_DISTINCT:
            return "categorical_candidate", None
        return "sensitive_numeric", None

    if base in ("boolean",):
        return "categorical_candidate", None

    return "sensitive", "text"
