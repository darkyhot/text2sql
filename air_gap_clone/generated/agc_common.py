"""Общие утилиты обеих программ: валидация идентификаторов, логирование.

Никакой бизнес-логики — только то, что нужно и профайлеру, и генератору.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sys


def stable_hash(*parts) -> int:
    """Детерминированный хэш строк в int — НЕ зависит от PYTHONHASHSEED (в отличие от
    builtin hash()). Нужен, чтобы --seed давал одинаковый результат между запусками."""
    s = "\x1f".join(str(p) for p in parts)
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")

# SQL-идентификатор: буква/underscore, далее буквы/цифры/underscore/$.
# Всё, что не проходит, — не подставляем в SQL как имя объекта (защита от инъекции).
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def validate_identifier(name: str, kind: str = "identifier") -> str:
    """Проверяет, что name — безопасный SQL-идентификатор. Иначе ValueError.

    Значения в запросы всегда идут параметрами; идентификаторы (schema/table/
    column) параметризовать нельзя, поэтому валидируем их этой функцией.
    """
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ValueError(f"Недопустимый SQL-{kind}: {name!r}")
    return name


def quote_ident(name: str) -> str:
    """Экранирует идентификатор двойными кавычками (для DDL/SQL)."""
    return '"' + name.replace('"', '""') + '"'


def qualified(schema: str, table: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}"


def get_logger(name: str = "agc") -> logging.Logger:
    """Единый логгер в stderr. Пишем: что прочитано, что засинтезировано, warning-и."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
