"""Генераторы значений по классу колонки.

Лёгкие, детерминированные (по seed), без ML-синтезаторов. numpy/Faker
опциональны — при их отсутствии работает stdlib-реализация (random/math).

Классы: categorical_keep, categorical_synth, sensitive_numeric, pii, datetime,
sensitive (generic). Ключи (key) обрабатывает key_linker.
"""
from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta

from agc_common import get_logger, stable_hash

log = get_logger("generator.values")

try:  # опционально — красивее pii; при отсутствии работает встроенный фолбэк
    from faker import Faker  # type: ignore
    _FAKER = Faker("ru_RU")
except Exception:  # noqa: BLE001
    _FAKER = None

# --- встроенные словари для pii-фолбэка (без внешних зависимостей) ---
# Значения ЯВНО синтетические (в духе «Иванов Иван Иванович», «ИНН 1234567890»),
# чтобы сразу было видно: это не реальные данные.
_FIRST_M = ["Александр", "Дмитрий", "Максим", "Сергей", "Андрей", "Алексей", "Иван", "Кирилл"]
_FIRST_F = ["Анна", "Елена", "Ольга", "Наталья", "Мария", "Татьяна", "Ирина", "Екатерина"]
_PATR_M = ["Иванович", "Петрович", "Сергеевич", "Андреевич", "Алексеевич", "Дмитриевич"]
_PATR_F = ["Ивановна", "Петровна", "Сергеевна", "Андреевна", "Алексеевна", "Дмитриевна"]
_LAST = ["Иванов", "Петров", "Смирнов", "Кузнецов", "Соколов", "Попов", "Лебедев", "Козлов"]
_STREETS = ["Ленина", "Мира", "Советская", "Гагарина", "Победы", "Садовая", "Лесная"]
_COMPANY = ["Ромашка", "Импульс", "Вектор", "Горизонт", "Альфа", "Дельта", "Пример", "Ресурс"]
_COMPANY_FORM = ["ООО", "АО", "ПАО", "ЗАО"]


def _apply_nulls(values: list, null_frac: float, rng: random.Random) -> list:
    if null_frac <= 0:
        return values
    return [None if rng.random() < null_frac else v for v in values]


# --------------------------------------------------------------------------- #
# Категориальные                                                              #
# --------------------------------------------------------------------------- #
def _weighted_all_present(labels: list, weights: list, n: int, rng: random.Random) -> list:
    """Сэмпл по весам, но КАЖДАЯ категория присутствует хотя бы раз (если n>=|labels|).
    Так все категории попадают в синтетику — важно для GROUP BY по редким значениям."""
    labels = labels or ["__empty__"]
    weights = [max(0.0, float(w)) for w in weights] or [1.0]
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]
    if n <= 0:
        return []
    if n >= len(labels):
        out = list(labels)  # по одному каждой категории
        out += rng.choices(labels, weights=weights, k=n - len(labels))
        rng.shuffle(out)
    else:
        out = rng.choices(labels, weights=weights, k=n)
    return out


def gen_categorical_keep(values: list, n: int, null_frac: float, rng: random.Random) -> list:
    """Сэмпл из реальных distinct-значений по их весам — сохраняет кардинальность и
    распределение групп; каждая категория присутствует хотя бы раз."""
    labels = [v[0] for v in values]
    weights = [v[1] for v in values]
    out = _weighted_all_present(labels, weights, n, rng)
    return _apply_nulls(out, null_frac, rng)


def gen_categorical_synth(k: int, freqs: list | None, n: int, null_frac: float,
                          rng: random.Random, prefix: str = "cat") -> list:
    """K синтетических токенов, сэмпл по весам freqs (или равномерно); каждая
    категория присутствует хотя бы раз. Реальные метки не используются."""
    k = max(1, k)
    tokens = [f"{prefix}_{i:04d}" for i in range(k)]
    if freqs:
        w = [max(0.0, float(x)) for x in freqs[:k]]
        while len(w) < k:
            w.append(min(w) if w else 1.0)
    else:
        w = [1.0] * k
    out = _weighted_all_present(tokens, w, n, rng)
    return _apply_nulls(out, null_frac, rng)


def gen_dependent_keep(determinant_values: list, value_map: dict, rng: random.Random,
                       null_frac: float = 0.0) -> list:
    """Зависимая колонка с реальными представителями: dependent = value_map[determinant].
    «Опросник» остаётся привязан к своему подтипу. Неизвестная категория → None."""
    out = [value_map.get(str(d)) if d is not None else None for d in determinant_values]
    return _apply_nulls(out, null_frac, rng)


def gen_dependent_synth(determinant_values: list, rng: random.Random,
                        prefix: str = "dep", null_frac: float = 0.0) -> list:
    """Зависимая колонка с СИНТЕТИЧЕСКИМ представителем: стабильный токен на категорию
    (одинаковый для одной категории), реальные значения не используются."""
    out = [None if d is None else f"{prefix}_{stable_hash(prefix, d) % 100000:05d}"
           for d in determinant_values]
    return _apply_nulls(out, null_frac, rng)


# --------------------------------------------------------------------------- #
# Числа                                                                       #
# --------------------------------------------------------------------------- #
def _parse_magnitude(avg_hint: str | None, default: float = 1000.0) -> float:
    if not avg_hint:
        return default
    s = str(avg_hint).lstrip("~").strip()
    try:
        return float(s)
    except ValueError:
        return default


def gen_sensitive_numeric(spec: dict, n: int, rng: random.Random) -> list:
    """Числа из указанной формы распределения. Правдоподобно для агрегатов,
    но не похоже на реальные суммы (форма и порядок — из профиля, значения — новые)."""
    dist = (spec.get("dist") or "lognormal").lower()
    scale = spec.get("scale")
    mean = _parse_magnitude(spec.get("avg_hint"), default=1000.0)
    null_frac = float(spec.get("null_frac") or 0.0)

    out: list = []
    if dist == "lognormal":
        sigma = 1.0
        mu = math.log(mean) - 0.5 * sigma * sigma if mean > 0 else 0.0
        for _ in range(n):
            out.append(math.exp(rng.gauss(mu, sigma)))
    elif dist == "normal":
        sigma = max(1.0, mean * 0.3)
        for _ in range(n):
            out.append(rng.gauss(mean, sigma))
    else:  # uniform
        lo, hi = 0.0, max(1.0, mean * 2)
        for _ in range(n):
            out.append(rng.uniform(lo, hi))

    # precision/scale и целочисленность.
    base = (spec.get("pg_type") or "").split("(")[0].strip().lower()
    is_int = base in ("smallint", "integer", "bigint")
    rounded = []
    for v in out:
        if is_int:
            rounded.append(int(round(v)))
        elif scale is not None:
            rounded.append(round(v, int(scale)))
        else:
            rounded.append(round(v, 2))
    return _apply_nulls(rounded, null_frac, rng)


# --------------------------------------------------------------------------- #
# PII                                                                         #
# --------------------------------------------------------------------------- #
def _pii_one(generator: str, rng: random.Random, seq: int) -> str:
    if _FAKER is not None:
        try:
            return _faker_value(generator, seq)
        except Exception:  # noqa: BLE001
            pass
    return _builtin_pii(generator, rng, seq)


def _faker_value(generator: str, seq: int) -> str:
    f = _FAKER
    mapping = {
        "full_name": f.name, "email": f.email, "phone": f.phone_number,
        "address": f.address, "company": f.company,
    }
    if generator in mapping:
        return str(mapping[generator]()).replace("\n", ", ")
    if generator == "inn":
        return f.numerify("##########")
    if generator == "snils":
        return f.numerify("###-###-### ##")
    if generator == "account":
        return f.numerify("#" * 20)
    if generator == "passport":
        return f.numerify("#### ######")
    return f"{generator}_{seq:06d}"


def _builtin_pii(generator: str, rng: random.Random, seq: int) -> str:
    if generator == "full_name":
        # ФИО с отчеством: «Иванов Иван Иванович» / «Иванова Анна Ивановна».
        if rng.random() < 0.5:
            return f"{rng.choice(_LAST)} {rng.choice(_FIRST_M)} {rng.choice(_PATR_M)}"
        return f"{rng.choice(_LAST)}а {rng.choice(_FIRST_F)} {rng.choice(_PATR_F)}"
    if generator == "email":
        return f"user{seq:06d}@example.test"
    if generator == "phone":
        return "+7" + "".join(str(rng.randint(0, 9)) for _ in range(10))
    if generator == "inn":
        return "".join(str(rng.randint(0, 9)) for _ in range(10))
    if generator == "snils":
        d = "".join(str(rng.randint(0, 9)) for _ in range(11))
        return f"{d[0:3]}-{d[3:6]}-{d[6:9]} {d[9:11]}"
    if generator == "account":
        return "".join(str(rng.randint(0, 9)) for _ in range(20))
    if generator == "passport":
        return "".join(str(rng.randint(0, 9)) for _ in range(4)) + " " + \
               "".join(str(rng.randint(0, 9)) for _ in range(6))
    if generator == "address":
        return f"ул. {rng.choice(_STREETS)}, д. {rng.randint(1, 120)}, кв. {rng.randint(1, 200)}"
    if generator == "company":
        return f'{rng.choice(_COMPANY_FORM)} "{rng.choice(_COMPANY)}"'
    # generic text
    return f"{generator}_{seq:06d}"


def gen_pii(generator: str, n: int, null_frac: float, unique_like: bool,
            rng: random.Random) -> list:
    out: list = []
    seen: set = set()
    seq = 0
    while len(out) < n:
        seq += 1
        val = _pii_one(generator, rng, seq)
        if unique_like:
            if val in seen:
                val = f"{val}#{seq}"  # гарантируем уникальность
            seen.add(val)
        out.append(val)
    return _apply_nulls(out, null_frac, rng)


# --------------------------------------------------------------------------- #
# Даты                                                                        #
# --------------------------------------------------------------------------- #
def _parse_date(s: str, default: date) -> date:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return default


def gen_datetime(spec: dict, n: int, rng: random.Random) -> list:
    rng_spec = spec.get("range") or {}
    start = _parse_date(rng_spec.get("start"), date(2018, 1, 1))
    end = _parse_date(rng_spec.get("end"), date(2024, 12, 31))
    span = max(1, (end - start).days)
    is_ts = "timestamp" in (spec.get("pg_type") or "").lower()
    out: list = []
    for _ in range(n):
        d = start + timedelta(days=rng.randint(0, span))
        if is_ts:
            out.append(datetime(d.year, d.month, d.day,
                                rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)))
        else:
            out.append(d)
    return _apply_nulls(out, float(spec.get("null_frac") or 0.0), rng)


def gen_generic(spec: dict, n: int, rng: random.Random) -> list:
    """sensitive (generic) — короткие псевдо-текстовые токены."""
    gen = spec.get("generator") or "val"
    out = [f"{gen}_{i:06d}" for i in range(n)]
    rng.shuffle(out)
    return _apply_nulls(out, float(spec.get("null_frac") or 0.0), rng)
