"""Производные бизнес-показатели отчёта (behaviors) и единая модель мер (Measure).

Сырых колонок мало — интересные показатели ПРОИЗВОДНЫЕ: доля закрытых/успешных,
ПРОСРОЧКА (факт закрытия позже плана), СРОК отработки (дни), денежный ЭФФЕКТ
(кол-во × вес, напр. отток × ЗП). Что именно считать — решает LLM по смыслу
колонок (фокусным вызовом), код лишь строит колонки и меры. Так надёжно для
слабых моделей и без хрупких эвристик.

Measure — единица анализа для майнинга: (имя, колонка, агрегат, вид, единица).
Виды: money (деньги, sum), count (штуки), rate (доля %, mean булева), duration
(срок в днях, mean). Деньги и количество — обе ключевые (ранжируем по деньгам,
но всегда показываем и людей/штуки)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Measure:
    name: str          # идентификатор меры: «Просрочка закрытия» или имя колонки
    col: str           # колонка в df (существующая или производная)
    agg: str           # sum | mean
    kind: str          # money | count | rate | duration | value
    unit: str          # ₽ | шт | % | дн
    label: str = ""    # бизнес-подпись для показа (по умолчанию = name)
    tech: str = ""     # техническое имя колонки для «подпись (tech)»; "" если синтетическая

    def __post_init__(self):
        if not self.label:
            self.label = self.name

    def title(self) -> str:
        """«Подпись (tech)» — если есть техническое имя-источник; иначе просто подпись."""
        if self.tech and self.label.lower() != self.tech.lower():
            return f"{self.label} ({self.tech})"
        return self.label


ROW_COL = "__row"       # техническая колонка-единица (для count по строкам)


_BEHAV_SYS = (
    "Ты бизнес-аналитик. По колонкам таблицы предложи ПРОИЗВОДНЫЕ бизнес-показатели, "
    "которые интересно считать (их нет как готовых колонок). Типы:\n"
    "- rate: доля по булеву флагу (напр. доля закрытых, успешных, в работе, эскалаций). "
    'Формат: {"name":"Доля закрытых","type":"rate","flag":"is_task_closed","unit":"%"}\n'
    "- overdue: ПРОСРОЧКА — факт наступил позже плана (или плановый срок прошёл). "
    'Формат: {"name":"Просрочка закрытия","type":"overdue","plan":"plan_close_task_dttm",'
    '"fact":"fact_close_task_dttm","unit":"%"}\n'
    "- duration: СРОК в днях между двумя датами (напр. срок отработки задачи). "
    'Формат: {"name":"Срок отработки, дн","type":"duration","start":"task_create_dt",'
    '"end":"fact_close_task_dttm","unit":"дн"}\n'
    "- impact: денежный ЭФФЕКТ = количество × денежный вес (напр. отток людей × средняя ЗП "
    "= потери в деньгах). Считай ТОЛЬКО если есть и счётная колонка, и денежный вес (ЗП/"
    'сумма). Формат: {"name":"Отток в деньгах","type":"impact","qty":"outflow_qty",'
    '"weight":"avg_salary","unit":"₽"}\n'
    "Правила: бери 3-8 самых полезных показателей по СМЫСЛУ таблицы. Не выдумывай колонки — "
    "используй только те, что перечислены. Если подходящих нет для типа — пропусти его.\n"
    "ОТДЕЛЬНО: record_count — нужно ли СЧИТАТЬ САМИ ЗАПИСИ как ключевую величину. Да, если "
    "строка таблицы = бизнес-СУЩНОСТЬ (задача, событие, сделка, клиент, оттёкший) — тогда "
    "«сколько их по разрезам» важно. Формат: {\"name\":\"Количество задач\",\"id_col\":\"task_code\"|null}. "
    "Нет (null), если это ВИТРИНА разных метрик (строка = набор показателей, а не сущность).\n"
    'Верни JSON: {"behaviors":[...], "record_count": {...}|null}'
)


def build_behaviors_llm(llm, table_desc: str, df: pd.DataFrame, meta: dict[str, dict]):
    """LLM предлагает производные показатели + решает, считать ли записи как сущность.
    Возвращает (behaviors: list[dict], record_count: dict|None)."""
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append(f"- {c} | {s.dtype} | {meta.get(c, {}).get('desc', '')}")
    user = (f"Таблица: {table_desc}\nКолонки [имя | тип | описание]:\n" + "\n".join(cols)
            + "\n\nПредложи производные показатели и реши про record_count.")
    try:
        # reasoning-модели (DeepSeek) тратят бюджет на размышление — даём с запасом
        out = llm.complete_json(_BEHAV_SYS, user, max_tokens=6000, node="report_behaviors")
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: производные показатели не сгенерированы (%s)", exc)
        return [], None
    beh = [b for b in (out.get("behaviors") or []) if isinstance(b, dict) and b.get("type")]
    rec = out.get("record_count") if isinstance(out.get("record_count"), dict) else None
    if rec and not str(rec.get("name") or "").strip():
        rec = None
    return beh, rec


def record_count_measure(df: pd.DataFrame, rec: dict | None) -> Measure | None:
    """Мера «Количество <сущностей>» = COUNT записей (по колонке-единице ROW_COL). Только
    если LLM решил, что строки таблицы — бизнес-сущности (не витрина метрик)."""
    if not rec:
        return None
    df[ROW_COL] = 1
    name = str(rec.get("name") or "Количество записей").strip()
    return Measure(name, ROW_COL, "sum", "count", "", label=name)


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return all(c and c in df.columns for c in cols)


def build_derived(df: pd.DataFrame, defs: list[dict], meta: dict[str, dict]) -> list[Measure]:
    """Строит производные колонки в df и возвращает соответствующие меры.
    Пропускает определения с несуществующими/некорректными колонками (fail-safe)."""
    measures: list[Measure] = []
    used_names: set[str] = set()
    for i, d in enumerate(defs):
        t = str(d.get("type", "")).lower()
        name = str(d.get("name") or t).strip()
        unit = str(d.get("unit") or "").strip()
        try:
            if t == "rate":
                flag = d.get("flag")
                if not _has(df, flag) or not _is_boolish(df[flag]):
                    continue
                col = f"__rate_{i}"
                df[col] = _as_bool(df[flag]).astype(float)
                measures.append(Measure(name, col, "mean", "rate", unit or "%"))
            elif t == "overdue":
                plan, fact = d.get("plan"), d.get("fact")
                if not _has(df, plan, fact):
                    continue
                p = pd.to_datetime(df[plan], errors="coerce")
                f = pd.to_datetime(df[fact], errors="coerce")
                # просрочка: закрыто позже плана. (не закрыто при прошедшем плане —
                # отдельный кейс, но fact=NaT не считаем просрочкой, чтобы не шуметь)
                col = f"__overdue_{i}"
                df[col] = ((f.notna()) & (p.notna()) & (f > p)).astype(float)
                measures.append(Measure(name, col, "mean", "rate", unit or "%"))
            elif t == "duration":
                start, end = d.get("start"), d.get("end")
                if not _has(df, start, end):
                    continue
                s = pd.to_datetime(df[start], errors="coerce")
                e = pd.to_datetime(df[end], errors="coerce")
                col = f"__dur_{i}"
                days = (e - s).dt.total_seconds() / 86400.0
                df[col] = days.where(days >= 0)      # отрицательные (грязь) → NaN
                measures.append(Measure(name, col, "mean", "duration", unit or "дн"))
            elif t == "impact":
                qty, weight = d.get("qty"), d.get("weight")
                if not _has(df, qty, weight):
                    continue
                q = pd.to_numeric(df[qty], errors="coerce")
                w = pd.to_numeric(df[weight], errors="coerce")
                col = f"__impact_{i}"
                df[col] = (q * w)
                measures.append(Measure(name, col, "sum", "money", unit or "₽"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("report: производный показатель %s не построен: %s", d, exc)
            continue
        used_names.add(name)
    logger.info("report: производных показателей построено: %d (%s)",
                len(measures), [m.name for m in measures])
    return measures


def _is_boolish(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    vals = set(str(v).strip().lower() for v in s.dropna().unique()[:20])
    return bool(vals) and vals <= {"true", "false", "0", "1", "да", "нет", "t", "f", "y", "n"}


def _as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.map(lambda v: str(v).strip().lower() in ("true", "1", "да", "t", "y")).fillna(False)


import re as _re

# СИЛЬНЫЕ признаки денег в имени (однозначные). Неоднозначные (potential/потенциал/
# qty/val) сюда НЕ входят — деньги обычно `_amt` и дробные, а не «потенциал» в штуках.
_STRONG_MONEY = _re.compile(
    r"(_amt($|_)|amount|salary|_fot($|_)|оклад|зарплат|(^|_)зп($|_)|руб|_cost($|_)|"
    r"price|стоим|оборот|выруч|доход|_sum($|_)|денеж|платеж)", _re.I)
_PERC_RE = _re.compile(r"(perc($|_)|percent|_share($|_)|_rate($|_)|доля|процент)", _re.I)
_AVG_RE = _re.compile(r"(avg|average|_mean($|_)|средн)", _re.I)


def _looks_money(df: pd.DataFrame, col: str, desc: str) -> bool:
    """Деньги, если СИЛЬНЫЙ токен имени/описания ИЛИ значения дробные (у денег копейки).
    Целочисленная колонка со слабым именем деньгами НЕ считается (люди/штуки/потенциал)."""
    if _STRONG_MONEY.search(col) or _STRONG_MONEY.search(desc or ""):
        return True
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return False
    return float((s % 1 != 0).mean()) > 0.05        # >5% значений с дробной частью


def money_from_metrics(df: pd.DataFrame, roles_meta: dict[str, dict], measures: list[Measure],
                       metric_cols: list[str], labels=None) -> None:
    """Числовые колонки → меры. ВИД величины (деньги/люди/количество/процент) определяет
    LLM (labels.unit_kind) — это надёжнее регекса/копеек. Деньги (₽) — только при уверенности
    LLM; при отсутствии вердикта консервативно: деньги ТОЛЬКО по сильному денежному токену
    имени, иначе количество (без ₽). avg/среднее → агрегат mean (не суммируем средние)."""
    lab = (lambda c: labels.of(c)) if labels is not None else (lambda c: c)
    uk = (lambda c: labels.unit_kind(c)) if labels is not None else (lambda c: None)
    have = {m.col for m in measures}
    for c in metric_cols:
        if c in have or c not in df.columns:
            continue
        desc = roles_meta.get(c, {}).get("desc", "")
        kind_llm = uk(c)
        agg = "mean" if (_AVG_RE.search(c) or _AVG_RE.search(desc)) else "sum"
        # процент/доля → нормируем к 0..1 (совпадёт с rate-мерами по порогам/формату)
        if kind_llm == "percent" or (kind_llm is None and (_PERC_RE.search(c) or _PERC_RE.search(desc))):
            s = pd.to_numeric(df[c], errors="coerce")
            col = c
            if float(s.dropna().abs().max() or 0) > 1.5:      # шкала 0..100 → 0..1
                col = f"__perc_{c}"
                df[col] = s / 100.0
            measures.append(Measure(c, col, "mean", "rate", "%", label=lab(c), tech=c))
            continue
        if kind_llm == "money":
            measures.append(Measure(c, c, agg, "money", "₽", label=lab(c), tech=c))
        elif kind_llm in ("people", "count"):
            unit = "чел." if kind_llm == "people" else ""
            measures.append(Measure(c, c, agg, "count", unit, label=lab(c), tech=c))
        else:                              # нет вердикта LLM → консервативно по имени
            is_money = bool(_STRONG_MONEY.search(c) or _STRONG_MONEY.search(desc or ""))
            measures.append(Measure(c, c, agg, "money" if is_money else "count",
                                    "₽" if is_money else "", label=lab(c), tech=c))
