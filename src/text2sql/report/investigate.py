"""Режим «Расследование»: не витрина слайсов, а ДИАГНОСТИКА конкретного вопроса
(«где мы потеряли 500к человек», «откуда взялся отток», «почему упал X»).

Как работает аналитик — так и здесь:
  1. РАМКА: LLM по вопросу выбирает целевую величину (target) и режим — «изменение»
     (со знаком: потери=отрицательные, net/gross) или «величина» (отток/количество).
  2. АТРИБУЦИЯ: детерминированно раскладываем target по разрезам — сколько КАЖДЫЙ
     внёс в потери (вклад, доля), где концентрация.
  3. СПУСК по дереву (где-внутри-где): идём в самый крупный вклад, раскладываем внутри,
     до концентрации/сущностей. Пороги материальности, лимит глубины, честный вывод
     «размазано», если концентрации нет.
  4. ПОЧЕМУ: разрезы-причины (причина оттока/статус) + СЛАГАЕМЫЕ target (напр. самозанятые
     vs ГПХ) — структурный сдвиг.
  5. КТО: конкретные сущности (компании/ИНН), составляющие потери, + их ЦЕННОСТЬ (потенциал)
     для приоритета возврата.
  6. СИНТЕЗ: LLM собирает причинную цепочку + Парето действий.

Всё числовое — pandas; LLM только рамка и синтез (надёжно, дёшево)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import PATHS
from . import core, interactive, labels as labels_mod, metrics, plan, render
from .builder import (_HTML_CSS, _img_b64, _load, _load_columns, _md_block_to_html,
                      _md_inline_to_html, _meta_for, _rel_chart)
from .core import _fmt, _save
from .labels import Labels, fmt_val
from .metrics import Measure, ROW_COL

logger = logging.getLogger(__name__)

_C_LOSS, _C_GAIN, _C_NEUT = "#C0504D", "#2E8B57", "#3B7DD8"
# сущность-ИДЕНТИФИКАТОР vs ЧИТАЕМОЕ имя; разрезы-ПРИЧИНЫ/статусы (детекция по смыслу)
_ID_ENT = re.compile(r"(^inn$|_inn$|_id$|_code$|saphr|epk|ogrn|kpp)", re.I)
_NAME_ENT = re.compile(r"(name|fio|компан|client|клиент|организ|holder|наимен)", re.I)
_REASON_RE = re.compile(r"(причин|reason|повод|cause|статус|status|основан|infopovod|признак|тип)", re.I)
# НАСТОЯЩАЯ причина (не статус). Сверяем по ИМЕНИ+ПОДПИСИ, но НЕ по описанию: описание статуса
# («дата задачи по отработке ОТТОКА») содержит «отток» и давало бы ложное совпадение.
_REASON_STRONG = re.compile(r"причин|reason|повод|cause|отток|outflow", re.I)
# причина-комментарий сотрудника — СВОБОДНЫЙ ТЕКСТ (сотни значений). Обычный потолок в 40
# уникальных её выбрасывал; для причин потолок высокий, а в разложении берём топ + «прочие».
_REASON_MAX_CARD = 500
_DIM_MAX_CARD = 40
_PRIMARY_MAX_CARD = 60      # дерево/водопад/кросс-таб: больше категорий график не читается
_WHERE_MAX_CARD = 500       # табличная раскладка «Где» (топ-6 + «прочие»): ГОСБ на проде ~110
_TOP_REASONS = 12
# даты ДЕЙСТВИЙ (не временна́я ось метрики): по ним нельзя раскладывать динамику target
_ACTION_DATE_RE = re.compile(r"(задач|task|создан|created|action|действ|заявк|обращен|звонок|визит|встреч|контакт)", re.I)


def _is_reason_col(col: str, label: str = "") -> bool:
    return bool(_REASON_STRONG.search(col + " " + (label or "")))


def _card_cap(col: str, label: str = "") -> int:
    return _REASON_MAX_CARD if _is_reason_col(col, label) else _DIM_MAX_CARD


def _to_num(s: pd.Series) -> pd.Series:
    """В float. Понимает обычные числа, Decimal (Postgres/Greenplum `numeric` → dtype object)
    и строки вида «1 616,81» (запятая-разделитель, пробелы-разряды). pd.NA/мусор → NaN."""
    out = pd.to_numeric(s, errors="coerce").astype("float64")
    bad = out.isna() & s.notna()
    if bad.any():
        txt = (s[bad].astype(str)
               .str.replace(r"[\s ]", "", regex=True)
               .str.replace(",", ".", regex=False))
        out.loc[bad] = pd.to_numeric(txt, errors="coerce")
    return out


def _numeric_like(s: pd.Series, thresh: float = 0.9) -> bool:
    """Колонка по сути числовая, даже если dtype=object (Decimal) или строка с запятой."""
    if pd.api.types.is_numeric_dtype(s):
        return True
    sample = s.dropna().head(500)
    return (not sample.empty) and float(_to_num(sample).notna().mean()) >= thresh


def _is_action_date(col: str, meta: dict) -> bool:
    desc = (meta.get(col, {}) or {}).get("desc", "") or ""
    return bool(_ACTION_DATE_RE.search(col) or _ACTION_DATE_RE.search(desc))


# ---------- подготовка данных (загрузка, роли, меры) ----------
@dataclass
class Prep:
    df: pd.DataFrame
    table_desc: str
    meta: dict
    lbls: Labels
    roles: object
    measures: list[Measure]


def _prepare(db, catalog, llm, fqn, where, progress) -> Prep:
    schema, table = fqn.split(".", 1)
    progress("загружаю данные…")
    df = _load(db, schema, table, where, _load_columns(catalog, fqn))
    if df.empty:
        raise ValueError("По заданному фильтру нет данных.")
    table_desc, meta = _meta_for(catalog, fqn)
    progress("подписи и единицы измерения…")
    lbls = labels_mod.build_labels_llm(llm, table_desc, meta, list(df.columns), df)
    roles = plan.build_roles_llm(llm, table_desc, df, meta) or core.profile(df, meta)
    for c in df.columns:
        if (c not in roles.entities and c not in roles.dimensions
                and core._ENTITY_RE.search(c) and df[c].nunique(dropna=True) > 10):
            roles.entities.append(c)
    core.normalize_roles(roles, list(df.columns))
    # Greenplum/Postgres `numeric` приезжает как Decimal (dtype=object), встречаются и строки
    # «1 616,81». is_numeric_dtype такие НЕ видит → колонка молча выпадала из анализа, а sum()
    # по ней давал 0. Нормализуем один раз, до производных. Разрезы/сущности/даты не трогаем.
    skip = set(roles.entities) | set(roles.dimensions) | set(getattr(roles, "dates", []) or [])
    fixed = []
    for c in df.columns:
        if c in skip or _ID_ENT.search(c) or pd.api.types.is_numeric_dtype(df[c]):
            continue
        if _numeric_like(df[c]):
            df[c] = _to_num(df[c])
            fixed.append(c)
    if fixed:
        logger.info("investigate: приведены к числу (были Decimal/строка): %s", fixed)
    progress("показатели…")
    behav, rec = metrics.build_behaviors_llm(llm, table_desc, df, meta)
    measures = metrics.build_derived(df, behav, meta)
    metrics.money_from_metrics(df, meta, measures, roles.metrics, lbls)
    cm = metrics.record_count_measure(df, rec)
    if cm:
        measures.insert(0, cm)
    return Prep(df, table_desc, meta, lbls, roles, measures)


# ---------- рамка расследования (LLM) ----------
@dataclass
class Frame:
    target: str
    mode: str                # change | magnitude
    direction: str           # loss | gain | top
    drill_dims: list[str] = field(default_factory=list)
    why_dims: list[str] = field(default_factory=list)
    why_asked: list[str] = field(default_factory=list)   # разрезы, ЯВНО названные в «разложи …»
    components: list[str] = field(default_factory=list)
    entity: str | None = None
    id_col: str | None = None            # id-партнёр организации (ИНН) — для подписи «Название (ИНН)»
    name_col: str | None = None          # name-партнёр организации — для подписи, если ключ = ИНН
    value: str | None = None
    restated: str = ""


_FRAME_SYS = (
    "Ты аналитик-исследователь. По ВОПРОСУ пользователя определи, ЧТО объяснять и как. "
    "Верни JSON:\n"
    "{\"target\": имя числовой колонки — величина, которую объясняем (изменение/убыль/отток/"
    "количество); \"mode\": \"change\" если это ИЗМЕНЕНИЕ со знаком (потери=отрицательные) | "
    "\"magnitude\" если положительная величина (отток/кол-во); \"direction\": \"loss\" (ищем "
    "где падение/потери) | \"gain\" | \"top\" (где сосредоточено); \"drill_dims\": [разрезы по "
    "убыванию важности для дерева «где внутри где»: сегмент, территория, холдинг…]; \"why_dims\": "
    "[разрезы-ПРИЧИНЫ: причина оттока, статус…]; \"components\": [колонки-СЛАГАЕМЫЕ target, если "
    "он раскладывается на части, напр. самозанятые+ГПХ]|[]; \"entity\": колонка-сущность "
    "(компания/ИНН — конкретные виновники)|null; \"value\": колонка-ЦЕННОСТЬ для приоритета "
    "(потенциал/сумма)|null; \"restated\": суть вопроса одной фразой}\n"
    "Бери ТОЛЬКО имена из предложенных списков."
)

# вопрос про ИЗМЕНЕНИЕ/динамику → target обязан быть дельта-метрикой со знаком (не уровнем)
_CHANGE_Q = re.compile(r"сниж|паден|отток|динамик|год.?к.?году|yoy|измен|прирост|убыл|сокращ|\bрост", re.I)
# мусорные значения-бакеты: как «драйвер причины» бесполезны
_NOISE_VAL = re.compile(r"^\s*(проч|не\s*указан|нет\s*данных|н/?д|unknown|other|пусто|основн|итого|всего|—|-|\(?пуст)", re.I)


def _frame(llm, question, prep: Prep) -> Frame:
    df = prep.df
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    lines = []
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        neg = "со знаком(есть отриц.)" if (s.min() is not None and s.min() < 0) else "положит."
        lines.append(f"- {c}: {prep.lbls.of(c)} | {neg} | {prep.meta.get(c, {}).get('desc', '')[:60]}")
    dims = "\n".join(f"- {d}: {prep.lbls.of(d)}" for d in prep.roles.dimensions)
    ents = "\n".join(f"- {e}: {prep.lbls.of(e)}" for e in prep.roles.entities)
    user = (f"Вопрос: {question}\n\nЧисловые колонки:\n" + "\n".join(lines)
            + f"\n\nРазрезы:\n{dims}\n\nСущности:\n{ents}")
    try:
        out = llm.complete_json(_FRAME_SYS, user, max_tokens=4000, node="investigate_frame")
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: рамка не построена (%s)", exc)
        out = {}
    numset, dimset = set(num_cols), set(prep.roles.dimensions)
    entset = set(prep.roles.entities)
    # B4: при промахе LLM берём НЕ произвольную первую числовую, а самую «мерную»:
    # исключаем id-подобные (inn/_id/code…), среди остальных — с макс. |сумма| (материальность).
    measures = [c for c in num_cols if not _ID_ENT.search(c)] or num_cols
    if out.get("target") in numset:
        target = out.get("target")
    elif measures:
        target = max(measures, key=lambda c: abs(float(pd.to_numeric(df[c], errors="coerce").sum())))
    else:
        target = None
    if target is None:
        raise ValueError("Не найдено числовой величины для расследования.")
    # ГАРД: вопрос про ИЗМЕНЕНИЕ/снижение ⇒ target обязан быть дельта-метрикой со знаком,
    # а не положительным уровнем (потенциал/остаток). Ловит дрейф LLM на level-колонки.
    if _CHANGE_Q.search(question):
        signed = [c for c in measures
                  if (pd.to_numeric(df[c], errors="coerce").min() or 0) < 0]
        deltas = [c for c in signed if _DELTA_RE.search(c)
                  or _DELTA_RE.search(prep.meta.get(c, {}).get("desc", "") or "")]
        pool = deltas or signed
        cur_min = pd.to_numeric(df[target], errors="coerce").min()
        if pool and not (cur_min is not None and cur_min < 0):     # текущий target — уровень
            target = max(pool, key=lambda c: abs(float(pd.to_numeric(df[c], errors="coerce").sum())))
    s = pd.to_numeric(df[target], errors="coerce")
    mode = out.get("mode") if out.get("mode") in ("change", "magnitude") else \
        ("change" if (s.min() is not None and s.min() < 0) else "magnitude")
    if _CHANGE_Q.search(question) and s.min() is not None and s.min() < 0:
        mode = "change"                       # target со знаком + вопрос про изменение ⇒ режим изменения
    # СУЩНОСТЬ-ОРГАНИЗАЦИЯ: у неё есть ИМЯ (company_name) и ИД (ИНН). Подпись всегда
    # «Название (ИНН)», а КЛЮЧУЕМ по самой ГРАНУЛЯРНОЙ колонке — обычно ИНН (300 организаций)
    # точнее названия (25 в синтетике/грубый пул), иначе теряем разбор по организациям.
    ents = list(prep.roles.entities)
    chosen = out.get("entity") if out.get("entity") in entset else None
    name_ents = [e for e in ents if _NAME_ENT.search(e) and not _ID_ENT.search(e)]
    id_ents = [e for e in ents if _ID_ENT.search(e)]
    name_col = name_ents[0] if name_ents else None
    if id_ents:
        id_col = next((c for c in id_ents if "inn" in c.lower()), id_ents[0])
    elif name_col:                        # ИНН не в сущностях — ищем среди всех колонок
        cands = [c for c in df.columns if _ID_ENT.search(c)
                 and df[c].nunique(dropna=True) >= df[name_col].nunique(dropna=True)]
        id_col = next((c for c in cands if "inn" in c.lower()), cands[0] if cands else None)
    else:
        id_col = None
    entity = chosen or (name_ents[0] if name_ents else (ents[0] if ents else None))
    # если сущность — колонка организации (имя/ИНН), ключуем по более гранулярной из пары
    if entity and (entity in (name_col, id_col) or _NAME_ENT.search(entity) or _ID_ENT.search(entity)):
        org_cols = [c for c in (name_col, id_col) if c and c in df.columns]
        if org_cols:
            entity = max(org_cols, key=lambda c: df[c].nunique(dropna=True))

    # ПОЧЕМУ-разрезы. LLM-выбор фильтруется по dimset, но профайлер часто кладёт причину/статус
    # во flags (не в dims) — тогда список пуст, и разрезы добавляет детерминированный проход по
    # колонкам (в порядке таблицы!). Даты не проходят: у них уникальных > 40.
    llm_why = [d for d in (out.get("why_dims") or []) if d in dimset]
    why = list(llm_why)
    for c in df.columns:
        if c in why or pd.api.types.is_numeric_dtype(df[c]):
            continue
        desc = prep.meta.get(c, {}).get("desc", "")
        if not (_REASON_RE.search(c) or _REASON_RE.search(desc)):
            continue
        # причина-комментарий = свободный текст: высокий потолок (иначе выпадает), топ берём ниже
        if 2 <= df[c].nunique(dropna=True) <= _card_cap(c, prep.lbls.of(c)):
            why.append(c)

    # что просят разложить: ТОЛЬКО объект команды «разложи …» (окно ~80 симв.). Весь промт брать
    # нельзя — глоссарий колонок содержит и «статус…», и «причина…». Сверяем с ПОДПИСЬЮ колонки
    # (не с описанием: описание статуса тоже содержит «оттока» и даёт ложное совпадение).
    m = re.search(r"разлож\w*(.{0,80})", question, re.I)
    obj = (m.group(1) if m else "").lower()

    def _asked(c: str) -> bool:
        if not obj:
            return False
        toks = re.findall(r"[а-яё]{5,}", prep.lbls.of(c).lower())
        return any(t[:5] in obj for t in toks)

    def _is_reason(c: str) -> bool:            # настоящая ПРИЧИНА (по имени/подписи, не по описанию)
        return _is_reason_col(c, prep.lbls.of(c))

    asked = [c for c in why if _asked(c)]
    # приоритет: явно запрошенный > настоящая причина > статус (порядок колонок таблицы — последний)
    why = sorted(why, key=lambda c: (_asked(c), _is_reason(c)), reverse=True)
    logger.info(
        "investigate FRAME: dims=%s | why от LLM=%s (после dimset=%s) | кандидаты-ПОЧЕМУ=%s | "
        "объект «разложи»=%r | asked=%s | причины=%s | ИТОГ why_dims=%s",
        sorted(dimset), out.get("why_dims"), llm_why, why,
        obj[:70], asked, [c for c in why if _is_reason(c)], why[:3])

    # ЦЕННОСТЬ (потенциал) для приоритета возврата. LLM врёт с именами колонок, поэтому если она
    # не дала валидную — подхватываем детерминированно, когда вопрос про потенциал.
    value = out.get("value") if out.get("value") in numset else None
    if value is None and re.search(r"потенциал|potential", question, re.I):
        vc = [c for c in measures if c != target
              and re.search(r"потенциал|potential", c + " " + prep.lbls.of(c), re.I)]
        pref = [c for c in vc if _year_of(c) == 26] or vc      # свежий период важнее
        if pref:
            value = max(pref, key=lambda c: abs(float(pd.to_numeric(df[c], errors="coerce").sum())))
    logger.info("investigate FRAME: target=%s | mode=%s | сущность=%s | ЦЕННОСТЬ(потенциал)=%s",
                target, mode, entity, value)

    return Frame(
        target=target, mode=mode,
        direction=out.get("direction") if out.get("direction") in ("loss", "gain", "top")
        else ("loss" if mode == "change" else "top"),
        drill_dims=[d for d in (out.get("drill_dims") or []) if d in dimset] or list(prep.roles.dimensions),
        why_dims=why[:3], why_asked=asked,
        components=[c for c in (out.get("components") or []) if c in numset and c != target],
        entity=entity, id_col=id_col, name_col=name_col,
        value=value,
        restated=str(out.get("restated") or question).strip(),
    )


# ---------- методология: «книги портфеля» (что входит/не входит в метрику) ----------
_BOOKS_SYS = (
    "Ты выделяешь «КНИГИ ПОРТФЕЛЯ» для анализа год-к-году по ОПИСАНИЯМ числовых колонок. "
    "Книга — показатель с фактами за ДВА года (y25=2025, y26=2026) и/или колонкой-дельтой (delta_col). "
    "role: 'headline' — ОСНОВНАЯ метрика портфеля (напр. ФЛ доля рынка, ФЛ КПЭ); "
    "'excluded' — категория, ИСКЛЮЧЁННАЯ из основной метрики доли (напр. ГПХ, самозанятые); "
    "'other' — прочее (пропусти). Значения y25/y26/delta_col — СТРОГО имена колонок из списка "
    "(или null). note: одна короткая фраза про методологию — что НЕ входит в долю и почему это "
    "важно для вывода. Верни JSON: {\"books\":[{\"label\":..,\"y25\":..,\"y26\":..,"
    "\"delta_col\":..,\"role\":..}],\"note\":..}"
)


@dataclass
class Book:
    label: str
    role: str            # headline | excluded
    v25: float
    v26: float
    delta: float


# --- разметка колонок в «книги» (единственное место, где нужна семантика таблицы) ---
_YEAR_TOK = re.compile(r"(?<!\d)(2025|2026|25|26)(?!\d)")
_DELTA_RE = re.compile(r"diff|delta|дельт|razn|разн|_izm|измен", re.I)
_EXCL_RE = re.compile(r"гпх|gpx|gph|самозанят|self.?empl|self.?emp", re.I)


def _year_of(col: str) -> int | None:
    m = _YEAR_TOK.search(col)
    return None if not m else (26 if m.group(1) in ("26", "2026") else 25)


def _book_label(col: str, desc: str) -> str:
    t = (col + " " + (desc or "")).lower()
    if re.search(r"гпх|gpx|gph", t):
        return "ГПХ"
    if re.search(r"самозанят|self.?empl", t):
        return "Самозанятые"
    if re.search(r"кпэ|unic|уник|kpi|кпи", t):
        return "ФЛ КПЭ"
    if "дол" in t or "market" in t or "share" in t:
        return "ФЛ доля"
    return re.sub(r"[_\s]+", " ", _YEAR_TOK.sub("", col)).strip() or col


def _books_specs_llm(df: pd.DataFrame, prep: Prep, llm, num_cols: list[str]) -> tuple[list[dict], str]:
    """LLM размечает числовые колонки в книги. Возвращает (specs, note) — сырьё для _books_compute.
    Обобщается на любую таблицу: семантика («что не входит в долю») берётся из описаний колонок."""
    listing = "\n".join(f"- {c}: {prep.meta.get(c, {}).get('desc', '') or prep.lbls.of(c)}"
                        for c in num_cols)
    try:
        out = llm.complete_json(_BOOKS_SYS, f"Таблица: {prep.table_desc}\n\nЧисловые колонки:\n{listing}",
                                max_tokens=4000, node="investigate_books")
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: LLM-разметка книг не удалась (%s)", exc)
        return [], ""
    specs = [{"label": str(b.get("label") or b.get("y26") or b.get("delta_col")),
              "role": b["role"], "y25": b.get("y25"), "y26": b.get("y26"),
              "delta_col": b.get("delta_col")}
             for b in (out.get("books") or [])
             if isinstance(b, dict) and b.get("role") in ("headline", "excluded")]
    return specs, str(out.get("note") or "").strip()


def _books_specs_heuristic(df: pd.DataFrame, prep: Prep, num_cols: list[str]) -> tuple[list[dict], str]:
    """Детерминированный фолбэк без LLM: пары колонок *_25/*_26 → книги; роль excluded по
    сигналам ГПХ/самозанятые в имени/описании. Ловит методологию, когда LLM недоступен/промолчал."""
    pairs: dict[str, dict[int, str]] = {}
    for c in num_cols:
        if _DELTA_RE.search(c):
            continue
        y = _year_of(c)
        if y is None:
            continue
        pairs.setdefault(_YEAR_TOK.sub("¤", c, count=1), {})[y] = c
    heads, excl = [], []
    for d in pairs.values():
        if 25 not in d or 26 not in d:
            continue
        col = d[26]
        desc = prep.meta.get(col, {}).get("desc", "") or prep.meta.get(d[25], {}).get("desc", "")
        spec = {"label": _book_label(col, desc), "y25": d[25], "y26": d[26], "delta_col": None}
        if _EXCL_RE.search(col + " " + (desc or "")):
            spec["role"] = "excluded"; excl.append(spec)
        else:
            spec["role"] = "headline"; heads.append(spec)
    note = ("Внимание: ГПХ и самозанятые не входят в ФЛ долю — их отток нельзя смешивать с долей."
            if excl else "")
    return heads + excl, note


def _valid_specs(specs: list[dict], df: pd.DataFrame) -> list[dict]:
    """Оставляем только книги с существующими колонками и вычислимой Δ."""
    return [s for s in specs
            if (s.get("y25") in df.columns and s.get("y26") in df.columns)
            or s.get("delta_col") in df.columns]


def _specs_ok(specs: list[dict]) -> bool:
    heads = [s for s in specs if s["role"] == "headline"]
    excl = [s for s in specs if s["role"] == "excluded"]
    return bool(heads) and (len(heads) >= 2 or bool(excl))


def _portfolio_books(df: pd.DataFrame, prep: Prep, llm) -> dict | None:
    """Методологический слой: делим числа на «книги портфеля» — основные метрики (доля/КПЭ)
    и ИСКЛЮЧЁННЫЕ из них категории (ГПХ/самозанятые). Разметку даёт LLM (обобщается на любую
    таблицу), при осечке — детерминированный эвристический фолбэк. Дальше ВСЁ детерминированно
    (_books_compute): Δ год-к-году по книге + Δ по сегментам. Ключ: метрики с разными
    знаменателями (отток ГПХ не входит в долю) смешивать нельзя."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                and not _ID_ENT.search(c)]
    if len(num_cols) < 4:
        return None
    specs, note = _books_specs_llm(df, prep, llm, num_cols)
    specs = _valid_specs(specs, df)
    if not _specs_ok(specs):                          # LLM пусто/невалидно → детерминированный фолбэк
        specs, note = _books_specs_heuristic(df, prep, num_cols)
        specs = _valid_specs(specs, df)
    return _books_compute(df, prep, specs, note)


def _books_compute(df: pd.DataFrame, prep: Prep, specs: list[dict], note: str) -> dict | None:
    """Чистая арифметика поверх разметки: суммы 25/26, Δ по книге, Δ по сегментам. Без LLM."""
    if not _specs_ok(specs):                          # нет методологии — блок не строим
        return None
    books: list[Book] = []
    for s in specs:
        y25, y26, dc = s.get("y25"), s.get("y26"), s.get("delta_col")
        v25 = float(pd.to_numeric(df[y25], errors="coerce").sum()) if y25 in df.columns else float("nan")
        v26 = float(pd.to_numeric(df[y26], errors="coerce").sum()) if y26 in df.columns else float("nan")
        if dc in df.columns:
            delta = float(pd.to_numeric(df[dc], errors="coerce").sum())
        elif v25 == v25 and v26 == v26:
            delta = v26 - v25
        else:
            continue
        books.append(Book(label=s["label"], role=s["role"], v25=v25, v26=v26, delta=delta))
    heads = [b for b in books if b.role == "headline"]
    excl = [b for b in books if b.role == "excluded"]
    if not heads or not (len(heads) >= 2 or excl):
        return None
    spec_by_label = {s["label"]: s for s in specs}
    # Δ по сегментам для headline-книг (за счёт чего изменение по каждой метрике)
    seg = next((d for d in prep.roles.dimensions if re.search(r"segment|сегмент", d, re.I)
                and d in df.columns), None)
    if seg is None:
        seg = next((d for d in prep.roles.dimensions if d in df.columns
                    and 2 <= df[d].nunique(dropna=True) <= 12), None)
    seg_series: dict = {}
    if seg:
        for b in heads:
            s = spec_by_label.get(b.label, {})
            y25, y26, dc = s.get("y25"), s.get("y26"), s.get("delta_col")
            if dc in df.columns:
                sr = pd.to_numeric(df[dc], errors="coerce").groupby(df[seg]).sum()
            elif y25 in df.columns and y26 in df.columns:
                sr = (pd.to_numeric(df[y26], errors="coerce").groupby(df[seg]).sum()
                      - pd.to_numeric(df[y25], errors="coerce").groupby(df[seg]).sum())
            else:
                continue
            seg_series[b.label] = sr.sort_values()
    return {"books": books, "note": note, "seg": seg, "seg_series": seg_series, "specs": specs,
            "head_sum": sum(b.delta for b in heads), "excl_sum": sum(b.delta for b in excl)}


# ---------- декомпозиция вклада ----------
@dataclass
class Contrib:
    dim: str
    table: pd.DataFrame        # index=value, cols: contrib, n, share
    total: float               # знаковый итог узла (net) или величина
    loss: float                # сумма отрицательных вкладов (для change), иначе = total
    top_share: float           # доля крупнейшего вклада в |loss|
    n_for_half: int            # сколько значений даёт 50% |loss|


def _decompose(df: pd.DataFrame, target: str, dim: str, mode: str,
               ref: float | None = None, side: str = "auto") -> Contrib:
    """ref — знаменатель для ДОЛИ (обычно общие потери всей таблицы), чтобы «доля лидера»
    была сопоставима между уровнями («49% ВСЕХ потерь»), а не 100% локально.
    side — какую сторону атрибутировать: 'loss' (отрицательные) | 'gain' | 'auto' (доминирующую).
    Важно: при вопросе про ПОТЕРИ берём потери, даже если net положительный."""
    s = pd.to_numeric(df[target], errors="coerce")
    g = s.groupby(df[dim]).sum()
    n = df.groupby(dim)[target].size()
    total = float(g.sum())
    if mode == "change":
        neg, pos = float(g[g < 0].sum()), float(g[g > 0].sum())
        want_loss = side == "loss" or (side == "auto" and abs(neg) >= abs(pos) and neg < 0)
        if want_loss:
            loss, asc = (neg if neg < 0 else -1e-9), True    # потери: самые отрицательные вперёд
        else:
            loss, asc = (pos if pos > 0 else 1e-9), False    # прирост: самые крупные вперёд
        contrib = g.sort_values(ascending=asc)
    else:
        loss, asc = total, False
        contrib = g.sort_values(ascending=False)
    ref_denom = abs(ref) if ref else (abs(loss) or 1.0)   # доля — от общей стороны
    local_denom = abs(loss) or 1.0                        # концентрация — внутри узла
    share = (contrib / (ref if ref else loss) * 100).clip(lower=0)
    tbl = pd.DataFrame({"contrib": contrib, "n": n.reindex(contrib.index), "share": share})
    cum = tbl["contrib"].abs().cumsum() / local_denom
    n_half = int((cum <= 0.5).sum()) + 1
    top_share = min(100.0, float(abs(tbl["contrib"].iloc[0]) / ref_denom * 100))
    return Contrib(dim, tbl, total, loss, top_share, n_half)


# ---------- спуск по дереву ----------
@dataclass
class Level:
    dim: str
    contrib: Contrib
    followed: object
    chart: str | None
    branch_mag: float = 0.0      # валовые потери/прирост ВЕТКИ (в которую спускаемся)
    parent_mag: float = 0.0      # валовые потери/прирост РОДИТЕЛЯ (текущего уровня)


def _branch_mag(x, frame: Frame, side: str) -> float:
    """Валовая величина исследуемой стороны в подмножестве (для честной доли ветки:
    подмножество ≤ родителя, так доля ребёнка не превышает родителя)."""
    v = pd.to_numeric(x[frame.target], errors="coerce")
    if frame.mode != "change":
        return abs(float(v.sum()))
    if side == "gain":
        return abs(float(v[v > 0].sum()))
    return abs(float(v[v < 0].sum()))            # loss/auto → потери


def _drill(df, frame: Frame, assets, lbls, ref, *, max_depth=4, min_rows=30,
           min_top_share=12.0, side="auto") -> list[Level]:
    """B2: рекурсивный спуск «горячего пути» — на каждом уровне берём разрез с макс.
    концентрацией вклада и уходим в его крупнейшее значение (сегмент→территория→холдинг→…),
    пока хватает разрезов/объёма и путь остаётся сконцентрированным (материальность)."""
    # разрезы для дерева: не сущности (они — «кто»), достаточно значений и объёма
    ent_concepts = {core._concept(e) for e in [frame.entity] if frame.entity}
    cand = [d for d in frame.drill_dims
            if d in df.columns and core._concept(d) not in ent_concepts
            and 2 <= df[d].nunique(dropna=True) <= 60]
    levels: list[Level] = []
    sub = df
    for _ in range(max_depth):
        cand = [d for d in cand if sub[d].nunique(dropna=True) >= 2]
        if not cand or len(sub) < min_rows:
            break
        # выбираем разрез с максимальной концентрацией вклада (детерминированно)
        decs = [_decompose(sub, frame.target, d, frame.mode, ref, side) for d in cand]
        decs = [c for c in decs if np.isfinite(c.top_share)]
        if not decs:
            break
        best = max(decs, key=lambda c: c.top_share)
        chart = _contrib_chart(best, frame, assets, lbls, path_len=len(levels))
        follow = best.table.index[0]                 # крупнейший вклад
        parent_mag = _branch_mag(sub, frame, side)
        branch = sub[sub[best.dim].astype(str) == str(follow)]
        levels.append(Level(best.dim, best, follow, chart,
                            _branch_mag(branch, frame, side), parent_mag))
        # стоп, если потери «размазаны» (нет доминирующего) — глубже не идём
        if best.top_share < min_top_share:
            break
        sub = branch
        cand = [d for d in cand if d != best.dim]
    return levels


def _drill_facts(levels: list[Level], lbls, unit, side: str) -> list[dict]:
    """Компактные факты горячего пути: разрез, значение, ВАЛОВЫЕ потери ветки и доля.
    Доля считается от ВАЛОВЫХ потерь родителя (а не от net-сумм) — тогда ветка ⊆ родителя
    и доля ребёнка не превышает родителя (net-суммы с взаимозачётом сиблингов давали >100%)."""
    out = []
    root = abs(float(levels[0].parent_mag)) if levels else 0.0
    sign = -1.0 if side != "gain" else 1.0
    for lv in levels:
        tbl = lv.contrib.table
        row = tbl.loc[lv.followed] if lv.followed in tbl.index else None
        n = int(row["n"]) if row is not None and pd.notna(row["n"]) else 0
        mag = abs(float(lv.branch_mag))
        share_parent = min(100.0, mag / (abs(float(lv.parent_mag)) or 1.0) * 100)
        share_all = min(100.0, mag / (root or 1.0) * 100)
        out.append({"dim": lv.dim, "разрез": lbls.of(lv.dim), "значение": fmt_val(lv.followed),
                    "вклад": _fmt_u(sign * mag, unit), "доля_родителя_%": round(share_parent, 1),
                    "доля_всех_%": round(share_all, 1), "строк": n, "chart": lv.chart})
    return out


# ---------- «почему» и «кто» ----------
def _why(df, frame: Frame, assets, lbls, ref, side="auto") -> list[tuple[str, dict, str | None]]:
    """«Почему» через LIFT к базе: причина ЗНАЧИМА, если её доля в ПОТЕРЯХ заметно выше
    её доли в СТРОКАХ («причина X даёт 45% потерь при 18% строк — ×2.5»). Это, в отличие
    от «преобладает Иное (80%)», отделяет настоящий драйвер от фонового значения."""
    out = []
    s = _to_num(df[frame.target])                 # pd.NA → NaN, Decimal/строка → float
    total_rows = len(df) or 1
    cands = [d for d in frame.why_dims[:3]
             if d in df.columns
             and 2 <= df[d].nunique(dropna=True) <= _card_cap(d, lbls.of(d))]

    def _has_meaningful(d: str) -> bool:      # есть ли осмысленный (не мусорный) значимый драйвер
        g = _side_series(s.groupby(df[d]).sum(), side)
        tl = abs(float(g.sum())) or 1.0
        return any(abs(float(cv)) / tl >= 0.05 and not _NOISE_VAL.search(str(v)) for v, cv in g.items())

    # ВЫБОР разрезов: явно запрошенные в «разложи …» — показываем их (даже с мусорным драйвером);
    # иначе — только те, где есть ОСМЫСЛЕННЫЙ драйвер (не тащим статус с одним «Прочее»).
    asked = [d for d in cands if d in set(frame.why_asked)]
    meaningful_dims = [d for d in cands if _has_meaningful(d)]
    dims = asked or meaningful_dims[:2] or cands[:1]
    logger.info("investigate ПОЧЕМУ: кандидаты=%s | asked=%s | с осмысленным драйвером=%s → показываем=%s",
                cands, asked, meaningful_dims, dims)

    for d in dims:
        c = _decompose(df, frame.target, d, frame.mode, ref, side)      # бар вклада
        g_loss = _side_series(s.groupby(df[d]).sum(), side)
        total_loss = abs(float(g_loss.sum())) or 1.0
        rows = df.groupby(d).size()
        recs = []
        for val, contrib in g_loss.items():
            ls = abs(float(contrib)) / total_loss
            bs = float(rows.get(val, 0)) / total_rows
            recs.append((val, ls, bs, (ls / bs) if bs > 0 else 0.0))
        material = [r for r in recs if r[1] >= 0.05] or recs
        # РАЗЛОЖЕНИЕ: все значения по частоте + их Δ-вклад в метрику («как влияют») — СТРОИМ ВСЕГДА,
        # чтобы явно запрошенный разрез (напр. статус) не выпал, даже если драйвер — «мусорный» бакет.
        gall = s.groupby(df[d]).sum()
        cnt = df.groupby(d).size().sort_values(ascending=False)
        top = list(cnt.index[:_TOP_REASONS])
        brk = [(fmt_val(v), int(cnt[v]), float(gall.get(v, 0.0) or 0.0)) for v in top]
        tail = list(cnt.index[_TOP_REASONS:])          # свободный текст: хвост не печатаем построчно
        if tail:
            brk.append((f"Прочие ({len(tail)} значений)", int(cnt[tail].sum()),
                        float(gall.reindex(tail).fillna(0).sum())))
        info = {"breakdown": brk}
        # заголовок-драйвер (lift) — только при ОСМЫСЛЕННОЙ причине (не Прочее/Не указано/пусто)
        meaningful = [r for r in material if not _NOISE_VAL.search(str(r[0]))]
        if meaningful:
            best = max(meaningful, key=lambda r: r[3])
            info.update(value=fmt_val(best[0]), loss_share=round(best[1] * 100, 1),
                        base_share=round(best[2] * 100, 1), lift=round(best[3], 1))
        out.append((d, info, _contrib_chart(c, frame, assets, lbls, tag="why")))
    return out


def _components(df, frame: Frame, lbls) -> list[tuple[str, float]]:
    """Структурный разбор target на слагаемые (напр. самозанятые vs ГПХ)."""
    res = []
    for c in frame.components:
        v = float(pd.to_numeric(df[c], errors="coerce").sum())
        res.append((lbls.of(c), v))
    return res


_COVER_OK = 0.8          # «примерно равнозначный» потенциал: покрывает ≥80% снижения


def _cover_txt(c) -> str:
    """Покрытие как ×N + пометка, покрывает ли потенциал снижение («и/или больше»).
    Выше ×999 точность бессмысленна (и обычно значит разные единицы) — показываем «>×999»."""
    if c is None or pd.isna(c):
        return "—"
    c = float(c)
    mark = "✓" if c >= _COVER_OK else ""
    if c > 999:
        return f">×999 {mark}".strip()
    return (f"×{c:.0f} {mark}" if c >= 10 else f"×{c:.1f} {mark}").strip()


def _recovery(df, frame: Frame, dim: str) -> list[tuple] | None:
    """Сопоставление «снижение vs потенциал» В РАЗРЕЗЕ `dim` (сегмент/ГОСБ): по каждому срезу —
    сколько просевших клиентов, у скольких потенциал покрывает снижение (≥80%), и итоговое
    покрытие среза. Отвечает на «сопоставь … имеющих потенциал ≈ сумме снижения и/или больше»."""
    ent, val = frame.entity, frame.value
    if not (ent and val) or ent not in df.columns or val not in df.columns or dim not in df.columns:
        return None
    s, v = _to_num(df[frame.target]), _to_num(df[val])
    lost = s.groupby(df[ent]).sum()
    lost = lost[lost < 0]                       # только просевшие клиенты
    if lost.empty:
        return None
    pot = v.groupby(df[ent]).sum(min_count=1).reindex(lost.index)
    # срез клиента = значение dim с наибольшим вкладом в его снижение (клиент может быть в нескольких)
    sub = df[df[ent].isin(set(lost.index))]
    gr = _to_num(sub[frame.target]).groupby([sub[ent], sub[dim]]).sum()
    pick = gr.groupby(level=0).idxmin()
    seg = pd.Series({e: (k[1] if isinstance(k, tuple) else None) for e, k in pick.items()})

    cov = (pot / lost.abs())
    d = pd.DataFrame({"seg": seg.reindex(lost.index), "lost": lost, "pot": pot, "cov": cov}).dropna(subset=["seg"])
    total_loss = abs(float(lost.sum())) or 1.0
    rows = []
    for sv, g in d.groupby("seg"):
        n, ok = len(g), int((g["cov"] >= _COVER_OK).sum())
        tl, tp = float(g["lost"].sum()), float(g["pot"].sum(min_count=1) or 0.0)
        if n < 3 and abs(tl) / total_loss < 0.01:   # шумовой срез (1-2 клиента, копейки) — не показываем
            continue
        rows.append((fmt_val(sv), n, ok, tl, tp, (tp / abs(tl)) if tl else float("nan")))
    rows.sort(key=lambda r: r[3])               # самые просевшие срезы вперёд
    return rows[:8] or None


# ---------- сводная таблица по явному запросу («построй сводную с колонками …») ----------
_PIVOT_RE = re.compile(
    r"свод\w*\s*(?:таблиц\w*|диаграмм\w*|отч[её]т\w*|)"          # сводная таблица/диаграмма/…
    r"[^.]*?\b(?:колон\w*|пол[яей]\w*|разрез\w*|измерени\w*)"    # …с колонками / с полями / по разрезам
    r"\s*[:\-—]?\s*(?P<cols>[^.]+)", re.I)
_PIVOT_TOP_DEFAULT = 10        # с сущностью (компанией) веток тысячи → топ-N на каждом уровне
# Предел строк сводной. Замерено в браузере (Tabulator, вложенные группы): раскрытие
# 500 строк ≈ 2.4с, 1000 ≈ 9с, 2000+ — зависание. Узкое место — рендер раскрытых строк,
# кэш подытогов не помогает. Поэтому единственный рычаг — не плодить строки.
_PIVOT_MAX_LEAVES = 500


def _level_caps(n_dims: int, topn: int) -> list[int]:
    """Топ-N на каждом уровне так, чтобы всего строк было ≤ _PIVOT_MAX_LEAVES.
    «Топ-10 везде» при 4 разрезах даёт 11⁴ ≈ 14 тыс. строк и вешает браузер. Ужимаем САМЫЙ
    БОЛЬШОЙ уровень (при равенстве — более глубокий): бюджет распределяется равномерно, и ни
    один разрез не схлопывается в 2 значения, пока остальные держат 10."""
    caps = [max(2, topn)] * max(1, n_dims)

    def est(cs: list[int]) -> int:
        p = 1
        for c in cs:
            p *= c + 1                     # +1 — строка «прочие» на каждом уровне
        return p

    while est(caps) > _PIVOT_MAX_LEAVES:
        j = max(range(len(caps)), key=lambda k: (caps[k], k))   # самый большой, при равенстве — глубже
        if caps[j] <= 2:
            break                          # дальше ужимать некуда
        caps[j] -= 1
    return caps
_PIVOT_TOP_RE = re.compile(r"топ[\s\-–]?(\d{1,3})", re.I)
# слова, не различающие книги («ФЛ доля рынка» vs «ФЛ КПЭ»): по ним матчить нельзя.
# Хранятся ОСНОВАМИ (см. _stems): «года/году/годом» → «год», «физических» → «физ».
_BOOK_STOP = {"фл", "год", "кол", "чел", "физ", "лиц", "шт", "вхо", "учт", "рын", "рас"}


def _stems(text: str, drop: set[str] | None = None) -> set[str]:
    """Основы значимых слов (первые 3 буквы): «доля»/«долю»/«доле» → «дол». Нужно, чтобы
    сопоставление меток от LLM не ломалось об русские окончания."""
    out = {t[:3] for t in re.findall(r"[а-яёa-z]{3,}", text.lower())}
    return out - (drop or set())

# синонимы для разрезов: как пользователь называет → что искать в имени/подписи колонки
_DIM_SYN = [
    (r"\bтб\b|террит\w*\s*банк", r"\btb\b|tb_name|террит"),
    (r"госб", r"gosb|госб"),
    (r"причин\w*\s*оттока|причин", r"outflow|reason|причин"),
    (r"сегмент", r"segment|сегмент"),
    (r"холдинг", r"holding|холдинг"),
    (r"статус", r"status|статус"),
    (r"компан\w*|клиент", r"company|компан|клиент"),
]


def _resolve_pivot_col(name: str, prep: Prep, df: pd.DataFrame, books) -> tuple[str, str] | None:
    """Запрошенное имя колонки → (колонка_или_метка_книги, вид: 'dim'|'book'|'num'). None — нет такой."""
    n = name.strip().strip("«»\"'").lower()
    if not n:
        return None
    # 1) МЕРА-книга: «ФЛ доля год к году» → Δ книги. Сравниваем по ОСНОВАМ слов: LLM формулирует
    # метку каждый раз по-своему («ФЛ доля рынка», «ФЛ, входящие в долю рынка»), и точное сравнение
    # токенов ломается об окончания («доля» ≠ «долю»).
    nt = _stems(n)
    best, best_score = None, 0
    for s in ((books or {}).get("specs") or []):
        lt = _stems(str(s["label"]), drop=_BOOK_STOP)
        score = len(lt & nt)
        if score > best_score:
            best, best_score = s["label"], score
    if best_score:
        return best, "book"
    # 2) РАЗРЕЗ по синонимам
    for pat, colpat in _DIM_SYN:
        if re.search(pat, n):
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    continue
                if re.search(colpat, c + " " + prep.lbls.of(c), re.I):
                    return c, "dim"
            return None                                        # синоним понятен, но колонки нет
    # 3) прямое совпадение с именем/подписью
    for c in df.columns:
        if n == c.lower() or n == prep.lbls.of(c).lower():
            return c, ("num" if pd.api.types.is_numeric_dtype(df[c]) else "dim")
    return None


def _pivot(df: pd.DataFrame, question: str, prep: Prep, books) -> dict | None:
    """Сводная таблица ПО ЯВНОМУ ЗАПРОСУ пользователя: разрезы × меры, суммы, детерминированно.
    Возвращает {dims, measures, rows, missing} — missing честно перечисляет, чего нет в таблице."""
    m = _PIVOT_RE.search(question)
    if not m:
        if re.search(r"свод\w*", question, re.I):     # просили сводную, но фразу не разобрали
            logger.warning("investigate СВОДНАЯ: в вопросе есть «сводн…», но список полей не распознан "
                           "(ждём «сводную таблицу/диаграмму с колонками/полями: A, B, C»)")
        return None
    raw = [p for p in re.split(r"[,;]| и ", m.group("cols")) if p.strip()]
    dims, meas, missing = [], [], []
    for p in raw:
        r = _resolve_pivot_col(p, prep, df, books)
        if r is None:
            missing.append(p.strip())
            continue
        col, kind = r
        (dims if kind == "dim" else meas).append((p.strip(), col, kind))
    if not dims or not meas:
        logger.info("investigate СВОДНАЯ: не собрана (разрезов=%d, мер=%d, нет=%s)",
                    len(dims), len(meas), missing)
        return None

    # значения мер: книга → Δ (delta_col или y26-y25); обычная числовая → сама колонка
    spec_by_label = {s["label"]: s for s in ((books or {}).get("specs") or [])}
    series: dict[str, pd.Series] = {}
    for title, col, kind in meas:
        if kind == "book":
            s = spec_by_label[col]
            dc, y25, y26 = s.get("delta_col"), s.get("y25"), s.get("y26")
            series[title] = (_to_num(df[dc]) if dc in df.columns
                             else _to_num(df[y26]) - _to_num(df[y25]))
        else:
            series[title] = _to_num(df[col])

    g = pd.DataFrame({t: v for t, v in series.items()})
    for _, col, _k in dims:
        g[col] = df[col].astype(str)
    dcols = [c for _, c, _k in dims]
    agg = g.groupby(dcols, dropna=False).sum(min_count=1).reset_index()
    key = list(series)[0]
    total = len(agg)

    tm = _PIVOT_TOP_RE.search(question)                        # «топ 10» из промта, иначе дефолт
    topn = min(int(tm.group(1)), 200) if tm else _PIVOT_TOP_DEFAULT

    # СВОДНАЯ: строки-ЛИСТЬЯ, у каждой все разрезы своими колонками (раскладка «вбок», как в Excel
    # в табличном виде). Иерархию/подытоги/сворачивание собирает Tabulator (groupBy) — заодно
    # получаем фильтры, сортировку и единый стиль. Длинные ветки режем до топ-N по мере, остаток —
    # честная строка «прочие (N)», поэтому групповые суммы Tabulator сходятся с полными данными.
    meas = list(series)
    leaves: list[tuple[list[str], list[float]]] = []
    truncated = False

    def _sums(part) -> list[float]:
        return [float(part[m].sum()) for m in meas]

    caps = _level_caps(len(dcols), topn)                     # топ-N на уровень под лимит строк

    def _walk(part: pd.DataFrame, lvl: int, path: list[str]) -> None:
        nonlocal truncated
        if len(leaves) >= _PIVOT_MAX_LEAVES * 2:             # аварийный стоп (не должен срабатывать)
            truncated = True
            return
        col = dcols[lvl]
        gb = part.groupby(col, dropna=False)
        order = list(gb[key].sum().sort_values().index)      # самое сильное снижение — вверх
        shown = order[:caps[lvl]]
        for val in shown:
            sub = gb.get_group(val)
            p = path + [fmt_val(val)]
            if lvl == len(dcols) - 1:
                leaves.append((p, _sums(sub)))
            else:
                _walk(sub, lvl + 1, p)
        rest = order[caps[lvl]:]
        if rest:                                             # честный остаток ветки
            sub = part[part[col].isin(rest)]
            pad = [""] * (len(dcols) - lvl - 1)
            leaves.append((path + [f"прочие ({len(rest)})"] + pad, _sums(sub)))

    _walk(agg, 0, [])
    grand = _sums(agg)
    logger.info("investigate СВОДНАЯ: разрезы=%s | меры=%s | комбинаций=%d | топ по уровням=%s | "
                "строк-листьев=%d%s | нет колонок=%s", dcols, meas, total, caps, len(leaves),
                " (ОБРЕЗАНО)" if truncated else "", missing)
    return {"dims": [t for t, _c, _k in dims], "measures": meas, "leaves": leaves, "grand": grand,
            "total": total, "topn": topn, "caps": caps, "key": key, "missing": missing,
            "truncated": truncated}


def _who(df, frame: Frame, assets, lbls, elab=fmt_val,
         ctx_dims=()) -> tuple[pd.DataFrame | None, str | None]:
    ent = frame.entity
    if not ent or ent not in df.columns:
        return None, None
    s = _to_num(df[frame.target])
    g = s.groupby(df[ent]).sum()
    g = g.sort_values() if frame.mode == "change" else g.sort_values(ascending=False)
    top = g.head(12)
    if frame.mode == "change":
        top = top[top < 0]
    if top.empty:
        return None, None
    val = None
    if frame.value and frame.value in df.columns:
        vser = _to_num(df[frame.value])            # Decimal/строка с запятой → float
        # min_count=1: группа из одних NULL → NaN («нет данных»), а не тихий 0
        vv = vser.groupby(df[ent]).sum(min_count=1)
        val = vv.reindex(top.index)
        logger.info(
            "investigate КТО: сущность=%s | ценность=%s | непустых в колонке=%d/%d | "
            "групп с ценностью=%d/%d | из топ-%d ценность получили=%d | примеры=%s",
            ent, frame.value, int(vser.notna().sum()), len(vser),
            int(vv.notna().sum()), len(vv), len(top), int(val.notna().sum()),
            [(str(k)[:18], None if pd.isna(v) else round(float(v), 1))
             for k, v in list(val.items())[:3]])
    # КЛАССИФИКАЦИЯ клиента (ГОСБ = территория, сегмент): один клиент может быть в НЕСКОЛЬКИХ
    # значениях — «модальное» врало бы. Показываем значение с НАИБОЛЬШИМ вкладом в снижение
    # этого клиента и помечаем «+N», если просело ещё где-то. Это и есть «где отрабатывать».
    ctx = [d for d in ctx_dims if d in df.columns and d != ent][:2]
    sub = df[df[ent].isin(set(top.index))]
    ssub = _to_num(sub[frame.target])
    ctx_vals: dict[str, list] = {}
    for d in ctx:
        gr = ssub.groupby([sub[ent], sub[d]]).sum()
        pick = gr.groupby(level=0).idxmin() if frame.mode == "change" else gr.groupby(level=0).idxmax()
        nneg = (gr < 0).groupby(level=0).sum() if frame.mode == "change" else None
        cells = []
        for e in top.index:
            key = pick.get(e)
            v = fmt_val(key[1]) if isinstance(key, tuple) else "—"
            k = int(nneg.get(e, 0)) if nneg is not None else 0
            cells.append(f"{v} +{k - 1}" if k > 1 else v)
        ctx_vals[d] = cells

    disp = top.copy(); disp.index = [elab(i) for i in top.index]     # «Имя (ID клиента …)»
    tbl = pd.DataFrame(index=disp.index)
    for d in ctx:
        tbl[d] = ctx_vals[d]
    tbl["contrib"] = disp.values
    if val is not None:
        tbl["value"] = val.values
        # ПОКРЫТИЕ: «потенциал примерно равнозначный сумме снижения и/или больше» — это ОТНОШЕНИЕ,
        # без него две колонки рядом ни на что не отвечают.
        with np.errstate(divide="ignore", invalid="ignore"):
            tbl["cover"] = val.values / np.abs(top.values)
    tbl.attrs["ctx"] = ctx
    if ctx:
        logger.info("investigate КТО: колонки-классификации=%s", ctx)
    chart = _entity_chart(disp, ent, frame, assets, lbls)
    return tbl, chart


# ---------- графики ----------
def _tick(v):
    s = fmt_val(v)
    return s if len(s) <= 24 else s[:22] + "…"


def _contrib_chart(c: Contrib, frame: Frame, assets, lbls, *, path_len=0, tag="lvl") -> str | None:
    try:
        t = c.table.head(12).iloc[::-1]              # крупнейшие сверху
        vals = t["contrib"].astype(float).values
        names = [_tick(x) for x in t.index]
        colors = [_C_LOSS if v < 0 else _C_GAIN for v in vals] if frame.mode == "change" else \
            [_C_NEUT] * len(vals)
        fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(t) + 0.6)))
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=9)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {_fmt(abs(v))}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color="#334")
        ax.axvline(0, color="#888", lw=1)
        ax.margins(x=0.16)
        side = "потери" if c.loss < 0 else "прирост"
        ttl = f"Вклад в {side}: {lbls.of(frame.target)} по «{lbls.of(c.dim)}»"
        ax.set_title(ttl, fontsize=11)
        ax.grid(axis="x", alpha=.3)
        bar_cols = (["loss" if v < 0 else "gain" for v in vals] if frame.mode == "change"
                    else ["neut"] * len(vals))
        spec = render.spec_barh(names, [float(v) for v in vals], title=ttl, colors=bar_cols)
        return _save(fig, assets, f"inv_{tag}{path_len}_{c.dim}", spec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: график вклада не построен: %s", exc)
        return None


def _entity_chart(top: pd.Series, ent: str, frame: Frame, assets, lbls) -> str | None:
    try:
        t = top.iloc[::-1]
        vals = t.astype(float).values
        fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(t) + 0.6)))
        ax.barh(range(len(vals)), vals, color=_C_LOSS if frame.mode == "change" else _C_NEUT)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels([_tick(x) for x in t.index], fontsize=9)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {_fmt(abs(v))}", va="center", ha="right" if v < 0 else "left",
                    fontsize=8, color="#334")
        ax.axvline(0, color="#888", lw=1); ax.margins(x=0.16)
        ttl = f"Кто: ТОП по «{lbls.of(ent)}» ({lbls.of(frame.target)})"
        ax.set_title(ttl, fontsize=11)
        ax.grid(axis="x", alpha=.3)
        spec = render.spec_barh([_tick(x) for x in t.index], [float(v) for v in vals], title=ttl,
                                color=("loss" if frame.mode == "change" else "neut"))
        return _save(fig, assets, f"inv_who_{ent}", spec)
    except Exception:  # noqa: BLE001
        return None


# ---------- ГЛУБОКАЯ ДЕКОМПОЗИЦИЯ ----------
def _side_of(frame: Frame) -> str:
    return frame.direction if frame.direction in ("loss", "gain") else "auto"


def _make_elab(df, frame: Frame, lbls):
    """Подпись значения организации ВСЕГДА как «Название (ИНН …)» — независимо от того,
    ключуем ли анализ по названию или по ИНН. Партнёр (имя/ИНН) показывается только если он
    ~однозначен на ключ (median distinct ≤ 1), иначе ввёл бы в заблуждение."""
    ent = frame.entity
    if not ent or ent not in df.columns:
        return lambda v: fmt_val(v)

    def _lookup(src):
        # карта ключ→значение src, если src ~однозначен на каждый ключ
        if not src or src not in df.columns or src == ent:
            return None
        if float(df.groupby(ent)[src].nunique(dropna=True).median()) > 1:
            return None
        return df.groupby(ent)[src].agg(lambda s: (s.dropna().mode().iloc[0]
                                                   if not s.dropna().mode().empty else None))

    if _NAME_ENT.search(ent) and not _ID_ENT.search(ent):     # ключ — название: «Название (ИНН)»
        idmap = _lookup(frame.id_col)
        idlabel = lbls.of(frame.id_col) if frame.id_col else ""

        def elab(v):
            iv = idmap.get(v) if idmap is not None else None
            return f"{fmt_val(v)} ({idlabel} {fmt_val(iv)})" if iv is not None else fmt_val(v)
    else:                                                     # ключ — ИНН/код: «Название (ИНН)»
        namemap = _lookup(frame.name_col)
        idlabel = lbls.of(ent)

        def elab(v):
            nm = namemap.get(v) if namemap is not None else None
            return f"{fmt_val(nm)} ({idlabel} {fmt_val(v)})" if nm is not None else fmt_val(v)
    return elab


def _side_series(g: pd.Series, side: str) -> pd.Series:
    if side == "loss":
        return g[g < 0].sort_values()
    if side == "gain":
        return g[g > 0].sort_values(ascending=False)
    return g.reindex(g.abs().sort_values(ascending=False).index)


@dataclass
class Child:
    label: str
    contrib: float
    share_in_parent: float


@dataclass
class Node:
    label: str
    contrib: float
    share_of_total: float
    children: list          # list[Child]
    tail_contrib: float     # «прочие» внутри узла
    tail_count: int


def _build_tree(df, frame: Frame, ref: float, top_dim: str, entity: str, side: str,
                elab=fmt_val, *, max_seg=6, max_comp=8) -> list[Node]:
    """Дерево: top_dim (сегмент) → топ-entity (компании) внутри + «прочие»."""
    # СОГЛАСОВАННАЯ loss-декомпозиция: узел (сегмент) = сумма ПРОИГРЫВАЮЩИХ сущностей внутри,
    # чтобы «топ + прочие» = узлу, а Σ узлов = общим потерям (без взаимозачёта с приростом).
    seg_side: dict = {}
    for segname, sub in df.groupby(top_dim):
        cs = _side_series(pd.to_numeric(sub[frame.target], errors="coerce").groupby(sub[entity]).sum(), side)
        val = float(cs.sum())
        if abs(val) > 1e-9:
            seg_side[segname] = (val, cs)
    nodes: list[Node] = []
    for segname, (val, cs) in sorted(seg_side.items(), key=lambda kv: abs(kv[1][0]), reverse=True)[:max_seg]:
        segabs = abs(val) or 1.0
        top = cs.head(max_comp)
        tail = float(cs.iloc[max_comp:].sum()) if len(cs) > max_comp else 0.0
        children = [Child(elab(i), float(v), abs(v) / segabs * 100) for i, v in top.items()]
        nodes.append(Node(fmt_val(segname), val, abs(val) / abs(ref) * 100,
                          children, tail, max(0, len(cs) - max_comp)))
    return nodes


def _treemap_chart(tree: list[Node], frame: Frame, assets, lbls, top_dim, entity) -> str | None:
    """Вложенный treemap: сегмент-прямоугольник, внутри — компании (размер = вклад)."""
    try:
        import squarify
        segs = [(n.label, abs(n.contrib), n) for n in tree if abs(n.contrib) > 0]
        if not segs:
            return None
        W, H = 100.0, 100.0
        seg_sizes = squarify.normalize_sizes([x[1] for x in segs], W, H)
        seg_rects = squarify.squarify(seg_sizes, 0, 0, W, H)
        leaf_pal = sns.color_palette("tab20", 20)                   # цвет — по КОМПАНИИ, не сегменту
        ci = 0
        fig, ax = plt.subplots(figsize=(12, 7.5))
        for (name, sz, node), rect in zip(segs, seg_rects):
            ax.add_patch(plt.Rectangle((rect["x"], rect["y"]), rect["dx"], rect["dy"],
                                       facecolor="none", edgecolor="#111", lw=3, zorder=5))
            parts = [(c.label, abs(c.contrib)) for c in node.children if abs(c.contrib) > 0]
            if node.tail_count:
                parts.append((f"прочие ({node.tail_count})", abs(node.tail_contrib)))
            if not parts:
                continue
            csz = squarify.normalize_sizes([p[1] for p in parts], rect["dx"], rect["dy"])
            crects = squarify.squarify(csz, rect["x"], rect["y"], rect["dx"], rect["dy"])
            for (clabel, cval), cr in zip(parts, crects):
                is_other = clabel.startswith("прочие")
                fc = "#c9d2de" if is_other else leaf_pal[ci % 20]
                if not is_other:
                    ci += 1
                ax.add_patch(plt.Rectangle((cr["x"], cr["y"]), cr["dx"], cr["dy"],
                                           facecolor=fc, edgecolor="white", lw=1.0, alpha=.92))
                if cr["dx"] > 7 and cr["dy"] > 5:
                    ax.text(cr["x"] + cr["dx"] / 2, cr["y"] + cr["dy"] / 2,
                            f"{clabel[:16]}\n{_fmt(cval)}", ha="center", va="center", fontsize=7)
            # подпись сегмента
            ax.text(rect["x"] + rect["dx"] / 2, rect["y"] + rect["dy"] - 2.5,
                    f"{name} · {_fmt(abs(node.contrib))}", ha="center", va="top",
                    fontsize=9, fontweight="bold", color="#111")
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
        ttl = f"Дерево потерь: «{lbls.of(top_dim)}» → «{lbls.of(entity)}» (площадь = вклад)"
        ax.set_title(ttl, fontsize=12)
        ids_, labs, pars, vals_ = [], [], [], []       # ECharts treemap: сегмент→компании
        for n in tree:
            children = [(c.label, abs(c.contrib)) for c in n.children if abs(c.contrib) > 0]
            if n.tail_count:
                children.append((f"прочие ({n.tail_count})", abs(n.tail_contrib)))
            seg_val = sum(v for _, v in children) or abs(n.contrib)
            if seg_val <= 0:
                continue
            sid = f"seg::{n.label}"
            ids_.append(sid); labs.append(n.label); pars.append(""); vals_.append(seg_val)
            for cl, cv in children:
                ids_.append(f"{sid}::{cl}"); labs.append(cl); pars.append(sid); vals_.append(cv)
        spec = render.spec_treemap(labs, pars, vals_, ids=ids_, title=ttl) if ids_ else None
        return _save(fig, assets, "inv_treemap", spec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: treemap не построен: %s", exc)
        return None


def _waterfall_chart(df, frame: Frame, dim, side, assets, lbls) -> str | None:
    """Водопад: как складывается итог из вкладов по разрезу (потери ↓, прирост ↑ → net)."""
    try:
        s = pd.to_numeric(df[frame.target], errors="coerce")
        g = s.groupby(df[dim]).sum().sort_values()          # от потерь к приросту
        items = list(g.items())
        if len(items) > 12:                                 # схлопнуть середину в «прочие»
            items = items[:6] + [("прочие", float(g.iloc[6:-3].sum()))] + items[-3:]
        labels_ = [_tick(k) for k, _ in items] + ["ИТОГ (net)"]
        vals = [float(v) for _, v in items]
        fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(labels_) + 2), 5))
        run = 0.0
        for i, v in enumerate(vals):
            ax.bar(i, v, bottom=run, color=_C_LOSS if v < 0 else _C_GAIN, edgecolor="white")
            ax.text(i, run + v + (max(map(abs, vals)) * 0.01 if v >= 0 else -max(map(abs, vals)) * 0.01),
                    _fmt(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
            run += v
        ax.bar(len(vals), run, color=_C_NEUT, edgecolor="white")
        ax.text(len(vals), run, _fmt(run), ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.axhline(0, color="#888", lw=1)
        ax.set_xticks(range(len(labels_))); ax.set_xticklabels(labels_, rotation=35, ha="right", fontsize=8)
        ttl = f"Как складывается {lbls.of(frame.target)} по «{lbls.of(dim)}»"
        ax.set_title(ttl, fontsize=11)
        ax.grid(axis="y", alpha=.3)
        spec = render.spec_waterfall(labels_, vals + [sum(vals)], title=ttl)
        return _save(fig, assets, f"inv_waterfall_{dim}", spec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: водопад не построен: %s", exc)
        return None


def _pareto_curve(df, frame: Frame, entity, side, assets, lbls) -> tuple[str | None, dict]:
    """Кривая Парето: накопленная доля потерь по сущностям (топ-N = X%)."""
    try:
        s = pd.to_numeric(df[frame.target], errors="coerce")
        g = _side_series(s.groupby(df[entity]).sum(), side)
        vals = g.abs().values
        total = vals.sum()
        if total <= 0 or len(vals) < 5:
            return None, {}
        cum = np.cumsum(vals) / total * 100
        n80 = int((cum < 80).sum()) + 1
        top10 = float(cum[min(9, len(cum) - 1)])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(range(1, len(cum) + 1), cum, color=_C_LOSS, lw=2)
        ax.axhline(80, color="gray", ls="--", lw=1); ax.axvline(n80, color="gray", ls=":", lw=1)
        ax.set_ylim(0, 105); ax.set_xlabel(f"число «{lbls.of(entity)}» (по убыванию вклада)")
        ax.set_ylabel("накопленная доля потерь, %")
        ttl = f"Концентрация: топ-{n80} «{lbls.of(entity)}» = 80% потерь"
        ax.set_title(ttl, fontsize=11)
        ax.grid(alpha=.3)
        spec = render.spec_line(list(range(1, len(cum) + 1)),
                                {"накопленная доля потерь, %": [float(v) for v in cum]},
                                title=ttl, unit="%", marker_x=n80)
        chart = _save(fig, assets, f"inv_pareto_{entity}", spec)
        return chart, {"n_for_80": n80, "total": int(len(vals)), "top10_share": round(top10, 1)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: Парето-кривая не построена: %s", exc)
        return None, {}


def _crosstab_heatmap(df, frame: Frame, da, db, side, assets, lbls) -> str | None:
    try:
        s = pd.to_numeric(df[frame.target], errors="coerce").astype("float64")   # pd.NA → NaN
        piv = s.groupby([df[da], df[db]]).sum().unstack()
        # оставляем сторону потерь/прироста, топ по объёму
        keep_a = _side_series(s.groupby(df[da]).sum(), side).head(8).index
        keep_b = _side_series(s.groupby(df[db]).sum(), side).head(8).index
        piv = piv.reindex(index=keep_a, columns=keep_b).astype("float64").fillna(0.0)
        if piv.empty:
            return None
        piv.index = [_tick(x) for x in piv.index]; piv.columns = [_tick(x) for x in piv.columns]
        fig, ax = plt.subplots(figsize=(min(12, 1.3 * len(piv.columns) + 3), min(8, 0.6 * len(piv.index) + 2)))
        sns.heatmap(piv / 1000, annot=True, fmt=".0f", cmap="RdYlGn", center=0, ax=ax,
                    cbar_kws={"label": f"{lbls.of(frame.target)}, тыс"}, linewidths=.5, annot_kws={"fontsize": 8})
        ttl = f"{lbls.of(frame.target)}: «{lbls.of(da)}» × «{lbls.of(db)}»"
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        pv = piv / 1000
        z = [[None if pd.isna(v) else round(float(v), 1) for v in row] for row in pv.values]
        spec = render.spec_heatmap(z, list(pv.columns), list(pv.index), title=ttl,
                                   unit=f"{lbls.of(frame.target)}, тыс", scale="RdYlGn")
        return _save(fig, assets, f"inv_heat_{da}_{db}", spec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: кросс-таб не построен: %s", exc)
        return None


def _quadrant_chart(df, frame: Frame, entity, value, side, assets, lbls) -> str | None:
    """Квадрант: убыль (X) vs ценность/потенциал (Y) по сущностям — приоритет возврата."""
    try:
        if not value or value not in df.columns:
            return None
        s = pd.to_numeric(df[frame.target], errors="coerce")
        loss = _side_series(s.groupby(df[entity]).sum(), side).abs()
        val = pd.to_numeric(df[value], errors="coerce").groupby(df[entity]).sum().reindex(loss.index)
        d = pd.DataFrame({"loss": loss, "val": val}).dropna().head(60)
        if len(d) < 4:
            return None
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(d["loss"], d["val"], s=40, color=_C_LOSS, alpha=.6)
        ax.axvline(d["loss"].median(), color="gray", ls="--", lw=1)
        ax.axhline(d["val"].median(), color="gray", ls="--", lw=1)
        # подписать топ-6 по (убыль×ценность)
        pri = (d["loss"] * d["val"]).sort_values(ascending=False).head(6).index
        for i in pri:
            ax.annotate(_tick(i), (d.loc[i, "loss"], d.loc[i, "val"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"убыль ({lbls.of(frame.target)})"); ax.set_ylabel(lbls.of(value))
        ttl = f"Приоритет возврата: убыль × «{lbls.of(value)}» по «{lbls.of(entity)}»"
        ax.set_title(ttl, fontsize=11)
        ax.grid(alpha=.3)
        spec = render.spec_scatter(d["loss"].tolist(), d["val"].tolist(), [_tick(i) for i in d.index],
                                   title=ttl, xlab=f"убыль ({lbls.of(frame.target)})", ylab=lbls.of(value),
                                   vline=float(d["loss"].median()), hline=float(d["val"].median()))
        return _save(fig, assets, f"inv_quadrant_{entity}", spec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: квадрант не построен: %s", exc)
        return None


# ---------- синтез (LLM) ----------
_SYNTH_SYS = (
    "Ты пишешь ВЫВОД расследования простым языком для руководителя. По фактам дай:\n"
    "answer — прямой ответ на вопрос (1-2 предложения, где именно сосредоточено);\n"
    "chain — причинная ЦЕПОЧКА: total → главный разрез → внутри него → причина → кто "
    "(3-5 коротких пунктов с числами);\n"
    "actions — 2-4 конкретных действия (куда смотреть/что делать, приоритет по ценности).\n"
    "СТРОГО опирайся на ФАКТЫ, НИЧЕГО не додумывай:\n"
    "• «Причину» бери из facts.почему по LIFT (чаще_базы_в_разы): значение с наибольшим lift — "
    "это НАСТОЯЩИЙ драйвер. Формулируй «по <разрез> значение «X» даёт N% потерь при M% строк "
    "(в L раз чаще базы)», а НЕ «преобладает X» и НЕ как доказанный механизм.\n"
    "• Сущности из facts.кто называй их ТИПОМ (facts.кто.сущность, напр. «компании по ИНН», "
    "«клиенты»), и НЕ путай с целевой величиной (facts.целевая) — ИНН это компания, а не физлицо.\n"
    "• Числа и названия — только из фактов.\n"
    "• ВАЖНО про МЕТОДОЛОГИЮ: если есть facts.методология_книги — НАЧНИ answer с неё: какие "
    "метрики выросли/упали и что реальное снижение сидит в ИСКЛЮЧЁННЫХ из доли категориях "
    "(ГПХ/самозанятые); подчеркни, что метрики с разными знаменателями смешивать нельзя. Только "
    "потом переходи к тому, где сосредоточено изменение внутри основной метрики.\n"
    "ЗАПРЕЩЕНО: матжаргон. Если потери РАЗМАЗАНЫ (нет доминирующего разреза) — честно скажи "
    "это в answer. Верни JSON: {\"answer\":\"...\",\"chain\":[...],\"actions\":[...]}"
)


def _when(df, frame: Frame, prep, unit: str, side: str, assets, lbls) -> tuple[str | None, dict]:
    """B3 «Когда»: динамика целевой величины по месяцам — где сконцентрировалось и резкий слом.
    Всё детерминированно (pandas): пик исследуемой стороны + самый резкий сдвиг период-к-периоду.

    ВАЖНО: берём только дату, которая является ВРЕМЕННО́Й ОСЬЮ метрики. Даты ДЕЙСТВИЙ
    (задача/заявка/создание/контакт) — это когда с организацией работали, а НЕ когда менялась
    измеряемая величина; раскладывать по ним потери — ложная атрибуция (см. фидбек по market_analysis)."""
    dates = [d for d in prep.roles.dates
             if d in df.columns and not _is_action_date(d, prep.meta)]
    if not dates:
        return None, {}
    dcol = dates[0]
    d = pd.to_datetime(df[dcol], errors="coerce")
    v = pd.to_numeric(df[frame.target], errors="coerce")
    ok = d.notna() & v.notna()
    if int(ok.sum()) < 12:
        return None, {}
    ts = v[ok].groupby(d[ok].dt.to_period("M")).sum().sort_index()
    if len(ts) < 4 or ts.index.nunique() < 4:
        return None, {}
    signed = _side_series(ts, side)                       # только вклад исследуемой стороны
    if signed.empty:
        return None, {}
    peak, peak_v = signed.index[0], float(signed.iloc[0])
    peak_share = abs(peak_v) / (abs(float(signed.sum())) or 1.0)
    diff = ts.diff().dropna()
    brk = None
    if len(diff):
        cand = diff[diff < 0] if side == "loss" else (diff[diff > 0] if side == "gain" else diff)
        cand = cand if not cand.empty else diff
        brk = cand.abs().idxmax()
    facts = {"период_пик": str(peak), "вклад_пика": _fmt_u(peak_v, unit),
             "доля_пика_%": round(peak_share * 100, 1),
             "слом": (str(brk) if brk is not None else None),
             "сдвиг_на_сломе": (_fmt_u(float(diff.get(brk, 0.0)), unit) if brk is not None else None)}
    return _when_chart(ts, brk, frame, assets, lbls), facts


def _when_chart(ts: pd.Series, brk, frame: Frame, assets, lbls) -> str | None:
    try:
        labels_ = [str(p) for p in ts.index]
        vals = [float(x) for x in ts.values]
        colors = [_C_NEUT if p == brk else (_C_LOSS if v < 0 else _C_GAIN)
                  for p, v in zip(ts.index, vals)]
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(ts) + 2), 3.6))
        ax.bar(range(len(vals)), vals, color=colors, edgecolor="white")
        ax.axhline(0, color="#888", lw=1)
        ax.set_xticks(range(len(labels_)))
        ax.set_xticklabels(labels_, rotation=90, fontsize=7)
        ttl = f"Динамика «{lbls.of(frame.target)}» по месяцам"
        ax.set_title(ttl, fontsize=11)
        ax.grid(axis="y", alpha=.3)
        bar_cols = ["neut" if p == brk else ("loss" if v < 0 else "gain")
                    for p, v in zip(ts.index, vals)]
        spec = render.spec_bar_v(labels_, vals, title=ttl, colors=bar_cols)
        return _save(fig, assets, "inv_when", spec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: динамика не построена: %s", exc)
        return None


def _flatten_item(x) -> str:
    """LLM иногда вставляет в chain/actions dict вместо текста ({'total': '-659 тыс'}) —
    разворачиваем в значения, а не в JSON-строку."""
    if isinstance(x, dict):
        return "; ".join(str(v) for v in x.values())
    s = str(x).strip()
    m = re.fullmatch(r"\{\s*['\"]?[\w\s]+['\"]?\s*:\s*['\"]?(.+?)['\"]?\s*\}", s)
    return m.group(1).strip() if m else s


def _synthesize(llm, question, facts: dict) -> dict:
    try:
        out = llm.complete_json(_SYNTH_SYS, f"Вопрос: {question}\n\nФакты:\n{facts}",
                                max_tokens=3000, node="investigate_synth")
        cid = labels_mod.clean_id_text                    # «ИНН» → «ID клиента» в прозе LLM
        return {"answer": cid(str(out.get("answer", "")).strip()),
                "chain": [cid(s) for x in (out.get("chain") or []) if (s := _flatten_item(x))],
                "actions": [cid(s) for x in (out.get("actions") or []) if (s := _flatten_item(x))]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("investigate: синтез не удался (%s)", exc)
        return {"answer": "", "chain": [], "actions": []}


# ---------- оркестратор ----------
def _route_playbook(fqn, prep, question, catalog, out_dir, progress, llm):
    """§8.1 роутинг внутри /investigate: строим семмодель, LLM-FRAME определяет тип; если
    сработал НЕ-дефолтный плейбук (сейчас — impact) — исполняем его; иначе None (обычный путь).
    Дефолтный loss_attribution оставляем текущей (отлаженной) реализации расследования."""
    try:
        from . import plans, semantics, surface
        model = semantics.build_semantic_table(fqn, prep.table_desc, prep.df, prep.roles,
                                               prep.measures, prep.lbls, meta=prep.meta, catalog=catalog)
        try:
            semantics.save(model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("investigate: семмодель не сохранена: %s", exc)
        progress("FRAME: определяю тип расследования…")
        frame = surface.frame_llm(llm, question, model, prep.df) or {}
        pb = None
        if frame.get("playbook"):                         # выбор плейбука по LLM-FRAME
            pb = next((p for p in plans.load_playbooks() if p.name == frame["playbook"]), None)
        if pb is None:                                    # фолбэк: ключевой роутинг §8.1
            pb = plans.select_playbook(question, has_flag=bool(model.tool_flags))
        if pb is None or pb.name == "loss_attribution":
            return None                                   # дефолт — обычное расследование
        if pb.name == "impact" and not model.tool_flags:
            return None                                   # impact без tool_flag неприменим
        progress(f"вопрос распознан как «{pb.name}» — исполняю плейбук…")
        return surface.run_playbook(prep.df, model, pb, question, table_desc=prep.table_desc,
                                    fqn=fqn, out_dir=out_dir, progress=progress, frame=frame)
    except Exception as exc:  # noqa: BLE001  (роутинг не должен ронять обычное расследование)
        logger.warning("investigate: роутинг в плейбук не удался (%s) — обычный путь", exc)
        return None


def investigate(db, catalog, llm, fqn: str, question: str, *, where: str | None = None,
                out_dir: Path | None = None, progress=lambda m: None) -> dict:
    schema, table = fqn.split(".", 1)
    out_dir = out_dir or PATHS.workspace_dir
    assets = out_dir / "report_assets" / f"{table}_investigate"
    if assets.exists():
        import shutil
        shutil.rmtree(assets, ignore_errors=True)

    prep = _prepare(db, catalog, llm, fqn, where, progress)
    df = prep.df

    # РОУТИНГ (§8.1): вопрос про эффективность инструмента + есть tool_flag → impact-плейбук;
    # иначе — дефолтное расследование «где/почему/кто» ниже. Отдельных команд НЕ плодим.
    routed = _route_playbook(fqn, prep, question, catalog, out_dir, progress, llm)
    if routed is not None:
        return routed

    progress("формулирую цель расследования…")
    frame = _frame(llm, question, prep)
    progress("проверяю методологию: книги портфеля (что входит/не входит)…")
    books = _portfolio_books(df, prep, llm)
    s = pd.to_numeric(df[frame.target], errors="coerce")
    net = float(s.sum())
    gross_loss = float(s[s < 0].sum()) if frame.mode == "change" else None
    gross_gain = float(s[s > 0].sum()) if frame.mode == "change" else None
    unit = _target_unit(prep, frame.target)
    side = _side_of(frame)                                # какую сторону расследуем
    # знаменатель «доли»: сторона потерь (если вопрос про потери), иначе доминирующая
    if frame.mode == "change":
        ref = gross_loss if side == "loss" else (gross_gain if side == "gain"
              else (gross_loss if abs(gross_loss) >= abs(gross_gain) else gross_gain))
    else:
        ref = net
    ref = ref or 1.0

    # главный разрез для дерева/водопада/кросс-таба — с макс. концентрацией вклада
    ent_con = {core._concept(e) for e in [frame.entity] if frame.entity}
    # кандидаты — не только drill_dims от LLM: он часто возвращает один разрез, и тогда «в каких
    # ГОСБ» остаётся без ответа. Добавляем остальные разрезы (кроме причин/статусов и сущности).
    extra = [d for d in prep.roles.dimensions
             if d not in frame.drill_dims and d not in frame.why_dims]
    cands = [d for d in (list(frame.drill_dims) + extra)
             if d in df.columns and core._concept(d) not in ent_con]

    def _dim_ok(d: str, cap: int) -> bool:
        n = df[d].nunique(dropna=True)
        return 2 <= n <= cap and n < 0.5 * len(df)      # не id-подобный

    # для дерева/водопада/хитмапа нужна ЧИТАЕМОСТЬ графика → жёсткий потолок категорий
    dcand = [d for d in cands if _dim_ok(d, _PRIMARY_MAX_CARD)]
    primary = max(dcand, key=lambda d: _decompose(df, frame.target, d, frame.mode, ref, side).top_share,
                  default=(dcand[0] if dcand else None))
    # «Где»: таблица (топ-6 + «прочие»), поэтому потолок мягкий — иначе ГОСБ (сотни значений)
    # выпадает, и вопрос «в каких ГОСБ» остаётся без ответа.
    wcand = [d for d in cands if _dim_ok(d, _WHERE_MAX_CARD)]
    wdims = ([primary] if primary else []) + [d for d in wcand if d != primary]
    where_dims = [(d, _decompose(df, frame.target, d, frame.mode, ref, side)) for d in wdims[:3]]
    logger.info("investigate ГДЕ: главный=%s | раскладываем по=%s | отсеяны по кардинальности=%s",
                primary, [d for d, _ in where_dims],
                [d for d in cands if d not in wcand])

    with core.report_style():                # локальный стиль графиков (не мутируем ноутбук)
        progress("строю водопад и кривую концентрации…")
        waterfall = _waterfall_chart(df, frame, primary, side, assets, prep.lbls) if primary else None
        pareto_chart, pareto_facts = (_pareto_curve(df, frame, frame.entity, side, assets, prep.lbls)
                                      if frame.entity else (None, {}))
        progress("раскладываю потери по дереву (сегмент → компании)…")
        elab = _make_elab(df, frame, prep.lbls)                # подпись «Имя (ИНН …)»
        tree = _build_tree(df, frame, ref, primary, frame.entity, side, elab) if (primary and frame.entity) else []
        treemap = _treemap_chart(tree, frame, assets, prep.lbls, primary, frame.entity) if tree else None
        progress("спускаюсь по горячему пути (сегмент→…→сущность)…")
        drill = _drill(df, frame, assets, prep.lbls, ref, side=side)
        drill_facts = _drill_facts(drill, prep.lbls, unit, side)
        progress("ищу причины и структуру…")
        why = _why(df, frame, assets, prep.lbls, ref, side)
        heat = (_crosstab_heatmap(df, frame, primary, frame.why_dims[0], side, assets, prep.lbls)
                if (primary and frame.why_dims) else None)
        comps = _components(df, frame, prep.lbls)
        progress("нахожу «виновников» и приоритет возврата…")
        who_tbl, who_chart = _who(df, frame, assets, prep.lbls, elab,
                                  ctx_dims=[d for d, _ in where_dims])
        rec_dim = primary if primary else None
        recovery = _recovery(df, frame, rec_dim) if rec_dim else None
        pivot = _pivot(df, question, prep, books)      # «построй сводную с колонками …»
        quadrant = _quadrant_chart(df, frame, frame.entity, frame.value, side, assets, prep.lbls) if frame.entity else None
        progress("смотрю динамику во времени…")
        when_chart, when_facts = _when(df, frame, prep, unit, side, assets, prep.lbls)

    facts = _facts_for_synth(frame, prep.lbls, net, gross_loss, gross_gain, unit, tree, why,
                             comps, who_tbl, pareto_facts, primary, when_facts, drill_facts)
    if books:                                        # методология в синтез (ведёт ответ)
        facts["методология_книги"] = {
            "note": books["note"],
            "книги": [{"книга": b.label, "роль": b.role, "дельта": _fmt_u(b.delta, unit)}
                      for b in books["books"]],
            "итог_основные": _fmt_u(books["head_sum"], unit),
            "итог_исключённые": _fmt_u(books["excl_sum"], unit)}
    progress("собираю причинную цепочку и действия…")
    synth = _synthesize(llm, question, facts)
    if books:                                        # детерминированная методологическая шапка — первой
        synth["answer"] = _books_lead(books, unit) + "\n\n" + (synth.get("answer") or "")
    if df.attrs.get("sample_note"):                  # плашка о сэмпле (гейт памяти)
        synth["answer"] = "⚠️ " + df.attrs["sample_note"] + "\n\n" + (synth.get("answer") or "")

    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = dict(prep=prep, frame=frame, table=table, fqn=fqn, question=question, where=where,
               net=net, gross_loss=gross_loss, gross_gain=gross_gain, unit=unit, primary=primary,
               waterfall=waterfall, pareto_chart=pareto_chart, pareto_facts=pareto_facts,
               tree=tree, treemap=treemap, why=why, heat=heat, comps=comps,
               who_tbl=who_tbl, who_chart=who_chart, quadrant=quadrant,
               when_chart=when_chart, when_facts=when_facts, drill_facts=drill_facts,
               books=books, synth=synth, where_dims=where_dims,
               recovery=recovery, rec_dim=rec_dim, pivot=pivot)
    md_path = out_dir / f"{table}_investigation.md"
    html_path = out_dir / f"{table}_investigation.html"
    md_path.write_text(_assemble_md(**ctx), encoding="utf-8")
    html_path.write_text(_assemble_html(**ctx), encoding="utf-8")
    return {"md_path": str(md_path), "html_path": str(html_path),
            "segments": len(tree), "rows": len(df)}


def _target_unit(prep: Prep, col: str) -> str:
    m = next((x for x in prep.measures if x.tech == col or x.col == col), None)
    if m:
        return m.unit
    uk = prep.lbls.unit_kind(col)
    return {"money": "₽", "people": "чел.", "percent": "%"}.get(uk, "")


def _fmt_u(v, unit):
    return f"{_fmt(v)} {unit}".strip()


def _side(loss) -> str:
    return "потерь" if loss < 0 else "прироста"


def _facts_for_synth(frame, lbls, net, gloss, ggain, unit, tree, why, comps, who_tbl,
                     pareto_facts, primary, when_facts=None, drill_facts=None) -> dict:
    f = {"целевая": f"{lbls.of(frame.target)} (единица: {unit or 'шт'})", "режим": frame.mode,
         "итог_net": _fmt_u(net, unit)}
    if gloss is not None:
        f["всего_потеряно"] = _fmt_u(gloss, unit)
        f["всего_приросло"] = _fmt_u(ggain, unit)
    f["главный_разрез"] = lbls.of(primary) if primary else None
    f["дерево"] = [{
        "узел": n.label, "вклад": _fmt_u(n.contrib, unit), "доля_от_общих_%": round(n.share_of_total, 1),
        "топ_внутри": [{"кто": c.label, "вклад": _fmt_u(c.contrib, unit),
                        "доля_в_узле_%": round(c.share_in_parent, 1)} for c in n.children[:4]],
    } for n in tree[:5]]
    if pareto_facts:
        f["концентрация"] = (f"топ-{pareto_facts.get('n_for_80')} «{lbls.of(frame.entity)}» = 80% потерь "
                             f"(всего {pareto_facts.get('total')}); топ-10 = {pareto_facts.get('top10_share')}%")
    f["почему"] = [{"разрез": lbls.of(d),
                    "значение": li.get("value"), "доля_потерь_%": li.get("loss_share"),
                    "доля_строк_%": li.get("base_share"), "чаще_базы_в_разы": li.get("lift"),
                    "разложение": [{"значение": v, "строк": n, "дельта": _fmt_u(c, unit)}
                                   for v, n, c in li.get("breakdown", [])[:8]]}
                   for d, li, _ in why]
    if comps:
        f["слагаемые"] = [{"часть": name, "значение": _fmt_u(v, unit)} for name, v in comps]
    if who_tbl is not None:
        # вместе с вкладом отдаём ЦЕННОСТЬ (потенциал) — иначе синтез про неё не напишет
        has_val = "value" in who_tbl.columns
        top = []
        for idx, row in who_tbl.head(6).iterrows():
            it = {"кто": fmt_val(idx), "вклад": _fmt_u(float(row["contrib"]), unit)}
            if has_val and pd.notna(row["value"]):
                it["потенциал"] = _fmt(float(row["value"]))
            top.append(it)
        f["кто"] = {"сущность": lbls.of(frame.entity) if frame.entity else "сущность", "топ": top}
        if frame.value:
            f["ценность"] = (f"{lbls.of(frame.value)} — сопоставляй с размером снижения "
                             f"(кого возвращать в первую очередь)")
    if when_facts:
        f["когда"] = when_facts
    if drill_facts:
        f["горячий_путь"] = [{"уровень": i, "разрез": d["разрез"], "значение": d["значение"],
                              "вклад": d["вклад"], "доля_родителя_%": d["доля_родителя_%"],
                              "доля_всех_%": d["доля_всех_%"], "строк": d["строк"]}
                             for i, d in enumerate(drill_facts, 1)]
    return f


# ---------- рендер ----------
_ROLE_RU = {"headline": "основная метрика", "excluded": "исключена из доли"}


def _books_takeaway(books, unit) -> str:
    """Детерминированный вывод методологии: основные метрики vs исключённые категории."""
    hs, es = books["head_sum"], books["excl_sum"]
    hw = "выросли" if hs > 0 else ("упали" if hs < 0 else "без изменений")
    parts = [f"Основные метрики портфеля {hw} на {_fmt_u(hs, unit)}"]
    if any(b.role == "excluded" for b in books["books"]):
        ew = "снизились" if es < 0 else "выросли"
        parts.append(f"а исключённые из доли категории (ГПХ/самозанятые) {ew} на {_fmt_u(es, unit)}")
    tail = ("— то есть реальное снижение портфеля вне доли; смешивать метрики нельзя."
            if hs >= 0 > es else "— методологии не смешиваем.")
    return ", ".join(parts) + " " + tail


def _books_lead(books, unit) -> str:
    """Детерминированная методологическая ШАПКА ответа: всегда первой строкой и с ВЕРНЫМ знаком,
    чтобы вывод не переворачивался (нельзя писать «снижение», если целевые метрики выросли)."""
    hs, es = books["head_sum"], books["excl_sum"]
    heads = [b.label for b in books["books"] if b.role == "headline"]
    excl = [b.label for b in books["books"] if b.role == "excluded"]
    hw = "вырос" if hs > 0 else ("снизился" if hs < 0 else "не изменился")
    amt = _fmt_u(hs, unit)
    dot = "" if amt.endswith(".") else "."
    lead = f"**По целевым метрикам ({', '.join(heads)}) портфель {hw} на {amt}{dot}**"
    if excl and es < 0 <= hs:
        lead += (f" Реальное падение — {_fmt_u(es, unit)} — сосредоточено в исключённых из доли "
                 f"категориях ({', '.join(excl)}), которые методологически нельзя смешивать с долей.")
    elif excl:
        w = "снизились" if es < 0 else "выросли"
        lead += f" Исключённые из доли ({', '.join(excl)}) {w} на {_fmt_u(es, unit)}."
    return lead


def _books_md(books, unit) -> str:
    L = ["## 📐 Методология: книги портфеля"]
    if books["note"]:
        L.append(f"_{books['note']}_\n")
    L.append("| Книга | 2025 | 2026 | Δ | Роль |")
    L.append("|---|---|---|---|---|")
    for b in books["books"]:
        v25 = _fmt_u(b.v25, unit) if b.v25 == b.v25 else "—"
        v26 = _fmt_u(b.v26, unit) if b.v26 == b.v26 else "—"
        L.append(f"| {b.label} | {v25} | {v26} | {_fmt_u(b.delta, unit)} | {_ROLE_RU.get(b.role, b.role)} |")
    L.append(f"\n**Вывод:** {_books_takeaway(books, unit)}\n")
    return "\n".join(L)


def _books_html(books, unit) -> str:
    import html as _h
    rows = "".join(
        f"<tr><td>{_h.escape(b.label)}</td>"
        f"<td>{_h.escape(_fmt_u(b.v25, unit) if b.v25 == b.v25 else '—')}</td>"
        f"<td>{_h.escape(_fmt_u(b.v26, unit) if b.v26 == b.v26 else '—')}</td>"
        f"<td>{_h.escape(_fmt_u(b.delta, unit))}</td>"
        f"<td>{_h.escape(_ROLE_RU.get(b.role, b.role))}</td></tr>" for b in books["books"])
    H = ["<h2>📐 Методология: книги портфеля</h2><div class='card'>"]
    if books["note"]:
        H.append(f"<p class='factline'>{_h.escape(books['note'])}</p>")
    H.append("<table><thead><tr><th>Книга</th><th>2025</th><th>2026</th><th>Δ</th><th>Роль</th>"
             f"</tr></thead><tbody>{rows}</tbody></table>")
    # Δ по каждой книге (headline=нейтраль, excluded=красный)
    labels = [b.label for b in books["books"]]
    vals = [round(b.delta, 1) for b in books["books"]]
    cols = ["neut" if b.role == "headline" else "loss" for b in books["books"]]
    spec = render.spec_bar_v(labels, vals, title="Δ по книгам портфеля (год к году)", colors=cols)
    H.append(render.chart("pl_books_delta", spec["option"], height=320))
    # Δ по сегментам для основных метрик
    ss = books.get("seg_series") or {}
    if ss:
        cats = list(next(iter(ss.values())).index)
        series = {lbl: [float(s.reindex(cats).fillna(0).loc[c]) for c in cats] for lbl, s in ss.items()}
        H.append(render.grouped_bar("pl_books_seg", [fmt_val(c) for c in cats], series,
                                    title="Δ по сегментам: основные метрики", unit=unit))
    H.append(f"<p class='insight'>💡 {_h.escape(_books_takeaway(books, unit))}</p>")
    H.append("</div>")
    return "".join(H)


def _where_rows(c, unit, top=6):
    """Строки раскладки по разрезу: значение | вклад | доля стороны. + «прочие»."""
    rows = [(fmt_val(v), _fmt_u(float(r["contrib"]), unit),
             f"{float(r['share']):.0f}%" if float(r["share"]) > 0 else "—")   # растущие: доли нет
            for v, r in c.table.head(top).iterrows()]
    rest = c.table.iloc[top:]
    if len(rest):
        rows.append((f"прочие ({len(rest)})", _fmt_u(float(rest["contrib"].sum()), unit), ""))
    return rows


def _assemble_md(prep, frame, table, fqn, question, where, net, gross_loss, gross_gain, unit,
                 primary, waterfall, pareto_chart, pareto_facts, tree, treemap, why, heat,
                 comps, who_tbl, who_chart, quadrant, when_chart, when_facts, drill_facts,
                 books, synth, where_dims=(), recovery=None, rec_dim=None, pivot=None) -> str:
    T = table + "_investigate"
    lb = prep.lbls
    def im(ch, alt=""):
        return f"![{alt}]({_rel_chart(T, ch)})\n" if ch else ""
    L = [f"# 🔎 Расследование: {prep.table_desc}", f"**Вопрос:** {labels_mod.clean_id_text(question)}\n"]
    meta = f"**Таблица:** `{fqn}` · **строк:** {len(prep.df):,}".replace(",", " ")
    if where:
        meta += f" · **фильтр:** `{where}`"
    L.append(meta + "\n")
    if synth.get("answer"):
        L.append("## 🎯 Ответ\n" + synth["answer"] + "\n")
    if books:
        L.append(_books_md(books, unit))
    # масштаб: net vs gross
    L.append("## 📊 Масштаб")
    if gross_loss is not None:
        L.append(f"- **Валовые потери:** {_fmt_u(gross_loss, unit)} · **приток:** {_fmt_u(gross_gain, unit)} "
                 f"· **net:** {_fmt_u(net, unit)}")
    else:
        L.append(f"- **{lb.of(frame.target)}:** {_fmt_u(net, unit)}")
    if pareto_facts:
        L.append(f"- **Концентрация:** топ-{pareto_facts['n_for_80']} «{lb.of(frame.entity)}» = 80% потерь "
                 f"(из {pareto_facts['total']}); топ-10 = {pareto_facts['top10_share']}%.")
    L.append("")
    L.append(im(waterfall, "водопад")); L.append(im(pareto_chart, "Парето"))
    if where_dims:
        L.append("## 📍 Где сосредоточено снижение")
        for d, c in where_dims:
            L.append(f"### По «{lb.col_title(d)}»")
            L.append(f"| {lb.col_title(d)} | {lb.of(frame.target)} | доля {_side(c.loss)} |")
            L.append("|---|--:|--:|")
            for v, contrib, sh in _where_rows(c, unit):
                L.append(f"| {v} | {contrib} | {sh} |")
            L.append("")
    if when_facts:
        L.append("## 🕒 Когда")
        line = f"- **Пик:** {when_facts['период_пик']} ({when_facts['вклад_пика']}, {when_facts['доля_пика_%']}% стороны)."
        if when_facts.get("слом"):
            line += f" **Резкий сдвиг:** {when_facts['слом']} ({when_facts['сдвиг_на_сломе']})."
        L.append(line)
        L.append(im(when_chart, "динамика")); L.append("")
    if comps:
        L.append("## 🧩 Из чего складывается")
        L += [f"- {name}: {_fmt_u(v, unit)}" for name, v in comps]
        L.append("")
    if drill_facts and len(drill_facts) >= 2:
        L.append("## 🧭 Горячий путь потерь")
        L.append("_Спуск в самый крупный вклад на каждом уровне (внутри предыдущего):_\n")
        for i, d in enumerate(drill_facts, 1):
            arrow = "\n   ↓" if i < len(drill_facts) else ""
            nrows = f"{d['строк']:,}".replace(",", " ")
            share = f"{d['доля_всех_%']}% всех потерь" if i == 1 else f"{d['доля_родителя_%']}% потерь родителя"
            L.append(f"{i}. **{d['разрез']} = {d['значение']}** — {d['вклад']} "
                     f"({share}, {nrows} строк){arrow}")
            L.append(im(d["chart"], d["dim"]))
        L.append("")
    if tree:
        L.append(f"## 🌳 Дерево потерь: «{lb.col_title(primary)}» → «{lb.col_title(frame.entity)}»")
        L.append(im(treemap, "treemap"))
        for n in tree:
            L.append(f"### {n.label} — {_fmt_u(n.contrib, unit)} ({n.share_of_total:.0f}% всех потерь)")
            for c in n.children[:5]:
                L.append(f"- {c.label}: {_fmt_u(c.contrib, unit)} ({c.share_in_parent:.0f}% узла)")
            if n.tail_count:
                L.append(f"- прочие ({n.tail_count}): {_fmt_u(n.tail_contrib, unit)}")
            L.append("")
    if why or heat:
        L.append("## ❓ Почему")
        for d, li, ch in why:
            if li.get("value"):
                L.append(f"### По «{lb.col_title(d)}»: **{li['value']}** — {li['loss_share']}% потерь "
                         f"при {li['base_share']}% строк (×{li['lift']} к базе)")
            else:
                L.append(f"### По «{lb.col_title(d)}» — разложение по частоте и влиянию")
            if li.get("breakdown"):
                L.append("")
                L.append(f"| {lb.col_title(d)} | строк | Δ {lb.of(frame.target)} |")
                L.append("|---|--:|--:|")
                for val, n, contrib in li["breakdown"]:
                    L.append(f"| {val} | {f'{n:,}'.replace(',', ' ')} | {_fmt_u(contrib, unit)} |")
                L.append("")
            L.append(im(ch, d))
        if heat:
            L.append(f"### Где какая причина: «{lb.of(primary)}» × «{lb.of(frame.why_dims[0])}»")
            L.append(im(heat, "heat"))
    if who_tbl is not None:
        L.append(f"## 👤 Кто + приоритет возврата ({lb.of(frame.entity)})")
        has_val = "value" in who_tbl.columns
        ctx = list(who_tbl.attrs.get("ctx", []))
        L.append(f"| {lb.of(frame.entity)} |" + "".join(f" {lb.of(c)} (осн. вклад) |" for c in ctx)
                 + f" {lb.of(frame.target)} |"
                 + (f" {lb.of(frame.value)} | покрытие |" if has_val else ""))
        L.append("|---|" + "---|" * (len(ctx) + 1) + ("--:|--:|" if has_val else ""))
        for idx, row in who_tbl.head(8).iterrows():
            line = (f"| {fmt_val(idx)} |" + "".join(f" {fmt_val(row[c])} |" for c in ctx)
                    + f" {_fmt_u(float(row['contrib']), unit)} |")
            if has_val:
                line += (f" {_fmt(float(row['value']))} |" if pd.notna(row['value']) else " — |")
                line += f" {_cover_txt(row.get('cover'))} |"
            L.append(line)
        L.append("")
    if pivot:
        L.append("## 📋 Сводная таблица")
        caps_txt = "/".join(map(str, pivot["caps"]))
        L.append(f"_Группировка: {' → '.join(pivot['dims'])}. Меры: {', '.join(pivot['measures'])}. "
                 f"Комбинаций: {pivot['total']}; показан топ по уровням {caps_txt} по «{pivot['key']}», "
                 f"остаток свёрнут в «прочие» — подытоги сходятся с полными данными._")
        if pivot["missing"]:
            L.append(f"\n> ⚠️ Нет в таблице, пропущены: {', '.join(pivot['missing'])}")
        L.append("")
        L.append("| " + " | ".join(pivot["dims"] + pivot["measures"]) + " |")
        L.append("|" + "---|" * len(pivot["dims"]) + "--:|" * len(pivot["measures"]))
        for path, vals in pivot["leaves"][:200]:
            L.append("| " + " | ".join(path) + " | " + " | ".join(_fmt(v) for v in vals) + " |")
        L.append("| " + " | ".join(["**ИТОГО**"] + [""] * (len(pivot["dims"]) - 1))
                 + " | " + " | ".join(f"**{_fmt(v)}**" for v in pivot["grand"]) + " |")
        L.append("")
    if recovery:
        L.append(f"## 🎯 Снижение vs потенциал в разрезе «{lb.col_title(rec_dim)}»")
        L.append(f"_Покрытие = потенциал ÷ |снижение|. «Покрыт» — потенциал ≥ {int(_COVER_OK*100)}% снижения._\n")
        L.append(f"| {lb.col_title(rec_dim)} | просевших клиентов | из них покрыты | снижение | потенциал | покрытие |")
        L.append("|---|--:|--:|--:|--:|--:|")
        for sv, n, ok, tl, tp, cov in recovery:
            L.append(f"| {sv} | {n} | {ok} | {_fmt_u(tl, unit)} | {_fmt(tp)} | {_cover_txt(cov)} |")
        L.append("")
        L.append("")
        L.append(im(quadrant, "квадрант"))          # who_chart убран: дублирует таблицу
    if synth.get("chain"):
        L.append("## 🧭 Причинная цепочка")
        L += [f"{i}. {x}" for i, x in enumerate(synth["chain"], 1)]
        L.append("")
    if synth.get("actions"):
        L.append("## ✅ Что делать")
        L += [f"- {a}" for a in synth["actions"]]
        L.append("")
    L.append("---\n_Расследование: pandas-декомпозиция + LLM-синтез._")
    return "\n".join(L)


def _assemble_html(prep, frame, table, fqn, question, where, net, gross_loss, gross_gain, unit,
                   primary, waterfall, pareto_chart, pareto_facts, tree, treemap, why, heat,
                   comps, who_tbl, who_chart, quadrant, when_chart, when_facts, drill_facts,
                   books, synth, where_dims=(), recovery=None, rec_dim=None, pivot=None) -> str:
    import html as _h
    lb = prep.lbls
    def im(ch):
        return render.embed(ch)                      # ECharts (сайдкар) или base64-PNG
    H = ["<!doctype html><html lang='ru'><head><meta charset='utf-8'>",
         f"<title>Расследование: {_h.escape(table)}</title><style>{_HTML_CSS}</style>",
         render.charts_head(), "</head><body>",
         f"<h1>🔎 Расследование: {_h.escape(prep.table_desc)}</h1>",
         f"<p class='angle'>{_h.escape(labels_mod.clean_id_text(question))}</p>"]
    meta = f"Таблица: <code>{_h.escape(fqn)}</code> · строк: {len(prep.df):,}".replace(",", " ")
    if where:
        meta += f" · фильтр: <code>{_h.escape(where)}</code>"
    H.append(f"<p class='meta'>{meta}</p>")
    if synth.get("answer"):
        H.append(f"<div class='card summary'><h2 style='border:none;margin-top:0'>🎯 Ответ</h2>"
                 f"<p class='insight'>{_h.escape(synth['answer'])}</p></div>")
    if books:
        H.append(_books_html(books, unit))
    # масштаб
    H.append("<h2>📊 Масштаб</h2><div class='card'>")
    if gross_loss is not None:
        H.append(f"<p><b>Валовые потери:</b> {_h.escape(_fmt_u(gross_loss, unit))} &nbsp;·&nbsp; "
                 f"<b>приток:</b> {_h.escape(_fmt_u(gross_gain, unit))} &nbsp;·&nbsp; "
                 f"<b>net:</b> {_h.escape(_fmt_u(net, unit))}</p>")
    if pareto_facts:
        H.append(f"<p class='factline'>Концентрация: топ-{pareto_facts['n_for_80']} «{_h.escape(lb.of(frame.entity))}» "
                 f"= 80% потерь (из {pareto_facts['total']}); топ-10 = {pareto_facts['top10_share']}%.</p>")
    H.append(im(waterfall) + im(pareto_chart) + "</div>")
    if where_dims:
        H.append("<h2>📍 Где сосредоточено снижение</h2>")
        for d, c in where_dims:
            rows = "".join(
                f"<tr><td>{_h.escape(v)}</td><td style='text-align:right'>{_h.escape(contrib)}</td>"
                f"<td style='text-align:right'>{_h.escape(sh)}</td></tr>"
                for v, contrib, sh in _where_rows(c, unit))
            H.append(f"<h3>По «{_h.escape(lb.col_title(d))}»</h3><div class='card'>"
                     f"<table><thead><tr><th>{_h.escape(lb.col_title(d))}</th>"
                     f"<th>{_h.escape(lb.of(frame.target))}</th>"
                     f"<th>доля {_h.escape(_side(c.loss))}</th></tr></thead>"
                     f"<tbody>{rows}</tbody></table></div>")
    if when_facts:
        line = (f"Пик: <b>{_h.escape(when_facts['период_пик'])}</b> "
                f"({_h.escape(when_facts['вклад_пика'])}, {when_facts['доля_пика_%']}% стороны).")
        if when_facts.get("слом"):
            line += (f" Резкий сдвиг: <b>{_h.escape(when_facts['слом'])}</b> "
                     f"({_h.escape(when_facts['сдвиг_на_сломе'])}).")
        H.append(f"<h2>🕒 Когда</h2><div class='card'><p class='factline'>{line}</p>"
                 + im(when_chart) + "</div>")
    if comps:
        H.append("<h2>🧩 Из чего складывается</h2><div class='card'><ul>"
                 + "".join(f"<li>{_h.escape(n)}: {_h.escape(_fmt_u(v, unit))}</li>" for n, v in comps) + "</ul></div>")
    if drill_facts and len(drill_facts) >= 2:
        H.append("<h2>🧭 Горячий путь потерь</h2><div class='card'>"
                 "<p class='meta'>Спуск в самый крупный вклад на каждом уровне (внутри предыдущего):</p><ol>")
        for i, d in enumerate(drill_facts, 1):
            nrows = f"{d['строк']:,}".replace(",", " ")
            share = f"{d['доля_всех_%']}% всех потерь" if i == 1 else f"{d['доля_родителя_%']}% потерь родителя"
            H.append(f"<li><b>{_h.escape(d['разрез'])} = {_h.escape(d['значение'])}</b> — "
                     f"{_h.escape(d['вклад'])} ({share}, {nrows} строк)</li>")
        H.append("</ol>" + "".join(im(d["chart"]) for d in drill_facts) + "</div>")
    if tree:
        H.append(f"<h2>🌳 Дерево потерь: «{_h.escape(lb.col_title(primary))}» → «{_h.escape(lb.col_title(frame.entity))}»</h2>")
        H.append("<div class='card'>" + im(treemap) + "</div>")
        for n in tree:
            H.append(f"<h3>{_h.escape(n.label)} — {_h.escape(_fmt_u(n.contrib, unit))} "
                     f"({n.share_of_total:.0f}% всех потерь)</h3><div class='card'><ul>")
            for c in n.children[:5]:
                H.append(f"<li>{_h.escape(c.label)}: {_h.escape(_fmt_u(c.contrib, unit))} "
                         f"({c.share_in_parent:.0f}% узла)</li>")
            if n.tail_count:
                H.append(f"<li>прочие ({n.tail_count}): {_h.escape(_fmt_u(n.tail_contrib, unit))}</li>")
            H.append("</ul></div>")
    if why or heat:
        H.append("<h2>❓ Почему</h2>")
        for d, li, ch in why:
            brk = ""
            if li.get("breakdown"):
                trows = "".join(
                    f"<tr><td>{_h.escape(val)}</td><td style='text-align:right'>{f'{n:,}'.replace(',', ' ')}</td>"
                    f"<td style='text-align:right'>{_h.escape(_fmt_u(contrib, unit))}</td></tr>"
                    for val, n, contrib in li["breakdown"])
                brk = (f"<table><thead><tr><th>{_h.escape(lb.col_title(d))}</th><th>строк</th>"
                       f"<th>Δ {_h.escape(lb.of(frame.target))}</th></tr></thead><tbody>{trows}</tbody></table>")
            hdr = (f"<p class='factline'><strong>{_h.escape(li['value'])}</strong> — "
                   f"{li['loss_share']}% потерь при {li['base_share']}% строк (×{li['lift']} к базе).</p>"
                   if li.get("value") else "")
            H.append(f"<h3>По «{_h.escape(lb.col_title(d))}»</h3><div class='card'>"
                     + hdr + brk + im(ch) + "</div>")
        if heat:
            H.append(f"<h3>Где какая причина: «{_h.escape(lb.of(primary))}» × «{_h.escape(lb.of(frame.why_dims[0]))}»</h3>"
                     f"<div class='card'>" + im(heat) + "</div>")
    if who_tbl is not None:
        has_val = "value" in who_tbl.columns
        H.append(f"<h2>👤 Кто + приоритет возврата ({_h.escape(lb.of(frame.entity))})</h2><div class='card pattern-card'>")
        ctx = list(who_tbl.attrs.get("ctx", []))
        rows = "".join(
            f"<tr><td>{_h.escape(fmt_val(idx))}</td>"
            + "".join(f"<td>{_h.escape(fmt_val(r[c]))}</td>" for c in ctx)
            + f"<td>{_h.escape(_fmt_u(float(r['contrib']), unit))}</td>"
            + ((f"<td>{_h.escape(_fmt(float(r['value'])) if pd.notna(r['value']) else '—')}</td>"
                f"<td>{_h.escape(_cover_txt(r.get('cover')))}</td>") if has_val else "")
            + "</tr>" for idx, r in who_tbl.head(8).iterrows())
        head = (f"<th>{_h.escape(lb.of(frame.entity))}</th>"
                + "".join(f"<th>{_h.escape(lb.of(c))} (осн. вклад)</th>" for c in ctx)
                + f"<th>{_h.escape(lb.of(frame.target))}</th>"
                + (f"<th>{_h.escape(lb.of(frame.value))}</th><th>покрытие</th>" if has_val else ""))
        H.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
        H.append(im(quadrant) + "</div>")          # who_chart убран: дублирует таблицу
    if pivot:
        # каждый разрез — СВОЯ колонка (раскладка «вбок»); у мер data-v = сырое число,
        # чтобы Tabulator сам считал подытоги групп. data-pivot = сколько колонок группировать.
        thead = ("".join(f"<th>{_h.escape(d)}</th>" for d in pivot["dims"])
                 + "".join(f"<th>{_h.escape(m)}</th>" for m in pivot["measures"]))
        prows = "".join(
            "<tr>" + "".join(f"<td>{_h.escape(c)}</td>" for c in path)
            + "".join(f"<td data-v='{v:.4f}' style='text-align:right'>{_h.escape(_fmt(v))}</td>"
                      for v in vals) + "</tr>"
            for path, vals in pivot["leaves"])
        note = (f"<p class='factline'>Группировка: {_h.escape(' → '.join(pivot['dims']))}. "
                f"Меры: {_h.escape(', '.join(pivot['measures']))}. "
                f"Комбинаций: {pivot['total']}; показан топ по уровням "
                f"{'/'.join(map(str, pivot['caps']))} по «{_h.escape(pivot['key'])}», остаток свёрнут "
                f"в «прочие», поэтому подытоги групп сходятся с полными данными. ИТОГО: "
                + " · ".join(f"<b>{_h.escape(_fmt(v))}</b>" for v in pivot["grand"]) + "</p>")
        if pivot["truncated"]:
            note += ("<p class='factline'>⚠️ Показана часть комбинаций (сработал предел строк) — "
                     "уменьшите число разрезов или укажите «топ N».</p>")
        if pivot["missing"]:
            note += (f"<p class='factline'>⚠️ Нет в таблице, пропущены: "
                     f"{_h.escape(', '.join(pivot['missing']))}</p>")
        # data-pivot = сколько ПЕРВЫХ колонок группировать. Группируем все разрезы КРОМЕ последнего:
        # иначе самый глубокий разрез дублировался бы (заголовок группы + идентичная строка-лист).
        # Последний разрез остаётся обычной колонкой — строки под группой и есть его значения.
        H.append("<h2>📋 Сводная таблица</h2><div class='card'>" + note
                 + f"<table data-pivot='{max(1, len(pivot['dims']) - 1)}'>"
                 f"<thead><tr>{thead}</tr></thead><tbody>{prows}</tbody></table></div>")
    if recovery:
        rows = "".join(
            f"<tr><td>{_h.escape(sv)}</td><td style='text-align:right'>{n}</td>"
            f"<td style='text-align:right'>{ok}</td>"
            f"<td style='text-align:right'>{_h.escape(_fmt_u(tl, unit))}</td>"
            f"<td style='text-align:right'>{_h.escape(_fmt(tp))}</td>"
            f"<td style='text-align:right'>{_h.escape(_cover_txt(cov))}</td></tr>"
            for sv, n, ok, tl, tp, cov in recovery)
        H.append(f"<h2>🎯 Снижение vs потенциал в разрезе «{_h.escape(lb.col_title(rec_dim))}»</h2>"
                 f"<div class='card'><p class='factline'>Покрытие = потенциал ÷ |снижение|. "
                 f"«Покрыт» — потенциал ≥ {int(_COVER_OK*100)}% снижения.</p>"
                 f"<table><thead><tr><th>{_h.escape(lb.col_title(rec_dim))}</th>"
                 f"<th>просевших клиентов</th><th>из них покрыты</th><th>снижение</th>"
                 f"<th>потенциал</th><th>покрытие</th></tr></thead>"
                 f"<tbody>{rows}</tbody></table></div>")
    if synth.get("chain"):
        H.append("<h2>🧭 Причинная цепочка</h2><div class='card'><ol>"
                 + "".join(f"<li>{_h.escape(x)}</li>" for x in synth["chain"]) + "</ol></div>")
    if synth.get("actions"):
        H.append("<div class='card attention'><h2 style='border:none;margin-top:0'>✅ Что делать</h2><ul>"
                 + "".join(f"<li>{_h.escape(a)}</li>" for a in synth["actions"]) + "</ul></div>")
    H.append("<p class='meta'>Расследование: pandas-декомпозиция + LLM-синтез.</p></body></html>")
    return interactive.enhance("\n".join(H))     # сортировка + фильтр таблиц (self-contained JS)
