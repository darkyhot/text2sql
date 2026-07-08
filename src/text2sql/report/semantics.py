"""Слой 1 — семантическая модель таблицы (§4 дизайна), pydantic-схема + JSON-store.

Декларативное описание того, «что такое эти данные»: меры (единица, агрегат, good_direction,
scope), разрезы (вид, id-партнёр), сущности (ключ/имя/ценность), временны́е оси (отчётная ли,
гранулярность, свежесть), join-пути, а также tool_flags / outcome_measures / boring_cols /
question_bank — всё, что нужно планировщику дашбордов, расследованиям и impact-плейбуку.

Персистентность (§3.3): один JSON-файл на таблицу `data_for_agent/semantics/<fqn>.json`,
АТОМАРНАЯ запись (tmp → os.replace, закрывает B-22), `human_overrides` НИКОГДА не перетираются
при пересборке.

ЧЕСТНЫЕ УПРОЩЕНИЯ против §4.2:
- НЕ «один enrich-LLM-вызов»: модель СОБИРАЕТСЯ из уже работающих профайлеров (build_roles_llm/
  build_labels_llm/build_behaviors_llm — их выходы приходят как roles/measures/lbls) + детермини-
  рованной статистики. Консолидация трёх вызовов в один (§12) — отдельный шаг, здесь не делается.
- JoinPath берётся из join-кандидатов каталога (overlap); кардинальность НЕ верифицируется живой
  check_join — помечается "candidate" с verified_at=None (честно). N:1/1:1 — TODO живой пробой.
- good_direction / kind разрезов / outcome_measures / tool_flags / question_bank определяются
  ДЕТЕРМИНИРОВАННО (регэкспы по имени/описанию + статистика), без LLM. С llm=... позже можно
  уточнить — точка расширения.
- `human_overrides` применяются к скалярам верхнего уровня и по ключу `measure:<name>:<field>`
  (не произвольный dotted-path §4.1).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from ..config import PATHS
from .labels import Labels, fmt_val
from .metrics import ROW_COL

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "text2sql.semantic_model/2"

# ---- классификация по смыслу (детерминированно) ----
_GEO_RE = re.compile(r"(террит|регион|region|gosb|госб|\btб\b|\btb\b|oktmo|октмо|город|city|area|субъект)", re.I)
_ORG_RE = re.compile(r"(компан|холдинг|holding|company|инн|inn|огрн|ogrn|организ|client|клиент)", re.I)
_REASON_RE = re.compile(r"(причин|reason|повод|cause|основан|infopovod)", re.I)
_STATUS_RE = re.compile(r"(статус|status|признак|стади|stage|этап|фаза)", re.I)
_SOURCE_RE = re.compile(r"(источник|source|канал|channel)", re.I)
_NAME_RE = re.compile(r"(name|fio|наимен|компан|holder)", re.I)
_ID_RE = re.compile(r"(^inn$|_inn$|_id$|_code$|saphr|epk|ogrn|kpp)", re.I)
_FLAG_RE = re.compile(r"(^is_|^has_|_flag$|flag_|признак|in_pipeline|used|использ)", re.I)
# good_direction: где рост — ПЛОХО / ХОРОШО
_BAD_UP_RE = re.compile(r"(просроч|отток|потер|loss|churn|задолж|дефолт|ошиб|escal|эскалац|риск|жалоб|overdue)", re.I)
_GOOD_UP_RE = re.compile(r"(закрыт|успех|выполн|продаж|выруч|доход|приб|потенциал|конверс|revenue|success|closed)", re.I)
_OUTCOME_RE = re.compile(r"(закрыт|успех|выполн|просроч|срок|duration|closed|success|overdue|конверс|resolve)", re.I)
_ACTION_DATE_RE = re.compile(r"(задач|task|создан|created|заявк|обращен|звонок|визит|контакт)", re.I)


# ================= схема (§4.1) =================
class Measure(BaseModel):  # noqa: F811 — семантическая Measure (не metrics.Measure)
    name: str
    label: str
    column: str
    agg: Literal["sum", "mean", "count", "nunique"]
    kind: str                                       # money|count|rate|duration|value
    unit: Literal["money", "people", "count", "percent", "days", "other"]
    good_direction: Literal["up", "down", "none"] = "none"
    priority: int = 0
    scope_mask: str | None = None
    is_outcome: bool = False


class Dimension(BaseModel):
    name: str
    label: str
    card: int
    kind: Literal["category", "geo", "org", "reason", "status", "channel", "source", "tool_flag"] = "category"
    id_partner: str | None = None
    top_values: list[str] = Field(default_factory=list)


class Entity(BaseModel):
    name: str
    label: str
    key_col: str
    name_col: str | None = None
    value_col: str | None = None


class TimeCol(BaseModel):
    name: str
    label: str
    is_reporting: bool = True
    granularity: Literal["day", "month"] = "day"
    fresh_until: str | None = None


class JoinPath(BaseModel):
    to_table: str
    on: list[tuple[str, str]]
    cardinality: Literal["N:1", "1:1", "candidate"] = "candidate"
    verified_at: str | None = None
    enrich_cols: list[str] = Field(default_factory=list)
    overlap: int | None = None


class SemanticTable(BaseModel):
    schema_version: str = SCHEMA_VERSION
    fqn: str
    table: str
    label: str = ""
    grain: str = ""
    row_meaning: str = ""
    audience_hint: str = ""
    row_count: int = 0
    time: list[TimeCol] = Field(default_factory=list)
    measures: list[Measure] = Field(default_factory=list)
    dimensions: list[Dimension] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    joins: list[JoinPath] = Field(default_factory=list)
    outcome_measures: list[str] = Field(default_factory=list)
    tool_flags: list[str] = Field(default_factory=list)
    boring_cols: list[str] = Field(default_factory=list)
    question_bank: list[str] = Field(default_factory=list)
    human_overrides: dict[str, Any] = Field(default_factory=dict)

    def measure(self, name: str) -> Measure | None:
        return next((m for m in self.measures if m.name == name or m.column == name), None)


# ================= детерминированные помощники =================
def _good_direction(text: str) -> str:
    if _BAD_UP_RE.search(text):
        return "down"
    if _GOOD_UP_RE.search(text):
        return "up"
    return "none"


def _unit_of(m, lbls: Labels) -> str:
    """m — metrics.Measure (из профайлера). Единицу берём из Labels, иначе из kind меры."""
    uk = lbls.unit_kind(m.tech or m.col) if lbls else None
    if uk in ("money", "people", "count", "percent"):
        return uk
    return {"money": "money", "rate": "percent", "duration": "days",
            "count": "count", "value": "other"}.get(m.kind, "other")


def _dim_kind(col: str, desc: str, card: int, is_flag: bool) -> str:
    if is_flag:
        return "tool_flag"
    blob = f"{col} {desc}"
    if _GEO_RE.search(blob):
        return "geo"
    if _ORG_RE.search(blob):
        return "org"
    if _REASON_RE.search(blob):
        return "reason"
    if _STATUS_RE.search(blob):
        return "status"
    if _SOURCE_RE.search(blob):
        return "source"
    return "category"


def _is_binary(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    vals = pd.to_numeric(s, errors="coerce").dropna().unique()
    return len(vals) > 0 and set(vals).issubset({0, 1})


def _granularity(s: pd.Series) -> str:
    d = pd.to_datetime(s, errors="coerce").dropna()
    if d.empty:
        return "day"
    # если почти все даты — первое число месяца → месячная
    return "month" if float((d.dt.day == 1).mean()) > 0.6 else "day"


# ================= сборка модели =================
def build_semantic_table(fqn: str, table_desc: str, df: pd.DataFrame, roles, measures,
                         lbls: Labels, *, meta: dict | None = None, catalog=None) -> SemanticTable:
    meta = meta or {}
    table = fqn.split(".", 1)[1] if "." in fqn else fqn
    cols = set(df.columns)

    def _desc(c: str) -> str:
        return str((meta.get(c, {}) or {}).get("desc", "") or "")

    # ---- меры ----
    mspecs: list[Measure] = []
    outcome_names: list[str] = []
    for i, m in enumerate(measures):
        if m.col not in cols and m.col != ROW_COL:
            continue
        agg = "count" if (m.kind == "count" and m.col == ROW_COL) else m.agg
        text = f"{m.name} {m.label} {_desc(m.tech or m.col)}"
        gd = _good_direction(text)
        is_out = bool(m.kind in ("rate", "duration") or _OUTCOME_RE.search(text))
        sm = Measure(name=m.name, label=m.label, column=m.tech or m.col, agg=agg,
                     kind=m.kind, unit=_unit_of(m, lbls), good_direction=gd, priority=i,
                     is_outcome=is_out)
        mspecs.append(sm)
        if is_out:
            outcome_names.append(sm.column)

    # ---- разрезы ----
    dspecs: list[Dimension] = []
    for d in roles.dimensions:
        if d not in cols:
            continue
        card = int(roles.card.get(d, df[d].nunique(dropna=True)))
        is_flag = _FLAG_RE.search(d) is not None and _is_binary(df[d])
        top = [fmt_val(v) for v in df[d].value_counts().head(25).index]
        idp = None
        conc = re.sub(r"(_id|_name|_fio|_code)$", "", d, flags=re.I)
        for other in df.columns:
            if other != d and re.sub(r"(_id|_name|_fio|_code)$", "", other, flags=re.I) == conc \
                    and bool(_ID_RE.search(other)) != bool(_ID_RE.search(d)):
                idp = other
                break
        dspecs.append(Dimension(name=d, label=lbls.of(d), card=card,
                                kind=_dim_kind(d, _desc(d), card, is_flag), id_partner=idp,
                                top_values=top))

    # ---- сущности ----
    especs: list[Entity] = []
    value_col = next((m.column for m in mspecs if "потенциал" in m.label.lower()
                      or "potential" in m.column.lower()), None)
    for e in roles.entities:
        if e not in cols:
            continue
        is_name = bool(_NAME_RE.search(e) and not _ID_RE.search(e))
        key, name_col = (e, None)
        if is_name:                                   # имя — найдём id-партнёра как ключ
            partner = next((c for c in df.columns if _ID_RE.search(c)
                            and df[c].nunique(dropna=True) >= df[e].nunique(dropna=True)), None)
            key, name_col = (partner or e), e
        else:                                         # id — найдём name-партнёра
            name_col = next((c for c in df.columns if _NAME_RE.search(c) and not _ID_RE.search(c)
                             and re.sub(r"(_id|_name)$", "", c, flags=re.I)
                             == re.sub(r"(_id|_name)$", "", e, flags=re.I)), None)
        especs.append(Entity(name=e, label=lbls.of(e), key_col=key, name_col=name_col, value_col=value_col))

    # ---- временны́е оси ----
    tspecs = [TimeCol(name=d, label=lbls.of(d), is_reporting=not bool(_ACTION_DATE_RE.search(d + _desc(d))),
                      granularity=_granularity(df[d]),
                      fresh_until=(str(pd.to_datetime(df[d], errors="coerce").max().date())
                                   if pd.to_datetime(df[d], errors="coerce").notna().any() else None))
              for d in roles.dates if d in cols]

    # ---- tool_flags (булевы, доля true 2–80%, НЕ исход) ----
    tool_flags = []
    for c in df.columns:
        if not _is_binary(df[c]):
            continue
        share = float(pd.to_numeric(df[c], errors="coerce").fillna(0).mean())
        looks_flag = _FLAG_RE.search(c) or (meta.get(c, {}) or {}).get("semantic_class") == "flag"
        if looks_flag and 0.02 <= share <= 0.80 and not _OUTCOME_RE.search(c + _desc(c)):
            tool_flags.append(c)
    # инструменты-«внедрения» (pipeline/tool/использование) — вперёд: они и есть treatment
    tool_flags.sort(key=lambda c: 0 if re.search(r"(pipeline|tool|инструм|adopt|использ)", c, re.I) else 1)

    # ---- join-пути (кандидаты из каталога) ----
    joins: list[JoinPath] = []
    if catalog is not None and getattr(catalog, "join_candidates", None):
        for jc in catalog.join_candidates_for([fqn]):
            other = jc["right"] if jc["left"] == fqn else jc["left"]
            on = [(p["left_col"], p["right_col"]) for p in jc.get("pairs", [])]
            overlap = max((p.get("overlap", 0) for p in jc.get("pairs", [])), default=None)
            joins.append(JoinPath(to_table=other, on=on, cardinality="candidate", overlap=overlap))

    # ---- boring_cols ----
    boring = [c for c in df.columns
              if (meta.get(c, {}) or {}).get("semantic_class") == "free_text"]

    # ---- question_bank (шаблоны из мер×разрезов; без LLM) ----
    qbank = _question_bank(mspecs, dspecs, especs)

    return SemanticTable(
        fqn=fqn, table=table, label=table_desc, row_meaning="", grain="", audience_hint="",
        row_count=int(len(df)), time=tspecs, measures=mspecs, dimensions=dspecs, entities=especs,
        joins=joins, outcome_measures=outcome_names, tool_flags=tool_flags, boring_cols=boring,
        question_bank=qbank)


def _question_bank(measures: list[Measure], dims: list[Dimension], ents: list[Entity]) -> list[str]:
    qs: list[str] = []
    money = next((m for m in measures if m.unit == "money"), None)
    main = money or (measures[0] if measures else None)
    geo = next((d for d in dims if d.kind == "geo"), None)
    if main and dims:
        qs.append(f"Как «{main.label}» распределён по «{dims[0].label}»?")
    if money and geo:
        qs.append(f"Где сосредоточены «{money.label}» (по «{geo.label}»)?")
    out = next((m for m in measures if m.is_outcome), None)
    if out and geo:
        qs.append(f"Где «{out.label}» хуже среднего?")
    if ents:
        qs.append(f"Кто ключевые «{ents[0].label}» и какая концентрация?")
    if any(d.kind == "tool_flag" for d in dims) or True:
        qs.append("Насколько эффективен инструмент (сравнить с/без)?")
    return qs[:8]


# ================= персистентность (§3.3) =================
def semantics_dir() -> Path:
    d = PATHS.data_dir / "semantics"
    return d


def atomic_write_json(path: Path, data: dict) -> Path:
    """Атомарная запись JSON (tmp → os.replace) — единственная точка записи (B-22)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _model_path(fqn: str) -> Path:
    return semantics_dir() / f"{fqn}.json"


def apply_overrides(model: SemanticTable) -> SemanticTable:
    """Применить human_overrides: скаляры/списки верхнего уровня и `measure:<name>:<field>`."""
    for key, val in (model.human_overrides or {}).items():
        if key.startswith("measure:"):
            _, name, field = key.split(":", 2)
            m = model.measure(name)
            if m is not None and hasattr(m, field):
                setattr(m, field, val)
        elif hasattr(model, key):
            setattr(model, key, val)
    return model


def save(model: SemanticTable, *, preserve_overrides: bool = True) -> Path:
    """Сохранить модель. human_overrides из существующего файла СОХРАНЯЮТСЯ и применяются."""
    path = _model_path(model.fqn)
    if preserve_overrides and path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
            model.human_overrides = {**(prev.get("human_overrides") or {}), **(model.human_overrides or {})}
        except Exception:  # noqa: BLE001
            pass
    apply_overrides(model)
    return atomic_write_json(path, model.model_dump())


def load(fqn: str) -> SemanticTable | None:
    path = _model_path(fqn)
    if not path.exists():
        return None
    try:
        return SemanticTable.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantics: не прочитал %s: %s", path, exc)
        return None


# ================= мост к плейбукам (§8) =================
def impact_bindings(model: SemanticTable, *, cutoff: str | None = None) -> dict | None:
    """Собрать bindings для impact-плейбука из семантической модели: treatment-флаг (tool_flag,
    не исход), outcome (первая outcome-мера), reporting-дата, cutoff (середина периода/свежесть).
    Возвращает None, если чего-то критичного нет — тогда impact не применим."""
    flag = model.tool_flags[0] if model.tool_flags else None
    outcome = model.outcome_measures[0] if model.outcome_measures else None
    tcol = next((t for t in model.time if t.is_reporting), model.time[0] if model.time else None)
    if not (flag and outcome and tcol):
        return None
    cut = cutoff or tcol.fresh_until
    b = {"flag": flag, "outcome": outcome, "time_col": tcol.name, "cutoff": cut}
    geo = next((d.name for d in model.dimensions if d.kind == "geo"), None)
    if geo:
        b["strata"] = geo
    return b
