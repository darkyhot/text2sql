"""Узлы графа. Каждый — маленький фокус-шаг. LLM принимает решения, код
исполняет инструменты и предохранители. Маршрутизация — Command(goto=...),
human-in-the-loop — interrupt()."""

from __future__ import annotations

import logging
import re
from typing import Any

from langgraph.types import Command, interrupt

logger = logging.getLogger(__name__)

from ..catalog.catalog import Catalog
from ..db.adapter import DbAdapter
from ..llm.client import LLMClient
from ..plan.model import StructuredPlan
from ..plan.render import normalize_plan, render_nl, render_sql, render_sql_plain, validate_plan
from ..tools.toolbox import Toolbox
from . import prompts
from .state import AgentState

MAX_CORRECTIONS = 6
_OK_WORDS = {"ok", "ок", "да", "окей", "go", "поехали", "выполняй", "согласен", ""}


def _concept_id(name: str) -> str:
    """Сущность идентификатора: токен перед _id. new_gosb_id/old_gosb_id → 'gosb'."""
    parts = name.lower().split("_")
    return parts[-2] if len(parts) >= 2 and parts[-1] == "id" else name.lower()


def _is_id_col(cm) -> bool:
    n = cm.name.lower()
    return (cm.semantic_class in ("join_key", "identifier") or n.endswith("_id")
            or n in ("inn", "kpp", "ogrn"))


def _is_texty(dtype: str) -> bool:
    d = (dtype or "").lower()
    return "char" in d or "text" in d


def _best_match(term: str, values: list[str]) -> str:
    """Из найденных значений выбрать лучшее под term (подстрока, регистронезависимо)."""
    t = term.lower()
    contains = [v for v in values if t in v.lower()]
    pool = contains or values
    return min(pool, key=len)  # самое короткое совпадение — обычно точнее


class Nodes:
    def __init__(self, catalog: Catalog, db: DbAdapter, llm: LLMClient, tools: Toolbox, *,
                 workspace_dir, tracer=None):
        self.catalog = catalog
        self.db = db
        self.llm = llm
        self.tools = tools
        self.workspace = workspace_dir
        self._trace = tracer or (lambda e: None)

    # ---------- SCOPE ----------
    def scope(self, state: AgentState) -> Command:
        q = state["question"]
        corrections = state.get("corrections", [])
        logger.info("node scope: вопрос=%r правок=%d", q, len(corrections))
        candidates = self.tools.search_tables(q, k=5)
        out = self.llm.complete_json(prompts.SCOPE_SYS, prompts.scope_user(q, corrections, candidates),
                                     max_tokens=4000, node="scope")
        chosen = [t for t in out.get("chosen_tables", []) if self.catalog.get(t)]
        ambiguity = out.get("ambiguity")
        self._trace({"kind": "node", "node": "scope", "chosen": chosen, "ambiguity": bool(ambiguity)})

        # Неоднозначность решает отдельный детерминированный узел clarify: нельзя
        # держать interrupt после недетерминированного LLM-вызова — при возобновлении
        # узел переисполнится, опции переупорядочатся и индекс выбора «съедет».
        if ambiguity and ambiguity.get("options"):
            return Command(goto="clarify", update={"candidates": candidates, "ambiguity": ambiguity,
                           "chosen_tables": chosen, "status": "awaiting_user",
                           "ambiguity_surfaced": True})
        if not chosen:
            return Command(goto="__end__", update={"status": "error",
                           "notes": ["Не удалось выбрать таблицы для вопроса."]})
        return Command(goto="plan_build", update={"candidates": candidates, "chosen_tables": chosen,
                       "ambiguity": None, "status": "planning"})

    # ---------- CLARIFY (детерминированный interrupt по неоднозначности) ----------
    def clarify(self, state: AgentState) -> Command:
        ambiguity = state["ambiguity"]
        options = ambiguity["options"]
        choice = interrupt({"type": "ambiguity", "question": ambiguity.get("question", ""),
                            "options": options})
        picked = self._resolve_ambiguity(options, choice)
        # Оставляем только реально существующие в каталоге таблицы.
        chosen = [t for t in picked if self.catalog.get(t)] or state.get("chosen_tables", [])
        logger.info("node clarify: выбор=%r -> таблицы=%s", choice, chosen)
        if not chosen:
            return Command(goto="__end__", update={"status": "error",
                           "notes": ["Не удалось разрешить неоднозначность."]})
        return Command(goto="plan_build", update={"chosen_tables": chosen,
                       "ambiguity": None, "status": "planning"})

    @staticmethod
    def _resolve_ambiguity(options: list[dict], choice: Any) -> list[str]:
        def _tables(opt: dict) -> list[str]:
            t = opt.get("tables", [])
            return [t] if isinstance(t, str) else list(t)

        if isinstance(choice, int) and 0 <= choice < len(options):
            return _tables(options[choice])
        s = str(choice).strip().lower()
        # индекс 0-based (как в выводе CLI: [0], [1]) или подстрока метки
        for i, opt in enumerate(options):
            if s == str(i) or (s and s in opt.get("label", "").lower()):
                return _tables(opt)
        return []

    # ---------- PLAN BUILD ----------
    def plan_build(self, state: AgentState) -> Command:
        q = state["question"]
        corrections = state.get("corrections", [])
        tables = [self.catalog.get(f) for f in state["chosen_tables"] if self.catalog.get(f)]
        jc = self.catalog.join_candidates_for([t.fqn for t in tables])
        raw = self.llm.complete_json(prompts.PLAN_SYS, prompts.plan_user(q, corrections, tables, jc),
                                     max_tokens=3500, node="plan_build")
        # RAW-SQL люк: сложные конструкции (окна/подзапросы/множества) вне SPJA
        raw_sql = str(raw.get("raw_sql") or "").strip()
        if raw_sql:
            logger.info("node plan_build: RAW-SQL режим")
            return Command(goto="present", update={
                "raw_sql": raw_sql, "plan": None,
                "raw_meta": {"intent": str(raw.get("intent", "")), "note": str(raw.get("note", ""))},
                "facts": {"join_candidates": jc}, "status": "planning"})
        try:
            plan = normalize_plan(StructuredPlan(**raw))
        except Exception as exc:  # noqa: BLE001
            logger.warning("node plan_build: план не распарсился: %s | raw=%s", exc, raw)
            return Command(goto="__end__", update={"status": "error",
                           "notes": [f"План не распарсился: {exc}"]})
        corr_notes = self._correct_count_ids(plan, q)   # выбор идентификатора для COUNT DISTINCT
        self._trace({"kind": "node", "node": "plan_build", "plan": plan.model_dump()})
        return Command(goto="resolve_values", update={"plan": plan.model_dump(), "raw_sql": "",
                       "notes": corr_notes, "facts": {"join_candidates": jc}})

    def _correct_count_ids(self, plan: StructuredPlan, question: str) -> list[str]:
        """Выбор ПРАВИЛЬНОГО идентификатора для COUNT DISTINCT сущности — решает LLM
        с учётом ВОПРОСА (уважает явный «по old_gosb_id») и описаний колонок.
        Вызывается только при реальной неоднозначности (≥2 id-колонки одной сущности),
        поэтому LLM-вызов редок. По умолчанию — актуальный id, не исторический/технический."""
        notes: list[str] = []
        for m in plan.metrics:
            if m.agg not in ("count", "count_distinct") or not m.column or m.column.endswith("*"):
                continue
            alias = m.column.split(".", 1)[0] if "." in m.column else (plan.tables[0].alias if plan.tables else None)
            col = m.column.split(".")[-1]
            tref = plan.table_by_alias(alias) if alias else None
            tmeta = self.catalog.get(tref.ref) if tref else None
            if not tmeta or not next((c for c in tmeta.columns if c.name == col), None):
                continue
            # кандидаты той же сущности (для группировки — эвристика; выбор делает LLM)
            concept = _concept_id(col)
            cands = [c for c in tmeta.columns if _concept_id(c.name) == concept and _is_id_col(c)]
            if len(cands) < 2:            # выбора нет — LLM не зовём
                continue
            try:
                out = self.llm.complete_json(prompts.COUNT_ID_SYS,
                                             prompts.count_id_user(question, col, cands),
                                             max_tokens=1500, node="count_id")
                chosen = str(out.get("column", "")).strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("count-id корректор: LLM не ответил (%s)", exc)
                continue
            chosen = next((c.name for c in cands if c.name == chosen), None)
            if chosen and chosen != col:
                m.column = f"{alias}.{chosen}" if alias else chosen
                notes.append(f"Для подсчёта выбран идентификатор {chosen} (вместо {col}).")
                logger.info("count-id корректор: %s → %s", col, chosen)
        return notes

    # ---------- RESOLVE VALUES (живой резолв текстовых фильтров) ----------
    def resolve_values(self, state: AgentState) -> Command:
        """Сопоставить значения текстовых фильтров с реальными значениями в БД
        (живая проба distinct_values). Закрывает кейс «task_subtype='фактический
        отток'»: значение берётся из данных, а не из устаревшей метадаты. Если
        значение лежит в ДРУГОЙ категориальной колонке — переключает фильтр на неё."""
        plan = StructuredPlan(**state["plan"])
        notes: list[str] = list(state.get("notes", []))
        changed = False
        for f in plan.filters:
            if f.op not in ("=", "ILIKE") or not isinstance(f.value, str) or not f.value.strip():
                continue
            alias = f.column.split(".", 1)[0] if "." in f.column else None
            col = f.column.split(".", 1)[-1]
            tref = plan.table_by_alias(alias) if alias else (plan.tables[0] if plan.tables else None)
            tmeta = self.catalog.get(tref.ref) if tref else None
            if not tmeta:
                continue
            colmeta = next((c for c in tmeta.columns if c.name == col), None)
            if colmeta and not _is_texty(colmeta.dtype):
                continue  # резолвим только текстовые колонки
            term = f.value.strip().strip("%")
            # 1) пробуем выбранную колонку
            vals = self.tools.distinct_values(tref.ref, col, like=term, n=10, enforce_cost=False)
            if vals:
                best = _best_match(term, vals)
                if f"%{best}%" != f.value or f.op != "ILIKE":
                    notes.append(f"Значение '{f.value}' сопоставлено с '{best}' (колонка {col}).")
                f.op, f.value, f.resolved_via = "ILIKE", f"%{best}%", "probe"
                changed = True
                continue
            # 2) ищем понятие в других категориальных колонках таблицы (мог ошибиться столбцом)
            found = None
            for c in tmeta.columns:
                if c.name == col or not _is_texty(c.dtype):
                    continue
                if c.semantic_class not in ("enum_like", "label", "free_text"):
                    continue
                alt = self.tools.distinct_values(tref.ref, c.name, like=term, n=5, enforce_cost=False)
                if alt:
                    found = (c.name, _best_match(term, alt))
                    break
            if found:
                newcol, newval = found
                notes.append(f"Значение '{f.value}' не найдено в {col}; найдено в {newcol}='{newval}' — переключаю фильтр.")
                f.column = f"{alias}.{newcol}" if alias else newcol
                f.op, f.value, f.resolved_via = "ILIKE", f"%{newval}%", "probe"
                changed = True
            else:
                notes.append(f"⚠ Значение '{f.value}' не найдено в БД (колонки таблицы) — проверьте фильтр.")
        if changed:
            self._trace({"kind": "node", "node": "resolve_values", "filters": [f.model_dump() for f in plan.filters]})
        return Command(goto="join_advisor", update={"plan": plan.model_dump(), "notes": notes})

    # ---------- JOIN ADVISOR (анти-fan-out) ----------
    def join_advisor(self, state: AgentState) -> Command:
        plan = StructuredPlan(**state["plan"])
        if not plan.joins:
            return Command(goto="present", update={})
        notes = self._verify_fix_joins(plan)
        # если что-то всё ещё небезопасно/невалидно — последняя попытка LLM-фикса
        if any((j.fanout_safe is False) for j in plan.joins):
            plan = self._fix_joins(state, plan)
            notes += self._verify_fix_joins(plan)
        self._trace({"kind": "node", "node": "join_advisor",
                     "joins": [(j.left_alias, j.right_alias, j.classification, j.fanout_safe) for j in plan.joins]})
        return Command(goto="present", update={"plan": plan.model_dump(), "notes": notes})

    def _verify_fix_joins(self, plan: StructuredPlan) -> list[str]:
        """Проверить каждый join (типы + fan-out). Если небезопасен/невалиден —
        ДЕТЕРМИНИРОВАННО подобрать ключ из join-графа + PK справочника (не полагаясь
        на LLM), затем перепроверить."""
        notes: list[str] = []
        for j in plan.joins:
            lt, rt = plan.table_by_alias(j.left_alias), plan.table_by_alias(j.right_alias)
            if not lt or not rt:
                continue
            right_cols = {rc for _, rc in j.on}
            left_cols = {lc for lc, _ in j.on}
            # 1) источник уже дедуплицирован по ключам join → N:1 гарантированно
            if rt.dedup and rt.dedup.by and set(rt.dedup.by) <= right_cols:
                j.classification, j.fanout_safe = "N:1", True
                notes.append(f"Источник {rt.alias} дедуплицирован по {rt.dedup.by} → join N:1 (без размножения).")
                continue
            if lt.dedup and lt.dedup.by and set(lt.dedup.by) <= left_cols:
                j.classification, j.fanout_safe = "N:1", True
                notes.append(f"Источник {lt.alias} дедуплицирован по {lt.dedup.by} → join N:1.")
                continue
            v = self.tools.check_join(lt.ref, rt.ref, [tuple(p) for p in j.on])
            if not v.ok:
                # 2) детерминированный подбор ключа (покрыть PK справочника)
                suggested = self.tools.suggest_join(lt.ref, rt.ref)
                if suggested and [tuple(p) for p in j.on] != suggested:
                    v2 = self.tools.check_join(lt.ref, rt.ref, suggested)
                    if v2.ok:
                        j.on = [list(p) for p in suggested]
                        notes.append(f"Ключ join исправлен на {j.on} (overlap + PK справочника).")
                        v = v2
                # 3) не вышло (join по неуникальному атрибуту справочника) → авто-dedup:
                #    свежая строка справочника на ключ (по дате актуальности)
                if not v.ok:
                    fixed = self._auto_dedup(j, lt, rt)
                    if fixed:
                        j.classification, j.fanout_safe = "N:1", True
                        notes.append(fixed)
                        continue
            j.classification, j.fanout_safe = v.classification, v.fanout_safe
            notes.append(v.note)
        return notes

    def _auto_dedup(self, j, lt, rt) -> str | None:
        """Фолбэк: join к справочнику по неуникальному атрибуту (напр. inn) — вместо
        размножения дедуплицируем справочную сторону по ключу (свежая карточка)."""
        best = self.tools.best_join_pair(lt.ref, rt.ref)
        if not best:
            return None
        lc, rc = best
        lmeta, rmeta = self.catalog.get(lt.ref), self.catalog.get(rt.ref)
        # какую сторону дедупить: справочную (role reference), иначе правую
        if rmeta and rmeta.role == "reference":
            side, col = rt, rc
        elif lmeta and lmeta.role == "reference":
            side, col = lt, lc
        else:
            side, col = rt, rc
        recency = self.tools.recency_col(side.ref)
        if not recency:
            return None
        from ..plan.model import Dedup
        j.on = [[lc, rc]]
        side.dedup = Dedup(by=[col], order_by=recency, desc=True)
        return (f"Справочник {side.alias} неуникален по «{col}» — беру СВЕЖУЮ строку "
                f"на «{col}» (по {recency} ↓), join по {j.on}. Размножения строк не будет.")

    def _fix_joins(self, state: AgentState, plan: StructuredPlan) -> StructuredPlan:
        bad = next((j for j in plan.joins if j.fanout_safe is False), None)
        if bad is None:
            return plan
        pk_by_alias = {t.alias: (self.catalog.get(t.ref).pk_hypothesis if self.catalog.get(t.ref) else [])
                       for t in plan.tables}
        jc = state.get("facts", {}).get("join_candidates", [])
        try:
            fix = self.llm.complete_json(
                prompts.JOIN_FIX_SYS,
                prompts.join_fix_user(plan.model_dump(), "N:M размножение строк", jc, pk_by_alias),
                max_tokens=1500, node="join_fix")
            new_joins = fix.get("joins")
            if new_joins:
                plan.joins = [type(plan.joins[0])(**j) for j in new_joins]
        except Exception as exc:  # noqa: BLE001
            self._trace({"kind": "node", "node": "join_fix", "error": str(exc)})
        return plan

    # ---------- PRESENT (human-in-the-loop) ----------
    def present(self, state: AgentState) -> Command:
        raw_sql = state.get("raw_sql")
        if raw_sql:
            rm = state.get("raw_meta", {})
            nl = (f"Намерение: {rm.get('intent','')}\n"
                  f"Режим: прямой SQL (сложный запрос — окна/подзапросы/множества).\n"
                  f"{rm.get('note','')}")
            issues: list[str] = []
            nl_full, sql_preview = nl, raw_sql
        else:
            plan = StructuredPlan(**state["plan"])
            nl_full = render_nl(plan)
            sql_preview = render_sql_plain(plan)   # читаемый SQL для аналитика
            issues = validate_plan(plan)
        nl = nl_full
        # логируем то, что показываем пользователю (чтобы не слать с экрана)
        logger.info("ПЛАН показан пользователю:\n%s\n-- SQL --\n%s\n-- заметки: %s%s",
                    nl, sql_preview, state.get("notes", []),
                    ("\n-- проблемы: " + str(issues)) if issues else "")
        decision = interrupt({"type": "approve_plan", "plan_nl": nl, "sql": sql_preview,
                              "issues": issues, "notes": state.get("notes", [])})
        text = str(decision).strip()
        if text.lower() in _OK_WORDS:
            return Command(goto="synth", update={"plan_nl": nl, "status": "approved"})
        corrections = state.get("corrections", []) + [text]
        if len(corrections) > MAX_CORRECTIONS:
            return Command(goto="__end__", update={"status": "error",
                           "notes": ["Слишком много итераций коррекции — остановка."]})
        return Command(goto="plan_build", update={"corrections": corrections, "raw_sql": "",
                       "status": "planning"})

    # ---------- SYNTH + VALIDATE + EXECUTE ----------
    def synth(self, state: AgentState) -> Command:
        raw_sql = state.get("raw_sql")
        if raw_sql:                          # RAW-режим: SQL уже готов, только валидируем
            sql, sql_display, issues = raw_sql, raw_sql, []
        else:
            plan = StructuredPlan(**state["plan"])
            try:
                sql = render_sql(plan, self.db)
            except Exception as exc:  # noqa: BLE001
                return Command(goto="__end__", update={"status": "error", "notes": [f"Сборка SQL: {exc}"]})
            sql_display = render_sql_plain(plan)
            issues = validate_plan(plan)
        try:
            cost = self.db.explain_cost(sql)
        except Exception as exc:  # noqa: BLE001
            logger.warning("node synth: EXPLAIN не прошёл: %s | sql=%s", exc, sql)
            return Command(goto="__end__", update={"status": "error", "sql": sql,
                           "notes": [f"EXPLAIN не прошёл (вероятно ошибка в SQL): {exc}"]})
        return Command(goto="execute", update={"sql": sql, "sql_display": sql_display,
                       "validation": {"issues": issues, "explain_cost": cost}})

    def execute(self, state: AgentState) -> Command:
        sql = state["sql"]
        logger.info("node execute: sql=%s", sql)
        try:
            # Финальная выгрузка: полный результат (без probe-LIMIT и cost-потолка).
            res = self.db.run_export(sql)
        except Exception as exc:  # noqa: BLE001
            logger.warning("node execute: ошибка выполнения: %s | sql=%s", exc, sql)
            return Command(goto="__end__", update={"status": "error", "notes": [f"Выполнение: {exc}"]})
        path = self.workspace / "last_query.csv"
        self._write_csv(path, res.columns, res.rows)
        self._trace({"kind": "node", "node": "execute", "rowcount": res.rowcount, "csv": str(path)})
        logger.info("РЕЗУЛЬТАТ: строк=%d, колонки=%s, csv=%s | первые строки: %s",
                    res.rowcount, res.columns, path, res.rows[:3])
        return Command(goto="__end__", update={"status": "done",
                       "result": {"csv": str(path), "rowcount": res.rowcount,
                                  "columns": res.columns, "truncated": res.truncated}})

    def _write_csv(self, path, columns: list[str], rows: list[dict]) -> None:
        import csv
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=columns)
            w.writeheader()
            w.writerows(rows)
