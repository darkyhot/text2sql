"""Узлы графа. Каждый — маленький фокус-шаг. LLM принимает решения, код
исполняет инструменты и предохранители. Маршрутизация — Command(goto=...),
human-in-the-loop — interrupt()."""

from __future__ import annotations

import logging
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
        try:
            plan = normalize_plan(StructuredPlan(**raw))
        except Exception as exc:  # noqa: BLE001
            logger.warning("node plan_build: план не распарсился: %s | raw=%s", exc, raw)
            return Command(goto="__end__", update={"status": "error",
                           "notes": [f"План не распарсился: {exc}"]})
        self._trace({"kind": "node", "node": "plan_build", "plan": plan.model_dump()})
        return Command(goto="resolve_values", update={"plan": plan.model_dump(), "facts": {"join_candidates": jc}})

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
            v = self.tools.check_join(lt.ref, rt.ref, [tuple(p) for p in j.on])
            if not v.ok:
                suggested = self.tools.suggest_join(lt.ref, rt.ref)
                if suggested and [tuple(p) for p in j.on] != suggested:
                    j.on = [list(p) for p in suggested]
                    v2 = self.tools.check_join(lt.ref, rt.ref, suggested)
                    if v2.ok:
                        notes.append(f"Ключ join автоматически исправлен на {j.on} (по overlap + PK справочника).")
                        v = v2
            j.classification, j.fanout_safe = v.classification, v.fanout_safe
            notes.append(v.note)
        return notes

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
        plan = StructuredPlan(**state["plan"])
        nl = render_nl(plan)
        sql_preview = render_sql_plain(plan)   # читаемый SQL для аналитика
        issues = validate_plan(plan)
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
        return Command(goto="plan_build", update={"corrections": corrections, "status": "planning"})

    # ---------- SYNTH + VALIDATE + EXECUTE ----------
    def synth(self, state: AgentState) -> Command:
        plan = StructuredPlan(**state["plan"])
        try:
            sql = render_sql(plan, self.db)
        except Exception as exc:  # noqa: BLE001
            return Command(goto="__end__", update={"status": "error", "notes": [f"Сборка SQL: {exc}"]})
        issues = validate_plan(plan)
        try:
            cost = self.db.explain_cost(sql)
        except Exception as exc:  # noqa: BLE001
            logger.warning("node synth: EXPLAIN не прошёл: %s | sql=%s", exc, sql)
            return Command(goto="__end__", update={"status": "error", "sql": sql,
                           "notes": [f"EXPLAIN не прошёл (вероятно ошибка в SQL): {exc}"]})
        return Command(goto="execute", update={"sql": sql, "sql_display": render_sql_plain(plan),
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
