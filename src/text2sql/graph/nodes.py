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
from ..plan.render import normalize_plan, render_nl, render_sql, validate_plan
from ..tools.toolbox import Toolbox
from . import prompts
from .state import AgentState

MAX_CORRECTIONS = 6
_OK_WORDS = {"ok", "ок", "да", "окей", "go", "поехали", "выполняй", "согласен", ""}


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
        chosen = self._resolve_ambiguity(options, choice) or state.get("chosen_tables", [])
        if not chosen:
            return Command(goto="__end__", update={"status": "error",
                           "notes": ["Не удалось разрешить неоднозначность."]})
        return Command(goto="plan_build", update={"chosen_tables": chosen,
                       "ambiguity": None, "status": "planning"})

    @staticmethod
    def _resolve_ambiguity(options: list[dict], choice: Any) -> list[str]:
        if isinstance(choice, int) and 0 <= choice < len(options):
            return options[choice].get("tables", [])
        s = str(choice).strip().lower()
        for i, opt in enumerate(options):
            if s == str(i) or s in opt.get("label", "").lower():
                return opt.get("tables", [])
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
        return Command(goto="join_advisor", update={"plan": plan.model_dump(), "facts": {"join_candidates": jc}})

    # ---------- JOIN ADVISOR (анти-fan-out) ----------
    def join_advisor(self, state: AgentState) -> Command:
        plan = StructuredPlan(**state["plan"])
        if not plan.joins:
            return Command(goto="present", update={})
        notes = self._annotate_joins(plan)
        if any(j.fanout_safe is False for j in plan.joins):
            plan = self._fix_joins(state, plan)
            notes += self._annotate_joins(plan)
        self._trace({"kind": "node", "node": "join_advisor",
                     "joins": [(j.left_alias, j.right_alias, j.classification, j.fanout_safe) for j in plan.joins]})
        return Command(goto="present", update={"plan": plan.model_dump(), "notes": notes})

    def _annotate_joins(self, plan: StructuredPlan) -> list[str]:
        notes: list[str] = []
        for j in plan.joins:
            lt, rt = plan.table_by_alias(j.left_alias), plan.table_by_alias(j.right_alias)
            if not lt or not rt:
                continue
            v = self.tools.check_join(lt.ref, rt.ref, [tuple(p) for p in j.on])
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
        issues = validate_plan(plan)
        decision = interrupt({"type": "approve_plan", "plan_nl": nl,
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
        return Command(goto="execute", update={"sql": sql,
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
