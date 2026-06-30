"""Сборка графа + фасад Agent для ноутбука.

Цикл взаимодействия:
    agent = Agent()
    st = agent.ask("Сколько всего ТБ и ГОСБ?")   # -> прерывание на показе плана
    print(st.plan_nl)
    st = agent.respond("ok")                       # окнуть -> выполнить -> CSV
    # либо agent.respond("группируй ещё по сегменту")  # коррекция плана
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.types import Command

from ..catalog.catalog import Catalog
from ..config import LLM, PATHS, LLMConfig
from ..db.adapter import make_adapter
from ..llm.client import LLMClient
from ..tools.toolbox import Toolbox
from ..trace import Tracer
from .nodes import Nodes
from .state import AgentState


def build_graph(nodes: Nodes, checkpointer: MemorySaver):
    g = StateGraph(AgentState)
    g.add_node("scope", nodes.scope)
    g.add_node("clarify", nodes.clarify)
    g.add_node("plan_build", nodes.plan_build)
    g.add_node("join_advisor", nodes.join_advisor)
    g.add_node("present", nodes.present)
    g.add_node("synth", nodes.synth)
    g.add_node("execute", nodes.execute)
    g.set_entry_point("scope")
    # маршрутизация полностью через Command(goto=...) внутри узлов
    return g.compile(checkpointer=checkpointer)


@dataclass
class Turn:
    """Снимок результата хода для ноутбука."""
    status: str
    interrupt: dict[str, Any] | None
    plan_nl: str
    result: dict[str, Any] | None
    notes: list[str]
    state: dict[str, Any]

    @property
    def plan_text(self) -> str:
        return self.plan_nl

    def __repr__(self) -> str:
        if self.interrupt:
            t = self.interrupt.get("type")
            if t == "approve_plan":
                return f"<Turn awaiting_approval>\n{self.interrupt.get('plan_nl','')}"
            if t == "ambiguity":
                opts = "\n".join(f"  [{i}] {o.get('label')}" for i, o in enumerate(self.interrupt.get("options", [])))
                return f"<Turn ambiguity> {self.interrupt.get('question')}\n{opts}"
        if self.status == "done" and self.result:
            return f"<Turn done> rows={self.result.get('rowcount')} csv={self.result.get('csv')}"
        return f"<Turn {self.status}> notes={self.notes}"


class Agent:
    def __init__(self, *, catalog: Catalog | None = None, session: str | None = None):
        self.tracer = Tracer(session)
        self.catalog = catalog or Catalog.load()
        self.db = make_adapter(tracer=self.tracer)
        self.llm = LLMClient(tracer=self.tracer)
        self.tools = Toolbox(self.catalog, self.db, tracer=self.tracer)
        self.nodes = Nodes(self.catalog, self.db, self.llm, self.tools,
                           workspace_dir=PATHS.workspace_dir, tracer=self.tracer)
        self.app = build_graph(self.nodes, MemorySaver())
        self._qid = 0
        self._thread = {"configurable": {"thread_id": f"{self.tracer.session_id}-0"}}

    def _rewire(self) -> None:
        """Пересобрать tools/nodes/graph после смены каталога/LLM/БД."""
        self.tools = Toolbox(self.catalog, self.db, tracer=self.tracer)
        self.nodes = Nodes(self.catalog, self.db, self.llm, self.tools,
                           workspace_dir=PATHS.workspace_dir, tracer=self.tracer)
        self.app = build_graph(self.nodes, MemorySaver())

    def reload_catalog(self) -> None:
        """Перечитать каталог после рефреша метаданных (для /refresh_metadata)."""
        self.catalog = Catalog.load()
        self._rewire()

    def reload_db(self) -> None:
        """Перечитать config_db.json и пересоздать engine (после /config_db_conn)."""
        self.db.reload()
        self._rewire()

    def set_llm(self, provider: str, model: str) -> None:
        """Сменить LLM (для /model). GigaChat читает токены из env."""
        cfg = LLMConfig(provider=provider, base_url=LLM.base_url, api_key=LLM.api_key,
                        model=model, max_tokens=LLM.max_tokens, temperature=LLM.temperature)
        self.llm = LLMClient(cfg=cfg, tracer=self.tracer)
        self._rewire()

    def ask(self, question: str) -> Turn:
        # Новый thread на каждый вопрос — чистый контекст в REPL.
        self._qid += 1
        self._thread = {"configurable": {"thread_id": f"{self.tracer.session_id}-{self._qid}"}}
        init: AgentState = {"question": question, "corrections": [], "notes": [], "status": "scoping"}
        return self._run(self.app.invoke(init, self._thread))

    def respond(self, text: str) -> Turn:
        """Ответ на прерывание: 'ok' для выполнения, иначе текст коррекции/выбора."""
        return self._run(self.app.invoke(Command(resume=text), self._thread))

    def _run(self, raw: dict[str, Any]) -> Turn:
        snap = self.app.get_state(self._thread)
        interrupts = _pending_interrupt(snap)
        st = snap.values
        return Turn(
            status=st.get("status", "?"),
            interrupt=interrupts,
            plan_nl=st.get("plan_nl", "") or (interrupts.get("plan_nl", "") if interrupts else ""),
            result=st.get("result"),
            notes=st.get("notes", []),
            state=st,
        )


def _pending_interrupt(snap) -> dict[str, Any] | None:
    tasks = getattr(snap, "tasks", ()) or ()
    for task in tasks:
        for intr in (getattr(task, "interrupts", ()) or ()):
            val = getattr(intr, "value", None)
            if isinstance(val, dict):
                return val
    return None
