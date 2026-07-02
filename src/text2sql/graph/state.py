"""Состояние графа. Персистентно между ходами (checkpointer) — цикл
план→коррекция дёшев: перезапускается только нужная часть, без переэксплоринга."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str                      # исходный вопрос
    corrections: list[str]             # накопленные правки пользователя

    candidates: list[dict[str, Any]]   # результат search_tables (Tier-0)
    chosen_tables: list[str]           # выбранные fqn
    ambiguity: dict[str, Any] | None   # сурфейс неоднозначности (для interrupt)
    ambiguity_surfaced: bool           # была ли неоднозначность показана (для eval)

    facts: dict[str, Any]              # describe_table выбранных таблиц + заметки
    plan: dict[str, Any] | None        # StructuredPlan.model_dump()
    raw_sql: str                       # прямой SQL (окна/подзапросы/множества — вне SPJA)
    raw_meta: dict[str, Any]           # {intent, note} для raw-режима
    plan_nl: str                       # человекочитаемый рендер (то, что окают)

    sql: str                           # исполняемый SQL (с кавычками идентификаторов)
    sql_display: str                   # читаемый SQL без кавычек (для показа)
    validation: dict[str, Any]
    result: dict[str, Any]             # {csv, rowcount, columns}

    status: str                        # scoping|exploring|planning|awaiting_user|done|error
    notes: list[str]                   # диагностические сообщения для пользователя
    step_count: int
