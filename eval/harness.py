"""Eval-харнес: гоняет агента по эталонным кейсам с авто-апрувом, проверяет
свойства результата/плана. Поддерживает несколько прогонов на кейс — чтобы
измерять нестабильность слабой модели (например, частоту детекта неоднозначности).

Запуск:
    python eval/harness.py            # 1 прогон на кейс
    python eval/harness.py 3          # 3 прогона на кейс (pass-rate)
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.graph.agent import Agent  # noqa: E402

from cases import CASES  # noqa: E402


def run_to_completion(agent: Agent, question: str, ambiguity_pick: int):
    """Прогнать вопрос до конца, авто-апрувя план и выбирая опцию при неоднозначности."""
    turn = agent.ask(question)
    guard = 0
    while turn.interrupt and guard < 8:
        guard += 1
        typ = turn.interrupt.get("type")
        if typ == "ambiguity":
            turn = agent.respond(str(ambiguity_pick))
        elif typ == "approve_plan":
            turn = agent.respond("ok")
        else:
            break
    return turn


def grade_case(case: dict, run_idx: int) -> dict:
    agent = Agent(session=f"eval-{case['id']}-{run_idx}")
    try:
        turn = run_to_completion(agent, case["question"], case.get("ambiguity_pick", 0))
    except Exception as exc:  # noqa: BLE001
        return {"id": case["id"], "error": f"{exc}", "checks": [], "passed": False,
                "trace": traceback.format_exc()}
    results = []
    for chk in case["checks"]:
        try:
            ok, detail = chk(turn, agent.db)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"check error: {exc}"
        results.append({"name": chk.__name__, "ok": ok, "detail": detail})
    return {"id": case["id"], "status": turn.status, "checks": results,
            "notes": turn.notes, "sql": turn.state.get("sql", ""),
            "passed": all(r["ok"] for r in results) and turn.status == "done"}


def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Eval: {len(CASES)} кейсов × {runs} прогон(ов)\n" + "=" * 60)
    totals = {c["id"]: 0 for c in CASES}
    for case in CASES:
        for r in range(runs):
            res = grade_case(case, r)
            mark = "PASS" if res["passed"] else "FAIL"
            totals[case["id"]] += int(res["passed"])
            print(f"[{mark}] {res['id']} (run {r+1}/{runs}) status={res.get('status','?')}")
            for c in res.get("checks", []):
                flag = "✓" if c["ok"] else "✗"
                print(f"       {flag} {c['name']}: {c['detail']}")
            if res.get("notes"):
                print(f"       notes: {res['notes']}")
            if res.get("error"):
                print(f"       ERROR: {res['error']}")
    print("=" * 60 + "\nИТОГ (pass-rate):")
    for cid, passed in totals.items():
        print(f"  {cid}: {passed}/{runs}")


if __name__ == "__main__":
    main()
