"""Смоук-тест семантической модели (§4): детерминированно, без БД и LLM.
Собираем модель из синтетических roles/measures/labels, проверяем классификацию, персистентность
(save/load + human_overrides) и мост в impact-плейбук (bindings → executor → вердикт).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.report import semantics as S  # noqa: E402
from text2sql.report.core import Roles  # noqa: E402
from text2sql.report.labels import Labels  # noqa: E402
from text2sql.report.metrics import Measure  # noqa: E402
from text2sql.report.plans import impact_verdict, load_playbooks, run_plan  # noqa: E402
from text2sql.report.store import FrameStore  # noqa: E402

FQN = "test_schema.sem_smoke"


class _FakeCat:
    def __init__(self, jc):
        self.join_candidates = jc

    def join_candidates_for(self, fqns):
        s = set(fqns)
        return [c for c in self.join_candidates if c["left"] in s or c["right"] in s]


def _fixture():
    rng = np.random.default_rng(0)
    n = 800
    region = rng.choice(["Москва", "СПб", "Казань"], size=n)      # geo
    reason = rng.choice(["закрытие счёта", "перевод", "прочее"], size=n)  # reason
    pipe = (rng.random(n) < 0.3).astype(int)                     # tool_flag
    post = rng.random(n) < 0.5
    rdt = np.where(post, pd.Timestamp("2025-09-01").value, pd.Timestamp("2025-03-01").value)
    close = (0.60 + 0.10 * (pipe.astype(bool) & post)).clip(0, 1)  # outcome-rate с uplift 0.10
    inn = rng.choice([f"{1000000000 + i}" for i in range(40)], size=n)
    company = pd.Series(inn).map({v: f"ООО-{i}" for i, v in enumerate(pd.unique(inn))}).to_numpy()
    df = pd.DataFrame({"region_name": region, "outflow_reason": reason, "is_in_pipeline": pipe,
                       "report_dt": pd.to_datetime(rdt), "task_dt": pd.to_datetime(rdt),
                       "close_rate": close, "inn": inn, "company_name": company})
    roles = Roles(dimensions=["region_name", "outflow_reason"], metrics=["close_rate"],
                  dates=["report_dt", "task_dt"], flags=["is_in_pipeline"],
                  entities=["company_name", "inn"],
                  card={"region_name": 3, "outflow_reason": 3, "company_name": 40, "inn": 40})
    measures = [Measure("Доля закрытий", "close_rate", "mean", "rate", "%", "Доля закрытий", "close_rate")]
    lbls = Labels({"region_name": "Территория", "outflow_reason": "Причина оттока",
                   "close_rate": "Доля закрытий", "company_name": "Компания", "inn": "ИНН",
                   "report_dt": "Отчётная дата", "task_dt": "Дата задачи"},
                  {"close_rate": "percent"})
    meta = {"is_in_pipeline": {"semantic_class": "flag", "desc": "Задача в инструменте пайплайна"},
            "task_dt": {"desc": "Дата постановки задачи"}}
    cat = _FakeCat([{"left": FQN, "right": "test_schema.dim_gosb",
                     "pairs": [{"left_col": "inn", "right_col": "inn", "overlap": 100}]}])
    return df, roles, measures, lbls, meta, cat


def main() -> int:
    ok = True
    df, roles, measures, lbls, meta, cat = _fixture()
    m = S.build_semantic_table(FQN, "Витрина оттока", df, roles, measures, lbls, meta=meta, catalog=cat)

    checks = {
        "geo-разрез": any(d.kind == "geo" and d.name == "region_name" for d in m.dimensions),
        "reason-разрез": any(d.kind == "reason" for d in m.dimensions),
        "сущность inn+name": any(e.name == "company_name" and e.key_col == "inn" for e in m.entities)
                             or any(e.name == "inn" and e.name_col == "company_name" for e in m.entities),
        "tool_flag найден": "is_in_pipeline" in m.tool_flags,
        "outcome-мера": "close_rate" in m.outcome_measures,
        "report_dt=reporting": any(t.name == "report_dt" and t.is_reporting for t in m.time),
        "task_dt=action": any(t.name == "task_dt" and not t.is_reporting for t in m.time),
        "join-кандидат": any(j.to_table == "test_schema.dim_gosb" for j in m.joins),
        "question_bank непустой": len(m.question_bank) > 0,
    }
    for name, val in checks.items():
        ok &= val
        print(f"[{'PASS' if val else 'FAIL'}] {name}")

    # --- персистентность + human_overrides ---
    m.human_overrides = {"audience_hint": "руководителям сети", "measure:Доля закрытий:unit": "percent"}
    path = S.save(m)
    m2 = S.load(FQN)
    r_persist = (m2 is not None and m2.row_count == 800
                 and m2.human_overrides.get("audience_hint") == "руководителям сети"
                 and m2.audience_hint == "руководителям сети")   # override применён
    ok &= r_persist
    print(f"[{'PASS' if r_persist else 'FAIL'}] round-trip JSON + human_overrides сохранены/применены")

    # пересборка не должна стереть human_overrides
    m3 = S.build_semantic_table(FQN, "Витрина оттока", df, roles, measures, lbls, meta=meta, catalog=cat)
    S.save(m3)
    m4 = S.load(FQN)
    r_keep = m4 is not None and m4.human_overrides.get("audience_hint") == "руководителям сети"
    ok &= r_keep
    print(f"[{'PASS' if r_keep else 'FAIL'}] пересборка сохранила human_overrides")

    # --- мост в impact-плейбук ---
    binds = S.impact_bindings(m, cutoff="2025-07-01")
    bridge_ok = binds is not None and binds["flag"] == "is_in_pipeline" and binds["outcome"] == "close_rate" \
        and binds["time_col"] == "report_dt"
    print(f"[{'PASS' if bridge_ok else 'FAIL'}] impact_bindings из модели: {binds}")
    ok &= bridge_ok
    if binds:
        pb = {p.name: p for p in load_playbooks()}["impact"]
        run = run_plan(FrameStore(df), pb.plan, binds)
        label, expl = impact_verdict(run)
        did = run.results["did"].facts["did"]
        verdict_ok = label == "подтверждается" and abs(did - 0.10) <= 0.03
        ok &= verdict_ok
        print(f"[{'PASS' if verdict_ok else 'FAIL'}] семмодель→impact: вердикт «{label}» · "
              f"DiD={did:+.3f} (ждём ≈+0.10) · {expl}")

    path.unlink(missing_ok=True)                                  # прибираем тестовый файл
    print("ИТОГ:", "семмодель OK" if ok else "есть провалы")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
