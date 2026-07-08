"""Eval impact-плейбука (§10.5 дизайна): система обязана
  1) на кейсе с ПОДЛОЖЕННЫМ uplift — подтвердить эффект с точностью ±2 п.п.;
  2) на кейсе с ЧИСТЫМ САМООТБОРОМ (флаг у заведомо сильных, реального эффекта нет) —
     ОТКЛОНИТЬ с формулировкой про самоотбор.
Детерминированно, без LLM и без БД: фикстуры генерятся в памяти.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.report.plans import impact_verdict, load_playbooks, run_plan  # noqa: E402
from text2sql.report.store import FrameStore  # noqa: E402

CUTOFF = "2025-07-01"


def _impact_plan():
    pbs = {p.name: p for p in load_playbooks()}
    assert "impact" in pbs, f"нет impact-плейбука: {list(pbs)}"
    return pbs["impact"].plan


def _run(df: pd.DataFrame, *, strata=None, placebo=None):
    binds = {"flag": "treat", "outcome": "closed", "time_col": "dt", "cutoff": CUTOFF}
    if strata:
        binds["strata"] = strata
    if placebo:
        binds["placebo"] = placebo
    return run_plan(FrameStore(df), _impact_plan(), binds)


def case_genuine_uplift(uplift=0.08, n=1500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.choice(["R1", "R2", "R3"], size=n)
    treat = (rng.random(n) < 0.35).astype(int)              # пилот во всех регионах, независимо
    post = rng.random(n) < 0.5
    base = pd.Series(region).map({"R1": 0.55, "R2": 0.60, "R3": 0.62}).to_numpy()
    closed = (base + uplift * (treat.astype(bool) & post)).clip(0, 1)
    dt = np.where(post, pd.Timestamp("2025-09-01").value, pd.Timestamp("2025-03-01").value)
    return pd.DataFrame({"region": region, "treat": treat, "closed": closed,
                         "placebo": base + rng.normal(0, 0.01, n), "dt": pd.to_datetime(dt)})


def case_self_selection(n=1500, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.choice(["R1", "R2", "R3"], size=n)
    strong = rng.random(n) < 0.3
    treat = strong.astype(int)                              # инструмент берут заведомо сильные
    post = rng.random(n) < 0.5
    closed = (0.55 + 0.20 * strong).clip(0, 1)             # константа во времени: эффекта НЕТ
    dt = np.where(post, pd.Timestamp("2025-09-01").value, pd.Timestamp("2025-03-01").value)
    return pd.DataFrame({"region": region, "treat": treat, "closed": closed,
                         "placebo": 0.5 + rng.normal(0, 0.01, n), "dt": pd.to_datetime(dt)})


def main() -> int:
    ok = True

    # --- Кейс 1: реальный uplift +8 п.п. ---
    run1 = _run(case_genuine_uplift(uplift=0.08), strata="region", placebo="placebo")
    label1, expl1 = impact_verdict(run1)
    did1 = run1.results["did"].facts["did"]
    c1 = (label1 == "подтверждается") and (abs(did1 - 0.08) <= 0.02)
    ok &= c1
    print(f"[{'PASS' if c1 else 'FAIL'}] uplift-кейс: вердикт «{label1}» · DiD={did1:+.3f} "
          f"(ждём ≈+0.08 ±0.02) · {expl1}")
    print(f"    selection pre_gap={run1.results['selection'].facts['pre_gap']:+.3f} "
          f"(ждём ≈0) · placebo_did={run1.results['placebo'].facts['placebo_did']:+.3f} · "
          f"экстраполяция +{run1.results['extrapolate'].facts['projected']:.0f} задач")

    # --- Кейс 2: чистый самоотбор, эффекта нет ---
    run2 = _run(case_self_selection(), strata="region", placebo="placebo")
    label2, expl2 = impact_verdict(run2)
    did2 = run2.results["did"].facts["did"]
    naive2 = run2.results["naive"].facts["diff"]
    sel2 = run2.results["selection"].facts
    c2 = (label2 == "не подтверждается") and (abs(did2) <= 0.02) and sel2["self_selection"]
    ok &= c2
    print(f"[{'PASS' if c2 else 'FAIL'}] самоотбор-кейс: вердикт «{label2}» · DiD={did2:+.3f} "
          f"(ждём ≈0) · наивно={naive2:+.3f} · pre_gap={sel2['pre_gap']:+.3f} (ждём >0) · {expl2}")

    print("ИТОГ:", "2/2 impact-проверок пройдены" if ok else "есть провалы")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
