"""AnalysisPlan — «план как данные» + executor (этап D, §8.2–8.3 дизайна).

Плейбук = заготовленный параметризуемый `AnalysisPlan` для КЛАССА вопросов (§8.3): цепочка
шагов-примитивов (`primitives.py`), исполняемая одним executor. Новый сценарий = новый
JSON-рецепт в `data_for_agent/playbooks/*.json`, а не новый модуль/команда. LLM (FRAME)
только подставляет параметры (`$target`, `$flag`, …); числа считают примитивы детерминированно.

Подстановка параметров в шаге:
  - `"$name"`          — параметр плейбука из bindings (заполняет FRAME/вызывающий код);
  - `"$stepid.facts.k"`— значение факта предыдущего шага (связи шагов, §8.2 `inputs`);
  - литерал            — как есть.
Если параметр-ссылка не разрешился (нет колонки/факта) — шаг ЧЕСТНО пропускается (не падаем):
на разных таблицах часть шагов неприменима (нет причины/даты/сущности).

ЧЕСТНЫЕ УПРОЩЕНИЯ против §8.2: `PlanStep.render` (WidgetSpec) не реализован — рендер плана
пока не подключён (см. §5.5 Plotly/Tabulator, отложено); маршрутизация в `/investigate`
(§8.1) не встроена — executor и рецепты работают автономно и покрыты самотестом.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PATHS
from . import primitives
from .store import FrameStore


@dataclass
class PlanStep:
    id: str
    primitive: str
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "PlanStep":
        return cls(id=d["id"], primitive=d["primitive"], params=dict(d.get("params", {})))


@dataclass
class AnalysisPlan:
    question: str
    steps: list[PlanStep]
    verdict_template: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisPlan":
        return cls(question=d.get("question", ""),
                   steps=[PlanStep.from_dict(s) for s in d.get("steps", [])],
                   verdict_template=d.get("verdict_template", ""))


@dataclass
class Playbook:
    name: str
    when: dict
    params: list[str]
    plan: AnalysisPlan

    @classmethod
    def from_dict(cls, d: dict) -> "Playbook":
        return cls(name=d["name"], when=d.get("when", {}), params=list(d.get("params", [])),
                   plan=AnalysisPlan.from_dict(d["plan"]))

    def matches(self, question: str, *, has_flag: bool = False) -> bool:
        """Срабатывание: паттерн вопроса + требования к модели (напр. наличие tool_flag)."""
        if self.when.get("requires_flag") and not has_flag:
            return False
        pats = self.when.get("question_patterns") or []
        q = (question or "").lower()
        return any(p.lower() in q for p in pats) if pats else True


@dataclass
class PlanRun:
    results: dict                 # step_id -> PrimitiveResult
    skipped: list                 # id шагов, пропущенных из-за неразрешённых параметров

    def facts(self) -> dict:
        return {sid: r.facts for sid, r in self.results.items()}


_REF = re.compile(r"^\$([A-Za-z0-9_]+)(?:\.facts\.([A-Za-z0-9_]+))?$")


def _resolve(value, bindings: dict, results: dict):
    """Разрешить одно значение параметра. Возвращает (resolved, is_missing_ref)."""
    if isinstance(value, (list, tuple)):
        out, miss = [], False
        for v in value:
            rv, m = _resolve(v, bindings, results)
            out.append(rv); miss = miss or m
        return out, miss
    if isinstance(value, str):
        m = _REF.match(value)
        if m:
            name, fact_key = m.group(1), m.group(2)
            if fact_key:                                  # $stepid.facts.key
                res = results.get(name)
                val = res.facts.get(fact_key) if res is not None else None
            else:                                         # $param из bindings
                val = bindings.get(name)
            return val, (val is None)
    return value, False


def run_plan(store: FrameStore, plan: AnalysisPlan, bindings: dict) -> PlanRun:
    """Исполнить план: подставить параметры, прогнать примитивы, собрать факты."""
    results: dict = {}
    skipped: list = []
    for step in plan.steps:
        params, missing = {}, False
        for k, v in step.params.items():
            rv, miss = _resolve(v, bindings, results)
            params[k] = rv
            missing = missing or miss
        if missing:
            skipped.append(step.id)
            continue
        results[step.id] = primitives.get(step.primitive)(store, **params)
    return PlanRun(results=results, skipped=skipped)


# ---------------- загрузка рецептов ----------------
def playbooks_dir() -> Path:
    return PATHS.data_dir / "playbooks" if hasattr(PATHS, "data_dir") else \
        Path(__file__).resolve().parents[3] / "data_for_agent" / "playbooks"


def load_playbooks(directory: Path | None = None) -> list[Playbook]:
    d = directory or playbooks_dir()
    out: list[Playbook] = []
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            out.append(Playbook.from_dict(json.loads(f.read_text(encoding="utf-8"))))
        except Exception:  # noqa: BLE001  (битый рецепт не должен ронять систему)
            continue
    return out


def select_playbook(question: str, *, has_flag: bool = False,
                    directory: Path | None = None) -> Playbook | None:
    """Роутинг §8.1: первый плейбук, чьи условия удовлетворены вопросом и моделью."""
    for pb in load_playbooks(directory):
        if pb.matches(question, has_flag=has_flag):
            return pb
    return None


def impact_verdict(run: PlanRun, *, min_pp: float = 0.02) -> tuple[str, str]:
    """Детерминированный вердикт impact-плейбука (§8.3 синтез): строго одна из форм —
    подтверждается / не подтверждается (самоотбор) / отрицательный / данных мало.
    Опирается на DiD (главная оценка) и selection_check (детект самоотбора)."""
    did_r = run.results.get("did")
    if did_r is None:
        return ("недостаточно данных", "нет DiD-оценки (нет даты внедрения или групп)")
    dv = float(did_r.facts["did"])
    naive_r = run.results.get("naive")
    nv = float(naive_r.facts["diff"]) if naive_r is not None else None
    sel = run.results.get("selection")
    strat = run.results.get("strat")
    supports = []
    if abs(dv) >= min_pp:
        supports.append(f"DiD {dv * 100:+.1f} п.п.")
    if strat is not None and abs(float(strat.facts.get("weighted_effect", 0.0))) >= min_pp:
        supports.append(f"стратификация {float(strat.facts['weighted_effect']) * 100:+.1f} п.п.")
    if dv >= min_pp:
        return ("подтверждается", "эффект " + ", ".join(supports))
    if dv <= -min_pp:
        return ("отрицательный", f"эффект {dv * 100:+.1f} п.п. (DiD)")
    # DiD ≈ 0
    if nv is not None and nv >= min_pp:
        gap = sel.facts.get("pre_gap") if sel is not None else None
        tail = (f"; пилоты были сильнее на {gap * 100:+.1f} п.п. ещё до внедрения"
                if gap is not None else "")
        return ("не подтверждается",
                f"наивная разница {nv * 100:+.1f} п.п. объясняется самоотбором{tail}")
    return ("недостаточно данных", "эффект не отличим от нуля")


if __name__ == "__main__":  # pragma: no cover
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(1)
    n = 400
    seg = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    diff = np.where(seg == "B", -rng.integers(50, 150, n), rng.integers(-5, 30, n)).astype(float)
    reason = np.where(seg == "B", "закрытие", rng.choice(["прочее", "перевод"], n))
    inn = rng.choice([f"inn{i}" for i in range(20)], size=n)
    treat = (seg == "A").astype(int)
    dt = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    outcome = (0.5 + 0.2 * (seg == "A") + 0.15 * ((dt >= "2025-07-01") & (treat == 1))).clip(0, 1)
    df = pd.DataFrame({"seg": seg, "diff": diff, "reason": reason, "inn": inn,
                       "treat": treat, "dt": dt, "closed": outcome})
    st = FrameStore(df)

    pbs = {p.name: p for p in load_playbooks()}
    assert "loss_attribution" in pbs and "impact" in pbs, list(pbs)

    # loss_attribution end-to-end
    la = run_plan(st, pbs["loss_attribution"].plan,
                  {"target": "diff", "dim": "seg", "entity": "inn", "reason": "reason",
                   "date": "dt", "side": "loss"})
    assert la.results["where"].facts["top_value"] == "B", la.facts()
    assert la.results["why"].facts["driver"] == "закрытие", la.facts()
    assert "when" in la.results and "who" in la.results, (la.results.keys(), la.skipped)

    # impact end-to-end
    im = run_plan(st, pbs["impact"].plan,
                  {"flag": "treat", "outcome": "closed", "time_col": "dt", "cutoff": "2025-07-01"})
    assert 0.08 < im.results["did"].facts["did"] < 0.22, im.facts()
    assert im.results["naive"].facts["diff"] > 0, im.facts()

    # роутинг по вопросу
    assert select_playbook("где мы потеряли клиентов").name == "loss_attribution"
    assert select_playbook("докажи эффективность инструмента", has_flag=True).name == "impact"
    assert select_playbook("докажи эффективность", has_flag=False) is None or \
        select_playbook("докажи эффективность", has_flag=False).name != "impact"

    print("plans self-test OK; skipped in loss_attribution:", la.skipped)
