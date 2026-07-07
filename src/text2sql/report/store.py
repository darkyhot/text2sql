"""FrameStore — реестр/кэш датафреймов и агрегаций (этап D).

Аналитика многократно считает одни и те же вещи: подмножество «где мера осмысленна»
(`_pop`), затем десятки `groupby` по нему. FrameStore держит базовый df, ленивo
материализует ИМЕНОВАННЫЕ производные подмножества и МЕМОИЗИРУЕТ результаты AggSpec —
одна и та же агрегация не считается дважды. Это и ускорение, и единая точка исполнения
для playbook'ов, собранных из `AggSpec`.

Кэш ключуется идентичностью датафрейма (`id`) и самим AggSpec (frozen → хэшируем),
поэтому агрегации по базе и по разным подмножествам не путаются.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .aggspec import AggResult, AggSpec, aggregate


class FrameStore:
    def __init__(self, base: pd.DataFrame):
        self.base = base
        self._subsets: dict[str, pd.DataFrame] = {}
        self._agg: dict[tuple[int, AggSpec], AggResult] = {}
        self.hits = 0
        self.misses = 0

    def subset(self, name: str, builder: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        """Именованное производное подмножество базы (напр. «мера X осмысленна»).
        Считается один раз, дальше отдаётся из кэша."""
        if name not in self._subsets:
            self._subsets[name] = builder(self.base)
        return self._subsets[name]

    def aggregate(self, spec: AggSpec, df: pd.DataFrame | None = None) -> AggResult:
        """Исполнить AggSpec с мемоизацией. По умолчанию — над базой; можно над подмножеством."""
        frame = self.base if df is None else df
        key = (id(frame), spec)
        cached = self._agg.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        res = aggregate(frame, spec)
        self._agg[key] = res
        return res

    @property
    def stats(self) -> dict[str, int]:
        return {"subsets": len(self._subsets), "agg_cached": len(self._agg),
                "hits": self.hits, "misses": self.misses}


if __name__ == "__main__":  # pragma: no cover
    df = pd.DataFrame({"seg": ["A", "A", "B"], "amt": [1, 2, 3], "ok": [1, 0, 1]})
    st = FrameStore(df)
    s = AggSpec(measure="amt", by=("seg",), agg="sum")
    r1 = st.aggregate(s)
    r2 = st.aggregate(s)                      # из кэша
    assert r1 is r2, "AggSpec должен мемоизироваться"
    sub = st.subset("ok", lambda d: d[d["ok"] == 1])
    assert len(sub) == 2 and st.subset("ok", lambda d: d) is sub  # второй builder игнорится (кэш)
    rp = st.aggregate(s, df=sub)              # агрегация по подмножеству — свой ключ
    assert rp is not r1
    assert st.stats["hits"] == 1 and st.stats["misses"] == 2, st.stats
    print("store self-test OK", st.stats)
