"""Засев тестового контейнера РЕАЛИСТИЧНЫМИ данными.

Зачем: прежняя синтетика рандомизировала строки, поэтому реальных значений
(напр. task_subtype='Фактический отток') в БД не было — value-резолв и баги
выбора колонки/значения не воспроизводились локально на DeepSeek.

Этот скрипт берёт пулы реальных значений из кураторского бэкапа
(data_for_agent_original_backup) и генерирует данные так, чтобы:
  • работали эталонные запросы (Q1: distinct tb/gosb; Q2: join fact↔dim по
    (tb_id, old_gosb_id); Q3: task_subtype='Фактический отток' и fact.is_task в фев-2026);
  • строковые колонки имели ПОВТОРЯЮЩИЕСЯ реальные значения (тогда и PK по сэмплу
    считается корректно — author_login не уникален; и value-резолв есть на чём проверять);
  • присутствовала реальная неоднозначность: task_type содержит 'Отток' (широкий),
    task_subtype — 'Фактический отток' (конкретный).

Запуск:  python scripts/seed_realistic_db.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2sql.config import DB  # noqa: E402

random.seed(42)
SCHEMA = "s_grnplm_ld_salesntwrk_pcap_sn_uzp"
BACKUP = ROOT / "data_for_agent_original_backup"

TB = [
    (2, "Центральный аппарат", "ЦА"), (13, "Байкальский банк", "ББ"),
    (16, "Уральский банк", "УБ"), (18, "Поволжский банк", "ПБ"),
    (38, "Юго-Западный банк", "ЮЗБ"), (40, "Дальневосточный банк", "ДВБ"),
    (42, "Волго-Вятский банк", "ВВБ"), (44, "Московский банк", "МБ"),
    (52, "Сибирский банк", "СибБ"), (54, "Среднерусский банк", "СРБ"),
    (55, "Северо-Западный банк", "СЗБ"), (70, "Центрально-Черноземный банк", "ЦЧБ"),
]
AUTHORS = [f"0{n} (Сотрудник {n})" for n in range(3496217, 3496222)]
# Системные load-таймстемпы: малый набор (как реальные батч-загрузки), НЕ уникальны —
# иначе ложно становятся PK. Имена *_dttm берут значения отсюда.
LOAD_TS = [dt.datetime(2026, 5, 19, h, m) for h, m in [(3, 0), (4, 1), (8, 0)]]


def load_pools() -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {}
    rr = json.loads((BACKUP / "rule_registry.json").read_text(encoding="utf-8"))
    for r in rr["rules"]:
        key = r["column_key"].split(".")[-1]
        vals = [v for v in r.get("value_candidates", []) if v and not v.islower()]
        if vals:
            pools.setdefault(key, list(dict.fromkeys(vals)))
    with (BACKUP / "attr_list.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            c, sv = row["column_name"], row["sample_values"]
            if sv and c not in pools:
                pools[c] = [x for x in sv.split("|") if x]
    return pools


def rand_date(feb_share: float = 0.15) -> dt.date:
    if random.random() < feb_share:
        return dt.date(2026, 2, random.randint(1, 28))
    start = dt.date(2025, 1, 1)
    return start + dt.timedelta(days=random.randint(0, 540))


def rand_dttm() -> dt.datetime:
    d = rand_date(0.1)
    return dt.datetime(d.year, d.month, d.day, random.randint(0, 23), random.randint(0, 59))


def columns(cur, table: str) -> list[tuple[str, str, int | None]]:
    cur.execute(
        "SELECT column_name, data_type, character_maximum_length "
        "FROM information_schema.columns "
        "WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
        (SCHEMA, table),
    )
    return [(r[0], r[1], r[2]) for r in cur.fetchall()]


class Gen:
    def __init__(self, pools, inns):
        self.pools = pools
        self.inns = inns

    def value(self, name: str, dtype: str, maxlen: int | None = None):
        d = dtype.lower()
        n = name.lower()
        if "bool" in d:
            return random.random() < 0.4
        if d == "date":
            return rand_date()
        if "timestamp" in d or d == "time":
            # системные *_dttm — из малого набора load-таймстемпов (не уникальны)
            if n.endswith("dttm") or any(k in n for k in ("insert", "modif", "updat", "load")):
                return random.choice(LOAD_TS)
            return rand_dttm()
        if n in self.pools and ("char" in d or "text" in d):
            return self._fit(random.choice(self.pools[n]), maxlen)
        if any(t in d for t in ("int", "smallint", "bigint")):
            if "inn" in n:
                return random.choice(self.inns)
            return random.randint(1, 5000)
        if any(t in d for t in ("numeric", "decimal", "double", "real")):
            return round(random.uniform(0, 1_000_000), 2)
        if "char" in d or "text" in d:
            return self._fit(random.choice(self.pools.get(n, ["Прочее", "Не указано", "—"])), maxlen)
        return None

    @staticmethod
    def _fit(value: str, maxlen: int | None) -> str:
        # уважаем varchar(N): обрезаем (для коротких кодовых полей — генерим цифры)
        if maxlen is None or len(value) <= maxlen:
            return value
        if maxlen <= 3:
            return "".join(random.choice("0123456789") for _ in range(maxlen))
        return value[:maxlen]


def build_rows(cur, table, n, gen, overrides):
    cols = columns(cur, table)
    rows = []
    for i in range(n):
        ov = overrides(i)
        row = []
        for cname, dtype, maxlen in cols:
            row.append(ov[cname] if cname in ov else gen.value(cname, dtype, maxlen))
        rows.append(row)
    return [c for c, _, _ in cols], rows


def insert(cur, table, colnames, rows):
    cur.execute(f'TRUNCATE TABLE "{SCHEMA}"."{table}"')
    collist = ", ".join(f'"{c}"' for c in colnames)
    execute_values(cur, f'INSERT INTO "{SCHEMA}"."{table}" ({collist}) VALUES %s', rows, page_size=500)


def main() -> None:
    pools = load_pools()
    inns = [random.randint(1_000_000_000, 9_999_999_999) for _ in range(200)]
    gen = Gen(pools, inns)
    conn = psycopg2.connect(DB.dsn)
    conn.autocommit = False
    cur = conn.cursor()

    # 1) dim_gosb: old_gosb_id ПОВТОРЯЕТСЯ между разными tb_id (как в реальности —
    # номера ГОСБ не глобально уникальны). Уникален только составной (tb_id, old_gosb_id).
    # Поэтому join fact↔dim по одному gosb_id даёт размножение строк, нужен и tb_id.
    old_pool = list(range(1000, 1035))  # 35 номеров ГОСБ на 60 пар (есть повторы между ТБ)
    gosb = []  # (tb_id, old_gosb_id, new_gosb_id)
    seen_pairs: set = set()
    while len(gosb) < 60:
        tb_id = random.choice(TB)[0]
        old_id = random.choice(old_pool)
        if (tb_id, old_id) in seen_pairs:
            continue
        seen_pairs.add((tb_id, old_id))
        gosb.append((tb_id, old_id, 2000 + len(gosb)))  # new_gosb_id уникален

    def dim_ov(i):
        tb_id, old_id, new_id = gosb[i]
        name = next(t for t in TB if t[0] == tb_id)
        return {"tb_id": tb_id, "old_gosb_id": old_id, "new_gosb_id": new_id,
                "tb_full_name": name[1], "tb_short_name": name[2],
                "author_login": random.choice(AUTHORS)}
    c, r = build_rows(cur, "uzp_dim_gosb", len(gosb), gen, dim_ov)
    insert(cur, "uzp_dim_gosb", c, r)

    # 2) fact_outflow: ссылается на dim (tb_id, gosb_id=old_gosb_id); PK (gosb_id, inn, report_dt)
    seen = set()
    def fact_ov(i):
        tb_id, old_id, _ = random.choice(gosb)
        while True:
            inn = random.choice(inns); rd = rand_date(0.2)
            if (old_id, inn, rd) not in seen:
                seen.add((old_id, inn, rd)); break
        return {"tb_id": tb_id, "gosb_id": old_id, "inn": inn, "report_dt": rd,
                "is_task": random.random() < 0.5, "author_login": random.choice(AUTHORS),
                "segment_name": random.choice(pools.get("segment_name", ["ММБ"]))}
    c, r = build_rows(cur, "uzp_dwh_fact_outflow", 1500, gen, fact_ov)
    insert(cur, "uzp_dwh_fact_outflow", c, r)

    # 3) sale_funnel_task: task_code уникален; task_subtype вкл. 'Фактический отток'
    def task_ov(i):
        tb_id, old_id, _ = random.choice(gosb)
        return {"task_code": f"T{i:07d}", "tb_id": tb_id, "gosb_id": old_id,
                "inn": random.choice(inns), "task_create_dt": rand_date(0.2),
                "report_dt": rand_date(0.2),
                "task_subtype": random.choice(pools["task_subtype"]),
                "task_type": random.choice(pools["task_type"])}
    c, r = build_rows(cur, "uzp_dwh_sale_funnel_task", 1500, gen, task_ov)
    insert(cur, "uzp_dwh_sale_funnel_task", c, r)

    # 4) epk_consolidation: epk_id уникален
    def epk_ov(i):
        tb_id, old_id, _ = random.choice(gosb)
        return {"epk_id": 5_000_000 + i, "tb_id": tb_id, "gosb_id": old_id,
                "inn": random.choice(inns)}
    c, r = build_rows(cur, "uzp_data_epk_consolidation", 800, gen, epk_ov)
    insert(cur, "uzp_data_epk_consolidation", c, r)

    conn.commit()
    cur.close(); conn.close()
    print("Готово. Реалистичные данные засеяны.")
    print("  task_subtype содержит 'Фактический отток':", "Фактический отток" in pools["task_subtype"])
    print("  task_type содержит 'Отток' (широкий):", "Отток" in pools["task_type"])


if __name__ == "__main__":
    main()
