"""Пересоздание и засев тестового контейнера РЕАЛИСТИЧНЫМИ данными по манифесту.

Generic: читает список таблиц и колонок из data_for_agent/attr_list.csv (актуальная
прод-метадата), пересоздаёт таблицы (DROP+CREATE) и засевает реалистичными значениями
из кураторских пулов (data_for_agent_original_backup + new_tables). Сохраняет связки
для эталонных запросов и бизнес-отчёта:
  • tb_id+gosb_id в fact-таблицах ссылаются на справочник uzp_dim_gosb (join работает);
  • old_gosb_id повторяется между ТБ → нужен составной ключ (анти-fanout);
  • task_subtype содержит 'Фактический отток', task_type — широкий 'Отток';
  • числовые метрики (amt/qty/perc/fot) в правдоподобных диапазонах — для отчёта;
  • даты покрывают 2025..2026 с долей февраля-2026.

Запуск:  python scripts/seed_realistic_db.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2sql.config import DB, PATHS  # noqa: E402

random.seed(42)
SCHEMA = "s_grnplm_ld_salesntwrk_pcap_sn_uzp"
BACKUP = ROOT / "data_for_agent_original_backup"
NROWS = {  # объём засева по таблицам (fact — крупнее, для отчёта)
    "uzp_dim_gosb": 60, "uzp_data_epk_consolidation": 800,
    "uzp_dwh_fact_outflow": 4000, "uzp_dwh_sale_funnel_task": 4000,
    "uzp_data_payroll_m": 6000, "uzp_dwh_company_holding_metric": 4000,
}
DEFAULT_ROWS = 3000

TB = [(2, "Центральный аппарат", "ЦА"), (13, "Байкальский банк", "ББ"),
      (16, "Уральский банк", "УБ"), (18, "Поволжский банк", "ПБ"),
      (38, "Юго-Западный банк", "ЮЗБ"), (40, "Дальневосточный банк", "ДВБ"),
      (42, "Волго-Вятский банк", "ВВБ"), (44, "Московский банк", "МБ"),
      (52, "Сибирский банк", "СибБ"), (54, "Среднерусский банк", "СРБ"),
      (55, "Северо-Западный банк", "СЗБ"), (70, "Центрально-Черноземный банк", "ЦЧБ")]
AUTHORS = [f"0{n} (Сотрудник {n})" for n in range(3496217, 3496222)]
LOAD_TS = [dt.datetime(2026, 5, 19, h, m) for h, m in [(3, 0), (4, 1), (8, 0)]]
ORG_TYPES = ["Холдинг", "Компания", "Группа компаний"]

_DDL_TYPE = {
    "bigint": "BIGINT", "integer": "INTEGER", "smallint": "SMALLINT",
    "character varying": "VARCHAR", "text": "TEXT", "boolean": "BOOLEAN",
    "numeric": "NUMERIC", "date": "DATE", "timestamp without time zone": "TIMESTAMP",
}


def load_pools() -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {}
    rr = json.loads((BACKUP / "rule_registry.json").read_text(encoding="utf-8"))
    for r in rr["rules"]:
        key = r["column_key"].split(".")[-1]
        vals = [v for v in r.get("value_candidates", []) if v and not v.islower()]
        if vals:
            pools.setdefault(key, list(dict.fromkeys(vals)))
    for src in (BACKUP / "attr_list.csv", ROOT / "data_for_agent" / "attr_list.csv"):
        if src.exists():
            for row in csv.DictReader(src.open(encoding="utf-8")):
                c, sv = row["column_name"], row.get("sample_values", "")
                if sv and c not in pools:
                    pools[c] = [x for x in sv.split("|") if x]
    return pools


def read_schema() -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = defaultdict(list)
    path = ROOT / "data_for_agent" / "attr_list.csv"
    for r in csv.DictReader(path.open(encoding="utf-8")):
        tables[r["table_name"]].append({
            "name": r["column_name"], "dtype": r["dType"],
            "pk": str(r.get("is_primary_key", "")).lower() in ("true", "1", "yes"),
        })
    return tables


def rand_date(feb_share: float = 0.15) -> dt.date:
    if random.random() < feb_share:
        return dt.date(2026, 2, random.randint(1, 28))
    return dt.date(2025, 1, 1) + dt.timedelta(days=random.randint(0, 540))


def rand_dttm() -> dt.datetime:
    d = rand_date(0.1)
    return dt.datetime(d.year, d.month, d.day, random.randint(0, 23), random.randint(0, 59))


class Gen:
    def __init__(self, pools, inns, epk_ids, org_ids):
        self.pools, self.inns, self.epk_ids, self.org_ids = pools, inns, epk_ids, org_ids

    def value(self, name: str, dtype: str):
        d, n = dtype.lower(), name.lower()
        if "bool" in d:
            return random.random() < 0.4
        if d == "date":
            return rand_date()
        if "timestamp" in d:
            if n.endswith("dttm") or any(k in n for k in ("insert", "modif", "updat", "load")):
                return random.choice(LOAD_TS)
            return rand_dttm()
        if any(t in d for t in ("char", "text")):
            if n in self.pools:
                return random.choice(self.pools[n])
            return random.choice(self.pools.get(n, ["Прочее", "Не указано", "Основные"]))
        if "numeric" in d:
            if "perc" in n:
                return round(random.uniform(0, 100), 2)
            if "amt" in n or "salary" in n or "fot" in n:
                return round(random.uniform(50_000, 5_000_000), 2)
            return round(random.uniform(0, 10_000), 2)
        if any(t in d for t in ("int", "smallint", "bigint")):
            if "inn" in n:
                return random.choice(self.inns)
            if any(k in n for k in ("qty", "cnt", "val", "num", "potential")):
                return random.randint(0, 5000)
            return random.randint(1, 1000)
        return None


# ---------- заложенные бизнес-сигналы (чтобы майнер отчёта находил реальное) ----------
HOT_EMP = ["Волков Д.А.", "Морозов И.С.", "Соколова Е.В.", "Новиков А.П.", "Зайцева О.Н."]


def _mk_dttm(base: dt.date, days: int = 0) -> dt.datetime:
    d = dt.datetime(base.year, base.month, base.day, random.randint(8, 20), random.randint(0, 59))
    return d + dt.timedelta(days=days)


def inject_signals(table: str, row: dict) -> None:
    """Правит строку, зашивая реалистичные закономерности:
    воронка — горячий срез (Москва×Микро×Фактический отток: просрочка+низкий успех,
    несколько «проблемных» сотрудников) и денежный перекос (Крупнейшие: мало задач,
    огромный потенциал); отток — дорогой срез (мало людей × высокая ЗП) против
    дешёвого массового. Меняет только существующие в строке колонки."""
    def put(k, v):
        if k in row:
            row[k] = v

    if table == "uzp_dwh_sale_funnel_task":
        create = row.get("task_create_dt")
        if not isinstance(create, dt.date):
            create = rand_date()
        put("task_create_dt", create)
        plan = _mk_dttm(create, 7)
        put("plan_close_task_dttm", plan)
        r = random.random()
        if r < 0.10:                       # горячий срез: просрочка, низкий успех
            put("tb_name", "Московский банк"); put("gosb_name", "Свердловское ГОСБ № 7003")
            put("segment_name", "Микро"); put("task_type", "Отток")
            put("task_subtype", "Фактический отток"); put("task_category", "Задача")
            put("is_task_closed", True); put("is_task_in_progress", False)
            put("is_task_closed_success", random.random() < 0.15)
            put("fact_close_task_dttm", plan + dt.timedelta(days=random.randint(5, 30)))  # просрочка
            put("emp_fio", random.choice(HOT_EMP))
            put("unrealized_deal_potential", random.randint(1000, 8000))
        elif r < 0.13:                     # денежный перекос: мало задач, огромный потенциал
            put("segment_name", "Крупнейшие"); put("task_type", "Привлечение")
            put("unrealized_deal_potential", random.randint(40_000, 90_000))
            put("is_task_closed", random.random() < 0.6)
            put("is_task_closed_success", random.random() < 0.5)
            put("fact_close_task_dttm", plan - dt.timedelta(days=random.randint(0, 3)))
        else:                              # база: обычно в срок, успех ~55%
            put("is_task_closed", random.random() < 0.7)
            put("is_task_closed_success", random.random() < 0.55)
            late = random.random() < 0.20
            put("fact_close_task_dttm",
                plan + dt.timedelta(days=random.randint(1, 10)) if late
                else plan - dt.timedelta(days=random.randint(0, 4)))
            put("unrealized_deal_potential", random.randint(0, 6000))

    elif table == "uzp_dwh_fact_outflow":
        r = random.random()
        if r < 0.03:                       # дорогой: мало людей × очень высокая ЗП
            put("outflow_qty", random.randint(3, 12))
            put("m_avg_salary_amt", round(random.uniform(1_500_000, 3_000_000), 2))
            put("segment_name", "Крупнейшие")
        elif r < 0.10:                     # дешёвый массовый: много людей × низкая ЗП
            put("outflow_qty", random.randint(200, 500))
            put("m_avg_salary_amt", round(random.uniform(30_000, 45_000), 2))
            put("segment_name", "Микро")
        else:                              # база: обычные ЗП, сегменты БЕЗ спец-значений
            put("outflow_qty", random.randint(10, 80))
            put("m_avg_salary_amt", round(random.uniform(50_000, 90_000), 2))
            put("segment_name", random.choice(["Средние", "Малые", "Клиенты машиностроения",
                                               "Рег. госсектор", "SBI"]))


def create_table(cur, table, cols):
    defs = []
    for c in cols:
        t = _DDL_TYPE.get(c["dtype"].lower(), "TEXT")
        defs.append(f'"{c["name"]}" {t}')
    cur.execute(f'DROP TABLE IF EXISTS "{SCHEMA}"."{table}"')
    cur.execute(f'CREATE TABLE "{SCHEMA}"."{table}" ({", ".join(defs)})')


def seed_table(cur, table, cols, gen, gosb, insert_fn):
    n = NROWS.get(table, DEFAULT_ROWS)
    colnames = [c["name"] for c in cols]
    pk_cols = [c["name"] for c in cols if c["pk"]]
    rows, seen_pk = [], set()
    for i in range(n):
        row = {c["name"]: gen.value(c["name"], c["dtype"]) for c in cols}
        # текстовые PK-колонки (task_code, acc_num) — уникальные коды
        for c in cols:
            if c["pk"] and any(t in c["dtype"].lower() for t in ("char", "text")):
                row[c["name"]] = f"{table[:3].upper()}-{c['name'][:4]}-{i:08d}"
        # связка с справочником: tb_id + gosb_id из реальной пары (join работает)
        if "tb_id" in row and "gosb_id" in row:
            tb_id, old_id, _ = random.choice(gosb)
            row["tb_id"], row["gosb_id"] = tb_id, old_id
        elif "tb_id" in row:
            row["tb_id"] = random.choice(TB)[0]
        # владельцы уникальных id
        if table == "uzp_data_epk_consolidation" and "epk_id" in row:
            row["epk_id"] = gen.epk_ids[i % len(gen.epk_ids)]
        if "org_id" in row:
            row["org_id"] = random.choice(gen.org_ids)
        if "org_type" in row and not gen.pools.get("org_type"):
            row["org_type"] = random.choice(ORG_TYPES)
        if "author_login" in row:
            row["author_login"] = random.choice(AUTHORS)
        inject_signals(table, row)          # зашиваем реалистичные закономерности
        # уникальность PK
        if pk_cols:
            key = tuple(row[c] for c in pk_cols)
            if key in seen_pk:
                continue
            seen_pk.add(key)
        rows.append([row[c] for c in colnames])
    insert_fn(cur, table, colnames, rows)
    return len(rows)


def build_dim_gosb(cur, cols, insert_fn):
    old_pool = list(range(1000, 1035))  # повторяются между ТБ
    gosb, seen = [], set()
    while len(gosb) < NROWS["uzp_dim_gosb"]:
        tb_id = random.choice(TB)[0]
        old_id = random.choice(old_pool)
        if (tb_id, old_id) in seen:
            continue
        seen.add((tb_id, old_id))
        gosb.append((tb_id, old_id, 2000 + len(gosb)))
    colnames = [c["name"] for c in cols]
    rows = []
    for i in range(len(gosb)):
        tb_id, old_id, new_id = gosb[i]
        name = next(t for t in TB if t[0] == tb_id)
        base = {c["name"]: None for c in cols}
        for c in cols:  # generic fill
            base[c["name"]] = _dim_val(c["name"], c["dtype"], name)
        base.update({"tb_id": tb_id, "old_gosb_id": old_id, "new_gosb_id": new_id,
                     "tb_full_name": name[1], "tb_short_name": name[2],
                     "author_login": random.choice(AUTHORS)})
        rows.append([base[c] for c in colnames])
    insert_fn(cur, "uzp_dim_gosb", colnames, rows)
    return gosb


def _dim_val(name, dtype, tb):
    d, n = dtype.lower(), name.lower()
    if "int" in d:
        return random.randint(1, 100)
    if "timestamp" in d:
        return random.choice(LOAD_TS)
    if any(t in d for t in ("char", "text")):
        if "region" in n:
            return random.choice(["Краснодарский край", "Свердловская область", "Ростовская область",
                                  "Тюменская область", "Московская область"])
        return f"{tb[2]}-{n[:6]}"
    return None


def insert(cur, table, colnames, rows):
    if not rows:
        return
    collist = ", ".join(f'"{c}"' for c in colnames)
    execute_values(cur, f'INSERT INTO "{SCHEMA}"."{table}" ({collist}) VALUES %s', rows, page_size=1000)


def main() -> None:
    pools = load_pools()
    schema = read_schema()
    inns = [random.randint(1_000_000_000, 9_999_999_999) for _ in range(300)]
    epk_ids = [5_000_000 + i for i in range(1000)]
    org_ids = [7_000_000 + i for i in range(500)]
    gen = Gen(pools, inns, epk_ids, org_ids)

    conn = psycopg2.connect(DB.dsn)
    conn.autocommit = False
    cur = conn.cursor()

    # создаём все таблицы
    for table, cols in schema.items():
        create_table(cur, table, cols)

    # dim_gosb первым (на него ссылаются fact-таблицы)
    gosb = build_dim_gosb(cur, schema["uzp_dim_gosb"], insert)
    print(f"  uzp_dim_gosb: {len(gosb)}")
    for table, cols in schema.items():
        if table == "uzp_dim_gosb":
            continue
        cnt = seed_table(cur, table, cols, gen, gosb, insert)
        print(f"  {table}: {cnt}")

    conn.commit()
    cur.close()
    conn.close()
    print("Готово. Реалистичные данные засеяны для всех таблиц манифеста.")


if __name__ == "__main__":
    main()
