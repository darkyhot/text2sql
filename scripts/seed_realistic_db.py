"""Пересоздание и засев тестового контейнера данными ПО МОТИВАМ ПРОДА.

Источник — папка data_for_agent_prom (кладётся с прода):
  • attr_list.csv — структура (схемы, таблицы, колонки, типы, PK, доля непустых);
  • tables_list.csv — манифест таблиц;
  • samples/<schema>.<table>.csv — синтетические ген-сэмплы (реальные по форме значения).

Из сэмплов строятся пулы значений по колонкам → генерятся ПОЛНЫЕ строки (без разреженности
самих сэмплов), с правдоподобной долей NULL из attr_list. Сохраняются связки и сигналы для
эталонных запросов и бизнес-отчёта:
  • tb_id+gosb_id в fact-таблицах ссылаются на uzp_dim_gosb (join работает);
  • old_gosb_id повторяется между ТБ → нужен составной ключ (анти-fanout);
  • task_subtype содержит 'Фактический отток', task_type — широкий 'Отток';
  • task_category='Задача' → сделочные поля пусты/0 (бизнес-правило);
  • горячий срез просрочки + денежный/потенциальный перекос — для майнера отчёта.
Поддерживает НЕСКОЛЬКО схем (в т.ч. новую таблицу yva_date_fl_market_analysis).

Запуск:  python scripts/seed_realistic_db.py
"""
from __future__ import annotations

import csv
import datetime as dt
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2sql.config import DB  # noqa: E402

random.seed(42)
PROM = ROOT / "data_for_agent_prom"          # источник структуры и сэмплов (с прода)
SAMPLES = PROM / "samples"
GOSB_SCHEMA = "s_grnplm_ld_salesntwrk_pcap_sn_uzp"   # схема со справочником uzp_dim_gosb
NROWS = {
    "uzp_dim_gosb": 60, "uzp_data_epk_consolidation": 800,
    "uzp_dwh_fact_outflow": 4000, "uzp_dwh_sale_funnel_task": 4000,
    "uzp_data_payroll_m": 6000, "uzp_dwh_company_holding_metric": 4000,
    "yva_date_fl_market_analysis": 3000,
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

_DDL_TYPE = {
    "bigint": "BIGINT", "integer": "INTEGER", "smallint": "SMALLINT",
    "character varying": "VARCHAR", "text": "TEXT", "boolean": "BOOLEAN",
    "numeric": "NUMERIC", "date": "DATE", "timestamp without time zone": "TIMESTAMP",
    "double precision": "DOUBLE PRECISION", "real": "REAL", "decimal": "NUMERIC",
}
_PROTECT = {"tb_id", "gosb_id", "inn"}       # не занулять (нужны для join/фильтров)


def read_schema() -> dict[tuple[str, str], list[dict]]:
    """{(schema, table): [ {name,dtype,pk,nn} ]} из prom attr_list.csv."""
    tables: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in csv.DictReader((PROM / "attr_list.csv").open(encoding="utf-8")):
        try:
            nn = float(r.get("not_null_perc") or 95)
        except ValueError:
            nn = 95.0
        tables[(r["schema_name"], r["table_name"])].append({
            "name": r["column_name"], "dtype": r["dType"],
            "pk": str(r.get("is_primary_key", "")).lower() in ("true", "1", "yes"),
            "nn": nn,
        })
    return tables


def load_pools() -> tuple[dict, dict]:
    """Пулы значений из сэмплов: точные (schema,table,col)→[vals] и общий col→[vals]."""
    exact: dict[tuple[str, str, str], list] = {}
    glob: dict[str, list] = defaultdict(list)
    for f in SAMPLES.glob("*.csv"):
        if f.name.endswith("Zone.Identifier"):
            continue
        try:
            schema, table = f.stem.split(".", 1)
        except ValueError:
            continue
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        if not rows:
            continue
        for col in rows[0].keys():
            vals = [r[col] for r in rows if r.get(col) not in (None, "", "None")]
            if vals:
                exact[(schema, table, col)] = list(dict.fromkeys(vals))
                for v in vals:
                    glob[col].append(v)
    glob = {k: list(dict.fromkeys(v)) for k, v in glob.items()}
    return exact, glob


def rand_date(feb_share: float = 0.15) -> dt.date:
    if random.random() < feb_share:
        return dt.date(2026, 2, random.randint(1, 28))
    return dt.date(2025, 1, 1) + dt.timedelta(days=random.randint(0, 540))


def rand_dttm() -> dt.datetime:
    d = rand_date(0.1)
    return dt.datetime(d.year, d.month, d.day, random.randint(0, 23), random.randint(0, 59))


def _coerce(v: str, dtype: str):
    """Значение из сэмпла (строка) → нужный python-тип по dtype."""
    d = dtype.lower()
    try:
        if "bool" in d:
            return str(v).strip().lower() in ("true", "1", "да", "t")
        if d == "date":
            return dt.date.fromisoformat(str(v)[:10])
        if "timestamp" in d:
            return dt.datetime.fromisoformat(str(v).replace("Z", "")[:26])
        if "int" in d or "smallint" in d or "bigint" in d:
            return int(float(v))
        if any(t in d for t in ("numeric", "double", "real", "decimal", "float")):
            return round(float(v), 2)
    except (ValueError, TypeError):
        return None
    return str(v)


class Gen:
    def __init__(self, exact, glob, inns, epk_ids):
        self.exact, self.glob, self.inns, self.epk_ids = exact, glob, inns, epk_ids

    def value(self, schema: str, table: str, name: str, dtype: str):
        d, n = dtype.lower(), name.lower()
        # КЛЮЧИ/ДАТЫ/булевы — широкая генерация (нужна для уникальности составного PK и
        # разброса по времени); сэмпл-пулы для них слишком узки → коллизии PK.
        if "inn" in n:
            return random.choice(self.inns)
        if "bool" in d:
            return random.random() < 0.4
        if d == "date":
            return rand_date()
        if "timestamp" in d:
            if n.endswith("dttm") or any(k in n for k in ("insert", "modif", "updat", "load")):
                return random.choice(LOAD_TS)
            return rand_dttm()
        # 1) значение из сэмпла (точный пул таблицы, затем общий по имени колонки)
        pool = self.exact.get((schema, table, name)) or self.glob.get(name)
        if pool:
            return _coerce(random.choice(pool), dtype)
        # 2) фолбэк по типу/имени (когда колонки нет в сэмплах)
        if any(t in d for t in ("numeric", "double", "real", "decimal", "float")):
            if "perc" in n:
                return round(random.uniform(0, 100), 2)
            if any(k in n for k in ("qty", "cnt", "count", "_num", "_fl_", "kol")):
                return random.randint(0, 5000)         # количества — целые (не копейки)
            if any(k in n for k in ("amt", "salary", "fot", "sum", "оборот", "выруч")):
                return round(random.uniform(50_000, 5_000_000), 2)
            return round(random.uniform(0, 10_000), 2)
        if any(t in d for t in ("int", "smallint", "bigint")):
            if "inn" in n:
                return random.choice(self.inns)
            if any(k in n for k in ("qty", "cnt", "val", "num", "potential")):
                return random.randint(0, 5000)
            return random.randint(1, 1000)
        return random.choice(["Прочее", "Не указано", "Основные"])


# ---------- заложенные бизнес-сигналы ----------
HOT_EMP = ["Волков Д.А.", "Морозов И.С.", "Соколова Е.В.", "Новиков А.П.", "Зайцева О.Н."]


def _mk_dttm(base: dt.date, days: int = 0) -> dt.datetime:
    d = dt.datetime(base.year, base.month, base.day, random.randint(8, 20), random.randint(0, 59))
    return d + dt.timedelta(days=days)


def inject_signals(table: str, row: dict) -> None:
    """Зашивает закономерности (только существующие в строке колонки)."""
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
        is_offer = random.random() < 0.5
        put("task_category", "Предложение" if is_offer else "Задача")
        if is_offer:
            put("deal_create_dttm", _mk_dttm(create, random.randint(1, 20)))
            put("plan_staff_deal_qty", random.randint(1, 20))
            put("fact_staff_deal_qty", random.randint(0, 20))
            put("unrealized_deal_potential", random.randint(0, 6000))
        else:                              # 'Задача' не несёт сделку
            put("plan_staff_deal_qty", 0); put("fact_staff_deal_qty", 0)
            put("deal_code", None); put("deal_create_dttm", None)
            put("unrealized_deal_potential", None)
        r = random.random()
        if r < 0.10:                       # горячий срез: просрочка, низкий успех
            put("tb_name", "Московский банк"); put("gosb_name", "Свердловское ГОСБ № 7003")
            put("segment_name", "Микро"); put("task_type", "Отток")
            put("task_subtype", "Фактический отток")
            put("is_task_closed", True); put("is_task_in_progress", False)
            put("is_task_closed_success", random.random() < 0.15)
            put("fact_close_task_dttm", plan + dt.timedelta(days=random.randint(5, 30)))
            put("emp_fio", random.choice(HOT_EMP))
        elif r < 0.13 and is_offer:        # перекос потенциала
            put("segment_name", "Крупнейшие"); put("task_type", "Привлечение")
            put("unrealized_deal_potential", random.randint(40_000, 90_000))
            put("is_task_closed", random.random() < 0.6)
            put("fact_close_task_dttm", plan - dt.timedelta(days=random.randint(0, 3)))
        else:                              # база
            put("is_task_closed", random.random() < 0.7)
            put("is_task_closed_success", random.random() < 0.55)
            late = random.random() < 0.20
            put("fact_close_task_dttm",
                plan + dt.timedelta(days=random.randint(1, 10)) if late
                else plan - dt.timedelta(days=random.randint(0, 4)))

    elif table == "yva_date_fl_market_analysis":
        # Целевая мера — diff_fl (fl_26 − fl_25). Закладываем КОНЦЕНТРИРОВАННЫЕ ПОТЕРИ:
        # горячий срез Холдинг Б × ММБ × причина «Закрытие расчётного счёта», убыль в ГПХ.
        fl25 = random.randint(500, 3000)
        r = random.random()
        if r < 0.15:                       # горячий срез потерь
            put("holding_name", "Холдинг Б"); put("segment_name_bp", "ММБ")
            put("gosb_name", "ГОСБ Москва"); put("outflow_reason", "Закрытие расчётного счёта")
            fl26 = max(0, fl25 - random.randint(300, 1500))
            put("potential_1q_26", random.randint(80_000, 250_000))   # ценные — вернуть
        elif r < 0.27:                     # прочие небольшие потери (фон)
            fl26 = max(0, fl25 - random.randint(10, 150))
        else:                              # база: рост/стабильно
            fl26 = fl25 + random.randint(0, 250)
        put("fl_25_qty", fl25); put("fl_26_qty", fl26)
        put("diff_fl_26_25_qty", fl26 - fl25)
        u25, u26 = int(fl25 * 0.9), int(fl26 * 0.9)
        put("fl_unic_25_qty", u25); put("fl_unic_26_qty", u26)
        put("diff_fl_unic_26_25_qty", u26 - u25)
        put("fl_gpx_25_qty", int(fl25 * 0.45)); put("fl_gpx_26_qty", int(fl26 * 0.30))
        put("fl_self_emplyeed_25_qty", int(fl25 * 0.25)); put("fl_self_emplyeed_26_qty", int(fl26 * 0.22))

    elif table == "uzp_dwh_fact_outflow":
        r = random.random()
        if r < 0.03:                       # дорогой: мало людей × высокая ЗП
            put("outflow_qty", random.randint(3, 12))
            put("m_avg_salary_amt", round(random.uniform(1_500_000, 3_000_000), 2))
            put("segment_name", "Крупнейшие")
        elif r < 0.10:                     # дешёвый массовый
            put("outflow_qty", random.randint(200, 500))
            put("m_avg_salary_amt", round(random.uniform(30_000, 45_000), 2))
            put("segment_name", "Микро")
        else:                              # база
            put("outflow_qty", random.randint(10, 80))
            put("m_avg_salary_amt", round(random.uniform(50_000, 90_000), 2))
            put("segment_name", random.choice(["Средние", "Малые", "Клиенты машиностроения",
                                               "Рег. госсектор", "SBI"]))


def create_schema(cur, schema: str) -> None:
    cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')


def create_table(cur, schema: str, table: str, cols) -> None:
    defs = [f'"{c["name"]}" {_DDL_TYPE.get(c["dtype"].lower(), "TEXT")}' for c in cols]
    cur.execute(f'DROP TABLE IF EXISTS "{schema}"."{table}"')
    cur.execute(f'CREATE TABLE "{schema}"."{table}" ({", ".join(defs)})')


def insert(cur, schema: str, table: str, colnames, rows) -> None:
    if not rows:
        return
    collist = ", ".join(f'"{c}"' for c in colnames)
    execute_values(cur, f'INSERT INTO "{schema}"."{table}" ({collist}) VALUES %s', rows, page_size=1000)


def seed_table(cur, schema: str, table: str, cols, gen, gosb) -> int:
    n = NROWS.get(table, DEFAULT_ROWS)
    colnames = [c["name"] for c in cols]
    pk_cols = [c["name"] for c in cols if c["pk"]]
    rows, seen_pk = [], set()
    for i in range(n):
        row = {c["name"]: gen.value(schema, table, c["name"], c["dtype"]) for c in cols}
        # правдоподобная доля NULL (кроме PK/связок), но не более 40% — иначе разрежённость
        for c in cols:
            if c["pk"] or c["name"] in _PROTECT:
                continue
            null_p = min(0.4, max(0.0, 1 - c["nn"] / 100.0))
            if null_p and random.random() < null_p:
                row[c["name"]] = None
        # текстовые PK — уникальные коды
        for c in cols:
            if c["pk"] and any(t in c["dtype"].lower() for t in ("char", "text")):
                row[c["name"]] = f"{table[:3].upper()}-{c['name'][:4]}-{i:08d}"
        # связка с справочником: tb_id + gosb_id из реальной пары
        if "tb_id" in row and "gosb_id" in row and gosb:
            tb_id, old_id, _ = random.choice(gosb)
            row["tb_id"], row["gosb_id"] = tb_id, old_id
        elif "tb_id" in row:
            row["tb_id"] = random.choice(TB)[0]
        if table == "uzp_data_epk_consolidation" and "epk_id" in row:
            row["epk_id"] = gen.epk_ids[i % len(gen.epk_ids)]
        if "author_login" in row:
            row["author_login"] = random.choice(AUTHORS)
        inject_signals(table, row)
        if pk_cols:
            key = tuple(row[c] for c in pk_cols)
            if key in seen_pk:
                continue
            seen_pk.add(key)
        rows.append([row[c] for c in colnames])
    insert(cur, schema, table, colnames, rows)
    return len(rows)


def build_dim_gosb(cur, schema: str, cols, gen) -> list[tuple]:
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
        tb = next(t for t in TB if t[0] == tb_id)
        base = {c["name"]: gen.value(schema, "uzp_dim_gosb", c["name"], c["dtype"]) for c in cols}
        base.update({"tb_id": tb_id, "old_gosb_id": old_id, "new_gosb_id": new_id,
                     "tb_full_name": tb[1], "tb_short_name": tb[2],
                     "author_login": random.choice(AUTHORS)})
        rows.append([base[c] for c in colnames])
    insert(cur, schema, "uzp_dim_gosb", colnames, rows)
    return gosb


def sync_metadata_to_active() -> None:
    """Скопировать prom-метадату в data_for_agent, чтобы каталог/агент видели прод-схему
    (в т.ч. новую таблицу). Сэмплы не трогаем."""
    active = ROOT / "data_for_agent"
    for fn in ("attr_list.csv", "tables_list.csv", "join_candidates.json"):
        src = PROM / fn
        if src.exists():
            shutil.copy2(src, active / fn)
    print(f"  метадата синхронизирована в {active.name}/ (attr_list, tables_list, join_candidates)")


def main() -> None:
    exact, glob = load_pools()
    schema = read_schema()
    inns = [random.randint(1_000_000_000, 9_999_999_999) for _ in range(300)]
    epk_ids = [5_000_000 + i for i in range(1000)]
    gen = Gen(exact, glob, inns, epk_ids)

    conn = psycopg2.connect(DB.dsn)
    conn.autocommit = False
    cur = conn.cursor()

    for sch in sorted({s for s, _ in schema}):
        create_schema(cur, sch)
    for (sch, table), cols in schema.items():
        create_table(cur, sch, table, cols)

    # dim_gosb первым (на него ссылаются fact-таблицы своей схемы)
    gosb = []
    if (GOSB_SCHEMA, "uzp_dim_gosb") in schema:
        gosb = build_dim_gosb(cur, GOSB_SCHEMA, schema[(GOSB_SCHEMA, "uzp_dim_gosb")], gen)
        print(f"  uzp_dim_gosb: {len(gosb)}")
    for (sch, table), cols in schema.items():
        if table == "uzp_dim_gosb":
            continue
        link = gosb if sch == GOSB_SCHEMA else []
        cnt = seed_table(cur, sch, table, cols, gen, link)
        print(f"  {sch}.{table}: {cnt}")

    conn.commit()
    cur.close()
    conn.close()
    sync_metadata_to_active()
    print("Готово. Тестовая БД засеяна по мотивам прода (data_for_agent_prom).")


if __name__ == "__main__":
    main()
