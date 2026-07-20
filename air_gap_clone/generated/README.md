# air-gap DB clone — profiler + generator

Инструмент «безопасного клонирования» БД через воздушный зазор из **двух отдельных
программ**. Через зазор едет **не выгрузка данных, а маленький человекочитаемый
`profile.json`** (схема + статистика) — его можно глазами проверить перед переносом.

```
закрытый контур (реальная БД)                 открытый контур (LLM, без реальной БД)
┌───────────────────────────┐   profile.json  ┌────────────────────────────────────┐
│  profiler                 │  ─────────────▶ │  generator                         │
│  каталог + сэмпл (pandas)  │   (+ policy)    │  schema.sql + синтетические данные  │
│  → profile.json (аудит)    │                 │  (CSV или INSERT-ы)                │
└───────────────────────────┘                 └────────────────────────────────────┘
```

СУБД: **Greenplum** (MPP-форк PostgreSQL). Код совместим с разными версиями и мягко
деградирует, если GPDB-специфики (`gp_distribution_policy`, `pg_partitions`,
`pg_appendonly`) нет. Модуль **самостоятельный** — ничего из внешних проектов не тянет.

---

## Что переносить между контурами

Наружу едет **только `profile.json`**. Реальные строки данных не выгружаются:
реальные значения попадают в профиль лишь для колонок, явно одобренных в policy как
`categorical_keep` (и представителей зависимостей с `keep_representative`). Линтер
строго проверяет это перед записью.

---

## Ключевые решения

- **PK не объявлен в DDL** → выводим **гипотезу по сэмплу** (минимальная уникальная
  комбинация, как в исходном проекте). **FK не выводим вообще** — ключи джойнов
  подбираются позже уже на синтетике.
- **Вся статистика считается в pandas на случайном сэмпле** (до ~1M строк на таблицу),
  а не отдельными `GROUP BY` к БД. Редкие категории в сэмпле могут потеряться — это
  допустимо.
- **Категория = меньше 200 уникальных значений** в сэмпле. Все попавшие в сэмпл
  категории и их доли сохраняются; генератор гарантирует, что **каждая категория
  присутствует** в синтетике (для корректных GROUP BY).
- **Функциональные зависимости** (например `task_subtype → task_questionary`): для
  каждой категории детерминанта берём одного представителя зависимой колонки (свежайшая
  строка по дате, где значение не NULL) и держим эту связь в синтетике — «опросник
  остаётся у своего подтипа задачи».
- **Чувствительные данные → синтетика**, явно «сгенерированная»: ФИО «Иванов Иван
  Иванович», ИНН «1234567890», компании «ООО Ромашка».

---

## Программа 1 — profiler (закрытый контур)

Подключение задаётся **явно** (в ячейке ноутбука, через `--dsn` или `db_config.json`) —
из проекта ничего не берётся.

```bash
python profiler.py --tables "public.tasks,public.clients" \
    --policy policy.yaml --out profile.json --sample-n 1000000 \
    --dsn "postgresql://user@host:5432/prom"

# либо список таблиц из CSV (колонки schema,table или schema_name,table_name)
python profiler.py --tables-csv tables.csv --policy policy.yaml --out profile.json
```

Порядок разрешения DSN: `--dsn` → `AGC_DB_DSN`/`DB_DSN` → `--db-config db_config.json`
(`host/port/database/user[/password]`). Движок — **read-only**. Пароль можно не
указывать (Kerberos/GSSAPI, как на проде GPDB).

Структуру (типы, nullability, default, storage, ключ распределения, партиции) берём из
системного каталога; PK/статистику/категории/зависимости — из сэмпла в pandas.

### Формат policy-YAML (whitelist)

```yaml
version: 1
columns:
  "public.tasks.task_type":    categorical_keep                 # хранит реальные значения
  "public.tasks.task_subtype": categorical_keep
  "public.clients.name":       {class: pii, generator: full_name}
  "public.tasks.amount":       {class: sensitive_numeric, dist: lognormal, avg_hint: "~5e4"}
tables:
  "public.tasks":
    order_groups: [["created_at", "updated_at"]]                # created <= updated
    dependencies:
      - determinant: task_subtype
        dependent:   task_questionary
        order_by:    task_date                                  # представитель — свежайший
        keep_representative: true                               # реальный представитель (whitelist)
```

Классы: `categorical_keep` | `sensitive_numeric` | `pii` | `key` | `datetime` | `sensitive`.
PK выводится автоматически; `categorical_keep` — только явно (граница безопасности).

---

## Программа 2 — generator (открытый контур)

```bash
python generator.py --profile profile.json --scale 0.001 --seed 42 --format csv --out out/
python generator.py --profile profile.json --format sql --keep-gpdb
```

- `--scale` — доля от исходного числа строк (1.2M × 0.001 ≈ 1200);
- `--keep-gpdb` — сохранить `DISTRIBUTED BY` и заметки о партициях (иначе всё heap);
- таблицы независимы (FK нет); суррогатный PK по гипотезе; категории — все;
  зависимости соблюдены; без ML-синтезаторов (SDV/CTGAN).

---

## Установка (закрытый контур без интернета)

```bash
pip install --no-index --find-links=./wheels -r requirements.txt
```

`SQLAlchemy`, `psycopg2`, `PyYAML`, `pandas` — для profiler. Generator работает на
stdlib; `Faker`/`numpy` — опциональные улучшения.

---

## Структура

```
agc_common.py            общие утилиты (валидация идентификаторов, логирование)
agc_profiler/            ПРОГРАММА 1
  db.py                  подключение (read-only), самостоятельное
  catalog_reader.py      структура из системного каталога (типы, storage, dist, партиции)
  sampler.py             адаптивный сэмпл таблицы в pandas
  analyze.py             PK-гипотеза, статистика, категории, представители зависимостей
  classifier.py          авто-эвристики (только ПРЕДЛАГАЮТ класс)
  policy.py              whitelist чувствительности + зависимости (финальное решение)
  linter.py              строгая проверка утечек реальных значений
  profile_builder.py     сборка profile.json
  cli.py                 CLI
agc_generator/           ПРОГРАММА 2
  profile_parser.py      парсер profile.json
  ddl_builder.py         schema.sql (PK-гипотеза, без FK)
  value_generators.py    генераторы значений по классам (+ зависимости)
  key_linker.py          суррогатный PK, категории, функциональные зависимости
  writer.py              CSV / INSERT-ы
  cli.py                 CLI
profiler.py, generator.py   точки входа
policy.example.yaml, tables.example.csv, examples/profile.example.json
```
