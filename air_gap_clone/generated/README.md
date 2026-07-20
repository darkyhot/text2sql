# air-gap DB clone — profiler + generator

Инструмент «безопасного клонирования» БД через воздушный зазор из **двух отдельных
программ**. Через зазор едет **не выгрузка данных, а маленький человекочитаемый
`profile.json`** (схема + статистика) — его можно глазами проверить перед переносом.

```
закрытый контур (реальная БД)                 открытый контур (LLM, без реальной БД)
┌───────────────────────────┐   profile.json  ┌────────────────────────────────────┐
│  profiler                 │  ─────────────▶ │  generator                         │
│  каталог + pg_stats        │   (+ policy)    │  schema.sql + синтетические данные  │
│  → profile.json (аудит)    │                 │  (CSV или INSERT-ы)                │
└───────────────────────────┘                 └────────────────────────────────────┘
```

СУБД: **Greenplum** (MPP-форк PostgreSQL). Код совместим с разными версиями и мягко
деградирует, если GPDB-специфики (`gp_distribution_policy`, `pg_partitions`,
`pg_appendonly`) нет.

---

## Что переносить между контурами

Наружу (закрытый → открытый) едет **только `profile.json`**. Опционально — DDL от
`pg_dump --schema-only` для сверки. Реальные строки данных **не** выгружаются:
реальные значения попадают в профиль лишь для колонок, явно одобренных в policy как
`categorical_keep` (см. линтер — строгая проверка перед записью).

---

## Программа 1 — profiler (закрытый контур)

```bash
# список таблиц через запятую
python profiler.py --tables "public.tasks,public.clients" \
    --policy policy.yaml --out profile.json --sample

# либо список из CSV (колонки schema,table или schema_name,table_name)
python profiler.py --tables-csv tables.csv --policy policy.yaml --out profile.json
```

Подключение к БД (в порядке приоритета): `--dsn` → `AGC_DB_DSN`/`DB_DSN` →
`--db-config db_config.json` → подключение проекта `text2sql` (`config_db.json`/`.env`),
если инструмент запущен внутри репозитория. Движок — **read-only**
(`default_transaction_read_only=on` + `statement_timeout`).

Откуда берётся информация:
- **структура** — из системного каталога (ground truth, без скана): типы/nullability/
  default, PK/UNIQUE/NOT NULL/FK/CHECK, `gp_distribution_policy`, партиции, тип хранения;
- **статистика** — из `pg_stats` одним запросом на схему: `null_frac`, `n_distinct`
  (знак сохраняем!), `avg_width`, `most_common_vals`/`freqs`, `histogram_bounds`;
- **число строк** — из `reltuples` (помечается как оценка);
- **сэмпл** (`--sample`) — только для инспекции форматов коротких текстовых полей;
  для кардинальности **не** используется (см. оговорку о смещении в `sampler.py`);
- `--recompute-missing` — точечно досчитать `null_frac`/`n_distinct` агрегатами там,
  где `pg_stats` пуст (это **скан**, по умолчанию выключено, логируется предупреждением).

### Формат policy-YAML (whitelist)

По умолчанию **каждая** колонка чувствительна и синтезируется. `categorical_keep`
недостижим автоматически — только явно в policy (граница безопасности).

```yaml
version: 1
columns:
  "public.tasks.task_type": categorical_keep                 # хранит реальные значения
  "public.clients.name":    {class: pii, generator: full_name}
  "public.tasks.amount":    {class: sensitive_numeric, dist: lognormal, avg_hint: "~5e4"}
tables:
  "public.tasks":
    order_groups: [["created_at", "updated_at"]]             # created <= updated
```

Классы: `categorical_keep` | `sensitive_numeric` | `pii` | `key` | `datetime` | `sensitive`.

---

## Программа 2 — generator (открытый контур)

```bash
python generator.py --profile profile.json --scale 0.001 --seed 42 \
    --format csv --out out/                # CSV на таблицу  (+ out/schema.sql)

python generator.py --profile profile.json --scale 0.001 \
    --format sql --keep-gpdb               # батч INSERT-ов + DISTRIBUTED BY
```

- `--scale` — доля от исходного числа строк (1.2M × 0.001 ≈ 1200);
- `--keep-gpdb` — сохранить `DISTRIBUTED BY` и заметки о партициях (иначе всё heap);
- согласованное масштабирование кардинальности и FK: **все джойны резолвятся**
  (FK берутся из уже сгенерированных родительских ключей);
- без ML-синтезаторов (SDV/CTGAN) — лёгкий детерминированный генератор по профилю.

---

## Установка (закрытый контур без интернета)

```bash
pip install --no-index --find-links=./wheels -r requirements.txt
```

Обязательны только `SQLAlchemy`, `psycopg2`, `PyYAML` (для profiler). Generator
работает на stdlib; `Faker`/`numpy` — опциональные улучшения.

---

## Структура

```
agc_common.py            общие утилиты (валидация идентификаторов, логирование)
agc_profiler/            ПРОГРАММА 1
  db.py                  подключение (read-only), переиспользует коннект проекта
  catalog_reader.py      структура из системного каталога
  stats_reader.py        pg_stats (+ парсер PG-массивов, досчёт по агрегатам)
  sampler.py             адаптивный сэмпл форматов (с оговоркой о смещении)
  classifier.py          авто-эвристики (только ПРЕДЛАГАЮТ класс)
  policy.py              whitelist чувствительности (финальное решение)
  linter.py              строгая проверка утечек реальных значений
  profile_builder.py     сборка profile.json
  cli.py                 CLI
agc_generator/           ПРОГРАММА 2
  profile_parser.py      парсер profile.json
  ddl_builder.py         schema.sql (топосортировка по FK)
  value_generators.py    генераторы значений по классам
  key_linker.py          суррогатные PK, FK, масштабирование кардинальности
  writer.py              CSV / INSERT-ы
  cli.py                 CLI
profiler.py, generator.py   точки входа
policy.example.yaml, tables.example.csv, examples/profile.example.json
```
