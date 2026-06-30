# Text2SQL Agent

LLM-first агент text2sql в стиле hermes/openclaw: тонкий tool-using граф на
LangGraph, минимум предвычисленных артефактов, защита от размножения строк.
Архитектура и обоснования — в [DESIGN.md](DESIGN.md).

## Возможности

- **План на естественном языке** перед выполнением; цикл `план → ОК / коррекция`.
- **Защита от размножения строк** (`check_join`): join к справочнику только по его
  полному составному ключу, кардинальность подтверждается живой пробой.
- **Неоднозначность показывается**, а не угадывается (interrupt с выбором).
- **Read-only пробы** к БД с предохранителями (только SELECT/EXPLAIN, авто-LIMIT,
  timeout, потолок cost по EXPLAIN).
- **Провайдер-абстракция LLM**: DeepSeek (сейчас) ↔ GigaChat (прод) без правок узлов.
- **Диалект-граница БД**: Postgres (тест) ↔ Greenplum (прод).
- **Трассировка** каждого шага в `traces/<session>.jsonl`.

## Структура

```
src/text2sql/
  config.py  trace.py
  llm/      base, openai_compat, gigachat, client   # провайдер-абстракция
  db/       adapter                                  # read-only guards + диалект
  catalog/  catalog, refresh                         # каталог, BM25, PK, join-граф
  tools/    toolbox                                  # search/describe/distinct/probe/check_join
  plan/     model, render                            # StructuredPlan → SQL/NL + валидация
  graph/    state, prompts, nodes, agent             # LangGraph: scope→clarify→plan→join_advisor→present→synth→execute
scripts/    smoke.py, refresh_metadata.py
eval/       cases.py, harness.py
notebooks/  text2sql_demo.ipynb
data_for_agent/        # метаданные (регенерируются из БД)
workspace/last_query.csv
```

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker start db-agent-test-pg          # тестовый Postgres на :55432
cp .env.example .env                   # при необходимости; ключ/DSN уже в .env
```

## Использование (ноутбук, одна ячейка)

```python
from text2sql.cli import CLI
CLI().run()
```

Дальше работа как в консоли: пишешь запрос → агент показывает **план** →
отвечаешь `ok` (выполнить) или текстом (правка плана); при неоднозначности —
номер варианта. Результат каждого запроса → `workspace/last_query.csv`.

Команды: `/help`, `/config_db_conn`, `/model`, `/table_list`,
`/add_table schema.table`, `/remove_table schema.table`, `/refresh_metadata`,
`/reset`, `/clear`, `/exit`.

- `/config_db_conn` — настроить подключение к БД (Greenplum/Postgres, Kerberos).
- `/model` — выбрать модель GigaChat (Gigachat-3-Ultra / Gigachat-2-Max). Выбор
  сохраняется в `config_runtime.json` и переживает перезапуск/обновление кода.
- `/table_list` — показать таблицы из метаданных.
- `/add_table schema.table` — добавить таблицу в манифест `tables_list.csv` и
  собрать по ней метаданные.
- `/remove_table schema.table` — удалить таблицу из метаданных.
- `/refresh_metadata` — пересобрать метаданные **только по таблицам манифеста**
  (адаптивный сэмпл по размеру таблицы, таймаут 5 мин/таблица; сбойная таблица
  пропускается с уведомлением, рефреш продолжается).

При истёкшем Kerberos-тикете агент выводит явное сообщение «Ошибка Kerberos…
перевыпустите kinit».

**Логи** для разбора инцидентов пишутся в `workspace/agent.log` (поток узлов,
исполняемый SQL, ошибки, трейсбэки) — этот файл удобно прислать при сбое.
См. [notebooks/text2sql_demo.ipynb](notebooks/text2sql_demo.ipynb).

Программный доступ (без REPL) — через `Agent.ask()/respond()`:
```python
from text2sql.graph.agent import Agent
agent = Agent()
turn = agent.ask('Посчитай сумму оттоков по дате и названию ГОСБ')
turn = agent.respond('ok')   # либо текст-коррекция; либо '0'/'1' при выборе
```

## Прод-контур (Greenplum + GigaChat)

Подключение к БД — SQLAlchemy `postgresql://user@host:port/db` (драйвер psycopg2,
без пароля: Kerberos/GSSAPI), как в боевом `core/database.py`. LLM — GigaChat.

**Что перенести на прод:**
```
src/text2sql/**           # весь пакет
requirements.txt
data_for_agent/**         # метаданные (или пересоберите на проде /refresh_metadata)
notebooks/text2sql_demo.ipynb
scripts/                  # опц. (refresh_metadata.py, smoke.py)
```
НЕ переносить: `.env` (создайте свой), `config_db.json` (создаётся командой),
`workspace/`, `traces/`, `.venv/`, `data_for_agent_original_backup/`.

**Шаги на проде:**
1. `pip install -r requirements.txt` (тянет sqlalchemy, psycopg2-binary, langchain-gigachat).
2. Создать `.env` без `DB_DSN` (коннект задаётся командой), с LLM:
   ```
   LLM_PROVIDER=gigachat
   LLM_MODEL=Gigachat-3-Ultra
   GIGACHAT_API_URL=<url>
   JPY_API_TOKEN=<token>
   ```
3. В ноутбуке выполнить ячейку `CLI().run()`, затем:
   - `/config_db_conn` — ввести user_id, host, port, database (dialect=greenplum);
   - `/model` — выбрать Gigachat-3-Ultra или Gigachat-2-Max (если не задано в .env);
   - `/refresh_metadata` — пересобрать метаданные из прод-БД (та же схема
     `s_grnplm_ld_salesntwrk_pcap_sn_uzp` с реальными значениями);
   - задавать запросы.

Пока коннект не настроен, на запрос данных агент отвечает:
`Не настроен коннект к БД, выполните /config_db_conn`.

Финальный запрос выгружается **полностью** (без probe-LIMIT/cost-потолка),
ограничен `EXPORT_MAX_ROWS` (по умолчанию 1 000 000) и `statement_timeout`.

## Метаданные

Регенерация из любой БД (generic). PK и описания/grain сидируются из кураторской
метадаты (на синтетических данных контейнера сэмпл-эвристика PK ненадёжна),
статистики/sample_values/join-граф — из реальных данных:

```bash
python scripts/refresh_metadata.py 's_grnplm%'
```

## Eval

Грейдинг по свойствам результата/плана (не по тексту SQL), multi-run для замера
стабильности:

```bash
python eval/harness.py        # 1 прогон
python eval/harness.py 3      # 3 прогона на кейс (pass-rate)
```

## Конфигурация (`.env`)

| Переменная | Назначение |
|---|---|
| `LLM_PROVIDER` | `deepseek` \| `gigachat` |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | параметры LLM |
| `DB_DIALECT` | `postgres` \| `greenplum` |
| `DB_DSN` | строка подключения |
| `PROBE_ROW_LIMIT`, `PROBE_TIMEOUT_MS`, `PROBE_MAX_COST` | предохранители проб |

Для прода (GigaChat): `LLM_PROVIDER=gigachat`, плюс `GIGACHAT_API_URL`,
`JPY_API_TOKEN`, `GIGACHAT_MODEL=Gigachat-3-ultra` (нужен `langchain-gigachat`).
