"""CLI-REPL для управления агентом из одной ячейки Jupyter.

Пользователь пишет запрос как в консоль, агент отвечает туда же, пользователь
отвечает туда же (ok / правка плана / номер варианта при неоднозначности).
Команды: /help, /refresh_metadata, /reset, /clear, /exit.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from .catalog.refresh import MetadataRefresh
from .db.adapter import KERBEROS_MESSAGE, is_kerberos_auth_error
from .db.connection import ConnectionConfig, load_connection, save_connection
from .graph.agent import Agent, Turn
from .logging_setup import setup_logging

logger = logging.getLogger(__name__)

# Прод-модели GigaChat (подсказка для /model)
_GIGACHAT_MODELS = ["Gigachat-3-Ultra", "Gigachat-2-Max"]

try:
    from IPython.display import clear_output, display
    import pandas as pd
    _HAS_IPY = True
except Exception:  # noqa: BLE001
    _HAS_IPY = False

HELP_TEXT = """
Доступные команды:
  /help               — показать этот список
  /config_db_conn     — настроить подключение к БД (user_id, host, port, database)
  /model              — выбрать модель LLM (Gigachat-3-Ultra / Gigachat-2-Max)
  /table_list         — показать таблицы (schema.table) из метаданных
  /add_table s.t      — добавить таблицу schema.table в манифест и собрать метаданные
  /remove_table s.t   — удалить таблицу schema.table из метаданных
  /refresh_metadata   — пересобрать метаданные по манифесту tables_list.csv
  /reset              — сбросить контекст (новая сессия)
  /clear              — очистить вывод консоли
  /exit               — завершить работу агента

Любой ввод без `/` обрабатывается как запрос к агенту.
В ответ на план: `ok` — выполнить; любой другой текст — правка плана.
При неоднозначности: введите номер варианта.
""".strip()

_OK_HINT = "Ваш ответ (ok / правка плана): "


class CLI:
    """Интерактивный REPL агента для Jupyter."""

    def __init__(self) -> None:
        self.log_file = setup_logging()
        logger.info("CLI запущен")
        self.agent = Agent()

    # ---------- вывод ----------
    @staticmethod
    def _status(msg: str) -> None:
        print(f"   …{msg}", flush=True)

    def _with_status(self, msg: str, fn: Callable[[], Turn]) -> Turn:
        t0 = time.time()
        self._status(msg)
        turn = fn()
        self._status(f"готово за {time.time()-t0:.0f}с")
        return turn

    def _banner(self) -> None:
        cat = self.agent.catalog
        schemas = sorted({t.schema for t in cat.all_tables()})
        ncols = sum(len(t.columns) for t in cat.all_tables())
        llm = self.agent.llm.cfg
        db_state = self.agent.db.connection_summary() if self.agent.db.is_configured else \
            "не настроено — выполните /config_db_conn"
        print(
            "╔══════════════════════════════════════════╗\n"
            "║        Text2SQL Agent — LLM-first        ║\n"
            "╚══════════════════════════════════════════╝\n"
            f"\nLLM: {llm.model} (provider={llm.provider})\n"
            f"БД: {db_state}\n"
            f"Схемы: {', '.join(schemas)}\n"
            f"Таблиц: {len(cat.all_tables())} | колонок: {ncols} | "
            f"join-кандидатов: {len(cat.join_candidates)}\n"
            "\nВведите запрос или /help. Результат → workspace/last_query.csv\n"
        )

    # ---------- команды ----------
    def _refresh_metadata(self) -> None:
        logger.info("CLI: /refresh_metadata (по манифесту tables_list.csv)")
        res = MetadataRefresh(self.agent.db, llm=self.agent.llm).refresh_manifest(progress_callback=self._status)
        self.agent.reload_catalog()
        print(f"✓ Метаданные пересобраны по манифесту: таблиц {res['tables']}, "
              f"колонок {res['columns']}, join-кандидатов {res['join_candidates']}.")
        if res["failed"]:
            print("⚠ Пропущены (таймаут/ошибка), старые метаданные сохранены:")
            for fqn, err in res["failed"]:
                print(f"   • {fqn}: {err}")

    def _add_table(self, ref: str) -> None:
        if "." not in ref:
            print("Формат: /add_table schema.table")
            return
        schema, table = ref.split(".", 1)
        logger.info("CLI: /add_table %s.%s", schema, table)
        self._status(f"добавляю и собираю метаданные: {schema}.{table}…")
        res = MetadataRefresh(self.agent.db, llm=self.agent.llm).add_table(schema, table)
        if res["status"] == "added":
            self.agent.reload_catalog()
            print(f"✓ Таблица {res['fqn']} добавлена в манифест. Колонок: {res['columns']}, PK-гипотеза: {res['pk']}.")
        elif res["status"] == "missing":
            print(f"✗ Таблица {res['fqn']} не найдена в БД.")
        else:
            print(f"✗ Не удалось собрать метаданные {res['fqn']}: {res.get('error')}")

    def _remove_table(self, ref: str) -> None:
        if "." not in ref:
            print("Формат: /remove_table schema.table")
            return
        schema, table = ref.split(".", 1)
        logger.info("CLI: /remove_table %s.%s", schema, table)
        res = MetadataRefresh(self.agent.db).remove_table(schema, table)
        if res["status"] == "removed":
            self.agent.reload_catalog()
            print(f"✓ Таблица {res['fqn']} удалена из метаданных.")
        else:
            print(f"✗ Таблица {res['fqn']} не найдена в метаданных.")

    def _table_list(self) -> None:
        tables = sorted(self.agent.catalog.all_tables(), key=lambda t: t.fqn)
        if not tables:
            print("Метаданные пусты. Добавьте таблицу: /add_table schema.table")
            return
        print(f"\nТаблицы в метаданных ({len(tables)}):")
        for t in tables:
            print(f"  {t.fqn}  [{t.role}/{t.grain}]  — {t.description}")

    def _reset(self) -> None:
        self.agent = Agent()
        print("✓ Контекст сброшен. Новая сессия.")

    def _config_db_conn(self) -> None:
        cur = load_connection() or ConnectionConfig()
        print("\n— Настройка подключения к БД (Greenplum/PostgreSQL) —")
        print("  (пароль не нужен — аутентификация Kerberos/GSSAPI, как на проде)")

        def ask(label: str, default: str) -> str:
            raw = input(f"  {label}" + (f" [{default}]" if default else "") + ": ").strip()
            return raw or default

        user_id = ask("user_id", cur.user_id)
        host = ask("host", cur.host)
        port_raw = ask("port", str(cur.port or 5432))
        database = ask("database", cur.database or "prom")
        dialect = ask("dialect (greenplum/postgres)", cur.dialect or "greenplum")
        try:
            port = int(port_raw)
        except ValueError:
            print("✗ port должен быть числом. Отмена.")
            return
        if not (user_id and host and database):
            print("✗ user_id, host и database обязательны. Отмена.")
            return

        cfg = ConnectionConfig(user_id=user_id, host=host, port=port, database=database, dialect=dialect)
        save_connection(cfg)
        self.agent.reload_db()
        self._status("проверяю подключение…")
        try:
            self.agent.db.test_connection()
        except Exception as exc:  # noqa: BLE001
            if is_kerberos_auth_error(exc):
                print(f"⚠ Конфиг сохранён. {KERBEROS_MESSAGE}")
            else:
                print(f"⚠ Конфиг сохранён, но подключиться не удалось: {exc}")
            return
        print(f"✓ Подключение настроено и проверено: {cfg.summary()}")
        print("  Подсказка: выполните /refresh_metadata, чтобы пересобрать метаданные из этой БД.")

    def _model(self) -> None:
        cur = self.agent.llm.cfg
        print(f"\nТекущая модель: {cur.model} (provider={cur.provider})")
        print("Доступные модели GigaChat (прод):")
        for i, m in enumerate(_GIGACHAT_MODELS):
            print(f"  [{i}] {m}")
        ans = input("Номер модели (или Enter — отмена): ").strip()
        if not ans:
            return
        try:
            model = _GIGACHAT_MODELS[int(ans)]
        except (ValueError, IndexError):
            print("✗ Некорректный номер.")
            return
        try:
            self.agent.set_llm("gigachat", model)
        except Exception as exc:  # noqa: BLE001
            print(f"✗ Не удалось переключить модель: {exc}\n"
                  "  Нужны: пакет langchain-gigachat и env GIGACHAT_API_URL, JPY_API_TOKEN.")
            return
        print(f"✓ Модель переключена: {model} (provider=gigachat).")

    # ---------- обработка запроса ----------
    def _process_query(self, question: str) -> None:
        if not self.agent.db.is_configured:
            print("Не настроен коннект к БД, выполните /config_db_conn")
            return
        turn = self._with_status("анализирую запрос и строю план", lambda: self.agent.ask(question))
        guard = 0
        while turn.interrupt and guard < 10:
            guard += 1
            itype = turn.interrupt.get("type")
            if itype == "ambiguity":
                print("\n❓ " + turn.interrupt.get("question", "Уточните вариант:"))
                opts = turn.interrupt.get("options", [])
                for i, o in enumerate(opts):
                    tbls = ", ".join(t.split(".")[-1] for t in o.get("tables", []))
                    print(f"  [{i}] {o.get('label','')}  ({tbls})")
                    if o.get("rationale"):
                        print(f"      {o['rationale']}")
                ans = input("\nНомер варианта: ").strip()
                logger.info("CLI ответ (выбор варианта): %r", ans)
                turn = self._with_status("планирую", lambda: self.agent.respond(ans))
            elif itype == "approve_plan":
                print("\n── ПЛАН ──")
                print(turn.interrupt.get("plan_nl", ""))
                for n in turn.interrupt.get("notes", []):
                    print(f"  • {n}")
                for iss in turn.interrupt.get("issues", []):
                    print(f"  ⚠ {iss}")
                sql_preview = turn.interrupt.get("sql", "")
                if sql_preview:
                    print("\n── SQL ──\n" + sql_preview)
                ans = input("\n" + _OK_HINT).strip()
                logger.info("CLI ответ на план: %r", ans)
                turn = self._with_status("обрабатываю", lambda: self.agent.respond(ans))
            else:
                break
        self._render_result(turn)

    def _render_result(self, turn: Turn) -> None:
        if turn.status == "done" and turn.result:
            sql = turn.state.get("sql_display") or turn.state.get("sql", "")
            res = turn.result
            print(f"\n✅ Готово. Строк: {res.get('rowcount')} → {res.get('csv')}")
            if sql:
                print("\nSQL:\n" + sql)
            if _HAS_IPY:
                try:
                    df = pd.read_csv(res["csv"])
                    display(df.head(20))
                except Exception:  # noqa: BLE001
                    pass
        else:
            print(f"\n⚠ Статус: {turn.status}")
            for n in turn.notes:
                print(f"  • {n}")

    # ---------- основной цикл ----------
    def run(self) -> None:
        self._banner()
        while True:
            try:
                user_input = input("\n🟢 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nДо свидания!")
                break
            if not user_input:
                continue

            parts = user_input.split()
            cmd = parts[0].lower()
            args = parts[1:]
            logger.info("CLI ввод: %s", cmd if cmd.startswith("/") else "<запрос>")
            try:
                if cmd in ("/exit", "/quit"):
                    print("До свидания!")
                    break
                if cmd == "/help":
                    print(HELP_TEXT)
                elif cmd in ("/config_db_conn", "/config_db", "/config"):
                    self._config_db_conn()
                elif cmd == "/model":
                    self._model()
                elif cmd == "/table_list":
                    self._table_list()
                elif cmd == "/add_table":
                    if not self.agent.db.is_configured:
                        print("Не настроен коннект к БД, выполните /config_db_conn")
                    elif not args:
                        print("Формат: /add_table schema.table")
                    else:
                        self._add_table(args[0])
                elif cmd == "/remove_table":
                    if not args:
                        print("Формат: /remove_table schema.table")
                    else:
                        self._remove_table(args[0])
                elif cmd in ("/refresh_metadata", "/refrash_metadata", "/refresh"):
                    if not self.agent.db.is_configured:
                        print("Не настроен коннект к БД, выполните /config_db_conn")
                    else:
                        self._refresh_metadata()
                elif cmd == "/reset":
                    self._reset()
                elif cmd == "/clear":
                    if _HAS_IPY:
                        clear_output(wait=True)
                    self._banner()
                elif cmd.startswith("/"):
                    print(f"Неизвестная команда: {user_input}. /help — список команд.")
                else:
                    self._process_query(user_input)
            except Exception as exc:  # noqa: BLE001  (REPL не должен падать)
                if is_kerberos_auth_error(exc):
                    logger.warning("CLI: ошибка Kerberos на вводе %r: %s", user_input, exc)
                    print(f"⚠ {KERBEROS_MESSAGE}")
                else:
                    logger.exception("CLI: необработанная ошибка на вводе %r", user_input)
                    print(f"⚠ Ошибка: {exc}\n  Подробности в логе: {self.log_file}")
