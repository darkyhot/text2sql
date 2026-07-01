"""Регенерация метаданных по манифесту tables_list.csv (адаптивный сэмпл,
таймаут 5 мин на таблицу, пропуск сбойных). Generic: любая БД.

Использование:
    python scripts/refresh_metadata.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.catalog.refresh import MetadataRefresh
from text2sql.db.adapter import make_adapter
from text2sql.llm.client import LLMClient
from text2sql.logging_setup import setup_logging
from text2sql.trace import Tracer


def main() -> None:
    setup_logging()
    tracer = Tracer("refresh")
    db = make_adapter(tracer=tracer)
    # llm — для генерации русских описаний колонок без комментария/сида
    res = MetadataRefresh(db, llm=LLMClient(tracer=tracer)).refresh_manifest(
        progress_callback=lambda m: print(f"  …{m}"))
    print(f"Таблиц: {res['tables']} | колонок: {res['columns']} | join-кандидатов: {res['join_candidates']}")
    if res["failed"]:
        print("Пропущены:")
        for fqn, err in res["failed"]:
            print(f"  {fqn}: {err}")


if __name__ == "__main__":
    main()
