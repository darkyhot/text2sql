"""DeepSeek-тест бизнес-отчёта: строит отчёты по fact-таблицам и проверяет
свойства (структура, графики, бизнес-язык без матжаргона). Для отладки функции.

Запуск: python eval/test_report.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.catalog.catalog import Catalog
from text2sql.db.adapter import make_adapter
from text2sql.llm.client import LLMClient
from text2sql.report.builder import build_business_report

S = "s_grnplm_ld_salesntwrk_pcap_sn_uzp"
_MATH_JARGON = re.compile(r"(z-score|z score|дисперси|стандартн\w* отклонен|корреляц|квантил|перцентил|p-value)", re.I)

CASES = [
    {"table": f"{S}.uzp_dwh_fact_outflow", "where": None, "focus": "", "want_patterns": True},
    {"table": f"{S}.uzp_dwh_company_holding_metric", "where": None,
     "focus": "аналитика по оттоку ФОТ и потенциалу"},
    {"table": f"{S}.uzp_data_payroll_m", "where": None, "focus": "разрез по сегментам и видам зачислений"},
    {"table": f"{S}.uzp_dwh_sale_funnel_task", "where": None,
     "focus": "аналитика по сотрудникам и закрытию задач", "want_entity": True},
]


def check(md: str, res: dict, case: dict) -> list[tuple[str, bool, str]]:
    html = Path(res["html_path"]).read_text(encoding="utf-8")
    checks = [
        ("секций >= 4", res["sections"] >= 4, f"sections={res['sections']}"),
        ("графиков >= 3", res["charts"] >= 3, f"charts={res['charts']}"),
        ("есть 'Главное'", "🎯 Главное" in md, ""),
        ("есть 'обратить внимание'", "обратить внимание" in md, ""),
        ("бизнес-язык (без матжаргона)", not _MATH_JARGON.search(md),
         (_MATH_JARGON.search(md).group(0) if _MATH_JARGON.search(md) else "")),
        ("выводы непустые", md.count("![") >= 3 and len(md) > 800, f"len={len(md)}"),
        ("HTML со встроенными графиками (ECharts/PNG)",
         html.count("T2S.plot(") + html.count("data:image/png;base64") >= 3,
         f"echarts={html.count('T2S.plot(')}, png={html.count('data:image/png;base64')}"),
        ("имя файла с префиксом таблицы", "_business_report" in Path(res["md_path"]).name, ""),
    ]
    if case.get("want_entity"):
        checks.append(("есть аналитика по сотрудникам (ФИО)",
                       bool(re.search(r"fio|сотрудник", md, re.I)), ""))
    if case.get("want_patterns"):
        checks.append(("есть блок закономерностей/аномалий", "🔍 Закономерности" in md, ""))
        checks.append(("есть сезонность/повторяемость/всплеск",
                       bool(re.search(r"Сезонность|Повторяемость|Всплеск", md)), ""))
    return checks


def main() -> None:
    db, cat, llm = make_adapter(), Catalog.load(), LLMClient()
    total_ok = 0
    for c in CASES:
        name = c["table"].split(".")[-1]
        try:
            res = build_business_report(db, cat, llm, c["table"], where=c["where"],
                                        focus=c["focus"], progress=lambda m: None)
            md = Path(res["md_path"]).read_text(encoding="utf-8")
            checks = check(md, res, c)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: {exc}")
            continue
        passed = all(ok for _, ok, _ in checks)
        total_ok += int(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name} (строк {res['rows']})")
        for label, ok, detail in checks:
            print(f"    {'✓' if ok else '✗'} {label} {('· '+detail) if detail and not ok else ''}")
    print(f"\nИТОГ: {total_ok}/{len(CASES)} отчётов прошли проверки")


if __name__ == "__main__":
    main()
