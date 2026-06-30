"""Smoke-тест фундамента: config + db-adapter (guards) + llm-adapter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.config import DB, LLM
from text2sql.db.adapter import GuardError, make_adapter
from text2sql.llm.client import LLMClient
from text2sql.trace import Tracer

tr = Tracer("smoke")
db = make_adapter(tracer=tr)

print("== DB: list tables ==")
for s, t in db.list_user_tables("s_grnplm%"):
    print(" ", s, t)

print("== DB: read-only SELECT with auto-limit ==")
S = "s_grnplm_ld_salesntwrk_pcap_sn_uzp"
res = db.run_select(f"select count(distinct tb_id) tb, count(distinct new_gosb_id) gosb from {S}.uzp_dim_gosb")
print("  cols:", res.columns, "rows:", res.rows, "cost:", res.cost)

print("== DB: guard rejects DML ==")
try:
    db.run_select("delete from foo")
    print("  FAIL: DML not rejected")
except GuardError as e:
    print("  OK rejected:", e)

print("== DB: guard rejects multi-statement ==")
try:
    db.run_select("select 1; select 2")
    print("  FAIL: multi not rejected")
except GuardError as e:
    print("  OK rejected:", e)

print("== LLM: plain completion ==")
llm = LLMClient(tracer=tr)
r = llm.complete("Ты лаконичный ассистент.", "Ответь одним словом: столица России?", max_tokens=2000)
print("  finish:", r.finish_reason, "| text:", repr(r.text), "| reasoning_len:", len(r.reasoning))

print("== LLM: JSON completion ==")
j = llm.complete_json(
    "Ты извлекаешь сущности.",
    'Верни {"city": <город>, "country": <страна>} для "Париж".',
    max_tokens=2000,
)
print("  json:", j)

print("\nALL SMOKE CHECKS DONE")
