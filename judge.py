"""Model judge for the EchoMedIQ evals: Qwen 3.8 27B on the team's LiteLLM proxy (self-hosted, $0).

Two checks, each one chat-completions call per row with a JSON answer:
    Value   does the audit Value have the same intent as the ground truth Value?   -> judge_qwen3.8-27b.json
    Reason  does the audit Reason agree with its own Criterion_Response and Value?  -> reason_qwen3.8-27b.json
            (audit only, no ground truth needed)

    evals/audit_results/<call>/ground_truth.json   GT Response + Value + Reason per param (build_cases.py)
    evals/audit_results/<call>/audit.csv           the audit being graded

Every verdict is stored with a fingerprint of the text it judged, so a rerun asks only rows that are new or whose
GT / audit text changed (--force asks every row again).

    python evals/judge.py [--check value|reason|all] [--force]
"""
import argparse
import csv
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

from schema import SCHEMA, audit_row, has_value

load_dotenv(Path(__file__).parent / ".env")  # LITELLM_PROXY_API_BASE, LITELLM_PROXY_API_KEY

JUDGE_NAME = "qwen3.8-27b"
# Qwen3.8-27B (FP8) on the proxy at $0. Thinking off (~1.5 s a request); 4 at a time to keep the shared GPU free.
# Chosen on a 167-sample benchmark (92.8%, close to Claude Haiku's 94.0%): see evals/archive/benchmark/.
JUDGE = {"base_url": os.environ.get("LITELLM_PROXY_API_BASE"), "model": "qwen3.8-27b",
         "api_key": os.environ.get("LITELLM_PROXY_API_KEY"), "workers": 4, "timeout": 300,
         "extra": {"reasoning_effort": "none"}}

VALUE_QUESTION = (
    "For this audit parameter, is the audit value correct: does it have the same intent as the ground truth value, "
    "even though the words are different? The audit value is often the call line where the doctor asks the question, "
    "while the ground truth value is the patient's answer to it; that is the same intent.\n"
    "Answer same_intent=true if same intent as the ground truth value; false if different intent: it is about "
    "something else or contradicts the ground truth value. Give a one-sentence reason.\n")
VALUE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["same_intent", "reason"],
    "properties": {"same_intent": {"type": "boolean"}, "reason": {"type": "string"}}}}}

REASON_QUESTION = (
    "This is one row of a medical call audit. Is the Reason consistent with the Criterion_Response and the Value? "
    "Consistent means the Reason supports the same answer as the Criterion_Response (it explains why it is True, or "
    "why it is False), does not argue for the opposite answer, and does not contradict the Value (it does not claim "
    "something was said when the Value is N/A, or describe different words than the Value quotes). A Reason that "
    "starts with a tag such as [from], [received], [approval], [order] or [none], or with 'Flagged:', is normal. "
    "Judge only consistency, not wording: a short or general Reason is fine if it agrees.\n"
    "Rules of this audit that decide the answer:\n"
    "- Facilitated By Tata1mg: only a case RECEIVED through Tata 1mg ([received]) is True. A doctor saying they are "
    "calling FROM Tata 1mg ([from]) is False and flagged, so 'Flagged: [from] ... calling from Tata 1mg' with False "
    "is consistent. [none] (Tata 1mg not mentioned) is False.\n"
    "- ERx Approval Conveyed: [approval] is True; [order] (only order or delivery talk) is False and flagged; [none] is False.\n"
    "- Empty is not a match: when either side of a comparison is missing, the answer is False. A Value like "
    "'ERx: X; Audio: N/A' means the call never stated it, so False is consistent with it.\n"
    "- 'N/A' in Value on a False row means there is nothing to quote.\n"
    "- Quotes are cut at about 15 words, so a Value may stop before the end of the sentence.\n"
    "Give a one-sentence explanation.")
REASON_SCHEMA = {"type": "json_schema", "json_schema": {"name": "consistency", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["consistent", "why"],
    "properties": {"consistent": {"type": "boolean"}, "why": {"type": "string"}}}}}


def rows_for_value(case):
    """(param, state) for every row where both the GT and the audit give a Value."""
    truth = json.loads((case / "ground_truth.json").read_text())
    with open(case / "audit.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for n, t in truth.items():
        row = audit_row(rows, n)
        if has_value(t.get("value")) and has_value(row.get("Value")):
            yield n, {"audit parameter": row.get("Audit_Param_Description", ""), "ground truth value": t["value"],
                      "ground truth evidence": re.sub(r"\s+", " ", t.get("evidence", "")).strip(" |"),
                      "audit value": row["Value"].strip()}


def rows_for_reason(case):
    """(param, state) for every audit row that is not NA: the question, the answer, the evidence and the reason."""
    with open(case / "audit.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("Criterion_Response") in (SCHEMA["yes"], SCHEMA["no"]):
                yield r["Param_ID"], {"audit parameter": r.get("Audit_Param_Description", ""),
                                      "criterion_response": r["Criterion_Response"],
                                      "value": r.get("Value", ""), "reason": r.get("Reason", "")}


def chat(client, prompt, schema):
    """One chat-completions call with a JSON schema -> the parsed JSON answer."""
    r = client.post(f"{JUDGE['base_url'].rstrip('/')}/chat/completions", timeout=JUDGE["timeout"],
                    headers={"Authorization": f"Bearer {JUDGE['api_key']}"},
                    json={"model": JUDGE["model"], "response_format": schema,
                          "messages": [{"role": "user", "content": prompt}], **JUDGE["extra"]})
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


def ask_value(client, state):
    out = chat(client, VALUE_QUESTION + "\n" + json.dumps(state, ensure_ascii=False, indent=1), VALUE_SCHEMA)
    return {"p_same": 1.0 if out["same_intent"] else 0.0, "reason": out["reason"]}


def ask_reason(client, state):
    out = chat(client, REASON_QUESTION + "\n\n" + json.dumps(state, ensure_ascii=False, indent=1), REASON_SCHEMA)
    return {"consistent": out["consistent"], "why": out["why"]}


def state_key(state):
    """Fingerprint of what was judged: a cached verdict is reused only while the text it saw is unchanged."""
    return hashlib.sha1(json.dumps(state, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def judge_case(client, out, states, fn, force):
    """Ask only the rows that are new or whose text changed since they were judged; drop rows no longer asked.
    -> (rows stored, rows asked now)."""
    done = {} if force or not out.exists() else json.loads(out.read_text())
    todo = {n: s for n, s in states.items() if done.get(n, {}).get("key") != state_key(s)}
    done = {n: v for n, v in done.items() if n in states}
    with ThreadPoolExecutor(JUDGE["workers"]) as pool:
        futures = {n: pool.submit(fn, client, s) for n, s in todo.items()}
        done.update({n: {**f.result(), "key": state_key(todo[n])} for n, f in futures.items()})
    out.write_text(json.dumps(dict(sorted(done.items())), indent=1, ensure_ascii=False))
    return len(done), len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=("value", "reason", "all"), default="all")
    ap.add_argument("--force", action="store_true", help="ask every row again, ignoring stored verdicts")
    ap.add_argument("--cases", type=Path, default=SCHEMA["cases"], help="case folder (default: evals/audit_results)")
    a = ap.parse_args()
    if not JUDGE["base_url"] or not JUDGE["api_key"]:
        raise SystemExit("Set LITELLM_PROXY_API_BASE and LITELLM_PROXY_API_KEY in .env")
    with httpx.Client() as client:
        for case in sorted(p for p in a.cases.iterdir() if (p / "ground_truth.json").exists()):
            line = case.name + ":"
            if a.check in ("value", "all"):
                n, new = judge_case(client, case / f"judge_{JUDGE_NAME}.json", dict(rows_for_value(case)), ask_value, a.force)
                line += f" value {n} rows ({new} asked)"
            if a.check in ("reason", "all"):
                n, new = judge_case(client, case / f"reason_{JUDGE_NAME}.json", dict(rows_for_reason(case)), ask_reason, a.force)
                line += f"  reason {n} rows ({new} asked)"
            print(line)


if __name__ == "__main__":
    main()
