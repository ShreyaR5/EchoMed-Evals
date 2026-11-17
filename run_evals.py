"""EchoMedIQ evals: grade each call's audit (23 params) against its ground truth.

    evals/audit_results/<call>/ground_truth.json    {"<param>": {"name", "answer", "value", "evidence"}, ...}
    evals/audit_results/<call>/audit.csv            the audit CSV: 23 rows, True / False / NA
    evals/audit_results/<call>/judge_*.json         Value verdicts from judge.py (Qwen 3.8 27B)
    evals/audit_results/<call>/reason_*.json        Reason consistency verdicts from judge.py

Writes evals/audit_results/<call>/eval_report.html for every call and evals/audit_results/summary.html across calls, and evals/eval_results/:
summary.html plus <call>/result.html per call (the files to commit).
Response (code): "False" means an issue was found, so it is the positive class: TP = both False, FN = audit True but
GT False. The NA params (P13, P23) are left out.
Value (judge): graded where the GT has a Value. An empty audit Value there is wrong; otherwise the judge decides whether
the audit Value has the same intent as the GT (p_same >= THRESHOLD).
Reason (judge, audit only): does each row's Reason agree with its own Response and Value?

    python evals/build_cases.py && python evals/judge.py && python evals/run_evals.py
"""
import argparse
import csv
import html
import json
import re
from pathlib import Path

from schema import SCHEMA, audit_row, has_value

THRESHOLD = 0.5
RESULTS = Path(__file__).parent / "eval_results"


def bucket(audit, truth, s):
    if truth == s["na"]:
        return "NA" if audit == s["na"] else "INVALID"
    if audit not in (s["yes"], s["no"]):
        return "INVALID"
    issue_audit, issue_truth = audit == s["no"], truth == s["no"]
    return {(True, True): "TP", (False, False): "TN", (True, False): "FP", (False, True): "FN"}[issue_audit, issue_truth]


def completeness(header, rows, truth, s):
    """Format problems in the CSV, one string each; empty when the CSV is complete and well formed."""
    problems = []
    if header != s["columns"]:
        problems.append(f"Header is {header}, expected {s['columns']}")
    if len(rows) != len(s["params"]):
        problems.append(f"{len(rows)} rows, expected {len(s['params'])}")
    for n, row in zip(s["params"], rows):
        if row.get("Param_ID", "").strip() != n:
            problems.append(f"Row {n} is '{row.get('Param_ID')}', expected '{n}'")
        response, reason = row.get("Criterion_Response"), row.get("Reason", "")
        allowed = (s["na"],) if n in s["na_params"] else (s["yes"], s["no"])
        if response not in allowed:
            problems.append(f"Row {n}: Criterion_Response '{response}' is not {'/'.join(allowed)}")
        if row.get("Confidence") not in ("Hi", "Med", "Lo"):
            problems.append(f"Row {n}: Confidence '{row.get('Confidence')}' is not Hi/Med/Lo")
        if row.get("Flag") != ("true" if response == s["no"] else "false"):
            problems.append(f"Row {n}: Flag '{row.get('Flag')}' does not follow Criterion_Response")
        for col, want in s["fixed"].get(n, {}).items():
            if row.get(col) != want:
                problems.append(f"Row {n}: {col} '{row.get(col)}' is not the fixed '{want}'")
        if not reason.strip():
            problems.append(f"Row {n}: Reason is empty")
        tag = re.match(r"(?:Flagged: )?\[(\w+)\]", reason)
        if n in s["tags"]:
            if not tag or tag.group(1) not in s["tags"][n]:
                problems.append(f"Row {n}: Reason does not start with one of the tags {s['tags'][n]}")
            elif reason.startswith("Flagged:") != (tag.group(1) == s["flag_tags"][n]):
                problems.append(f"Row {n}: 'Flagged:' must be on the [{s['flag_tags'][n]}] tag only")
        elif tag or reason.startswith("Flagged:"):
            problems.append(f"Row {n}: only P7 and P8 carry a tag or 'Flagged:'")
    return problems


def value_verdict(gt_value, audit_value, judged):
    """'same' / 'different' / 'missing' for a graded row, None when the GT has no Value to compare with."""
    if not has_value(gt_value):
        return None
    if not has_value(audit_value):
        return "missing"
    if judged is None:
        return "not judged"
    return "same" if judged["p_same"] >= THRESHOLD else "different"


def evaluate(case, s):
    truth = json.loads((case / "ground_truth.json").read_text())
    with open(case / "audit.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)
    judges = {p.stem.removeprefix("judge_"): json.loads(p.read_text()) for p in sorted(case.glob("judge_*.json"))}
    results = []
    for n in s["params"]:
        t, row = truth[n], audit_row(rows, n)
        audit, value = row.get("Criterion_Response", "missing"), row.get("Value", "").strip()
        results.append({"param": n, "name": t["name"], "audit": audit, "truth": t["answer"],
                        "bucket": bucket(audit, t["answer"], s),
                        "value": value, "gt_value": t.get("value", ""),
                        "judged": {j: v.get(n) for j, v in judges.items()},
                        "verdict": {j: value_verdict(t.get("value", ""), value, v.get(n)) for j, v in judges.items()}})
    counts = {b: sum(r["bucket"] == b for r in results) for b in ("TP", "TN", "FP", "FN", "INVALID", "NA")}
    correct = counts["TP"] + counts["TN"]
    graded = correct + counts["FP"] + counts["FN"] + counts["INVALID"]
    values = {j: {"right": sum(r["verdict"][j] == "same" for r in results),
                  "graded": sum(r["verdict"][j] is not None for r in results)} for j in judges}
    # Reason consistency (audit only): does each row's Reason agree with its own Response and Value?
    reasons = {p.stem.removeprefix("reason_"): json.loads(p.read_text()) for p in sorted(case.glob("reason_*.json"))}
    reason = {j: {"consistent": sum(v["consistent"] for v in d.values()), "checked": len(d),
                  "flagged": [(n, v["why"]) for n, v in sorted(d.items(), key=lambda x: s["params"].index(x[0]))
                              if not v["consistent"]]} for j, d in reasons.items()}
    return {"call": case.name, "results": results, "counts": counts, "correct": correct, "graded": graded,
            "accuracy": correct / graded if graded else 0.0, "problems": completeness(header, rows, truth, s),
            "judges": list(judges), "values": values, "reason": reason}


STYLE = """
:root { --bg: #F7FAFA; --surface: #FFFFFF; --fg: #14262B; --muted: #5A6B70; --line: #D7E3E4;
  --accent: #0B5C6B; --good: #1E7A55; --bad: #B4472F; --bad-bg: #FBEDEA }
@media (prefers-color-scheme: dark) { :root { --bg: #0E1A1D; --surface: #14252A; --fg: #E4EEEF; --muted: #93A8AC;
  --line: #26393E; --accent: #6CC3CF; --good: #6FD0A4; --bad: #F08A72; --bad-bg: #3A211C; color-scheme: dark } }
body { background: var(--bg); color: var(--fg); font: 15px/1.5 'Segoe UI', Arial, sans-serif; margin: 0; padding: 32px 16px }
main { max-width: 1100px; margin: 0 auto; display: flex; flex-direction: column; gap: 24px }
h1, h2 { color: var(--accent); margin: 0 } h1 { font-size: 1.7rem } h2 { font-size: 1.15rem }
.tiles { display: flex; flex-wrap: wrap; gap: 12px }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 16px; min-width: 90px }
.tile b { display: block; font-size: 1.4rem } .tile span { color: var(--muted); font-size: .8rem }
.wrap { overflow-x: auto }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top }
th { font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted) }
tr.wrong td { background: var(--bad-bg) }
.FP, .FN, .INVALID, .bad, .different, .missing { color: var(--bad); font-weight: 700 }
.TP, .TN, .good, .same { color: var(--good); font-weight: 700 }
.NA, .muted { color: var(--muted) } a { color: var(--accent) }
"""
SKIPPED = ("NA",)


def page(title, body):
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>")


def tiles(items):
    return "<div class='tiles'>" + "".join(f"<div class='tile'><b>{v}</b><span>{k}</span></div>" for k, v in items) + "</div>"


def value_score(values):
    return f"{values['right']}/{values['graded']}" if values["graded"] else "–"


def call_report(r, s):
    c = r["counts"]
    rows = "".join(
        f"<tr class='{'' if x['bucket'] in ('TP', 'TN') + SKIPPED else 'wrong'}'><td>{x['param']}</td><td>{html.escape(x['name'])}</td>"
        f"<td>{x['audit']}</td><td>{x['truth']}</td>"
        f"<td class='{x['bucket']}'>{x['bucket']}</td></tr>"
        for x in r["results"])
    problems = (f"<p class='good'>All {len(s['params'])} rows present, in order, valid values.</p>" if not r["problems"] else
                "<ul>" + "".join(f"<li class='bad'>{html.escape(p)}</li>" for p in r["problems"]) + "</ul>")
    body = (f"<h1>Eval report · {r['call']}</h1><p class='muted'><a href='../summary.html'>All calls</a></p>"
            + tiles([("Response accuracy", f"{r['accuracy']:.1%}"), ("Correct", f"{r['correct']}/{r['graded']}"), ("TP", c["TP"]),
                     ("TN", c["TN"]), ("FP", c["FP"]), ("FN", c["FN"])]
                    + [(f"Value · {j}", value_score(r["values"][j])) for j in r["judges"]])
            + f"<h2>Response ({s['yes']} / {s['no']})</h2><p class='muted'>{s['no']} = issue found. NA params "
            "are left out of accuracy.</p>"
            + "<div class='wrap'><table><tr><th>#</th><th>Param</th><th>Audit</th><th>Ground truth</th><th>Result</th></tr>"
            + f"{rows}</table></div>" + value_section(r) + reason_section(r) + f"<h2>Completeness</h2>{problems}")
    return page(f"Eval {r['call']}", body)


def value_section(r):
    if not r["judges"]:
        return "<h2>Value</h2><p class='muted'>No judge results yet: run judge.py.</p>"

    def cell(x, j):
        p = x["judged"][j]
        score = f" <span class='muted'>{p['p_same']:.2f}</span>" if p else ""
        return f"<td class='{x['verdict'][j]}'>{x['verdict'][j]}{score}</td>"

    graded = [x for x in r["results"] if has_value(x["gt_value"])]
    head = "".join(f"<th>{html.escape(j)}</th>" for j in r["judges"])
    rows = "".join(
        f"<tr><td>{x['param']}</td><td>{html.escape(x['gt_value'])}</td>"
        f"<td>{html.escape(x['value']) or '<span class=muted>empty</span>'}</td>"
        + "".join(cell(x, j) for j in r["judges"]) + "</tr>"
        for x in graded)
    return (f"<h2>Value</h2><p class='muted'>Rows where the GT has a Value. The judge decides whether the audit Value points to the "
            f"same fact (p_same ≥ {THRESHOLD}); an empty audit Value is missing.</p><div class='wrap'><table><tr><th>#</th>"
            f"<th>GT Value</th><th>Audit Value</th>{head}</tr>{rows}</table></div>")


def reason_score(d):
    return f"{d['consistent']}/{d['checked']}" if d["checked"] else "–"


def reason_section(r):
    if not r["reason"]:
        return "<h2>Reason consistency</h2><p class='muted'>No results yet: run judge.py.</p>"
    out = ""
    for j, d in r["reason"].items():
        rows = "".join(f"<tr><td>{n}</td><td class='bad'>{html.escape(why)}</td></tr>" for n, why in d["flagged"]) \
            or "<tr><td colspan='2' class='good'>Every Reason agrees with its Response and Value</td></tr>"
        out += (f"<h2>Reason consistency · {html.escape(j)} · {reason_score(d)}</h2><p class='muted'>Audit only: does each "
                "row's Reason agree with its own Response and Value?</p>"
                f"<div class='wrap'><table><tr><th>#</th><th>Why it is inconsistent</th></tr>{rows}</table></div>")
    return out


def summary_report(reports, s):
    total = {b: sum(r["counts"][b] for r in reports) for b in ("TP", "TN", "FP", "FN")}
    correct, n = sum(r["correct"] for r in reports), sum(r["graded"] for r in reports)
    judges = sorted({j for r in reports for j in r["judges"]})
    value_totals = {j: {k: sum(r["values"].get(j, {}).get(k, 0) for r in reports) for k in ("right", "graded")} for j in judges}
    reason_judges = sorted({j for r in reports for j in r["reason"]})
    reason_totals = {j: {k: sum(r["reason"].get(j, {}).get(k, 0) for r in reports) for k in ("consistent", "checked")}
                     for j in reason_judges}
    calls = "".join(
        f"<tr><td><a href='{r['call']}/eval_report.html'>{r['call']}</a></td><td>{r['accuracy']:.1%}</td><td>{r['graded']}</td>"
        f"<td>{r['counts']['TP']}</td><td>{r['counts']['TN']}</td><td class='{'FP' if r['counts']['FP'] else ''}'>"
        f"{r['counts']['FP']}</td><td class='{'FN' if r['counts']['FN'] else ''}'>{r['counts']['FN']}</td>"
        + "".join(f"<td>{value_score(r['values'][j]) if j in r['values'] else '–'}</td>" for j in judges)
        + f"<td class='{'bad' if r['problems'] else 'good'}'>{'issues' if r['problems'] else 'ok'}</td></tr>"
        for r in reports)
    misses = [(r["call"], x) for r in reports for x in r["results"] if x["bucket"] not in ("TP", "TN") + SKIPPED]
    miss_rows = "".join(f"<tr><td>{call}</td><td>{x['param']} {html.escape(x['name'])}</td><td>{x['audit']}</td>"
                        f"<td>{x['truth']}</td><td class='{x['bucket']}'>{x['bucket']}</td></tr>"
                        for call, x in misses) or "<tr><td colspan='5' class='good'>No misses</td></tr>"
    body = (f"<h1>EchoMedIQ eval summary</h1><p class='muted'>{len(reports)} calls · {len(s['params'])} params each · "
            f"{n} graded</p>"
            + tiles([("Response accuracy", f"{correct / n:.1%}" if n else "–"), ("Correct", f"{correct}/{n}"), ("TP", total["TP"]),
                     ("TN", total["TN"]), ("FP", total["FP"]), ("FN", total["FN"])]
                    + [(f"Value · {j}", value_score(value_totals[j])) for j in judges]
                    + [(f"Reason · {j}", reason_score(reason_totals[j])) for j in reason_judges])
            + "<h2>Per call</h2><div class='wrap'><table><tr><th>Call</th><th>Response</th><th>Graded</th><th>TP</th><th>TN</th>"
            + "<th>FP</th><th>FN</th>" + "".join(f"<th>Value · {html.escape(j)}</th>" for j in judges)
            + f"<th>Completeness</th></tr>{calls}</table></div>"
            + "<h2>Response misses</h2><div class='wrap'><table><tr><th>Call</th><th>Param</th><th>Audit</th>"
            + f"<th>Ground truth</th><th>Result</th></tr>{miss_rows}</table></div>")
    return page("EchoMedIQ eval summary", body)


def pct(a, b):
    return f"{a}/{b} · {a / b:.1%}" if b else "–"


RESULT_WORD = {"TP": "correct", "TN": "correct", "FP": "wrong", "FN": "wrong", "INVALID": "invalid",
               "NA": "NA"}
VERDICT_CLASS = {"correct": "good", "same": "good", "consistent": "good",
                 "wrong": "bad", "invalid": "bad", "different": "bad", "missing": "bad", "not consistent": "bad"}


def scores(r, judge):
    v = r["values"].get(judge, {"right": 0, "graded": 0})
    rs = r["reason"].get(judge, {"consistent": 0, "checked": 0, "flagged": []})
    return (r["correct"], r["graded"], v["right"], v["graded"], rs["consistent"], rs["checked"]), rs


def score_cells(n):
    return (f"<td>{pct(n[0], n[1])}</td><td>{pct(n[2], n[3])}</td><td>{pct(n[4], n[5])}</td>"
            f"<td><b>{pct(n[0] + n[2] + n[4], n[1] + n[3] + n[5])}</b></td>")


def cell(word):
    return f"<td class='{VERDICT_CLASS.get(word, 'muted')}'>{html.escape(word)}</td>"


def results_summary(reports, s, judge):
    """All calls: scores per call and every miss."""
    graded = [r for r in reports if r["graded"]]
    rows, misses, T = "", "", [0] * 6
    for r in graded:
        n, rs = scores(r, judge)
        T = [a + b for a, b in zip(T, n)]
        rows += f"<tr><td><a href='{r['call']}/result.html'>{r['call']}</a></td>{score_cells(n)}</tr>"
        flagged = {p for p, _ in rs["flagged"]}
        for x in r["results"]:
            checks = ([("Response", f"audit {x['audit']}, GT {x['truth']}")] if x["bucket"] in ("FP", "FN", "INVALID") else []) \
                + ([("Value", x["verdict"][judge])] if x["verdict"].get(judge) in ("different", "missing") else []) \
                + ([("Reason", "not consistent")] if x["param"] in flagged else [])
            misses += "".join(f"<tr><td>{r['call']}</td><td>{x['param']} {html.escape(x['name'])}</td><td>{c}</td>"
                              f"<td class='bad'>{html.escape(w)}</td></tr>" for c, w in checks)
    body = ("<h1>EchoMedIQ Evaluation Results</h1>"
            + tiles([("Response", pct(T[0], T[1])), ("Value", pct(T[2], T[3])), ("Reason", pct(T[4], T[5])),
                     ("Overall", pct(T[0] + T[2] + T[4], T[1] + T[3] + T[5]))])
            + "<h2>Per call</h2><div class='wrap'><table><tr><th>Call</th><th>Response</th><th>Value</th><th>Reason</th>"
            + f"<th>Overall</th></tr>{rows}<tr><td><b>Total</b></td>{score_cells(T)}</tr></table></div>"
            + "<h2>Misses</h2><div class='wrap'><table><tr><th>Call</th><th>Param</th><th>Check</th><th>Result</th></tr>"
            + (misses or "<tr><td colspan='4' class='good'>No misses</td></tr>") + "</table></div>")
    return page("EchoMedIQ Evaluation Results", body)


def result_call(r, judge):
    """One call: its scores, and per param the answers, both Values and the verdicts, with the judge's reason on misses."""
    n, rs = scores(r, judge)
    flagged = dict(rs["flagged"])
    rows = ""
    for x in r["results"]:
        reason = ("–" if x["audit"] not in ("True", "False") or not rs["checked"]
                  else "not consistent" if x["param"] in flagged else "consistent")
        verdict = x["verdict"].get(judge) or "–"
        why = []
        if verdict == "different" and x["judged"].get(judge):
            why.append("Value: " + x["judged"][judge].get("reason", ""))
        if reason == "not consistent":
            why.append("Reason: " + flagged[x["param"]])
        rows += (f"<tr class='{'wrong' if 'bad' in {VERDICT_CLASS.get(w) for w in (RESULT_WORD[x['bucket']], verdict, reason)} else ''}'>"
                 f"<td>{x['param']}</td><td>{html.escape(x['name'])}</td><td>{x['audit']}</td><td>{x['truth'] or '–'}</td>"
                 + cell(RESULT_WORD[x["bucket"]])
                 + f"<td>{html.escape(x['gt_value']) or '–'}</td><td>{html.escape(x['value']) or '–'}</td>"
                 + cell(verdict) + cell(reason) + f"<td class='muted'>{html.escape(' '.join(why))}</td></tr>")
    body = (f"<h1>Evaluation Result · Call {r['call']}</h1><p class='muted'><a href='../summary.html'>All calls</a></p>"
            + tiles([("Response", pct(n[0], n[1])), ("Value", pct(n[2], n[3])), ("Reason", pct(n[4], n[5])),
                     ("Overall", pct(n[0] + n[2] + n[4], n[1] + n[3] + n[5]))])
            + "<div class='wrap'><table><tr><th>Param</th><th>Name</th><th>Audit</th><th>GT</th><th>Response</th>"
            + "<th>GT Value</th><th>Audit Value</th><th>Value</th><th>Reason</th><th>Judge's Note</th></tr>"
            + f"{rows}</table></div>")
    return page(f"Evaluation Result {r['call']}", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=SCHEMA["cases"], help="case folder (default: evals/audit_results)")
    a = ap.parse_args()
    s, cases = SCHEMA, a.cases
    reports = []
    for case in sorted(p for p in cases.iterdir() if (p / "audit.csv").exists() and (p / "ground_truth.json").exists()):
        r = evaluate(case, s)
        (case / "eval_report.html").write_text(call_report(r, s), encoding="utf-8")
        reports.append(r)
        print(f"{r['call']}: response {r['correct']}/{r['graded']} = {r['accuracy']:.1%}  TP {r['counts']['TP']} TN {r['counts']['TN']} "
              f"FP {r['counts']['FP']} FN {r['counts']['FN']}"
              f"  completeness {'ok' if not r['problems'] else len(r['problems'])}"
              + "".join(f"  value[{j}] {value_score(r['values'][j])}" for j in r["judges"])
              + "".join(f"  reason[{j}] {reason_score(d)}" for j, d in r["reason"].items()))
    (cases / "summary.html").write_text(summary_report(reports, s), encoding="utf-8")
    RESULTS.mkdir(exist_ok=True)
    judge = next((j for r in reports for j in r["judges"]), None)
    out = RESULTS / "summary.html"
    out.write_text(results_summary(reports, s, judge), encoding="utf-8")
    for r in reports:
        if r["graded"]:
            (RESULTS / r["call"]).mkdir(exist_ok=True)
            (RESULTS / r["call"] / "result.html").write_text(result_call(r, judge), encoding="utf-8")
    correct, graded = sum(r["correct"] for r in reports), sum(r["graded"] for r in reports)
    print(f"overall response: {correct}/{graded} = {correct / graded:.1%}  -> {cases / 'summary.html'}, {out}")


if __name__ == "__main__":
    main()
