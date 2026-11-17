"""Build evals/audit_results/<call>/ from the GT CSVs and the audits. Rerun it after editing the GT.

GT (--gt), either layout:
    <gt>/[<num>_]<call>.csv              flat: every GT in one folder
    <gt>/<call>/<call>_GT.csv            one folder per call, next to its mp3 / pdf / srt
Audit (--audits):
    <audits>/<call>/audit.csv            the 23-row audit being graded (inference output)
A GT is matched to the audit with the same call id (the mp3 name).

    python evals/build_cases.py                                         # uses GT_DIR and AUDIT_DIR below
    python evals/build_cases.py --gt <gt folder> --audits <audit folder>  # or pass the folders
"""
import csv
import json
import shutil
from pathlib import Path

# ---- Input folders: put your paths here, or pass --gt / --audits on the command line ----
GT_DIR = Path("path/to/ground_truth")        # the GT CSVs
AUDIT_DIR = Path("path/to/audits")           # the audit CSVs from the inference pipeline, <call>/audit.csv
# -----------------------------------------------------------------------------------------

CASES = Path(__file__).parent / "audit_results"


def parse_gt(path):
    truth = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            truth[r["Param_ID"]] = {"name": r["Audit_Param_Name"], "answer": r["Criterion_Response"],
                                    "value": r["Value"], "evidence": r["Reason"]}
    return truth


def gt_files(folder):
    """{call id: GT csv} from a flat folder of GTs or from one folder per call."""
    flat = {p.stem.split("_")[-1]: p for p in folder.glob("*.csv")}
    nested = {p.parent.name: p for p in folder.glob("*/*_GT.csv")}
    return dict(sorted({**flat, **nested}.items()))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=GT_DIR, help=f"GT folder (default: {GT_DIR})")
    ap.add_argument("--audits", type=Path, default=AUDIT_DIR, help=f"audit folder (default: {AUDIT_DIR})")
    ap.add_argument("--cases", type=Path, default=CASES, help="eval case folder to write (default: evals/audit_results)")
    a = ap.parse_args()
    for folder in (a.gt, a.audits):
        if not folder.is_dir():
            raise SystemExit(f"Folder not found: {folder}. Set GT_DIR / AUDIT_DIR in this file, or pass --gt / --audits.")
    gts = gt_files(a.gt)
    if not gts:
        raise SystemExit(f"No GT CSVs found in {a.gt}")
    # start clean: drop calls left over from earlier inputs
    if a.cases.exists():
        for old in a.cases.iterdir():
            if old.is_dir() and old.name not in {c[:5] for c in gts}:
                shutil.rmtree(old)
    for call, gt in gts.items():
        audit = a.audits / call / "audit.csv"
        if not audit.exists():
            print(f"{call[:5]}: no audit at {audit}, skipped")
            continue
        case = a.cases / call[:5]
        case.mkdir(parents=True, exist_ok=True)
        truth = parse_gt(gt)
        (case / "ground_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
        shutil.copy(audit, case / "audit.csv")
        print(f"{call[:5]}: {len(truth)} params")


if __name__ == "__main__":
    main()
