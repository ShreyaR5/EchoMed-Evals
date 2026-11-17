"""The audit CSV format the evals read: 23 params.

23 params matched by Param_ID, True / False / NA (P13 and P23 always NA), 12 columns, evidence in Value.
P7 and P8 start their Reason with a tag, and their flagged Falses with "Flagged:".
Name, description, domain, significance and Source are fixed per param (params.json).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
PARAMS = json.loads((HERE / "params.json").read_text())

SCHEMA = {
    "cases": HERE / "audit_results", "key": "Param_ID", "value": "Value", "description": "Audit_Param_Description",
    "yes": "True", "no": "False", "na": "NA",
    "columns": ["Param_ID", "Audit_Param_Name", "Audit_Param_Description", "Domain", "Significance_Level",
                "Criterion_Response", "Flag", "Confidence", "Value", "Source", "Reason", "Note"],
    "params": [p["id"] for p in PARAMS], "na_params": {p["id"] for p in PARAMS if p["phase"] == 2},
    "fixed": {p["id"]: {"Audit_Param_Name": p["name"], "Audit_Param_Description": p["description"],
                        "Domain": p["domain"], "Significance_Level": p["significance"], "Source": p["source"]}
              for p in PARAMS},
    "tags": {"P7": ("received", "from", "none"), "P8": ("approval", "order", "none")},
    "flag_tags": {"P7": "from", "P8": "order"},
}


def has_value(v):
    """A Value to compare: empty and 'N/A' (a False row with nothing quoted) are no value."""
    return bool(v and v.strip() and v.strip().upper() != "N/A")


def audit_row(rows, n):
    """The audit CSV row for param n (by Param_ID); {} when it is missing."""
    return next((r for r in rows if r.get("Param_ID", "").strip() == n), {})
