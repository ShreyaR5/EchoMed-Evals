# EchoMedIQ Evals

Grades each call's audit (23 params) against its ground truth (GT).
Standalone for now: not yet integrated with the orchestrator.

| Check | What is compared | Done by |
|---|---|---|
| Response | True / False answer: audit vs GT | Code |
| Value | Audit evidence has the same meaning as the GT | Qwen 3.8 27B |
| Reason | Audit Reason agrees with its own Response and Value | Qwen 3.8 27B |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in LITELLM_PROXY_API_BASE and LITELLM_PROXY_API_KEY
```

## Inputs

- **GT folder:** `<call_id>.csv`, or `<call_id>/<call_id>_GT.csv` (one folder per call).
- **Audit folder:** `<call_id>/audit.csv` (the inference output).

The GT and audit are matched by call ID.

## Set the paths (either way)

1. **In `build_cases.py`:** put your folders in `GT_DIR` and `AUDIT_DIR` at the top of the file.
2. **On the command line:** `python build_cases.py --gt <gt folder> --audits <audit folder>`

## Run

```bash
python build_cases.py      # pair each GT with its audit
python judge.py            # Qwen verdicts for Value and Reason
python run_evals.py        # scores and reports
```

## Results

- `eval_results/summary.html`: all calls, scores and misses
- `eval_results/<call_id>/result.html`: one call, all 23 params
