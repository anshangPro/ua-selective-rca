# Uncertainty-Aware Selective RCA

Research prototype for *posterior-weighted, selective root-cause analysis under uncertain detection boundaries*.  The package is intentionally independent of a particular RCA implementation: CIRCA, RCD, and BARO rankings are supplied through a small adapter interface and the uncertainty layer performs the common boundary, aggregation, and selective-decision logic.

## What is implemented

- Two lightweight boundary detectors: a change-point score and an EWMA residual score.
- Temperature-calibrated posterior probabilities over candidate fault start times.
- Posterior-weighted multi-window aggregation with quality and ranking-stability factors.
- Selective decisions: Top-1, minimal cumulative-mass Top-k, or abstention with an explanation.
- Case-level evaluation: AC@k, MRR, Avg@5, coverage/risk, AURC, Top-k set recall, calibration error, and boundary-offset robustness.
- Reproducible JSON configuration, run manifest, synthetic smoke data, and unit tests.

This repository does **not** claim to bundle CIRCA, RCD, BARO, RCAEval, or their results.  Use the adapters in `src/ua_rca/adapters.py` to load rankings emitted by their official implementations, then run this layer on the same candidate windows.

## Quick start

The project has no required third-party runtime dependency.  Python 3.10+ is sufficient.

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_synthetic_demo.py
```

The demo writes an auditable run record to `artifacts/synthetic_demo/`.  For real data, start from `configs/re1_template.json` and follow `docs/REPRODUCIBILITY.md`.

## Canonical integration contract

For each fault case, external RCA tools provide a ranking for every candidate window:

```json
{
  "case_id": "case-001",
  "windows": {
    "1710000000": {"checkout": 0.82, "payment": 0.18},
    "1710000060": {"checkout": 0.71, "payment": 0.29}
  }
}
```

The key is the candidate start timestamp. Scores can be probabilities, arbitrary positive scores, or negative ranks; the adapter normalizes them.  The final result records the posterior, all window factors, aggregate ranking, confidence components, and any abstention reason.

To run a real precomputed case, use an input JSON with a `windows` list whose entries contain `start`, `detector_scores`, `ranking`, and optional `quality` fields:

```powershell
python scripts/run_case.py --input path/to/case.json --output artifacts/case-001.json --config configs/re1_template.json
```

## Reproducibility boundary

Run real experiments only after obtaining the official datasets and implementations under their respective licenses.  Commit configuration and summary metrics, not raw production telemetry or credentials.
