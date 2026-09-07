from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ua_rca import CandidateWindow, SelectiveConfig, SelectiveRCAPipeline
from ua_rca.evaluation import ranking_metrics


def main() -> None:
    windows = [
        CandidateWindow(960, {"changepoint": 1.7, "residual": 1.3}, {"checkout": 0.72, "payment": 0.18, "catalog": 0.10}, 0.92),
        CandidateWindow(1020, {"changepoint": 2.4, "residual": 2.1}, {"checkout": 0.83, "payment": 0.11, "catalog": 0.06}, 1.00),
        CandidateWindow(1080, {"changepoint": 1.4, "residual": 1.6}, {"checkout": 0.62, "payment": 0.26, "catalog": 0.12}, 0.83),
    ]
    pipeline = SelectiveRCAPipeline(SelectiveConfig(top1_confidence=0.55, top1_margin=0.12, min_stability=0.35))
    decision, audit = pipeline.diagnose(windows)
    ranking = list(decision.aggregate_ranking)
    audit["metrics"] = ranking_metrics(ranking, {"checkout"})
    audit["manifest"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "dataset": "synthetic-smoke-case", "seed": 7,
    }
    output = ROOT / "artifacts" / "synthetic_demo"
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit["decision"], indent=2))
    print(f"Audit record: {output / 'result.json'}")


if __name__ == "__main__":
    main()
