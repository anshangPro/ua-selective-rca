from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ua_rca import CandidateWindow, SelectiveConfig, SelectiveRCAPipeline


def gpu_metadata() -> dict[str, str | None]:
    """Best-effort GPU audit without adding a CUDA/PyTorch runtime dependency."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, capture_output=True, timeout=10, check=False,
        )
        return {"nvidia_smi": result.stdout.strip() or None}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"nvidia_smi": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run posterior-weighted selective RCA on one external-RCA case.")
    parser.add_argument("--input", required=True, type=Path, help="JSON case with a windows list")
    parser.add_argument("--output", required=True, type=Path, help="auditable result JSON")
    parser.add_argument("--config", type=Path, help="optional experiment configuration JSON")
    parser.add_argument("--ablations", action="store_true", help="also emit single/equal/posterior/factor ablations")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    config_payload = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    selective = SelectiveConfig(**config_payload.get("selective", {}))
    detector_config = config_payload.get("detectors", {})
    weights = {name: item.get("weight", 1.0) for name, item in detector_config.items() if isinstance(item, dict)}
    windows = [
        CandidateWindow(float(row["start"]), row["detector_scores"], row["ranking"], float(row.get("quality", 1.0)))
        for row in payload["windows"]
    ]
    aggregation_config = config_payload.get("aggregation", {})
    pipeline = SelectiveRCAPipeline(
        selective, float(detector_config.get("temperature", 1.0)), weights,
        str(aggregation_config.get("mode", "posterior_quality_stability")),
        int(aggregation_config.get("stability_k", 5)),
    )
    decision, audit = pipeline.diagnose(windows)
    if args.ablations:
        audit["aggregation_ablations"] = pipeline.diagnose_ablations(windows)
    audit["case_id"] = payload.get("case_id")
    audit["manifest"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "gpu": gpu_metadata(),
        "config": str(args.config) if args.config else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(decision.as_dict(), indent=2))


if __name__ == "__main__":
    main()
