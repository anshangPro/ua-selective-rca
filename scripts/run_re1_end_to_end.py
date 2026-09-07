"""Non-oracle RE1 RCA run: detection boundaries are inferred from metrics."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from RCAEval.e2e.baro import baro
from RCAEval.utility import read_metrics
from ua_rca import CandidateWindow, SelectiveConfig, SelectiveRCAPipeline, build_boundary_candidates
from ua_rca.adapters import ranking_from_ordered_services
from ua_rca.evaluation import ranking_metrics


def anomaly_signal(frame):
    """Mean absolute baseline-standardized metric deviation, without labels."""
    metric_columns = [name for name in frame.columns if name != "time"]
    baseline = frame[metric_columns].iloc[:max(2, len(frame) // 4)]
    center = baseline.median()
    scale = (baseline - center).abs().median().replace(0, 1e-8)
    return ((frame[metric_columns] - center).abs() / scale).mean(axis=1).tolist()


def root_service(case_name):
    return case_name.split("_", 3)[1]


def service_ranking(metric_ranking):
    result = []
    for metric in metric_ranking:
        service = metric.split("_", 1)[0]
        if service not in result:
            result.append(service)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run BARO with inferred, probabilistic RE1 onset boundaries.")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("re1-ob", "re1-ss", "re1-tt"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-limit", type=int, default=5)
    args = parser.parse_args()

    data = read_metrics(args.case)
    boundaries = build_boundary_candidates(
        anomaly_signal(data), data["time"].tolist(), candidate_limit=args.candidate_limit,
    )
    windows = []
    for candidate in boundaries["candidates"]:
        ranking = baro(data.copy(), inject_time=candidate["timestamp"], dataset=args.dataset)["ranks"]
        windows.append(CandidateWindow(candidate["timestamp"], candidate["detector_scores"], ranking_from_ordered_services(service_ranking(ranking))))
    config = json.loads((ROOT / "configs" / "re1_template.json").read_text())
    decision, audit = SelectiveRCAPipeline(SelectiveConfig(**config["selective"]), config["detectors"]["temperature"],
                                            {"changepoint": 1.0, "residual": 1.0}).diagnose(windows)
    truth = root_service(args.case.name)
    audit.update({"case_id": args.case.name, "boundary_detection": boundaries, "oracle_metrics":
                  ranking_metrics(list(decision.aggregate_ranking), {truth})})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2))
    print(json.dumps({"decision": decision.as_dict(), "oracle_metrics": audit["oracle_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
