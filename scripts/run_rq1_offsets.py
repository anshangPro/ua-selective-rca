"""Case-level RQ1: measure RCA degradation under onset-boundary offsets."""
import argparse
import json
import time
from pathlib import Path

from RCAEval.utility import read_metrics
from RCAEval.e2e.baro import baro
from RCAEval.e2e.circa import circa

ROOT = Path(__file__).resolve().parents[1]


def service_ranking(metrics):
    """Collapse metric rankings to unique service rankings for RE1 labels."""
    result = []
    for metric in metrics:
        service = metric.split("_", 1)[0]
        if service not in result:
            result.append(service)
    return result


def scores(ranking, truth):
    position = next((i + 1 for i, value in enumerate(ranking) if value == truth), None)
    return {"ac_at_1": float(position == 1), "ac_at_3": float(position is not None and position <= 3),
            "ac_at_5": float(position is not None and position <= 5), "mrr": 1.0 / position if position else 0.0,
            "avg_at_5": float(position is not None and position <= 5) / min(5, len(ranking))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "RCAEval")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "rq1_offsets.json")
    parser.add_argument("--limit-per-system", type=int, default=1)
    parser.add_argument("--systems", nargs="+", choices=("ob", "ss", "tt"), default=("ob", "ss", "tt"))
    parser.add_argument("--offsets", nargs="+", type=int, help="subset of configured offsets")
    parser.add_argument("--resume", action="store_true", help="reuse completed rows in --output")
    args = parser.parse_args()
    offsets = args.offsets or json.loads((ROOT / "configs" / "re1_template.json").read_text())["offset_seconds"]
    methods = {"BARO": baro, "CIRCA": circa}
    existing = json.loads(args.output.read_text()).get("rows", []) if args.resume and args.output.exists() else []
    rows = list(existing)
    completed = {(row["case_id"], row["baseline"], row["offset_seconds"]) for row in rows}

    def persist():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"offsets": offsets, "rows": rows}, indent=2))
    all_systems = (("ob", "re1ob", "re1-ob"), ("ss", "re1ss", "re1-ss"), ("tt", "re1tt", "re1-tt"))
    for short_name, prefix, dataset in all_systems:
        if short_name not in args.systems:
            continue
        cases = sorted(args.data.glob(prefix + "_*"))[:args.limit_per_system]
        for case in cases:
            _, truth, _, _ = case.name.split("_", 3)
            data = read_metrics(case)
            onset = int((case / "inject_time.txt").read_text().strip())
            for offset in offsets:
                for name, method in methods.items():
                    if (case.name, name, offset) in completed:
                        continue
                    started = time.perf_counter()
                    try:
                        rank = service_ranking(method(data.copy(), inject_time=onset + offset, dataset=dataset)["ranks"])
                        row = scores(rank, truth)
                        row.update({"case_id": case.name, "system": prefix, "baseline": name, "offset_seconds": offset,
                                    "elapsed_seconds": round(time.perf_counter() - started, 4), "status": "ok"})
                    except Exception as error:
                        row = {"case_id": case.name, "system": prefix, "baseline": name, "offset_seconds": offset,
                               "elapsed_seconds": round(time.perf_counter() - started, 4), "status": "failed", "error": repr(error)}
                    rows.append(row)
                    completed.add((case.name, name, offset))
                    persist()
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {}
    for name in methods:
        summary[name] = {str(offset): {key: sum(r[key] for r in ok if r["baseline"] == name and r["offset_seconds"] == offset) /
            len([r for r in ok if r["baseline"] == name and r["offset_seconds"] == offset]) for key in ("ac_at_1", "ac_at_3", "ac_at_5", "mrr", "avg_at_5")}
            for offset in offsets}
    args.output.write_text(json.dumps({"offsets": offsets, "rows": rows, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
