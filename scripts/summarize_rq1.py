"""Create the final, reproducible RQ1 offset-robustness summary."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

METRICS = ("ac_at_1", "ac_at_3", "ac_at_5", "mrr", "avg_at_5")


def mean(rows, metric):
    return sum(row[metric] for row in rows) / len(rows)


def offset_auc(points):
    """Trapezoid area normalized to [-300, 300], so result remains in [0, 1]."""
    ordered = sorted(points.items())
    area = sum((right[0] - left[0]) * (left[1] + right[1]) / 2
               for left, right in zip(ordered, ordered[1:]))
    return area / (ordered[-1][0] - ordered[0][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())["rows"]
    valid = [row for row in rows if row["status"] == "ok"]
    failed = [row for row in rows if row["status"] != "ok"]
    by_baseline_offset = defaultdict(list)
    by_system_baseline_offset = defaultdict(list)
    for row in valid:
        by_baseline_offset[(row["baseline"], row["offset_seconds"])].append(row)
        by_system_baseline_offset[(row["system"], row["baseline"], row["offset_seconds"])].append(row)
    offset_table = []
    for (baseline, offset), group in sorted(by_baseline_offset.items()):
        offset_table.append({"baseline": baseline, "offset_seconds": offset, "n": len(group),
                             **{metric: mean(group, metric) for metric in METRICS}})
    robustness = []
    for baseline in sorted({row["baseline"] for row in valid}):
        baseline_rows = [row for row in offset_table if row["baseline"] == baseline]
        item = {"baseline": baseline}
        for metric in METRICS:
            values = {row["offset_seconds"]: row[metric] for row in baseline_rows}
            average = sum(values.values()) / len(values)
            item[metric + "_mean_offsets"] = average
            item[metric + "_worst_offset"] = min(values.values())
            item[metric + "_variance_offsets"] = sum((value - average) ** 2 for value in values.values()) / len(values)
            item[metric + "_offset_auc"] = offset_auc(values)
        robustness.append(item)
    extremes = []
    for system, baseline, offset in sorted(by_system_baseline_offset):
        if offset in (-300, 0, 300):
            group = by_system_baseline_offset[(system, baseline, offset)]
            extremes.append({"system": system, "baseline": baseline, "offset_seconds": offset, "n": len(group),
                             **{metric: mean(group, metric) for metric in METRICS}})
    failure_counts = defaultdict(int)
    for row in failed:
        failure_counts[(row["system"], row["baseline"], row.get("error", "unknown"))] += 1
    result = {"planned_calls": len(rows), "valid_calls": len(valid), "failed_calls": len(failed),
              "offset_table": offset_table, "robustness": robustness, "extreme_offsets_by_system": extremes,
              "failures": [{"system": system, "baseline": baseline, "error": error, "count": count}
                           for (system, baseline, error), count in sorted(failure_counts.items())]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
