from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .types import Decision


def ranking_metrics(ranking: Sequence[str], truth: set[str]) -> dict[str, float]:
    if not truth:
        raise ValueError("ground truth cannot be empty")
    hits = [int(name in truth) for name in ranking]
    first = next((index + 1 for index, hit in enumerate(hits) if hit), None)
    return {
        "ac_at_1": float(any(hits[:1])), "ac_at_3": float(any(hits[:3])), "ac_at_5": float(any(hits[:5])),
        "mrr": 1.0 / first if first else 0.0,
        "avg_at_5": sum(hits[:5]) / min(5, len(ranking)),
    }


def aggregate_metric_rows(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def selective_metrics(decisions: Sequence[Decision], truths: Sequence[set[str]]) -> dict[str, float]:
    if len(decisions) != len(truths) or not decisions:
        raise ValueError("decisions and truths must be non-empty and aligned")
    accepted = [(d, truth) for d, truth in zip(decisions, truths) if d.mode != "abstain"]
    coverage = len(accepted) / len(decisions)
    errors = [0.0 if set(d.causes) & truth else 1.0 for d, truth in accepted]
    set_recall = sum(1.0 - error for error in errors) / len(errors) if errors else 0.0
    return {
        "coverage": coverage,
        "selective_risk": sum(errors) / len(errors) if errors else 0.0,
        "topk_set_recall": set_recall,
        "mean_set_size": sum(len(d.causes) for d, _ in accepted) / len(accepted) if accepted else 0.0,
        "abstention_rate": 1.0 - coverage,
    }


def aurc(decisions: Sequence[Decision], truths: Sequence[set[str]]) -> float:
    """Area under risk-coverage curve after sorting by decision confidence."""
    pairs = sorted(zip(decisions, truths), key=lambda pair: pair[0].confidence, reverse=True)
    risks = []
    errors = 0
    for index, (decision, truth) in enumerate(pairs, start=1):
        errors += int(not (set(decision.causes) & truth))
        risks.append(errors / index)
    return sum(risks) / len(risks) if risks else 0.0


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[int], bins: int = 10) -> float:
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidence and correctness vectors must be non-empty and aligned")
    error = 0.0
    for bucket in range(bins):
        lower, upper = bucket / bins, (bucket + 1) / bins
        indices = [i for i, value in enumerate(confidences) if lower <= value < upper or (bucket == bins - 1 and value == 1.0)]
        if indices:
            accuracy = sum(correct[i] for i in indices) / len(indices)
            mean_confidence = sum(confidences[i] for i in indices) / len(indices)
            error += len(indices) / len(confidences) * abs(accuracy - mean_confidence)
    return error
