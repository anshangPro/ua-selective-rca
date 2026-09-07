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
    """Evaluate the realized selective outputs without treating abstentions as errors.

    ``selective_risk`` is the conditional error rate among accepted cases.  The
    set metrics are deliberately reported separately: a set can contain a true
    cause while still being large or imprecise.
    """
    if len(decisions) != len(truths) or not decisions:
        raise ValueError("decisions and truths must be non-empty and aligned")
    if any(not truth for truth in truths):
        raise ValueError("ground truth cannot be empty")
    accepted = [(d, truth) for d, truth in zip(decisions, truths) if d.mode != "abstain"]
    coverage = len(accepted) / len(decisions)
    errors = [0.0 if set(d.causes) & truth else 1.0 for d, truth in accepted]
    set_hits = [1.0 - error for error in errors]
    truth_recalls = [len(set(d.causes) & truth) / len(truth) for d, truth in accepted]
    set_precisions = [len(set(d.causes) & truth) / len(d.causes) for d, truth in accepted]
    exact_matches = [float(set(d.causes) == truth) for d, truth in accepted]
    top1 = [(d, truth) for d, truth in accepted if d.mode == "top1"]
    return {
        "coverage": coverage,
        "selective_risk": sum(errors) / len(errors) if errors else 0.0,
        # Kept as a compatibility alias for existing result consumers.
        "topk_set_recall": sum(set_hits) / len(set_hits) if set_hits else 0.0,
        "set_hit_rate": sum(set_hits) / len(set_hits) if set_hits else 0.0,
        "mean_truth_recall": sum(truth_recalls) / len(truth_recalls) if truth_recalls else 0.0,
        "mean_set_precision": sum(set_precisions) / len(set_precisions) if set_precisions else 0.0,
        "exact_set_match_rate": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        "mean_set_size": sum(len(d.causes) for d, _ in accepted) / len(accepted) if accepted else 0.0,
        "abstention_rate": 1.0 - coverage,
        "top1_rate": len(top1) / len(decisions),
        "top1_accuracy": (sum(float(d.causes[0] in truth) for d, truth in top1) / len(top1) if top1 else 0.0),
        "topk_rate": sum(d.mode == "topk" for d, _ in accepted) / len(decisions),
    }


def risk_coverage_curve(decisions: Sequence[Decision], truths: Sequence[set[str]]) -> list[dict[str, float]]:
    """Return the attainable confidence-threshold risk--coverage curve.

    Explicit abstentions are not silently recast as wrong diagnoses.  Coverage
    is still normalized by every evaluated case, so the final point exposes the
    maximum coverage the policy actually attains.  Equal confidence values are
    accepted as one group to avoid an arbitrary tie-order changing the curve.
    """
    if len(decisions) != len(truths) or not decisions:
        raise ValueError("decisions and truths must be non-empty and aligned")
    if any(not truth for truth in truths):
        raise ValueError("ground truth cannot be empty")
    accepted = [(decision, truth) for decision, truth in zip(decisions, truths)
                if decision.mode != "abstain"]
    pairs = sorted(accepted, key=lambda pair: pair[0].confidence, reverse=True)
    points = [{"coverage": 0.0, "selective_risk": 0.0, "accepted": 0.0,
               "confidence_threshold": 1.0}]
    errors = 0
    index = 0
    while index < len(pairs):
        confidence = pairs[index][0].confidence
        group_end = index
        while group_end < len(pairs) and pairs[group_end][0].confidence == confidence:
            decision, truth = pairs[group_end]
            errors += int(not (set(decision.causes) & truth))
            group_end += 1
        accepted_count = group_end
        points.append({"coverage": accepted_count / len(decisions),
                       "selective_risk": errors / accepted_count,
                       "accepted": float(accepted_count),
                       "confidence_threshold": confidence})
        index = group_end
    return points


def aurc(decisions: Sequence[Decision], truths: Sequence[set[str]]) -> float:
    """Compute AURC over the attainable non-abstained coverage range.

    This is a right-step integral, equivalent to the standard mean-prefix-risk
    estimator when every case can be accepted.  Compare it with ``coverage``;
    an abstaining policy cannot claim unearned coverage by obtaining a small
    area over a short interval.
    """
    points = risk_coverage_curve(decisions, truths)
    return sum((point["coverage"] - previous["coverage"]) * point["selective_risk"]
               for previous, point in zip(points, points[1:]))


def normalized_aurc(decisions: Sequence[Decision], truths: Sequence[set[str]]) -> float:
    """AURC divided by attained coverage; returns zero when all cases abstain."""
    points = risk_coverage_curve(decisions, truths)
    maximum_coverage = points[-1]["coverage"]
    return aurc(decisions, truths) / maximum_coverage if maximum_coverage else 0.0


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
