from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .detectors import change_point_scores, ewma_residual_scores


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not values:
        raise ValueError("cannot normalize empty scores")
    scaled = [v / temperature for v in values]
    maximum = max(scaled)
    exps = [math.exp(v - maximum) for v in scaled]
    total = sum(exps)
    return [v / total for v in exps]


def posterior_from_detectors(
    detector_scores: Sequence[Mapping[str, float]],
    detector_weights: Mapping[str, float] | None = None,
    temperature: float = 1.0,
) -> list[float]:
    """Fuse detector evidence and return p(boundary | telemetry, detectors)."""
    if not detector_scores:
        raise ValueError("at least one candidate is required")
    names = sorted({name for row in detector_scores for name in row})
    if not names:
        raise ValueError("candidate detector scores are empty")
    weights = detector_weights or {}
    fused = [sum(row.get(name, 0.0) * weights.get(name, 1.0) for name in names) for row in detector_scores]
    return softmax(fused, temperature)


def normalized_entropy(probabilities: Sequence[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return entropy / math.log(len(probabilities))


def build_boundary_candidates(
    series: Sequence[float],
    timestamps: Sequence[float] | None = None,
    candidate_limit: int = 5,
    lookback: int = 12,
    lookahead: int = 12,
    min_separation: int | None = None,
    detector_weights: Mapping[str, float] | None = None,
    temperature: float = 1.0,
) -> dict:
    """Create a diverse candidate-onset set and calibrated boundary posterior.

    The returned candidates retain each detector's raw evidence; ``posterior`` is
    a probability distribution over candidates, not a forced single alarm.
    """
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    if len(series) < lookback + lookahead + 1:
        raise ValueError("series is too short for the requested windows")
    if timestamps is not None and len(timestamps) != len(series):
        raise ValueError("timestamps must align with series")
    indices = list(range(lookback, len(series) - lookahead))
    cp = change_point_scores(series, indices, lookback, lookahead)
    residual = ewma_residual_scores(series, indices)
    weights = detector_weights or {}

    def scale(scores: Mapping[int, float]) -> dict[int, float]:
        low, high = min(scores.values()), max(scores.values())
        width = high - low
        return {index: (value - low) / width if width else 0.0 for index, value in scores.items()}

    cp_scaled, residual_scaled = scale(cp), scale(residual)
    fused = {index: cp_scaled[index] * weights.get("changepoint", 1.0) + residual_scaled[index] * weights.get("residual", 1.0) for index in indices}
    separation = min_separation if min_separation is not None else max(1, lookback)
    selected = []
    for index in sorted(indices, key=lambda item: fused[item], reverse=True):
        if all(abs(index - chosen) >= separation for chosen in selected):
            selected.append(index)
        if len(selected) == candidate_limit:
            break
    rows = [{"index": index, "timestamp": timestamps[index] if timestamps is not None else index,
             "detector_scores": {"changepoint": cp_scaled[index], "residual": residual_scaled[index]},
             "raw_detector_scores": {"changepoint": cp[index], "residual": residual[index]}}
            for index in selected]
    posterior = posterior_from_detectors([row["detector_scores"] for row in rows], detector_weights, temperature)
    for row, probability in zip(rows, posterior):
        row["probability"] = probability
    return {"candidates": rows, "posterior": posterior, "entropy": normalized_entropy(posterior),
            "top_confidence": max(posterior), "temperature": temperature}


def fit_temperature_grid(logits: Sequence[Sequence[float]], labels: Sequence[int], grid: Sequence[float] | None = None) -> float:
    """Choose temperature by minimum NLL on a validation set; never fit on test cases."""
    if len(logits) != len(labels) or not logits:
        raise ValueError("validation logits and labels must be non-empty and aligned")
    candidates = grid or tuple(0.25 + 0.25 * i for i in range(16))
    best_t, best_loss = candidates[0], float("inf")
    for temp in candidates:
        loss = 0.0
        for row, label in zip(logits, labels):
            probs = softmax(row, temp)
            if label < 0 or label >= len(probs):
                raise ValueError("label is outside candidate range")
            loss -= math.log(max(probs[label], 1e-12))
        if loss < best_loss:
            best_t, best_loss = temp, loss
    return float(best_t)
