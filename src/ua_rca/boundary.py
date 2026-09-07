from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


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
