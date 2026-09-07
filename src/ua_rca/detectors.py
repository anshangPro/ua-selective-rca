from __future__ import annotations

import math
from collections.abc import Sequence


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / (len(values) - 1))


def change_point_scores(series: Sequence[float], candidates: Sequence[int], lookback: int = 12, lookahead: int = 12) -> dict[int, float]:
    """Absolute standardized mean shift around each candidate boundary."""
    scores: dict[int, float] = {}
    for index in candidates:
        before = series[max(0, index - lookback):index]
        after = series[index:min(len(series), index + lookahead)]
        if len(before) < 2 or len(after) < 2:
            scores[index] = -10.0
            continue
        pooled = max((_std(before) + _std(after)) / 2, 1e-8)
        scores[index] = abs(_mean(after) - _mean(before)) / pooled
    return scores


def ewma_residual_scores(series: Sequence[float], candidates: Sequence[int], alpha: float = 0.25, horizon: int = 5) -> dict[int, float]:
    """Scores post-boundary deviations from a pre-boundary EWMA forecast."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    scores: dict[int, float] = {}
    for index in candidates:
        history = series[:index]
        future = series[index:min(len(series), index + horizon)]
        if len(history) < 2 or not future:
            scores[index] = -10.0
            continue
        level = history[0]
        residuals = []
        for value in history[1:]:
            residuals.append(value - level)
            level = alpha * value + (1 - alpha) * level
        scale = max(_std(residuals), 1e-8)
        scores[index] = _mean([abs(value - level) / scale for value in future])
    return scores
