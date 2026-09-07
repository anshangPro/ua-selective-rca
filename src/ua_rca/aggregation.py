from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .adapters import normalize_ranking
from .types import CandidateWindow

AGGREGATION_MODES = (
    "single_boundary",
    "equal_window",
    "posterior",
    "posterior_quality",
    "posterior_stability",
    "posterior_quality_stability",
)


def _top_items(ranking: Mapping[str, float], k: int = 5) -> set[str]:
    return {name for name, _ in sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:k]}


def topk_jaccard_stability(rankings: Sequence[Mapping[str, float]], k: int = 5) -> list[float]:
    """Per-window stability, measured against all other windows' top-k sets."""
    if len(rankings) == 1:
        return [1.0]
    sets = [_top_items(ranking, k) for ranking in rankings]
    output = []
    for i, left in enumerate(sets):
        similarities = []
        for j, right in enumerate(sets):
            if i == j:
                continue
            union = left | right
            similarities.append(len(left & right) / len(union) if union else 1.0)
        output.append(sum(similarities) / len(similarities))
    return output


def window_quality_scores(windows: Sequence[CandidateWindow]) -> list[float]:
    """Return bounded, auditable evidence-quality factors for candidate windows.

    ``quality`` is deliberately an input to the uncertainty layer: an RCA adapter
    can derive it from anomaly strength, normal/abnormal sample sufficiency, and
    causal-graph availability without coupling this package to a particular RCA
    implementation.  Values outside [0, 1] are rejected rather than silently
    changing the interpretation of posterior weights.
    """
    quality = [float(window.quality) for window in windows]
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in quality):
        raise ValueError("window quality scores must be finite values in [0, 1]")
    return quality


def score_window_quality(
    anomaly_strength: float, normal_samples: int, abnormal_samples: int, graph_available: bool = True,
    target_anomaly_strength: float = 3.0, min_normal_samples: int = 20, min_abnormal_samples: int = 10,
) -> float:
    """Score a window's usable evidence in [0, 1] without using fault labels.

    Adapters should feed anomaly strength (for example, median robust-z residual),
    sample counts on the two sides of the candidate boundary, and whether their
    causal graph/statistical test was usable.  A geometric mean means one missing
    evidence type cannot be hidden by a strong score in another dimension.
    """
    if target_anomaly_strength <= 0 or min_normal_samples <= 0 or min_abnormal_samples <= 0:
        raise ValueError("quality score targets must be positive")
    if not math.isfinite(anomaly_strength) or normal_samples < 0 or abnormal_samples < 0:
        raise ValueError("window evidence must be finite and sample counts non-negative")
    factors = (
        min(max(anomaly_strength / target_anomaly_strength, 0.0), 1.0),
        min(normal_samples / min_normal_samples, 1.0),
        min(abnormal_samples / min_abnormal_samples, 1.0),
        1.0 if graph_available else 0.0,
    )
    return math.prod(factors) ** (1.0 / len(factors))


def _raw_window_weights(
    mode: str, posterior: Sequence[float], quality: Sequence[float], stability: Sequence[float]
) -> list[float]:
    if mode not in AGGREGATION_MODES:
        raise ValueError("unknown aggregation mode: %s" % mode)
    if mode == "single_boundary":
        selected = max(range(len(posterior)), key=lambda index: posterior[index])
        return [1.0 if index == selected else 0.0 for index in range(len(posterior))]
    if mode == "equal_window":
        return [1.0] * len(posterior)
    if mode == "posterior":
        return list(posterior)
    if mode == "posterior_quality":
        return [p * q for p, q in zip(posterior, quality)]
    if mode == "posterior_stability":
        return [p * s for p, s in zip(posterior, stability)]
    return [p * q * s for p, q, s in zip(posterior, quality, stability)]


def aggregate_rankings(
    windows: Sequence[CandidateWindow], posterior: Sequence[float], stability_k: int = 5,
    mode: str = "posterior_quality_stability",
) -> tuple[dict[str, float], list[float], list[float]]:
    """Aggregate rankings under a named ablation or the full weighting model.

    Modes isolate the contribution of posterior, evidence quality, and ranking
    stability.  This makes equal-window and single-boundary baselines run through
    exactly the same score normalization and ranking path as the full method.
    """
    if len(windows) != len(posterior) or not windows:
        raise ValueError("windows and posterior must be non-empty and aligned")
    if any(p < 0 for p in posterior) or not math.isclose(sum(posterior), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("posterior must sum to one")
    rankings = [normalize_ranking(window.ranking) for window in windows]
    stability = topk_jaccard_stability(rankings, stability_k)
    quality = window_quality_scores(windows)
    raw_weights = _raw_window_weights(mode, posterior, quality, stability)
    if sum(raw_weights) <= 0:
        # A quality/stability factor can be zero for every window.  Fall back to
        # posterior rather than producing an undefined ranking, and retain this
        # fact in the audit through the effective weights.
        raw_weights = list(posterior)
    weight_total = sum(raw_weights)
    weights = [weight / weight_total for weight in raw_weights]
    aggregate: dict[str, float] = {}
    for weight, ranking in zip(weights, rankings):
        for service, score in ranking.items():
            aggregate[service] = aggregate.get(service, 0.0) + weight * score
    return dict(sorted(aggregate.items(), key=lambda item: item[1], reverse=True)), weights, stability


def aggregate_ablations(
    windows: Sequence[CandidateWindow], posterior: Sequence[float], stability_k: int = 5
) -> dict[str, dict[str, object]]:
    """Run all aggregation ablations on identical windows and posterior input."""
    results: dict[str, dict[str, object]] = {}
    for mode in AGGREGATION_MODES:
        ranking, weights, stability = aggregate_rankings(windows, posterior, stability_k, mode)
        results[mode] = {
            "aggregate_ranking": ranking,
            "effective_window_weights": weights,
            "window_stability": stability,
            "window_quality": window_quality_scores(windows),
        }
    return results
