from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .adapters import normalize_ranking
from .types import CandidateWindow


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


def aggregate_rankings(
    windows: Sequence[CandidateWindow], posterior: Sequence[float], stability_k: int = 5
) -> tuple[dict[str, float], list[float], list[float]]:
    """Aggregate normalized rankings by posterior × quality × cross-window stability."""
    if len(windows) != len(posterior) or not windows:
        raise ValueError("windows and posterior must be non-empty and aligned")
    if any(p < 0 for p in posterior) or not math.isclose(sum(posterior), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("posterior must sum to one")
    rankings = [normalize_ranking(window.ranking) for window in windows]
    stability = topk_jaccard_stability(rankings, stability_k)
    raw_weights = [p * max(window.quality, 0.0) * s for p, window, s in zip(posterior, windows, stability)]
    if sum(raw_weights) <= 0:
        raw_weights = list(posterior)
    weight_total = sum(raw_weights)
    weights = [weight / weight_total for weight in raw_weights]
    aggregate: dict[str, float] = {}
    for weight, ranking in zip(weights, rankings):
        for service, score in ranking.items():
            aggregate[service] = aggregate.get(service, 0.0) + weight * score
    return dict(sorted(aggregate.items(), key=lambda item: item[1], reverse=True)), weights, stability
