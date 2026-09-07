from __future__ import annotations

from collections.abc import Mapping


def normalize_ranking(scores: Mapping[str, float]) -> dict[str, float]:
    """Normalize arbitrary positive / signed external RCA scores to a distribution."""
    if not scores:
        raise ValueError("external RCA ranking cannot be empty")
    minimum = min(scores.values())
    shifted = {service: score - minimum + 1e-12 for service, score in scores.items()}
    total = sum(shifted.values())
    if total <= 0:
        raise ValueError("external RCA scores cannot be normalized")
    return {service: score / total for service, score in shifted.items()}


def ranking_from_ordered_services(services: list[str]) -> dict[str, float]:
    """Adapter for tools that emit an ordered list rather than confidence scores."""
    if not services:
        raise ValueError("ordered service list cannot be empty")
    n = len(services)
    return normalize_ranking({service: float(n - rank) for rank, service in enumerate(services)})
