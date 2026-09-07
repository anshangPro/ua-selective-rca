from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .boundary import normalized_entropy
from .types import Decision, SelectiveConfig


def _entropy(scores: Sequence[float]) -> float:
    total = sum(scores)
    return normalized_entropy([score / total for score in scores]) if total else 1.0


def select_diagnosis(
    aggregate_ranking: Mapping[str, float],
    boundary_posterior: Sequence[float],
    window_stability: Sequence[float],
    config: SelectiveConfig = SelectiveConfig(),
) -> Decision:
    if not aggregate_ranking:
        raise ValueError("aggregate ranking cannot be empty")
    ordered = sorted(aggregate_ranking.items(), key=lambda item: item[1], reverse=True)
    services, scores = zip(*ordered)
    total = sum(scores)
    probs = [score / total for score in scores]
    confidence = probs[0]
    margin = confidence - (probs[1] if len(probs) > 1 else 0.0)
    stability = sum(window_stability) / len(window_stability) if window_stability else 0.0
    boundary_entropy = normalized_entropy(boundary_posterior)
    ranking_entropy = _entropy(scores)
    diagnostics = {
        "top1_probability": confidence,
        "margin": margin,
        "mean_window_stability": stability,
        "boundary_entropy": boundary_entropy,
        "ranking_entropy": ranking_entropy,
    }
    if (confidence >= config.top1_confidence and margin >= config.top1_margin and stability >= config.min_stability
            and boundary_entropy <= config.max_boundary_entropy):
        return Decision("top1", (services[0],), confidence, None, aggregate_ranking, diagnostics)
    cumulative, chosen = 0.0, []
    for service, probability in zip(services[:config.top_k_limit], probs[:config.top_k_limit]):
        chosen.append(service)
        cumulative += probability
        if cumulative >= config.set_mass:
            break
    if cumulative >= config.min_set_confidence:
        reason = "boundary_or_ranking_ambiguity"
        return Decision("topk", tuple(chosen), cumulative, reason, aggregate_ranking, diagnostics)
    return Decision("abstain", tuple(), confidence, "insufficient_rank_concentration", aggregate_ranking, diagnostics)
