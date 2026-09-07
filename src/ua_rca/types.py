from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class CandidateWindow:
    """A candidate fault-start boundary and the ranking produced for its window."""

    start: float
    detector_scores: Mapping[str, float]
    ranking: Mapping[str, float]
    quality: float = 1.0


@dataclass(frozen=True)
class SelectiveConfig:
    top_k_limit: int = 5
    top1_confidence: float = 0.60
    top1_margin: float = 0.15
    min_stability: float = 0.45
    max_boundary_entropy: float = 0.90
    set_mass: float = 0.80
    min_set_confidence: float = 0.45


@dataclass(frozen=True)
class Decision:
    mode: str  # top1, topk, abstain
    causes: tuple[str, ...]
    confidence: float
    reason: str | None
    aggregate_ranking: Mapping[str, float]
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        result = asdict(self)
        result["causes"] = list(self.causes)
        return result
