from __future__ import annotations

from collections.abc import Mapping, Sequence

from .aggregation import aggregate_rankings
from .boundary import posterior_from_detectors
from .selective import select_diagnosis
from .types import CandidateWindow, Decision, SelectiveConfig


class SelectiveRCAPipeline:
    def __init__(self, config: SelectiveConfig = SelectiveConfig(), temperature: float = 1.0, detector_weights: Mapping[str, float] | None = None):
        self.config = config
        self.temperature = temperature
        self.detector_weights = detector_weights

    def diagnose(self, windows: Sequence[CandidateWindow]) -> tuple[Decision, dict]:
        posterior = posterior_from_detectors([window.detector_scores for window in windows], self.detector_weights, self.temperature)
        aggregate, weights, stability = aggregate_rankings(windows, posterior)
        decision = select_diagnosis(aggregate, posterior, stability, self.config)
        audit = {
            "candidate_starts": [window.start for window in windows],
            "boundary_posterior": posterior,
            "effective_window_weights": weights,
            "window_stability": stability,
            "decision": decision.as_dict(),
        }
        return decision, audit
