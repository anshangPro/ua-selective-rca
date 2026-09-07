from __future__ import annotations

from collections.abc import Mapping, Sequence

from .aggregation import AGGREGATION_MODES, aggregate_ablations, aggregate_rankings, window_quality_scores
from .boundary import posterior_from_detectors
from .selective import select_diagnosis
from .types import CandidateWindow, Decision, SelectiveConfig


class SelectiveRCAPipeline:
    def __init__(
        self, config: SelectiveConfig = SelectiveConfig(), temperature: float = 1.0,
        detector_weights: Mapping[str, float] | None = None,
        aggregation_mode: str = "posterior_quality_stability", stability_k: int = 5,
    ):
        if aggregation_mode not in AGGREGATION_MODES:
            raise ValueError("unknown aggregation mode: %s" % aggregation_mode)
        if stability_k < 1:
            raise ValueError("stability_k must be positive")
        self.config = config
        self.temperature = temperature
        self.detector_weights = detector_weights
        self.aggregation_mode = aggregation_mode
        self.stability_k = stability_k

    def diagnose(self, windows: Sequence[CandidateWindow]) -> tuple[Decision, dict]:
        posterior = posterior_from_detectors([window.detector_scores for window in windows], self.detector_weights, self.temperature)
        aggregate, weights, stability = aggregate_rankings(
            windows, posterior, self.stability_k, self.aggregation_mode,
        )
        decision = select_diagnosis(aggregate, posterior, stability, self.config)
        audit = {
            "candidate_starts": [window.start for window in windows],
            "boundary_posterior": posterior,
            "aggregation_mode": self.aggregation_mode,
            "effective_window_weights": weights,
            "window_quality": window_quality_scores(windows),
            "window_stability": stability,
            "decision": decision.as_dict(),
        }
        return decision, audit

    def diagnose_ablations(self, windows: Sequence[CandidateWindow]) -> dict[str, dict]:
        """Return decisions and factor audits for every aggregation ablation.

        Every mode shares the same RCA window rankings and boundary posterior, so
        reported differences isolate only the aggregation factors.
        """
        posterior = posterior_from_detectors(
            [window.detector_scores for window in windows], self.detector_weights, self.temperature,
        )
        output: dict[str, dict] = {}
        for mode, result in aggregate_ablations(windows, posterior, self.stability_k).items():
            decision = select_diagnosis(
                result["aggregate_ranking"], posterior, result["window_stability"], self.config,
            )
            output[mode] = {**result, "boundary_posterior": posterior, "decision": decision.as_dict()}
        return output
