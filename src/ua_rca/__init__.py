"""Posterior-weighted selective RCA research prototype."""

from .types import CandidateWindow, Decision, SelectiveConfig
from .pipeline import SelectiveRCAPipeline
from .boundary import build_boundary_candidates

__all__ = ["CandidateWindow", "Decision", "SelectiveConfig", "SelectiveRCAPipeline", "build_boundary_candidates"]
