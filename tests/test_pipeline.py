import unittest

from ua_rca import CandidateWindow, SelectiveConfig, SelectiveRCAPipeline
from ua_rca.evaluation import selective_metrics


class PipelineTests(unittest.TestCase):
    def test_pipeline_returns_auditable_top1(self):
        windows = [
            CandidateWindow(0, {"cp": 1.0}, {"a": 0.9, "b": 0.1}),
            CandidateWindow(1, {"cp": 2.0}, {"a": 0.8, "b": 0.2}),
        ]
        decision, audit = SelectiveRCAPipeline(SelectiveConfig(top1_confidence=0.55, min_stability=0.2)).diagnose(windows)
        self.assertEqual(decision.mode, "top1")
        self.assertEqual(decision.causes, ("a",))
        self.assertIn("boundary_posterior", audit)

    def test_ambiguous_ranking_returns_topk(self):
        windows = [CandidateWindow(0, {"cp": 1.0}, {"a": 0.5, "b": 0.5})]
        decision, _ = SelectiveRCAPipeline().diagnose(windows)
        self.assertEqual(decision.mode, "topk")
        self.assertGreaterEqual(len(decision.causes), 1)

    def test_selective_metrics(self):
        windows = [CandidateWindow(0, {"cp": 1.0}, {"a": 0.9, "b": 0.1})]
        decision, _ = SelectiveRCAPipeline(SelectiveConfig(top1_confidence=0.5, min_stability=0.0)).diagnose(windows)
        metrics = selective_metrics([decision], [{"a"}])
        self.assertEqual(metrics["coverage"], 1.0)
        self.assertEqual(metrics["selective_risk"], 0.0)


if __name__ == "__main__":
    unittest.main()
