import unittest

from ua_rca import CandidateWindow, SelectiveRCAPipeline
from ua_rca.aggregation import AGGREGATION_MODES, aggregate_rankings, score_window_quality, window_quality_scores


class AggregationTests(unittest.TestCase):
    def setUp(self):
        # The first two windows agree while the third is a ranking outlier.
        self.windows = [
            CandidateWindow(0, {"d": 1.0}, {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}, 1.0),
            CandidateWindow(1, {"d": 1.0}, {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}, 0.2),
            CandidateWindow(2, {"d": 1.0}, {"x": 5, "y": 4, "z": 3, "u": 2, "v": 1}, 1.0),
        ]
        self.posterior = [0.5, 0.4, 0.1]

    def test_equal_and_posterior_modes_are_independent(self):
        _, equal_weights, _ = aggregate_rankings(self.windows, self.posterior, mode="equal_window")
        _, posterior_weights, _ = aggregate_rankings(self.windows, self.posterior, mode="posterior")
        self.assertEqual(equal_weights, [1 / 3, 1 / 3, 1 / 3])
        self.assertEqual(posterior_weights, self.posterior)

    def test_quality_and_stability_factors_change_only_their_ablation(self):
        _, posterior_weights, _ = aggregate_rankings(self.windows, self.posterior, mode="posterior")
        _, quality_weights, _ = aggregate_rankings(self.windows, self.posterior, mode="posterior_quality")
        _, stability_weights, stability = aggregate_rankings(
            self.windows, self.posterior, mode="posterior_stability",
        )
        self.assertEqual(window_quality_scores(self.windows), [1.0, 0.2, 1.0])
        self.assertLess(quality_weights[1], posterior_weights[1])
        self.assertLess(stability[2], stability[0])
        self.assertLess(stability_weights[2], posterior_weights[2])

    def test_all_ablations_have_auditable_outputs(self):
        output = SelectiveRCAPipeline().diagnose_ablations(self.windows)
        self.assertEqual(set(output), set(AGGREGATION_MODES))
        for mode, result in output.items():
            self.assertEqual(len(result["effective_window_weights"]), len(self.windows), mode)
            self.assertAlmostEqual(sum(result["effective_window_weights"]), 1.0)
            self.assertIn("decision", result)

    def test_invalid_quality_is_rejected(self):
        with self.assertRaises(ValueError):
            window_quality_scores([CandidateWindow(0, {"d": 1.0}, {"a": 1.0}, 1.1)])

    def test_window_evidence_quality_requires_all_evidence_dimensions(self):
        self.assertEqual(score_window_quality(3.0, 20, 10, True), 1.0)
        self.assertEqual(score_window_quality(3.0, 20, 10, False), 0.0)
        self.assertLess(score_window_quality(1.5, 20, 10, True), 1.0)


if __name__ == "__main__":
    unittest.main()
