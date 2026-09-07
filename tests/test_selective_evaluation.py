import unittest

from ua_rca.evaluation import aurc, normalized_aurc, risk_coverage_curve, selective_metrics
from ua_rca.types import Decision


def decision(mode, causes, confidence):
    return Decision(mode, tuple(causes), confidence, None, {}, {})


class SelectiveEvaluationTests(unittest.TestCase):
    def test_curve_excludes_explicit_abstentions_but_coverage_counts_all_cases(self):
        decisions = [
            decision("top1", ["a"], 0.9),
            decision("topk", ["b", "c"], 0.7),
            decision("abstain", [], 0.99),
        ]
        curve = risk_coverage_curve(decisions, [{"a"}, {"x"}, {"z"}])
        self.assertEqual([point["coverage"] for point in curve], [0.0, 1 / 3, 2 / 3])
        self.assertEqual([point["selective_risk"] for point in curve], [0.0, 0.0, 0.5])
        self.assertAlmostEqual(aurc(decisions, [{"a"}, {"x"}, {"z"}]), 1 / 6)
        self.assertAlmostEqual(normalized_aurc(decisions, [{"a"}, {"x"}, {"z"}]), 0.25)

    def test_equal_confidences_are_a_single_threshold_group(self):
        decisions = [decision("top1", ["a"], 0.8), decision("top1", ["b"], 0.8)]
        curve = risk_coverage_curve(decisions, [{"a"}, {"x"}])
        self.assertEqual(len(curve), 2)
        self.assertEqual(curve[-1]["coverage"], 1.0)
        self.assertEqual(curve[-1]["selective_risk"], 0.5)

    def test_set_diagnostics_report_partial_recall_and_precision(self):
        decisions = [decision("topk", ["a", "noise"], 0.7), decision("abstain", [], 0.1)]
        metrics = selective_metrics(decisions, [{"a", "b"}, {"x"}])
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["set_hit_rate"], 1.0)
        self.assertEqual(metrics["mean_truth_recall"], 0.5)
        self.assertEqual(metrics["mean_set_precision"], 0.5)
        self.assertEqual(metrics["mean_set_size"], 2.0)
        self.assertEqual(metrics["topk_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
