import unittest

from ua_rca.boundary import fit_temperature_grid, normalized_entropy, posterior_from_detectors


class BoundaryTests(unittest.TestCase):
    def test_posterior_is_normalized_and_prefers_strong_evidence(self):
        posterior = posterior_from_detectors([{"a": 1.0, "b": 0.5}, {"a": 3.0, "b": 2.0}])
        self.assertAlmostEqual(sum(posterior), 1.0)
        self.assertGreater(posterior[1], posterior[0])

    def test_entropy_and_temperature(self):
        self.assertAlmostEqual(normalized_entropy([1.0]), 0.0)
        temperature = fit_temperature_grid([[0.0, 2.0], [2.0, 0.0]], [1, 0], [0.5, 1.0, 2.0])
        self.assertEqual(temperature, 0.5)


if __name__ == "__main__":
    unittest.main()
