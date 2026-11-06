import unittest

import numpy as np

from cfaf_ddct.train import _select_source_for_adaptation


class SourceConfidenceSelectionTests(unittest.TestCase):
    def test_max_softmax_confidence_controls_global_ranking(self):
        source_x = np.arange(6 * 2 * 3).reshape(6, 2, 3)
        source_y = np.array([2, 0, 1, 2, 1, 0])
        probabilities = np.array([
            [0.34, 0.33, 0.33],
            [0.80, 0.10, 0.10],
            [0.20, 0.20, 0.60],
            [0.05, 0.05, 0.90],
            [0.40, 0.35, 0.25],
            [0.10, 0.70, 0.20],
        ])
        selected_x, selected_y, indices, confidence = \
            _select_source_for_adaptation(
                source_x, source_y, probabilities, target_count=2, ratio=1.5
            )
        np.testing.assert_array_equal(indices, [3, 1, 5])
        np.testing.assert_array_equal(selected_x, source_x[indices])
        np.testing.assert_array_equal(selected_y, source_y[indices])
        np.testing.assert_allclose(confidence, probabilities.max(axis=1))

    def test_requested_count_is_capped_by_available_source_samples(self):
        source_x = np.zeros((4, 2, 3))
        source_y = np.arange(4)
        probabilities = np.eye(4)
        selected_x, _, indices, _ = _select_source_for_adaptation(
            source_x, source_y, probabilities, target_count=10, ratio=2
        )
        self.assertEqual(len(selected_x), 4)
        self.assertEqual(len(indices), 4)


if __name__ == "__main__":
    unittest.main()
