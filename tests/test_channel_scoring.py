import unittest

import numpy as np

from cfaf_ddct.data import (
    channel_score,
    hard_vote_window_scores,
    sample_window_channel_weights,
    select_and_weight_channels,
    sliding_window_bounds,
    window_channel_weights,
)


class ChannelScoringTests(unittest.TestCase):
    def test_sliding_window_boundaries(self):
        self.assertEqual(
            sliding_window_bounds(10, window_samples=4, step_samples=3),
            [(0, 4), (3, 7), (6, 10)],
        )
        self.assertEqual(sliding_window_bounds(10, 20, 3), [(0, 10)])

    def test_absolute_pearson_treats_negative_correlation_as_strong(self):
        x = np.arange(8, dtype=np.float64)
        window = np.stack([x, -x, np.array([0, 1, 0, 1, 0, 1, 0, 1])])
        weights = window_channel_weights(window)
        correlation = np.abs(np.corrcoef(window))
        expected = correlation.mean(axis=1)
        np.testing.assert_allclose(weights, expected, atol=1e-12)
        self.assertGreater(weights[0], 2.0 / 3.0)

    def test_each_window_is_kept_and_hard_voted_before_aggregation(self):
        trial = np.arange(3 * 10, dtype=np.float64).reshape(3, 10)
        window_weights = sample_window_channel_weights(trial, 4, 3)
        self.assertEqual(window_weights.shape, (3, 3))
        continuous = np.array([[0.1, 0.5, 0.9], [0.8, 0.7, 0.2]])
        score, votes = hard_vote_window_scores(continuous, "median")
        np.testing.assert_array_equal(votes, [[0, 0, 1], [1, 0, 0]])
        np.testing.assert_allclose(score, [0.5, 0.0, 0.5])
        self.assertFalse(np.allclose(score, continuous.mean(axis=0)))

    def test_subject_group_target_fusion_and_index_alignment(self):
        rng = np.random.default_rng(4)
        source = [(rng.normal(size=(3, 4, 20)).astype(np.float32), np.arange(3))
                  for _ in range(2)]
        target = rng.normal(size=(3, 4, 20)).astype(np.float32)
        indices, weights, details = select_and_weight_channels(
            source, target, retained=2, alpha=0.4,
            window_samples=8, step_samples=4, return_details=True,
        )
        self.assertEqual(details["source_subject_scores"].shape, (2, 4))
        expected = (0.4 * details["source_group_score"]
                    + 0.6 * details["target_subject_score"])
        np.testing.assert_allclose(details["fused_score"], expected)
        expected_indices = np.argsort(-expected, kind="stable")[:2]
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_allclose(weights, expected[expected_indices])
        self.assertEqual(channel_score(target, 8, 4).shape, (4,))


if __name__ == "__main__":
    unittest.main()
