import unittest

import torch
from torch.nn import functional as F

from cfaf_ddct.model import EEGConformer


class CFAFModelTests(unittest.TestCase):
    def test_shapes_probabilities_and_backward(self):
        weights = torch.linspace(0.2, 0.8, 12)
        model = EEGConformer(12, 128, channel_weights=weights, dropout=0.0)
        x = torch.randn(2, 12, 128, requires_grad=True)
        intermediate = model(x, return_intermediates=True)

        self.assertEqual(intermediate["input_4d"].shape, (2, 1, 12, 128))
        self.assertEqual(intermediate["temporal"].shape, (2, 40, 12, 97))
        self.assertEqual(intermediate["broadcast_weights"].shape, (1, 1, 12, 1))
        self.assertEqual(intermediate["encoder1"].shape, (2, 40, 12, 2))
        self.assertEqual(intermediate["pointwise"].shape, (2, 40, 12, 2))
        self.assertEqual(intermediate["sequence"].shape, (2, 24, 40))
        self.assertEqual(intermediate["residual"].shape,
                         intermediate["attention"].shape)
        self.assertEqual(intermediate["fused"].shape, (2, 24, 40))
        self.assertEqual(intermediate["features"].shape, (2, 256))
        self.assertEqual(intermediate["logits"].shape, (2, 3))
        self.assertEqual(model.attention.num_heads, 4)
        torch.testing.assert_close(
            intermediate["probabilities"].sum(dim=1), torch.ones(2)
        )
        expected_weighted = (
            intermediate["temporal"]
            * weights.view(1, 1, 12, 1)
        )
        torch.testing.assert_close(intermediate["weighted"], expected_weighted)

        loss = F.cross_entropy(intermediate["logits"], torch.tensor([0, 2]))
        loss.backward()
        self.assertIsNotNone(model.channel_weights.grad)
        self.assertTrue(torch.isfinite(model.channel_weights.grad).all())


if __name__ == "__main__":
    unittest.main()
