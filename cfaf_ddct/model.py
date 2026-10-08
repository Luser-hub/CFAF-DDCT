"""Compact EEG feature extractor used by the synthetic runnable example."""

from __future__ import annotations

import torch
from torch import nn


class EEGConformer(nn.Module):
    """Small convolutional/Transformer classifier.

    The input is ``[batch, channels, samples]``.  The convolutional stem
    represents the temporal and channel-mixing stages; the Transformer and
    residual attention path provide the FT/RA part of the reference model.
    """

    def __init__(self, n_channels: int, n_times: int, n_classes: int = 3,
                 width: int = 32, use_residual_attention: bool = True):
        super().__init__()
        if n_channels < 1 or n_times < 16:
            raise ValueError("n_channels must be positive and n_times >= 16")
        self.use_residual_attention = bool(use_residual_attention)
        temporal = min(15, max(3, n_times // 8 * 2 + 1))
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=(1, temporal), padding=(0, temporal // 2), bias=False),
            nn.BatchNorm2d(width),
            nn.ELU(),
            nn.Conv2d(width, width, kernel_size=(n_channels, 1), groups=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(0.1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=width * 2,
            dropout=0.1, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.residual_attention = nn.MultiheadAttention(
            embed_dim=width, num_heads=4, dropout=0.1, batch_first=True
        )
        self.feature_norm = nn.LayerNorm(width)
        self.classifier = nn.Sequential(
            nn.Linear(width, width), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(width, n_classes),
        )
        self.feature_dim = width

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError("Expected input shape [batch, channels, samples]")
        tokens = self.stem(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        if self.use_residual_attention:
            attended, _ = self.residual_attention(tokens, tokens, tokens, need_weights=False)
            tokens = tokens + attended
        feature = self.feature_norm(tokens.mean(dim=1))
        return self.classifier(feature), feature
