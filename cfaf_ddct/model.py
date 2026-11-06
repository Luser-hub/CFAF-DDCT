"""Channel-Feature Attention Fusion network."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn


class EEGConformer(nn.Module):
    """CFAF classifier accepting ``[batch, channels, samples]`` inputs."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 3,
        width: int = 40,
        use_residual_attention: bool = True,
        channel_weights: Optional[torch.Tensor] = None,
        temporal_kernel: int = 32,
        pool_kernel: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
        attention_heads: int = 4,
        classifier_hidden: int = 256,
    ):
        super().__init__()
        if n_channels < 1 or n_classes < 2:
            raise ValueError("n_channels must be positive and n_classes >= 2")
        if n_times - temporal_kernel + 1 < pool_kernel:
            raise ValueError(
                "Input is too short for temporal convolution and 1x75 pooling"
            )
        if width != 40 or temporal_kernel != 32 or pool_kernel != 75:
            raise ValueError(
                "CFAF requires 40 filters, a 1x32 temporal "
                "kernel, and 1x75 temporal pooling"
            )
        if attention_heads != 4 or width % attention_heads:
            raise ValueError("CFAF requires four compatible attention heads")

        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)
        self.feature_dim = int(classifier_hidden)
        self.use_residual_attention = bool(use_residual_attention)

        initial_weights = (
            torch.ones(n_channels, dtype=torch.float32)
            if channel_weights is None
            else torch.as_tensor(channel_weights, dtype=torch.float32).flatten()
        )
        if initial_weights.numel() != n_channels:
            raise ValueError("channel_weights must contain one value per channel")
        self.channel_weights = nn.Parameter(initial_weights.clone())

        self.temporal_convolution = nn.Conv2d(
            1, width, kernel_size=(1, temporal_kernel), stride=(1, 1), bias=False
        )
        self.batch_norm = nn.BatchNorm2d(width)
        self.elu = nn.ELU()
        self.temporal_pool = nn.AvgPool2d(
            kernel_size=(1, pool_kernel), stride=(1, pool_stride)
        )
        self.dropout = nn.Dropout(dropout)
        self.pointwise_convolution = nn.Conv2d(
            width, width, kernel_size=(1, 1), stride=(1, 1), bias=False
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(width)

        convolved_times = n_times - temporal_kernel + 1
        pooled_times = (convolved_times - pool_kernel) // pool_stride + 1
        self.sequence_length = n_channels * pooled_times
        self.fc1 = nn.Linear(self.sequence_length * width, classifier_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(classifier_hidden, n_classes)

    def _reshape_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, features, channels, times = tensor.shape
        return tensor.permute(0, 2, 3, 1).reshape(
            batch, channels * times, features
        )

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError("Expected input shape [batch, channels, samples]")
        if x.shape[1] != self.n_channels or x.shape[2] != self.n_times:
            raise ValueError(
                f"Expected [batch, {self.n_channels}, {self.n_times}], got {tuple(x.shape)}"
            )

        x4d = x.unsqueeze(1)
        temporal = self.temporal_convolution(x4d)
        broadcast_weights = self.channel_weights.view(1, 1, self.n_channels, 1)
        weighted = temporal * broadcast_weights
        encoder1 = self.dropout(
            self.temporal_pool(self.elu(self.batch_norm(weighted)))
        )
        pointwise = self.pointwise_convolution(encoder1)
        sequence = self._reshape_sequence(pointwise)
        residual = self._reshape_sequence(encoder1)
        attended, _ = self.attention(
            sequence, sequence, sequence, need_weights=False
        )
        fused = self.layer_norm(
            attended + residual if self.use_residual_attention else attended
        )
        flattened = fused.flatten(start_dim=1)
        features = self.relu(self.fc1(flattened))
        logits = self.fc2(features)

        if return_intermediates:
            return {
                "input_4d": x4d,
                "temporal": temporal,
                "broadcast_weights": broadcast_weights,
                "weighted": weighted,
                "encoder1": encoder1,
                "pointwise": pointwise,
                "sequence": sequence,
                "residual": residual,
                "attention": attended,
                "fused": fused,
                "features": features,
                "logits": logits,
                "probabilities": torch.softmax(logits, dim=1),
            }
        return logits, features

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x)[0], dim=1)
