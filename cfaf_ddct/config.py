from dataclasses import dataclass
from typing import Optional

LOSS_WEIGHTS = {"icl": 0.10, "mmd": 0.10, "lmmd": 0.01,
                "coral": 0.05, "adver": 0.03, "none": 0.0}


@dataclass
class Config:
    loss: str = "icl"
    fine_tune_epochs: int = 20
    source_pretrain_epochs: int = 5
    batch_size: int = 32
    seed: int = 42
    retained_channels: int = 12
    icl_margin: float = 1.0
    kernel_mul: float = 2.0
    kernel_num: int = 5
    source_target_ratio: float = 1.0
    normalize_transfer_features: bool = True
    lambda_override: Optional[float] = None
    channel_fusion_alpha: float = 0.5
    sampling_rate: float = 250.0
    channel_window_seconds: float = 3.0
    channel_step_seconds: float = 0.5
    hard_vote_threshold: str = "median"
    temporal_filters: int = 40
    temporal_kernel: int = 32
    pool_kernel: int = 75
    pool_stride: int = 15
    model_dropout: float = 0.5
    attention_heads: int = 4
    classifier_hidden: int = 256

    def validate(self):
        if self.loss not in LOSS_WEIGHTS:
            raise ValueError("loss must be icl, mmd, lmmd, coral, adver, or none")
        if self.fine_tune_epochs not in (200,20):
            raise ValueError("fine_tune_epochs must be 180 or 200")
        if self.source_pretrain_epochs < 1 or self.batch_size < 1:
            raise ValueError("Epoch and batch-size values must be positive")
        if not 1 <= self.retained_channels <= 60:
            raise ValueError("retained_channels must be in [1, 60]")
        if self.icl_margin <= 0 or self.kernel_mul <= 0 or self.kernel_num < 1:
            raise ValueError("Invalid adaptation-loss hyperparameter")
        if self.source_target_ratio <= 0:
            raise ValueError("source_target_ratio must be positive")
        if not 0.0 <= self.channel_fusion_alpha <= 1.0:
            raise ValueError("channel_fusion_alpha must be in [0, 1]")
        if self.sampling_rate <= 0 or self.channel_window_seconds <= 0:
            raise ValueError("Channel-scoring sampling rate and window must be positive")
        if self.channel_step_seconds <= 0:
            raise ValueError("channel_step_seconds must be positive")
        if self.hard_vote_threshold not in ("median", "mean"):
            raise ValueError("hard_vote_threshold must be 'median' or 'mean'")
        if self.temporal_filters < 1 or self.temporal_kernel < 1:
            raise ValueError("Temporal-convolution settings must be positive")
        if self.pool_kernel < 1 or self.pool_stride < 1:
            raise ValueError("Temporal-pooling settings must be positive")
        if not 0.0 <= self.model_dropout < 1.0:
            raise ValueError("model_dropout must be in [0, 1)")
        if self.attention_heads < 1 or self.temporal_filters % self.attention_heads:
            raise ValueError("temporal_filters must be divisible by attention_heads")
        if self.classifier_hidden < 1:
            raise ValueError("classifier_hidden must be positive")
        if self.loss == "none" and self.lambda_override not in (None, 0.0):
            raise ValueError("The classification-only baseline requires lambda=0")

    @property
    def lambda_value(self):
        if self.loss == "none":
            return 0.0
        return LOSS_WEIGHTS[self.loss] if self.lambda_override is None else self.lambda_override
