from dataclasses import dataclass
from typing import Optional

LOSS_WEIGHTS = {"icl": 0.10, "mmd": 0.10, "lmmd": 0.01,
                "coral": 0.05, "adver": 0.03, "none": 0.0}


@dataclass
class Config:
    loss: str = "icl"
    fine_tune_epochs: int = 200
    source_pretrain_epochs: int = 70
    batch_size: int = 32
    seed: int = 42
    retained_channels: int = 12
    icl_margin: float = 1.0
    kernel_mul: float = 2.0
    kernel_num: int = 5
    source_target_ratio: float = 1.0
    normalize_transfer_features: bool = True
    lambda_override: Optional[float] = None

    def validate(self):
        if self.loss not in LOSS_WEIGHTS:
            raise ValueError("loss must be icl, mmd, lmmd, coral, adver, or none")
        if self.fine_tune_epochs not in (180, 200):
            raise ValueError("fine_tune_epochs must be 180 or 200")
        if self.source_pretrain_epochs < 1 or self.batch_size < 1:
            raise ValueError("Epoch and batch-size values must be positive")
        if not 1 <= self.retained_channels <= 60:
            raise ValueError("retained_channels must be in [1, 60]")
        if self.icl_margin <= 0 or self.kernel_mul <= 0 or self.kernel_num < 1:
            raise ValueError("Invalid adaptation-loss hyperparameter")
        if self.source_target_ratio <= 0:
            raise ValueError("source_target_ratio must be positive")
        if self.loss == "none" and self.lambda_override not in (None, 0.0):
            raise ValueError("The classification-only baseline requires lambda=0")

    @property
    def lambda_value(self):
        if self.loss == "none":
            return 0.0
        return LOSS_WEIGHTS[self.loss] if self.lambda_override is None else self.lambda_override
