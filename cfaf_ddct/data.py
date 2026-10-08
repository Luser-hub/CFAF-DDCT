from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def split_target(labels: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return approximately 64% adaptation, 16% validation, 20% test indices."""
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for cls in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == cls))
        if len(indices) < 6:
            raise ValueError("Synthetic data need at least six trials per class")
        n_test = max(1, round(0.20 * len(indices)))
        n_val = max(1, round(0.16 * len(indices)))
        n_train = len(indices) - n_val - n_test
        train.extend(indices[:n_train])
        validation.extend(indices[n_train:n_train + n_val])
        test.extend(indices[n_train + n_val:])
    return tuple(rng.permutation(part) for part in (train, validation, test))


def channel_score(x: np.ndarray) -> np.ndarray:
    scores = np.zeros(x.shape[1], dtype=np.float64)
    for trial in x:
        corr = np.nan_to_num(np.corrcoef(trial), nan=0.0)
        values = corr.mean(axis=1)
        scores += values > np.median(values)
    scores /= max(len(x), 1)
    return scores / max(scores.sum(), 1e-12)


def select_and_weight_channels(source_subjects, target_train, retained, alpha,
                               use_ct=True, use_cw=True):
    source_arrays = [item[0] if isinstance(item, (tuple, list)) else item
                     for item in source_subjects]
    if not source_arrays:
        raise ValueError("At least one source subject is required")
    n_channels = source_arrays[0].shape[1]
    source_score = np.mean([channel_score(item) for item in source_arrays], axis=0)
    target_score = channel_score(target_train)
    fused = (alpha * source_score + (1.0 - alpha) * target_score
             if (use_ct or use_cw) else np.ones(n_channels))
    indices = np.argsort(-fused, kind="stable")[:retained] if use_ct else np.arange(retained)
    weights = np.exp(fused[indices] - fused[indices].max()) if use_cw else np.ones(len(indices))
    if use_cw:
        weights /= weights.sum()
    return indices, weights.astype(np.float32)


def apply_channel_transform(x, indices, weights):
    return (x[:, indices, :] * weights[None, :, None]).astype(np.float32)


def loader(x, y, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)
