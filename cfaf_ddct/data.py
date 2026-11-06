from typing import List, Tuple, Union

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


def sliding_window_bounds(n_times: int, window_samples: int,
                          step_samples: int) -> List[Tuple[int, int]]:
    """Return complete sliding-window boundaries."""
    if n_times < 2 or window_samples < 2 or step_samples < 1:
        raise ValueError("Need n_times/window_samples >= 2 and step_samples >= 1")
    if window_samples >= n_times:
        return [(0, n_times)]
    return [(start, start + window_samples)
            for start in range(0, n_times - window_samples + 1, step_samples)]


def window_channel_weights(window: np.ndarray) -> np.ndarray:
    """Calculate row means of the absolute Pearson correlation matrix."""
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2 or window.shape[1] < 2:
        raise ValueError("A window must have shape [channels, time>=2]")
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(window)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    return np.abs(correlation).mean(axis=1)


def sample_window_channel_weights(trial: np.ndarray, window_samples: int,
                                  step_samples: int) -> np.ndarray:
    """Return one continuous channel-weight vector for every time window."""
    trial = np.asarray(trial)
    if trial.ndim != 2:
        raise ValueError("A trial must have shape [channels, time]")
    bounds = sliding_window_bounds(trial.shape[1], window_samples, step_samples)
    return np.stack([window_channel_weights(trial[:, start:end])
                     for start, end in bounds], axis=0)


def hard_vote_window_scores(
    window_weights: np.ndarray,
    threshold: Union[str, float] = "median",
) -> Tuple[np.ndarray, np.ndarray]:
    """Binarize each window independently and return vote proportions."""
    weights = np.asarray(window_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] == 0:
        raise ValueError("window_weights must have shape [windows, channels]")
    if threshold == "median":
        thresholds = np.median(weights, axis=1, keepdims=True)
    elif threshold == "mean":
        thresholds = np.mean(weights, axis=1, keepdims=True)
    elif np.isscalar(threshold) and not isinstance(threshold, str):
        thresholds = np.full((weights.shape[0], 1), float(threshold))
    else:
        raise ValueError("threshold must be 'median', 'mean', or a number")
    votes = (weights > thresholds).astype(np.float64)
    return votes.mean(axis=0), votes


def sample_channel_score(trial: np.ndarray, window_samples: int,
                         step_samples: int,
                         threshold: Union[str, float] = "median") -> np.ndarray:
    window_weights = sample_window_channel_weights(
        trial, window_samples, step_samples
    )
    score, _ = hard_vote_window_scores(window_weights, threshold)
    return score


def channel_score(x: np.ndarray, window_samples: int, step_samples: int,
                  threshold: Union[str, float] = "median") -> np.ndarray:
    """Average hard-vote sample scores to obtain one subject score."""
    x = np.asarray(x)
    if x.ndim != 3 or len(x) == 0:
        raise ValueError("x must have shape [trials, channels, time]")
    return np.mean([
        sample_channel_score(trial, window_samples, step_samples, threshold)
        for trial in x
    ], axis=0)


def select_and_weight_channels(source_subjects, target_train, retained, alpha,
                               use_ct=True, use_cw=True, window_samples=None,
                               step_samples=None, threshold="median",
                               return_details=False):
    source_arrays = [item[0] if isinstance(item, (tuple, list)) else item
                     for item in source_subjects]
    if not source_arrays:
        raise ValueError("At least one source subject is required")
    n_channels = source_arrays[0].shape[1]
    n_times = source_arrays[0].shape[2]
    window_samples = n_times if window_samples is None else int(window_samples)
    step_samples = window_samples if step_samples is None else int(step_samples)
    subject_scores = [
        channel_score(item, window_samples, step_samples, threshold)
        for item in source_arrays
    ]
    source_score = np.mean(subject_scores, axis=0)
    target_score = channel_score(
        target_train, window_samples, step_samples, threshold
    )
    fused = (alpha * source_score + (1.0 - alpha) * target_score
             if (use_ct or use_cw) else np.ones(n_channels))
    indices = np.argsort(-fused, kind="stable")[:retained] if use_ct else np.arange(retained)
    weights = fused[indices] if use_cw else np.ones(len(indices))
    result = (indices, weights.astype(np.float32))
    if return_details:
        return result + ({
            "source_subject_scores": np.asarray(subject_scores),
            "source_group_score": source_score,
            "target_subject_score": target_score,
            "fused_score": fused,
        },)
    return result


def apply_channel_transform(x, indices, weights):
    """Select channels in the requested order."""
    del weights
    return x[:, indices, :].astype(np.float32)


def loader(x, y, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)
