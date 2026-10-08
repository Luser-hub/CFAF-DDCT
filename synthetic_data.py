"""Generate artificial three-class EEG-like arrays in memory.

The data are for software checks only and have no scientific meaning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def generate_subject(
    subject_id: int,
    trials_per_class: int = 12,
    channels: int = 60,
    samples: int = 128,
    seed: int = 2026,
    target_domain_shift: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return x with shape [trials, channels, samples] and labels [trials]."""
    if trials_per_class < 6 or channels < 12 or samples < 128:
        raise ValueError("Need at least 6 trials/class, 12 channels, and 128 samples")
    rng = np.random.default_rng(seed + int(subject_id))
    time = np.arange(samples, dtype=np.float32) / 250.0
    signals, labels = [], []
    is_target = int(subject_id) == 1 and target_domain_shift
    subject_scale = 0.62 if is_target else 0.85 + 0.03 * (int(subject_id) % 7)
    noise_scale = 0.50 if is_target else 0.35
    for cls in range(3):
        # The target domain has a sensor-layout and frequency shift. Labels
        # remain identical, so this is a domain shift rather than a label bug.
        channel_offset = 16 if is_target else 0
        active = np.arange(cls * 4 + channel_offset, cls * 4 + channel_offset + 8) % channels
        frequency = (9.5 if is_target else 8.0) + 3.0 * cls
        for _ in range(trials_per_class):
            signal = rng.normal(0.0, noise_scale, (channels, samples)).astype(np.float32)
            phase = rng.uniform(-np.pi, np.pi)
            signal[active] += (
                subject_scale * np.sin(2 * np.pi * frequency * time + phase)
            )[None, :]
            signals.append(signal)
            labels.append(cls)
    order = rng.permutation(len(labels))
    return np.asarray(signals, dtype=np.float32)[order], np.asarray(labels, dtype=np.int64)[order]


def generate_dataset(
    subjects: int = 25,
    trials_per_class: int = 12,
    channels: int = 60,
    samples: int = 128,
    seed: int = 2026,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return an in-memory dictionary keyed by subject number."""
    if not 1 <= subjects <= 25:
        raise ValueError("subjects must be between 1 and 25")
    return {
        subject: generate_subject(
            subject, trials_per_class, channels, samples, seed
        )
        for subject in range(1, subjects + 1)
    }


def generate_experiment_data(seed: int = 2026):
    """Return source subjects and one target subject for ``main.py``.

    The target is intentionally held separately so the training module can
    perform its own stratified adaptation/validation/test split.
    """
    # Subject 1 is the target domain; subjects 2-16 form 15 source domains.
    # Every subject has the same number of trials, so source_total / target = 15.
    data = generate_dataset(subjects=16, trials_per_class=12, samples=128, seed=seed)
    target = data.pop(1)
    return [data[key] for key in sorted(data)], target[0], target[1]


def save_experiment_data(output_dir: Path, seed: int = 2026) -> Path:
    """Generate the demo data and save one compressed NPZ file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_subjects, target_x, target_y = generate_experiment_data(seed=seed)
    source_x = np.concatenate([x for x, _ in source_subjects], axis=0)
    source_y = np.concatenate([y for _, y in source_subjects], axis=0)
    source_subject_ids = np.concatenate([
        np.full(len(x), subject_index, dtype=np.int64)
        for subject_index, (x, _) in enumerate(source_subjects)
    ])
    path = output_dir / "synthetic_data.npz"
    np.savez_compressed(
        path, source_x=source_x, source_y=source_y,
        source_subject_ids=source_subject_ids,
        target_x=target_x, target_y=target_y,
    )
    return path


def load_experiment_data(path: Path):
    """Load a file created by :func:`save_experiment_data`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic data file not found: {path}. "
            "Run `python synthetic_data.py --output-dir outputs` first."
        )
    with np.load(path) as saved:
        required = {"source_x", "source_y", "source_subject_ids", "target_x", "target_y"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Synthetic data file is missing fields: {sorted(missing)}")
        source_x = np.asarray(saved["source_x"], dtype=np.float32)
        source_y = np.asarray(saved["source_y"], dtype=np.int64)
        source_subject_ids = np.asarray(saved["source_subject_ids"], dtype=np.int64)
        target_x = np.asarray(saved["target_x"], dtype=np.float32)
        target_y = np.asarray(saved["target_y"], dtype=np.int64)
    if len(source_x) != len(source_y) or len(source_x) != len(source_subject_ids):
        raise ValueError("Source arrays have inconsistent lengths")
    source_subjects = []
    for subject_id in np.unique(source_subject_ids):
        mask = source_subject_ids == subject_id
        source_subjects.append((source_x[mask], source_y[mask]))
    return source_subjects, target_x, target_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CFAF-DDCT synthetic data")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    saved_path = save_experiment_data(args.output_dir, seed=args.seed)
    source_subjects, target_x, target_y = load_experiment_data(saved_path)
    print(f"[data] saved: {saved_path.resolve()}")
    print(f"[data] source_subjects={len(source_subjects)} "
          f"per_subject={[tuple(x.shape) for x, _ in source_subjects]}")
    print(f"[data] target_x={tuple(target_x.shape)} target_y={tuple(target_y.shape)}")
