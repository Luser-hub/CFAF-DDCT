"""In-memory source pretraining and target adaptation."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .data import loader, seed_everything, select_and_weight_channels, split_target
from .losses import DomainDiscriminator, compute_adaptation_loss
from .model import EEGConformer


def _accuracy(model, data_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in data_loader:
            logits, _ = model(x.to(device))
            correct += (logits.argmax(1).cpu() == y).sum().item()
            total += len(y)
    return correct / max(total, 1)


def _source_pretrain(model, source_loader, epochs, device, verbose=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in source_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device))[0], y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        mean_loss = total_loss / max(len(source_loader), 1)
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        if verbose:
            print(f"[source pretrain] epoch {epoch + 1}/{epochs} loss={mean_loss:.6f}")
    return history


def _select_source_for_adaptation(source_x, source_y, target_y, ratio, seed):
    """Select gamma times as many source samples as target samples per class."""
    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(target_y):
        source_indices = np.flatnonzero(source_y == cls)
        target_count = int(np.sum(target_y == cls))
        requested = max(1, int(round(target_count * ratio)))
        if requested > len(source_indices):
            raise ValueError(
                f"gamma={ratio} requests {requested} source samples for class {cls}, "
                f"but only {len(source_indices)} are available"
            )
        selected.extend(rng.choice(source_indices, requested, replace=False).tolist())
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return source_x[selected], source_y[selected]


def run(source_subjects, target_x, target_y, config: Config, device=None, verbose=True):
    """Run the complete experiment and return metrics/history in memory."""
    config.validate()
    seed_everything(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if verbose:
        print(f"[device] {device}")
        print(f"[input] target_x={tuple(target_x.shape)} target_y={tuple(target_y.shape)}")
        print(f"[input] source_subjects={len(source_subjects)} "
              f"per_subject={[tuple(x.shape) for x, _ in source_subjects]}")
    train_idx, val_idx, test_idx = split_target(target_y, config.seed)
    if verbose:
        print(f"[split] adaptation={len(train_idx)} validation={len(val_idx)} test={len(test_idx)}")
    channel_idx, channel_weights = select_and_weight_channels(
        source_subjects, target_x[train_idx], config.retained_channels, alpha=0.5
    )
    if verbose:
        print(f"[CT/CW] selected {len(channel_idx)} channels; "
              f"indices={channel_idx.tolist()}")

    def transform(x):
        return (x[:, channel_idx, :] * channel_weights[None, :, None]).astype(np.float32)

    source_x = np.concatenate([transform(x) for x, _ in source_subjects], axis=0)
    source_y = np.concatenate([y for _, y in source_subjects], axis=0)
    target_train, target_val, target_test = (transform(target_x[i]) for i in (train_idx, val_idx, test_idx))
    adaptation_source_x, adaptation_source_y = _select_source_for_adaptation(
        source_x, source_y, target_y[train_idx], config.source_target_ratio, config.seed
    )
    model = EEGConformer(len(channel_idx), target_x.shape[2]).to(device)
    if verbose:
        print(f"[model] transformed source={tuple(source_x.shape)} "
              f"target_train={tuple(target_train.shape)} "
              f"feature_dim={model.feature_dim}")
        print(f"[gamma] source/target adaptation ratio={config.source_target_ratio:.3f}; "
              f"selected source={len(adaptation_source_x)}, target={len(target_train)} "
              f"(actual={len(adaptation_source_x) / max(len(target_train), 1):.3f})")
    source_loader = loader(source_x, source_y, config.batch_size, True, config.seed)
    adaptation_source_loader = loader(
        adaptation_source_x, adaptation_source_y,
        min(len(adaptation_source_x), max(1, int(round(
            config.source_target_ratio * config.batch_size
        )))), True, config.seed + 4
    )
    train_loader = loader(target_train, target_y[train_idx], config.batch_size, True, config.seed + 1)
    val_loader = loader(target_val, target_y[val_idx], config.batch_size, False, config.seed + 2)
    test_loader = loader(target_test, target_y[test_idx], config.batch_size, False, config.seed + 3)
    source_history = _source_pretrain(
        model, source_loader, config.source_pretrain_epochs, device, verbose=verbose
    )
    if verbose:
        print(f"[fine-tune] loss={config.loss} lambda={config.lambda_value:.3f} "
              f"epochs={config.fine_tune_epochs}")
        print(f"[adapt] feature_normalization={config.normalize_transfer_features}")
        if config.loss == "icl":
            print("[adapt] ICL is a hinge triplet loss; raw adapt=0 means all "
                  "sampled triplets already satisfy the margin, not that labels are missing.")

    discriminator = DomainDiscriminator(model.feature_dim).to(device) if config.loss == "adver" else None
    parameters = list(model.parameters()) + ([] if discriminator is None else list(discriminator.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=5e-4, weight_decay=1e-5)
    best_accuracy, best_state = -1.0, None
    history = []
    step = 0
    for epoch in range(config.fine_tune_epochs):
        model.train()
        if discriminator is not None:
            discriminator.train()
        total_cls = total_adapt = total = 0.0
        source_iter = iter(adaptation_source_loader)
        target_iter = iter(train_loader)
        num_steps = max(len(adaptation_source_loader), len(train_loader))
        for _ in range(num_steps):
            try:
                source_batch, source_labels = next(source_iter)
            except StopIteration:
                source_iter = iter(adaptation_source_loader)
                source_batch, source_labels = next(source_iter)
            try:
                target_batch, target_labels = next(target_iter)
            except StopIteration:
                target_iter = iter(train_loader)
                target_batch, target_labels = next(target_iter)
            target_batch, target_labels = target_batch.to(device), target_labels.to(device)
            source_batch, source_labels = source_batch.to(device), source_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            target_logits, target_features = model(target_batch)
            _, source_features = model(source_batch)
            if config.normalize_transfer_features:
                source_features = F.normalize(source_features, dim=1)
                target_features = F.normalize(target_features, dim=1)
            cls = F.cross_entropy(target_logits, target_labels)
            adaptation = compute_adaptation_loss(
                config.loss, source_features, target_features, source_labels,
                target_labels, target_logits, discriminator, step, config
            ) if config.lambda_value else target_features.sum() * 0.0
            objective = cls + config.lambda_value * adaptation
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite training objective")
            objective.backward()
            nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total_cls += cls.item(); total_adapt += adaptation.item(); total += objective.item()
            step += 1
        validation_accuracy = _accuracy(model, val_loader, device)
        history.append({"epoch": epoch + 1, "classification_loss": total_cls / num_steps,
                        "adaptation_loss": total_adapt / num_steps,
                        "total_loss": total / num_steps,
                        "validation_accuracy": validation_accuracy})
        if verbose:
            row = history[-1]
            print(f"[fine-tune] epoch {row['epoch']}/{config.fine_tune_epochs} "
                  f"cls={row['classification_loss']:.6f} "
                  f"adapt={row['adaptation_loss']:.3e} "
                  f"weighted_adapt={config.lambda_value * row['adaptation_loss']:.3e} "
                  f"total={row['total_loss']:.6f} "
                  f"val_acc={row['validation_accuracy']:.4f}")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    test_accuracy = _accuracy(model, test_loader, device)
    if verbose:
        print(f"[test] locked_test_acc={test_accuracy:.4f}")
    return {
        "loss": config.loss,
        "lambda": config.lambda_value,
        "source_target_ratio": config.source_target_ratio,
        "normalize_transfer_features": config.normalize_transfer_features,
        "selected_channels": channel_idx.tolist(),
        "validation_accuracy": float(best_accuracy),
        "test_accuracy": float(test_accuracy),
        "device": str(device),
        "input_shapes": {
            "target_x": list(target_x.shape),
            "target_y": list(target_y.shape),
            "source_subjects": [list(x.shape) for x, _ in source_subjects],
            "transformed_source_x": list(source_x.shape),
            "target_train": list(target_train.shape),
            "target_validation": list(target_val.shape),
            "target_test": list(target_test.shape),
            "adaptation_source_x": list(adaptation_source_x.shape),
        },
        "split_sizes": {
            "adaptation": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "source_pretrain_history": source_history,
        "history": history,
    }
