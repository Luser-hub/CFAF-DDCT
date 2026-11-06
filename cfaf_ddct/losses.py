"""Feature-transfer losses used in the CFAF-DDCT comparison."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def gaussian_kernel(source, target, multiplier=2.0, number=5):
    joined = torch.cat((source, target), dim=0)
    distances = torch.cdist(joined, joined).square()
    count = joined.shape[0]
    denominator = max(count * (count - 1), 1)
    bandwidth = distances.detach().sum() / denominator
    bandwidth = bandwidth.clamp_min(1e-6) / (multiplier ** (number // 2))
    return sum(torch.exp(-distances / (bandwidth * multiplier ** i)) for i in range(number))


def mmd_loss(source, target, multiplier=2.0, number=5):
    if len(source) == 0 or len(target) == 0:
        return _zero(source)
    kernels = gaussian_kernel(source, target, multiplier, number)
    n = len(source)
    return (kernels[:n, :n].mean() + kernels[n:, n:].mean()
            - 2.0 * kernels[:n, n:].mean())


def lmmd_loss(source, target, source_labels, target_logits, multiplier=2.0, number=5):
    if len(source) == 0 or len(target) == 0:
        return _zero(source)
    kernels = gaussian_kernel(source, target, multiplier, number)
    n_source = len(source)
    n_classes = target_logits.shape[1]
    source_onehot = F.one_hot(source_labels, n_classes).float()
    target_prob = target_logits.detach().softmax(dim=1)
    source_weight = source_onehot / source_onehot.sum(0, keepdim=True).clamp_min(1.0)
    target_weight = target_prob / target_prob.sum(0, keepdim=True).clamp_min(1e-6)
    ss = source_weight @ source_weight.T
    tt = target_weight @ target_weight.T
    st = source_weight @ target_weight.T
    return ((ss * kernels[:n_source, :n_source]).sum()
            + (tt * kernels[n_source:, n_source:]).sum()
            - 2.0 * (st * kernels[:n_source, n_source:]).sum())


def coral_loss(source, target):
    if min(len(source), len(target)) < 2:
        return _zero(source)
    source_centered = source - source.mean(0, keepdim=True)
    target_centered = target - target.mean(0, keepdim=True)
    source_cov = source_centered.T @ source_centered / (len(source) - 1)
    target_cov = target_centered.T @ target_centered / (len(target) - 1)
    return (source_cov - target_cov).square().mean()


def triplet_icl(source, target, source_labels, target_labels, margin=1.0):
    anchors, positives, negatives = [], [], []
    for index, label in enumerate(source_labels):
        positive = torch.nonzero(source_labels == label).flatten()
        positive = positive[positive != index]
        negative = torch.nonzero(target_labels != label).flatten()
        if len(positive) and len(negative):
            anchors.append(index)
            positives.append(positive[torch.randint(len(positive), (1,), device=source.device)].item())
            negatives.append(negative[torch.randint(len(negative), (1,), device=source.device)].item())
    if not anchors:
        return _zero(source)
    return F.triplet_margin_loss(
        source[anchors], source[positives], target[negatives], margin=margin
    )


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.coefficient * grad, None


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x, coefficient):
        return self.layers(GradientReverse.apply(x, coefficient))


def adversarial_loss(source, target, discriminator, step, max_iter=1000):
    coefficient = 2.0 / (1.0 + math.exp(-min(step / max_iter, 1.0))) - 1.0
    source_logits = discriminator(source, coefficient)
    target_logits = discriminator(target, coefficient)
    source_loss = F.binary_cross_entropy_with_logits(source_logits, torch.zeros_like(source_logits))
    target_loss = F.binary_cross_entropy_with_logits(target_logits, torch.ones_like(target_logits))
    return 0.5 * (source_loss + target_loss)


def compute_adaptation_loss(name, source, target, source_labels, target_labels,
                            target_logits, discriminator, step, config):
    if name == "none":
        return _zero(target)
    if name == "icl":
        return triplet_icl(source, target, source_labels, target_labels, config.icl_margin)
    if name == "mmd":
        return mmd_loss(source, target, config.kernel_mul, config.kernel_num)
    if name == "lmmd":
        return lmmd_loss(source, target, source_labels, target_logits,
                         config.kernel_mul, config.kernel_num)
    if name == "coral":
        return coral_loss(source, target)
    if name == "adver":
        return adversarial_loss(source, target, discriminator, step)
    raise ValueError("Unknown adaptation loss: " + name)
