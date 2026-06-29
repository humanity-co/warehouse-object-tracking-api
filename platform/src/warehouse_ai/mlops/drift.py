from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    eps = 1e-6
    breaks = np.unique(
        np.quantile(np.concatenate([expected, actual]), np.linspace(0, 1, bins + 1))
    )
    if len(breaks) < 3:
        return 0.0
    exp_hist, _ = np.histogram(expected, bins=breaks)
    act_hist, _ = np.histogram(actual, bins=breaks)
    exp_pct = exp_hist / max(exp_hist.sum(), 1)
    act_pct = act_hist / max(act_hist.sum(), 1)
    return float(np.sum((act_pct - exp_pct) * np.log((act_pct + eps) / (exp_pct + eps))))


def kl_divergence(expected: np.ndarray, actual: np.ndarray, bins: int = 20) -> float:
    eps = 1e-6
    breaks = np.unique(
        np.quantile(np.concatenate([expected, actual]), np.linspace(0, 1, bins + 1))
    )
    if len(breaks) < 3:
        return 0.0
    exp_hist, _ = np.histogram(expected, bins=breaks)
    act_hist, _ = np.histogram(actual, bins=breaks)
    exp_pct = exp_hist / max(exp_hist.sum(), 1)
    act_pct = act_hist / max(act_hist.sum(), 1)
    return float(np.sum((act_pct + eps) * np.log((act_pct + eps) / (exp_pct + eps))))


@dataclass
class DriftDecision:
    psi: float
    kl_divergence: float
    should_retrain: bool


class DistributionShiftDetector:
    def __init__(self, psi_threshold: float = 0.2, kl_threshold: float = 0.15) -> None:
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold

    def evaluate(self, baseline: np.ndarray, current: np.ndarray) -> DriftDecision:
        psi = population_stability_index(baseline, current)
        kl = kl_divergence(baseline, current)
        return DriftDecision(
            psi=psi,
            kl_divergence=kl,
            should_retrain=psi >= self.psi_threshold or kl >= self.kl_threshold,
        )

