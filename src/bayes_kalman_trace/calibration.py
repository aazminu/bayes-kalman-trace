from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class CalibrationMetrics:
    expected_calibration_error: float
    brier_score: float
    negative_log_loss: float
    num_samples: int


class LaplaceHistogramLikelihood:
    """Empirical histogram density estimator with additive Laplace smoothing."""

    def __init__(
        self,
        min_val: float,
        max_val: float,
        num_bins: int = 20,
        laplace_alpha: float = 1.0,
    ) -> None:
        if max_val <= min_val:
            raise ValueError("max_val must be greater than min_val")
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if laplace_alpha <= 0.0:
            raise ValueError("laplace_alpha must be positive")
        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = num_bins
        self.laplace_alpha = laplace_alpha
        self.bin_width = (max_val - min_val) / num_bins
        self._bin_probs: List[float] = [1.0 / num_bins] * num_bins

    def _bin_index(self, x: float) -> int:
        if x <= self.min_val:
            return 0
        if x >= self.max_val:
            return self.num_bins - 1
        idx = int((x - self.min_val) / self.bin_width)
        return min(max(idx, 0), self.num_bins - 1)

    def fit(self, samples: Sequence[float]) -> None:
        valid = [x for x in samples if x is not None and not math.isnan(x)]
        if not valid:
            raise ValueError("Cannot fit LaplaceHistogramLikelihood on empty samples")
        counts = [0] * self.num_bins
        for x in valid:
            counts[self._bin_index(x)] += 1
        denom = len(valid) + self.num_bins * self.laplace_alpha
        self._bin_probs = [(c + self.laplace_alpha) / denom for c in counts]

    def log_likelihood(self, x: float) -> float:
        idx = self._bin_index(x)
        density = self._bin_probs[idx] / self.bin_width
        return math.log(max(density, 1e-12))


def evaluate_posterior_calibration(
    predicted_probs: Sequence[float],
    true_labels: Sequence[int],
    num_bins: int = 10,
) -> CalibrationMetrics:
    if len(predicted_probs) != len(true_labels) or not predicted_probs:
        raise ValueError("predicted_probs and true_labels must be non-empty and equal length")
    if any(y not in (0, 1) for y in true_labels):
        raise ValueError("true_labels must be binary (0 or 1)")
    if any(p < 0.0 or p > 1.0 or math.isnan(p) for p in predicted_probs):
        raise ValueError("predicted_probs must be valid probabilities in [0, 1]")

    n = len(predicted_probs)
    brier = sum((p - y) ** 2 for p, y in zip(predicted_probs, true_labels)) / n
    eps = 1e-12
    nll = -sum(
        y * math.log(max(p, eps)) + (1 - y) * math.log(max(1.0 - p, eps))
        for p, y in zip(predicted_probs, true_labels)
    ) / n

    bin_probs: List[List[float]] = [[] for _ in range(num_bins)]
    bin_labels: List[List[int]] = [[] for _ in range(num_bins)]
    for p, y in zip(predicted_probs, true_labels):
        b = min(int(p * num_bins), num_bins - 1)
        bin_probs[b].append(p)
        bin_labels[b].append(y)

    ece = 0.0
    for b in range(num_bins):
        if not bin_probs[b]:
            continue
        avg_conf = sum(bin_probs[b]) / len(bin_probs[b])
        avg_acc = sum(bin_labels[b]) / len(bin_labels[b])
        ece += (len(bin_probs[b]) / n) * abs(avg_acc - avg_conf)

    return CalibrationMetrics(
        expected_calibration_error=round(ece, 5),
        brier_score=round(brier, 5),
        negative_log_loss=round(nll, 5),
        num_samples=n,
    )
