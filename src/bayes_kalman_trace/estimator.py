from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class TrajectoryResult:
    final_posterior_class_1: float
    predicted_class: int
    log_bayes_factor: float
    observed_steps: int
    missing_steps: int
    posterior_trace: List[float]


class RecursiveBayesianClassifier:
    """Recursive Bayesian filter with Markov transition matrix, missing-data propagation, and log-linear pooling."""

    def __init__(
        self,
        stay_prob_0: float = 0.985,
        stay_prob_1: float = 0.985,
        alpha_speed: float = 0.65,
        alpha_volatility: float = 0.55,
        window_size: int = 12,
    ) -> None:
        self.t00 = stay_prob_0
        self.t01 = 1.0 - stay_prob_0
        self.t11 = stay_prob_1
        self.t10 = 1.0 - stay_prob_1
        self.alpha_speed = alpha_speed
        self.alpha_volatility = alpha_volatility
        self.window_size = window_size

    @staticmethod
    def _gaussian_log_pdf(x: float, mu: float, sigma: float, laplace_floor: float = 1e-4) -> float:
        z = (x - mu) / max(sigma, 1e-6)
        density = (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * math.exp(-0.5 * z * z)
        return math.log(max(density, laplace_floor))

    def _rolling_volatility(self, window: Sequence[float]) -> Optional[float]:
        valid = [v for v in window if v is not None and not math.isnan(v)]
        if len(valid) < 3:
            return None
        mean = sum(valid) / len(valid)
        var = sum((x - mean) ** 2 for x in valid) / (len(valid) - 1)
        return math.sqrt(var)

    def filter_series(self, velocities: Sequence[Optional[float]]) -> TrajectoryResult:
        p0, p1 = 0.5, 0.5
        trace: List[float] = []
        observed = 0
        missing = 0
        recent: List[float] = []

        for v in velocities:
            # 1. Markov transition prior update
            pred0 = self.t00 * p0 + self.t10 * p1
            pred1 = self.t01 * p0 + self.t11 * p1

            # 2. Missing data behavior: propagate transition prior without measurement update
            if v is None or math.isnan(v):
                missing += 1
                p0, p1 = pred0, pred1
                trace.append(round(p1, 6))
                continue

            observed += 1
            recent.append(v)
            if len(recent) > self.window_size:
                recent.pop(0)

            # Class 0: biological/low-volatility agile regime; Class 1: ballistic/high-speed low-jitter regime
            ll0 = self.alpha_speed * self._gaussian_log_pdf(v, mu=11.5, sigma=4.2)
            ll1 = self.alpha_speed * self._gaussian_log_pdf(v, mu=22.0, sigma=5.0)

            vol = self._rolling_volatility(recent)
            if vol is not None:
                ll0 += self.alpha_volatility * self._gaussian_log_pdf(vol, mu=3.8, sigma=1.1)
                ll1 += self.alpha_volatility * self._gaussian_log_pdf(vol, mu=1.1, sigma=0.55)

            log_post0 = math.log(max(pred0, 1e-12)) + ll0
            log_post1 = math.log(max(pred1, 1e-12)) + ll1
            max_lp = max(log_post0, log_post1)
            w0 = math.exp(log_post0 - max_lp)
            w1 = math.exp(log_post1 - max_lp)
            norm = w0 + w1
            p0, p1 = w0 / norm, w1 / norm
            trace.append(round(p1, 6))

        eps = 1e-9
        log_bf = math.log(max(p1, eps) / max(p0, eps))
        return TrajectoryResult(
            final_posterior_class_1=round(p1, 6),
            predicted_class=1 if p1 >= 0.5 else 0,
            log_bayes_factor=round(log_bf, 4),
            observed_steps=observed,
            missing_steps=missing,
            posterior_trace=trace,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="BayesKalmanTrace Trajectory Estimator")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--dropout", type=float, default=0.08)
    args = parser.parse_args()

    rng = random.Random(42)
    synthetic = [
        None if rng.random() < args.dropout else rng.gauss(21.8, 1.05)
        for _ in range(args.samples)
    ]
    clf = RecursiveBayesianClassifier()
    res = clf.filter_series(synthetic)
    print(
        json.dumps(
            {
                "predicted_class": res.predicted_class,
                "final_posterior_class_1": res.final_posterior_class_1,
                "log_bayes_factor": res.log_bayes_factor,
                "observed_steps": res.observed_steps,
                "missing_steps": res.missing_steps,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
