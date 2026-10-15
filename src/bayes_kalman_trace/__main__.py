from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .calibration import LaplaceHistogramLikelihood, evaluate_posterior_calibration
from .estimator import RecursiveBayesianClassifier
from .rts_smoother import KinematicRTSSmoother


def run_benchmark(seed: int = 42) -> List[Dict[str, Any]]:
    """Run reproducible Bayesian trajectory filtering, RTS smoothing, and calibration benchmarks."""
    rng = random.Random(seed)

    # 1. Recursive Bayesian trajectory classification with sensor dropouts
    clf = RecursiveBayesianClassifier()
    c0_series = [None if rng.random() < 0.12 else rng.gauss(11.2, 3.9) for _ in range(250)]
    c1_series = [None if rng.random() < 0.12 else rng.gauss(22.4, 1.0) for _ in range(250)]
    res0 = clf.filter_series(c0_series)
    res1 = clf.filter_series(c1_series)

    # 2. Kinematic Rauch-Tung-Striebel (RTS) backward smoothing with missing observations
    true_vel = 2.5
    true_positions = [true_vel * t for t in range(40)]
    noisy_obs = [
        None if t in (10, 11, 25) else pos + rng.gauss(0.0, 1.5)
        for t, pos in enumerate(true_positions)
    ]
    smoother = KinematicRTSSmoother(dt=1.0, process_accel_std=0.2, measurement_pos_std=1.5)
    smooth_res = smoother.filter_and_smooth(noisy_obs, init_pos=0.0, init_vel=2.0)
    filt_mse = sum((p - tp) ** 2 for p, tp in zip(smooth_res.filtered_positions, true_positions)) / 40.0
    smooth_mse = sum((p - tp) ** 2 for p, tp in zip(smooth_res.smoothed_positions, true_positions)) / 40.0

    # 3. Laplace histogram likelihood & posterior ECE/Brier calibration
    hist = LaplaceHistogramLikelihood(min_val=0.0, max_val=30.0, num_bins=15, laplace_alpha=1.0)
    hist_samples = [11.0 + rng.gauss(0.0, 0.6) for _ in range(12)]
    hist.fit(hist_samples)
    ll_gap = hist.log_likelihood(11.0) - hist.log_likelihood(28.0)

    cal = evaluate_posterior_calibration(
        predicted_probs=[
            0.95 + rng.uniform(-0.02, 0.02),
            0.92 + rng.uniform(-0.02, 0.02),
            0.88 + rng.uniform(-0.02, 0.02),
            0.08 + rng.uniform(-0.02, 0.02),
            0.04 + rng.uniform(-0.02, 0.02),
            0.10 + rng.uniform(-0.02, 0.02),
        ],
        true_labels=[1, 1, 1, 0, 0, 0],
        num_bins=5,
    )

    return [
        {
            "benchmark": "recursive_bayesian_classifier",
            "PrimaryMetric": "class1_posterior_separation",
            "value": round(res1.final_posterior_class_1 - res0.final_posterior_class_1, 6),
            "SecondaryMetric": "missing_dropouts_handled",
            "secondary_value": res0.missing_steps + res1.missing_steps,
            "status": "PASS" if (res0.predicted_class == 0 and res1.predicted_class == 1) else "FAIL",
        },
        {
            "benchmark": "kinematic_rts_smoother",
            "PrimaryMetric": "smoothed_mse",
            "value": round(smooth_mse, 4),
            "SecondaryMetric": "filtered_mse",
            "secondary_value": round(filt_mse, 4),
            "status": "REDUCED_MSE" if smooth_mse < filt_mse else "NO_GAIN",
        },
        {
            "benchmark": "posterior_calibration_ece",
            "PrimaryMetric": "expected_calibration_error",
            "value": round(cal.expected_calibration_error, 4),
            "SecondaryMetric": "log_likelihood_gap",
            "secondary_value": round(ll_gap, 4),
            "status": "CALIBRATED" if cal.brier_score < 0.02 and ll_gap > 0 else "UNCALIBRATED",
        },
    ]


def format_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = [
        "benchmark",
        "PrimaryMetric",
        "value",
        "SecondaryMetric",
        "secondary_value",
        "status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible bayes-kalman-trace state estimation and calibration benchmarks."
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for benchmark results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible benchmark execution.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to write benchmark output.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = run_benchmark(seed=args.seed)
    rendered = (
        json.dumps({"seed": args.seed, "benchmarks": rows}, indent=2)
        if args.format == "json"
        else format_markdown_table(rows)
    )
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"Error writing output to {args.output}: {exc}", file=sys.stderr)
            return 1
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
