import random
import pytest
from bayes_kalman_trace import (
    KinematicRTSSmoother,
    LaplaceHistogramLikelihood,
    evaluate_posterior_calibration,
)


def test_rts_smoother_reduces_variance_and_handles_missing_steps() -> None:
    rng = random.Random(123)
    true_vel = 2.5
    true_positions = [true_vel * t for t in range(40)]
    noisy_obs = [
        None if t in (10, 11, 25) else pos + rng.gauss(0.0, 1.5)
        for t, pos in enumerate(true_positions)
    ]

    smoother = KinematicRTSSmoother(dt=1.0, process_accel_std=0.2, measurement_pos_std=1.5)
    res = smoother.filter_and_smooth(noisy_obs, init_pos=0.0, init_vel=2.0)

    assert res.missing_count == 3
    # RTS backward smoothing must strictly reduce or equal interior position variance
    for t in range(len(true_positions) - 1):
        assert res.smoothed_pos_variances[t] <= res.filtered_pos_variances[t] + 1e-9

    filt_mse = sum((p - tp) ** 2 for p, tp in zip(res.filtered_positions, true_positions)) / 40
    smooth_mse = sum((p - tp) ** 2 for p, tp in zip(res.smoothed_positions, true_positions)) / 40
    assert smooth_mse < filt_mse


def test_laplace_histogram_and_posterior_calibration() -> None:
    hist = LaplaceHistogramLikelihood(min_val=0.0, max_val=30.0, num_bins=15, laplace_alpha=1.0)
    hist.fit([10.2, 11.5, 10.8, 12.1, 11.0, 10.9, 11.3])
    # Density near 11.0 must exceed out-of-support density near 28.0
    assert hist.log_likelihood(11.0) > hist.log_likelihood(28.0)

    probs = [0.95, 0.92, 0.88, 0.08, 0.04, 0.10]
    labels = [1, 1, 1, 0, 0, 0]
    cal = evaluate_posterior_calibration(probs, labels, num_bins=5)
    assert cal.brier_score < 0.02
    assert cal.expected_calibration_error < 0.10
    assert cal.num_samples == 6

    with pytest.raises(ValueError):
        evaluate_posterior_calibration([], [])
    with pytest.raises(ValueError):
        evaluate_posterior_calibration([1.2], [1])
    with pytest.raises(ValueError, match="num_bins"):
        evaluate_posterior_calibration([0.5], [1], num_bins=0)

    smoother = KinematicRTSSmoother()
    with pytest.raises(ValueError, match="init_var"):
        smoother.filter_and_smooth([1.0, 2.0], init_var=-1.0)

