---
name: bayes-kalman-trace
description: Run recursive Bayesian state-space filtering and log-linear likelihood pooling over noisy kinematic or biological velocity time series with missing-observation marginalization.
---

# BayesKalmanTrace Agent Skill

Use this skill inside Google Antigravity to perform calibrated Bayesian trajectory classification and uncertainty quantification on time-series telemetry with sensor dropouts.

## Usage

```bash
uv run python -m bayes_kalman_trace.estimator --samples 600 --dropout 0.08
```
