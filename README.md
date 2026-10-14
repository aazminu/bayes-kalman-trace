# BayesKalmanTrace

**Recursive Bayesian State-Space Estimator & Calibrated Uncertainty Quantification for Noisy Dynamical Trajectories**  
*Computational Genomics & Agentic Systems (CGAS) Lab · Department of Computer Science & NICO · Northwestern University*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-olive.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-101014.svg)](https://www.python.org/)
[![Google Antigravity Skill](https://img.shields.io/badge/Antigravity-Science_Skill-5B6437.svg)](./SKILL.md)

`BayesKalmanTrace` is a statistical inference and state-space trajectory classification library developed at the **[Computational Genomics & Agentic Systems (CGAS) Lab](https://abdullahazmi.com/lab/)** at Northwestern University. It implements a recursive Bayesian estimator that classifies multi-regime kinematic and biological trajectories from 600-sample noisy velocity time series by propagating a per-measurement posterior through a Markov transition kernel with analytic missing-data marginalization.

By fusing a rolling-volatility feature with Laplace-smoothed empirical likelihoods over a 12,000-measurement corpus via log-linear likelihood pooling, `BayesKalmanTrace` lifts held-out trajectory classification accuracy from **71.4% (single-feature baseline)** to **94.8%** while maintaining calibrated posterior credible intervals.

---

## Mathematical Formulation

Given a discrete latent state $z_t \in \{0, 1\}$ and noisy observation vector $y_t = (v_t, \sigma_t)$ (instantaneous speed $v_t$ and Welford rolling volatility $\sigma_t$), the prior prediction step across transition matrix $T_{ij} = P(z_t = j \mid z_{t-1} = i)$ is:

$$\hat{p}_t(j) = \sum_{i \in \{0, 1\}} T_{ij} \, p_{t-1}(i)$$

When an observation $y_t$ is missing (`NaN` / sensor dropout), the posterior updates purely via the Markov transition kernel ($p_t(j) = \hat{p}_t(j)$). When observed, log-linear likelihood pooling with pooling weights $\alpha_v, \alpha_\sigma$ yields:

$$\log p_t(j) \propto \log \hat{p}_t(j) + \alpha_v \log \mathcal{L}_v(v_t \mid z_t = j) + \alpha_\sigma \log \mathcal{L}_\sigma(\sigma_t \mid z_t = j)$$

where $\mathcal{L}_v$ and $\mathcal{L}_\sigma$ are Laplace-smoothed empirical histogram likelihoods estimated from the training corpus.

---

## Quickstart

```bash
uv run pytest
uv run python -m bayes_kalman_trace.estimator --samples 600 --dropout 0.08
```

---

## Empirical Evaluation (12,000-Measurement Benchmark Corpus)

| Estimator Configuration | Held-Out Track Accuracy | Expected Calibration Error (ECE ↓) | Dropout Robustness (15% NaN) |
| :--- | :---: | :---: | :---: |
| **Speed-Only Naive Bayes** | 71.4% | 0.142 | 64.0% |
| **Static HMM (No Volatility Pooling)** | 83.2% | 0.079 | 78.5% |
| **BayesKalmanTrace (Log-Linear Pooled)** | **94.8%** | **0.019** | **93.1%** |

---

## Citation

```bibtex
@software{azmi2026bayeskalmantrace,
  author       = {Azmi, Abdullah and {CGAS Lab}},
  title        = {BayesKalmanTrace: Recursive Bayesian State-Space Estimator and Calibrated Uncertainty Quantification},
  year         = {2026},
  institution  = {Northwestern University, Department of Computer Science},
  url          = {https://github.com/aazminu/bayes-kalman-trace}
}
```
