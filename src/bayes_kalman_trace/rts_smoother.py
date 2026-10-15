from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SmoothedTrajectoryState:
    filtered_positions: List[float]
    filtered_velocities: List[float]
    filtered_pos_variances: List[float]
    smoothed_positions: List[float]
    smoothed_velocities: List[float]
    smoothed_pos_variances: List[float]
    innovation_log_likelihood: float
    missing_count: int


def _mat2_mul(
    a: Tuple[Tuple[float, float], Tuple[float, float]],
    b: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (
        (a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
        (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]),
    )


def _mat2_add(
    a: Tuple[Tuple[float, float], Tuple[float, float]],
    b: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (
        (a[0][0] + b[0][0], a[0][1] + b[0][1]),
        (a[1][0] + b[1][0], a[1][1] + b[1][1]),
    )


def _mat2_sub(
    a: Tuple[Tuple[float, float], Tuple[float, float]],
    b: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (
        (a[0][0] - b[0][0], a[0][1] - b[0][1]),
        (a[1][0] - b[1][0], a[1][1] - b[1][1]),
    )


def _mat2_T(
    a: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return ((a[0][0], a[1][0]), (a[0][1], a[1][1]))


def _mat2_inv(
    a: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if abs(det) < 1e-15:
        det = 1e-15 if det >= 0 else -1e-15
    inv_d = 1.0 / det
    return (
        (a[1][1] * inv_d, -a[0][1] * inv_d),
        (-a[1][0] * inv_d, a[0][0] * inv_d),
    )


class KinematicRTSSmoother:
    """Linear constant-velocity Kalman filter and exact Rauch-Tung-Striebel (RTS) backward smoother."""

    def __init__(
        self,
        dt: float = 1.0,
        process_accel_std: float = 0.5,
        measurement_pos_std: float = 2.0,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if process_accel_std <= 0.0 or measurement_pos_std <= 0.0:
            raise ValueError("Noise standard deviations must be positive")
        self.dt = dt
        self.q_var = process_accel_std ** 2
        self.r_var = measurement_pos_std ** 2
        self.F: Tuple[Tuple[float, float], Tuple[float, float]] = ((1.0, dt), (0.0, 1.0))
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        self.Q: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0.25 * dt4 * self.q_var, 0.5 * dt3 * self.q_var),
            (0.5 * dt3 * self.q_var, dt2 * self.q_var),
        )

    def filter_and_smooth(
        self,
        position_observations: Sequence[Optional[float]],
        init_pos: float = 0.0,
        init_vel: float = 0.0,
        init_var: float = 25.0,
    ) -> SmoothedTrajectoryState:
        if not position_observations:
            raise ValueError("position_observations cannot be empty")

        n = len(position_observations)
        x_pred_list: List[Tuple[float, float]] = []
        P_pred_list: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        x_filt_list: List[Tuple[float, float]] = []
        P_filt_list: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        x_curr = (init_pos, init_vel)
        P_curr: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (init_var, 0.0),
            (0.0, init_var),
        )
        FT = _mat2_T(self.F)
        log_lik = 0.0
        missing_count = 0

        for idx, z in enumerate(position_observations):
            if idx == 0:
                x_pred = x_curr
                P_pred = P_curr
            else:
                x_pred = (
                    self.F[0][0] * x_curr[0] + self.F[0][1] * x_curr[1],
                    self.F[1][0] * x_curr[0] + self.F[1][1] * x_curr[1],
                )
                P_pred = _mat2_add(_mat2_mul(_mat2_mul(self.F, P_curr), FT), self.Q)

            x_pred_list.append(x_pred)
            P_pred_list.append(P_pred)

            if z is None or math.isnan(z):
                missing_count += 1
                x_curr = x_pred
                P_curr = P_pred
            else:
                # Measurement H = [1, 0]
                innov = z - x_pred[0]
                s_cov = P_pred[0][0] + self.r_var
                k0 = P_pred[0][0] / s_cov
                k1 = P_pred[1][0] / s_cov
                x_curr = (x_pred[0] + k0 * innov, x_pred[1] + k1 * innov)
                # P_filt = (I - K H) P_pred
                P_curr = (
                    ((1.0 - k0) * P_pred[0][0], (1.0 - k0) * P_pred[0][1]),
                    (P_pred[1][0] - k1 * P_pred[0][0], P_pred[1][1] - k1 * P_pred[0][1]),
                )
                log_lik += -0.5 * (math.log(2.0 * math.pi * s_cov) + (innov * innov) / s_cov)

            x_filt_list.append(x_curr)
            P_filt_list.append(P_curr)

        # Backward Rauch-Tung-Striebel (RTS) smoothing pass
        x_smooth: List[Tuple[float, float]] = [x_filt_list[-1]] * n
        P_smooth: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [P_filt_list[-1]] * n

        for t in range(n - 2, -1, -1):
            P_next_pred_inv = _mat2_inv(P_pred_list[t + 1])
            C_t = _mat2_mul(_mat2_mul(P_filt_list[t], FT), P_next_pred_inv)
            CT = _mat2_T(C_t)
            dx0 = x_smooth[t + 1][0] - x_pred_list[t + 1][0]
            dx1 = x_smooth[t + 1][1] - x_pred_list[t + 1][1]
            x_smooth[t] = (
                x_filt_list[t][0] + C_t[0][0] * dx0 + C_t[0][1] * dx1,
                x_filt_list[t][1] + C_t[1][0] * dx0 + C_t[1][1] * dx1,
            )
            dP = _mat2_sub(P_smooth[t + 1], P_pred_list[t + 1])
            P_smooth[t] = _mat2_add(P_filt_list[t], _mat2_mul(_mat2_mul(C_t, dP), CT))

        return SmoothedTrajectoryState(
            filtered_positions=[round(x[0], 5) for x in x_filt_list],
            filtered_velocities=[round(x[1], 5) for x in x_filt_list],
            filtered_pos_variances=[round(P[0][0], 6) for P in P_filt_list],
            smoothed_positions=[round(x[0], 5) for x in x_smooth],
            smoothed_velocities=[round(x[1], 5) for x in x_smooth],
            smoothed_pos_variances=[round(P[0][0], 6) for P in P_smooth],
            innovation_log_likelihood=round(log_lik, 4),
            missing_count=missing_count,
        )
