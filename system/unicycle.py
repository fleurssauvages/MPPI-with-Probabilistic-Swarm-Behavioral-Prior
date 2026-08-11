from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import controller as ctrl
except ImportError:
    import controller as ctrl

from numba import njit, prange

Array = np.ndarray
NUMBA_AVAILABLE = True
MODEL_NAME = "unicycle"

ControllerVariant = ctrl.ControllerVariant
Scene = ctrl.Scene
SimulationResult = ctrl.SimulationResult
DynamicWallScenario = ctrl.DynamicWallScenario
MPPIHomotopyMode = ctrl.MPPIHomotopyMode


@dataclass
class MPPIConfig(ctrl.ControllerConfig):
    goal_tolerance: float = 0.05
    v_min: float = -1.0
    v_max: float = 2.8
    omega_min: float = -4.5
    omega_max: float = 4.5
    noise_v: float = 0.5
    noise_omega: float = 1.0

    prior_ilqr_iterations: int = 2
    prior_ilqr_line_search_steps: int = 2
    prior_ilqr_mahalanobis_weight: float = 2.5
    prior_ilqr_covariance_floor: float = 0.12
    prior_ilqr_covariance_fallback_std: float = 0.25
    prior_ilqr_heading_weight: float = 2.5
    prior_ilqr_progress_weight: float = 1.5
    prior_ilqr_control_v_weight: float = 0.015
    prior_ilqr_control_omega_weight: float = 0.03
    prior_ilqr_regularization: float = 0.02

    max_delta_v: float = 0.7
    max_delta_omega: float = 1.4
    enforce_one_step_safety: bool = True
    one_step_safety_clearance: float = 0.0


    def __post_init__(self) -> None:
        super().__post_init__()
        if self.v_min > self.v_max or self.omega_min > self.omega_max:
            raise ValueError("Invalid unicycle control bounds.")
        for name, value in (
            ("prior_ilqr_mahalanobis_weight", self.prior_ilqr_mahalanobis_weight),
            ("prior_ilqr_covariance_floor", self.prior_ilqr_covariance_floor),
            ("prior_ilqr_covariance_fallback_std", self.prior_ilqr_covariance_fallback_std),
            ("prior_ilqr_heading_weight", self.prior_ilqr_heading_weight),
            ("prior_ilqr_progress_weight", self.prior_ilqr_progress_weight),
            ("prior_ilqr_control_v_weight", self.prior_ilqr_control_v_weight),
            ("prior_ilqr_control_omega_weight", self.prior_ilqr_control_omega_weight),
            ("prior_ilqr_regularization", self.prior_ilqr_regularization),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        self.prior_ilqr_iterations = max(1, int(self.prior_ilqr_iterations))
        self.prior_ilqr_line_search_steps = max(1, int(self.prior_ilqr_line_search_steps))


@njit(cache=True)
def _wrap_angle_nb(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

@njit(cache=True)
def _softplus_scalar_nb(z):
        if z > 40.0:
            return z
        if z < -40.0:
            return math.exp(z)
        return math.log1p(math.exp(z))


@njit(cache=True)
def unicycle_step_nb(x, u, dt):
    out = np.empty(3, dtype=np.float64)
    theta = x[2]
    out[0] = x[0] + u[0] * math.cos(theta) * dt
    out[1] = x[1] + u[0] * math.sin(theta) * dt
    out[2] = _wrap_angle_nb(theta + u[1] * dt)
    return out


@njit(cache=True, parallel=True)
def rollout_unicycle_batch_nb(x0, U, dt):
    N = U.shape[0]
    H = U.shape[1]
    X = np.zeros((N, H + 1, 3), dtype=np.float64)
    for n in prange(N):
        X[n, 0, 0] = x0[0]
        X[n, 0, 1] = x0[1]
        X[n, 0, 2] = x0[2]
        for t in range(H):
            X[n, t + 1] = unicycle_step_nb(X[n, t], U[n, t], dt)
    return X

@njit(cache=True)
def rollout_unicycle_single_nb(x0, U, dt):
    H = U.shape[0]
    X = np.zeros((H + 1, 3), dtype=np.float64)
    X[0] = x0
    for t in range(H):
        X[t + 1] = unicycle_step_nb(X[t], U[t], dt)
    return X

@njit(cache=True)
def _path_intercept_point_nb(ref, best_seg, best_qx, best_qy, lookahead_distance):
    """Return a point ``lookahead_distance`` forward from the path projection."""
    ref_len = ref.shape[0]
    tx = best_qx
    ty = best_qy
    remaining = max(0.0, lookahead_distance)
    if ref_len < 2 or best_seg >= ref_len - 1:
        return tx, ty

    dx = ref[best_seg + 1, 0] - best_qx
    dy = ref[best_seg + 1, 1] - best_qy
    seg_len = math.sqrt(dx * dx + dy * dy)
    if seg_len > 1e-12:
        if remaining <= seg_len:
            alpha = remaining / seg_len
            return best_qx + alpha * dx, best_qy + alpha * dy
        remaining -= seg_len
        tx = ref[best_seg + 1, 0]
        ty = ref[best_seg + 1, 1]

    for i in range(best_seg + 1, ref_len - 1):
        dx = ref[i + 1, 0] - ref[i, 0]
        dy = ref[i + 1, 1] - ref[i, 1]
        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len <= 1e-12:
            continue
        if remaining <= seg_len:
            alpha = remaining / seg_len
            return ref[i, 0] + alpha * dx, ref[i, 1] + alpha * dy
        remaining -= seg_len
        tx = ref[i + 1, 0]
        ty = ref[i + 1, 1]
    return tx, ty

@njit(cache=True)
def _path_arc_lengths_ilqr_nb(ref):
    n = ref.shape[0]
    arc = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dx = ref[i, 0] - ref[i - 1, 0]
        dy = ref[i, 1] - ref[i - 1, 1]
        arc[i] = arc[i - 1] + math.sqrt(dx * dx + dy * dy)
    return arc


@njit(cache=True)
def _project_path_forward_ilqr_nb(ref, arc, px, py, start_seg):
    n = ref.shape[0]
    if n < 2:
        return 0.0, ref[0, 0], ref[0, 1], 1.0, 0.0, 0.0, 0
    first = min(max(int(start_seg), 0), n - 2)
    best_d2 = 1e300
    best_seg = first
    best_alpha = 0.0
    best_qx = ref[first, 0]
    best_qy = ref[first, 1]
    best_tx = 1.0
    best_ty = 0.0
    found = False
    for i in range(first, n - 1):
        ax = ref[i, 0]
        ay = ref[i, 1]
        dx = ref[i + 1, 0] - ax
        dy = ref[i + 1, 1] - ay
        l2 = dx * dx + dy * dy
        if l2 <= 1e-16:
            continue
        alpha = ((px - ax) * dx + (py - ay) * dy) / l2
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0
        qx = ax + alpha * dx
        qy = ay + alpha * dy
        ex = px - qx
        ey = py - qy
        d2 = ex * ex + ey * ey
        if d2 < best_d2:
            inv_l = 1.0 / math.sqrt(l2)
            best_d2 = d2
            best_seg = i
            best_alpha = alpha
            best_qx = qx
            best_qy = qy
            best_tx = dx * inv_l
            best_ty = dy * inv_l
            found = True
    if not found:
        for i in range(first - 1, -1, -1):
            dx = ref[i + 1, 0] - ref[i, 0]
            dy = ref[i + 1, 1] - ref[i, 1]
            l2 = dx * dx + dy * dy
            if l2 > 1e-16:
                inv_l = 1.0 / math.sqrt(l2)
                best_seg = i
                best_alpha = 1.0
                best_qx = ref[i + 1, 0]
                best_qy = ref[i + 1, 1]
                best_tx = dx * inv_l
                best_ty = dy * inv_l
                break
    seg_len = arc[best_seg + 1] - arc[best_seg]
    progress = arc[best_seg] + best_alpha * max(seg_len, 0.0)
    heading = math.atan2(best_ty, best_tx)
    return progress, best_qx, best_qy, best_tx, best_ty, heading, best_seg


@njit(cache=True)
def _spatial_precision_ilqr_nb(cov_blocks, arc, seg, progress, covariance_floor):
    """Inverse of Sigma(s)+sigma_floor^2 I at geometric projected progress."""
    n = cov_blocks.shape[0]
    if n == 0:
        floor_var = max(covariance_floor * covariance_floor, 1e-8)
        inv = 1.0 / floor_var
        return inv, 0.0, inv
    i = min(max(int(seg), 0), max(0, n - 2))
    j = min(i + 1, n - 1)
    ds = arc[min(i + 1, arc.shape[0] - 1)] - arc[min(i, arc.shape[0] - 1)]
    alpha = 0.0 if ds <= 1e-12 else (progress - arc[i]) / ds
    alpha = min(max(alpha, 0.0), 1.0)
    beta = 1.0 - alpha
    c00 = beta * cov_blocks[i, 0, 0] + alpha * cov_blocks[j, 0, 0]
    c01a = beta * cov_blocks[i, 0, 1] + alpha * cov_blocks[j, 0, 1]
    c01b = beta * cov_blocks[i, 1, 0] + alpha * cov_blocks[j, 1, 0]
    c11 = beta * cov_blocks[i, 1, 1] + alpha * cov_blocks[j, 1, 1]
    floor_var = max(covariance_floor * covariance_floor, 1e-8)
    a = max(c00 + floor_var, floor_var)
    d = max(c11 + floor_var, floor_var)
    b = 0.5 * (c01a + c01b)
    max_b = 0.999999 * math.sqrt(max(a * d, 0.0))
    b = min(max(b, -max_b), max_b)
    det = max(a * d - b * b, 1e-14)
    return d / det, -b / det, a / det


@njit(cache=True)
def _project_unicycle_rollout_ilqr_nb(X, ref, arc, cov_blocks, covariance_floor):
    count = X.shape[0]
    progress = np.zeros(count, dtype=np.float64)
    qx = np.zeros(count, dtype=np.float64)
    qy = np.zeros(count, dtype=np.float64)
    tx = np.zeros(count, dtype=np.float64)
    ty = np.zeros(count, dtype=np.float64)
    heading = np.zeros(count, dtype=np.float64)
    p00 = np.zeros(count, dtype=np.float64)
    p01 = np.zeros(count, dtype=np.float64)
    p11 = np.zeros(count, dtype=np.float64)
    cursor = 0
    for t in range(count):
        values = _project_path_forward_ilqr_nb(ref, arc, X[t, 0], X[t, 1], cursor)
        progress[t] = values[0]
        qx[t] = values[1]
        qy[t] = values[2]
        tx[t] = values[3]
        ty[t] = values[4]
        heading[t] = values[5]
        cursor = values[6]
        a, b, d = _spatial_precision_ilqr_nb(cov_blocks, arc, cursor, progress[t], covariance_floor)
        p00[t] = a
        p01[t] = b
        p11[t] = d
    return progress, qx, qy, tx, ty, heading, p00, p01, p11

@njit(cache=True)
def _unicycle_ilqr_initial_controls_nb(x0, ref, arc, horizon, dt, v_min, v_max, omega_min, omega_max):
    U = np.zeros((horizon, 2), dtype=np.float64)
    x = np.empty(3, dtype=np.float64)
    x[0] = x0[0]
    x[1] = x0[1]
    x[2] = x0[2]
    cursor = 0
    lower_v = max(0.0, v_min)
    for t in range(horizon):
        s, qx, qy, tx, ty, heading, cursor = _project_path_forward_ilqr_nb(
            ref, arc, x[0], x[1], cursor
        )
        _ = (s, heading)
        lookahead = 0.55
        gx, gy = _path_intercept_point_nb(ref, cursor, qx, qy, lookahead)
        desired = math.atan2(gy - x[1], gx - x[0])
        err = _wrap_angle_nb(desired - x[2])
        alignment = max(0.0, math.cos(err))
        v = v_max * (0.25 + 0.75 * alignment * alignment)
        v = min(max(v, lower_v), v_max)
        omega = 3.0 * err
        omega = min(max(omega, omega_min), omega_max)
        U[t, 0] = v
        U[t, 1] = omega
        x = unicycle_step_nb(x, U[t], dt)
    return U


@njit(cache=True)
def _unicycle_ilqr_total_cost_nb(
    X, U, ref, arc, cov_blocks, covariance_floor,
    mahalanobis_weight, heading_weight, progress_weight,
    control_v_weight, control_omega_weight,
):
    progress, qx, qy, tx, ty, heading, p00, p01, p11 = _project_unicycle_rollout_ilqr_nb(
        X, ref, arc, cov_blocks, covariance_floor
    )
    H = U.shape[0]
    cost = 0.0
    for t in range(H):
        dx = X[t, 0] - qx[t]
        dy = X[t, 1] - qy[t]
        mx = p00[t] * dx + p01[t] * dy
        my = p01[t] * dx + p11[t] * dy
        mahal = dx * mx + dy * my
        eh = _wrap_angle_nb(X[t, 2] - heading[t])
        cost += mahalanobis_weight * mahal + heading_weight * eh * eh - progress_weight * progress[t]
        cost += control_v_weight * U[t, 0] * U[t, 0] + control_omega_weight * U[t, 1] * U[t, 1]
    return cost

@njit(cache=True)
def _invert_regularized_2x2_ilqr_nb(a00, a01, a11, regularization):
    r = max(regularization, 1e-9)
    a00 += r
    a11 += r
    det = a00 * a11 - a01 * a01
    if det <= 1e-12:
        a00 += 10.0 * r + 1e-6
        a11 += 10.0 * r + 1e-6
        det = max(a00 * a11 - a01 * a01, 1e-12)
    return a11 / det, -a01 / det, a00 / det


@njit(cache=True)
def _unicycle_ilqr_backward_nb(
    X, U, ref, arc, cov_blocks, covariance_floor, dt,
    mahalanobis_weight, heading_weight, progress_weight,
    control_v_weight, control_omega_weight, regularization,
):
    H = U.shape[0]
    progress, qx, qy, tx, ty, heading, p00, p01, p11 = _project_unicycle_rollout_ilqr_nb(
        X, ref, arc, cov_blocks, covariance_floor
    )
    kff = np.zeros((H, 2), dtype=np.float64)
    Kfb = np.zeros((H, 2, 3), dtype=np.float64)

    Vx = np.zeros(3, dtype=np.float64)
    Vxx = np.zeros((3, 3), dtype=np.float64)

    for t in range(H - 1, -1, -1):
        theta = X[t, 2]
        v = U[t, 0]
        c = math.cos(theta)
        sn = math.sin(theta)
        A = np.eye(3, dtype=np.float64)
        A[0, 2] = -v * sn * dt
        A[1, 2] = v * c * dt
        B = np.zeros((3, 2), dtype=np.float64)
        B[0, 0] = c * dt
        B[1, 0] = sn * dt
        B[2, 1] = dt

        dx = X[t, 0] - qx[t]
        dy = X[t, 1] - qy[t]
        mx = p00[t] * dx + p01[t] * dy
        my = p01[t] * dx + p11[t] * dy
        eh = _wrap_angle_nb(X[t, 2] - heading[t])
        lx = np.zeros(3, dtype=np.float64)
        lx[0] = 2.0 * mahalanobis_weight * mx - progress_weight * tx[t]
        lx[1] = 2.0 * mahalanobis_weight * my - progress_weight * ty[t]
        lx[2] = 2.0 * heading_weight * eh
        lxx = np.zeros((3, 3), dtype=np.float64)
        lxx[0, 0] = 2.0 * mahalanobis_weight * p00[t]
        lxx[0, 1] = 2.0 * mahalanobis_weight * p01[t]
        lxx[1, 0] = lxx[0, 1]
        lxx[1, 1] = 2.0 * mahalanobis_weight * p11[t]
        lxx[2, 2] = 2.0 * heading_weight
        lu = np.array((2.0 * control_v_weight * U[t, 0], 2.0 * control_omega_weight * U[t, 1]), dtype=np.float64)
        luu = np.zeros((2, 2), dtype=np.float64)
        luu[0, 0] = 2.0 * control_v_weight
        luu[1, 1] = 2.0 * control_omega_weight

        Qx = lx + A.T @ Vx
        Qu = lu + B.T @ Vx
        Qxx = lxx + A.T @ Vxx @ A
        Quu = luu + B.T @ Vxx @ B
        Qux = B.T @ Vxx @ A

        inv00, inv01, inv11 = _invert_regularized_2x2_ilqr_nb(
            Quu[0, 0], 0.5 * (Quu[0, 1] + Quu[1, 0]), Quu[1, 1], regularization
        )
        k0 = -(inv00 * Qu[0] + inv01 * Qu[1])
        k1 = -(inv01 * Qu[0] + inv11 * Qu[1])
        kff[t, 0] = k0
        kff[t, 1] = k1
        for j in range(3):
            Kfb[t, 0, j] = -(inv00 * Qux[0, j] + inv01 * Qux[1, j])
            Kfb[t, 1, j] = -(inv01 * Qux[0, j] + inv11 * Qux[1, j])

        K = Kfb[t]
        kval = kff[t]
        Vx = Qx + K.T @ Quu @ kval + K.T @ Qu + Qux.T @ kval
        Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
        Vxx = 0.5 * (Vxx + Vxx.T)
    return kff, Kfb

@njit(cache=True)
def _unicycle_ilqr_forward_update_nb(x0, U, X, kff, Kfb, alpha, dt, v_min, v_max, omega_min, omega_max):
    H = U.shape[0]
    Unew = np.zeros_like(U)
    Xnew = np.zeros_like(X)
    Xnew[0, 0] = x0[0]
    Xnew[0, 1] = x0[1]
    Xnew[0, 2] = x0[2]
    lower_v = max(0.0, v_min)
    for t in range(H):
        dx0 = Xnew[t, 0] - X[t, 0]
        dx1 = Xnew[t, 1] - X[t, 1]
        dx2 = _wrap_angle_nb(Xnew[t, 2] - X[t, 2])
        du0 = alpha * kff[t, 0] + Kfb[t, 0, 0] * dx0 + Kfb[t, 0, 1] * dx1 + Kfb[t, 0, 2] * dx2
        du1 = alpha * kff[t, 1] + Kfb[t, 1, 0] * dx0 + Kfb[t, 1, 1] * dx1 + Kfb[t, 1, 2] * dx2
        u0 = min(max(U[t, 0] + du0, lower_v), v_max)
        u1 = min(max(U[t, 1] + du1, omega_min), omega_max)
        Unew[t, 0] = u0
        Unew[t, 1] = u1
        Xnew[t + 1] = unicycle_step_nb(Xnew[t], Unew[t], dt)
    return Unew, Xnew


@njit(cache=True)
def _unicycle_ilqr_nominal_and_positions_nb(
    x0, ref, cov_blocks, horizon, dt, v_min, v_max, omega_min, omega_max,
    iterations, line_search_steps,
    mahalanobis_weight, covariance_floor, heading_weight, progress_weight,
    control_v_weight, control_omega_weight, regularization,
):
    Uzero = np.zeros((horizon, 2), dtype=np.float64)
    positions = np.zeros(horizon, dtype=np.float64)
    if ref.shape[0] < 2:
        return Uzero, positions
    arc = _path_arc_lengths_ilqr_nb(ref)
    if arc[arc.shape[0] - 1] <= 1e-10:
        return Uzero, positions

    U = _unicycle_ilqr_initial_controls_nb(x0, ref, arc, horizon, dt, v_min, v_max, omega_min, omega_max)
    X = rollout_unicycle_single_nb(x0, U, dt)
    best_cost = _unicycle_ilqr_total_cost_nb(
        X, U, ref, arc, cov_blocks, covariance_floor,
        mahalanobis_weight, heading_weight, progress_weight,
        control_v_weight, control_omega_weight,
    )

    for _ in range(max(1, int(iterations))):
        kff, Kfb = _unicycle_ilqr_backward_nb(
            X, U, ref, arc, cov_blocks, covariance_floor, dt,
            mahalanobis_weight, heading_weight, progress_weight,
            control_v_weight, control_omega_weight, regularization,
        )
        improved = False
        alpha = 1.0
        for _ls in range(max(1, int(line_search_steps))):
            Utrial, Xtrial = _unicycle_ilqr_forward_update_nb(
                x0, U, X, kff, Kfb, alpha, dt, v_min, v_max, omega_min, omega_max
            )
            trial_cost = _unicycle_ilqr_total_cost_nb(
                Xtrial, Utrial, ref, arc, cov_blocks, covariance_floor,
                mahalanobis_weight, heading_weight, progress_weight,
                control_v_weight, control_omega_weight,
            )
            if math.isfinite(trial_cost) and trial_cost < best_cost - 1e-9:
                U = Utrial
                X = Xtrial
                best_cost = trial_cost
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break

    final_progress, _, _, _, _, _, _, _, _ = _project_unicycle_rollout_ilqr_nb(
        X, ref, arc, cov_blocks, covariance_floor
    )
    for t in range(horizon):
        positions[t] = final_progress[t]
    return U, positions


@njit(cache=True)
def nominal_controls_to_goal_nb(x0, goal, horizon, dt, v_min, v_max, omega_min, omega_max):
        U = np.zeros((horizon, 2), dtype=np.float64)
        px = x0[0]
        py = x0[1]
        theta = x0[2]
        for t in range(horizon):
            dx = goal[0] - px
            dy = goal[1] - py
            desired_heading = math.atan2(dy, dx)
            err = _wrap_angle_nb(desired_heading - theta)
            forward = math.cos(err)
            if forward < 0.0:
                forward = 0.0
            v = v_max * forward * forward
            if v < v_min:
                v = v_min
            omega = 3.0 * err
            if omega < omega_min:
                omega = omega_min
            elif omega > omega_max:
                omega = omega_max
            U[t, 0] = v
            U[t, 1] = omega
            px += v * math.cos(theta) * dt
            py += v * math.sin(theta) * dt
            theta = _wrap_angle_nb(theta + omega * dt)
        return U

@njit(cache=True, parallel=True)
def standard_mppi_costs_batch_nb(X, U, circle_centers, circle_radii, goal, horizon, robot_radius, w_goal, w_obstacle, w_control, w_control_smooth):
        N = U.shape[0]
        H = horizon
        M = circle_radii.shape[0]
        costs = np.zeros(N, dtype=np.float64)
        for n in prange(N):
            cost = 0.0
            for t in range(H):
                px = X[n, t + 1, 0]
                py = X[n, t + 1, 1]
                gx = px - goal[0]
                gy = py - goal[1]
                cost += w_goal / H * (gx * gx + gy * gy)
                for j in range(M):
                    dx = px - circle_centers[j, 0]
                    dy = py - circle_centers[j, 1]
                    d = math.sqrt(dx * dx + dy * dy) - circle_radii[j]
                    margin = robot_radius
                    sp = _softplus_scalar_nb(8.0 * (margin - d))
                    cost += w_obstacle * sp * sp
            ctrl_cost = 0.0
            for t in range(H):
                v = U[n, t, 0]
                om = U[n, t, 1]
                ctrl_cost += v * v + 0.15 * om * om
            cost += w_control * ctrl_cost
            smooth_cost = 0.0
            for t in range(H - 1):
                dv = U[n, t + 1, 0] - U[n, t, 0]
                dom = U[n, t + 1, 1] - U[n, t, 1]
                smooth_cost += dv * dv + 0.2 * dom * dom
            cost += w_control_smooth * smooth_cost
            costs[n] = cost
        return costs

@njit(cache=True)
def point_in_poly_nb(px, py, poly):
        inside = False
        n = poly.shape[0]
        for i in range(n):
            xi = poly[i, 0]
            yi = poly[i, 1]
            j = i + 1
            if j == n:
                j = 0
            xj = poly[j, 0]
            yj = poly[j, 1]
            if (yi > py) != (yj > py):
                x_cross = xi + (py - yi) * (xj - xi) / (yj - yi + 1e-18)
                if px < x_cross:
                    inside = not inside
        return inside

@njit(cache=True)
def point_segment_dist_nb(px, py, ax, ay, bx, by):
        abx = bx - ax
        aby = by - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-12:
            cx = ax
            cy = ay
        else:
            u = ((px - ax) * abx + (py - ay) * aby) / denom
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            cx = ax + u * abx
            cy = ay + u * aby
        dx = px - cx
        dy = py - cy
        return math.sqrt(dx * dx + dy * dy)

@njit(cache=True)
def min_clearance_nb(states, polys_padded, poly_lengths, robot_radius):
        best = 1e+18
        S = states.shape[0]
        M = poly_lengths.shape[0]
        for s in range(S):
            px = states[s, 0]
            py = states[s, 1]
            for m in range(M):
                n = poly_lengths[m]
                poly = polys_padded[m]
                min_d = 1e+18
                for i in range(n):
                    j = i + 1
                    if j == n:
                        j = 0
                    d = point_segment_dist_nb(px, py, poly[i, 0], poly[i, 1], poly[j, 0], poly[j, 1])
                    if d < min_d:
                        min_d = d
                if point_in_poly_nb(px, py, poly[:n]):
                    signed = -min_d
                else:
                    signed = min_d
                clearance = signed - robot_radius
                if clearance < best:
                    best = clearance
        return best

@njit(cache=True, parallel=True)
def sensitivity_projected_covariances_nb(x0, nominal_controls, position_covariances, lookahead_steps, fd_v, fd_omega, pseudoinverse_damping, covariance_jitter, dt, v_min, v_max, omega_min, omega_max):
        """Project planar trajectory covariance into unicycle control space."""
        horizon = nominal_controls.shape[0]
        nominal_states = rollout_unicycle_single_nb(x0, nominal_controls, dt)
        projected = np.zeros((horizon, 2, 2), dtype=np.float64)
        damping_sq = pseudoinverse_damping * pseudoinverse_damping
        for t in prange(horizon):
            interval = min(max(1, int(lookahead_steps)), horizon - t)
            jacobian = np.zeros((2, 2), dtype=np.float64)
            for control_index in range(2):
                delta = fd_v if control_index == 0 else fd_omega
                lower = v_min if control_index == 0 else omega_min
                upper = v_max if control_index == 0 else omega_max
                center = nominal_controls[t, control_index]
                plus_value = min(center + delta, upper)
                minus_value = max(center - delta, lower)
                denominator = plus_value - minus_value
                if denominator <= 1e-12:
                    continue
                plus_state = nominal_states[t].copy()
                minus_state = nominal_states[t].copy()
                for k in range(interval):
                    v_plus = nominal_controls[t + k, 0]
                    omega_plus = nominal_controls[t + k, 1]
                    v_minus = v_plus
                    omega_minus = omega_plus
                    if k == 0:
                        if control_index == 0:
                            v_plus = plus_value
                            v_minus = minus_value
                        else:
                            omega_plus = plus_value
                            omega_minus = minus_value
                    plus_theta = plus_state[2]
                    plus_state[0] += v_plus * math.cos(plus_theta) * dt
                    plus_state[1] += v_plus * math.sin(plus_theta) * dt
                    plus_state[2] = _wrap_angle_nb(plus_theta + omega_plus * dt)
                    minus_theta = minus_state[2]
                    minus_state[0] += v_minus * math.cos(minus_theta) * dt
                    minus_state[1] += v_minus * math.sin(minus_theta) * dt
                    minus_state[2] = _wrap_angle_nb(minus_theta + omega_minus * dt)
                jacobian[0, control_index] = (plus_state[0] - minus_state[0]) / denominator
                jacobian[1, control_index] = (plus_state[1] - minus_state[1]) / denominator
            a00 = jacobian[0, 0] * jacobian[0, 0] + jacobian[0, 1] * jacobian[0, 1] + damping_sq
            a01 = jacobian[0, 0] * jacobian[1, 0] + jacobian[0, 1] * jacobian[1, 1]
            a11 = jacobian[1, 0] * jacobian[1, 0] + jacobian[1, 1] * jacobian[1, 1] + damping_sq
            determinant = a00 * a11 - a01 * a01
            if determinant < 1e-18:
                determinant = 1e-18
            inv00 = a11 / determinant
            inv01 = -a01 / determinant
            inv11 = a00 / determinant
            pinv00 = jacobian[0, 0] * inv00 + jacobian[1, 0] * inv01
            pinv01 = jacobian[0, 0] * inv01 + jacobian[1, 0] * inv11
            pinv10 = jacobian[0, 1] * inv00 + jacobian[1, 1] * inv01
            pinv11 = jacobian[0, 1] * inv01 + jacobian[1, 1] * inv11
            s00 = position_covariances[t, 0, 0]
            s01 = 0.5 * (position_covariances[t, 0, 1] + position_covariances[t, 1, 0])
            s11 = position_covariances[t, 1, 1]
            q00 = pinv00 * s00 + pinv01 * s01
            q01 = pinv00 * s01 + pinv01 * s11
            q10 = pinv10 * s00 + pinv11 * s01
            q11 = pinv10 * s01 + pinv11 * s11
            c00 = q00 * pinv00 + q01 * pinv01
            c01 = q00 * pinv10 + q01 * pinv11
            c10 = q10 * pinv00 + q11 * pinv01
            c11 = q10 * pinv10 + q11 * pinv11
            projected[t, 0, 0] = c00 + covariance_jitter
            projected[t, 0, 1] = 0.5 * (c01 + c10)
            projected[t, 1, 0] = projected[t, 0, 1]
            projected[t, 1, 1] = c11 + covariance_jitter
        return projected

def _poly_vertices(obs) -> Array:
    if hasattr(obs, 'vertices'):
        return np.asarray(obs.vertices, dtype=np.float64)[:, :2]
    return np.asarray(obs, dtype=np.float64)[:, :2]

def point_in_poly(p: Array, poly: Array) -> bool:
    x, y = p
    inside = False
    n = poly.shape[0]
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i + 1) % n]
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi + 1e-18)
            if x < x_cross:
                inside = not inside
    return inside

def min_clearance(states: Array, obstacles: Sequence, robot_radius: float) -> float:
    state_array = np.asarray(states, dtype=np.float64)
    if state_array.size == 0 or not obstacles:
        return 1e309
    padded, lengths = obstacles_to_padded_arrays(obstacles)
    return float(min_clearance_nb(state_array, padded, lengths, float(robot_radius)))

def unicycle_step(x: Array, u: Array, dt: float) -> Array:
    return unicycle_step_nb(np.asarray(x, dtype=np.float64), np.asarray(u, dtype=np.float64), float(dt))

def rollout_unicycle(x0: Array, U: Array, dt: float) -> Array:
    return rollout_unicycle_single_nb(np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), float(dt))

def rollout_unicycle_batch(x0: Array, U: Array, dt: float) -> Array:
    return rollout_unicycle_batch_nb(np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), float(dt))

def obstacle_circles_to_arrays(obstacle_circles: List[Tuple[Array, float]]) -> Tuple[Array, Array]:
    if not obstacle_circles:
        return (np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64))
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    return (centers, radii)

def apply_smooth_safe_control(x_current: Array, u: Array, previous_control: Optional[Array], obstacle_circles: List[Tuple[Array, float]], cfg: MPPIConfig) -> Array:
    cmd = np.asarray(u, dtype=np.float64).copy()
    if previous_control is not None:
        dv = float(np.clip(cmd[0] - previous_control[0], -cfg.max_delta_v, cfg.max_delta_v))
        domega = float(np.clip(cmd[1] - previous_control[1], -cfg.max_delta_omega, cfg.max_delta_omega))
        cmd[0] = previous_control[0] + dv
        cmd[1] = previous_control[1] + domega
    cmd[0] = np.clip(cmd[0], cfg.v_min, cfg.v_max)
    cmd[1] = np.clip(cmd[1], cfg.omega_min, cfg.omega_max)
    if cfg.enforce_one_step_safety and obstacle_circles:
        x_next = unicycle_step(x_current, cmd, cfg.dt)
        centers, radii = obstacle_circles_to_arrays(obstacle_circles)
        current_clearance = float(np.min(np.linalg.norm(x_current[None, :2] - centers, axis=1) - radii - cfg.robot_radius))
        next_clearance = float(np.min(np.linalg.norm(x_next[None, :2] - centers, axis=1) - radii - cfg.robot_radius))
        moving_deeper = next_clearance < current_clearance - 0.0001
        below_required_clearance = next_clearance < cfg.one_step_safety_clearance
        if below_required_clearance and moving_deeper:
            cmd[0] = 0.0
    return cmd

def path_min_clearance_to_circles(path: Array, obstacle_circles: List[Tuple[Array, float]], robot_radius: float, substeps: int=2) -> float:
    p = np.asarray(path, dtype=np.float64)
    if len(p) == 0 or not obstacle_circles:
        return 1e309
    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    samples = [p]
    if len(p) > 1:
        for q in range(1, max(0, int(substeps)) + 1):
            alpha = q / float(max(0, int(substeps)) + 1)
            samples.append(p[:-1] + alpha * (p[1:] - p[:-1]))
    points = np.vstack(samples)
    clearance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2) - radii[None, :] - float(robot_radius)
    return float(np.min(clearance))

def obstacles_to_padded_arrays(obstacles: Sequence) -> Tuple[Array, Array]:
    polys = [_poly_vertices(o).astype(np.float64) for o in obstacles]
    if not polys:
        return (np.zeros((0, 0, 2), dtype=np.float64), np.zeros(0, dtype=np.int64))
    max_n = max((p.shape[0] for p in polys))
    padded = np.zeros((len(polys), max_n, 2), dtype=np.float64)
    lengths = np.zeros(len(polys), dtype=np.int64)
    for i, p in enumerate(polys):
        padded[i, :p.shape[0], :] = p
        lengths[i] = p.shape[0]
    return (padded, lengths)

def _prepare_ilqr_covariance(ref: Array, cov_blocks: Optional[Array], cfg) -> Array:
    path = np.asarray(ref, dtype=np.float64)
    n = len(path)
    if cov_blocks is None:
        var = float(cfg.prior_ilqr_covariance_fallback_std) ** 2
        cov = np.zeros((n, 2, 2), dtype=np.float64)
        cov[:, 0, 0] = var
        cov[:, 1, 1] = var
        return np.ascontiguousarray(cov)
    cov = np.asarray(cov_blocks, dtype=np.float64)
    if cov.shape != (n, 2, 2):
        raise ValueError(f"cov_blocks must have shape ({n},2,2), got {cov.shape}")
    cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))
    return np.ascontiguousarray(cov)


def nominal_controls_and_arc_positions(
    x0: Array, ref: Array, cfg, cov_blocks: Optional[Array] = None
) -> Tuple[Array, Array]:
    path = np.asarray(ref, dtype=np.float64)
    cov = _prepare_ilqr_covariance(path, cov_blocks, cfg)
    return _unicycle_ilqr_nominal_and_positions_nb(
        np.asarray(x0, dtype=np.float64), path, cov, int(cfg.horizon),
        float(cfg.dt), float(cfg.v_min), float(cfg.v_max), float(cfg.omega_min), float(cfg.omega_max),
        int(cfg.prior_ilqr_iterations), int(cfg.prior_ilqr_line_search_steps),
        float(cfg.prior_ilqr_mahalanobis_weight), float(cfg.prior_ilqr_covariance_floor),
        float(cfg.prior_ilqr_heading_weight), float(cfg.prior_ilqr_progress_weight),
        float(cfg.prior_ilqr_control_v_weight), float(cfg.prior_ilqr_control_omega_weight),
        float(cfg.prior_ilqr_regularization),
    )


def prior_control_arc_positions(
    x0: Array, ref: Array, cfg, cov_blocks: Optional[Array] = None
) -> Array:
    _, positions = nominal_controls_and_arc_positions(x0, ref, cfg, cov_blocks)
    return positions


def nominal_controls_to_track_path(
    x0: Array, ref: Array, cfg, cov_blocks: Optional[Array] = None
) -> Array:
    controls, _ = nominal_controls_and_arc_positions(x0, ref, cfg, cov_blocks)
    return controls


def nominal_controls_to_goal(x0: Array, goal: Array, cfg) -> Array:
    return nominal_controls_to_goal_nb(
        np.asarray(x0, dtype=np.float64), np.asarray(goal, dtype=np.float64), int(cfg.horizon),
        float(cfg.dt), float(cfg.v_min), float(cfg.v_max), float(cfg.omega_min), float(cfg.omega_max),
    )

def standard_mppi_costs_batch(X: Array, U: Array, obstacle_circles: List[Tuple[Array, float]], goal: Array, cfg: MPPIConfig) -> Array:
    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    return standard_mppi_costs_batch_nb(
        np.asarray(X, dtype=np.float64), np.asarray(U, dtype=np.float64), centers, radii,
        np.asarray(goal, dtype=np.float64), int(cfg.horizon), float(cfg.robot_radius),
        float(cfg.w_goal), float(cfg.w_obstacle), float(cfg.w_control), float(cfg.w_control_smooth),
    )

def stable_representation_costs(X: Array, U: Array, obstacle_circles: List[Tuple[Array, float]], goal: Array, cfg: MPPIConfig) -> Array:
    return standard_mppi_costs_batch(X, U, obstacle_circles, goal, cfg)

def enforce_forward_curve_proposals(U: Array, cfg: MPPIConfig) -> Array:
    U = np.asarray(U, dtype=np.float64)
    if U.size == 0:
        return U
    U[:, :, 0] = np.clip(U[:, :, 0], cfg.v_min, cfg.v_max)
    U[:, :, 1] = np.clip(U[:, :, 1], cfg.omega_min, cfg.omega_max)
    return U

def sensitivity_projected_control_covariances(x0: Array, nominal_controls: Array, position_covariances: Array, cfg: MPPIConfig) -> Array:
    """Compute Eq. (26) using the Numba sensitivity-projection kernel."""
    nominal = np.asarray(nominal_controls, dtype=np.float64)
    covariances = np.asarray(position_covariances, dtype=np.float64)
    horizon = int(nominal.shape[0])
    if nominal.shape != (horizon, 2):
        raise ValueError(f'nominal_controls must have shape (H,2), got {nominal.shape}')
    if covariances.shape != (horizon, 2, 2):
        raise ValueError(f'position_covariances must have shape (H,2,2), got {covariances.shape}')
    return sensitivity_projected_covariances_nb(
        np.asarray(x0, dtype=np.float64), nominal, covariances,
        int(cfg.spg_lookahead_steps), float(cfg.spg_fd_accel), float(cfg.spg_fd_steering_rate),
        float(cfg.spg_pseudoinverse_damping), float(cfg.spg_covariance_jitter),
        float(cfg.dt), float(cfg.v_min), float(cfg.v_max), float(cfg.omega_min), float(cfg.omega_max),
    )

@njit(cache=True, parallel=True)
def rollout_collision_mask_nb(X, circle_centers, circle_radii, robot_radius, hard_collision_clearance):
    """Parallel early-exit collision test over independent MPPI rollouts."""
    N = X.shape[0]
    H = X.shape[1] - 1
    M = circle_radii.shape[0]
    mask = np.zeros(N, dtype=np.bool_)
    for n in prange(N):
        colliding = False
        for t in range(H):
            px = X[n, t + 1, 0]
            py = X[n, t + 1, 1]
            for j in range(M):
                dx = px - circle_centers[j, 0]
                dy = py - circle_centers[j, 1]
                clearance = math.sqrt(dx * dx + dy * dy) - circle_radii[j] - robot_radius
                if clearance < hard_collision_clearance:
                    colliding = True
                    break
            if colliding:
                break
        mask[n] = colliding
    return mask


def rollout_collision_mask(X: Array, obstacle_circles: Sequence[Tuple[Array, float]], cfg: MPPIConfig) -> Array:
    if not obstacle_circles or X.shape[0] == 0:
        return np.zeros(X.shape[0], dtype=bool)
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    return rollout_collision_mask_nb(
        np.asarray(X, dtype=np.float64),
        centers,
        radii,
        float(cfg.robot_radius),
        float(cfg.hard_collision_clearance),
    )

def update_display_trajectory(info: Dict[str, object], x_current: Array, executed_u: Array, cfg: MPPIConfig) -> None:
    sequence = info.get('planned_control_sequence')
    if sequence is None:
        return
    display_u = np.asarray(sequence, dtype=np.float64).copy()
    if display_u.ndim != 2 or display_u.shape[1] != 2 or len(display_u) == 0:
        return
    display_u[0] = np.asarray(executed_u, dtype=np.float64)
    info['optimal_traj'] = rollout_unicycle(x_current, display_u, cfg.dt)

def initial_pose(start: Array, goal: Array) -> Array:
    direction = goal - start
    heading = math.atan2(direction[1], direction[0])
    return np.array([start[0], start[1], heading], dtype=np.float64)

def control_noise_scale(cfg: MPPIConfig) -> Array:
    return np.asarray([cfg.noise_v, cfg.noise_omega], dtype=np.float64)


def clip_control_batch(controls: Array, cfg: MPPIConfig) -> Array:
    return enforce_forward_curve_proposals(np.asarray(controls, dtype=np.float64), cfg)


def rollout_batch(x_current: Array, controls: Array, cfg: MPPIConfig) -> Array:
    return rollout_unicycle_batch(x_current, controls, cfg.dt)


def rollout_single(x_current: Array, controls: Array, cfg: MPPIConfig) -> Array:
    return rollout_unicycle(x_current, controls, cfg.dt)


def project_control_covariances(x_current: Array, nominal: Array, covariances: Array, cfg: MPPIConfig) -> Array:
    return sensitivity_projected_control_covariances(x_current, nominal, covariances, cfg)


def trajectory_costs(states: Array, controls: Array, obstacle_circles, goal: Array, cfg: MPPIConfig) -> Array:
    return stable_representation_costs(states, controls, obstacle_circles, goal, cfg)


def collision_mask(states: Array, obstacle_circles, goal: Array, cfg: MPPIConfig) -> Array:
    del goal
    return rollout_collision_mask(states, obstacle_circles, cfg)


def mean_path_clearance(path: Array, obstacle_circles, cfg: MPPIConfig) -> float:
    return path_min_clearance_to_circles(
        path,
        obstacle_circles,
        cfg.robot_radius,
        substeps=cfg.mode_blocking_substeps,
    )


def apply_final_output(
    x_current: Array,
    control: Array,
    previous_control: Optional[Array],
    obstacle_circles,
    goal: Array,
    cfg: MPPIConfig,
) -> Array:
    del goal
    return apply_smooth_safe_control(x_current, control, previous_control, obstacle_circles, cfg)


def render_output_trajectory(info, x_current: Array, control: Array, goal: Array, cfg: MPPIConfig) -> None:
    del goal
    update_display_trajectory(info, x_current, control, cfg)


def goal_reached(state: Array, goal: Array, cfg: MPPIConfig) -> bool:
    return bool(np.linalg.norm(np.asarray(state[:2]) - np.asarray(goal)) <= cfg.goal_tolerance)


def advance_state(state: Array, control: Array, goal: Array, cfg: MPPIConfig) -> Tuple[Array, bool]:
    next_state = unicycle_step(state, control, cfg.dt)
    return next_state, goal_reached(next_state, goal, cfg)


def minimum_clearance(states: Array, obstacles: Sequence, cfg: MPPIConfig) -> float:
    return min_clearance(states, obstacles, cfg.robot_radius)


SUPPORTED_VARIANTS = set(ControllerVariant)

build_default_scene = ctrl.build_default_scene
default_dynamic_wall_scenarios = ctrl.default_dynamic_wall_scenarios
obstacle_bounding_circles = ctrl.obstacle_bounding_circles
obstacle_center = ctrl.obstacle_center
make_wall_blockers_between_centers = ctrl.make_wall_blockers_between_centers
localize_mode_for_state = ctrl.localize_mode_for_state
localize_path_for_state = ctrl.localize_path_for_state


def build_homotopy_modes(scene: Scene, obstacles: Sequence, seed: int):
    return ctrl.build_homotopy_modes(scene, obstacles, seed)


def run_controller(variant, modes, base_obstacles, blockers, scene, **kwargs):
    return ctrl.run_controller(sys.modules[__name__], variant, modes, base_obstacles, blockers, scene, **kwargs)