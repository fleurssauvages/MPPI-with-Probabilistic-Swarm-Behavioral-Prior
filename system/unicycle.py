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

from numba import njit

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
    v_min: float = -1.0
    v_max: float = 2.8
    omega_min: float = -4.5
    omega_max: float = 4.5
    noise_v: float = 0.5
    noise_omega: float = 0.9

    # Geometric-prior to unicycle-control conversion.
    prior_reference_speed: float = 2.0
    prior_tracking_heading_gain: float = 2.5
    prior_tracking_lateral_gain: float = 1.0
    prior_tracking_terminal_distance_gain: float = 1.8
    # Intercept the geometric prior before switching to local Frenet tracking.
    prior_intercept_lateral_threshold: float = 0.60
    prior_intercept_heading_threshold: float = 0.65
    prior_intercept_lookahead: float = 0.90
    prior_intercept_heading_gain: float = 2.8

    max_delta_v: float = 0.7
    max_delta_omega: float = 1.4
    enforce_one_step_safety: bool = True
    one_step_safety_clearance: float = 0.0

    terminal_slowdown_radius: float = 0.75
    terminal_max_speed: float = 0.55
    terminal_distance_gain: float = 1.8
    terminal_heading_gain: float = 1.2
    terminal_heading_deadzone: float = 0.45
    terminal_blend_power: float = 1.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.v_min > self.v_max or self.omega_min > self.omega_max:
            raise ValueError("Invalid unicycle control bounds.")


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


@njit(cache=True)
def rollout_unicycle_batch_nb(x0, U, dt):
    N = U.shape[0]
    H = U.shape[1]
    X = np.zeros((N, H + 1, 3), dtype=np.float64)
    for n in range(N):
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
def nominal_controls_to_track_path_nb(
    x0, ref, horizon, reference_speed, heading_gain, lateral_gain,
    terminal_distance_gain, intercept_lateral_threshold,
    intercept_heading_threshold, intercept_lookahead, intercept_heading_gain,
    dt, v_min, v_max, omega_min, omega_max
):
    U = np.zeros((horizon, 2), dtype=np.float64)
    px = x0[0]
    py = x0[1]
    theta = x0[2]
    ref_len = ref.shape[0]
    progress_idx = 0

    terminal_endpoint = False
    if ref_len >= 2:
        ex = ref[ref_len - 1, 0] - ref[ref_len - 2, 0]
        ey = ref[ref_len - 1, 1] - ref[ref_len - 2, 1]
        terminal_endpoint = ex * ex + ey * ey <= 1e-12

    for t in range(horizon):
        if ref_len < 2:
            U[t, 0] = 0.0
            U[t, 1] = 0.0
            continue

        first_seg = min(progress_idx, ref_len - 2)
        best_seg = first_seg
        best_d2 = 1e300
        best_qx = ref[first_seg, 0]
        best_qy = ref[first_seg, 1]
        for i in range(first_seg, ref_len - 1):
            sx = ref[i + 1, 0] - ref[i, 0]
            sy = ref[i + 1, 1] - ref[i, 1]
            seg2 = sx * sx + sy * sy
            if seg2 <= 1e-14:
                qx = ref[i, 0]
                qy = ref[i, 1]
            else:
                tau = ((px - ref[i, 0]) * sx + (py - ref[i, 1]) * sy) / seg2
                tau = min(max(tau, 0.0), 1.0)
                qx = ref[i, 0] + tau * sx
                qy = ref[i, 1] + tau * sy
            dxq = px - qx
            dyq = py - qy
            d2 = dxq * dxq + dyq * dyq
            if d2 < best_d2:
                best_d2 = d2
                best_seg = i
                best_qx = qx
                best_qy = qy
        progress_idx = best_seg

        tangent_seg = best_seg
        sx = ref[tangent_seg + 1, 0] - ref[tangent_seg, 0]
        sy = ref[tangent_seg + 1, 1] - ref[tangent_seg, 1]
        seg_len = math.sqrt(sx * sx + sy * sy)
        if seg_len <= 1e-12:
            found = False
            for i in range(best_seg + 1, ref_len - 1):
                tx0 = ref[i + 1, 0] - ref[i, 0]
                ty0 = ref[i + 1, 1] - ref[i, 1]
                ll = math.sqrt(tx0 * tx0 + ty0 * ty0)
                if ll > 1e-12:
                    tangent_seg = i
                    sx = tx0
                    sy = ty0
                    seg_len = ll
                    found = True
                    break
            if not found:
                for i in range(best_seg - 1, -1, -1):
                    tx0 = ref[i + 1, 0] - ref[i, 0]
                    ty0 = ref[i + 1, 1] - ref[i, 1]
                    ll = math.sqrt(tx0 * tx0 + ty0 * ty0)
                    if ll > 1e-12:
                        tangent_seg = i
                        sx = tx0
                        sy = ty0
                        seg_len = ll
                        break

        tx = sx / max(seg_len, 1e-12)
        ty = sy / max(seg_len, 1e-12)
        path_heading = math.atan2(ty, tx)
        lateral_error = -ty * (px - best_qx) + tx * (py - best_qy)
        heading_error = _wrap_angle_nb(theta - path_heading)
        cross_track_distance = math.sqrt(max(best_d2, 0.0))

        intercept_mode = (
            cross_track_distance > intercept_lateral_threshold
            or abs(heading_error) > intercept_heading_threshold
        )

        if intercept_mode:
            target_x, target_y = _path_intercept_point_nb(
                ref, best_seg, best_qx, best_qy, intercept_lookahead
            )
            dx = target_x - px
            dy = target_y - py
            capture_heading = math.atan2(dy, dx)
            capture_error = _wrap_angle_nb(capture_heading - theta)
            forward = max(0.0, math.cos(capture_error))
            desired_speed = min(max(reference_speed * forward * forward, 0.0), v_max)
            omega = intercept_heading_gain * capture_error
        else:
            curvature = 0.0
            if ref_len >= 3:
                center = min(max(tangent_seg, 1), ref_len - 2)
                x0p = ref[center - 1, 0]
                y0p = ref[center - 1, 1]
                x1p = ref[center, 0]
                y1p = ref[center, 1]
                x2p = ref[center + 1, 0]
                y2p = ref[center + 1, 1]
                a0x = x1p - x0p
                a0y = y1p - y0p
                b0x = x2p - x0p
                b0y = y2p - y0p
                l01 = math.sqrt(a0x * a0x + a0y * a0y)
                d12x = x2p - x1p
                d12y = y2p - y1p
                l12 = math.sqrt(d12x * d12x + d12y * d12y)
                l02 = math.sqrt(b0x * b0x + b0y * b0y)
                denom = l01 * l12 * l02
                if denom > 1e-12:
                    cross = a0x * b0y - a0y * b0x
                    curvature = 2.0 * cross / denom
                if abs(curvature) < 1e-4:
                    curvature = 0.0

            forward = max(0.0, math.cos(heading_error))
            desired_speed = min(max(reference_speed * forward * forward, 0.0), v_max)
            omega = desired_speed * curvature - heading_gain * heading_error - lateral_gain * lateral_error

        if terminal_endpoint:
            remaining = math.sqrt(
                (ref[best_seg + 1, 0] - best_qx) ** 2
                + (ref[best_seg + 1, 1] - best_qy) ** 2
            )
            for i in range(best_seg + 1, ref_len - 1):
                rx = ref[i + 1, 0] - ref[i, 0]
                ry = ref[i + 1, 1] - ref[i, 1]
                remaining += math.sqrt(rx * rx + ry * ry)
            desired_speed = min(desired_speed, max(0.0, terminal_distance_gain * remaining))

        if omega < omega_min:
            omega = omega_min
        elif omega > omega_max:
            omega = omega_max
        if desired_speed < v_min:
            desired_speed = v_min

        U[t, 0] = desired_speed
        U[t, 1] = omega
        px += desired_speed * math.cos(theta) * dt
        py += desired_speed * math.sin(theta) * dt
        theta = _wrap_angle_nb(theta + omega * dt)
    return U


@njit(cache=True)
def nominal_controls_to_track_paths_batch_nb(
    x0, refs, horizon, reference_speed, heading_gain, lateral_gain,
    terminal_distance_gain, intercept_lateral_threshold,
    intercept_heading_threshold, intercept_lookahead, intercept_heading_gain,
    dt, v_min, v_max, omega_min, omega_max
):
    count = refs.shape[0]
    output = np.empty((count, horizon, 2), dtype=np.float64)
    for n in range(count):
        controls = nominal_controls_to_track_path_nb(
            x0, refs[n], horizon, reference_speed, heading_gain, lateral_gain,
            terminal_distance_gain, intercept_lateral_threshold,
            intercept_heading_threshold, intercept_lookahead, intercept_heading_gain,
            dt, v_min, v_max, omega_min, omega_max,
        )
        for t in range(horizon):
            output[n, t, 0] = controls[t, 0]
            output[n, t, 1] = controls[t, 1]
    return output

@njit(cache=True)
def nominal_controls_to_goal_nb(x0, goal, horizon, dt, v_min, v_max, omega_min, omega_max):
        U = np.zeros((horizon, 2), dtype=np.float64)
        px = x0[0]
        py = x0[1]
        theta = x0[2]
        for t in range(horizon):
            dx = goal[0] - px
            dy = goal[1] - py
            dist = math.sqrt(dx * dx + dy * dy)
            desired_heading = math.atan2(dy, dx)
            err = _wrap_angle_nb(desired_heading - theta)
            forward = math.cos(err)
            if forward < 0.0:
                forward = 0.0
            heading_scale = forward * forward
            v = 0.2 + 2.2 * dist * heading_scale
            if v < v_min:
                v = v_min
            elif v > v_max:
                v = v_max
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

@njit(cache=True)
def standard_mppi_costs_batch_nb(X, U, circle_centers, circle_radii, goal, horizon, robot_radius, w_goal, w_obstacle, w_control, w_control_smooth):
        N = U.shape[0]
        H = horizon
        M = circle_radii.shape[0]
        costs = np.zeros(N, dtype=np.float64)
        for n in range(N):
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

@njit(cache=True)
def sensitivity_projected_covariances_nb(x0, nominal_controls, position_covariances, lookahead_steps, fd_v, fd_omega, pseudoinverse_damping, covariance_jitter, dt, v_min, v_max, omega_min, omega_max):
        """Project planar trajectory covariance into unicycle control space."""
        horizon = nominal_controls.shape[0]
        nominal_states = rollout_unicycle_single_nb(x0, nominal_controls, dt)
        projected = np.zeros((horizon, 2, 2), dtype=np.float64)
        damping_sq = pseudoinverse_damping * pseudoinverse_damping
        for t in range(horizon):
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

def point_segment_distance_and_normal(p: Array, a: Array, b: Array) -> Tuple[float, Array]:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        closest = a
    else:
        u = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
        closest = a + u * ab
    dvec = p - closest
    dist = float(np.linalg.norm(dvec))
    normal = np.array([1.0, 0.0]) if dist <= 1e-12 else dvec / dist
    return (dist, normal)

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

def polygon_signed_distance_and_normal(p: Array, obs) -> Tuple[float, Array]:
    poly = _poly_vertices(obs)
    best_dist = 1e309
    best_normal = np.array([1.0, 0.0])
    for i in range(poly.shape[0]):
        d, n = point_segment_distance_and_normal(p, poly[i], poly[(i + 1) % poly.shape[0]])
        if d < best_dist:
            best_dist = d
            best_normal = n
    if point_in_poly(p, poly):
        return (-best_dist, best_normal)
    return (best_dist, best_normal)

def min_clearance(states: Array, obstacles: Sequence, robot_radius: float) -> float:
    state_array = np.asarray(states, dtype=np.float64)
    if state_array.size == 0 or not obstacles:
        return 1e309
    padded, lengths = obstacles_to_padded_arrays(obstacles)
    return float(min_clearance_nb(state_array, padded, lengths, float(robot_radius)))

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def unicycle_step(x: Array, u: Array, dt: float) -> Array:
    return unicycle_step_nb(np.asarray(x, dtype=np.float64), np.asarray(u, dtype=np.float64), float(dt))

def segment_goal_entry_state(x0: Array, x1: Array, goal: Array, goal_tolerance: float) -> Tuple[bool, Array]:
    p0 = np.asarray(x0[:2], dtype=np.float64)
    p1 = np.asarray(x1[:2], dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    r = float(goal_tolerance)
    if np.linalg.norm(p0 - g) <= r:
        return (True, np.asarray(x0, dtype=np.float64).copy())
    if np.linalg.norm(p1 - g) <= r:
        return (True, np.asarray(x1, dtype=np.float64).copy())
    d = p1 - p0
    a = float(d @ d)
    if a <= 1e-16:
        return (False, np.asarray(x1, dtype=np.float64).copy())
    f = p0 - g
    b = 2.0 * float(f @ d)
    c = float(f @ f) - r * r
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return (False, np.asarray(x1, dtype=np.float64).copy())
    root = math.sqrt(max(0.0, disc))
    roots = [q for q in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if 0.0 <= q <= 1.0]
    if not roots:
        return (False, np.asarray(x1, dtype=np.float64).copy())
    alpha = float(min(roots))
    hit = np.asarray(x0, dtype=np.float64).copy()
    hit[:2] = p0 + alpha * d
    hit[2] = wrap_angle(float(x0[2]) + alpha * wrap_angle(float(x1[2]) - float(x0[2])))
    return (True, hit)

def rollout_unicycle(x0: Array, U: Array, dt: float) -> Array:
    return rollout_unicycle_single_nb(np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), float(dt))

def rollout_unicycle_batch(x0: Array, U: Array, dt: float) -> Array:
    return rollout_unicycle_batch_nb(np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), float(dt))

def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

def obstacle_circles_to_arrays(obstacle_circles: List[Tuple[Array, float]]) -> Tuple[Array, Array]:
    if not obstacle_circles:
        return (np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64))
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    return (centers, radii)

def apply_terminal_goal_approach(x_current: Array, u: Array, goal: Array, goal_tolerance: float, cfg: MPPIConfig) -> Array:
    cmd = np.asarray(u, dtype=np.float64).copy()
    dx = float(goal[0] - x_current[0])
    dy = float(goal[1] - x_current[1])
    distance = math.hypot(dx, dy)
    radius = max(float(cfg.terminal_slowdown_radius), goal_tolerance + 1e-06)
    if distance >= radius:
        return cmd
    desired_heading = math.atan2(dy, dx)
    heading_error = wrap_angle(desired_heading - float(x_current[2]))
    remaining = max(0.0, distance)
    heading_scale = max(0.0, math.cos(heading_error)) ** 2
    terminal_v = min(float(cfg.terminal_max_speed), float(cfg.terminal_distance_gain) * remaining) * heading_scale
    heading_fade = float(np.clip((distance - goal_tolerance) / max(float(cfg.terminal_heading_deadzone), 1e-06), 0.0, 1.0))
    terminal_omega = heading_fade * float(np.clip(float(cfg.terminal_heading_gain) * heading_error, cfg.omega_min, cfg.omega_max))
    normalized = np.clip((radius - distance) / max(radius - goal_tolerance, 1e-06), 0.0, 1.0)
    blend = float(normalized ** max(float(cfg.terminal_blend_power), 1e-06))
    cmd[0] = (1.0 - blend) * cmd[0] + blend * terminal_v
    cmd[1] = (1.0 - blend) * cmd[1] + blend * terminal_omega
    cmd[0] = max(0.0, cmd[0])
    return cmd

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

def nominal_controls_to_track_path(x0: Array, ref: Array, cfg) -> Array:
    return nominal_controls_to_track_path_nb(
        np.asarray(x0, dtype=np.float64), np.asarray(ref, dtype=np.float64), int(cfg.horizon),
        float(cfg.prior_reference_speed), float(cfg.prior_tracking_heading_gain),
        float(cfg.prior_tracking_lateral_gain), float(cfg.prior_tracking_terminal_distance_gain),
        float(cfg.prior_intercept_lateral_threshold),
        float(cfg.prior_intercept_heading_threshold),
        float(cfg.prior_intercept_lookahead),
        float(cfg.prior_intercept_heading_gain),
        float(cfg.dt), float(cfg.v_min), float(cfg.v_max), float(cfg.omega_min), float(cfg.omega_max),
    )

def nominal_controls_to_track_paths(x0: Array, refs: Array, cfg) -> Array:
    reference_batch = np.asarray(refs, dtype=np.float64)
    if reference_batch.ndim != 3 or reference_batch.shape[1:] != (int(cfg.horizon), 2):
        raise ValueError(f"refs must have shape (N,{int(cfg.horizon)},2), got {reference_batch.shape}")
    return nominal_controls_to_track_paths_batch_nb(
        np.asarray(x0, dtype=np.float64), reference_batch, int(cfg.horizon),
        float(cfg.prior_reference_speed), float(cfg.prior_tracking_heading_gain),
        float(cfg.prior_tracking_lateral_gain), float(cfg.prior_tracking_terminal_distance_gain),
        float(cfg.prior_intercept_lateral_threshold),
        float(cfg.prior_intercept_heading_threshold),
        float(cfg.prior_intercept_lookahead),
        float(cfg.prior_intercept_heading_gain),
        float(cfg.dt), float(cfg.v_min), float(cfg.v_max), float(cfg.omega_min), float(cfg.omega_max),
    )

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

def rollout_collision_mask(X: Array, obstacle_circles: Sequence[Tuple[Array, float]], cfg: MPPIConfig) -> Array:
    if not obstacle_circles or X.shape[0] == 0:
        return np.zeros(X.shape[0], dtype=bool)
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    points = np.asarray(X[:, 1:, :2], dtype=np.float64)
    delta = points[:, :, None, :] - centers[None, None, :, :]
    clearance = np.linalg.norm(delta, axis=-1) - radii[None, None, :] - float(cfg.robot_radius)
    return np.any(clearance < float(cfg.hard_collision_clearance), axis=(1, 2))

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

# Generic controller adapter -------------------------------------------------

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
    command = apply_terminal_goal_approach(x_current, control, goal, cfg.goal_tolerance, cfg)
    return apply_smooth_safe_control(x_current, command, previous_control, obstacle_circles, cfg)


def render_output_trajectory(info, x_current: Array, control: Array, goal: Array, cfg: MPPIConfig) -> None:
    del goal
    update_display_trajectory(info, x_current, control, cfg)


def goal_reached(state: Array, goal: Array, cfg: MPPIConfig) -> bool:
    return bool(np.linalg.norm(np.asarray(state[:2]) - np.asarray(goal)) <= cfg.goal_tolerance)


def advance_state(state: Array, control: Array, goal: Array, cfg: MPPIConfig) -> Tuple[Array, bool]:
    next_state = unicycle_step(state, control, cfg.dt)
    arrived, state_at_goal = segment_goal_entry_state(state, next_state, goal, cfg.goal_tolerance)
    return (state_at_goal if arrived else next_state), bool(arrived)


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