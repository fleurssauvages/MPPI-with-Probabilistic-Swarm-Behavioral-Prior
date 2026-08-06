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

try:
    from numba import njit, prange
except Exception:
    njit = None

Array = np.ndarray
NUMBA_AVAILABLE = njit is not None
MODEL_NAME = "unicycle"

ControllerVariant = ctrl.ControllerVariant
Scene = ctrl.Scene
SimulationResult = ctrl.SimulationResult
DynamicWallScenario = ctrl.DynamicWallScenario
MPPIHomotopyMode = ctrl.MPPIHomotopyMode


@dataclass
class MPPIConfig(ctrl.ControllerConfig):
    gaussian_covariance_scale: float = 4.0
    v_min: float = -1.0
    v_max: float = 2.8
    omega_min: float = -4.5
    omega_max: float = 4.5
    noise_v: float = 0.5
    noise_omega: float = 0.9

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


if njit is not None:
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

    @njit(cache=True, parallel=True)
    def rollout_unicycle_batch_nb(x0, U, dt):
            N = U.shape[0]
            H = U.shape[1]
            X = np.zeros((N, H + 1, 3), dtype=np.float64)
            for n in prange(N):
                X[n, 0, 0] = x0[0]
                X[n, 0, 1] = x0[1]
                X[n, 0, 2] = x0[2]
            for n in prange(N):
                for t in range(H):
                    th = X[n, t, 2]
                    v = U[n, t, 0]
                    om = U[n, t, 1]
                    X[n, t + 1, 0] = X[n, t, 0] + v * math.cos(th) * dt
                    X[n, t + 1, 1] = X[n, t, 1] + v * math.sin(th) * dt
                    X[n, t + 1, 2] = _wrap_angle_nb(X[n, t, 2] + om * dt)
            return X

    @njit(cache=True)
    def rollout_unicycle_single_nb(x0, U, dt):
            H = U.shape[0]
            X = np.zeros((H + 1, 3), dtype=np.float64)
            X[0, 0] = x0[0]
            X[0, 1] = x0[1]
            X[0, 2] = x0[2]
            for t in range(H):
                th = X[t, 2]
                v = U[t, 0]
                om = U[t, 1]
                X[t + 1, 0] = X[t, 0] + v * math.cos(th) * dt
                X[t + 1, 1] = X[t, 1] + v * math.sin(th) * dt
                X[t + 1, 2] = _wrap_angle_nb(X[t, 2] + om * dt)
            return X

    @njit(cache=True)
    def nominal_controls_to_track_path_nb(x0, ref, horizon, dt, v_min, v_max, omega_min, omega_max):
            U = np.zeros((horizon, 2), dtype=np.float64)
            px = x0[0]
            py = x0[1]
            theta = x0[2]
            ref_len = ref.shape[0]
            for t in range(horizon):
                target_idx = t + 3
                if target_idx >= ref_len:
                    target_idx = ref_len - 1
                dx = ref[target_idx, 0] - px
                dy = ref[target_idx, 1] - py
                dist = math.sqrt(dx * dx + dy * dy)
                desired_heading = math.atan2(dy, dx)
                err = _wrap_angle_nb(desired_heading - theta)
                forward = math.cos(err)
                if forward < 0.0:
                    forward = 0.0
                heading_scale = forward * forward
                v = 0.2 + 2.4 * dist * heading_scale
                if v < v_min:
                    v = v_min
                elif v > v_max:
                    v = v_max
                omega = 3.2 * err
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
    def rollout_collision_mask_nb(X, circle_centers, circle_radii, robot_radius, hard_collision_clearance):
        mask = np.zeros(X.shape[0], dtype=np.bool_)
        for n in prange(X.shape[0]):
            collided = False
            for t in range(1, X.shape[1]):
                px = X[n, t, 0]
                py = X[n, t, 1]
                for obstacle in range(circle_radii.shape[0]):
                    dx = px - circle_centers[obstacle, 0]
                    dy = py - circle_centers[obstacle, 1]
                    clearance = math.sqrt(dx * dx + dy * dy) - circle_radii[obstacle] - robot_radius
                    if clearance < hard_collision_clearance:
                        collided = True
                        break
                if collided:
                    break
            mask[n] = collided
        return mask

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

    @njit(cache=True)
    def centerline_connection_clearance_nb(start_point, end_point, circle_centers, circle_radii, robot_radius):
        if circle_radii.shape[0] == 0:
            return 1e18
        ax = start_point[0]
        ay = start_point[1]
        bx = end_point[0]
        by = end_point[1]
        abx = bx - ax
        aby = by - ay
        denominator = abx * abx + aby * aby
        best = 1e18
        for index in range(circle_radii.shape[0]):
            if denominator <= 1e-16:
                alpha = 0.0
            else:
                alpha = ((circle_centers[index, 0] - ax) * abx + (circle_centers[index, 1] - ay) * aby) / denominator
                alpha = min(1.0, max(0.0, alpha))
            qx = ax + alpha * abx
            qy = ay + alpha * aby
            dx = circle_centers[index, 0] - qx
            dy = circle_centers[index, 1] - qy
            clearance = math.sqrt(dx * dx + dy * dy) - circle_radii[index] - robot_radius
            if clearance < best:
                best = clearance
        return best
else:
    _wrap_angle_nb = None
    _softplus_scalar_nb = None
    rollout_unicycle_batch_nb = None
    rollout_unicycle_single_nb = None
    nominal_controls_to_track_path_nb = None
    nominal_controls_to_goal_nb = None
    standard_mppi_costs_batch_nb = None
    point_in_poly_nb = None
    point_segment_dist_nb = None
    min_clearance_nb = None
    rollout_collision_mask_nb = None
    sensitivity_projected_covariances_nb = None
    centerline_connection_clearance_nb = None


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
    if min_clearance_nb is not None:
        padded, lengths = obstacles_to_padded_arrays(obstacles)
        return float(min_clearance_nb(np.asarray(states, dtype=np.float64), padded, lengths, float(robot_radius)))
    vals = []
    for x in states:
        p = x[:2]
        for obs in obstacles:
            d, _ = polygon_signed_distance_and_normal(p, obs)
            vals.append(d - robot_radius)
    return float(np.min(vals)) if vals else 1e309

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def unicycle_step(x: Array, u: Array, dt: float) -> Array:
    px, py, th = x
    v, om = u
    return np.array([px + v * math.cos(th) * dt, py + v * math.sin(th) * dt, wrap_angle(th + om * dt)], dtype=np.float64)

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
    if rollout_unicycle_single_nb is not None:
        return rollout_unicycle_single_nb(np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), float(dt))
    X = np.zeros((len(U) + 1, 3), dtype=np.float64)
    X[0] = x0
    for t, u in enumerate(U):
        X[t + 1] = unicycle_step(X[t], u, dt)
    return X

def rollout_unicycle_batch(x0: Array, U: Array, dt: float) -> Array:
    if rollout_unicycle_batch_nb is not None:
        return rollout_unicycle_batch_nb(np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), float(dt))
    N, H, _ = U.shape
    X = np.zeros((N, H + 1, 3), dtype=np.float64)
    X[:, 0, :] = x0[None, :]
    for t in range(H):
        th = X[:, t, 2]
        v = U[:, t, 0]
        om = U[:, t, 1]
        X[:, t + 1, 0] = X[:, t, 0] + v * np.cos(th) * dt
        X[:, t + 1, 1] = X[:, t, 1] + v * np.sin(th) * dt
        X[:, t + 1, 2] = wrap_angle(X[:, t, 2] + om * dt)
    return X

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
    if nominal_controls_to_track_path_nb is not None:
        return nominal_controls_to_track_path_nb(np.asarray(x0, dtype=np.float64), np.asarray(ref, dtype=np.float64), int(cfg.horizon), float(cfg.dt), float(cfg.v_min), float(cfg.v_max), float(cfg.omega_min), float(cfg.omega_max))
    H = cfg.horizon
    U = np.zeros((H, 2), dtype=np.float64)
    x = x0.copy()
    for t in range(H):
        target = ref[min(t + 3, len(ref) - 1)]
        delta = target - x[:2]
        dist = float(np.linalg.norm(delta))
        desired_heading = math.atan2(delta[1], delta[0])
        err = wrap_angle(desired_heading - x[2])
        heading_scale = max(0.0, math.cos(err)) ** 2
        v = np.clip(0.2 + 2.4 * dist * heading_scale, 0.0, cfg.v_max)
        omega = np.clip(3.2 * err, cfg.omega_min, cfg.omega_max)
        U[t] = [v, omega]
        x = unicycle_step(x, U[t], cfg.dt)
    return U

def nominal_controls_to_goal(x0: Array, goal: Array, cfg) -> Array:
    if nominal_controls_to_goal_nb is not None:
        return nominal_controls_to_goal_nb(np.asarray(x0, dtype=np.float64), np.asarray(goal, dtype=np.float64), int(cfg.horizon), float(cfg.dt), float(cfg.v_min), float(cfg.v_max), float(cfg.omega_min), float(cfg.omega_max))
    H = cfg.horizon
    U = np.zeros((H, 2), dtype=np.float64)
    x = x0.copy()
    for t in range(H):
        delta = goal - x[:2]
        dist = float(np.linalg.norm(delta))
        desired_heading = math.atan2(delta[1], delta[0])
        err = wrap_angle(desired_heading - x[2])
        heading_scale = max(0.0, math.cos(err)) ** 2
        v = np.clip(0.2 + 2.2 * dist * heading_scale, 0.0, cfg.v_max)
        omega = np.clip(3.0 * err, cfg.omega_min, cfg.omega_max)
        U[t] = [v, omega]
        x = unicycle_step(x, U[t], cfg.dt)
    return U

def standard_mppi_costs_batch(X: Array, U: Array, obstacle_circles: List[Tuple[Array, float]], goal: Array, cfg: MPPIConfig) -> Array:
    if standard_mppi_costs_batch_nb is not None:
        centers, radii = obstacle_circles_to_arrays(obstacle_circles)
        costs = standard_mppi_costs_batch_nb(np.asarray(X, dtype=np.float64), np.asarray(U, dtype=np.float64), centers, radii, np.asarray(goal, dtype=np.float64), int(cfg.horizon), float(cfg.robot_radius), float(cfg.w_goal), float(cfg.w_obstacle), float(cfg.w_control), float(cfg.w_control_smooth))
        return costs
    N, H, _ = U.shape
    costs = np.zeros(N, dtype=np.float64)
    P = X[:, 1:H + 1, :2]
    for t in range(H):
        p = P[:, t, :]
        goal_error = p - goal[None, :]
        costs += cfg.w_goal / H * np.sum(goal_error ** 2, axis=1)
        for center, radius in obstacle_circles:
            d = np.linalg.norm(p - center[None, :], axis=1) - radius
            margin = cfg.robot_radius
            costs += cfg.w_obstacle * softplus(8.0 * (margin - d)) ** 2
    costs += cfg.w_control * np.sum(U[:, :, 0] ** 2 + 0.15 * U[:, :, 1] ** 2, axis=1)
    dU = np.diff(U, axis=1)
    costs += cfg.w_control_smooth * np.sum(dU[:, :, 0] ** 2 + 0.2 * dU[:, :, 1] ** 2, axis=1)
    return costs

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
    """Compute Eq. (26) for the unicycle controls (v, omega)."""
    nominal = np.asarray(nominal_controls, dtype=np.float64)
    covariances = np.asarray(position_covariances, dtype=np.float64)
    horizon = int(nominal.shape[0])
    if nominal.shape != (horizon, 2):
        raise ValueError(f'nominal_controls must have shape (H,2), got {nominal.shape}')
    if covariances.shape != (horizon, 2, 2):
        raise ValueError(f'position_covariances must have shape (H,2,2), got {covariances.shape}')
    if sensitivity_projected_covariances_nb is not None:
        return sensitivity_projected_covariances_nb(np.asarray(x0, dtype=np.float64), nominal, covariances, int(cfg.spg_lookahead_steps), float(cfg.spg_fd_accel), float(cfg.spg_fd_steering_rate), float(cfg.spg_pseudoinverse_damping), float(cfg.spg_covariance_jitter), float(cfg.dt), float(cfg.v_min), float(cfg.v_max), float(cfg.omega_min), float(cfg.omega_max))
    nominal_states = rollout_unicycle(x0, nominal, cfg.dt)
    projected = np.zeros((horizon, 2, 2), dtype=np.float64)
    damping = float(cfg.spg_pseudoinverse_damping)
    covariance_scale = 1.0
    eye2 = np.eye(2, dtype=np.float64)
    for t in range(horizon):
        interval = min(max(1, int(cfg.spg_lookahead_steps)), horizon - t)
        jacobian = np.zeros((2, 2), dtype=np.float64)
        for control_index, delta in enumerate((float(cfg.spg_fd_accel), float(cfg.spg_fd_steering_rate))):
            lower = cfg.v_min if control_index == 0 else cfg.omega_min
            upper = cfg.v_max if control_index == 0 else cfg.omega_max
            plus_value = float(np.clip(nominal[t, control_index] + delta, lower, upper))
            minus_value = float(np.clip(nominal[t, control_index] - delta, lower, upper))
            denominator = plus_value - minus_value
            if denominator <= 1e-12:
                continue
            plus_controls = nominal[t:t + interval].copy()
            minus_controls = plus_controls.copy()
            plus_controls[0, control_index] = plus_value
            minus_controls[0, control_index] = minus_value
            plus_position = rollout_unicycle(nominal_states[t], plus_controls, cfg.dt)[-1, :2]
            minus_position = rollout_unicycle(nominal_states[t], minus_controls, cfg.dt)[-1, :2]
            jacobian[:, control_index] = (plus_position - minus_position) / denominator
        regularized = jacobian @ jacobian.T + (damping * damping + 1e-18) * eye2
        pseudoinverse = np.linalg.solve(regularized, jacobian).T
        position_covariance = 0.5 * (covariances[t] + covariances[t].T)
        control_covariance = pseudoinverse @ position_covariance @ pseudoinverse.T
        projected[t] = covariance_scale * 0.5 * (control_covariance + control_covariance.T) + float(cfg.spg_covariance_jitter) * eye2
    return projected

def rollout_collision_mask(X: Array, obstacle_circles: Sequence[Tuple[Array, float]], cfg: MPPIConfig) -> Array:
    if not obstacle_circles or X.shape[0] == 0:
        return np.zeros(X.shape[0], dtype=bool)
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    if rollout_collision_mask_nb is not None:
        return rollout_collision_mask_nb(
            np.asarray(X, dtype=np.float64),
            centers,
            radii,
            float(cfg.robot_radius),
            float(cfg.hard_collision_clearance),
        )
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


def centerline_connection_clearance(
    x_current: Array,
    closest_point: Array,
    obstacle_circles,
    cfg: MPPIConfig,
) -> float:
    """Exact swept-disc clearance from the robot position to the centerline."""
    if not obstacle_circles:
        return 1e309
    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    start = np.asarray(x_current[:2], dtype=np.float64)
    point = np.asarray(closest_point, dtype=np.float64).reshape(2)
    if centerline_connection_clearance_nb is not None:
        return float(
            centerline_connection_clearance_nb(
                start,
                point,
                centers,
                radii,
                float(cfg.robot_radius),
            )
        )
    segment = point - start
    denominator = float(segment @ segment)
    if denominator <= 1e-16:
        closest = np.repeat(start[None, :], len(centers), axis=0)
    else:
        alpha = np.clip(((centers - start[None, :]) @ segment) / denominator, 0.0, 1.0)
        closest = start[None, :] + alpha[:, None] * segment[None, :]
    return float(np.min(np.linalg.norm(centers - closest, axis=1) - radii - float(cfg.robot_radius)))


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


__all__ = [
    "ControllerVariant", "MPPIConfig", "Scene", "SimulationResult", "DynamicWallScenario",
    "build_default_scene", "default_dynamic_wall_scenarios", "obstacle_center",
    "make_wall_blockers_between_centers", "build_homotopy_modes", "run_controller",
    "min_clearance", "minimum_clearance", "obstacle_bounding_circles",
    "localize_mode_for_state", "localize_path_for_state", "NUMBA_AVAILABLE",
]
