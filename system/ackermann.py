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
MODEL_NAME = "ackermann"

ControllerVariant = ctrl.ControllerVariant
Scene = ctrl.Scene
SimulationResult = ctrl.SimulationResult
DynamicWallScenario = ctrl.DynamicWallScenario
MPPIHomotopyMode = ctrl.MPPIHomotopyMode


@dataclass
class MPPIConfig(ctrl.ControllerConfig):
    rear_axle_distance: float = 0.275
    front_axle_distance: float = 0.275
    mass: float = 18.0
    yaw_inertia: float = 1.2
    cornering_stiffness_front: float = 85.0
    cornering_stiffness_rear: float = 95.0
    tire_friction_coefficient: float = 0.95
    gravity: float = 9.81
    aerodynamic_drag_coefficient: float = 0.12
    rolling_resistance_force: float = 0.7
    minimum_tire_speed: float = 0.4
    dynamics_substeps: int = 4
    lateral_velocity_limit: float = 3.0
    yaw_rate_limit: float = 7.0

    v_min: float = -2.8
    v_max: float = 2.8
    accel_min: float = -3.5
    accel_max: float = 5.0
    steering_min: float = -1.2
    steering_max: float = 1.2
    steering_rate_min: float = -50.0
    steering_rate_max: float = 50.0
    noise_accel: float = 0.5
    noise_steering_rate: float = 1.0

    prior_ilqr_iterations: int = 2
    prior_ilqr_line_search_steps: int = 2
    prior_ilqr_mahalanobis_weight: float = 2.5
    prior_ilqr_covariance_floor: float = 0.12
    prior_ilqr_covariance_fallback_std: float = 0.25
    prior_ilqr_heading_weight: float = 2.5
    prior_ilqr_progress_weight: float = 2.0
    prior_ilqr_control_accel_weight: float = 0.01
    prior_ilqr_control_steering_rate_weight: float = 0.01
    prior_ilqr_regularization: float = 0.05

    vehicle_length: float = 0.80
    vehicle_width: float = 0.35
    collision_substeps: int = 5
    rollout_goal_tolerance: float = 0.305

    max_delta_accel: float = 1.2
    max_delta_steering_rate: float = 5.2
    enforce_one_step_safety: bool = False
    one_step_safety_clearance: float = 0.0

    @property
    def wheelbase(self) -> float:
        return self.front_axle_distance + self.rear_axle_distance

    def __post_init__(self) -> None:
        super().__post_init__()
        self.goal_tolerance = float(self.rollout_goal_tolerance)
        if self.wheelbase <= 0.0:
            raise ValueError("Ackermann axle distances must sum to a positive wheelbase.")
        positive = {
            "mass": self.mass,
            "yaw_inertia": self.yaw_inertia,
            "cornering_stiffness_front": self.cornering_stiffness_front,
            "cornering_stiffness_rear": self.cornering_stiffness_rear,
            "tire_friction_coefficient": self.tire_friction_coefficient,
            "minimum_tire_speed": self.minimum_tire_speed,
            "vehicle_length": self.vehicle_length,
            "vehicle_width": self.vehicle_width,
            "prior_ilqr_mahalanobis_weight": self.prior_ilqr_mahalanobis_weight,
            "prior_ilqr_covariance_floor": self.prior_ilqr_covariance_floor,
            "prior_ilqr_covariance_fallback_std": self.prior_ilqr_covariance_fallback_std,
            "prior_ilqr_heading_weight": self.prior_ilqr_heading_weight,
            "prior_ilqr_progress_weight": self.prior_ilqr_progress_weight,
            "prior_ilqr_control_accel_weight": self.prior_ilqr_control_accel_weight,
            "prior_ilqr_control_steering_rate_weight": self.prior_ilqr_control_steering_rate_weight,
            "prior_ilqr_regularization": self.prior_ilqr_regularization,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError("These values must be positive: " + ", ".join(invalid))
        self.prior_ilqr_iterations = max(1, int(self.prior_ilqr_iterations))
        self.prior_ilqr_line_search_steps = max(1, int(self.prior_ilqr_line_search_steps))
        self.dynamics_substeps = max(1, int(self.dynamics_substeps))
        for lower, upper, name in (
            (self.v_min, self.v_max, "velocity"),
            (self.accel_min, self.accel_max, "acceleration"),
            (self.steering_min, self.steering_max, "steering"),
            (self.steering_rate_min, self.steering_rate_max, "steering rate"),
        ):
            if lower > upper:
                raise ValueError(f"Invalid {name} bounds.")


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
def _dynamic_ackermann_step_nb(state, accel_cmd, steering_rate_cmd, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        px = state[0]
        py = state[1]
        psi = state[2]
        vx = state[3]
        vy = state[4]
        yaw_rate = state[5]
        steering = state[6]
        accel = min(max(accel_cmd, accel_min), accel_max)
        steering_rate = min(max(steering_rate_cmd, steering_rate_min), steering_rate_max)
        substeps = max(1, int(dynamics_substeps))
        h = dt / substeps
        wheelbase = max(front_axle_distance + rear_axle_distance, 1e-09)
        normal_front = mass * gravity * rear_axle_distance / wheelbase
        normal_rear = mass * gravity * front_axle_distance / wheelbase
        front_limit = tire_friction_coefficient * normal_front
        rear_limit = tire_friction_coefficient * normal_rear
        for _ in range(substeps):
            slip_speed = max(abs(vx), minimum_tire_speed)
            speed_scale = min(1.0, abs(vx) / minimum_tire_speed)
            alpha_front = math.atan2(vy + front_axle_distance * yaw_rate, slip_speed) - steering
            alpha_rear = math.atan2(vy - rear_axle_distance * yaw_rate, slip_speed)
            force_y_front = -cornering_stiffness_front * alpha_front * speed_scale
            force_y_rear = -cornering_stiffness_rear * alpha_rear * speed_scale
            force_y_front = min(max(force_y_front, -front_limit), front_limit)
            force_y_rear = min(max(force_y_rear, -rear_limit), rear_limit)
            drag_force = aerodynamic_drag_coefficient * vx * abs(vx)
            rolling_force = rolling_resistance_force * math.tanh(vx / 0.1)
            force_x = mass * accel - drag_force - rolling_force
            cos_psi = math.cos(psi)
            sin_psi = math.sin(psi)
            cos_delta = math.cos(steering)
            sin_delta = math.sin(steering)
            px_dot = vx * cos_psi - vy * sin_psi
            py_dot = vx * sin_psi + vy * cos_psi
            psi_dot = yaw_rate
            vx_dot = (force_x - force_y_front * sin_delta) / mass + yaw_rate * vy
            vy_dot = (force_y_front * cos_delta + force_y_rear) / mass - yaw_rate * vx
            yaw_accel = (front_axle_distance * force_y_front * cos_delta - rear_axle_distance * force_y_rear) / yaw_inertia
            px += h * px_dot
            py += h * py_dot
            psi = _wrap_angle_nb(psi + h * psi_dot)
            vx = min(max(vx + h * vx_dot, v_min), v_max)
            vy = min(max(vy + h * vy_dot, -lateral_velocity_limit), lateral_velocity_limit)
            yaw_rate = min(max(yaw_rate + h * yaw_accel, -yaw_rate_limit), yaw_rate_limit)
            steering = min(max(steering + h * steering_rate, steering_min), steering_max)
        return (px, py, psi, vx, vy, yaw_rate, steering)

@njit(cache=True, parallel=True)
def rollout_ackermann_batch_nb(x0, U, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        N = U.shape[0]
        H = U.shape[1]
        X = np.zeros((N, H + 1, 7), dtype=np.float64)
        for n in prange(N):
            for j in range(7):
                X[n, 0, j] = x0[j]
            for t in range(H):
                values = _dynamic_ackermann_step_nb(X[n, t], U[n, t, 0], U[n, t, 1], dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
                for j in range(7):
                    X[n, t + 1, j] = values[j]
        return X

@njit(cache=True)
def rollout_ackermann_single_nb(x0, U, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        H = U.shape[0]
        X = np.zeros((H + 1, 7), dtype=np.float64)
        for j in range(7):
            X[0, j] = x0[j]
        for t in range(H):
            values = _dynamic_ackermann_step_nb(X[t], U[t, 0], U[t, 1], dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
            for j in range(7):
                X[t + 1, j] = values[j]
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
def _project_ackermann_rollout_ilqr_nb(X, ref, arc, cov_blocks, covariance_floor):
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
def _ackermann_forward_only_step_ilqr_nb(
    x, accel, steering_rate,
    dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
    cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
    gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
    minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
    yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
    steering_rate_min, steering_rate_max,
):
    values = _dynamic_ackermann_step_nb(
        x, accel, steering_rate, dt, front_axle_distance, rear_axle_distance,
        mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear,
        tire_friction_coefficient, gravity, aerodynamic_drag_coefficient,
        rolling_resistance_force, minimum_tire_speed, dynamics_substeps,
        v_min, v_max, lateral_velocity_limit, yaw_rate_limit,
        accel_min, accel_max, steering_min, steering_max,
        steering_rate_min, steering_rate_max,
    )
    if values[3] >= -1e-8:
        return accel, values

    high = accel_max
    high_values = _dynamic_ackermann_step_nb(
        x, high, steering_rate, dt, front_axle_distance, rear_axle_distance,
        mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear,
        tire_friction_coefficient, gravity, aerodynamic_drag_coefficient,
        rolling_resistance_force, minimum_tire_speed, dynamics_substeps,
        v_min, v_max, lateral_velocity_limit, yaw_rate_limit,
        accel_min, accel_max, steering_min, steering_max,
        steering_rate_min, steering_rate_max,
    )
    if high_values[3] < 0.0:
        return high, high_values
    low = accel
    best_accel = high
    best_values = high_values
    for _ in range(4):
        mid = 0.5 * (low + high)
        mid_values = _dynamic_ackermann_step_nb(
            x, mid, steering_rate, dt, front_axle_distance, rear_axle_distance,
            mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear,
            tire_friction_coefficient, gravity, aerodynamic_drag_coefficient,
            rolling_resistance_force, minimum_tire_speed, dynamics_substeps,
            v_min, v_max, lateral_velocity_limit, yaw_rate_limit,
            accel_min, accel_max, steering_min, steering_max,
            steering_rate_min, steering_rate_max,
        )
        if mid_values[3] >= 0.0:
            high = mid
            best_accel = mid
            best_values = mid_values
        else:
            low = mid
    return best_accel, best_values


@njit(cache=True)
def _ackermann_ilqr_initial_controls_nb(
    x0, ref, arc, horizon,
    dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
    cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
    gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
    minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
    yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
    steering_rate_min, steering_rate_max,
):
    U = np.zeros((horizon, 2), dtype=np.float64)
    x = np.empty(7, dtype=np.float64)
    for j in range(7):
        x[j] = x0[j]
    cursor = 0
    wheelbase = max(front_axle_distance + rear_axle_distance, 1e-9)
    for t in range(horizon):
        s, qx, qy, tx, ty, heading, cursor = _project_path_forward_ilqr_nb(
            ref, arc, x[0], x[1], cursor
        )
        _ = (tx, ty, heading)
        remaining = max(0.0, arc[arc.shape[0] - 1] - s)
        lookahead = min(1.2, max(0.55, 0.55 + 0.22 * max(x[3], 0.0)))
        gx, gy = _path_intercept_point_nb(ref, cursor, qx, qy, lookahead)
        desired_heading = math.atan2(gy - x[1], gx - x[0])
        heading_error = _wrap_angle_nb(desired_heading - x[2])
        alignment = max(0.0, math.cos(heading_error))
        desired_speed = min(v_max * (0.30 + 0.70 * alignment * alignment), 1.5 * remaining)
        accel = 3.0 * (desired_speed - x[3])
        accel = min(max(accel, accel_min), accel_max)
        curvature = 2.0 * math.sin(heading_error) / max(lookahead, 1e-6)
        desired_steering = math.atan(wheelbase * curvature)
        desired_steering = min(max(desired_steering, steering_min), steering_max)
        steering_rate = 5.0 * (desired_steering - x[6]) - 0.20 * x[5]
        steering_rate = min(max(steering_rate, steering_rate_min), steering_rate_max)
        accel, values = _ackermann_forward_only_step_ilqr_nb(
            x, accel, steering_rate, dt, front_axle_distance, rear_axle_distance,
            mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear,
            tire_friction_coefficient, gravity, aerodynamic_drag_coefficient,
            rolling_resistance_force, minimum_tire_speed, dynamics_substeps,
            v_min, v_max, lateral_velocity_limit, yaw_rate_limit,
            accel_min, accel_max, steering_min, steering_max,
            steering_rate_min, steering_rate_max,
        )
        U[t, 0] = accel
        U[t, 1] = steering_rate
        for j in range(7):
            x[j] = values[j]
    return U


@njit(cache=True)
def _ackermann_ilqr_total_cost_nb(
    X, U, ref, arc, cov_blocks, covariance_floor,
    mahalanobis_weight, heading_weight, progress_weight,
    control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight,
):
    progress, qx, qy, tx, ty, heading, p00, p01, p11 = _project_ackermann_rollout_ilqr_nb(
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
        cost += control_accel_weight * U[t, 0] * U[t, 0] + control_steering_rate_weight * U[t, 1] * U[t, 1]
    exT = X[H, 0] - ref[ref.shape[0] - 1, 0]
    eyT = X[H, 1] - ref[ref.shape[0] - 1, 1]
    cost += terminal_position_weight * (exT * exT + eyT * eyT)
    cost += terminal_velocity_weight * (X[H, 3] * X[H, 3] + X[H, 4] * X[H, 4])
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
def _ackermann_ilqr_linearize_nb(
    x, u, xnext,
    dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
    cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
    gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
    minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
    yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
    steering_rate_min, steering_rate_max,
):
    A = np.zeros((7, 7), dtype=np.float64)
    B = np.zeros((7, 2), dtype=np.float64)
    A[0, 0] = 1.0
    A[1, 1] = 1.0

    for j in range(2, 7):
        eps = 1e-4 if j == 2 or j == 6 else 1e-3
        step = eps
        if j == 3:
            if x[j] + eps > v_max:
                step = -eps
            elif x[j] - eps < v_min:
                step = eps
        elif j == 4:
            if x[j] + eps > lateral_velocity_limit:
                step = -eps
            elif x[j] - eps < -lateral_velocity_limit:
                step = eps
        elif j == 5:
            if x[j] + eps > yaw_rate_limit:
                step = -eps
            elif x[j] - eps < -yaw_rate_limit:
                step = eps
        elif j == 6:
            if x[j] + eps > steering_max:
                step = -eps
            elif x[j] - eps < steering_min:
                step = eps
        xp = np.empty(7, dtype=np.float64)
        for k in range(7):
            xp[k] = x[k]
        xp[j] += step
        if j == 2:
            xp[j] = _wrap_angle_nb(xp[j])
        values = _dynamic_ackermann_step_nb(
            xp, u[0], u[1], dt, front_axle_distance, rear_axle_distance,
            mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear,
            tire_friction_coefficient, gravity, aerodynamic_drag_coefficient,
            rolling_resistance_force, minimum_tire_speed, dynamics_substeps,
            v_min, v_max, lateral_velocity_limit, yaw_rate_limit,
            accel_min, accel_max, steering_min, steering_max,
            steering_rate_min, steering_rate_max,
        )
        for i in range(7):
            diff = values[i] - xnext[i]
            if i == 2:
                diff = _wrap_angle_nb(diff)
            A[i, j] = diff / step

    for j in range(2):
        eps = 1e-3
        step = eps
        if j == 0:
            if u[j] + eps > accel_max:
                step = -eps
            elif u[j] - eps < accel_min:
                step = eps
        else:
            if u[j] + eps > steering_rate_max:
                step = -eps
            elif u[j] - eps < steering_rate_min:
                step = eps
        up0 = u[0]
        up1 = u[1]
        if j == 0:
            up0 += step
        else:
            up1 += step
        values = _dynamic_ackermann_step_nb(
            x, up0, up1, dt, front_axle_distance, rear_axle_distance,
            mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear,
            tire_friction_coefficient, gravity, aerodynamic_drag_coefficient,
            rolling_resistance_force, minimum_tire_speed, dynamics_substeps,
            v_min, v_max, lateral_velocity_limit, yaw_rate_limit,
            accel_min, accel_max, steering_min, steering_max,
            steering_rate_min, steering_rate_max,
        )
        for i in range(7):
            diff = values[i] - xnext[i]
            if i == 2:
                diff = _wrap_angle_nb(diff)
            B[i, j] = diff / step
    return A, B


@njit(cache=True)
def _ackermann_ilqr_backward_nb(
    X, U, ref, arc, cov_blocks, covariance_floor,
    mahalanobis_weight, heading_weight, progress_weight,
    control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight, regularization,
    dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
    cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
    gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
    minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
    yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
    steering_rate_min, steering_rate_max,
):
    H = U.shape[0]
    progress, qx, qy, tx, ty, heading, p00, p01, p11 = _project_ackermann_rollout_ilqr_nb(
        X, ref, arc, cov_blocks, covariance_floor
    )
    kff = np.zeros((H, 2), dtype=np.float64)
    Kfb = np.zeros((H, 2, 7), dtype=np.float64)

    Vx = np.zeros(7, dtype=np.float64)
    Vxx = np.zeros((7, 7), dtype=np.float64)
    exT = X[H, 0] - ref[ref.shape[0] - 1, 0]
    eyT = X[H, 1] - ref[ref.shape[0] - 1, 1]
    Vx[0] = 2.0 * terminal_position_weight * exT
    Vx[1] = 2.0 * terminal_position_weight * eyT
    Vxx[0, 0] = 2.0 * terminal_position_weight
    Vxx[1, 1] = 2.0 * terminal_position_weight
    Vx[3] = 2.0 * terminal_velocity_weight * X[H, 3]
    Vx[4] = 2.0 * terminal_velocity_weight * X[H, 4]
    Vxx[3, 3] = 2.0 * terminal_velocity_weight
    Vxx[4, 4] = 2.0 * terminal_velocity_weight

    for t in range(H - 1, -1, -1):
        A, B = _ackermann_ilqr_linearize_nb(
            X[t], U[t], X[t + 1],
            dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
            cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
            gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
            minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
            yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
            steering_rate_min, steering_rate_max,
        )
        dx = X[t, 0] - qx[t]
        dy = X[t, 1] - qy[t]
        mx = p00[t] * dx + p01[t] * dy
        my = p01[t] * dx + p11[t] * dy
        eh = _wrap_angle_nb(X[t, 2] - heading[t])
        lx = np.zeros(7, dtype=np.float64)
        lx[0] = 2.0 * mahalanobis_weight * mx - progress_weight * tx[t]
        lx[1] = 2.0 * mahalanobis_weight * my - progress_weight * ty[t]
        lx[2] = 2.0 * heading_weight * eh
        lxx = np.zeros((7, 7), dtype=np.float64)
        lxx[0, 0] = 2.0 * mahalanobis_weight * p00[t]
        lxx[0, 1] = 2.0 * mahalanobis_weight * p01[t]
        lxx[1, 0] = lxx[0, 1]
        lxx[1, 1] = 2.0 * mahalanobis_weight * p11[t]
        lxx[2, 2] = 2.0 * heading_weight
        lu = np.array((2.0 * control_accel_weight * U[t, 0], 2.0 * control_steering_rate_weight * U[t, 1]), dtype=np.float64)
        luu = np.zeros((2, 2), dtype=np.float64)
        luu[0, 0] = 2.0 * control_accel_weight
        luu[1, 1] = 2.0 * control_steering_rate_weight

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
        for j in range(7):
            Kfb[t, 0, j] = -(inv00 * Qux[0, j] + inv01 * Qux[1, j])
            Kfb[t, 1, j] = -(inv01 * Qux[0, j] + inv11 * Qux[1, j])

        K = Kfb[t]
        kval = kff[t]
        Vx = Qx + K.T @ Quu @ kval + K.T @ Qu + Qux.T @ kval
        Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
        Vxx = 0.5 * (Vxx + Vxx.T)
    return kff, Kfb

@njit(cache=True)
def _ackermann_ilqr_forward_update_nb(
    x0, U, X, kff, Kfb, alpha,
    dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
    cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
    gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
    minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
    yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
    steering_rate_min, steering_rate_max,
):
    H = U.shape[0]
    Unew = np.zeros_like(U)
    Xnew = np.zeros_like(X)
    for j in range(7):
        Xnew[0, j] = x0[j]
    for t in range(H):
        du0 = alpha * kff[t, 0]
        du1 = alpha * kff[t, 1]
        for j in range(7):
            dx = Xnew[t, j] - X[t, j]
            if j == 2:
                dx = _wrap_angle_nb(dx)
            du0 += Kfb[t, 0, j] * dx
            du1 += Kfb[t, 1, j] * dx
        u0 = min(max(U[t, 0] + du0, accel_min), accel_max)
        u1 = min(max(U[t, 1] + du1, steering_rate_min), steering_rate_max)
        u0, values = _ackermann_forward_only_step_ilqr_nb(
            Xnew[t], u0, u1, dt, front_axle_distance, rear_axle_distance,
            mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear,
            tire_friction_coefficient, gravity, aerodynamic_drag_coefficient,
            rolling_resistance_force, minimum_tire_speed, dynamics_substeps,
            v_min, v_max, lateral_velocity_limit, yaw_rate_limit,
            accel_min, accel_max, steering_min, steering_max,
            steering_rate_min, steering_rate_max,
        )
        Unew[t, 0] = u0
        Unew[t, 1] = u1
        for j in range(7):
            Xnew[t + 1, j] = values[j]
    return Unew, Xnew


@njit(cache=True)
def _ackermann_ilqr_nominal_and_positions_nb(
    x0, ref, cov_blocks, horizon,
    iterations, line_search_steps,
    mahalanobis_weight, covariance_floor, heading_weight, progress_weight,
    control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight, regularization,
    dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
    cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
    gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
    minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
    yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
    steering_rate_min, steering_rate_max,
):
    Uzero = np.zeros((horizon, 2), dtype=np.float64)
    positions = np.zeros(horizon, dtype=np.float64)
    if ref.shape[0] < 2:
        return Uzero, positions
    arc = _path_arc_lengths_ilqr_nb(ref)
    if arc[arc.shape[0] - 1] <= 1e-10:
        return Uzero, positions

    ilqr_steering_rate_min = max(steering_rate_min, -8.0)
    ilqr_steering_rate_max = min(steering_rate_max, 8.0)

    U = _ackermann_ilqr_initial_controls_nb(
        x0, ref, arc, horizon,
        dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
        cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
        gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
        minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
        yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
        ilqr_steering_rate_min, ilqr_steering_rate_max,
    )
    X = rollout_ackermann_single_nb(
        x0, U, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
        cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
        gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
        minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
        yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
        ilqr_steering_rate_min, ilqr_steering_rate_max,
    )
    best_cost = _ackermann_ilqr_total_cost_nb(
        X, U, ref, arc, cov_blocks, covariance_floor,
        mahalanobis_weight, heading_weight, progress_weight,
        control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight,
    )

    for _ in range(max(1, int(iterations))):
        kff, Kfb = _ackermann_ilqr_backward_nb(
            X, U, ref, arc, cov_blocks, covariance_floor,
            mahalanobis_weight, heading_weight, progress_weight,
            control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight, regularization,
            dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
            cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
            gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
            minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
            yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
            ilqr_steering_rate_min, ilqr_steering_rate_max,
        )
        improved = False
        alpha = 1.0
        for _ls in range(max(1, int(line_search_steps))):
            Utrial, Xtrial = _ackermann_ilqr_forward_update_nb(
                x0, U, X, kff, Kfb, alpha,
                dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
                cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
                gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
                minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
                yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
                ilqr_steering_rate_min, ilqr_steering_rate_max,
            )
            trial_cost = _ackermann_ilqr_total_cost_nb(
                Xtrial, Utrial, ref, arc, cov_blocks, covariance_floor,
                mahalanobis_weight, heading_weight, progress_weight,
                control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight,
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

    final_progress, _, _, _, _, _, _, _, _ = _project_ackermann_rollout_ilqr_nb(
        X, ref, arc, cov_blocks, covariance_floor
    )
    for t in range(horizon):
        positions[t] = final_progress[t]
    return U, positions


@njit(cache=True)
def nominal_controls_to_goal_nb(x0, goal, horizon, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        U = np.zeros((horizon, 2), dtype=np.float64)
        x = np.zeros(7, dtype=np.float64)
        for j in range(7):
            x[j] = x0[j]
        wheelbase = front_axle_distance + rear_axle_distance
        for t in range(horizon):
            dx = goal[0] - x[0]
            dy = goal[1] - x[1]
            dist = math.sqrt(dx * dx + dy * dy)
            desired_heading = math.atan2(dy, dx)
            heading_error = _wrap_angle_nb(desired_heading - x[2])
            heading_scale = max(0.0, math.cos(heading_error)) ** 2
            desired_speed = min(v_max * heading_scale, 1.5 * dist)
            accel = min(max(3.0 * (desired_speed - x[3]), accel_min), accel_max)
            lookahead = max(dist, 0.35)
            curvature = 2.0 * math.sin(heading_error) / lookahead
            desired_steering = math.atan(wheelbase * curvature)
            desired_steering = min(max(desired_steering, steering_min), steering_max)
            steering_rate = 4.0 * (desired_steering - x[6]) - 0.15 * x[5]
            steering_rate = min(max(steering_rate, steering_rate_min), steering_rate_max)
            U[t, 0] = accel
            U[t, 1] = steering_rate
            values = _dynamic_ackermann_step_nb(x, accel, steering_rate, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
            for j in range(7):
                x[j] = values[j]
        return U

@njit(cache=True)
def rectangle_circle_clearance_nb(px, py, heading, circle_x, circle_y, circle_radius, vehicle_length, vehicle_width):
        dx = circle_x - px
        dy = circle_y - py
        c = math.cos(heading)
        s = math.sin(heading)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        qx = abs(local_x) - 0.5 * vehicle_length
        qy = abs(local_y) - 0.5 * vehicle_width
        outside_x = max(qx, 0.0)
        outside_y = max(qy, 0.0)
        outside = math.sqrt(outside_x * outside_x + outside_y * outside_y)
        inside = min(max(qx, qy), 0.0)
        return outside + inside - circle_radius

@njit(cache=True)
def minimum_rectangle_circle_clearance_nb(state, circle_centers, circle_radii, vehicle_length, vehicle_width):
        if circle_radii.shape[0] == 0:
            return 1e+18
        px = state[0]
        py = state[1]
        heading = state[2]
        best = 1e+18
        for j in range(circle_radii.shape[0]):
            clearance = rectangle_circle_clearance_nb(px, py, heading, circle_centers[j, 0], circle_centers[j, 1], circle_radii[j], vehicle_length, vehicle_width)
            if clearance < best:
                best = clearance
        return best

@njit(cache=True)
def path_min_clearance_to_circles_nb(path, circle_centers, circle_radii, vehicle_length, vehicle_width, substeps):
        count = path.shape[0]
        if count == 0 or circle_radii.shape[0] == 0:
            return 1e+18
        headings = np.zeros(count, dtype=np.float64)
        if count > 1:
            for i in range(count):
                if i == 0:
                    dx = path[1, 0] - path[0, 0]
                    dy = path[1, 1] - path[0, 1]
                elif i == count - 1:
                    dx = path[count - 1, 0] - path[count - 2, 0]
                    dy = path[count - 1, 1] - path[count - 2, 1]
                else:
                    dx = path[i + 1, 0] - path[i - 1, 0]
                    dy = path[i + 1, 1] - path[i - 1, 1]
                headings[i] = math.atan2(dy, dx)
            for i in range(1, count):
                headings[i] = headings[i - 1] + _wrap_angle_nb(headings[i] - headings[i - 1])
        best = 1e+18
        for i in range(count):
            state = np.empty(3, dtype=np.float64)
            state[0] = path[i, 0]
            state[1] = path[i, 1]
            state[2] = headings[i]
            clearance = minimum_rectangle_circle_clearance_nb(state, circle_centers, circle_radii, vehicle_length, vehicle_width)
            if clearance < best:
                best = clearance
        interpolation_count = max(0, int(substeps))
        if count > 1 and interpolation_count > 0:
            denominator = float(interpolation_count + 1)
            state = np.empty(3, dtype=np.float64)
            for i in range(count - 1):
                dh = _wrap_angle_nb(headings[i + 1] - headings[i])
                for q in range(1, interpolation_count + 1):
                    alpha = q / denominator
                    state[0] = path[i, 0] + alpha * (path[i + 1, 0] - path[i, 0])
                    state[1] = path[i, 1] + alpha * (path[i + 1, 1] - path[i, 1])
                    state[2] = _wrap_angle_nb(headings[i] + alpha * dh)
                    clearance = minimum_rectangle_circle_clearance_nb(state, circle_centers, circle_radii, vehicle_length, vehicle_width)
                    if clearance < best:
                        best = clearance
        return best

@njit(cache=True, parallel=True)
def rollout_collision_mask_nb(X, circle_centers, circle_radii, vehicle_length, vehicle_width, hard_collision_clearance):
        N = X.shape[0]
        H = X.shape[1] - 1
        mask = np.zeros(N, dtype=np.bool_)
        for n in prange(N):
            for t in range(H):
                state = X[n, t + 1]
                clearance = minimum_rectangle_circle_clearance_nb(state, circle_centers, circle_radii, vehicle_length, vehicle_width)
                if clearance < hard_collision_clearance:
                    mask[n] = True
                    break
        return mask

@njit(cache=True)
def apply_smooth_safe_control_nb(x_current, u, previous_control, has_previous_control, circle_centers, circle_radii, max_delta_accel, max_delta_steering_rate, enforce_one_step_safety, one_step_safety_clearance, vehicle_length, vehicle_width, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        cmd = np.empty(2, dtype=np.float64)
        cmd[0] = u[0]
        cmd[1] = u[1]
        if has_previous_control:
            delta_accel = cmd[0] - previous_control[0]
            delta_accel = min(max(delta_accel, -max_delta_accel), max_delta_accel)
            delta_steering_rate = cmd[1] - previous_control[1]
            delta_steering_rate = min(max(delta_steering_rate, -max_delta_steering_rate), max_delta_steering_rate)
            cmd[0] = previous_control[0] + delta_accel
            cmd[1] = previous_control[1] + delta_steering_rate
        cmd[0] = min(max(cmd[0], accel_min), accel_max)
        cmd[1] = min(max(cmd[1], steering_rate_min), steering_rate_max)
        if enforce_one_step_safety and circle_radii.shape[0] > 0:
            values = _dynamic_ackermann_step_nb(x_current, cmd[0], cmd[1], dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
            x_next = np.empty(7, dtype=np.float64)
            for j in range(7):
                x_next[j] = values[j]
            current_clearance = minimum_rectangle_circle_clearance_nb(x_current, circle_centers, circle_radii, vehicle_length, vehicle_width)
            next_clearance = minimum_rectangle_circle_clearance_nb(x_next, circle_centers, circle_radii, vehicle_length, vehicle_width)
            moving_deeper = next_clearance < current_clearance - 0.0001
            below_required = next_clearance < one_step_safety_clearance
            if below_required and moving_deeper:
                if x_current[3] > 0.0:
                    cmd[0] = accel_min
                else:
                    cmd[0] = min(0.0, cmd[0])
        return cmd

@njit(cache=True, parallel=True)
def standard_mppi_costs_batch_nb(X, U, circle_centers, circle_radii, goal, horizon, vehicle_length, vehicle_width, w_goal, w_obstacle, w_control, w_control_smooth, w_terminal_position, w_terminal_velocity):
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
                heading = X[n, t + 1, 2]
                for j in range(M):
                    clearance = rectangle_circle_clearance_nb(px, py, heading, circle_centers[j, 0], circle_centers[j, 1], circle_radii[j], vehicle_length, vehicle_width)
                    sp = _softplus_scalar_nb(8.0 * (0.0 - clearance))
                    cost += w_obstacle * sp * sp
            ctrl_cost = 0.0
            for t in range(H):
                u0 = U[n, t, 0]
                u1 = U[n, t, 1]
                ctrl_cost += u0 * u0 + 0.15 * u1 * u1
            cost += w_control * ctrl_cost
            smooth_cost = 0.0
            for t in range(H - 1):
                du0 = U[n, t + 1, 0] - U[n, t, 0]
                du1 = U[n, t + 1, 1] - U[n, t, 1]
                smooth_cost += du0 * du0 + 0.2 * du1 * du1
            cost += w_control_smooth * smooth_cost
            gxT = X[n, H, 0] - goal[0]
            gyT = X[n, H, 1] - goal[1]
            cost += w_terminal_position * (gxT * gxT + gyT * gyT)
            cost += w_terminal_velocity * (X[n, H, 3] * X[n, H, 3] + X[n, H, 4] * X[n, H, 4])
            costs[n] = cost
        return costs

@njit(cache=True)
def point_in_poly_nb(px, py, poly, n):
        inside = False
        for i in range(n):
            j = i + 1
            if j == n:
                j = 0
            xi = poly[i, 0]
            yi = poly[i, 1]
            xj = poly[j, 0]
            yj = poly[j, 1]
            if (yi > py) != (yj > py):
                x_cross = xi + (py - yi) * (xj - xi) / (yj - yi + 1e-18)
                if px < x_cross:
                    inside = not inside
        return inside

@njit(cache=True)
def point_in_oriented_rectangle_nb(px, py, cx, cy, heading, half_length, half_width):
        dx = px - cx
        dy = py - cy
        c = math.cos(heading)
        s = math.sin(heading)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        return abs(local_x) <= half_length + 1e-12 and abs(local_y) <= half_width + 1e-12

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
def orientation_2d_nb(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

@njit(cache=True)
def point_on_segment_nb(px, py, ax, ay, bx, by):
        if abs(orientation_2d_nb(ax, ay, bx, by, px, py)) > 1e-12:
            return False
        return min(ax, bx) - 1e-12 <= px <= max(ax, bx) + 1e-12 and min(ay, by) - 1e-12 <= py <= max(ay, by) + 1e-12

@njit(cache=True)
def segments_intersect_nb(ax, ay, bx, by, cx, cy, dx, dy):
        o1 = orientation_2d_nb(ax, ay, bx, by, cx, cy)
        o2 = orientation_2d_nb(ax, ay, bx, by, dx, dy)
        o3 = orientation_2d_nb(cx, cy, dx, dy, ax, ay)
        o4 = orientation_2d_nb(cx, cy, dx, dy, bx, by)
        if (o1 > 0.0 and o2 < 0.0 or (o1 < 0.0 and o2 > 0.0)) and (o3 > 0.0 and o4 < 0.0 or (o3 < 0.0 and o4 > 0.0)):
            return True
        return abs(o1) <= 1e-12 and point_on_segment_nb(cx, cy, ax, ay, bx, by) or (abs(o2) <= 1e-12 and point_on_segment_nb(dx, dy, ax, ay, bx, by)) or (abs(o3) <= 1e-12 and point_on_segment_nb(ax, ay, cx, cy, dx, dy)) or (abs(o4) <= 1e-12 and point_on_segment_nb(bx, by, cx, cy, dx, dy))

@njit(cache=True)
def segment_segment_distance_nb(ax, ay, bx, by, cx, cy, dx, dy):
        if segments_intersect_nb(ax, ay, bx, by, cx, cy, dx, dy):
            return 0.0
        return min(point_segment_dist_nb(ax, ay, cx, cy, dx, dy), point_segment_dist_nb(bx, by, cx, cy, dx, dy), point_segment_dist_nb(cx, cy, ax, ay, bx, by), point_segment_dist_nb(dx, dy, ax, ay, bx, by))

@njit(cache=True)
def rectangle_polygon_clearance_nb(state, poly, n, vehicle_length, vehicle_width):
        cx = state[0]
        cy = state[1]
        heading = state[2]
        half_length = 0.5 * vehicle_length
        half_width = 0.5 * vehicle_width
        c = math.cos(heading)
        s = math.sin(heading)
        corners = np.empty((4, 2), dtype=np.float64)
        local_x = np.array((-half_length, half_length, half_length, -half_length))
        local_y = np.array((-half_width, -half_width, half_width, half_width))
        for i in range(4):
            corners[i, 0] = cx + c * local_x[i] - s * local_y[i]
            corners[i, 1] = cy + s * local_x[i] + c * local_y[i]
            if point_in_poly_nb(corners[i, 0], corners[i, 1], poly, n):
                return -1e-09
        for i in range(n):
            if point_in_oriented_rectangle_nb(poly[i, 0], poly[i, 1], cx, cy, heading, half_length, half_width):
                return -1e-09
        best = 1e+18
        for i in range(4):
            ni = i + 1
            if ni == 4:
                ni = 0
            ax = corners[i, 0]
            ay = corners[i, 1]
            bx = corners[ni, 0]
            by = corners[ni, 1]
            for j in range(n):
                nj = j + 1
                if nj == n:
                    nj = 0
                distance = segment_segment_distance_nb(ax, ay, bx, by, poly[j, 0], poly[j, 1], poly[nj, 0], poly[nj, 1])
                if distance <= 1e-12:
                    return -1e-09
                if distance < best:
                    best = distance
        return best

@njit(cache=True)
def min_clearance_nb(states, polys_padded, poly_lengths, vehicle_length, vehicle_width):
        best = 1e+18
        for state_index in range(states.shape[0]):
            state = states[state_index]
            for polygon_index in range(poly_lengths.shape[0]):
                n = int(poly_lengths[polygon_index])
                clearance = rectangle_polygon_clearance_nb(state, polys_padded[polygon_index], n, vehicle_length, vehicle_width)
                if clearance < best:
                    best = clearance
        return best

@njit(cache=True, parallel=True)
def sensitivity_projected_covariances_nb(x0, nominal_controls, position_covariances, lookahead_steps, fd_accel, fd_steering_rate, pseudoinverse_damping, covariance_jitter, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        """Project position covariance into control space with Eq. (26)."""
        horizon = nominal_controls.shape[0]
        nominal_states = rollout_ackermann_single_nb(x0, nominal_controls, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
        projected = np.zeros((horizon, 2, 2), dtype=np.float64)
        damping_sq = pseudoinverse_damping * pseudoinverse_damping
        for t in prange(horizon):
            interval = min(max(1, int(lookahead_steps)), horizon - t)
            jacobian = np.zeros((2, 2), dtype=np.float64)
            for control_index in range(2):
                delta = fd_accel if control_index == 0 else fd_steering_rate
                lower = accel_min if control_index == 0 else steering_rate_min
                upper = accel_max if control_index == 0 else steering_rate_max
                center = nominal_controls[t, control_index]
                plus_value = min(center + delta, upper)
                minus_value = max(center - delta, lower)
                denominator = plus_value - minus_value
                if denominator <= 1e-12:
                    continue
                plus_state = nominal_states[t].copy()
                minus_state = nominal_states[t].copy()
                for k in range(interval):
                    accel_plus = nominal_controls[t + k, 0]
                    steer_plus = nominal_controls[t + k, 1]
                    accel_minus = accel_plus
                    steer_minus = steer_plus
                    if k == 0:
                        if control_index == 0:
                            accel_plus = plus_value
                            accel_minus = minus_value
                        else:
                            steer_plus = plus_value
                            steer_minus = minus_value
                    plus_values = _dynamic_ackermann_step_nb(plus_state, accel_plus, steer_plus, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
                    minus_values = _dynamic_ackermann_step_nb(minus_state, accel_minus, steer_minus, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
                    for state_index in range(7):
                        plus_state[state_index] = plus_values[state_index]
                        minus_state[state_index] = minus_values[state_index]
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

def ackermann_rectangle_corners(state: Array, vehicle_length: float, vehicle_width: float) -> Array:
    """Return the four world-frame corners of the centered Ackermann body."""
    x, y, heading = map(float, np.asarray(state, dtype=np.float64)[:3])
    half_length = 0.5 * float(vehicle_length)
    half_width = 0.5 * float(vehicle_width)
    local = np.array([[-half_length, -half_width], [half_length, -half_width], [half_length, half_width], [-half_length, half_width]], dtype=np.float64)
    c = math.cos(heading)
    s = math.sin(heading)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + np.array([x, y], dtype=np.float64)

def rectangle_circle_clearance(state: Array, circle_center: Array, circle_radius: float, vehicle_length: float, vehicle_width: float) -> float:
    """Signed clearance between an oriented rectangle and a circle."""
    x, y, heading = map(float, np.asarray(state, dtype=np.float64)[:3])
    center = np.asarray(circle_center, dtype=np.float64)
    dx = float(center[0] - x)
    dy = float(center[1] - y)
    c = math.cos(heading)
    s = math.sin(heading)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    qx = abs(local_x) - 0.5 * float(vehicle_length)
    qy = abs(local_y) - 0.5 * float(vehicle_width)
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return outside + inside - float(circle_radius)

def minimum_rectangle_circle_clearance(state: Array, centers: Array, radii: Array, vehicle_length: float, vehicle_width: float) -> float:
    state_array = np.asarray(state, dtype=np.float64)
    center_array = np.asarray(centers, dtype=np.float64)
    radius_array = np.asarray(radii, dtype=np.float64)
    if radius_array.size == 0:
        return 1e309
    return float(minimum_rectangle_circle_clearance_nb(
        state_array, center_array, radius_array, float(vehicle_length), float(vehicle_width)
    ))

def _point_in_oriented_rectangle(point: Array, state: Array, vehicle_length: float, vehicle_width: float) -> bool:
    x, y, heading = map(float, np.asarray(state, dtype=np.float64)[:3])
    dx = float(point[0] - x)
    dy = float(point[1] - y)
    c = math.cos(heading)
    s = math.sin(heading)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return bool(abs(local_x) <= 0.5 * float(vehicle_length) + 1e-12 and abs(local_y) <= 0.5 * float(vehicle_width) + 1e-12)

def _orientation_2d(a: Array, b: Array, c: Array) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

def _point_on_segment(point: Array, a: Array, b: Array, eps: float=1e-12) -> bool:
    if abs(_orientation_2d(a, b, point)) > eps:
        return False
    return bool(min(a[0], b[0]) - eps <= point[0] <= max(a[0], b[0]) + eps and min(a[1], b[1]) - eps <= point[1] <= max(a[1], b[1]) + eps)

def _segments_intersect(a: Array, b: Array, c: Array, d: Array) -> bool:
    o1 = _orientation_2d(a, b, c)
    o2 = _orientation_2d(a, b, d)
    o3 = _orientation_2d(c, d, a)
    o4 = _orientation_2d(c, d, b)
    if (o1 > 0.0 and o2 < 0.0 or (o1 < 0.0 and o2 > 0.0)) and (o3 > 0.0 and o4 < 0.0 or (o3 < 0.0 and o4 > 0.0)):
        return True
    return bool(abs(o1) <= 1e-12 and _point_on_segment(c, a, b) or (abs(o2) <= 1e-12 and _point_on_segment(d, a, b)) or (abs(o3) <= 1e-12 and _point_on_segment(a, c, d)) or (abs(o4) <= 1e-12 and _point_on_segment(b, c, d)))

def _point_segment_distance(point: Array, a: Array, b: Array) -> float:
    ab = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    denom = float(ab @ ab)
    if denom <= 1e-16:
        return float(np.linalg.norm(np.asarray(point) - np.asarray(a)))
    t = float(np.clip((np.asarray(point) - np.asarray(a)) @ ab / denom, 0.0, 1.0))
    closest = np.asarray(a) + t * ab
    return float(np.linalg.norm(np.asarray(point) - closest))

def _segment_segment_distance(a: Array, b: Array, c: Array, d: Array) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d), _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))

def rectangle_polygon_clearance(state: Array, obstacle, vehicle_length: float, vehicle_width: float) -> float:
    """Signed rectangle--polygon clearance; negative means overlap."""
    rectangle = ackermann_rectangle_corners(state, vehicle_length, vehicle_width)
    polygon = _poly_vertices(obstacle)
    if any((point_in_poly(corner, polygon) for corner in rectangle)):
        return -1e-09
    if any((_point_in_oriented_rectangle(vertex, state, vehicle_length, vehicle_width) for vertex in polygon)):
        return -1e-09
    minimum = 1e309
    for i in range(len(rectangle)):
        a = rectangle[i]
        b = rectangle[(i + 1) % len(rectangle)]
        for j in range(len(polygon)):
            c = polygon[j]
            d = polygon[(j + 1) % len(polygon)]
            distance = _segment_segment_distance(a, b, c, d)
            if distance <= 1e-12:
                return -1e-09
            minimum = min(minimum, distance)
    return minimum

def min_clearance(states: Array, obstacles: Sequence=0.18, vehicle_length: float=0.81, vehicle_width: float=0.36) -> float:
    """Minimum signed clearance using the Numba oriented-rectangle kernel."""
    state_array = np.asarray(states, dtype=np.float64)
    if state_array.size == 0 or not obstacles:
        return 1e309
    padded, lengths = obstacles_to_padded_arrays(obstacles)
    return float(min_clearance_nb(state_array, padded, lengths, float(vehicle_length), float(vehicle_width)))

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def ackermann_step(x: Array, u: Array, cfg: MPPIConfig) -> Array:
    state = np.asarray(x, dtype=np.float64)
    control = np.asarray(u, dtype=np.float64)
    values = _dynamic_ackermann_step_nb(
        state, float(control[0]), float(control[1]), *_dynamic_model_arguments(cfg)
    )
    return np.asarray(values, dtype=np.float64)

def goal_pose_satisfied(state: Array, goal: Array, goal_tolerance: float, cfg: MPPIConfig) -> bool:
    state = np.asarray(state, dtype=np.float64)
    position_ok = np.linalg.norm(state[:2] - np.asarray(goal, dtype=np.float64)) <= goal_tolerance
    speed_ok = math.hypot(float(state[3]), float(state[4])) <= float(cfg.terminal_velocity_tolerance)
    return bool(position_ok and speed_ok)

def _dynamic_model_arguments(cfg: MPPIConfig) -> Tuple[float, ...]:
    return (float(cfg.dt), float(cfg.front_axle_distance), float(cfg.rear_axle_distance), float(cfg.mass), float(cfg.yaw_inertia), float(cfg.cornering_stiffness_front), float(cfg.cornering_stiffness_rear), float(cfg.tire_friction_coefficient), float(cfg.gravity), float(cfg.aerodynamic_drag_coefficient), float(cfg.rolling_resistance_force), float(cfg.minimum_tire_speed), int(cfg.dynamics_substeps), float(cfg.v_min), float(cfg.v_max), float(cfg.lateral_velocity_limit), float(cfg.yaw_rate_limit), float(cfg.accel_min), float(cfg.accel_max), float(cfg.steering_min), float(cfg.steering_max), float(cfg.steering_rate_min), float(cfg.steering_rate_max))

def rollout_ackermann(x0: Array, U: Array, cfg: MPPIConfig) -> Array:
    return rollout_ackermann_single_nb(
        np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), *_dynamic_model_arguments(cfg)
    )

def rollout_ackermann_batch(x0: Array, U: Array, cfg: MPPIConfig) -> Array:
    return rollout_ackermann_batch_nb(
        np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), *_dynamic_model_arguments(cfg)
    )

def obstacle_circles_to_arrays(obstacle_circles: List[Tuple[Array, float]]) -> Tuple[Array, Array]:
    if not obstacle_circles:
        return (np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64))
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    return (centers, radii)

def apply_smooth_safe_control(x_current: Array, u: Array, previous_control: Optional[Array], obstacle_circles: List[Tuple[Array, float]], cfg: MPPIConfig) -> Array:
    state = np.asarray(x_current, dtype=np.float64)
    command = np.asarray(u, dtype=np.float64)
    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    has_previous = previous_control is not None
    previous = np.asarray(previous_control, dtype=np.float64) if has_previous else np.zeros(2, dtype=np.float64)
    return apply_smooth_safe_control_nb(
        state, command, previous, has_previous, centers, radii,
        float(cfg.max_delta_accel), float(cfg.max_delta_steering_rate),
        bool(cfg.enforce_one_step_safety), float(cfg.one_step_safety_clearance),
        float(cfg.vehicle_length), float(cfg.vehicle_width), *_dynamic_model_arguments(cfg)
    )

def path_min_clearance_to_circles(path: Array, obstacle_circles: List[Tuple[Array, float]], vehicle_length: float, vehicle_width: float, substeps: int=2) -> float:
    p = np.asarray(path, dtype=np.float64)
    if len(p) == 0 or not obstacle_circles:
        return 1e309
    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    return float(path_min_clearance_to_circles_nb(
        p, centers, radii, float(vehicle_length), float(vehicle_width), int(substeps)
    ))

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

def _prepare_ilqr_covariance(ref: Array, cov_blocks: Optional[Array], cfg: MPPIConfig) -> Array:
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
    x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array] = None
) -> Tuple[Array, Array]:
    path = np.asarray(ref, dtype=np.float64)
    cov = _prepare_ilqr_covariance(path, cov_blocks, cfg)
    return _ackermann_ilqr_nominal_and_positions_nb(
        np.asarray(x0, dtype=np.float64), path, cov, int(cfg.horizon),
        int(cfg.prior_ilqr_iterations), int(cfg.prior_ilqr_line_search_steps),
        float(cfg.prior_ilqr_mahalanobis_weight), float(cfg.prior_ilqr_covariance_floor),
        float(cfg.prior_ilqr_heading_weight), float(cfg.prior_ilqr_progress_weight),
        float(cfg.prior_ilqr_control_accel_weight), float(cfg.prior_ilqr_control_steering_rate_weight),
        float(cfg.w_terminal_position), float(cfg.w_terminal_velocity),
        float(cfg.prior_ilqr_regularization), *_dynamic_model_arguments(cfg),
    )


def prior_control_arc_positions(
    x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array] = None
) -> Array:
    _, positions = nominal_controls_and_arc_positions(x0, ref, cfg, cov_blocks)
    return positions


def nominal_controls_to_track_path(
    x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array] = None
) -> Array:
    controls, _ = nominal_controls_and_arc_positions(x0, ref, cfg, cov_blocks)
    return controls


def nominal_controls_to_goal(x0: Array, goal: Array, cfg: MPPIConfig) -> Array:
    return nominal_controls_to_goal_nb(
        np.asarray(x0, dtype=np.float64), np.asarray(goal, dtype=np.float64), int(cfg.horizon),
        *_dynamic_model_arguments(cfg)
    )

def standard_mppi_costs_batch(X: Array, U: Array, obstacle_circles: List[Tuple[Array, float]], goal: Array, cfg: MPPIConfig) -> Array:
    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    costs = standard_mppi_costs_batch_nb(
        np.asarray(X, dtype=np.float64), np.asarray(U, dtype=np.float64), centers, radii,
        np.asarray(goal, dtype=np.float64), int(cfg.horizon),
        float(cfg.vehicle_length), float(cfg.vehicle_width), float(cfg.w_goal),
        float(cfg.w_obstacle), float(cfg.w_control), float(cfg.w_control_smooth),
        float(cfg.w_terminal_position), float(cfg.w_terminal_velocity),
    )
    return costs

def stable_representation_costs(X: Array, U: Array, obstacle_circles: List[Tuple[Array, float]], goal: Array, cfg: MPPIConfig) -> Array:
    return standard_mppi_costs_batch(X, U, obstacle_circles, goal, cfg)

def enforce_ackermann_control_bounds(U: Array, cfg: MPPIConfig) -> Array:
    U = np.asarray(U, dtype=np.float64)
    if U.size == 0:
        return U
    U[:, :, 0] = np.clip(U[:, :, 0], cfg.accel_min, cfg.accel_max)
    U[:, :, 1] = np.clip(U[:, :, 1], cfg.steering_rate_min, cfg.steering_rate_max)
    return U

def sensitivity_projected_control_covariances(x_current: Array, nominal_controls: Array, position_covariances: Array, cfg: MPPIConfig) -> Array:
    """Compute Sigma_u,t = J_t^dagger Sigma_p,t J_t^dagger.T using the Numba kernel."""
    x0 = np.asarray(x_current, dtype=np.float64)
    nominal = np.asarray(nominal_controls, dtype=np.float64)
    covariances = np.asarray(position_covariances, dtype=np.float64)
    return sensitivity_projected_covariances_nb(
        x0, nominal, covariances,
        int(cfg.spg_lookahead_steps), float(cfg.spg_fd_accel), float(cfg.spg_fd_steering_rate),
        float(cfg.spg_pseudoinverse_damping), float(cfg.spg_covariance_jitter),
        *_dynamic_model_arguments(cfg),
    )

def rollout_collision_mask(X: Array, obstacle_circles: Sequence[Tuple[Array, float]], goal: Array, cfg: MPPIConfig) -> Array:
    state_batch = np.asarray(X, dtype=np.float64)
    if not obstacle_circles or state_batch.shape[0] == 0:
        return np.zeros(state_batch.shape[0], dtype=bool)
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    del goal
    return rollout_collision_mask_nb(
        state_batch, centers, radii,
        float(cfg.vehicle_length), float(cfg.vehicle_width),
        float(cfg.hard_collision_clearance),
    )

def update_display_trajectory(info: Dict[str, object], x_current: Array, executed_u: Array, goal: Array, cfg: MPPIConfig) -> None:
    sequence = info.get('planned_control_sequence')
    if sequence is None:
        return
    display_u = np.asarray(sequence, dtype=np.float64).copy()
    if display_u.ndim != 2 or display_u.shape[1] != 2 or len(display_u) == 0:
        return
    display_u[0] = np.asarray(executed_u, dtype=np.float64)
    del goal
    info['optimal_traj'] = rollout_ackermann(x_current, display_u, cfg)

def initial_pose(start: Array, goal: Array) -> Array:
    direction = goal - start
    heading = math.atan2(direction[1], direction[0])
    return np.array([start[0], start[1], heading, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

def control_noise_scale(cfg: MPPIConfig) -> Array:
    return np.asarray([cfg.noise_accel, cfg.noise_steering_rate], dtype=np.float64)


def clip_control_batch(controls: Array, cfg: MPPIConfig) -> Array:
    return enforce_ackermann_control_bounds(np.asarray(controls, dtype=np.float64), cfg)


def rollout_batch(x_current: Array, controls: Array, cfg: MPPIConfig) -> Array:
    return rollout_ackermann_batch(x_current, controls, cfg)


def rollout_single(x_current: Array, controls: Array, cfg: MPPIConfig) -> Array:
    return rollout_ackermann(x_current, controls, cfg)


def project_control_covariances(x_current: Array, nominal: Array, covariances: Array, cfg: MPPIConfig) -> Array:
    return sensitivity_projected_control_covariances(x_current, nominal, covariances, cfg)


def trajectory_costs(states: Array, controls: Array, obstacle_circles, goal: Array, cfg: MPPIConfig) -> Array:
    return stable_representation_costs(states, controls, obstacle_circles, goal, cfg)


def collision_mask(states: Array, obstacle_circles, goal: Array, cfg: MPPIConfig) -> Array:
    return rollout_collision_mask(states, obstacle_circles, goal, cfg)


def mean_path_clearance(path: Array, obstacle_circles, cfg: MPPIConfig) -> float:
    return path_min_clearance_to_circles(
        path,
        obstacle_circles,
        cfg.vehicle_length,
        cfg.vehicle_width,
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
    update_display_trajectory(info, x_current, control, goal, cfg)


def goal_reached(state: Array, goal: Array, cfg: MPPIConfig) -> bool:
    return goal_pose_satisfied(state, goal, cfg.rollout_goal_tolerance, cfg)


def advance_state(state: Array, control: Array, goal: Array, cfg: MPPIConfig) -> Tuple[Array, bool]:
    next_state = ackermann_step(state, control, cfg)
    return next_state, goal_reached(next_state, goal, cfg)


def minimum_clearance(states: Array, obstacles: Sequence, cfg: MPPIConfig) -> float:
    return min_clearance(states, obstacles, cfg.vehicle_length, cfg.vehicle_width)


SUPPORTED_VARIANTS = {
    ControllerVariant.PLANNER_ILQR,
    ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI,
    ControllerVariant.GAUSSIAN_PRIOR_MPPI,
    ControllerVariant.CORRIDOR_PRIOR_MPPI,
    ControllerVariant.CONTROL_BANK_MPPI,
    ControllerVariant.STANDARD_MPPI,
}

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