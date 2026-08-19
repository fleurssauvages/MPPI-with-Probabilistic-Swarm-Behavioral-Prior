from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
try:
    from . import controller as ctrl
except ImportError:
    import controller as ctrl
from numba import njit, prange
Array = np.ndarray
MODEL_NAME = 'ackermann'
DISPLAY_NAME = 'Ackermann'
STATE_DIM = 7
BODY_COLOR = '#17becf'

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
    v_min: float = -3.5
    v_max: float = 3.5
    accel_min: float = -5.0
    accel_max: float = 7.0
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
    vehicle_length: float = 0.8
    vehicle_width: float = 0.35
    collision_substeps: int = 5
    rollout_goal_tolerance: float = 0.305
    max_delta_accel: float = 1.2
    max_delta_steering_rate: float = 5.2

    @property
    def wheelbase(self) -> float:
        return self.front_axle_distance + self.rear_axle_distance

    def __post_init__(self) -> None:
        super().__post_init__()
        self.goal_tolerance = float(self.rollout_goal_tolerance)
        if self.wheelbase <= 0.0:
            raise ValueError('Ackermann axle distances must sum to a positive wheelbase.')
        positive = {'mass': self.mass, 'yaw_inertia': self.yaw_inertia, 'cornering_stiffness_front': self.cornering_stiffness_front, 'cornering_stiffness_rear': self.cornering_stiffness_rear, 'tire_friction_coefficient': self.tire_friction_coefficient, 'minimum_tire_speed': self.minimum_tire_speed, 'vehicle_length': self.vehicle_length, 'vehicle_width': self.vehicle_width, 'prior_ilqr_mahalanobis_weight': self.prior_ilqr_mahalanobis_weight, 'prior_ilqr_covariance_floor': self.prior_ilqr_covariance_floor, 'prior_ilqr_covariance_fallback_std': self.prior_ilqr_covariance_fallback_std, 'prior_ilqr_heading_weight': self.prior_ilqr_heading_weight, 'prior_ilqr_progress_weight': self.prior_ilqr_progress_weight, 'prior_ilqr_control_accel_weight': self.prior_ilqr_control_accel_weight, 'prior_ilqr_control_steering_rate_weight': self.prior_ilqr_control_steering_rate_weight, 'prior_ilqr_regularization': self.prior_ilqr_regularization}
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError('These values must be positive: ' + ', '.join(invalid))
        self.prior_ilqr_iterations = max(1, int(self.prior_ilqr_iterations))
        self.prior_ilqr_line_search_steps = max(1, int(self.prior_ilqr_line_search_steps))
        self.dynamics_substeps = max(1, int(self.dynamics_substeps))
        for lower, upper, name in ((self.v_min, self.v_max, 'velocity'), (self.accel_min, self.accel_max, 'acceleration'), (self.steering_min, self.steering_max, 'steering'), (self.steering_rate_min, self.steering_rate_max, 'steering rate')):
            if lower > upper:
                raise ValueError(f'Invalid {name} bounds.')

@njit(cache=True)
def _wrap_angle_nb(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

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
        return (tx, ty)
    dx = ref[best_seg + 1, 0] - best_qx
    dy = ref[best_seg + 1, 1] - best_qy
    seg_len = math.sqrt(dx * dx + dy * dy)
    if seg_len > 1e-12:
        if remaining <= seg_len:
            alpha = remaining / seg_len
            return (best_qx + alpha * dx, best_qy + alpha * dy)
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
            return (ref[i, 0] + alpha * dx, ref[i, 1] + alpha * dy)
        remaining -= seg_len
        tx = ref[i + 1, 0]
        ty = ref[i + 1, 1]
    return (tx, ty)

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
        return (0.0, ref[0, 0], ref[0, 1], 1.0, 0.0, 0.0, 0)
    first = min(max(int(start_seg), 0), n - 2)
    best_d2 = 1e+300
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
    return (progress, best_qx, best_qy, best_tx, best_ty, heading, best_seg)

@njit(cache=True)
def _spatial_precision_ilqr_nb(cov_blocks, arc, seg, progress, covariance_floor):
    """Inverse of Sigma(s)+sigma_floor^2 I at geometric projected progress."""
    n = cov_blocks.shape[0]
    if n == 0:
        floor_var = max(covariance_floor * covariance_floor, 1e-08)
        inv = 1.0 / floor_var
        return (inv, 0.0, inv)
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
    floor_var = max(covariance_floor * covariance_floor, 1e-08)
    a = max(c00 + floor_var, floor_var)
    d = max(c11 + floor_var, floor_var)
    b = 0.5 * (c01a + c01b)
    max_b = 0.999999 * math.sqrt(max(a * d, 0.0))
    b = min(max(b, -max_b), max_b)
    det = max(a * d - b * b, 1e-14)
    return (d / det, -b / det, a / det)

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
    return (progress, qx, qy, tx, ty, heading, p00, p01, p11)

@njit(cache=True)
def _ackermann_forward_only_step_ilqr_nb(x, accel, steering_rate, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
    values = _dynamic_ackermann_step_nb(x, accel, steering_rate, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
    if values[3] >= -1e-08:
        return (accel, values)
    high = accel_max
    high_values = _dynamic_ackermann_step_nb(x, high, steering_rate, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
    if high_values[3] < 0.0:
        return (high, high_values)
    low = accel
    best_accel = high
    best_values = high_values
    for _ in range(4):
        mid = 0.5 * (low + high)
        mid_values = _dynamic_ackermann_step_nb(x, mid, steering_rate, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
        if mid_values[3] >= 0.0:
            high = mid
            best_accel = mid
            best_values = mid_values
        else:
            low = mid
    return (best_accel, best_values)

@njit(cache=True)
def _ackermann_ilqr_initial_controls_nb(x0, ref, arc, horizon, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
    U = np.zeros((horizon, 2), dtype=np.float64)
    x = np.empty(7, dtype=np.float64)
    for j in range(7):
        x[j] = x0[j]
    cursor = 0
    wheelbase = max(front_axle_distance + rear_axle_distance, 1e-09)
    for t in range(horizon):
        s, qx, qy, tx, ty, heading, cursor = _project_path_forward_ilqr_nb(ref, arc, x[0], x[1], cursor)
        _ = (tx, ty, heading)
        remaining = max(0.0, arc[arc.shape[0] - 1] - s)
        lookahead = min(1.2, max(0.55, 0.55 + 0.22 * max(x[3], 0.0)))
        gx, gy = _path_intercept_point_nb(ref, cursor, qx, qy, lookahead)
        desired_heading = math.atan2(gy - x[1], gx - x[0])
        heading_error = _wrap_angle_nb(desired_heading - x[2])
        alignment = max(0.0, math.cos(heading_error))
        desired_speed = min(v_max * (0.3 + 0.7 * alignment * alignment), 1.5 * remaining)
        accel = 3.0 * (desired_speed - x[3])
        accel = min(max(accel, accel_min), accel_max)
        curvature = 2.0 * math.sin(heading_error) / max(lookahead, 1e-06)
        desired_steering = math.atan(wheelbase * curvature)
        desired_steering = min(max(desired_steering, steering_min), steering_max)
        steering_rate = 5.0 * (desired_steering - x[6]) - 0.2 * x[5]
        steering_rate = min(max(steering_rate, steering_rate_min), steering_rate_max)
        accel, values = _ackermann_forward_only_step_ilqr_nb(x, accel, steering_rate, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
        U[t, 0] = accel
        U[t, 1] = steering_rate
        for j in range(7):
            x[j] = values[j]
    return U

@njit(cache=True)
def _ackermann_ilqr_total_cost_nb(X, U, ref, arc, cov_blocks, covariance_floor, mahalanobis_weight, heading_weight, progress_weight, control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight):
    progress, qx, qy, tx, ty, heading, p00, p01, p11 = _project_ackermann_rollout_ilqr_nb(X, ref, arc, cov_blocks, covariance_floor)
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
    r = max(regularization, 1e-09)
    a00 += r
    a11 += r
    det = a00 * a11 - a01 * a01
    if det <= 1e-12:
        a00 += 10.0 * r + 1e-06
        a11 += 10.0 * r + 1e-06
        det = max(a00 * a11 - a01 * a01, 1e-12)
    return (a11 / det, -a01 / det, a00 / det)

@njit(cache=True)
def _ilqr_backward_update_fill_nb(A, B, lx, lxx, lu, luu, Vx, Vxx, regularization, krow, Krow, Qx, Qu, Qxx, Quu, Qux, VxxA, VxxB, Vx_new, Vxx_new):
    nx = Vx.shape[0]
    for r in range(nx):
        for j in range(nx):
            value = 0.0
            for q in range(nx):
                value += Vxx[r, q] * A[q, j]
            VxxA[r, j] = value
        for a in range(2):
            value = 0.0
            for q in range(nx):
                value += Vxx[r, q] * B[q, a]
            VxxB[r, a] = value
    for i in range(nx):
        value = lx[i]
        for r in range(nx):
            value += A[r, i] * Vx[r]
        Qx[i] = value
    for a in range(2):
        value = lu[a]
        for r in range(nx):
            value += B[r, a] * Vx[r]
        Qu[a] = value
    for i in range(nx):
        for j in range(nx):
            value = lxx[i, j]
            for r in range(nx):
                value += A[r, i] * VxxA[r, j]
            Qxx[i, j] = value
    for a in range(2):
        for b in range(2):
            value = luu[a, b]
            for r in range(nx):
                value += B[r, a] * VxxB[r, b]
            Quu[a, b] = value
    for a in range(2):
        for j in range(nx):
            value = 0.0
            for r in range(nx):
                value += B[r, a] * VxxA[r, j]
            Qux[a, j] = value
    inv00, inv01, inv11 = _invert_regularized_2x2_ilqr_nb(Quu[0, 0], 0.5 * (Quu[0, 1] + Quu[1, 0]), Quu[1, 1], regularization)
    krow[0] = -(inv00 * Qu[0] + inv01 * Qu[1])
    krow[1] = -(inv01 * Qu[0] + inv11 * Qu[1])
    for j in range(nx):
        Krow[0, j] = -(inv00 * Qux[0, j] + inv01 * Qux[1, j])
        Krow[1, j] = -(inv01 * Qux[0, j] + inv11 * Qux[1, j])
    for i in range(nx):
        value = Qx[i]
        for a in range(2):
            value += Krow[a, i] * Qu[a] + Qux[a, i] * krow[a]
            for b in range(2):
                value += Krow[a, i] * Quu[a, b] * krow[b]
        Vx_new[i] = value
    for i in range(nx):
        for j in range(nx):
            value = Qxx[i, j]
            for a in range(2):
                value += Krow[a, i] * Qux[a, j] + Qux[a, i] * Krow[a, j]
                for b in range(2):
                    value += Krow[a, i] * Quu[a, b] * Krow[b, j]
            Vxx_new[i, j] = value
    for i in range(nx):
        Vx[i] = Vx_new[i]
        for j in range(nx):
            Vxx[i, j] = 0.5 * (Vxx_new[i, j] + Vxx_new[j, i])

@njit(cache=True)
def _ackermann_ilqr_linearize_fill_nb(x, u, xnext, A, B, xp, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
    for i in range(7):
        for j in range(7):
            A[i, j] = 0.0
        B[i, 0] = 0.0
        B[i, 1] = 0.0
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    for j in range(2, 7):
        eps = 0.0001 if j == 2 or j == 6 else 0.001
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
        for k in range(7):
            xp[k] = x[k]
        xp[j] += step
        if j == 2:
            xp[j] = _wrap_angle_nb(xp[j])
        values = _dynamic_ackermann_step_nb(xp, u[0], u[1], dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
        for i in range(7):
            diff = values[i] - xnext[i]
            if i == 2:
                diff = _wrap_angle_nb(diff)
            A[i, j] = diff / step
    for j in range(2):
        eps = 0.001
        step = eps
        if j == 0:
            if u[j] + eps > accel_max:
                step = -eps
            elif u[j] - eps < accel_min:
                step = eps
        elif u[j] + eps > steering_rate_max:
            step = -eps
        elif u[j] - eps < steering_rate_min:
            step = eps
        up0 = u[0]
        up1 = u[1]
        if j == 0:
            up0 += step
        else:
            up1 += step
        values = _dynamic_ackermann_step_nb(x, up0, up1, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
        for i in range(7):
            diff = values[i] - xnext[i]
            if i == 2:
                diff = _wrap_angle_nb(diff)
            B[i, j] = diff / step

@njit(cache=True)
def _ackermann_ilqr_backward_nb(X, U, ref, arc, cov_blocks, covariance_floor, mahalanobis_weight, heading_weight, progress_weight, control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight, regularization, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
    H = U.shape[0]
    progress, qx, qy, tx, ty, heading, p00, p01, p11 = _project_ackermann_rollout_ilqr_nb(X, ref, arc, cov_blocks, covariance_floor)
    kff = np.zeros((H, 2), dtype=np.float64)
    Kfb = np.zeros((H, 2, 7), dtype=np.float64)
    Vx = np.zeros(7, dtype=np.float64)
    Vxx = np.zeros((7, 7), dtype=np.float64)
    A = np.empty((7, 7), dtype=np.float64)
    B = np.empty((7, 2), dtype=np.float64)
    xp = np.empty(7, dtype=np.float64)
    lx = np.empty(7, dtype=np.float64)
    lxx = np.empty((7, 7), dtype=np.float64)
    lu = np.empty(2, dtype=np.float64)
    luu = np.empty((2, 2), dtype=np.float64)
    Qx = np.empty(7, dtype=np.float64)
    Qu = np.empty(2, dtype=np.float64)
    Qxx = np.empty((7, 7), dtype=np.float64)
    Quu = np.empty((2, 2), dtype=np.float64)
    Qux = np.empty((2, 7), dtype=np.float64)
    VxxA = np.empty((7, 7), dtype=np.float64)
    VxxB = np.empty((7, 2), dtype=np.float64)
    Vx_new = np.empty(7, dtype=np.float64)
    Vxx_new = np.empty((7, 7), dtype=np.float64)
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
        _ackermann_ilqr_linearize_fill_nb(X[t], U[t], X[t + 1], A, B, xp, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
        for i in range(7):
            lx[i] = 0.0
            for j in range(7):
                lxx[i, j] = 0.0
        lu[0] = 2.0 * control_accel_weight * U[t, 0]
        lu[1] = 2.0 * control_steering_rate_weight * U[t, 1]
        luu[0, 0] = 2.0 * control_accel_weight
        luu[0, 1] = 0.0
        luu[1, 0] = 0.0
        luu[1, 1] = 2.0 * control_steering_rate_weight
        dx = X[t, 0] - qx[t]
        dy = X[t, 1] - qy[t]
        mx = p00[t] * dx + p01[t] * dy
        my = p01[t] * dx + p11[t] * dy
        eh = _wrap_angle_nb(X[t, 2] - heading[t])
        lx[0] = 2.0 * mahalanobis_weight * mx - progress_weight * tx[t]
        lx[1] = 2.0 * mahalanobis_weight * my - progress_weight * ty[t]
        lx[2] = 2.0 * heading_weight * eh
        lxx[0, 0] = 2.0 * mahalanobis_weight * p00[t]
        lxx[0, 1] = 2.0 * mahalanobis_weight * p01[t]
        lxx[1, 0] = lxx[0, 1]
        lxx[1, 1] = 2.0 * mahalanobis_weight * p11[t]
        lxx[2, 2] = 2.0 * heading_weight
        _ilqr_backward_update_fill_nb(A, B, lx, lxx, lu, luu, Vx, Vxx, regularization, kff[t], Kfb[t], Qx, Qu, Qxx, Quu, Qux, VxxA, VxxB, Vx_new, Vxx_new)
    return (kff, Kfb)

@njit(cache=True)
def _ackermann_ilqr_forward_update_nb(x0, U, X, kff, Kfb, alpha, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
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
        u0, values = _ackermann_forward_only_step_ilqr_nb(Xnew[t], u0, u1, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
        Unew[t, 0] = u0
        Unew[t, 1] = u1
        for j in range(7):
            Xnew[t + 1, j] = values[j]
    return (Unew, Xnew)

@njit(cache=True)
def _ackermann_linearize_trajectory_nb(X, U, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
    """Return the A/B sequence from the same local model used by Ackermann iLQR."""
    H = U.shape[0]
    A = np.empty((H, 7, 7), dtype=np.float64)
    B = np.empty((H, 7, 2), dtype=np.float64)
    xp = np.empty(7, dtype=np.float64)
    for t in range(H):
        _ackermann_ilqr_linearize_fill_nb(X[t], U[t], X[t + 1], A[t], B[t], xp, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
    return (A, B)

@njit(cache=True, nogil=True)
def _ackermann_ilqr_nominal_and_positions_nb(x0, ref, cov_blocks, horizon, iterations, line_search_steps, mahalanobis_weight, covariance_floor, heading_weight, progress_weight, control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight, regularization, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
    Uzero = np.zeros((horizon, 2), dtype=np.float64)
    positions = np.zeros(horizon, dtype=np.float64)
    if ref.shape[0] < 2:
        return (Uzero, positions, np.zeros((horizon, 7, 7), dtype=np.float64), np.zeros((horizon, 7, 2), dtype=np.float64), np.zeros((horizon, 2), dtype=np.float64))
    arc = _path_arc_lengths_ilqr_nb(ref)
    if arc[arc.shape[0] - 1] <= 1e-10:
        return (Uzero, positions, np.zeros((horizon, 7, 7), dtype=np.float64), np.zeros((horizon, 7, 2), dtype=np.float64), np.zeros((horizon, 2), dtype=np.float64))
    ilqr_steering_rate_min = max(steering_rate_min, -8.0)
    ilqr_steering_rate_max = min(steering_rate_max, 8.0)
    U = _ackermann_ilqr_initial_controls_nb(x0, ref, arc, horizon, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, ilqr_steering_rate_min, ilqr_steering_rate_max)
    X = rollout_ackermann_single_nb(x0, U, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, ilqr_steering_rate_min, ilqr_steering_rate_max)
    best_cost = _ackermann_ilqr_total_cost_nb(X, U, ref, arc, cov_blocks, covariance_floor, mahalanobis_weight, heading_weight, progress_weight, control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight)
    for _ in range(max(1, int(iterations))):
        kff, Kfb = _ackermann_ilqr_backward_nb(X, U, ref, arc, cov_blocks, covariance_floor, mahalanobis_weight, heading_weight, progress_weight, control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight, regularization, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, ilqr_steering_rate_min, ilqr_steering_rate_max)
        improved = False
        alpha = 1.0
        for _ls in range(max(1, int(line_search_steps))):
            Utrial, Xtrial = _ackermann_ilqr_forward_update_nb(x0, U, X, kff, Kfb, alpha, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, ilqr_steering_rate_min, ilqr_steering_rate_max)
            trial_cost = _ackermann_ilqr_total_cost_nb(Xtrial, Utrial, ref, arc, cov_blocks, covariance_floor, mahalanobis_weight, heading_weight, progress_weight, control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight)
            if math.isfinite(trial_cost) and trial_cost < best_cost - 1e-09:
                U = Utrial
                X = Xtrial
                best_cost = trial_cost
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
    final_progress, _, _, _, _, _, _, _, _ = _project_ackermann_rollout_ilqr_nb(X, ref, arc, cov_blocks, covariance_floor)
    for t in range(horizon):
        positions[t] = final_progress[t]
    A, B = _ackermann_linearize_trajectory_nb(X, U, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, ilqr_steering_rate_min, ilqr_steering_rate_max)
    ilqr_xy = np.empty((horizon, 2), dtype=np.float64)
    for t in range(horizon):
        ilqr_xy[t, 0] = X[t, 0]
        ilqr_xy[t, 1] = X[t, 1]
    return (U, positions, A, B, ilqr_xy)


@njit(cache=True, parallel=True)
def _ackermann_ilqr_batch_nb(x0, refs, cov_blocks, lengths, use_covariance, horizon, iterations, line_search_steps, mahalanobis_weight, covariance_floor, fallback_variance, heading_weight, progress_weight, control_accel_weight, control_steering_rate_weight, terminal_position_weight, terminal_velocity_weight, regularization, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
    count = refs.shape[0]
    controls = np.zeros((count, horizon, 2), dtype=np.float64)
    positions = np.zeros((count, horizon), dtype=np.float64)
    As = np.zeros((count, horizon, 7, 7), dtype=np.float64)
    Bs = np.zeros((count, horizon, 7, 2), dtype=np.float64)
    trajectories = np.zeros((count, horizon, 2), dtype=np.float64)
    for m in prange(count):
        n = int(lengths[m])
        if n < 1:
            continue
        ref = refs[m, :n]
        if use_covariance:
            cov = cov_blocks[m, :n]
        else:
            cov = np.zeros((n, 2, 2), dtype=np.float64)
            for i in range(n):
                cov[i, 0, 0] = fallback_variance
                cov[i, 1, 1] = fallback_variance
        U, pos, A, B, xy = _ackermann_ilqr_nominal_and_positions_nb(
            x0, ref, cov, horizon, iterations, line_search_steps,
            mahalanobis_weight, covariance_floor, heading_weight, progress_weight,
            control_accel_weight, control_steering_rate_weight,
            terminal_position_weight, terminal_velocity_weight, regularization,
            dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
            cornering_stiffness_front, cornering_stiffness_rear,
            tire_friction_coefficient, gravity, aerodynamic_drag_coefficient,
            rolling_resistance_force, minimum_tire_speed, dynamics_substeps,
            v_min, v_max, lateral_velocity_limit, yaw_rate_limit,
            accel_min, accel_max, steering_min, steering_max,
            steering_rate_min, steering_rate_max,
        )
        controls[m] = U
        positions[m] = pos
        As[m] = A
        Bs[m] = B
        trajectories[m] = xy
    return controls, positions, As, Bs, trajectories

@njit(cache=True, parallel=True)
def clip_ackermann_controls_inplace_nb(U, accel_min, accel_max, steering_rate_min, steering_rate_max):
    flat = U.reshape((-1, 2))
    for i in prange(flat.shape[0]):
        flat[i, 0] = min(max(flat[i, 0], accel_min), accel_max)
        flat[i, 1] = min(max(flat[i, 1], steering_rate_min), steering_rate_max)
    return U

@njit(cache=True, parallel=True)
def _spg_from_jacobians_nb(A, B, cov, lookahead_steps, pseudoinverse_damping, covariance_jitter):
    """Project planar trajectory covariance using the iLQR dynamics Jacobians."""
    H = B.shape[0]
    nx = B.shape[1]
    out = np.empty((H, 2, 2), dtype=np.float64)
    damp2 = pseudoinverse_damping * pseudoinverse_damping
    for t in prange(H):
        ell = min(max(1, int(lookahead_steps)), H - t)
        S = np.empty((nx, 2), dtype=np.float64)
        tmpS = np.empty((nx, 2), dtype=np.float64)
        for r in range(nx):
            S[r, 0] = B[t, r, 0]
            S[r, 1] = B[t, r, 1]
        for k in range(1, ell):
            At = A[t + k]
            for r in range(nx):
                s0 = 0.0
                s1 = 0.0
                for c in range(nx):
                    s0 += At[r, c] * S[c, 0]
                    s1 += At[r, c] * S[c, 1]
                tmpS[r, 0] = s0
                tmpS[r, 1] = s1
            swap = S
            S = tmpS
            tmpS = swap
        j00 = S[0, 0]
        j01 = S[0, 1]
        j10 = S[1, 0]
        j11 = S[1, 1]
        m00 = j00 * j00 + j01 * j01 + damp2
        m01 = j00 * j10 + j01 * j11
        m11 = j10 * j10 + j11 * j11 + damp2
        det = m00 * m11 - m01 * m01
        if abs(det) < 1e-15:
            det = 1e-15 if det >= 0.0 else -1e-15
        im00 = m11 / det
        im01 = -m01 / det
        im11 = m00 / det
        d00 = j00 * im00 + j10 * im01
        d01 = j00 * im01 + j10 * im11
        d10 = j01 * im00 + j11 * im01
        d11 = j01 * im01 + j11 * im11
        c00 = cov[t, 0, 0]
        c01 = 0.5 * (cov[t, 0, 1] + cov[t, 1, 0])
        c11 = cov[t, 1, 1]
        q00 = d00 * c00 + d01 * c01
        q01 = d00 * c01 + d01 * c11
        q10 = d10 * c00 + d11 * c01
        q11 = d10 * c01 + d11 * c11
        out[t, 0, 0] = q00 * d00 + q01 * d01 + covariance_jitter
        off01 = q00 * d10 + q01 * d11
        off10 = q10 * d00 + q11 * d01
        off = 0.5 * (off01 + off10)
        out[t, 0, 1] = off
        out[t, 1, 0] = off
        out[t, 1, 1] = q10 * d10 + q11 * d11 + covariance_jitter
    return out

def ackermann_step(x: Array, u: Array, cfg: MPPIConfig) -> Array:
    state = np.asarray(x, dtype=np.float64)
    control = np.asarray(u, dtype=np.float64)
    values = _dynamic_ackermann_step_nb(state, float(control[0]), float(control[1]), *_dynamic_model_arguments(cfg))
    return np.asarray(values, dtype=np.float64)

def _dynamic_model_arguments(cfg: MPPIConfig) -> Tuple[float, ...]:
    return (float(cfg.dt), float(cfg.front_axle_distance), float(cfg.rear_axle_distance), float(cfg.mass), float(cfg.yaw_inertia), float(cfg.cornering_stiffness_front), float(cfg.cornering_stiffness_rear), float(cfg.tire_friction_coefficient), float(cfg.gravity), float(cfg.aerodynamic_drag_coefficient), float(cfg.rolling_resistance_force), float(cfg.minimum_tire_speed), int(cfg.dynamics_substeps), float(cfg.v_min), float(cfg.v_max), float(cfg.lateral_velocity_limit), float(cfg.yaw_rate_limit), float(cfg.accel_min), float(cfg.accel_max), float(cfg.steering_min), float(cfg.steering_max), float(cfg.steering_rate_min), float(cfg.steering_rate_max))

def rollout_ackermann(x0: Array, U: Array, cfg: MPPIConfig) -> Array:
    return rollout_ackermann_single_nb(np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), *_dynamic_model_arguments(cfg))

def obstacle_circles_to_arrays(obstacle_circles: List[Tuple[Array, float]]) -> Tuple[Array, Array]:
    if not obstacle_circles:
        return (np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64))
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    return (centers, radii)

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
        raise ValueError(f'cov_blocks must have shape ({n},2,2), got {cov.shape}')
    cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))
    return np.ascontiguousarray(cov)

def _nominal_controls_full(x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array]=None):
    path = np.ascontiguousarray(np.asarray(ref, dtype=np.float64))
    cov = _prepare_ilqr_covariance(path, cov_blocks, cfg)
    return _ackermann_ilqr_nominal_and_positions_nb(np.asarray(x0, dtype=np.float64), path, cov, int(cfg.horizon), int(cfg.prior_ilqr_iterations), int(cfg.prior_ilqr_line_search_steps), float(cfg.prior_ilqr_mahalanobis_weight), float(cfg.prior_ilqr_covariance_floor), float(cfg.prior_ilqr_heading_weight), float(cfg.prior_ilqr_progress_weight), float(cfg.prior_ilqr_control_accel_weight), float(cfg.prior_ilqr_control_steering_rate_weight), float(cfg.w_terminal_position), float(cfg.w_terminal_velocity), float(cfg.prior_ilqr_regularization), *_dynamic_model_arguments(cfg))

def nominal_controls_and_arc_positions(x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array]=None) -> Tuple[Array, Array]:
    controls, positions, _, _, _ = _nominal_controls_full(x0, ref, cfg, cov_blocks)
    return (controls, positions)

def nominal_controls_and_arc_positions_with_jacobians(x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array]=None, initial_controls: Optional[Array]=None):
    del initial_controls
    controls, positions, A, B, _ = _nominal_controls_full(x0, ref, cfg, cov_blocks)
    return (controls, positions, A, B)

def nominal_controls_and_arc_positions_with_trajectory(x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array]=None):
    controls, positions, _, _, ilqr_xy = _nominal_controls_full(x0, ref, cfg, cov_blocks)
    return (controls, positions, ilqr_xy)

def nominal_controls_and_arc_positions_with_jacobians_and_trajectory(x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array]=None, initial_controls: Optional[Array]=None):
    del initial_controls
    return _nominal_controls_full(x0, ref, cfg, cov_blocks)

def prior_control_arc_positions(x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array]=None) -> Array:
    _, positions = nominal_controls_and_arc_positions(x0, ref, cfg, cov_blocks)
    return positions

def nominal_controls_to_track_path(x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array]=None) -> Array:
    controls, _ = nominal_controls_and_arc_positions(x0, ref, cfg, cov_blocks)
    return controls

def batch_nominal_solutions(x0: Array, refs: Array, lengths: Array, cfg: MPPIConfig, cov_blocks: Optional[Array]=None) -> Tuple[Array, Array, Array, Array, Array]:
    packed_refs = np.ascontiguousarray(np.asarray(refs, dtype=np.float64))
    packed_lengths = np.ascontiguousarray(np.asarray(lengths, dtype=np.int64).reshape(-1))
    if packed_refs.ndim != 3 or packed_refs.shape[2] != 2:
        raise ValueError(f'refs must have shape (M,K,2), got {packed_refs.shape}')
    if packed_lengths.shape[0] != packed_refs.shape[0]:
        raise ValueError('lengths must contain one entry per reference path.')
    if cov_blocks is None:
        packed_cov = np.zeros((packed_refs.shape[0], packed_refs.shape[1], 2, 2), dtype=np.float64)
        use_covariance = False
    else:
        packed_cov = np.ascontiguousarray(np.asarray(cov_blocks, dtype=np.float64))
        if packed_cov.shape != (packed_refs.shape[0], packed_refs.shape[1], 2, 2):
            raise ValueError(f'cov_blocks must have shape {(packed_refs.shape[0], packed_refs.shape[1], 2, 2)}, got {packed_cov.shape}')
        use_covariance = True
    fallback_variance = float(cfg.prior_ilqr_covariance_fallback_std) ** 2
    return _ackermann_ilqr_batch_nb(
        np.ascontiguousarray(np.asarray(x0, dtype=np.float64)),
        packed_refs,
        packed_cov,
        packed_lengths,
        use_covariance,
        int(cfg.horizon),
        int(cfg.prior_ilqr_iterations),
        int(cfg.prior_ilqr_line_search_steps),
        float(cfg.prior_ilqr_mahalanobis_weight),
        float(cfg.prior_ilqr_covariance_floor),
        fallback_variance,
        float(cfg.prior_ilqr_heading_weight),
        float(cfg.prior_ilqr_progress_weight),
        float(cfg.prior_ilqr_control_accel_weight),
        float(cfg.prior_ilqr_control_steering_rate_weight),
        float(cfg.w_terminal_position),
        float(cfg.w_terminal_velocity),
        float(cfg.prior_ilqr_regularization),
        *_dynamic_model_arguments(cfg),
    )


def nominal_controls_batch_to_track_paths(x0: Array, refs: Array, cfg: MPPIConfig, cov_blocks: Optional[Array]=None) -> Array:
    packed_refs = np.ascontiguousarray(np.asarray(refs, dtype=np.float64))
    if packed_refs.ndim != 3 or packed_refs.shape[2] != 2:
        raise ValueError(f'refs must have shape (M,K,2), got {packed_refs.shape}')
    lengths = np.full(packed_refs.shape[0], packed_refs.shape[1], dtype=np.int64)
    controls, _, _, _, _ = batch_nominal_solutions(x0, packed_refs, lengths, cfg, cov_blocks)
    return controls


def project_control_covariances_from_jacobians(A: Array, B: Array, covariances: Array, cfg: MPPIConfig) -> Array:
    """Project trajectory covariance with the local A/B model returned by iLQR."""
    return _spg_from_jacobians_nb(np.ascontiguousarray(np.asarray(A, dtype=np.float64)), np.ascontiguousarray(np.asarray(B, dtype=np.float64)), np.ascontiguousarray(np.asarray(covariances, dtype=np.float64)), int(cfg.spg_lookahead_steps), float(cfg.spg_pseudoinverse_damping), float(cfg.spg_covariance_jitter))

def sensitivity_projected_control_covariances(x_current: Array, nominal_controls: Array, position_covariances: Array, cfg: MPPIConfig) -> Array:
    """Compatibility wrapper using the same local linearization model as iLQR."""
    x0 = np.asarray(x_current, dtype=np.float64)
    nominal = np.ascontiguousarray(np.asarray(nominal_controls, dtype=np.float64))
    X = rollout_ackermann_single_nb(x0, nominal, *_dynamic_model_arguments(cfg))
    ilqr_steering_rate_min = max(float(cfg.steering_rate_min), -8.0)
    ilqr_steering_rate_max = min(float(cfg.steering_rate_max), 8.0)
    args = list(_dynamic_model_arguments(cfg))
    args[-2] = ilqr_steering_rate_min
    args[-1] = ilqr_steering_rate_max
    A, B = _ackermann_linearize_trajectory_nb(X, nominal, *args)
    return project_control_covariances_from_jacobians(A, B, position_covariances, cfg)

def control_noise_scale(cfg: MPPIConfig) -> Array:
    return np.asarray([cfg.noise_accel, cfg.noise_steering_rate], dtype=np.float64)

def clip_control_batch_inplace(controls: Array, cfg: MPPIConfig) -> Array:
    U = np.ascontiguousarray(np.asarray(controls, dtype=np.float64))
    if U.size == 0:
        return U
    return clip_ackermann_controls_inplace_nb(U, float(cfg.accel_min), float(cfg.accel_max), float(cfg.steering_rate_min), float(cfg.steering_rate_max))

def clip_control_batch(controls: Array, cfg: MPPIConfig) -> Array:
    return clip_control_batch_inplace(np.ascontiguousarray(np.asarray(controls, dtype=np.float64)), cfg)

def pack_obstacle_circles(obstacle_circles) -> Tuple[Array, Array]:
    return obstacle_circles_to_arrays(list(obstacle_circles))

vehicle_step = ackermann_step

def rollout_single(x_current: Array, controls: Array, cfg: MPPIConfig) -> Array:
    return rollout_ackermann(x_current, controls, cfg)

def project_control_covariances(x_current: Array, nominal: Array, covariances: Array, cfg: MPPIConfig) -> Array:
    return sensitivity_projected_control_covariances(x_current, nominal, covariances, cfg)

def apply_final_output(x_current: Array, control: Array, previous_control: Optional[Array], obstacle_circles, goal: Array, cfg: MPPIConfig) -> Array:
    """Execute the same bounded command that the rollout model evaluates.

    MPPI rollouts use the absolute actuator bounds directly and do not impose the
    previous-command slew limits.  Applying those slew limits only after MPPI
    created a prediction/execution mismatch: the optimized first control could be
    very different from the control actually propagated by the simulator.

    Keep execution consistent with the rollout dynamics by applying only the same
    absolute acceleration and steering-rate bounds here.
    """
    del x_current, previous_control, obstacle_circles, goal
    command = np.asarray(control, dtype=np.float64).reshape(1, 2).copy()
    return clip_ackermann_controls_inplace_nb(command, float(cfg.accel_min), float(cfg.accel_max), float(cfg.steering_rate_min), float(cfg.steering_rate_max))[0]
