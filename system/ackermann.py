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
    noise_accel: float = 0.8
    noise_steering_rate: float = 1.0

    # Geometric-prior to Ackermann-control conversion.
    prior_reference_speed: float = 2.0
    prior_tracking_heading_gain: float = 2.5
    prior_tracking_lateral_gain: float = 1.0
    prior_tracking_softening_speed: float = 0.8
    prior_tracking_steering_gain: float = 3.5
    prior_tracking_yaw_rate_gain: float = 0.4
    prior_tracking_terminal_distance_gain: float = 1.8
    # Intercept the geometric prior before switching to local Frenet tracking.
    prior_intercept_lateral_threshold: float = 0.60
    prior_intercept_heading_threshold: float = 0.65
    prior_intercept_lookahead: float = 0.90
    prior_intercept_min_speed: float = 0.40
    prior_intercept_heading_gain: float = 0.35

    vehicle_length: float = 0.81
    vehicle_width: float = 0.36
    collision_substeps: int = 5
    hard_collision_penalty: float = 800000.0
    rollout_goal_tolerance: float = 0.305
    w_boundary: float = 500.0
    boundary_xmin: float = 0.0
    boundary_xmax: float = 10.0
    boundary_ymin: float = 0.0
    boundary_ymax: float = 10.0

    max_delta_accel: float = 1.2
    max_delta_steering_rate: float = 5.2
    enforce_one_step_safety: bool = True
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
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError("These values must be positive: " + ", ".join(invalid))
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

@njit(cache=True)
def rollout_ackermann_batch_nb(x0, U, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        N = U.shape[0]
        H = U.shape[1]
        X = np.zeros((N, H + 1, 7), dtype=np.float64)
        for n in range(N):
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
def nominal_controls_to_track_path_nb(
    x0, ref, horizon, reference_speed, heading_gain, lateral_gain, softening_speed,
    steering_gain, yaw_rate_gain, terminal_distance_gain,
    intercept_lateral_threshold, intercept_heading_threshold, intercept_lookahead,
    intercept_min_speed, intercept_heading_gain,
    dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
    cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
    gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
    minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
    yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
    steering_rate_min, steering_rate_max
):
    U = np.zeros((horizon, 2), dtype=np.float64)
    x = np.zeros(7, dtype=np.float64)
    for j in range(7):
        x[j] = x0[j]
    ref_len = ref.shape[0]
    wheelbase = front_axle_distance + rear_axle_distance
    progress_idx = 0

    terminal_endpoint = False
    if ref_len >= 2:
        ex = ref[ref_len - 1, 0] - ref[ref_len - 2, 0]
        ey = ref[ref_len - 1, 1] - ref[ref_len - 2, 1]
        terminal_endpoint = ex * ex + ey * ey <= 1e-12

    for t in range(horizon):
        if ref_len < 2:
            accel = min(max(-3.0 * x[3], accel_min), accel_max)
            steering_rate = min(max(-steering_gain * x[6] - yaw_rate_gain * x[5], steering_rate_min), steering_rate_max)
            U[t, 0] = accel
            U[t, 1] = steering_rate
            values = _dynamic_ackermann_step_nb(
                x, accel, steering_rate, dt, front_axle_distance, rear_axle_distance,
                mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear,
                tire_friction_coefficient, gravity, aerodynamic_drag_coefficient,
                rolling_resistance_force, minimum_tire_speed, dynamics_substeps,
                v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min,
                accel_max, steering_min, steering_max, steering_rate_min,
                steering_rate_max,
            )
            for j in range(7):
                x[j] = values[j]
            continue

        # Closest forward projection on the prior.
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
                tau = ((x[0] - ref[i, 0]) * sx + (x[1] - ref[i, 1]) * sy) / seg2
                tau = min(max(tau, 0.0), 1.0)
                qx = ref[i, 0] + tau * sx
                qy = ref[i, 1] + tau * sy
            dxq = x[0] - qx
            dyq = x[1] - qy
            d2 = dxq * dxq + dyq * dyq
            if d2 < best_d2:
                best_d2 = d2
                best_seg = i
                best_qx = qx
                best_qy = qy
        progress_idx = best_seg

        # Non-degenerate local tangent.
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
        lateral_error = -ty * (x[0] - best_qx) + tx * (x[1] - best_qy)
        heading_error = _wrap_angle_nb(x[2] - path_heading)
        cross_track_distance = math.sqrt(max(best_d2, 0.0))

        # Interception is used whenever either position or orientation is too far
        # from the local Frenet tube. Once both errors are small, switch to Frenet.
        intercept_mode = (
            cross_track_distance > intercept_lateral_threshold
            or abs(heading_error) > intercept_heading_threshold
        )

        desired_speed = min(max(reference_speed, 0.0), v_max)
        if intercept_mode:
            target_x, target_y = _path_intercept_point_nb(
                ref, best_seg, best_qx, best_qy, intercept_lookahead
            )
            dx = target_x - x[0]
            dy = target_y - x[1]
            target_distance = math.sqrt(dx * dx + dy * dy)
            capture_heading = math.atan2(dy, dx)
            capture_error = _wrap_angle_nb(capture_heading - x[2])
            capture_curvature = 2.0 * math.sin(capture_error) / max(target_distance, 0.35)
            desired_steering = math.atan(wheelbase * capture_curvature) + intercept_heading_gain * capture_error
            desired_steering = min(max(desired_steering, steering_min), steering_max)

            forward = max(0.0, math.cos(capture_error))
            desired_speed = reference_speed * forward * forward
            desired_speed = max(intercept_min_speed, desired_speed)
            desired_speed = min(max(desired_speed, 0.0), v_max)
        else:
            # Frenet feed-forward curvature from three nearby path samples.
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

            feedforward = math.atan(wheelbase * curvature)
            feedback = -heading_gain * heading_error - math.atan2(
                lateral_gain * lateral_error, abs(x[3]) + softening_speed
            )
            desired_steering = min(max(feedforward + feedback, steering_min), steering_max)

        steering_rate = steering_gain * (desired_steering - x[6]) - yaw_rate_gain * x[5]
        steering_rate = min(max(steering_rate, steering_rate_min), steering_rate_max)

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

        accel = min(max(3.0 * (desired_speed - x[3]), accel_min), accel_max)
        U[t, 0] = accel
        U[t, 1] = steering_rate
        values = _dynamic_ackermann_step_nb(
            x, accel, steering_rate, dt, front_axle_distance, rear_axle_distance,
            mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear,
            tire_friction_coefficient, gravity, aerodynamic_drag_coefficient,
            rolling_resistance_force, minimum_tire_speed, dynamics_substeps,
            v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min,
            accel_max, steering_min, steering_max, steering_rate_min,
            steering_rate_max,
        )
        for j in range(7):
            x[j] = values[j]
    return U


@njit(cache=True)
def nominal_controls_to_track_paths_batch_nb(
    x0, refs, horizon, reference_speed, heading_gain, lateral_gain, softening_speed,
    steering_gain, yaw_rate_gain, terminal_distance_gain,
    intercept_lateral_threshold, intercept_heading_threshold, intercept_lookahead,
    intercept_min_speed, intercept_heading_gain,
    dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
    cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
    gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
    minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
    yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
    steering_rate_min, steering_rate_max
):
    count = refs.shape[0]
    output = np.empty((count, horizon, 2), dtype=np.float64)
    for n in range(count):
        controls = nominal_controls_to_track_path_nb(
            x0, refs[n], horizon, reference_speed, heading_gain, lateral_gain, softening_speed,
            steering_gain, yaw_rate_gain, terminal_distance_gain,
            intercept_lateral_threshold, intercept_heading_threshold, intercept_lookahead,
            intercept_min_speed, intercept_heading_gain,
            dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia,
            cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient,
            gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
            minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit,
            yaw_rate_limit, accel_min, accel_max, steering_min, steering_max,
            steering_rate_min, steering_rate_max,
        )
        for t in range(horizon):
            output[n, t, 0] = controls[t, 0]
            output[n, t, 1] = controls[t, 1]
    return output

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
            desired_speed = min(max(0.2 + 2.2 * dist * heading_scale, 0.0), v_max)
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

@njit(cache=True)
def rollout_collision_mask_nb(X, circle_centers, circle_radii, goal, vehicle_length, vehicle_width, hard_collision_clearance, rollout_goal_tolerance):
        N = X.shape[0]
        H = X.shape[1] - 1
        mask = np.zeros(N, dtype=np.bool_)
        goal_radius_sq = rollout_goal_tolerance * rollout_goal_tolerance
        for n in range(N):
            for t in range(H):
                state = X[n, t + 1]
                clearance = minimum_rectangle_circle_clearance_nb(state, circle_centers, circle_radii, vehicle_length, vehicle_width)
                if clearance < hard_collision_clearance:
                    mask[n] = True
                    break
                gx = state[0] - goal[0]
                gy = state[1] - goal[1]
                if gx * gx + gy * gy <= goal_radius_sq:
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

@njit(cache=True)
def standard_mppi_costs_batch_nb(X, U, circle_centers, circle_radii, goal, horizon, vehicle_length, vehicle_width, w_goal, rollout_goal_tolerance, w_obstacle, w_control, w_control_smooth):
        N = U.shape[0]
        H = horizon
        M = circle_radii.shape[0]
        costs = np.zeros(N, dtype=np.float64)
        goal_radius_sq = rollout_goal_tolerance * rollout_goal_tolerance
        for n in range(N):
            cost = 0.0
            arrival_index = H
            for t in range(H):
                px = X[n, t + 1, 0]
                py = X[n, t + 1, 1]
                gx = px - goal[0]
                gy = py - goal[1]
                goal_distance_sq = gx * gx + gy * gy
                cost += w_goal / H * goal_distance_sq
                yaw_rate = X[n, t + 1, 5]
                steering_angle = X[n, t + 1, 6]
                heading = X[n, t + 1, 2]
                for j in range(M):
                    clearance = rectangle_circle_clearance_nb(px, py, heading, circle_centers[j, 0], circle_centers[j, 1], circle_radii[j], vehicle_length, vehicle_width)
                    sp = _softplus_scalar_nb(8.0 * (0.0 - clearance))
                    cost += w_obstacle * sp * sp
                if goal_distance_sq <= goal_radius_sq:
                    arrival_index = t + 1
                    break
            ctrl_cost = 0.0
            for t in range(arrival_index):
                v = U[n, t, 0]
                om = U[n, t, 1]
                ctrl_cost += v * v + 0.15 * om * om
            cost += w_control * ctrl_cost
            smooth_cost = 0.0
            for t in range(max(0, arrival_index - 1)):
                dv = U[n, t + 1, 0] - U[n, t, 0]
                dom = U[n, t + 1, 1] - U[n, t, 1]
                smooth_cost += dv * dv + 0.2 * dom * dom
            cost += w_control_smooth * smooth_cost
            costs[n] = cost
        return costs

@njit(cache=True)
def boundary_penalty_nb(X, xmin, xmax, ymin, ymax, vehicle_length, vehicle_width, w_boundary, collision_substeps, hard_collision_clearance, hard_collision_penalty):
        N = X.shape[0]
        H = X.shape[1] - 1
        extras = np.zeros(N, dtype=np.float64)
        substeps = max(0, int(collision_substeps))
        denominator = float(substeps + 1)
        half_length = 0.5 * vehicle_length
        half_width = 0.5 * vehicle_width
        for n in range(N):
            cost = 0.0
            for t in range(H):
                x0 = X[n, t, 0]
                y0 = X[n, t, 1]
                h0 = X[n, t, 2]
                x1 = X[n, t + 1, 0]
                y1 = X[n, t + 1, 1]
                dh = _wrap_angle_nb(X[n, t + 1, 2] - h0)
                for q in range(1, substeps + 2):
                    alpha = q / denominator
                    px = x0 + alpha * (x1 - x0)
                    py = y0 + alpha * (y1 - y0)
                    heading = _wrap_angle_nb(h0 + alpha * dh)
                    c = abs(math.cos(heading))
                    s = abs(math.sin(heading))
                    extent_x = half_length * c + half_width * s
                    extent_y = half_length * s + half_width * c
                    clearance = min(px - xmin - extent_x, xmax - px - extent_x, py - ymin - extent_y, ymax - py - extent_y)
                    if q <= substeps:
                        sp = _softplus_scalar_nb(8.0 * (0.0 - clearance))
                        cost += w_boundary / denominator * sp * sp
                    if clearance < hard_collision_clearance:
                        penetration = hard_collision_clearance - clearance
                        cost += hard_collision_penalty * (1.0 + penetration * penetration)
            extras[n] = cost
        return extras

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

@njit(cache=True)
def sensitivity_projected_covariances_nb(x0, nominal_controls, position_covariances, lookahead_steps, fd_accel, fd_steering_rate, pseudoinverse_damping, covariance_jitter, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        """Project position covariance into control space with Eq. (26)."""
        horizon = nominal_controls.shape[0]
        nominal_states = rollout_ackermann_single_nb(x0, nominal_controls, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
        projected = np.zeros((horizon, 2, 2), dtype=np.float64)
        damping_sq = pseudoinverse_damping * pseudoinverse_damping
        for t in range(horizon):
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
    _ = cfg
    return bool(np.linalg.norm(np.asarray(state[:2]) - np.asarray(goal)) <= goal_tolerance)

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

def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

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

def _path_tangent_headings(path: Array) -> Array:
    p = np.asarray(path, dtype=np.float64)
    if len(p) <= 1:
        return np.zeros(len(p), dtype=np.float64)
    tangent = np.empty_like(p)
    tangent[0] = p[1] - p[0]
    tangent[-1] = p[-1] - p[-2]
    if len(p) > 2:
        tangent[1:-1] = p[2:] - p[:-2]
    headings = np.arctan2(tangent[:, 1], tangent[:, 0])
    for i in range(1, len(headings)):
        headings[i] = headings[i - 1] + wrap_angle(headings[i] - headings[i - 1])
    return headings

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

def nominal_controls_to_track_path(x0: Array, ref: Array, cfg: MPPIConfig) -> Array:
    return nominal_controls_to_track_path_nb(
        np.asarray(x0, dtype=np.float64),
        np.asarray(ref, dtype=np.float64),
        int(cfg.horizon),
        float(cfg.prior_reference_speed),
        float(cfg.prior_tracking_heading_gain),
        float(cfg.prior_tracking_lateral_gain),
        float(cfg.prior_tracking_softening_speed),
        float(cfg.prior_tracking_steering_gain),
        float(cfg.prior_tracking_yaw_rate_gain),
        float(cfg.prior_tracking_terminal_distance_gain),
        float(cfg.prior_intercept_lateral_threshold),
        float(cfg.prior_intercept_heading_threshold),
        float(cfg.prior_intercept_lookahead),
        float(cfg.prior_intercept_min_speed),
        float(cfg.prior_intercept_heading_gain),
        *_dynamic_model_arguments(cfg),
    )

def nominal_controls_to_track_paths(x0: Array, refs: Array, cfg: MPPIConfig) -> Array:
    reference_batch = np.asarray(refs, dtype=np.float64)
    if reference_batch.ndim != 3 or reference_batch.shape[1:] != (int(cfg.horizon), 2):
        raise ValueError(f"refs must have shape (N,{int(cfg.horizon)},2), got {reference_batch.shape}")
    return nominal_controls_to_track_paths_batch_nb(
        np.asarray(x0, dtype=np.float64),
        reference_batch,
        int(cfg.horizon),
        float(cfg.prior_reference_speed),
        float(cfg.prior_tracking_heading_gain),
        float(cfg.prior_tracking_lateral_gain),
        float(cfg.prior_tracking_softening_speed),
        float(cfg.prior_tracking_steering_gain),
        float(cfg.prior_tracking_yaw_rate_gain),
        float(cfg.prior_tracking_terminal_distance_gain),
        float(cfg.prior_intercept_lateral_threshold),
        float(cfg.prior_intercept_heading_threshold),
        float(cfg.prior_intercept_lookahead),
        float(cfg.prior_intercept_min_speed),
        float(cfg.prior_intercept_heading_gain),
        *_dynamic_model_arguments(cfg),
    )

def nominal_controls_to_goal(x0: Array, goal: Array, cfg: MPPIConfig) -> Array:
    return nominal_controls_to_goal_nb(
        np.asarray(x0, dtype=np.float64), np.asarray(goal, dtype=np.float64), int(cfg.horizon),
        *_dynamic_model_arguments(cfg)
    )

def boundary_penalty(X: Array, cfg: MPPIConfig) -> Array:
    states = np.asarray(X, dtype=np.float64)
    if states.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return boundary_penalty_nb(
        states,
        float(cfg.boundary_xmin), float(cfg.boundary_xmax),
        float(cfg.boundary_ymin), float(cfg.boundary_ymax),
        float(cfg.vehicle_length), float(cfg.vehicle_width), float(cfg.w_boundary),
        int(cfg.collision_substeps), float(cfg.hard_collision_clearance),
        float(cfg.hard_collision_penalty),
    )

def standard_mppi_costs_batch(X: Array, U: Array, obstacle_circles: List[Tuple[Array, float]], goal: Array, cfg: MPPIConfig) -> Array:
    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    costs = standard_mppi_costs_batch_nb(
        np.asarray(X, dtype=np.float64), np.asarray(U, dtype=np.float64), centers, radii,
        np.asarray(goal, dtype=np.float64), int(cfg.horizon),
        float(cfg.vehicle_length), float(cfg.vehicle_width), float(cfg.w_goal),
        float(cfg.rollout_goal_tolerance), float(cfg.w_obstacle), float(cfg.w_control),
        float(cfg.w_control_smooth),
    )
    return costs + boundary_penalty(X, cfg)

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
    return rollout_collision_mask_nb(
        state_batch, centers, radii, np.asarray(goal, dtype=np.float64),
        float(cfg.vehicle_length), float(cfg.vehicle_width),
        float(cfg.hard_collision_clearance), float(cfg.rollout_goal_tolerance),
    )

def update_display_trajectory(info: Dict[str, object], x_current: Array, executed_u: Array, goal: Array, cfg: MPPIConfig) -> None:
    sequence = info.get('planned_control_sequence')
    if sequence is None:
        return
    display_u = np.asarray(sequence, dtype=np.float64).copy()
    if display_u.ndim != 2 or display_u.shape[1] != 2 or len(display_u) == 0:
        return
    display_u[0] = np.asarray(executed_u, dtype=np.float64)
    trajectory = rollout_ackermann(x_current, display_u, cfg)
    distances = np.linalg.norm(trajectory[:, :2] - np.asarray(goal, dtype=np.float64)[None, :], axis=1)
    reached = np.flatnonzero(distances <= float(cfg.rollout_goal_tolerance))
    if len(reached):
        trajectory = trajectory[:int(reached[0]) + 1]
    info['optimal_traj'] = trajectory

def initial_pose(start: Array, goal: Array) -> Array:
    direction = goal - start
    heading = math.atan2(direction[1], direction[0])
    return np.array([start[0], start[1], heading, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Generic controller adapter -------------------------------------------------

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