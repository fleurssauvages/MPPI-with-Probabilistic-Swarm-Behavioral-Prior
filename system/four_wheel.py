from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from . import ackermann

Array = np.ndarray
MODEL_NAME = 'four_wheel'
DISPLAY_NAME = 'Four-wheel'
STATE_DIM = 13
BODY_COLOR = '#d6278f'


@dataclass
class MPPIConfig(ackermann.MPPIConfig):
    track_width: float = 0.30
    wheel_radius: float = 0.08
    wheel_inertia: float = 0.025
    longitudinal_tire_stiffness: float = 100.0
    roll_inertia: float = 0.38
    roll_stiffness: float = 32.0
    roll_damping: float = 4.5
    cg_height: float = 0.15
    roll_center_height: float = 0.05
    wheel_damping: float = 0.004
    wheel_speed_limit: float = 95.0
    drive_bias_front: float = 0.45
    minimum_normal_load_fraction: float = 0.02
    roll_angle_limit: float = 0.45
    roll_rate_limit: float = 5.0
    dynamics_substeps: int = 6
    prior_ilqr_regularization: float = 0.08
    prior_ilqr_roll_weight: float = 0.5
    prior_ilqr_roll_rate_weight: float = 0.05

    def __post_init__(self) -> None:
        super().__post_init__()
        positive = {
            'track_width': self.track_width,
            'wheel_radius': self.wheel_radius,
            'wheel_inertia': self.wheel_inertia,
            'longitudinal_tire_stiffness': self.longitudinal_tire_stiffness,
            'roll_inertia': self.roll_inertia,
            'roll_stiffness': self.roll_stiffness,
            'roll_damping': self.roll_damping,
            'cg_height': self.cg_height,
            'wheel_speed_limit': self.wheel_speed_limit,
            'prior_ilqr_roll_weight': self.prior_ilqr_roll_weight,
            'prior_ilqr_roll_rate_weight': self.prior_ilqr_roll_rate_weight,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError('These values must be positive: ' + ', '.join(invalid))
        if not 0.0 <= self.drive_bias_front <= 1.0:
            raise ValueError('drive_bias_front must be in [0, 1].')
        if not 0.0 <= self.minimum_normal_load_fraction < 0.25:
            raise ValueError('minimum_normal_load_fraction must be in [0, 0.25).')


@njit(cache=True, inline='always')
def _wrap_angle_nb(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@njit(cache=True, inline='always')
def _combined_tire_force_nb(
    kappa: float,
    alpha: float,
    normal_load: float,
    longitudinal_stiffness: float,
    cornering_stiffness: float,
    friction_coefficient: float,
) -> tuple[float, float]:
    limit = max(friction_coefficient * normal_load, 1e-6)
    fx = limit * math.tanh(longitudinal_stiffness * kappa / limit)
    fy = -limit * math.tanh(cornering_stiffness * alpha / limit)
    ratio = math.sqrt((fx / limit) ** 2 + (fy / limit) ** 2)
    if ratio > 1.0:
        scale = 1.0 / ratio
        fx *= scale
        fy *= scale
    return fx, fy


@njit(cache=True)
def _dynamic_four_wheel_step_nb(
    state,
    accel_cmd,
    steering_rate_cmd,
    dt,
    front_axle_distance,
    rear_axle_distance,
    mass,
    yaw_inertia,
    cornering_stiffness_front,
    cornering_stiffness_rear,
    tire_friction_coefficient,
    gravity,
    aerodynamic_drag_coefficient,
    rolling_resistance_force,
    minimum_tire_speed,
    dynamics_substeps,
    v_min,
    v_max,
    lateral_velocity_limit,
    yaw_rate_limit,
    accel_min,
    accel_max,
    steering_min,
    steering_max,
    steering_rate_min,
    steering_rate_max,
    track_width,
    wheel_radius,
    wheel_inertia,
    longitudinal_tire_stiffness,
    roll_inertia,
    roll_stiffness,
    roll_damping,
    cg_height,
    roll_center_height,
    wheel_damping,
    wheel_speed_limit,
    drive_bias_front,
    minimum_normal_load_fraction,
    roll_angle_limit,
    roll_rate_limit,
):
    px = state[0]
    py = state[1]
    psi = state[2]
    vx = state[3]
    vy = state[4]
    yaw_rate = state[5]
    steering = state[6]
    roll = state[7]
    roll_rate = state[8]
    omega_fl = state[9]
    omega_fr = state[10]
    omega_rl = state[11]
    omega_rr = state[12]
    accel = min(max(accel_cmd, accel_min), accel_max)
    steering_rate = min(max(steering_rate_cmd, steering_rate_min), steering_rate_max)
    h = dt / max(1, int(dynamics_substeps))
    wheelbase = max(front_axle_distance + rear_axle_distance, 1e-9)
    half_track = 0.5 * track_width
    front_fraction = rear_axle_distance / wheelbase
    rear_fraction = front_axle_distance / wheelbase
    min_normal = max(1e-6, minimum_normal_load_fraction * mass * gravity)
    for _ in range(max(1, int(dynamics_substeps))):
        longitudinal_transfer = mass * accel * cg_height / wheelbase
        front_total = mass * gravity * front_fraction - longitudinal_transfer
        rear_total = mass * gravity * rear_fraction + longitudinal_transfer
        lateral_estimate = vx * yaw_rate
        suspension_moment = roll_stiffness * roll + roll_damping * roll_rate
        lateral_transfer = (mass * lateral_estimate * cg_height + suspension_moment) / max(track_width, 1e-6)
        front_transfer = lateral_transfer * front_fraction
        rear_transfer = lateral_transfer * rear_fraction
        fz0 = max(min_normal, 0.5 * front_total - 0.5 * front_transfer)
        fz1 = max(min_normal, 0.5 * front_total + 0.5 * front_transfer)
        fz2 = max(min_normal, 0.5 * rear_total - 0.5 * rear_transfer)
        fz3 = max(min_normal, 0.5 * rear_total + 0.5 * rear_transfer)
        total_fx = 0.0
        total_fy = 0.0
        yaw_moment = 0.0
        tire_fx = np.empty(4, dtype=np.float64)
        omegas = np.empty(4, dtype=np.float64)
        omegas[0] = omega_fl
        omegas[1] = omega_fr
        omegas[2] = omega_rl
        omegas[3] = omega_rr
        fz = np.empty(4, dtype=np.float64)
        fz[0] = fz0
        fz[1] = fz1
        fz[2] = fz2
        fz[3] = fz3
        for wheel in range(4):
            front = wheel < 2
            left = wheel == 0 or wheel == 2
            xw = front_axle_distance if front else -rear_axle_distance
            yw = half_track if left else -half_track
            delta = steering if front else 0.0
            local_vx = vx - yaw_rate * yw
            local_vy = vy + yaw_rate * xw
            cd = math.cos(delta)
            sd = math.sin(delta)
            wheel_vx = cd * local_vx + sd * local_vy
            wheel_vy = -sd * local_vx + cd * local_vy
            speed = max(abs(wheel_vx), minimum_tire_speed)
            kappa = (wheel_radius * omegas[wheel] - wheel_vx) / speed
            alpha = math.atan2(wheel_vy, speed)
            cornering = 0.5 * (cornering_stiffness_front if front else cornering_stiffness_rear)
            fx_wheel, fy_wheel = _combined_tire_force_nb(
                kappa,
                alpha,
                fz[wheel],
                longitudinal_tire_stiffness,
                cornering,
                tire_friction_coefficient,
            )
            tire_fx[wheel] = fx_wheel
            fx_body = cd * fx_wheel - sd * fy_wheel
            fy_body = sd * fx_wheel + cd * fy_wheel
            total_fx += fx_body
            total_fy += fy_body
            yaw_moment += xw * fy_body - yw * fx_body
        requested_force = mass * accel
        front_torque = 0.5 * requested_force * drive_bias_front * wheel_radius
        rear_torque = 0.5 * requested_force * (1.0 - drive_bias_front) * wheel_radius
        omega_fl_dot = (front_torque - tire_fx[0] * wheel_radius - wheel_damping * omega_fl) / wheel_inertia
        omega_fr_dot = (front_torque - tire_fx[1] * wheel_radius - wheel_damping * omega_fr) / wheel_inertia
        omega_rl_dot = (rear_torque - tire_fx[2] * wheel_radius - wheel_damping * omega_rl) / wheel_inertia
        omega_rr_dot = (rear_torque - tire_fx[3] * wheel_radius - wheel_damping * omega_rr) / wheel_inertia
        total_fx -= aerodynamic_drag_coefficient * vx * abs(vx)
        total_fx -= rolling_resistance_force * math.tanh(vx / 0.1)
        cpsi = math.cos(psi)
        spsi = math.sin(psi)
        px_dot = vx * cpsi - vy * spsi
        py_dot = vx * spsi + vy * cpsi
        vx_dot = total_fx / mass + yaw_rate * vy
        vy_dot = total_fy / mass - yaw_rate * vx
        yaw_accel = yaw_moment / yaw_inertia
        roll_moment = total_fy * max(cg_height - roll_center_height, 0.0)
        roll_moment -= roll_stiffness * roll + roll_damping * roll_rate
        roll_accel = roll_moment / roll_inertia
        px += h * px_dot
        py += h * py_dot
        psi = _wrap_angle_nb(psi + h * yaw_rate)
        vx = min(max(vx + h * vx_dot, v_min), v_max)
        vy = min(max(vy + h * vy_dot, -lateral_velocity_limit), lateral_velocity_limit)
        yaw_rate = min(max(yaw_rate + h * yaw_accel, -yaw_rate_limit), yaw_rate_limit)
        steering = min(max(steering + h * steering_rate, steering_min), steering_max)
        roll = min(max(roll + h * roll_rate, -roll_angle_limit), roll_angle_limit)
        roll_rate = min(max(roll_rate + h * roll_accel, -roll_rate_limit), roll_rate_limit)
        omega_fl = min(max(omega_fl + h * omega_fl_dot, -wheel_speed_limit), wheel_speed_limit)
        omega_fr = min(max(omega_fr + h * omega_fr_dot, -wheel_speed_limit), wheel_speed_limit)
        omega_rl = min(max(omega_rl + h * omega_rl_dot, -wheel_speed_limit), wheel_speed_limit)
        omega_rr = min(max(omega_rr + h * omega_rr_dot, -wheel_speed_limit), wheel_speed_limit)
    return (
        px,
        py,
        psi,
        vx,
        vy,
        yaw_rate,
        steering,
        roll,
        roll_rate,
        omega_fl,
        omega_fr,
        omega_rl,
        omega_rr,
    )


@njit(cache=True)
def rollout_four_wheel_single_nb(x0, controls, *args):
    horizon = controls.shape[0]
    states = np.empty((horizon + 1, STATE_DIM), dtype=np.float64)
    for j in range(STATE_DIM):
        states[0, j] = x0[j]
    for t in range(horizon):
        values = _dynamic_four_wheel_step_nb(states[t], controls[t, 0], controls[t, 1], *args)
        for j in range(STATE_DIM):
            states[t + 1, j] = values[j]
    return states


@njit(cache=True)
def _linearize_trajectory_nb(states, controls, *args):
    horizon = controls.shape[0]
    A = np.empty((horizon, STATE_DIM, STATE_DIM), dtype=np.float64)
    B = np.empty((horizon, STATE_DIM, 2), dtype=np.float64)
    perturbed = np.empty(STATE_DIM, dtype=np.float64)
    for t in range(horizon):
        for j in range(STATE_DIM):
            for k in range(STATE_DIM):
                perturbed[k] = states[t, k]
            eps = 1e-4 if j in (2, 6, 7) else 1e-3
            perturbed[j] += eps
            if j == 2:
                perturbed[j] = _wrap_angle_nb(perturbed[j])
            values = _dynamic_four_wheel_step_nb(perturbed, controls[t, 0], controls[t, 1], *args)
            for i in range(STATE_DIM):
                diff = values[i] - states[t + 1, i]
                if i == 2:
                    diff = _wrap_angle_nb(diff)
                A[t, i, j] = diff / eps
        for j in range(2):
            u0 = controls[t, 0]
            u1 = controls[t, 1]
            eps = 1e-3
            if j == 0:
                u0 += eps
            else:
                u1 += eps
            values = _dynamic_four_wheel_step_nb(states[t], u0, u1, *args)
            for i in range(STATE_DIM):
                diff = values[i] - states[t + 1, i]
                if i == 2:
                    diff = _wrap_angle_nb(diff)
                B[t, i, j] = diff / eps
    return A, B




@njit(cache=True)
def _four_wheel_ilqr_total_cost_nb(
    states,
    controls,
    ref,
    arc,
    cov_blocks,
    covariance_floor,
    mahalanobis_weight,
    heading_weight,
    progress_weight,
    control_accel_weight,
    control_steering_rate_weight,
    terminal_position_weight,
    terminal_velocity_weight,
    roll_weight,
    roll_rate_weight,
):
    cost = ackermann._ackermann_ilqr_total_cost_nb(
        states,
        controls,
        ref,
        arc,
        cov_blocks,
        covariance_floor,
        mahalanobis_weight,
        heading_weight,
        progress_weight,
        control_accel_weight,
        control_steering_rate_weight,
        terminal_position_weight,
        terminal_velocity_weight,
    )
    horizon = controls.shape[0]
    for t in range(horizon):
        cost += roll_weight * states[t, 7] * states[t, 7]
        cost += roll_rate_weight * states[t, 8] * states[t, 8]
    cost += roll_weight * states[horizon, 7] * states[horizon, 7]
    cost += roll_rate_weight * states[horizon, 8] * states[horizon, 8]
    return cost


@njit(cache=True)
def _four_wheel_ilqr_backward_nb(
    states,
    controls,
    ref,
    arc,
    cov_blocks,
    covariance_floor,
    mahalanobis_weight,
    heading_weight,
    progress_weight,
    control_accel_weight,
    control_steering_rate_weight,
    terminal_position_weight,
    terminal_velocity_weight,
    regularization,
    roll_weight,
    roll_rate_weight,
    *args,
):
    horizon = controls.shape[0]
    A, B = _linearize_trajectory_nb(states, controls, *args)
    progress, qx, qy, tx, ty, heading, p00, p01, p11 = ackermann._project_ackermann_rollout_ilqr_nb(
        states, ref, arc, cov_blocks, covariance_floor
    )
    kff = np.zeros((horizon, 2), dtype=np.float64)
    Kfb = np.zeros((horizon, 2, STATE_DIM), dtype=np.float64)
    Vx = np.zeros(STATE_DIM, dtype=np.float64)
    Vxx = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
    ex = states[horizon, 0] - ref[ref.shape[0] - 1, 0]
    ey = states[horizon, 1] - ref[ref.shape[0] - 1, 1]
    Vx[0] = 2.0 * terminal_position_weight * ex
    Vx[1] = 2.0 * terminal_position_weight * ey
    Vxx[0, 0] = 2.0 * terminal_position_weight
    Vxx[1, 1] = 2.0 * terminal_position_weight
    Vx[3] = 2.0 * terminal_velocity_weight * states[horizon, 3]
    Vx[4] = 2.0 * terminal_velocity_weight * states[horizon, 4]
    Vxx[3, 3] = 2.0 * terminal_velocity_weight
    Vxx[4, 4] = 2.0 * terminal_velocity_weight
    Vx[7] = 2.0 * roll_weight * states[horizon, 7]
    Vx[8] = 2.0 * roll_rate_weight * states[horizon, 8]
    Vxx[7, 7] = 2.0 * roll_weight
    Vxx[8, 8] = 2.0 * roll_rate_weight
    lx = np.empty(STATE_DIM, dtype=np.float64)
    lxx = np.empty((STATE_DIM, STATE_DIM), dtype=np.float64)
    lu = np.empty(2, dtype=np.float64)
    luu = np.empty((2, 2), dtype=np.float64)
    Qx = np.empty(STATE_DIM, dtype=np.float64)
    Qu = np.empty(2, dtype=np.float64)
    Qxx = np.empty((STATE_DIM, STATE_DIM), dtype=np.float64)
    Quu = np.empty((2, 2), dtype=np.float64)
    Qux = np.empty((2, STATE_DIM), dtype=np.float64)
    VxxA = np.empty((STATE_DIM, STATE_DIM), dtype=np.float64)
    VxxB = np.empty((STATE_DIM, 2), dtype=np.float64)
    Vx_new = np.empty(STATE_DIM, dtype=np.float64)
    Vxx_new = np.empty((STATE_DIM, STATE_DIM), dtype=np.float64)
    for t in range(horizon - 1, -1, -1):
        for i in range(STATE_DIM):
            lx[i] = 0.0
            for j in range(STATE_DIM):
                lxx[i, j] = 0.0
        lu[0] = 2.0 * control_accel_weight * controls[t, 0]
        lu[1] = 2.0 * control_steering_rate_weight * controls[t, 1]
        luu[0, 0] = 2.0 * control_accel_weight
        luu[0, 1] = 0.0
        luu[1, 0] = 0.0
        luu[1, 1] = 2.0 * control_steering_rate_weight
        dx = states[t, 0] - qx[t]
        dy = states[t, 1] - qy[t]
        mx = p00[t] * dx + p01[t] * dy
        my = p01[t] * dx + p11[t] * dy
        eh = _wrap_angle_nb(states[t, 2] - heading[t])
        lx[0] = 2.0 * mahalanobis_weight * mx - progress_weight * tx[t]
        lx[1] = 2.0 * mahalanobis_weight * my - progress_weight * ty[t]
        lx[2] = 2.0 * heading_weight * eh
        lx[7] = 2.0 * roll_weight * states[t, 7]
        lx[8] = 2.0 * roll_rate_weight * states[t, 8]
        lxx[0, 0] = 2.0 * mahalanobis_weight * p00[t]
        lxx[0, 1] = 2.0 * mahalanobis_weight * p01[t]
        lxx[1, 0] = lxx[0, 1]
        lxx[1, 1] = 2.0 * mahalanobis_weight * p11[t]
        lxx[2, 2] = 2.0 * heading_weight
        lxx[7, 7] = 2.0 * roll_weight
        lxx[8, 8] = 2.0 * roll_rate_weight
        ackermann._ilqr_backward_update_fill_nb(
            A[t], B[t], lx, lxx, lu, luu, Vx, Vxx, regularization,
            kff[t], Kfb[t], Qx, Qu, Qxx, Quu, Qux, VxxA, VxxB, Vx_new, Vxx_new
        )
    return kff, Kfb


@njit(cache=True)
def _four_wheel_ilqr_forward_update_nb(x0, controls, states, kff, Kfb, alpha, *args):
    horizon = controls.shape[0]
    trial_controls = np.empty_like(controls)
    trial_states = np.empty_like(states)
    for j in range(STATE_DIM):
        trial_states[0, j] = x0[j]
    accel_min = args[17]
    accel_max = args[18]
    steering_rate_min = args[21]
    steering_rate_max = args[22]
    for t in range(horizon):
        d0 = np.empty(STATE_DIM, dtype=np.float64)
        for j in range(STATE_DIM):
            d0[j] = trial_states[t, j] - states[t, j]
        d0[2] = _wrap_angle_nb(d0[2])
        u0 = controls[t, 0] + alpha * kff[t, 0]
        u1 = controls[t, 1] + alpha * kff[t, 1]
        for j in range(STATE_DIM):
            u0 += Kfb[t, 0, j] * d0[j]
            u1 += Kfb[t, 1, j] * d0[j]
        u0 = min(max(u0, accel_min), accel_max)
        u1 = min(max(u1, steering_rate_min), steering_rate_max)
        trial_controls[t, 0] = u0
        trial_controls[t, 1] = u1
        values = _dynamic_four_wheel_step_nb(trial_states[t], u0, u1, *args)
        for j in range(STATE_DIM):
            trial_states[t + 1, j] = values[j]
    return trial_controls, trial_states


def _dynamic_model_arguments(cfg: MPPIConfig) -> Tuple[float, ...]:
    return (
        float(cfg.dt),
        float(cfg.front_axle_distance),
        float(cfg.rear_axle_distance),
        float(cfg.mass),
        float(cfg.yaw_inertia),
        float(cfg.cornering_stiffness_front),
        float(cfg.cornering_stiffness_rear),
        float(cfg.tire_friction_coefficient),
        float(cfg.gravity),
        float(cfg.aerodynamic_drag_coefficient),
        float(cfg.rolling_resistance_force),
        float(cfg.minimum_tire_speed),
        int(cfg.dynamics_substeps),
        float(cfg.v_min),
        float(cfg.v_max),
        float(cfg.lateral_velocity_limit),
        float(cfg.yaw_rate_limit),
        float(cfg.accel_min),
        float(cfg.accel_max),
        float(cfg.steering_min),
        float(cfg.steering_max),
        float(cfg.steering_rate_min),
        float(cfg.steering_rate_max),
        float(cfg.track_width),
        float(cfg.wheel_radius),
        float(cfg.wheel_inertia),
        float(cfg.longitudinal_tire_stiffness),
        float(cfg.roll_inertia),
        float(cfg.roll_stiffness),
        float(cfg.roll_damping),
        float(cfg.cg_height),
        float(cfg.roll_center_height),
        float(cfg.wheel_damping),
        float(cfg.wheel_speed_limit),
        float(cfg.drive_bias_front),
        float(cfg.minimum_normal_load_fraction),
        float(cfg.roll_angle_limit),
        float(cfg.roll_rate_limit),
    )


def _ackermann_config(cfg: MPPIConfig) -> ackermann.MPPIConfig:
    values = {field.name: getattr(cfg, field.name) for field in fields(ackermann.MPPIConfig)}
    return ackermann.MPPIConfig(**values)


def _reduced_state(state: Array) -> Array:
    values = np.asarray(state, dtype=np.float64)
    if values.size < 7:
        raise ValueError('Four-wheel state must contain at least seven shared vehicle states.')
    return np.ascontiguousarray(values[:7])


def four_wheel_step(x: Array, u: Array, cfg: MPPIConfig) -> Array:
    state = np.asarray(x, dtype=np.float64)
    control = np.asarray(u, dtype=np.float64)
    return np.asarray(
        _dynamic_four_wheel_step_nb(state, float(control[0]), float(control[1]), *_dynamic_model_arguments(cfg)),
        dtype=np.float64,
    )


vehicle_step = four_wheel_step


def rollout_single(x_current: Array, controls: Array, cfg: MPPIConfig) -> Array:
    return rollout_four_wheel_single_nb(
        np.ascontiguousarray(np.asarray(x_current, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(controls, dtype=np.float64)),
        *_dynamic_model_arguments(cfg),
    )


def _full_model_ilqr_solution(
    x0: Array,
    ref: Array,
    cfg: MPPIConfig,
    cov_blocks: Optional[Array] = None,
    initial_controls: Optional[Array] = None,
    need_jacobians: bool = True,
):
    path = np.ascontiguousarray(np.asarray(ref, dtype=np.float64)[:, :2])
    if len(path) < 2:
        raise ValueError('Reference path must contain at least two points.')
    cov = ackermann._prepare_ilqr_covariance(path, cov_blocks, _ackermann_config(cfg))
    arc = ackermann._path_arc_lengths_ilqr_nb(path)
    state = np.ascontiguousarray(np.asarray(x0, dtype=np.float64))
    if initial_controls is None:
        controls = np.asarray(
            ackermann.nominal_controls_to_track_path(_reduced_state(state), path, _ackermann_config(cfg), cov),
            dtype=np.float64,
        )
    else:
        controls = np.asarray(initial_controls, dtype=np.float64).copy()
    controls = np.ascontiguousarray(clip_control_batch(controls[None, :, :], cfg)[0])
    args = _dynamic_model_arguments(cfg)
    states = rollout_four_wheel_single_nb(state, controls, *args)
    weights = (
        float(cfg.prior_ilqr_covariance_floor),
        float(cfg.prior_ilqr_mahalanobis_weight),
        float(cfg.prior_ilqr_heading_weight),
        float(cfg.prior_ilqr_progress_weight),
        float(cfg.prior_ilqr_control_accel_weight),
        float(cfg.prior_ilqr_control_steering_rate_weight),
        float(cfg.w_terminal_position),
        float(cfg.w_terminal_velocity),
        float(cfg.prior_ilqr_roll_weight),
        float(cfg.prior_ilqr_roll_rate_weight),
    )
    best_cost = _four_wheel_ilqr_total_cost_nb(states, controls, path, arc, cov, *weights)
    for _ in range(int(cfg.prior_ilqr_iterations)):
        kff, Kfb = _four_wheel_ilqr_backward_nb(
            states,
            controls,
            path,
            arc,
            cov,
            float(cfg.prior_ilqr_covariance_floor),
            float(cfg.prior_ilqr_mahalanobis_weight),
            float(cfg.prior_ilqr_heading_weight),
            float(cfg.prior_ilqr_progress_weight),
            float(cfg.prior_ilqr_control_accel_weight),
            float(cfg.prior_ilqr_control_steering_rate_weight),
            float(cfg.w_terminal_position),
            float(cfg.w_terminal_velocity),
            float(cfg.prior_ilqr_regularization),
            float(cfg.prior_ilqr_roll_weight),
            float(cfg.prior_ilqr_roll_rate_weight),
            *args,
        )
        accepted = False
        for line_search in range(int(cfg.prior_ilqr_line_search_steps)):
            alpha = 0.5 ** line_search
            trial_controls, trial_states = _four_wheel_ilqr_forward_update_nb(
                state, controls, states, kff, Kfb, alpha, *args
            )
            trial_cost = _four_wheel_ilqr_total_cost_nb(
                trial_states, trial_controls, path, arc, cov, *weights
            )
            if trial_cost < best_cost:
                controls = np.ascontiguousarray(trial_controls)
                states = np.ascontiguousarray(trial_states)
                best_cost = float(trial_cost)
                accepted = True
                break
        if not accepted:
            break
    progress, _, _, _, _, _, _, _, _ = ackermann._project_ackermann_rollout_ilqr_nb(
        states, path, arc, cov, float(cfg.prior_ilqr_covariance_floor)
    )
    positions = np.ascontiguousarray(np.asarray(progress[:int(cfg.horizon)], dtype=np.float64))
    trajectory = np.ascontiguousarray(states[:int(cfg.horizon), :2])
    if need_jacobians:
        A, B = _linearize_trajectory_nb(states, controls, *args)
    else:
        A = np.zeros((0, 0, 0), dtype=np.float64)
        B = np.zeros((0, 0, 0), dtype=np.float64)
    return controls, positions, A, B, trajectory


def nominal_controls_to_track_path(
    x0: Array,
    ref: Array,
    cfg: MPPIConfig,
    cov_blocks: Optional[Array] = None,
) -> Array:
    controls, _, _, _, _ = _full_model_ilqr_solution(
        x0, ref, cfg, cov_blocks, need_jacobians=False
    )
    return controls


def nominal_controls_and_arc_positions(
    x0: Array,
    ref: Array,
    cfg: MPPIConfig,
    cov_blocks: Optional[Array] = None,
) -> Tuple[Array, Array]:
    controls, positions, _, _, _ = _full_model_ilqr_solution(
        x0, ref, cfg, cov_blocks, need_jacobians=False
    )
    return controls, positions


def nominal_controls_and_arc_positions_with_trajectory(
    x0: Array,
    ref: Array,
    cfg: MPPIConfig,
    cov_blocks: Optional[Array] = None,
):
    controls, positions, _, _, trajectory = _full_model_ilqr_solution(
        x0, ref, cfg, cov_blocks, need_jacobians=False
    )
    return controls, positions, trajectory


def nominal_controls_and_arc_positions_with_jacobians(
    x0: Array,
    ref: Array,
    cfg: MPPIConfig,
    cov_blocks: Optional[Array] = None,
    initial_controls: Optional[Array] = None,
):
    controls, positions, A, B, _ = _full_model_ilqr_solution(
        x0, ref, cfg, cov_blocks, initial_controls=initial_controls, need_jacobians=True
    )
    return controls, positions, A, B


def nominal_controls_and_arc_positions_with_jacobians_and_trajectory(
    x0: Array,
    ref: Array,
    cfg: MPPIConfig,
    cov_blocks: Optional[Array] = None,
    initial_controls: Optional[Array] = None,
):
    return _full_model_ilqr_solution(
        x0, ref, cfg, cov_blocks, initial_controls=initial_controls, need_jacobians=True
    )


def prior_control_arc_positions(
    x0: Array,
    ref: Array,
    cfg: MPPIConfig,
    cov_blocks: Optional[Array] = None,
) -> Array:
    _, positions = nominal_controls_and_arc_positions(x0, ref, cfg, cov_blocks)
    return positions


def _ackermann_batch_initial_controls(
    x0: Array,
    refs: Array,
    lengths: Array,
    cfg: MPPIConfig,
    cov_blocks: Optional[Array],
):
    controls, _, _, _, _ = ackermann.batch_nominal_solutions(
        _reduced_state(x0), refs, lengths, _ackermann_config(cfg), cov_blocks
    )
    return np.ascontiguousarray(np.asarray(controls, dtype=np.float64))


def batch_nominal_controls(
    x0: Array,
    refs: Array,
    lengths: Array,
    cfg: MPPIConfig,
    cov_blocks: Optional[Array] = None,
) -> Array:
    return _ackermann_batch_initial_controls(x0, refs, lengths, cfg, cov_blocks)


def batch_nominal_controls_and_trajectories(
    x0: Array,
    refs: Array,
    lengths: Array,
    cfg: MPPIConfig,
    cov_blocks: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
    controls, positions, _, _, trajectories = batch_nominal_solutions(
        x0, refs, lengths, cfg, cov_blocks, need_final_jacobians=False
    )
    return controls, positions, trajectories


def batch_nominal_solutions(
    x0: Array,
    refs: Array,
    lengths: Array,
    cfg: MPPIConfig,
    cov_blocks: Optional[Array] = None,
    need_final_jacobians: bool = True,
) -> Tuple[Array, Array, Array, Array, Array]:
    packed_refs = np.ascontiguousarray(np.asarray(refs, dtype=np.float64))
    packed_lengths = np.ascontiguousarray(np.asarray(lengths, dtype=np.int64))
    initial = _ackermann_batch_initial_controls(x0, packed_refs, packed_lengths, cfg, cov_blocks)
    count = packed_refs.shape[0]
    horizon = int(cfg.horizon)
    controls = np.zeros((count, horizon, 2), dtype=np.float64)
    positions = np.zeros((count, horizon), dtype=np.float64)
    As = np.zeros((count, horizon, STATE_DIM, STATE_DIM), dtype=np.float64)
    Bs = np.zeros((count, horizon, STATE_DIM, 2), dtype=np.float64)
    trajectories = np.zeros((count, horizon, 2), dtype=np.float64)
    for m in range(count):
        n = int(packed_lengths[m])
        mode_cov = None if cov_blocks is None else np.ascontiguousarray(np.asarray(cov_blocks[m, :n], dtype=np.float64))
        result = _full_model_ilqr_solution(
            x0,
            np.ascontiguousarray(packed_refs[m, :n]),
            cfg,
            mode_cov,
            initial_controls=initial[m],
            need_jacobians=need_final_jacobians,
        )
        controls[m] = result[0]
        positions[m] = result[1]
        trajectories[m] = result[4]
        if need_final_jacobians:
            As[m] = result[2]
            Bs[m] = result[3]
    return controls, positions, As, Bs, trajectories


def nominal_controls_batch_to_track_paths(
    x0: Array,
    refs: Array,
    cfg: MPPIConfig,
    cov_blocks: Optional[Array] = None,
) -> Array:
    packed_refs = np.ascontiguousarray(np.asarray(refs, dtype=np.float64))
    lengths = np.full(packed_refs.shape[0], packed_refs.shape[1], dtype=np.int64)
    return _ackermann_batch_initial_controls(x0, packed_refs, lengths, cfg, cov_blocks)


def project_control_covariances_from_jacobians(
    A: Array,
    B: Array,
    covariances: Array,
    cfg: MPPIConfig,
) -> Array:
    return ackermann.project_control_covariances_from_jacobians(A, B, covariances, cfg)


def project_control_covariances(
    x_current: Array,
    nominal: Array,
    covariances: Array,
    cfg: MPPIConfig,
) -> Array:
    states = rollout_single(x_current, nominal, cfg)
    A, B = _linearize_trajectory_nb(
        states,
        np.ascontiguousarray(np.asarray(nominal, dtype=np.float64)),
        *_dynamic_model_arguments(cfg),
    )
    return project_control_covariances_from_jacobians(A, B, covariances, cfg)


def control_noise_scale(cfg: MPPIConfig) -> Array:
    return np.asarray([cfg.noise_accel, cfg.noise_steering_rate], dtype=np.float64)


def clip_control_batch_inplace(controls: Array, cfg: MPPIConfig) -> Array:
    return ackermann.clip_control_batch_inplace(controls, cfg)


def clip_control_batch(controls: Array, cfg: MPPIConfig) -> Array:
    return ackermann.clip_control_batch(controls, cfg)


def pack_obstacle_circles(obstacle_circles) -> Tuple[Array, Array]:
    return ackermann.pack_obstacle_circles(obstacle_circles)


def obstacle_circles_to_arrays(obstacle_circles: List[Tuple[Array, float]]) -> Tuple[Array, Array]:
    return ackermann.obstacle_circles_to_arrays(obstacle_circles)


def apply_final_output(
    x_current: Array,
    control: Array,
    previous_control: Optional[Array],
    obstacle_circles,
    goal: Array,
    cfg: MPPIConfig,
) -> Array:
    return ackermann.apply_final_output(
        _reduced_state(x_current), control, previous_control, obstacle_circles, goal, cfg
    )
