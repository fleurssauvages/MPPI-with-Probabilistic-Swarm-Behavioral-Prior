from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit, prange

try:
    from . import controller as ctrl
except ImportError:
    import controller as ctrl

Array = np.ndarray
NUMBA_AVAILABLE = True
MODEL_NAME = "planar_quadrotor_payload"
INITIAL_POSE_USES_CONFIG = True
STATE_DIM = 10

ControllerVariant = ctrl.ControllerVariant
Scene = ctrl.Scene
SimulationResult = ctrl.SimulationResult
DynamicWallScenario = ctrl.DynamicWallScenario
MPPIHomotopyMode = ctrl.MPPIHomotopyMode


@dataclass
class MPPIConfig(ctrl.ControllerConfig):
    # Rigid-body parameters from Morbidi & Pisarski (ICRA 2021), Table I.
    mass: float = 1.0
    inertia: float = 0.081
    gravity: float = 9.8066
    drag_x: float = 0.3
    drag_y: float = 0.3
    thrust_factor: float = 3.8281e-5

    rotor_arm: float = 0.3
    rotor_radius: float = 0.07
    body_radius: float = 0.08

    rotor_speed_max: float = 1047.2
    rotor_accel_max: float = 1000.0
    rotor_speed_tracking_gain: float = 5.0
    attitude_kp: float = 12.0
    attitude_kd: float = 7.0

    cable_length: float = 0.6
    payload_mass: float = 0.25
    payload_radius: float = 0.08
    payload_angular_damping: float = 6.0
    payload_swing_weight: float = 0.0
    payload_swing_rate_weight: float = 0.0
    payload_terminal_swing_weight: float = 100.0
    payload_terminal_swing_rate_weight: float = 100.0
    payload_angle_tolerance: float = 0.12
    payload_rate_tolerance: float = 0.25

    dynamics_substeps: int = 5
    max_translational_speed: float = 2.5

    noise_thrust: float = 300.0
    noise_moment: float = 300.0

    spg_fd_thrust: float = 100.0
    spg_fd_moment: float = 100.0

    prior_ilqr_iterations: int = 2
    prior_ilqr_line_search_steps: int = 2
    prior_ilqr_mahalanobis_weight: float = 2.5
    prior_ilqr_covariance_floor: float = 0.12
    prior_ilqr_covariance_fallback_std: float = 0.25
    prior_ilqr_progress_weight: float = 1.5
    prior_ilqr_control_thrust_weight: float = 0.015
    prior_ilqr_control_moment_weight: float = 0.03
    prior_ilqr_regularization: float = 0.02

    enforce_one_step_safety: bool = False
    one_step_safety_clearance: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (
            ("mass", self.mass),
            ("inertia", self.inertia),
            ("gravity", self.gravity),
            ("thrust_factor", self.thrust_factor),
            ("rotor_arm", self.rotor_arm),
            ("rotor_radius", self.rotor_radius),
            ("body_radius", self.body_radius),
            ("rotor_speed_max", self.rotor_speed_max),
            ("rotor_accel_max", self.rotor_accel_max),
            ("rotor_speed_tracking_gain", self.rotor_speed_tracking_gain),
            ("cable_length", self.cable_length),
            ("payload_mass", self.payload_mass),
            ("payload_radius", self.payload_radius),
            ("max_translational_speed", self.max_translational_speed),
            ("prior_ilqr_mahalanobis_weight", self.prior_ilqr_mahalanobis_weight),
            ("prior_ilqr_covariance_floor", self.prior_ilqr_covariance_floor),
            ("prior_ilqr_covariance_fallback_std", self.prior_ilqr_covariance_fallback_std),
            ("prior_ilqr_progress_weight", self.prior_ilqr_progress_weight),
            ("prior_ilqr_control_thrust_weight", self.prior_ilqr_control_thrust_weight),
            ("prior_ilqr_control_moment_weight", self.prior_ilqr_control_moment_weight),
            ("prior_ilqr_regularization", self.prior_ilqr_regularization),
            ("attitude_kp", self.attitude_kp),
            ("attitude_kd", self.attitude_kd),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        for name, value in (
            ("payload_angular_damping", self.payload_angular_damping),
            ("payload_swing_weight", self.payload_swing_weight),
            ("payload_swing_rate_weight", self.payload_swing_rate_weight),
            ("payload_terminal_swing_weight", self.payload_terminal_swing_weight),
            ("payload_terminal_swing_rate_weight", self.payload_terminal_swing_rate_weight),
            ("payload_angle_tolerance", self.payload_angle_tolerance),
            ("payload_rate_tolerance", self.payload_rate_tolerance),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        self.dynamics_substeps = max(1, int(self.dynamics_substeps))
        self.prior_ilqr_iterations = max(1, int(self.prior_ilqr_iterations))
        self.prior_ilqr_line_search_steps = max(1, int(self.prior_ilqr_line_search_steps))
        if self.hover_rotor_speed > self.rotor_speed_max + 1e-9:
            raise ValueError(
                "payload_mass is too large for hover with the configured rotor_speed_max/thrust_factor."
            )

    @property
    def total_drone_radius(self) -> float:
        """Center-to-outer-rotor radius used for collision and display geometry."""
        return float(self.rotor_arm + self.rotor_radius)

    @property
    def total_mass(self) -> float:
        return float(self.mass + self.payload_mass)

    @property
    def payload_weight_newtons(self) -> float:
        return float(self.payload_mass * self.gravity)

    @property
    def hover_thrust(self) -> float:
        return self.total_mass * self.gravity

    @property
    def hover_rotor_speed(self) -> float:
        return math.sqrt(self.hover_thrust / (2.0 * self.thrust_factor))

def _dynamic_args(cfg: MPPIConfig) -> tuple:
    return (
        float(cfg.dt),
        float(cfg.mass),
        float(cfg.inertia),
        float(cfg.gravity),
        float(cfg.drag_x),
        float(cfg.drag_y),
        float(cfg.thrust_factor),
        float(cfg.rotor_arm),
        float(cfg.rotor_speed_max),
        float(cfg.rotor_accel_max),
        float(cfg.cable_length),
        float(cfg.payload_mass),
        float(cfg.payload_angular_damping),
        int(cfg.dynamics_substeps),
    )


@njit(cache=True)
def _wrap_angle_nb(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@njit(cache=True)
def _softplus_scalar_nb(z):
    if z > 40.0:
        return z
    if z < -40.0:
        return math.exp(z)
    return math.log1p(math.exp(z))


@njit(cache=True)
def _clip_quad_values_nb(f, tau, thrust_factor, rotor_arm, rotor_speed_max):
    """Project a desired total thrust/moment onto the feasible rotor-speed box."""
    w2max = rotor_speed_max * rotor_speed_max
    wr2 = (f + tau / rotor_arm) / (2.0 * thrust_factor)
    wl2 = (f - tau / rotor_arm) / (2.0 * thrust_factor)
    wr2 = min(max(wr2, 0.0), w2max)
    wl2 = min(max(wl2, 0.0), w2max)
    return thrust_factor * (wr2 + wl2), rotor_arm * thrust_factor * (wr2 - wl2)


@njit(cache=True)
def _desired_rotor_speeds_nb(f, tau, thrust_factor, rotor_arm, rotor_speed_max):
    w2max = rotor_speed_max * rotor_speed_max
    wr2 = (f + tau / rotor_arm) / (2.0 * thrust_factor)
    wl2 = (f - tau / rotor_arm) / (2.0 * thrust_factor)
    wr2 = min(max(wr2, 0.0), w2max)
    wl2 = min(max(wl2, 0.0), w2max)
    return math.sqrt(wr2), math.sqrt(wl2)


@njit(cache=True)
def _clip_rotor_accel_nb(value, rotor_accel_max):
    return min(max(value, -rotor_accel_max), rotor_accel_max)


@njit(cache=True)
def _clip_quad_control_nb(control, rotor_accel_max):
    out = np.empty(2, dtype=np.float64)
    out[0] = _clip_rotor_accel_nb(control[0], rotor_accel_max)
    out[1] = _clip_rotor_accel_nb(control[1], rotor_accel_max)
    return out


@njit(cache=True)
def _bounded_rotor_accel_nb(w, u, rotor_speed_max, rotor_accel_max):
    u = _clip_rotor_accel_nb(u, rotor_accel_max)
    if w <= 0.0 and u < 0.0:
        return 0.0
    if w >= rotor_speed_max and u > 0.0:
        return 0.0
    return u


@njit(cache=True)
def _payload_position_nb(state, cable_length):
    """Point-mass package position for phi measured from downward vertical."""
    phi = state[8]
    return (
        state[0] + cable_length * math.sin(phi),
        state[1] - cable_length * math.cos(phi),
    )


@njit(cache=True)
def _quad_rhs_nb(x, control, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                 thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
                 cable_length, payload_mass, payload_angular_damping):
    """Coupled planar quadrotor + single-link suspended point-mass dynamics.

    The cable is massless, inextensible, and attached at the quadrotor COM.  With
    q = [x_q, y_q, phi], phi measured from downward vertical, Euler-Lagrange gives

      (m+mL) xdd + mL*l*cos(phi)*phidd - mL*l*sin(phi)*phidot^2 = Fx
      (m+mL) ydd + mL*l*sin(phi)*phidd + mL*l*cos(phi)*phidot^2 = Fy-(m+mL)g
      cos(phi)*xdd + sin(phi)*ydd + l*phidd = -g*sin(phi)

    where Fx/Fy contain rotor thrust and the original quadrotor drag model.  The
    equations below use the equivalent closed-form solution for phidd, xdd, ydd.
    """
    # State: [x, y, vx, vy, theta, omega, omega_r, omega_l, phi, phi_dot]
    wr = min(max(x[6], 0.0), rotor_speed_max)
    wl = min(max(x[7], 0.0), rotor_speed_max)
    ur = _bounded_rotor_accel_nb(wr, control[0], rotor_speed_max, rotor_accel_max)
    ul = _bounded_rotor_accel_nb(wl, control[1], rotor_speed_max, rotor_accel_max)

    f = thrust_factor * (wr * wr + wl * wl)
    tau = rotor_arm * thrust_factor * (wr * wr - wl * wl)

    vx = x[2]
    vy = x[3]
    theta = x[4]
    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    bxx = drag_x_coeff * ctheta * ctheta + drag_y_coeff * stheta * stheta
    bxy = (drag_x_coeff - drag_y_coeff) * ctheta * stheta
    byy = drag_x_coeff * stheta * stheta + drag_y_coeff * ctheta * ctheta
    # The original planar-quadrotor implementation treats these coefficients as
    # acceleration damping. Convert back to force so the mL->0 limit is identical.
    drag_ax = bxx * vx + bxy * vy
    drag_ay = bxy * vx + byy * vy
    force_x = -f * stheta - mass * drag_ax
    force_y = f * ctheta - mass * drag_ay

    phi = x[8]
    phi_dot = x[9]
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    total_mass = mass + payload_mass

    # Closed-form solution of the 3x3 translational/pendulum mass matrix.
    # Strong viscous angular damping removes residual payload swing energy.
    # The damping rate is expressed directly in 1/s so it stays easy to tune
    # when payload mass or cable length changes.
    phi_ddot = (
        -(cphi * force_x + sphi * force_y) / (mass * cable_length)
        - payload_angular_damping * phi_dot
    )
    x_ddot = (
        force_x
        + payload_mass * cable_length * sphi * phi_dot * phi_dot
        - payload_mass * cable_length * cphi * phi_ddot
    ) / total_mass
    y_ddot = (
        force_y
        - total_mass * gravity
        - payload_mass * cable_length * cphi * phi_dot * phi_dot
        - payload_mass * cable_length * sphi * phi_ddot
    ) / total_mass

    out = np.empty(STATE_DIM, dtype=np.float64)
    out[0] = vx
    out[1] = vy
    out[2] = x_ddot
    out[3] = y_ddot
    out[4] = x[5]
    out[5] = tau / inertia
    out[6] = ur
    out[7] = ul
    out[8] = phi_dot
    out[9] = phi_ddot
    return out


@njit(cache=True)
def _planar_quadrotor_step_nb(state, control, dt, mass, inertia, gravity,
                              drag_x_coeff, drag_y_coeff, thrust_factor,
                              rotor_arm, rotor_speed_max, rotor_accel_max,
                              cable_length, payload_mass, payload_angular_damping, dynamics_substeps):
    x = state.copy()
    u = _clip_quad_control_nb(control, rotor_accel_max)
    substeps = max(1, int(dynamics_substeps))
    h = dt / substeps
    for _ in range(substeps):
        k1 = _quad_rhs_nb(x, u, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                          thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
                          cable_length, payload_mass, payload_angular_damping)
        k2 = _quad_rhs_nb(x + 0.5 * h * k1, u, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                          thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
                          cable_length, payload_mass, payload_angular_damping)
        k3 = _quad_rhs_nb(x + 0.5 * h * k2, u, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                          thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
                          cable_length, payload_mass, payload_angular_damping)
        k4 = _quad_rhs_nb(x + h * k3, u, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                          thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
                          cable_length, payload_mass, payload_angular_damping)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x[4] = _wrap_angle_nb(x[4])
        x[8] = _wrap_angle_nb(x[8])
        x[6] = min(max(x[6], 0.0), rotor_speed_max)
        x[7] = min(max(x[7], 0.0), rotor_speed_max)
    return x


@njit(cache=True)
def _rollout_quad_single_nb(x0, U, dt, mass, inertia, gravity,
                            drag_x_coeff, drag_y_coeff, thrust_factor,
                            rotor_arm, rotor_speed_max, rotor_accel_max,
                            cable_length, payload_mass, payload_angular_damping, dynamics_substeps):
    H = U.shape[0]
    X = np.zeros((H + 1, STATE_DIM), dtype=np.float64)
    for j in range(STATE_DIM):
        X[0, j] = x0[j]
    for t in range(H):
        X[t + 1] = _planar_quadrotor_step_nb(
            X[t], U[t], dt, mass, inertia, gravity,
            drag_x_coeff, drag_y_coeff, thrust_factor,
            rotor_arm, rotor_speed_max, rotor_accel_max,
            cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
        )
    return X


@njit(cache=True, parallel=True)
def _rollout_quad_batch_nb(x0, U, dt, mass, inertia, gravity,
                           drag_x_coeff, drag_y_coeff, thrust_factor,
                           rotor_arm, rotor_speed_max, rotor_accel_max,
                           cable_length, payload_mass, payload_angular_damping, dynamics_substeps):
    N = U.shape[0]
    H = U.shape[1]
    X = np.zeros((N, H + 1, STATE_DIM), dtype=np.float64)
    for n in prange(N):
        for j in range(STATE_DIM):
            X[n, 0, j] = x0[j]
        for t in range(H):
            X[n, t + 1] = _planar_quadrotor_step_nb(
                X[n, t], U[n, t], dt, mass, inertia, gravity,
                drag_x_coeff, drag_y_coeff, thrust_factor,
                rotor_arm, rotor_speed_max, rotor_accel_max,
                cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
            )
    return X


@njit(cache=True, parallel=True)
def _clip_quad_rows_nb(U, rotor_accel_max):
    out = U.copy()
    for n in prange(out.shape[0]):
        out[n, 0] = _clip_rotor_accel_nb(out[n, 0], rotor_accel_max)
        out[n, 1] = _clip_rotor_accel_nb(out[n, 1], rotor_accel_max)
    return out


@njit(cache=True, parallel=True)
def _trajectory_costs_quad_nb(X, U, circle_centers, circle_radii, goal,
                              robot_radius, payload_radius, cable_length, rotor_accel_max,
                              w_goal, w_obstacle, w_control, w_control_smooth,
                              w_terminal_position, w_terminal_velocity,
                              payload_swing_weight, payload_swing_rate_weight,
                              payload_terminal_swing_weight, payload_terminal_swing_rate_weight):
    N = U.shape[0]
    H = U.shape[1]
    M = circle_radii.shape[0]
    costs = np.zeros(N, dtype=np.float64)
    inv_h = 1.0 / max(1, H)
    inv_a = 1.0 / max(rotor_accel_max, 1e-9)
    for n in prange(N):
        cost = 0.0
        for t in range(H):
            state = X[n, t + 1]
            px = state[0]
            py = state[1]
            phi = _wrap_angle_nb(state[8])
            phi_dot = state[9]
            load_x = px + cable_length * math.sin(phi)
            load_y = py - cable_length * math.cos(phi)
            dx = px - goal[0]
            dy = py - goal[1]
            cost += w_goal * inv_h * (dx * dx + dy * dy)
            cost += inv_h * (
                payload_swing_weight * phi * phi
                + payload_swing_rate_weight * phi_dot * phi_dot
            )
            for m in range(M):
                cx = px - circle_centers[m, 0]
                cy = py - circle_centers[m, 1]
                clearance_q = math.sqrt(cx * cx + cy * cy) - circle_radii[m] - robot_radius
                sp_q = _softplus_scalar_nb(8.0 * (-clearance_q))
                cost += w_obstacle * sp_q * sp_q

                lx = load_x - circle_centers[m, 0]
                ly = load_y - circle_centers[m, 1]
                clearance_l = math.sqrt(lx * lx + ly * ly) - circle_radii[m] - payload_radius
                sp_l = _softplus_scalar_nb(8.0 * (-clearance_l))
                cost += w_obstacle * sp_l * sp_l
        for t in range(H):
            ur = U[n, t, 0] * inv_a
            ul = U[n, t, 1] * inv_a
            cost += w_control * (ur * ur + ul * ul)
        for t in range(H - 1):
            dur = (U[n, t + 1, 0] - U[n, t, 0]) * inv_a
            dul = (U[n, t + 1, 1] - U[n, t, 1]) * inv_a
            cost += w_control_smooth * (dur * dur + dul * dul)
        gxT = X[n, H, 0] - goal[0]
        gyT = X[n, H, 1] - goal[1]
        phiT = _wrap_angle_nb(X[n, H, 8])
        phiDotT = X[n, H, 9]
        cost += w_terminal_position * (gxT * gxT + gyT * gyT)
        cost += w_terminal_velocity * (X[n, H, 2] * X[n, H, 2] + X[n, H, 3] * X[n, H, 3])
        cost += payload_terminal_swing_weight * phiT * phiT
        cost += payload_terminal_swing_rate_weight * phiDotT * phiDotT
        costs[n] = cost
    return costs


@njit(cache=True, parallel=True)
def _collision_mask_quad_nb(X, circle_centers, circle_radii,
                            robot_radius, payload_radius, cable_length,
                            hard_collision_clearance):
    N = X.shape[0]
    H = X.shape[1] - 1
    M = circle_radii.shape[0]
    mask = np.zeros(N, dtype=np.bool_)
    for n in prange(N):
        hit = False
        for t in range(H):
            state = X[n, t + 1]
            px = state[0]
            py = state[1]
            load_x, load_y = _payload_position_nb(state, cable_length)
            for m in range(M):
                dx = px - circle_centers[m, 0]
                dy = py - circle_centers[m, 1]
                threshold_q = circle_radii[m] + robot_radius + hard_collision_clearance
                if math.sqrt(dx * dx + dy * dy) < threshold_q:
                    hit = True
                    break
                ldx = load_x - circle_centers[m, 0]
                ldy = load_y - circle_centers[m, 1]
                threshold_l = circle_radii[m] + payload_radius + hard_collision_clearance
                if math.sqrt(ldx * ldx + ldy * ldy) < threshold_l:
                    hit = True
                    break
            if hit:
                break
        mask[n] = hit
    return mask


@njit(cache=True, parallel=True)
def _spg_quad_nb(x0, U, cov, lookahead_steps, fd_thrust, fd_moment,
                 pseudoinverse_damping, covariance_jitter,
                 dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                 thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max, cable_length, payload_mass, payload_angular_damping, dynamics_substeps):
    H = U.shape[0]
    X = _rollout_quad_single_nb(
        x0, U, dt, mass, inertia, gravity,
        drag_x_coeff, drag_y_coeff, thrust_factor,
        rotor_arm, rotor_speed_max, rotor_accel_max, cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
    )
    out = np.zeros((H, 2, 2), dtype=np.float64)
    damp2 = pseudoinverse_damping * pseudoinverse_damping
    for t in prange(H):
        ell = min(max(1, int(lookahead_steps)), H - t)
        J = np.zeros((2, 2), dtype=np.float64)
        for j in range(2):
            eps = fd_thrust if j == 0 else fd_moment
            plus = X[t].copy()
            minus = X[t].copy()
            for k in range(ell):
                up = U[t + k].copy()
                um = U[t + k].copy()
                if k == 0:
                    up[j] += eps
                    um[j] -= eps
                plus = _planar_quadrotor_step_nb(
                    plus, up, dt, mass, inertia, gravity,
                    drag_x_coeff, drag_y_coeff, thrust_factor,
                    rotor_arm, rotor_speed_max, rotor_accel_max, cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
                )
                minus = _planar_quadrotor_step_nb(
                    minus, um, dt, mass, inertia, gravity,
                    drag_x_coeff, drag_y_coeff, thrust_factor,
                    rotor_arm, rotor_speed_max, rotor_accel_max, cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
                )
            J[0, j] = (plus[0] - minus[0]) / (2.0 * eps)
            J[1, j] = (plus[1] - minus[1]) / (2.0 * eps)
        M = J @ J.T + damp2 * np.eye(2)
        Jdag = J.T @ np.linalg.inv(M)
        C = np.empty((2, 2), dtype=np.float64)
        C[0, 0] = cov[t, 0, 0]
        C[0, 1] = 0.5 * (cov[t, 0, 1] + cov[t, 1, 0])
        C[1, 0] = C[0, 1]
        C[1, 1] = cov[t, 1, 1]
        projected = Jdag @ C @ Jdag.T
        out[t, 0, 0] = projected[0, 0] + covariance_jitter
        out[t, 0, 1] = projected[0, 1]
        out[t, 1, 0] = projected[1, 0]
        out[t, 1, 1] = projected[1, 1] + covariance_jitter
    return out


def _circle_arrays(obstacle_circles) -> Tuple[Array, Array]:
    """Convert Python obstacle-circle objects once at the model boundary."""
    if not obstacle_circles:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    centers = np.asarray(
        [np.asarray(center, dtype=np.float64)[:2] for center, _ in obstacle_circles],
        dtype=np.float64,
    )
    radii = np.asarray([float(radius) for _, radius in obstacle_circles], dtype=np.float64)
    return np.ascontiguousarray(centers), np.ascontiguousarray(radii)


@njit(cache=True)
def _path_arc_nb(path):
    n = path.shape[0]
    arc = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dx = path[i, 0] - path[i - 1, 0]
        dy = path[i, 1] - path[i - 1, 1]
        arc[i] = arc[i - 1] + math.sqrt(dx * dx + dy * dy)
    return arc


@njit(cache=True)
def _project_to_path_nb(px, py, path, arc):
    n = path.shape[0]
    if n < 2:
        return 0.0, path[0, 0], path[0, 1], 1.0, 0.0, 0.0, 0
    best_d2 = 1e300
    best_progress = 0.0
    best_qx = path[0, 0]
    best_qy = path[0, 1]
    best_tx = 1.0
    best_ty = 0.0
    best_alpha = 0.0
    best_seg = 0
    for i in range(n - 1):
        ax = path[i, 0]
        ay = path[i, 1]
        dx = path[i + 1, 0] - ax
        dy = path[i + 1, 1] - ay
        dd = dx * dx + dy * dy
        if dd <= 1e-15:
            alpha = 0.0
        else:
            alpha = ((px - ax) * dx + (py - ay) * dy) / dd
            alpha = min(max(alpha, 0.0), 1.0)
        qx = ax + alpha * dx
        qy = ay + alpha * dy
        ex = px - qx
        ey = py - qy
        d2 = ex * ex + ey * ey
        if d2 < best_d2:
            length = math.sqrt(max(dd, 1e-15))
            if length > 1e-12:
                tx = dx / length
                ty = dy / length
            else:
                tx = 1.0
                ty = 0.0
            best_d2 = d2
            best_qx = qx
            best_qy = qy
            best_tx = tx
            best_ty = ty
            best_alpha = alpha
            best_seg = i
            best_progress = arc[i] + alpha * max(0.0, arc[i + 1] - arc[i])
    return best_progress, best_qx, best_qy, best_tx, best_ty, best_alpha, best_seg


@njit(cache=True)
def _precision_at_nb(cov, seg, alpha, covariance_floor):
    n = cov.shape[0]
    i = min(max(int(seg), 0), max(0, n - 1))
    j = min(i + 1, n - 1)
    beta = 1.0 - alpha
    c00 = beta * cov[i, 0, 0] + alpha * cov[j, 0, 0]
    c01 = 0.5 * (
        beta * (cov[i, 0, 1] + cov[i, 1, 0])
        + alpha * (cov[j, 0, 1] + cov[j, 1, 0])
    )
    c11 = beta * cov[i, 1, 1] + alpha * cov[j, 1, 1]
    floor_var = max(covariance_floor * covariance_floor, 1e-12)
    a = c00 + floor_var
    d = c11 + floor_var
    det = a * d - c01 * c01
    if det <= 1e-15:
        a += floor_var
        d += floor_var
        det = max(a * d - c01 * c01, 1e-15)
    inv_det = 1.0 / det
    return d * inv_det, -c01 * inv_det, a * inv_det


def _prepare_cov(path: Array, cov_blocks: Optional[Array], cfg: MPPIConfig) -> Array:
    """Shape/type adapter; all covariance arithmetic is performed in Numba."""
    n = len(path)
    if cov_blocks is None:
        out = np.zeros((n, 2, 2), dtype=np.float64)
        variance = float(cfg.prior_ilqr_covariance_fallback_std) ** 2
        out[:, 0, 0] = variance
        out[:, 1, 1] = variance
        return np.ascontiguousarray(out)
    cov = np.asarray(cov_blocks, dtype=np.float64)
    if cov.shape != (n, 2, 2):
        raise ValueError(f"cov_blocks must have shape ({n},2,2), got {cov.shape}")
    return np.ascontiguousarray(0.5 * (cov + np.swapaxes(cov, 1, 2)))


@njit(cache=True)
def _state_difference_nb(a, b):
    d = a - b
    d[4] = _wrap_angle_nb(d[4])
    d[8] = _wrap_angle_nb(d[8])
    return d


@njit(cache=True)
def _stage_terms_nb(x, u, path, arc, cov, covariance_floor,
                    mahalanobis_weight, progress_weight,
                    control_thrust_weight, control_moment_weight, rotor_accel_max,
                    payload_swing_weight, payload_swing_rate_weight):
    progress, qx, qy, tx, ty, alpha, seg = _project_to_path_nb(x[0], x[1], path, arc)
    p00, p01, p11 = _precision_at_nb(cov, seg, alpha, covariance_floor)
    ex = x[0] - qx
    ey = x[1] - qy
    mx = p00 * ex + p01 * ey
    my = p01 * ex + p11 * ey
    phi = _wrap_angle_nb(x[8])
    phi_dot = x[9]

    inv_a = 1.0 / max(rotor_accel_max, 1e-9)
    collective = 0.5 * (u[0] + u[1]) * inv_a
    differential = 0.5 * (u[0] - u[1]) * inv_a
    cost = (
        mahalanobis_weight * (ex * mx + ey * my)
        - progress_weight * progress
        + control_thrust_weight * collective * collective
        + control_moment_weight * differential * differential
        + payload_swing_weight * phi * phi
        + payload_swing_rate_weight * phi_dot * phi_dot
    )

    lx = np.zeros(STATE_DIM, dtype=np.float64)
    lx[0] = 2.0 * mahalanobis_weight * mx - progress_weight * tx
    lx[1] = 2.0 * mahalanobis_weight * my - progress_weight * ty
    lx[8] = 2.0 * payload_swing_weight * phi
    lx[9] = 2.0 * payload_swing_rate_weight * phi_dot
    lxx = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
    lxx[0, 0] = 2.0 * mahalanobis_weight * p00
    lxx[0, 1] = 2.0 * mahalanobis_weight * p01
    lxx[1, 0] = lxx[0, 1]
    lxx[1, 1] = 2.0 * mahalanobis_weight * p11
    lxx[8, 8] = 2.0 * payload_swing_weight
    lxx[9, 9] = 2.0 * payload_swing_rate_weight

    lu = np.empty(2, dtype=np.float64)
    lu[0] = (control_thrust_weight * collective + control_moment_weight * differential) * inv_a
    lu[1] = (control_thrust_weight * collective - control_moment_weight * differential) * inv_a
    luu = np.zeros((2, 2), dtype=np.float64)
    scale = 0.5 * inv_a * inv_a
    luu[0, 0] = scale * (control_thrust_weight + control_moment_weight)
    luu[1, 1] = luu[0, 0]
    luu[0, 1] = scale * (control_thrust_weight - control_moment_weight)
    luu[1, 0] = luu[0, 1]
    lux = np.zeros((2, STATE_DIM), dtype=np.float64)
    return cost, lx, lxx, lu, luu, lux, progress


@njit(cache=True)
def _linearize_nb(x, u, dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                  thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
                  cable_length, payload_mass, payload_angular_damping, dynamics_substeps):
    nx = STATE_DIM
    nu = 2
    A = np.zeros((nx, nx), dtype=np.float64)
    B = np.zeros((nx, nu), dtype=np.float64)
    eps_x = np.array([
        1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-2, 1e-2, 1e-5, 1e-4
    ], dtype=np.float64)
    eps_u = np.array([1.0, 1.0], dtype=np.float64)
    for j in range(nx):
        xp = x.copy()
        xm = x.copy()
        xp[j] += eps_x[j]
        xm[j] -= eps_x[j]
        fp = _planar_quadrotor_step_nb(
            xp, u, dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
            thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
            cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
        )
        fm = _planar_quadrotor_step_nb(
            xm, u, dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
            thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
            cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
        )
        diff = _state_difference_nb(fp, fm)
        for r in range(nx):
            A[r, j] = diff[r] / (2.0 * eps_x[j])
    for j in range(nu):
        up = u.copy()
        um = u.copy()
        up[j] += eps_u[j]
        um[j] -= eps_u[j]
        fp = _planar_quadrotor_step_nb(
            x, up, dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
            thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
            cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
        )
        fm = _planar_quadrotor_step_nb(
            x, um, dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
            thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
            cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
        )
        diff = _state_difference_nb(fp, fm)
        for r in range(nx):
            B[r, j] = diff[r] / (2.0 * eps_u[j])
    return A, B


@njit(cache=True)
def _guidance_control_nb(x, qx, qy, tx, ty, max_translational_speed,
                         mass, payload_mass, inertia, gravity, thrust_factor, rotor_arm,
                         rotor_speed_max, rotor_accel_max, rotor_speed_tracking_gain,
                         attitude_kp, attitude_kd):
    target_vx = max_translational_speed * tx
    target_vy = max_translational_speed * ty
    desired_ax = 2.2 * (target_vx - x[2]) + 1.2 * (qx - x[0])
    desired_ay = 2.2 * (target_vy - x[3]) + 1.2 * (qy - x[1])
    thrust_x = desired_ax
    thrust_y = desired_ay + gravity
    # Use the combined supported mass. The iLQR refinement then handles the
    # pendulum coupling and swing explicitly through the full 10-state dynamics.
    f = (mass + payload_mass) * math.sqrt(thrust_x * thrust_x + thrust_y * thrust_y)
    theta_des = math.atan2(-thrust_x, thrust_y)
    theta_error = _wrap_angle_nb(theta_des - x[4])
    tau = inertia * (attitude_kp * theta_error - attitude_kd * x[5])

    wr_des, wl_des = _desired_rotor_speeds_nb(
        f, tau, thrust_factor, rotor_arm, rotor_speed_max
    )
    ur = rotor_speed_tracking_gain * (wr_des - x[6])
    ul = rotor_speed_tracking_gain * (wl_des - x[7])
    out = np.empty(2, dtype=np.float64)
    out[0] = _bounded_rotor_accel_nb(x[6], ur, rotor_speed_max, rotor_accel_max)
    out[1] = _bounded_rotor_accel_nb(x[7], ul, rotor_speed_max, rotor_accel_max)
    return out


@njit(cache=True)
def _initial_path_controls_nb(x0, path, H, max_translational_speed,
                              rotor_speed_tracking_gain, attitude_kp, attitude_kd,
                              dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                              thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
                              cable_length, payload_mass, payload_angular_damping, dynamics_substeps):
    U = np.zeros((H, 2), dtype=np.float64)
    x = x0.copy()
    arc = _path_arc_nb(path)
    for t in range(H):
        progress, qx, qy, tx, ty, _, _ = _project_to_path_nb(x[0], x[1], path, arc)
        remaining = max(0.0, arc[arc.shape[0] - 1] - progress)
        target_speed = min(max_translational_speed, 1.5 * remaining)
        U[t] = _guidance_control_nb(
            x, qx, qy, tx, ty, target_speed,
            mass, payload_mass, inertia, gravity, thrust_factor, rotor_arm,
            rotor_speed_max, rotor_accel_max, rotor_speed_tracking_gain, attitude_kp, attitude_kd,
        )
        x = _planar_quadrotor_step_nb(
            x, U[t], dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
            thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
            cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
        )
    return U


@njit(cache=True)
def _invert_regularized_2x2_nb(M):
    a = M[0, 0]
    b = 0.5 * (M[0, 1] + M[1, 0])
    d = M[1, 1]
    det = a * d - b * b
    if abs(det) < 1e-12:
        jitter = 1e-6
        a += jitter
        d += jitter
        det = a * d - b * b
        if abs(det) < 1e-15:
            det = 1e-15 if det >= 0.0 else -1e-15
    inv = np.empty((2, 2), dtype=np.float64)
    inv[0, 0] = d / det
    inv[0, 1] = -b / det
    inv[1, 0] = -b / det
    inv[1, 1] = a / det
    return inv


@njit(cache=True)
def _ilqr_total_cost_nb(x0, U, path, arc, cov, covariance_floor,
                        mahalanobis_weight, progress_weight,
                        control_thrust_weight, control_moment_weight, rotor_accel_max,
                        payload_swing_weight, payload_swing_rate_weight,
                        terminal_position_weight, terminal_velocity_weight,
                        payload_terminal_swing_weight, payload_terminal_swing_rate_weight,
                        dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                        thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max_dyn,
                        cable_length, payload_mass, payload_angular_damping, dynamics_substeps):
    X = _rollout_quad_single_nb(
        x0, U, dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
        thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max_dyn,
        cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
    )
    H = U.shape[0]
    positions = np.zeros(H, dtype=np.float64)
    cost = 0.0
    for t in range(H):
        c, _, _, _, _, _, progress = _stage_terms_nb(
            X[t], U[t], path, arc, cov, covariance_floor,
            mahalanobis_weight, progress_weight,
            control_thrust_weight, control_moment_weight, rotor_accel_max,
            payload_swing_weight, payload_swing_rate_weight,
        )
        cost += c
        positions[t] = progress
    exT = X[H, 0] - path[path.shape[0] - 1, 0]
    eyT = X[H, 1] - path[path.shape[0] - 1, 1]
    phiT = _wrap_angle_nb(X[H, 8])
    phiDotT = X[H, 9]
    cost += terminal_position_weight * (exT * exT + eyT * eyT)
    cost += terminal_velocity_weight * (X[H, 2] * X[H, 2] + X[H, 3] * X[H, 3])
    cost += payload_terminal_swing_weight * phiT * phiT
    cost += payload_terminal_swing_rate_weight * phiDotT * phiDotT
    return cost, X, positions


@njit(cache=True)
def _ilqr_nominal_nb(x0, path, cov, H, iterations, line_search_steps,
                     max_translational_speed, covariance_floor,
                     mahalanobis_weight, progress_weight,
                     control_thrust_weight, control_moment_weight,
                     payload_swing_weight, payload_swing_rate_weight,
                     terminal_position_weight, terminal_velocity_weight,
                     payload_terminal_swing_weight, payload_terminal_swing_rate_weight,
                     regularization,
                     rotor_speed_tracking_gain, attitude_kp, attitude_kd,
                     dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                     thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
                     cable_length, payload_mass, payload_angular_damping, dynamics_substeps):
    if path.shape[0] < 2:
        return np.zeros((H, 2), dtype=np.float64), np.zeros(H, dtype=np.float64)

    arc = _path_arc_nb(path)
    U = _initial_path_controls_nb(
        x0, path, H, max_translational_speed,
        rotor_speed_tracking_gain, attitude_kp, attitude_kd,
        dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
        thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
        cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
    )
    best_cost, X, positions = _ilqr_total_cost_nb(
        x0, U, path, arc, cov, covariance_floor,
        mahalanobis_weight, progress_weight,
        control_thrust_weight, control_moment_weight, rotor_accel_max,
        payload_swing_weight, payload_swing_rate_weight,
        terminal_position_weight, terminal_velocity_weight,
        payload_terminal_swing_weight, payload_terminal_swing_rate_weight,
        dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
        thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
        cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
    )

    for _ in range(iterations):
        A = np.zeros((H, STATE_DIM, STATE_DIM), dtype=np.float64)
        B = np.zeros((H, STATE_DIM, 2), dtype=np.float64)
        lx = np.zeros((H, STATE_DIM), dtype=np.float64)
        lxx = np.zeros((H, STATE_DIM, STATE_DIM), dtype=np.float64)
        lu = np.zeros((H, 2), dtype=np.float64)
        luu = np.zeros((H, 2, 2), dtype=np.float64)
        lux = np.zeros((H, 2, STATE_DIM), dtype=np.float64)
        for t in range(H):
            At, Bt = _linearize_nb(
                X[t], U[t], dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
                cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
            )
            _, lxt, lxxt, lut, luut, luxt, _ = _stage_terms_nb(
                X[t], U[t], path, arc, cov, covariance_floor,
                mahalanobis_weight, progress_weight,
                control_thrust_weight, control_moment_weight, rotor_accel_max,
                payload_swing_weight, payload_swing_rate_weight,
            )
            A[t] = At
            B[t] = Bt
            lx[t] = lxt
            lxx[t] = lxxt
            lu[t] = lut
            luu[t] = luut
            lux[t] = luxt

        Vx = np.zeros(STATE_DIM, dtype=np.float64)
        Vxx = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        exT = X[H, 0] - path[path.shape[0] - 1, 0]
        eyT = X[H, 1] - path[path.shape[0] - 1, 1]
        phiT = _wrap_angle_nb(X[H, 8])
        phiDotT = X[H, 9]
        Vx[0] = 2.0 * terminal_position_weight * exT
        Vx[1] = 2.0 * terminal_position_weight * eyT
        Vxx[0, 0] = 2.0 * terminal_position_weight
        Vxx[1, 1] = 2.0 * terminal_position_weight
        Vx[2] = 2.0 * terminal_velocity_weight * X[H, 2]
        Vx[3] = 2.0 * terminal_velocity_weight * X[H, 3]
        Vxx[2, 2] = 2.0 * terminal_velocity_weight
        Vxx[3, 3] = 2.0 * terminal_velocity_weight
        Vx[8] = 2.0 * payload_terminal_swing_weight * phiT
        Vx[9] = 2.0 * payload_terminal_swing_rate_weight * phiDotT
        Vxx[8, 8] = 2.0 * payload_terminal_swing_weight
        Vxx[9, 9] = 2.0 * payload_terminal_swing_rate_weight
        k = np.zeros((H, 2), dtype=np.float64)
        K = np.zeros((H, 2, STATE_DIM), dtype=np.float64)

        for t in range(H - 1, -1, -1):
            At = A[t]
            Bt = B[t]
            Qx = lx[t] + At.T @ Vx
            Qu = lu[t] + Bt.T @ Vx
            Qxx = lxx[t] + At.T @ Vxx @ At
            Quu = luu[t] + Bt.T @ Vxx @ Bt
            Quu[0, 0] += regularization
            Quu[1, 1] += regularization
            Qux = lux[t] + Bt.T @ Vxx @ At
            inv_Quu = _invert_regularized_2x2_nb(Quu)
            k[t] = -(inv_Quu @ Qu)
            K[t] = -(inv_Quu @ Qux)
            Vx = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu + Qux.T @ k[t]
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]
            Vxx = 0.5 * (Vxx + Vxx.T)

        improved = False
        for ls in range(line_search_steps):
            alpha_ls = 0.5 ** ls
            Unew = np.zeros_like(U)
            Xnew = np.zeros_like(X)
            Xnew[0] = x0
            for t in range(H):
                state_delta = _state_difference_nb(Xnew[t], X[t])
                du = alpha_ls * k[t] + K[t] @ state_delta
                Unew[t, 0] = _clip_rotor_accel_nb(U[t, 0] + du[0], rotor_accel_max)
                Unew[t, 1] = _clip_rotor_accel_nb(U[t, 1] + du[1], rotor_accel_max)
                Xnew[t + 1] = _planar_quadrotor_step_nb(
                    Xnew[t], Unew[t], dt, mass, inertia, gravity,
                    drag_x_coeff, drag_y_coeff, thrust_factor, rotor_arm,
                    rotor_speed_max, rotor_accel_max, cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
                )
            candidate_cost, candidate_X, candidate_pos = _ilqr_total_cost_nb(
                x0, Unew, path, arc, cov, covariance_floor,
                mahalanobis_weight, progress_weight,
                control_thrust_weight, control_moment_weight, rotor_accel_max,
                payload_swing_weight, payload_swing_rate_weight,
                terminal_position_weight, terminal_velocity_weight,
                payload_terminal_swing_weight, payload_terminal_swing_rate_weight,
                dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
                cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
            )
            if candidate_cost < best_cost:
                U = Unew
                X = candidate_X
                positions = candidate_pos
                best_cost = candidate_cost
                improved = True
                break
        if not improved:
            break
    return U, positions


@njit(cache=True)
def _nominal_controls_to_goal_nb(x0, goal, H, max_translational_speed,
                                 rotor_speed_tracking_gain, attitude_kp, attitude_kd,
                                 dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                                 thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
                                 cable_length, payload_mass, payload_angular_damping, dynamics_substeps):
    x = x0.copy()
    U = np.zeros((H, 2), dtype=np.float64)
    for t in range(H):
        dx = goal[0] - x[0]
        dy = goal[1] - x[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1e-9:
            tx = dx / dist
            ty = dy / dist
        else:
            tx = 0.0
            ty = 0.0
        target_speed = min(max_translational_speed, 1.5 * dist)
        U[t] = _guidance_control_nb(
            x, goal[0], goal[1], tx, ty, target_speed,
            mass, payload_mass, inertia, gravity, thrust_factor, rotor_arm, rotor_speed_max,
            rotor_accel_max, rotor_speed_tracking_gain, attitude_kp, attitude_kd,
        )
        x = _planar_quadrotor_step_nb(
            x, U[t], dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
            thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
            cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
        )
    return U


def planar_quadrotor_step(state: Array, control: Array, cfg: MPPIConfig) -> Array:
    return _planar_quadrotor_step_nb(
        np.asarray(state, dtype=np.float64),
        np.asarray(control, dtype=np.float64),
        *_dynamic_args(cfg),
    )


def rollout_single(x_current: Array, controls: Array, cfg: MPPIConfig) -> Array:
    return _rollout_quad_single_nb(
        np.asarray(x_current, dtype=np.float64),
        np.ascontiguousarray(np.asarray(controls, dtype=np.float64)),
        *_dynamic_args(cfg),
    )


def rollout_batch(x_current: Array, controls: Array, cfg: MPPIConfig) -> Array:
    U = np.ascontiguousarray(np.asarray(controls, dtype=np.float64))
    if U.ndim != 3 or U.shape[2] != 2:
        raise ValueError("controls must have shape (N,H,2).")
    return _rollout_quad_batch_nb(
        np.asarray(x_current, dtype=np.float64), U, *_dynamic_args(cfg)
    )


def nominal_controls_and_arc_positions(x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array] = None) -> Tuple[Array, Array]:
    path = np.ascontiguousarray(np.asarray(ref, dtype=np.float64))
    cov = _prepare_cov(path, cov_blocks, cfg)
    return _ilqr_nominal_nb(
        np.asarray(x0, dtype=np.float64), path, cov,
        int(cfg.horizon), int(cfg.prior_ilqr_iterations), int(cfg.prior_ilqr_line_search_steps),
        float(cfg.max_translational_speed), float(cfg.prior_ilqr_covariance_floor),
        float(cfg.prior_ilqr_mahalanobis_weight), float(cfg.prior_ilqr_progress_weight),
        float(cfg.prior_ilqr_control_thrust_weight), float(cfg.prior_ilqr_control_moment_weight),
        float(cfg.payload_swing_weight), float(cfg.payload_swing_rate_weight),
        float(cfg.w_terminal_position), float(cfg.w_terminal_velocity),
        float(cfg.payload_terminal_swing_weight), float(cfg.payload_terminal_swing_rate_weight),
        float(cfg.prior_ilqr_regularization),
        float(cfg.rotor_speed_tracking_gain), float(cfg.attitude_kp), float(cfg.attitude_kd),
        *_dynamic_args(cfg),
    )


def prior_control_arc_positions(x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array] = None) -> Array:
    return nominal_controls_and_arc_positions(x0, ref, cfg, cov_blocks)[1]


def nominal_controls_to_track_path(x0: Array, ref: Array, cfg: MPPIConfig, cov_blocks: Optional[Array] = None) -> Array:
    return nominal_controls_and_arc_positions(x0, ref, cfg, cov_blocks)[0]


def nominal_controls_to_goal(x0: Array, goal: Array, cfg: MPPIConfig) -> Array:
    return _nominal_controls_to_goal_nb(
        np.asarray(x0, dtype=np.float64), np.asarray(goal, dtype=np.float64),
        int(cfg.horizon), float(cfg.max_translational_speed),
        float(cfg.rotor_speed_tracking_gain), float(cfg.attitude_kp), float(cfg.attitude_kd),
        *_dynamic_args(cfg),
    )


def control_noise_scale(cfg: MPPIConfig) -> Array:
    return np.asarray([cfg.noise_thrust, cfg.noise_moment], dtype=np.float64)


def clip_control_batch(controls: Array, cfg: MPPIConfig) -> Array:
    U = np.ascontiguousarray(np.asarray(controls, dtype=np.float64))
    if U.size == 0:
        return U.copy()
    if U.shape[-1] != 2:
        raise ValueError("controls must have final dimension 2.")
    shape = U.shape
    flat = np.ascontiguousarray(U.reshape(-1, 2))
    clipped = _clip_quad_rows_nb(flat, float(cfg.rotor_accel_max))
    return clipped.reshape(shape)


def project_control_covariances(x_current: Array, nominal: Array, covariances: Array, cfg: MPPIConfig) -> Array:
    return _spg_quad_nb(
        np.asarray(x_current, dtype=np.float64),
        np.ascontiguousarray(np.asarray(nominal, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(covariances, dtype=np.float64)),
        int(cfg.spg_lookahead_steps),
        float(cfg.spg_fd_thrust),
        float(cfg.spg_fd_moment),
        float(cfg.spg_pseudoinverse_damping),
        float(cfg.spg_covariance_jitter),
        *_dynamic_args(cfg),
    )


def trajectory_costs(states: Array, controls: Array, obstacle_circles, goal: Array, cfg: MPPIConfig) -> Array:
    centers, radii = _circle_arrays(obstacle_circles)
    return _trajectory_costs_quad_nb(
        np.ascontiguousarray(np.asarray(states, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(controls, dtype=np.float64)),
        centers,
        radii,
        np.asarray(goal, dtype=np.float64),
        float(cfg.total_drone_radius),
        float(cfg.payload_radius),
        float(cfg.cable_length),
        float(cfg.rotor_accel_max),
        float(cfg.w_goal),
        float(cfg.w_obstacle),
        float(cfg.w_control),
        float(cfg.w_control_smooth),
        float(cfg.w_terminal_position),
        float(cfg.w_terminal_velocity),
        float(cfg.payload_swing_weight),
        float(cfg.payload_swing_rate_weight),
        float(cfg.payload_terminal_swing_weight),
        float(cfg.payload_terminal_swing_rate_weight),
    )


def collision_mask(states: Array, obstacle_circles, goal: Array, cfg: MPPIConfig) -> Array:
    del goal
    centers, radii = _circle_arrays(obstacle_circles)
    return _collision_mask_quad_nb(
        np.ascontiguousarray(np.asarray(states, dtype=np.float64)),
        centers,
        radii,
        float(cfg.total_drone_radius),
        float(cfg.payload_radius),
        float(cfg.cable_length),
        float(cfg.hard_collision_clearance),
    )


@njit(cache=True, parallel=True)
def _mean_path_clearance_nb(path, circle_centers, circle_radii,
                            robot_radius, payload_radius, cable_length):
    """Prior screening with the package assumed to hang vertically below the path."""
    n = path.shape[0]
    m = circle_radii.shape[0]
    if n == 0 or m == 0:
        return math.inf
    local_best = np.full(n, math.inf, dtype=np.float64)
    for i in prange(n):
        px = path[i, 0]
        py = path[i, 1]
        load_x = px
        load_y = py - cable_length
        best = math.inf
        for j in range(m):
            dx = px - circle_centers[j, 0]
            dy = py - circle_centers[j, 1]
            value_q = math.sqrt(dx * dx + dy * dy) - circle_radii[j] - robot_radius
            if value_q < best:
                best = value_q
            ldx = load_x - circle_centers[j, 0]
            ldy = load_y - circle_centers[j, 1]
            value_l = math.sqrt(ldx * ldx + ldy * ldy) - circle_radii[j] - payload_radius
            if value_l < best:
                best = value_l
        local_best[i] = best
    return np.min(local_best)


def mean_path_clearance(path: Array, obstacle_circles, cfg: MPPIConfig) -> float:
    centers, radii = _circle_arrays(obstacle_circles)
    return float(_mean_path_clearance_nb(
        np.ascontiguousarray(np.asarray(path, dtype=np.float64)),
        centers,
        radii,
        float(cfg.total_drone_radius),
        float(cfg.payload_radius),
        float(cfg.cable_length),
    ))


@njit(cache=True)
def _quad_payload_circle_clearance_nb(state, circle_centers, circle_radii,
                                      robot_radius, payload_radius, cable_length):
    if circle_radii.shape[0] == 0:
        return math.inf
    load_x, load_y = _payload_position_nb(state, cable_length)
    best = math.inf
    for j in range(circle_radii.shape[0]):
        dx = state[0] - circle_centers[j, 0]
        dy = state[1] - circle_centers[j, 1]
        value_q = math.sqrt(dx * dx + dy * dy) - circle_radii[j] - robot_radius
        if value_q < best:
            best = value_q
        ldx = load_x - circle_centers[j, 0]
        ldy = load_y - circle_centers[j, 1]
        value_l = math.sqrt(ldx * ldx + ldy * ldy) - circle_radii[j] - payload_radius
        if value_l < best:
            best = value_l
    return best


@njit(cache=True)
def _apply_final_output_nb(x_current, control,
                           circle_centers, circle_radii,
                           enforce_one_step_safety, one_step_safety_clearance,
                           robot_radius, payload_radius,
                           rotor_speed_tracking_gain, attitude_kp, attitude_kd,
                           dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
                           thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
                           cable_length, payload_mass, payload_angular_damping, dynamics_substeps):
    cmd = _clip_quad_control_nb(control, rotor_accel_max)
    if enforce_one_step_safety and circle_radii.shape[0] > 0:
        nxt = _planar_quadrotor_step_nb(
            x_current, cmd, dt, mass, inertia, gravity, drag_x_coeff, drag_y_coeff,
            thrust_factor, rotor_arm, rotor_speed_max, rotor_accel_max,
            cable_length, payload_mass, payload_angular_damping, dynamics_substeps,
        )
        next_clearance = _quad_payload_circle_clearance_nb(
            nxt, circle_centers, circle_radii, robot_radius, payload_radius, cable_length
        )
        current_clearance = _quad_payload_circle_clearance_nb(
            x_current, circle_centers, circle_radii, robot_radius, payload_radius, cable_length
        )
        if next_clearance < one_step_safety_clearance and next_clearance < current_clearance - 1e-4:
            # Hold quadrotor position; the coupled model naturally lets the payload settle.
            cmd = _guidance_control_nb(
                x_current, x_current[0], x_current[1], 0.0, 0.0, 0.0,
                mass, payload_mass, inertia, gravity, thrust_factor, rotor_arm, rotor_speed_max,
                rotor_accel_max, rotor_speed_tracking_gain, attitude_kp, attitude_kd,
            )
    return cmd


def apply_final_output(x_current: Array, control: Array, previous_control: Optional[Array], obstacle_circles, goal: Array, cfg: MPPIConfig) -> Array:
    del goal, previous_control
    centers, radii = _circle_arrays(obstacle_circles)
    return _apply_final_output_nb(
        np.asarray(x_current, dtype=np.float64),
        np.asarray(control, dtype=np.float64),
        centers,
        radii,
        bool(cfg.enforce_one_step_safety),
        float(cfg.one_step_safety_clearance),
        float(cfg.total_drone_radius),
        float(cfg.payload_radius),
        float(cfg.rotor_speed_tracking_gain),
        float(cfg.attitude_kp),
        float(cfg.attitude_kd),
        *_dynamic_args(cfg),
    )


def render_output_trajectory(info: Dict[str, object], x_current: Array, control: Array, goal: Array, cfg: MPPIConfig) -> None:
    del goal
    sequence = info.get("planned_control_sequence")
    if sequence is None:
        return
    U = np.asarray(sequence, dtype=np.float64).copy()
    if U.ndim == 2 and U.shape[1] == 2 and len(U):
        U[0] = np.asarray(control, dtype=np.float64)
        info["optimal_traj"] = rollout_single(x_current, U, cfg)


@njit(cache=True)
def _initial_pose_nb(start, hover_rotor_speed):
    return np.array([
        start[0], start[1], 0.0, 0.0, 0.0, 0.0,
        hover_rotor_speed, hover_rotor_speed, 0.0, 0.0,
    ], dtype=np.float64)


def initial_pose(start: Array, goal: Array, cfg: MPPIConfig) -> Array:
    del goal
    return _initial_pose_nb(np.asarray(start, dtype=np.float64), float(cfg.hover_rotor_speed))


def payload_position(state: Array, cfg: MPPIConfig) -> Array:
    state = np.asarray(state, dtype=np.float64)
    if state.size < STATE_DIM:
        raise ValueError(f"payload quadrotor state must contain {STATE_DIM} values.")
    px, py = _payload_position_nb(state, float(cfg.cable_length))
    return np.asarray([px, py], dtype=np.float64)


@njit(cache=True)
def _goal_reached_nb(state, goal, tolerance):
    dx = state[0] - goal[0]
    dy = state[1] - goal[1]
    return math.sqrt(dx * dx + dy * dy) <= tolerance


def goal_reached(state: Array, goal: Array, cfg: MPPIConfig) -> bool:
    state = np.asarray(state, dtype=np.float64)
    position_ok = bool(_goal_reached_nb(
        state, np.asarray(goal, dtype=np.float64), float(cfg.goal_tolerance),
    ))
    speed_ok = math.hypot(float(state[2]), float(state[3])) <= float(cfg.terminal_velocity_tolerance)
    swing_ok = abs(float(_wrap_angle_nb(state[8]))) <= float(cfg.payload_angle_tolerance)
    swing_rate_ok = abs(float(state[9])) <= float(cfg.payload_rate_tolerance)
    return bool(position_ok and speed_ok and swing_ok and swing_rate_ok)


def advance_state(state: Array, control: Array, goal: Array, cfg: MPPIConfig) -> Tuple[Array, bool]:
    nxt = _planar_quadrotor_step_nb(
        np.asarray(state, dtype=np.float64), np.asarray(control, dtype=np.float64),
        *_dynamic_args(cfg),
    )
    return nxt, goal_reached(nxt, goal, cfg)


def _obstacle_polygon_arrays(obstacles: Sequence) -> Tuple[Array, Array]:
    """Pack arbitrary Python obstacle objects once before entering Numba."""
    if not obstacles:
        return np.zeros((0, 0, 2), dtype=np.float64), np.zeros(0, dtype=np.int64)
    polygons = [
        np.asarray(getattr(obstacle, "vertices", obstacle), dtype=np.float64)[:, :2]
        for obstacle in obstacles
    ]
    max_vertices = max(len(poly) for poly in polygons)
    padded = np.zeros((len(polygons), max_vertices, 2), dtype=np.float64)
    lengths = np.zeros(len(polygons), dtype=np.int64)
    for i, poly in enumerate(polygons):
        padded[i, :len(poly)] = poly
        lengths[i] = len(poly)
    return np.ascontiguousarray(padded), np.ascontiguousarray(lengths)


@njit(cache=True)
def _point_segment_distance_nb(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    dd = dx * dx + dy * dy
    if dd <= 1e-15:
        ex = px - ax
        ey = py - ay
        return math.sqrt(ex * ex + ey * ey)
    alpha = ((px - ax) * dx + (py - ay) * dy) / dd
    alpha = min(max(alpha, 0.0), 1.0)
    qx = ax + alpha * dx
    qy = ay + alpha * dy
    ex = px - qx
    ey = py - qy
    return math.sqrt(ex * ex + ey * ey)


@njit(cache=True)
def _point_in_polygon_nb(px, py, polygon, n):
    inside = False
    for i in range(n):
        j = i + 1
        if j == n:
            j = 0
        x0 = polygon[i, 0]
        y0 = polygon[i, 1]
        x1 = polygon[j, 0]
        y1 = polygon[j, 1]
        if (y0 > py) != (y1 > py):
            xcross = x0 + (py - y0) * (x1 - x0) / (y1 - y0 + 1e-18)
            if px < xcross:
                inside = not inside
    return inside


@njit(cache=True)
def _circle_polygon_clearance_nb(px, py, radius, poly, n):
    distance = math.inf
    for i in range(n):
        j = i + 1
        if j == n:
            j = 0
        d = _point_segment_distance_nb(
            px, py, poly[i, 0], poly[i, 1], poly[j, 0], poly[j, 1]
        )
        if d < distance:
            distance = d
    if _point_in_polygon_nb(px, py, poly, n):
        distance = -distance
    return distance - radius


@njit(cache=True, parallel=True)
def _minimum_clearance_nb(states, polygons, polygon_lengths,
                          robot_radius, payload_radius, cable_length):
    count = states.shape[0]
    if count == 0 or polygon_lengths.shape[0] == 0:
        return math.inf
    per_state = np.full(count, math.inf, dtype=np.float64)
    for sidx in prange(count):
        state = states[sidx]
        px = state[0]
        py = state[1]
        load_x, load_y = _payload_position_nb(state, cable_length)
        best = math.inf
        for obs in range(polygon_lengths.shape[0]):
            n = int(polygon_lengths[obs])
            poly = polygons[obs]
            cq = _circle_polygon_clearance_nb(px, py, robot_radius, poly, n)
            if cq < best:
                best = cq
            cl = _circle_polygon_clearance_nb(load_x, load_y, payload_radius, poly, n)
            if cl < best:
                best = cl
        per_state[sidx] = best
    return np.min(per_state)


def minimum_clearance(states: Array, obstacles: Sequence, cfg: MPPIConfig) -> float:
    polygons, lengths = _obstacle_polygon_arrays(obstacles)
    return float(_minimum_clearance_nb(
        np.ascontiguousarray(np.asarray(states, dtype=np.float64)),
        polygons,
        lengths,
        float(cfg.total_drone_radius),
        float(cfg.payload_radius),
        float(cfg.cable_length),
    ))


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