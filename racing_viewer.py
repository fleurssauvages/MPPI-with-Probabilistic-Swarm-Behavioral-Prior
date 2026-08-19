from __future__ import annotations

import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from numba import njit, prange

from system import ackermann, four_wheel, controller as controller_core
from system.ackermann import _dynamic_ackermann_step_nb
from system.four_wheel import _dynamic_four_wheel_step_nb

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:
    raise SystemExit("Tkinter is required to run the racing viewer.") from exc

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Polygon


VARIANTS = [
    ("Centerline iLQR", "planner_ilqr"),
    ("Standard MPPI", "standard_mppi"),
    ("Control bank", "control_bank_mppi"),
    ("Corridor prior", "corridor_prior_mppi"),
    ("Gaussian prior", "gaussian_prior_mppi"),
    ("SPG prior", "sensitivity_projected_gaussian_prior_mppi"),
]
DISPLAY_TO_VARIANT = dict(VARIANTS)
VARIANT_TO_DISPLAY = {value: label for label, value in VARIANTS}
VEHICLE_SYSTEMS = list(controller_core.VEHICLE_SYSTEMS)
DISPLAY_TO_VEHICLE = dict(VEHICLE_SYSTEMS)
VEHICLE_TO_DISPLAY = {value: label for label, value in VEHICLE_SYSTEMS}


TRACK_WIDTH = 20.0
TRACK_HEIGHT = 8.0
ROAD_WIDTH = 3.0
TRACK_X0 = 0.0
TRACK_Y0 = 0.0
TRACK_CENTER_X = TRACK_X0 + 0.5 * TRACK_WIDTH
TRACK_CENTER_Y = TRACK_Y0 + 0.5 * TRACK_HEIGHT
OUTER_RADIUS = 0.5 * TRACK_HEIGHT
CENTERLINE_RADIUS = OUTER_RADIUS - 0.5 * ROAD_WIDTH
LEFT_ARC_X = TRACK_X0 + OUTER_RADIUS
RIGHT_ARC_X = TRACK_X0 + TRACK_WIDTH - OUTER_RADIUS
STRAIGHT_LENGTH = RIGHT_ARC_X - LEFT_ARC_X
TRACK_LENGTH = 2.0 * STRAIGHT_LENGTH + 2.0 * math.pi * CENTERLINE_RADIUS
START_S = 0.5 * STRAIGHT_LENGTH


@dataclass
class RaceResult:
    states: np.ndarray
    controls: np.ndarray
    nominal_predictions: list[np.ndarray]
    mppi_predictions: list[np.ndarray]
    temperatures: np.ndarray
    esses: np.ndarray
    feasible_counts: np.ndarray
    cumulative_progress: np.ndarray
    lap_times: list[float]
    requested_laps: int
    completed_laps: int
    off_track: bool
    runtime_s: float
    variant_value: str
    cfg: object
    model_name: str


@njit(cache=True, inline="always")
def _clamp_nb(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


@njit(cache=True, inline="always")
def _project_centerline_nb(px: float, py: float) -> tuple[float, float]:
    """Return (arc coordinate modulo lap, squared distance to the centerline)."""
    r = CENTERLINE_RADIUS
    x_left = LEFT_ARC_X
    x_right = RIGHT_ARC_X
    yc = TRACK_CENTER_Y
    straight = STRAIGHT_LENGTH


    qx = _clamp_nb(px, x_left, x_right)
    qy = yc - r
    dx = px - qx
    dy = py - qy
    best_d2 = dx * dx + dy * dy
    best_s = qx - x_left


    theta = math.atan2(py - yc, px - x_right)
    theta = _clamp_nb(theta, -0.5 * math.pi, 0.5 * math.pi)
    qx = x_right + r * math.cos(theta)
    qy = yc + r * math.sin(theta)
    dx = px - qx
    dy = py - qy
    d2 = dx * dx + dy * dy
    if d2 < best_d2:
        best_d2 = d2
        best_s = straight + r * (theta + 0.5 * math.pi)


    qx = _clamp_nb(px, x_left, x_right)
    qy = yc + r
    dx = px - qx
    dy = py - qy
    d2 = dx * dx + dy * dy
    if d2 < best_d2:
        best_d2 = d2
        best_s = straight + math.pi * r + (x_right - qx)


    theta = math.atan2(py - yc, px - x_left)
    if theta < 0.5 * math.pi:
        theta += 2.0 * math.pi
    theta = _clamp_nb(theta, 0.5 * math.pi, 1.5 * math.pi)
    qx = x_left + r * math.cos(theta)
    qy = yc + r * math.sin(theta)
    dx = px - qx
    dy = py - qy
    d2 = dx * dx + dy * dy
    if d2 < best_d2:
        best_d2 = d2
        best_s = 2.0 * straight + math.pi * r + r * (theta - 0.5 * math.pi)

    if best_s >= TRACK_LENGTH:
        best_s -= TRACK_LENGTH
    return best_s, best_d2


@njit(cache=True, inline="always")
def _signed_progress_delta_nb(new_s: float, old_s: float) -> float:
    ds = new_s - old_s
    half = 0.5 * TRACK_LENGTH
    if ds > half:
        ds -= TRACK_LENGTH
    elif ds < -half:
        ds += TRACK_LENGTH
    return ds


@njit(cache=True, inline="always")
def _vehicle_on_track_nb(state: np.ndarray, vehicle_length: float, vehicle_width: float, hard_collision_clearance: float) -> bool:
    """Check a dense set of footprint points against the exact 2 m road tube."""
    px = state[0]
    py = state[1]
    psi = state[2]
    c = math.cos(psi)
    s = math.sin(psi)
    half_l = 0.5 * vehicle_length
    half_w = 0.5 * vehicle_width
    half_road = max(0.0, 0.5 * ROAD_WIDTH - hard_collision_clearance)
    allowed2 = half_road * half_road


    for ia in range(5):
        a = -half_l + (2.0 * half_l) * (ia / 4.0)
        for ib in range(3):
            b = -half_w + (2.0 * half_w) * (ib / 2.0)
            wx = px + c * a - s * b
            wy = py + s * a + c * b
            _, d2 = _project_centerline_nb(wx, wy)
            if d2 > allowed2:
                return False
    return True


@njit(cache=True, parallel=True)
def _racing_rollout_costs_ackermann_nb(
    x0: np.ndarray,
    controls: np.ndarray,
    current_s: float,
    vehicle_length: float,
    vehicle_width: float,
    hard_collision_clearance: float,
    dt: float,
    front_axle_distance: float,
    rear_axle_distance: float,
    mass: float,
    yaw_inertia: float,
    cornering_stiffness_front: float,
    cornering_stiffness_rear: float,
    tire_friction_coefficient: float,
    gravity: float,
    aerodynamic_drag_coefficient: float,
    rolling_resistance_force: float,
    minimum_tire_speed: float,
    dynamics_substeps: int,
    v_min: float,
    v_max: float,
    lateral_velocity_limit: float,
    yaw_rate_limit: float,
    accel_min: float,
    accel_max: float,
    steering_min: float,
    steering_max: float,
    steering_rate_min: float,
    steering_rate_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rollouts = controls.shape[0]
    horizon = controls.shape[1]
    costs = np.empty(n_rollouts, dtype=np.float64)
    collisions = np.zeros(n_rollouts, dtype=np.bool_)
    terminal_progress = np.empty(n_rollouts, dtype=np.float64)
    for n in prange(n_rollouts):
        state = np.empty(7, dtype=np.float64)
        next_state = np.empty(7, dtype=np.float64)
        for j in range(7):
            state[j] = x0[j]
        previous_s = current_s
        cumulative = 0.0
        prefix_sum = 0.0
        hit = False
        for t in range(horizon):
            values = _dynamic_ackermann_step_nb(
                state,
                controls[n, t, 0],
                controls[n, t, 1],
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
            )
            for j in range(7):
                next_state[j] = values[j]
            if not _vehicle_on_track_nb(next_state, vehicle_length, vehicle_width, hard_collision_clearance):
                hit = True
                break
            s_mod, _ = _project_centerline_nb(next_state[0], next_state[1])
            cumulative += _signed_progress_delta_nb(s_mod, previous_s)
            previous_s = s_mod
            prefix_sum += cumulative
            for j in range(7):
                state[j] = next_state[j]
        collisions[n] = hit
        terminal_progress[n] = cumulative
        costs[n] = math.inf if hit else -prefix_sum / max(1, horizon)
    return costs, collisions, terminal_progress



@njit(cache=True, parallel=True)
def _racing_rollout_costs_four_wheel_nb(
    x0: np.ndarray,
    controls: np.ndarray,
    current_s: float,
    vehicle_length: float,
    vehicle_width: float,
    hard_collision_clearance: float,
    dt: float,
    front_axle_distance: float,
    rear_axle_distance: float,
    mass: float,
    yaw_inertia: float,
    cornering_stiffness_front: float,
    cornering_stiffness_rear: float,
    tire_friction_coefficient: float,
    gravity: float,
    aerodynamic_drag_coefficient: float,
    rolling_resistance_force: float,
    minimum_tire_speed: float,
    dynamics_substeps: int,
    v_min: float,
    v_max: float,
    lateral_velocity_limit: float,
    yaw_rate_limit: float,
    accel_min: float,
    accel_max: float,
    steering_min: float,
    steering_max: float,
    steering_rate_min: float,
    steering_rate_max: float,
    track_width: float,
    wheel_radius: float,
    wheel_inertia: float,
    longitudinal_tire_stiffness: float,
    roll_inertia: float,
    roll_stiffness: float,
    roll_damping: float,
    cg_height: float,
    roll_center_height: float,
    wheel_damping: float,
    wheel_speed_limit: float,
    drive_bias_front: float,
    minimum_normal_load_fraction: float,
    roll_angle_limit: float,
    roll_rate_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rollouts = controls.shape[0]
    horizon = controls.shape[1]
    costs = np.empty(n_rollouts, dtype=np.float64)
    collisions = np.zeros(n_rollouts, dtype=np.bool_)
    terminal_progress = np.empty(n_rollouts, dtype=np.float64)
    for n in prange(n_rollouts):
        state = np.empty(13, dtype=np.float64)
        next_state = np.empty(13, dtype=np.float64)
        for j in range(13):
            state[j] = x0[j]
        previous_s = current_s
        cumulative = 0.0
        prefix_sum = 0.0
        hit = False
        for t in range(horizon):
            values = _dynamic_four_wheel_step_nb(
                state,
                controls[n, t, 0],
                controls[n, t, 1],
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
            )
            for j in range(13):
                next_state[j] = values[j]
            if not _vehicle_on_track_nb(next_state, vehicle_length, vehicle_width, hard_collision_clearance):
                hit = True
                break
            s_mod, _ = _project_centerline_nb(next_state[0], next_state[1])
            cumulative += _signed_progress_delta_nb(s_mod, previous_s)
            previous_s = s_mod
            prefix_sum += cumulative
            for j in range(13):
                state[j] = next_state[j]
        collisions[n] = hit
        terminal_progress[n] = cumulative
        costs[n] = math.inf if hit else -prefix_sum / max(1, horizon)
    return costs, collisions, terminal_progress


def _evaluate_control_batch(
    state: np.ndarray,
    controls: np.ndarray,
    current_s: float,
    cfg: object,
    model: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = _racing_rollout_costs_ackermann_nb if model.MODEL_NAME == 'ackermann' else _racing_rollout_costs_four_wheel_nb
    return kernel(
        np.ascontiguousarray(np.asarray(state, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(controls, dtype=np.float64)),
        float(current_s),
        float(cfg.vehicle_length),
        float(cfg.vehicle_width),
        float(cfg.hard_collision_clearance),
        *model._dynamic_model_arguments(cfg),
    )


def centerline_point_tangent(s_value: float) -> tuple[np.ndarray, np.ndarray]:
    s = float(s_value % TRACK_LENGTH)
    r = CENTERLINE_RADIUS
    straight = STRAIGHT_LENGTH
    yc = TRACK_CENTER_Y

    if s < straight:
        return (
            np.asarray([LEFT_ARC_X + s, yc - r], dtype=np.float64),
            np.asarray([1.0, 0.0], dtype=np.float64),
        )

    s -= straight
    arc_len = math.pi * r
    if s < arc_len:
        theta = -0.5 * math.pi + s / r
        return (
            np.asarray([RIGHT_ARC_X + r * math.cos(theta), yc + r * math.sin(theta)], dtype=np.float64),
            np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64),
        )

    s -= arc_len
    if s < straight:
        return (
            np.asarray([RIGHT_ARC_X - s, yc + r], dtype=np.float64),
            np.asarray([-1.0, 0.0], dtype=np.float64),
        )

    s -= straight
    theta = 0.5 * math.pi + s / r
    return (
        np.asarray([LEFT_ARC_X + r * math.cos(theta), yc + r * math.sin(theta)], dtype=np.float64),
        np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64),
    )


REFERENCE_CACHE_COUNT = 8192


def _build_reference_cache() -> tuple[np.ndarray, np.ndarray, float]:
    points = np.empty((REFERENCE_CACHE_COUNT, 2), dtype=np.float64)
    tangents = np.empty((REFERENCE_CACHE_COUNT, 2), dtype=np.float64)
    ds = TRACK_LENGTH / float(REFERENCE_CACHE_COUNT)
    for i in range(REFERENCE_CACHE_COUNT):
        point, tangent = centerline_point_tangent(i * ds)
        points[i] = point
        tangents[i] = tangent
    return np.ascontiguousarray(points), np.ascontiguousarray(tangents), ds


_REFERENCE_CACHE_POINTS, _REFERENCE_CACHE_TANGENTS, _REFERENCE_CACHE_DS = _build_reference_cache()


@njit(cache=True, inline='always')
def _cached_centerline_point_tangent_nb(s_value: float, points: np.ndarray, tangents: np.ndarray, ds: float) -> tuple[float, float, float, float]:
    wrapped = s_value % TRACK_LENGTH
    u = wrapped / ds
    i0 = int(math.floor(u)) % points.shape[0]
    w = u - math.floor(u)
    i1 = (i0 + 1) % points.shape[0]
    x = (1.0 - w) * points[i0, 0] + w * points[i1, 0]
    y = (1.0 - w) * points[i0, 1] + w * points[i1, 1]
    tx = (1.0 - w) * tangents[i0, 0] + w * tangents[i1, 0]
    ty = (1.0 - w) * tangents[i0, 1] + w * tangents[i1, 1]
    norm = math.sqrt(tx * tx + ty * ty)
    if norm > 1e-15:
        tx /= norm
        ty /= norm
    return x, y, tx, ty


@njit(cache=True)
def _sample_cached_centerline_nb(start_s: float, distance: float, count: int, points: np.ndarray, tangents: np.ndarray, ds: float) -> np.ndarray:
    n = max(2, int(count))
    out = np.empty((n, 2), dtype=np.float64)
    denom = max(1, n - 1)
    for i in range(n):
        s_value = start_s + distance * (i / float(denom))
        x, y, _, _ = _cached_centerline_point_tangent_nb(s_value, points, tangents, ds)
        out[i, 0] = x
        out[i, 1] = y
    return out


@njit(cache=True, parallel=True)
def _sample_offset_reference_bank_nb(start_s: float, distance: float, count: int, offsets: np.ndarray, points: np.ndarray, tangents: np.ndarray, ds: float) -> np.ndarray:
    n_paths = offsets.shape[0]
    n = max(2, int(count))
    out = np.empty((n_paths, n, 2), dtype=np.float64)
    denom = max(1, n - 1)
    for m in prange(n_paths):
        offset = offsets[m]
        for i in range(n):
            s_value = start_s + distance * (i / float(denom))
            x, y, tx, ty = _cached_centerline_point_tangent_nb(s_value, points, tangents, ds)
            out[m, i, 0] = x - offset * ty
            out[m, i, 1] = y + offset * tx
    return out


def sample_centerline(start_s: float, distance: float, count: int) -> np.ndarray:
    return np.ascontiguousarray(_sample_cached_centerline_nb(
        float(start_s), float(distance), int(count),
        _REFERENCE_CACHE_POINTS, _REFERENCE_CACHE_TANGENTS, float(_REFERENCE_CACHE_DS),
    ))


def sample_closed_centerline(count: int = 600) -> np.ndarray:
    return sample_centerline(0.0, TRACK_LENGTH, count)



def racing_prior_covariance(reference: np.ndarray) -> np.ndarray:
    """Centerline prior covariance whose 1-sigma diameter equals the road width."""
    sigma = 0.5 * ROAD_WIDTH
    variance = sigma * sigma
    cov = np.zeros((len(reference), 2, 2), dtype=np.float64)
    cov[:, 0, 0] = variance
    cov[:, 1, 1] = variance
    return np.ascontiguousarray(cov)



def _control_bank_controls(
    state: np.ndarray,
    current_s: float,
    cfg: object,
    model: object,
    preview_distance: float,
    ref_count: int,
    count: int,
) -> np.ndarray:
    count = max(1, int(count))
    half_available = max(0.0, 0.5 * ROAD_WIDTH - 0.5 * float(cfg.vehicle_width))
    offsets = np.linspace(-half_available, half_available, count, dtype=np.float64)
    references = _sample_offset_reference_bank_nb(
        float(current_s),
        float(preview_distance),
        int(ref_count),
        np.ascontiguousarray(offsets),
        _REFERENCE_CACHE_POINTS,
        _REFERENCE_CACHE_TANGENTS,
        float(_REFERENCE_CACHE_DS),
    )
    controls = model.nominal_controls_batch_to_track_paths(
        np.ascontiguousarray(np.asarray(state, dtype=np.float64)),
        np.ascontiguousarray(references),
        cfg,
    )
    return model.clip_control_batch(np.asarray(controls, dtype=np.float64), cfg)

def _stadium_boundary(radius: float, count_per_arc: int = 120) -> np.ndarray:
    bottom = np.asarray([[LEFT_ARC_X, TRACK_CENTER_Y - radius], [RIGHT_ARC_X, TRACK_CENTER_Y - radius]])
    right_angles = np.linspace(-0.5 * math.pi, 0.5 * math.pi, count_per_arc)
    right = np.column_stack((
        RIGHT_ARC_X + radius * np.cos(right_angles),
        TRACK_CENTER_Y + radius * np.sin(right_angles),
    ))
    top = np.asarray([[RIGHT_ARC_X, TRACK_CENTER_Y + radius], [LEFT_ARC_X, TRACK_CENTER_Y + radius]])
    left_angles = np.linspace(0.5 * math.pi, 1.5 * math.pi, count_per_arc)
    left = np.column_stack((
        LEFT_ARC_X + radius * np.cos(left_angles),
        TRACK_CENTER_Y + radius * np.sin(left_angles),
    ))
    return np.vstack((bottom, right[1:], top[1:], left[1:]))


def initial_race_state(model_name: str = 'ackermann') -> np.ndarray:
    point, tangent = centerline_point_tangent(START_S)
    heading = math.atan2(float(tangent[1]), float(tangent[0]))
    if model_name == 'four_wheel':
        return np.asarray([point[0], point[1], heading, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray([point[0], point[1], heading, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def make_racing_config(
    num_rollouts: int,
    lbps_delta: float,
    *,
    horizon: int,
    temporal_noise_smoothing: float,
    sigma0_scale: float,
    v_max: float,
    hard_collision_clearance: float,
    model_name: str = 'ackermann',
) -> object:
    model = controller_core.resolve_vehicle_model(model_name)
    cfg = model.MPPIConfig(num_rollouts=int(num_rollouts))
    cfg.adaptive_temperature_lbps = True
    cfg.lbps_delta = float(lbps_delta)
    cfg.horizon = int(horizon)
    cfg.temporal_noise_smoothing = float(temporal_noise_smoothing)


    sigma0_scale = float(sigma0_scale)
    sigma0_std_scale = math.sqrt(sigma0_scale)
    cfg.noise_accel = float(cfg.noise_accel) * sigma0_std_scale
    cfg.noise_steering_rate = float(cfg.noise_steering_rate) * sigma0_std_scale
    cfg.v_max = float(v_max)
    cfg.hard_collision_clearance = float(hard_collision_clearance)
    return cfg


def racing_controller_step(
    variant_value: str,
    state: np.ndarray,
    current_s: float,
    cfg: object,
    model: object,
    rng: np.random.Generator,
    record_predictions: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        variant = controller_core.ControllerVariant(variant_value)
    except ValueError as exc:
        raise ValueError(f"Unsupported racing controller variant: {variant_value}") from exc

    preview_distance = max(2.0 * TRACK_LENGTH, 2.0 * cfg.v_max * cfg.dt * cfg.horizon)
    ref_count = max(240, 5 * int(cfg.horizon))
    reference = sample_centerline(current_s, preview_distance, ref_count)
    prior_cov: Optional[np.ndarray] = None
    controls: Optional[np.ndarray] = None

    if variant == controller_core.ControllerVariant.STANDARD_MPPI:
        nominal = model.nominal_controls_to_track_path(state, reference, cfg)
        controls = controller_core.sample_controls_around_nominal(
            model, nominal, int(cfg.num_rollouts), cfg, rng
        )
    else:
        prior_cov = racing_prior_covariance(reference)
        if variant == controller_core.ControllerVariant.CORRIDOR_PRIOR_MPPI:
            nominal = model.nominal_controls_to_track_path(state, reference, cfg, prior_cov)
            controls = controller_core.sample_controls_around_nominal(
                model, nominal, int(cfg.num_rollouts), cfg, rng
            )
        elif variant == controller_core.ControllerVariant.GAUSSIAN_PRIOR_MPPI:
            prior_mode = controller_core.MPPIHomotopyMode(
                signature=(0,),
                probability=1.0,
                mean_path=np.ascontiguousarray(reference),
                cov_blocks=np.ascontiguousarray(prior_cov),
                sample_paths=None,
                arc_length=np.linspace(0.0, preview_distance, len(reference), dtype=np.float64),
                gaussian_variance=np.full(len(reference), 0.5 * (prior_cov[0, 0, 0] + prior_cov[0, 1, 1]), dtype=np.float64),
            )
            controls, nominal = controller_core.sample_gaussian_controls_with_nominal(
                model, state, prior_mode, int(cfg.num_rollouts), cfg, rng
            )
        elif variant == controller_core.ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI:
            prior_mode = controller_core.MPPIHomotopyMode(
                signature=(0,),
                probability=1.0,
                mean_path=np.ascontiguousarray(reference),
                cov_blocks=np.ascontiguousarray(prior_cov),
                sample_paths=None,
                arc_length=np.linspace(0.0, preview_distance, len(reference), dtype=np.float64),
                gaussian_variance=np.full(len(reference), 0.5 * (prior_cov[0, 0, 0] + prior_cov[0, 1, 1]), dtype=np.float64),
            )
            controls, nominal = controller_core.sample_sensitivity_projected_gaussian_controls_with_nominal(
                model, state, prior_mode, int(cfg.num_rollouts), cfg, rng
            )
        elif variant in {
            controller_core.ControllerVariant.PLANNER_ILQR,
            controller_core.ControllerVariant.CONTROL_BANK_MPPI,
        }:
            nominal = model.nominal_controls_to_track_path(state, reference, cfg, prior_cov)
        else:
            raise ValueError(f"Unsupported racing controller variant: {variant_value}")

    nominal = np.ascontiguousarray(np.asarray(nominal, dtype=np.float64))
    nominal_states = (
        np.asarray(model.rollout_single(state, nominal, cfg), dtype=np.float64)
        if record_predictions else np.zeros((0, int(model.STATE_DIM)), dtype=np.float64)
    )

    if variant == controller_core.ControllerVariant.PLANNER_ILQR:
        candidate = nominal.copy()
        candidate_costs, candidate_collision, candidate_progress = _evaluate_control_batch(
            state, candidate[None, :, :], current_s, cfg, model
        )
        finite_count = int(not bool(candidate_collision[0]))
        temperature = float("nan")
        ess = 1.0 if finite_count else 0.0
        collision_rollouts = int(bool(candidate_collision[0]))
        best_terminal_progress = float(candidate_progress[0]) if finite_count else float("nan")
        rollout_count = 1
        output_source = "centerline_ilqr"
    elif variant == controller_core.ControllerVariant.CONTROL_BANK_MPPI:
        bank_controls = _control_bank_controls(
            state, current_s, cfg, model, preview_distance, ref_count, int(cfg.num_rollouts)
        )
        costs, collisions, terminal_progress = _evaluate_control_batch(
            state, bank_controls, current_s, cfg, model
        )
        finite_ids = np.flatnonzero(np.isfinite(costs))
        finite_count = int(finite_ids.size)
        collision_rollouts = int(np.count_nonzero(collisions))
        if finite_ids.size:
            best_id = int(finite_ids[np.argmin(costs[finite_ids])])
            candidate = np.asarray(bank_controls[best_id], dtype=np.float64).copy()
            candidate_costs = np.asarray([costs[best_id]], dtype=np.float64)
            candidate_collision = np.asarray([False], dtype=np.bool_)
            candidate_progress = np.asarray([terminal_progress[best_id]], dtype=np.float64)
            best_terminal_progress = float(np.max(terminal_progress[finite_ids]))
            output_source = "best_control_bank_trajectory"
        else:
            candidate = nominal.copy()
            candidate_costs, candidate_collision, candidate_progress = _evaluate_control_batch(
                state, candidate[None, :, :], current_s, cfg, model
            )
            best_terminal_progress = float("nan")
            output_source = "centerline_ilqr_no_feasible_bank"
        temperature = float("nan")
        ess = 1.0 if finite_count else 0.0
        rollout_count = len(bank_controls)
    else:
        controls = np.ascontiguousarray(np.asarray(controls, dtype=np.float64))
        if len(controls):
            controls[0] = nominal
        costs, collisions, terminal_progress = _evaluate_control_batch(
            state, controls, current_s, cfg, model
        )
        temperature_info = controller_core.resolve_mppi_temperature(costs, cfg)
        temperature = float(temperature_info[0])
        ess = float(temperature_info[2])
        candidate = controller_core.mppi_weighted_control_sequence(
            model, costs, controls, cfg, temperature=temperature
        )
        candidate_costs, candidate_collision, candidate_progress = _evaluate_control_batch(
            state, np.asarray(candidate, dtype=np.float64)[None, :, :], current_s, cfg, model
        )
        finite_mask = np.isfinite(costs)
        finite_count = int(np.count_nonzero(finite_mask))
        collision_rollouts = int(np.count_nonzero(collisions))
        best_terminal_progress = (
            float(np.max(terminal_progress[finite_mask])) if finite_count else float("nan")
        )
        rollout_count = len(controls)
        output_source = "weighted_candidate"

    candidate = np.ascontiguousarray(np.asarray(candidate, dtype=np.float64))
    candidate_states = (
        np.asarray(model.rollout_single(state, candidate, cfg), dtype=np.float64)
        if record_predictions else np.zeros((0, int(model.STATE_DIM)), dtype=np.float64)
    )
    info: dict[str, object] = {
        "planned_control_sequence": candidate,
        "optimal_traj": candidate_states,
        "nominal_traj": nominal_states,
        "temperature": temperature,
        "ess": ess,
        "finite_rollouts": finite_count,
        "rollout_count": int(rollout_count),
        "collision_rollouts": collision_rollouts,
        "candidate_collision": bool(candidate_collision[0]),
        "candidate_cost": float(candidate_costs[0]),
        "candidate_progress": float(candidate_progress[0]),
        "best_terminal_progress": best_terminal_progress,
        "output_source": output_source,
    }
    return candidate[0].copy(), info


def _warm_racing_kernels(
    variant_value: str,
    state: np.ndarray,
    current_s: float,
    cfg: object,
    model: object,
    seed: int,
) -> None:
    warm_cfg = replace(cfg)
    warm_cfg.horizon = min(int(cfg.horizon), 8)
    warm_cfg.num_rollouts = min(max(16, int(cfg.num_rollouts)), 32)
    racing_controller_step(
        variant_value,
        state,
        current_s,
        warm_cfg,
        model,
        np.random.default_rng(int(seed) + 999983),
        record_predictions=False,
    )

def run_race(
    *,
    variant_value: str = "standard_mppi",
    laps: int,
    num_rollouts: int,
    lbps_delta: float,
    seed: int,
    horizon: int = 50,
    temporal_noise_smoothing: float = 0.1,
    sigma0_scale: float = 1.0,
    v_max: float = 8.4,
    hard_collision_clearance: float = 0.01,
    model_name: str = 'ackermann',
    record_predictions: bool = True,
    max_steps: Optional[int] = None,
) -> RaceResult:
    if variant_value not in VARIANT_TO_DISPLAY:
        raise ValueError(f"Unsupported racing variant: {variant_value}")
    model = controller_core.resolve_vehicle_model(model_name)
    laps = int(laps)
    if laps < 1:
        raise ValueError("Laps must be at least 1.")
    if num_rollouts < 32:
        raise ValueError("Rollouts per step must be at least 32.")
    if not math.isfinite(lbps_delta) or not (0.0 < lbps_delta < 1.0):
        raise ValueError("LBPS delta must be strictly between 0 and 1.")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("Horizon H must be at least 1.")
    temporal_noise_smoothing = float(temporal_noise_smoothing)
    if not math.isfinite(temporal_noise_smoothing) or not (0.0 <= temporal_noise_smoothing < 1.0):
        raise ValueError("Temporal noise smoothing must be in [0, 1).")
    sigma0_scale = float(sigma0_scale)
    if not math.isfinite(sigma0_scale) or sigma0_scale <= 0.0:
        raise ValueError("Sigma_0 covariance scale must be positive.")
    v_max = float(v_max)
    if not math.isfinite(v_max) or v_max <= 0.0:
        raise ValueError("v_max must be positive.")
    hard_collision_clearance = float(hard_collision_clearance)
    if not math.isfinite(hard_collision_clearance) or hard_collision_clearance < 0.0:
        raise ValueError("hard_collision_clearance must be nonnegative.")

    cfg = make_racing_config(
        num_rollouts,
        lbps_delta,
        horizon=horizon,
        temporal_noise_smoothing=temporal_noise_smoothing,
        sigma0_scale=sigma0_scale,
        v_max=v_max,
        hard_collision_clearance=hard_collision_clearance,
        model_name=model_name,
    )
    rng = np.random.default_rng(int(seed))
    state = initial_race_state(model_name)
    start_s, _ = _project_centerline_nb(float(state[0]), float(state[1]))
    current_s = float(start_s)
    cumulative = 0.0
    target_progress = laps * TRACK_LENGTH

    if max_steps is None:


        theoretical_steps = TRACK_LENGTH / max(float(cfg.v_max) * float(cfg.dt), 1e-9)
        max_steps = int(math.ceil(4.0 * laps * theoretical_steps + 10.0 / cfg.dt))

    states = [state.copy()]
    controls: list[np.ndarray] = []
    nominal_predictions: list[np.ndarray] = []
    mppi_predictions: list[np.ndarray] = []
    temperatures: list[float] = []
    esses: list[float] = []
    feasible_counts: list[int] = []
    cumulative_history = [0.0]
    lap_times: list[float] = []
    completed_laps = 0
    off_track = False

    _warm_racing_kernels(variant_value, state, current_s, cfg, model, int(seed))

    t0 = time.perf_counter()
    for step in range(int(max_steps)):
        control, info = racing_controller_step(
            variant_value, state, current_s, cfg, model, rng, record_predictions=record_predictions
        )
        control = model.apply_final_output(
            state, control, controls[-1] if controls else None, [], np.zeros(2), cfg
        )
        next_state = np.asarray(model.vehicle_step(state, control, cfg), dtype=np.float64)

        new_s, _ = _project_centerline_nb(float(next_state[0]), float(next_state[1]))
        ds = float(_signed_progress_delta_nb(float(new_s), float(current_s)))
        cumulative += ds
        current_s = float(new_s)

        controls.append(control.copy())
        states.append(next_state.copy())
        temperatures.append(float(info["temperature"]))
        esses.append(float(info["ess"]))
        feasible_counts.append(int(info["finite_rollouts"]))
        cumulative_history.append(cumulative)
        if record_predictions:
            nominal_predictions.append(np.asarray(info["nominal_traj"], dtype=np.float64).copy())
            mppi_predictions.append(np.asarray(info["optimal_traj"], dtype=np.float64).copy())

        new_completed = int(math.floor(max(cumulative, 0.0) / TRACK_LENGTH + 1e-12))
        while completed_laps < min(new_completed, laps):
            completed_laps += 1
            lap_times.append((step + 1) * float(cfg.dt))

        if not bool(_vehicle_on_track_nb(next_state, float(cfg.vehicle_length), float(cfg.vehicle_width), float(cfg.hard_collision_clearance))):
            off_track = True
            state = next_state
            break

        state = next_state
        if cumulative >= target_progress:
            completed_laps = laps
            break

    runtime = time.perf_counter() - t0
    return RaceResult(
        states=np.asarray(states, dtype=np.float64),
        controls=np.asarray(controls, dtype=np.float64),
        nominal_predictions=nominal_predictions,
        mppi_predictions=mppi_predictions,
        temperatures=np.asarray(temperatures, dtype=np.float64),
        esses=np.asarray(esses, dtype=np.float64),
        feasible_counts=np.asarray(feasible_counts, dtype=np.int64),
        cumulative_progress=np.asarray(cumulative_history, dtype=np.float64),
        lap_times=lap_times,
        requested_laps=laps,
        completed_laps=completed_laps,
        off_track=off_track,
        runtime_s=runtime,
        variant_value=variant_value,
        cfg=cfg,
        model_name=model_name,
    )


class NASCARViewer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MPPI NASCAR racing viewer")
        self.root.minsize(1080, 700)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.geometry(
                f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0"
            )

        self.worker: Optional[threading.Thread] = None
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.result: Optional[RaceResult] = None
        self.frame_index = 0
        self.playing = False
        self.after_id: Optional[str] = None
        self.poll_after_id: Optional[str] = None
        self.closing = False
        self.updating_slider = False

        self._outer = _stadium_boundary(OUTER_RADIUS)
        self._inner = _stadium_boundary(OUTER_RADIUS - ROAD_WIDTH)
        self._centerline = sample_closed_centerline()

        self._build_ui()
        self.poll_after_id = self.root.after(100, self._poll_worker)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0, minsize=330)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=12, width=330)
        controls.grid(row=0, column=0, sticky="ns")
        controls.grid_propagate(False)
        controls.columnconfigure(0, weight=1)

        plot_frame = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        ttk.Label(controls, text="NASCAR race", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        
        base_cfg = ackermann.MPPIConfig()
        self.vehicle_var = tk.StringVar(value="Ackermann")
        self.variant_var = tk.StringVar(value="SPG prior")
        self.laps_var = tk.StringVar(value="10")
        self.rollouts_var = tk.StringVar(value="4096")
        self.lbps_delta_var = tk.StringVar(value="0.9")
        self.seed_var = tk.StringVar(value="1")
        self.speed_var = tk.StringVar(value="1.0")
        self.horizon_var = tk.DoubleVar(value=float(10))
        self.temporal_noise_var = tk.DoubleVar(value=float(base_cfg.temporal_noise_smoothing))
        self.sigma0_scale_var = tk.DoubleVar(value=1.0)
        self.vmax_var = tk.DoubleVar(value=float(8.0))
        self.hard_collision_clearance_var = tk.DoubleVar(value=float(base_cfg.hard_collision_clearance))
        self.status_var = tk.StringVar(value="Ready")
        self.frame_label_var = tk.StringVar(value="Frame 0 / 0")

        row = 2
        row = self._add_combo(controls, row, "Vehicle model", self.vehicle_var, [label for label, _ in VEHICLE_SYSTEMS])
        row = self._add_combo(controls, row, "Controller variant", self.variant_var, [label for label, _ in VARIANTS])
        row = self._add_entry(controls, row, "Number of laps", self.laps_var)
        row = self._add_entry(controls, row, "Rollouts per step", self.rollouts_var)
        row = self._add_entry(controls, row, "LBPS delta", self.lbps_delta_var)
        row = self._add_entry(controls, row, "Controller seed", self.seed_var)

        ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=(10, 6))
        row += 1
        ttk.Label(controls, text="Racing controller config", font=("TkDefaultFont", 11, "bold")).grid(
            row=row, column=0, sticky="w"
        )
        row += 1
        row = self._add_slider(
            controls, row, "Horizon H", self.horizon_var, 5.0, 15.0,
            formatter=lambda value: str(int(round(value))),
        )
        row = self._add_slider(
            controls, row, "Temporal noise", self.temporal_noise_var, 0.0, 0.95,
            formatter=lambda value: f"{value:.2f}",
        )
        row = self._add_slider(
            controls, row, "Sigma_0 covariance scale", self.sigma0_scale_var, 0.1, 4.0,
            formatter=lambda value: f"{value:.2f}x",
        )
        row = self._add_slider(
            controls, row, "v_max [m/s]", self.vmax_var, 1.0, 15.0,
            formatter=lambda value: f"{value:.1f}",
        )
        row = self._add_slider(
            controls, row, "hard_collision_clearance [m]", self.hard_collision_clearance_var, 0.0, 0.1,
            formatter=lambda value: f"{value:.3f}",
        )

        buttons = ttk.Frame(controls)
        buttons.grid(row=row, column=0, sticky="ew", pady=(14, 6))
        buttons.columnconfigure((0, 1), weight=1)
        self.run_button = ttk.Button(buttons, text="Run race", command=self.run_selected)
        self.run_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.play_button = ttk.Button(buttons, text="Play", command=self.toggle_play, state="disabled")
        self.play_button.grid(row=1, column=0, sticky="ew", padx=(0, 3))
        self.restart_button = ttk.Button(buttons, text="Restart", command=self.restart_animation, state="disabled")
        self.restart_button.grid(row=1, column=1, sticky="ew", padx=(3, 0))
        row += 1

        ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=10)
        row += 1
        ttk.Label(controls, text="Animation", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Label(controls, textvariable=self.frame_label_var).grid(row=row, column=0, sticky="w", pady=(4, 2))
        row += 1
        self.frame_scale = ttk.Scale(
            controls, from_=0, to=0, orient="horizontal", command=self._on_frame_slider, state="disabled"
        )
        self.frame_scale.grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=10)
        row += 1
        ttk.Label(controls, text="Status", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, sticky="w")
        row += 1
        self.status_label = ttk.Label(
            controls, textvariable=self.status_var, wraplength=295, justify="left", anchor="nw"
        )
        self.status_label.grid(row=row, column=0, sticky="nsew", pady=(4, 0))
        controls.rowconfigure(row, weight=1)

        self.figure = Figure(figsize=(10, 7))
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.93)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")
        self._init_plot_artists()
        self._draw_frame(0)

    @staticmethod
    def _add_entry(parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(5, 2))
        ttk.Entry(parent, textvariable=variable, width=32).grid(row=row + 1, column=0, sticky="ew")
        return row + 2

    @staticmethod
    def _add_combo(parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, values: list[str]) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(5, 2))
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=30).grid(
            row=row + 1, column=0, sticky="ew"
        )
        return row + 2

    @staticmethod
    def _add_slider(
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.DoubleVar,
        lower: float,
        upper: float,
        *,
        formatter,
    ) -> int:
        header = ttk.Frame(parent)
        header.grid(row=row, column=0, sticky="ew", pady=(5, 0))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=label).grid(row=0, column=0, sticky="w")
        value_var = tk.StringVar(value=formatter(float(variable.get())))
        ttk.Label(header, textvariable=value_var, width=8, anchor="e").grid(row=0, column=1, sticky="e")

        def update_value(raw: str) -> None:
            value = float(raw)
            value_var.set(formatter(value))

        slider = ttk.Scale(
            parent, from_=float(lower), to=float(upper), orient="horizontal",
            variable=variable, command=update_value,
        )
        slider.grid(row=row + 1, column=0, sticky="ew", pady=(0, 2))
        return row + 2

    def run_selected(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Simulation running", "Wait for the current race to finish.")
            return
        try:
            vehicle_label = self.vehicle_var.get()
            if vehicle_label not in DISPLAY_TO_VEHICLE:
                raise ValueError("Choose a valid vehicle model.")
            model_name = DISPLAY_TO_VEHICLE[vehicle_label]
            variant_label = self.variant_var.get()
            if variant_label not in DISPLAY_TO_VARIANT:
                raise ValueError("Choose a valid controller variant.")
            variant_value = DISPLAY_TO_VARIANT[variant_label]
            laps = int(self.laps_var.get())
            rollouts = int(self.rollouts_var.get())
            lbps_delta = float(self.lbps_delta_var.get())
            seed = int(self.seed_var.get())
            horizon = int(round(float(self.horizon_var.get())))
            temporal_noise = float(self.temporal_noise_var.get())
            sigma0_scale = float(self.sigma0_scale_var.get())
            v_max = float(self.vmax_var.get())
            hard_collision_clearance = float(self.hard_collision_clearance_var.get())
            if laps < 1:
                raise ValueError("Number of laps must be at least 1.")
            if laps > 100:
                raise ValueError("Number of laps must be 100 or less for the interactive viewer.")
            if rollouts < 32:
                raise ValueError("Rollouts per step must be at least 32.")
            if not (0.0 < lbps_delta < 1.0):
                raise ValueError("LBPS delta must be strictly between 0 and 1.")
            if not (10 <= horizon <= 100):
                raise ValueError("Horizon H must be between 10 and 100.")
            if not (0.0 <= temporal_noise < 1.0):
                raise ValueError("Temporal noise smoothing must be in [0, 1).")
            if sigma0_scale <= 0.0:
                raise ValueError("Sigma_0 covariance scale must be positive.")
            if v_max <= 0.0:
                raise ValueError("v_max must be positive.")
            if hard_collision_clearance < 0.0:
                raise ValueError("hard_collision_clearance must be nonnegative.")
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self._stop_animation()
        self.result = None
        self.frame_index = 0
        self.run_button.configure(state="disabled")
        self.play_button.configure(state="disabled", text="Play")
        self.restart_button.configure(state="disabled")
        self.frame_scale.configure(state="disabled")
        self.status_var.set(
            f"Running {vehicle_label} / {variant_label}: H={horizon}, temporal={temporal_noise:.2f}, "
            f"Sigma_0 scale={sigma0_scale:.2f}x, v_max={v_max:.1f} m/s, clearance={hard_collision_clearance:.3f} m. "
            "First run may take longer while Numba/iLQR kernels compile."
        )
        settings = dict(
            variant_value=variant_value,
            laps=laps,
            num_rollouts=rollouts,
            lbps_delta=lbps_delta,
            seed=seed,
            horizon=horizon,
            temporal_noise_smoothing=temporal_noise,
            sigma0_scale=sigma0_scale,
            v_max=v_max,
            hard_collision_clearance=hard_collision_clearance,
            model_name=model_name,
        )
        self.worker = threading.Thread(target=self._worker_run, kwargs=settings, daemon=True)
        self.worker.start()

    def _worker_run(self, **settings: object) -> None:
        try:
            result = run_race(**settings)
            self.worker_queue.put(("success", result))
        except Exception:
            self.worker_queue.put(("error", traceback.format_exc()))

    def _poll_worker(self) -> None:
        if self.closing:
            return
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "success":
                    self._on_ready(payload)
                else:
                    self._on_error(str(payload))
        except queue.Empty:
            pass
        if not self.closing:
            self.poll_after_id = self.root.after(100, self._poll_worker)

    def _on_ready(self, payload: object) -> None:
        self.result = payload if isinstance(payload, RaceResult) else None
        self.run_button.configure(state="normal")
        if self.result is None:
            self.status_var.set("Race returned no result.")
            return
        total_frames = max(0, len(self.result.states) - 1)
        self.frame_scale.configure(from_=0, to=total_frames, state="normal")
        self.play_button.configure(state="normal")
        self.restart_button.configure(state="normal")
        self.frame_index = 0
        self._set_slider(0)
        self._draw_frame(0)

        if self.result.completed_laps >= self.result.requested_laps:
            lap_text = ", ".join(f"{value:.2f}s" for value in self.result.lap_times)
            self.status_var.set(
                f"Finished {self.result.completed_laps}/{self.result.requested_laps} laps. "
                f"Race time {len(self.result.controls) * self.result.cfg.dt:.2f}s. "
                f"Lap crossing times: {lap_text}. Compute time {self.result.runtime_s:.2f}s."
            )
        elif self.result.off_track:
            self.status_var.set(
                f"DNF: left the 2 m road after {self.result.completed_laps}/{self.result.requested_laps} laps. "
                f"Simulated time {len(self.result.controls) * self.result.cfg.dt:.2f}s."
            )
        else:
            self.status_var.set(
                f"DNF: step limit reached after {self.result.completed_laps}/{self.result.requested_laps} laps."
            )


        if len(self.result.states) > 1:
            self.playing = True
            self.play_button.configure(text="Pause")
            self._schedule_next_frame()

    def _on_error(self, text: str) -> None:
        self.run_button.configure(state="normal")
        self.play_button.configure(state="disabled", text="Play")
        self.restart_button.configure(state="disabled")
        self.status_var.set("Race failed. See error dialog.")
        messagebox.showerror("Race failed", text)

    def toggle_play(self) -> None:
        if self.result is None or len(self.result.states) <= 1:
            return
        if self.playing:
            self._stop_animation()
            return
        if self.frame_index >= len(self.result.states) - 1:
            self.frame_index = 0
        self.playing = True
        self.play_button.configure(text="Pause")
        self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        if not self.playing or self.result is None:
            return
        try:
            playback = max(float(self.speed_var.get()), 0.1)
        except ValueError:
            playback = 1.0
        delay_ms = max(10, int(1000.0 * self.result.cfg.dt / playback))
        self.after_id = self.root.after(delay_ms, self._advance_frame)

    def _advance_frame(self) -> None:
        self.after_id = None
        if not self.playing or self.result is None:
            return
        if self.frame_index >= len(self.result.states) - 1:
            self._stop_animation()
            return
        self.frame_index += 1
        self._set_slider(self.frame_index)
        self._draw_frame(self.frame_index)
        self._schedule_next_frame()

    def _stop_animation(self) -> None:
        self.playing = False
        if hasattr(self, "play_button"):
            self.play_button.configure(text="Play")
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def restart_animation(self) -> None:
        self._stop_animation()
        self.frame_index = 0
        self._set_slider(0)
        self._draw_frame(0)

    def _set_slider(self, value: int) -> None:
        self.updating_slider = True
        self.frame_scale.set(value)
        self.updating_slider = False

    def _on_frame_slider(self, value: str) -> None:
        if self.updating_slider or self.result is None:
            return
        try:
            frame = int(round(float(value)))
        except ValueError:
            return
        frame = int(np.clip(frame, 0, len(self.result.states) - 1))
        self.frame_index = frame
        self._draw_frame(frame)

    @staticmethod
    def _transform_vehicle_points(
        points: np.ndarray, origin: tuple[float, float], angle: float
    ) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        rotation = np.asarray([[c, -s], [s, c]], dtype=float)
        return np.asarray(points, dtype=float) @ rotation.T + np.asarray(origin, dtype=float)

    def _init_plot_artists(self) -> None:
        self.ax.set_xlim(-0.8, TRACK_WIDTH + 0.8)
        self.ax.set_ylim(-0.8, TRACK_HEIGHT + 0.8)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.grid(True, alpha=0.15)
        outside_color = '0.92'
        self.ax.set_facecolor(outside_color)
        self.ax.add_patch(Polygon(
            self._outer, closed=True, facecolor='white', edgecolor='0.25', linewidth=2.0, zorder=1
        ))
        self.ax.add_patch(Polygon(
            self._inner, closed=True, facecolor=outside_color, edgecolor='0.25', linewidth=2.0, zorder=2
        ))
        start, _ = centerline_point_tangent(START_S)
        self.ax.plot(
            [start[0], start[0]], [start[1] - 1.0, start[1] + 1.0],
            color='0.15', linewidth=2.0, zorder=4,
        )
        self.executed_line, = self.ax.plot(
            [], [], color='#1f77b4', linewidth=2.4, zorder=5, label='Executed trajectory'
        )
        self.nominal_line, = self.ax.plot(
            [], [], color='#0066cc', linewidth=2.0, linestyle='--', alpha=0.95,
            zorder=6, label='Nominal'
        )
        self.prediction_line, = self.ax.plot(
            [], [], color='#ff7f0e', linewidth=2.2, linestyle='--', alpha=0.95,
            zorder=7, label='MPPI prediction'
        )
        self.vehicle_body = Polygon(
            np.zeros((4, 2)), closed=True, facecolor='#17becf', edgecolor='black',
            linewidth=1.1, alpha=0.92, zorder=12,
        )
        self.ax.add_patch(self.vehicle_body)
        self.vehicle_wheels = []
        for _ in range(4):
            wheel = Polygon(
                np.zeros((4, 2)), closed=True, facecolor='0.10', edgecolor='black',
                linewidth=0.6, zorder=13,
            )
            self.ax.add_patch(wheel)
            self.vehicle_wheels.append(wheel)
        self.velocity_arrow = FancyArrowPatch(
            (0.0, 0.0), (0.0, 0.0), arrowstyle='-|>', mutation_scale=10.0,
            linewidth=1.2, color='#1f77b4', zorder=14,
        )
        self.ax.add_patch(self.velocity_arrow)
        self.nose_line, = self.ax.plot([], [], color='black', linewidth=1.1, zorder=14)
        self.ax.legend(loc='upper center', ncol=4, fontsize=8)

    def _update_vehicle_artists(self, state: np.ndarray, cfg: object, model_name: str) -> None:
        if state.size < 7:
            raise ValueError('Vehicle state must contain [x, y, psi, vx, vy, r, delta].')
        x, y, heading, vx, vy, _, steering = map(float, state[:7])
        lf = float(cfg.front_axle_distance)
        lr = float(cfg.rear_axle_distance)
        body_length = float(cfg.vehicle_length)
        body_width = float(cfg.vehicle_width)
        body = np.asarray([
            [-0.5 * body_length, -0.5 * body_width],
            [0.5 * body_length, -0.5 * body_width],
            [0.5 * body_length, 0.5 * body_width],
            [-0.5 * body_length, 0.5 * body_width],
        ])
        self.vehicle_body.set_facecolor(four_wheel.BODY_COLOR if model_name == 'four_wheel' else ackermann.BODY_COLOR)
        self.vehicle_body.set_xy(self._transform_vehicle_points(body, (x, y), heading))
        wheel_shape = np.asarray([
            [-0.11, -0.0275], [0.11, -0.0275], [0.11, 0.0275], [-0.11, 0.0275]
        ])
        half_track = 0.5 * float(getattr(cfg, 'track_width', 0.86 * body_width))
        c = math.cos(heading)
        s = math.sin(heading)

        def body_to_world(longitudinal: float, lateral: float) -> tuple[float, float]:
            return (
                x + longitudinal * c - lateral * s,
                y + longitudinal * s + lateral * c,
            )

        wheel_specs = (
            (lf, half_track, heading + steering),
            (lf, -half_track, heading + steering),
            (-lr, half_track, heading),
            (-lr, -half_track, heading),
        )
        for wheel_index, (artist, (longitudinal, lateral, wheel_heading)) in enumerate(zip(self.vehicle_wheels, wheel_specs)):
            center = body_to_world(longitudinal, lateral)
            artist.set_xy(self._transform_vehicle_points(wheel_shape, center, wheel_heading))
            if model_name == 'four_wheel' and state.size >= 13:
                normalized_spin = min(1.0, abs(float(state[9 + wheel_index])) / max(float(cfg.wheel_speed_limit), 1e-9))
                artist.set_linewidth(0.6 + 0.8 * normalized_spin)
            else:
                artist.set_linewidth(0.6)
        world_vx = vx * c - vy * s
        world_vy = vx * s + vy * c
        if math.hypot(world_vx, world_vy) > 0.001:
            self.velocity_arrow.set_positions((x, y), (x + 0.22 * world_vx, y + 0.22 * world_vy))
            self.velocity_arrow.set_visible(True)
        else:
            self.velocity_arrow.set_visible(False)
        nose = body_to_world(0.52 * body_length, 0.0)
        self.nose_line.set_data([x, nose[0]], [y, nose[1]])

    def _draw_frame(self, frame: int) -> None:
        if self.result is None:
            self.executed_line.set_data([], [])
            self.nominal_line.set_data([], [])
            self.prediction_line.set_data([], [])
            model_name = DISPLAY_TO_VEHICLE.get(self.vehicle_var.get(), 'ackermann')
            model = controller_core.resolve_vehicle_model(model_name)
            self._update_vehicle_artists(initial_race_state(model_name), model.MPPIConfig(), model_name)
            self.ax.set_title('No-planner MPPI racing - choose a variant, laps, and run')
            self.frame_label_var.set('Frame 0 / 0')
            self.canvas.draw_idle()
            return

        frame = int(np.clip(frame, 0, len(self.result.states) - 1))
        state = self.result.states[frame]
        path = self.result.states[:frame + 1, :2]
        recent = path[-100:]
        self.executed_line.set_data(recent[:, 0], recent[:, 1])
        pred_idx = min(frame, max(0, len(self.result.mppi_predictions) - 1))
        if self.result.mppi_predictions and pred_idx < len(self.result.mppi_predictions):
            pred = self.result.mppi_predictions[pred_idx]
            self.prediction_line.set_data(pred[:, 0], pred[:, 1])
            self.prediction_line.set_visible(True)
        else:
            self.prediction_line.set_visible(False)
        if self.result.nominal_predictions and pred_idx < len(self.result.nominal_predictions):
            nominal = self.result.nominal_predictions[pred_idx]
            self.nominal_line.set_data(nominal[:, 0], nominal[:, 1])
            self.nominal_line.set_visible(True)
        else:
            self.nominal_line.set_visible(False)
        self._update_vehicle_artists(state, self.result.cfg, self.result.model_name)

        speed = math.hypot(float(state[3]), float(state[4]))
        progress = float(self.result.cumulative_progress[frame])
        laps_progress = max(progress, 0.0) / TRACK_LENGTH
        completed = min(int(math.floor(laps_progress + 1e-12)), self.result.requested_laps)
        sim_time = min(frame, len(self.result.controls)) * float(self.result.cfg.dt)
        variant_label = VARIANT_TO_DISPLAY.get(self.result.variant_value, self.result.variant_value)
        vehicle_label = VEHICLE_TO_DISPLAY.get(self.result.model_name, self.result.model_name)
        title = (
            f'{vehicle_label} / {variant_label}  |  Lap {min(completed + 1, self.result.requested_laps)}/{self.result.requested_laps}  |  '
            f'progress {laps_progress:.2f} laps  |  speed {speed:.2f} m/s  |  t={sim_time:.1f}s'
        )
        if self.result.model_name == 'four_wheel' and state.size >= 13:
            title += f'  |  roll={math.degrees(float(state[7])):.1f} deg'
        if frame > 0 and frame - 1 < len(self.result.temperatures):
            lam = self.result.temperatures[frame - 1]
            ess = self.result.esses[frame - 1]
            feasible = int(self.result.feasible_counts[frame - 1])
            denominator = 1 if self.result.variant_value == 'planner_ilqr' else self.result.cfg.num_rollouts
            if math.isfinite(float(lam)):
                title += f'  |  lambda={lam:.3g}, ESS={ess:.1f}, feasible={feasible}/{denominator}'
            else:
                title += f'  |  feasible={feasible}/{denominator}'
        self.ax.set_title(title)
        self.frame_label_var.set(f'Frame {frame} / {len(self.result.states) - 1}')
        self.canvas.draw_idle()

    def _on_close(self) -> None:
        self.closing = True
        self._stop_animation()
        if self.poll_after_id is not None:
            try:
                self.root.after_cancel(self.poll_after_id)
            except tk.TclError:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    NASCARViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
