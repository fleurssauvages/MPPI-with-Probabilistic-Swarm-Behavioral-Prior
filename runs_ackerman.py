from __future__ import annotations

import csv
import math
import pickle
import time
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

try:
    import pandas as pd
except Exception:
    pd = None


try:
    from numba import njit
except Exception:
    njit = None


try:
    from geometry.utils import round_obstacle, PolyObstacle, obstacles_to_segs
    from planner.env import FishGoalEnv2D
    from graph.graph import build_full_graph
    from planner.planner import HomotopyAwareGenerativePlanner, trajectory_cost
except Exception as exc:
    raise ImportError(
        "Could not import your project modules. Run from the root of your project, "
        "where geometry/, RL/, graph/, planner.py, and save/ exist.\n"
        f"Original import error: {exc}"
    )


Array = np.ndarray


RUN_SEEDS = list(range(100))
RUN_SWARM_SEED = 5
OUTPUT_PREFIX = "dynamic_block_soft"


class ControllerVariant(str, Enum):
    GAUSSIAN_PRIOR_MPPI = "gaussian_prior_mppi"


    CORRIDOR_PRIOR_MPPI = "corridor_prior_mppi"
    CONTROL_BANK_MPPI = "control_bank_mppi"

    STANDARD_MPPI = "standard_mppi"
    STANDARD_MPPI_128 = "standard_mppi_128_rollouts"


@dataclass
class MPPIConfig:
    dt: float = 0.12
    horizon: int = 50
    num_rollouts: int = 32
    lambda_temperature: float = 2.2


    wheelbase: float = 0.55
    rear_axle_distance: float = 0.275
    front_axle_distance: float = 0.275
    mass: float = 18.0
    yaw_inertia: float = 1.20
    cornering_stiffness_front: float = 85.0
    cornering_stiffness_rear: float = 95.0
    tire_friction_coefficient: float = 0.95
    gravity: float = 9.81
    aerodynamic_drag_coefficient: float = 0.12
    rolling_resistance_force: float = 0.70
    minimum_tire_speed: float = 0.40
    dynamics_substeps: int = 4
    lateral_velocity_limit: float = 3.0
    yaw_rate_limit: float = 7.0

    v_min: float = 0.0
    v_max: float = 2.8
    accel_min: float = -3.5
    accel_max: float = 2.5

    steering_min: float = -1.20
    steering_max: float = 1.20
    steering_rate_min: float = -7.50
    steering_rate_max: float = 7.50


    noise_accel: float = 0.80
    noise_steering_rate: float = 0.70
    temporal_noise_smoothing: float = 0.72

    gaussian_covariance_scale: float = 2.0

    swarm_init_probability: float = 0.60
    max_empirical_nominals_per_mode: int = 16

    # Oriented rectangular collision footprint. These dimensions match
    # the Ackermann body drawn by the viewer: wheelbase + 0.26 by 0.36 m.
    vehicle_length: float = 0.81
    vehicle_width: float = 0.36
    # Retained for backward compatibility with existing callers/plots.
    robot_radius: float = 0.18
    base_safety_margin: float = 0.0
    uncertainty_margin_gain: float = 0.25


    collision_substeps: int = 5
    hard_collision_clearance: float = 0.01
    hard_collision_penalty: float = 800_000.0


    suppress_blocked_modes: bool = True
    mode_blocking_clearance: float = 0.02
    mode_blocking_substeps: int = 2


    w_goal: float = 110.0
    w_time_to_goal: float = 0.0
    rollout_goal_tolerance: float = 0.30
    w_obstacle: float = 500.0
    w_boundary: float = 500.0
    boundary_xmin: float = 0.0
    boundary_xmax: float = 10.0
    boundary_ymin: float = 0.0
    boundary_ymax: float = 10.0
    w_control: float = 0.004
    w_control_smooth: float = 0.40


    w_steering_angle: float = 0.0
    w_yaw_rate: float = 0.0
    w_heading: float = 0.0
    w_mode_prior: float = 0.25
    sigma_floor: float = 0.25


    w_reference_tracking: float = 1.20


    smooth_accel_weight: float = 0.5
    smooth_steering_rate_weight: float = 2.0

    max_precision: float = 10.0


    use_monotonic_reference_progress: bool = True
    max_reference_index_advance: int = 4


    apply_control_lowpass: bool = False
    control_lowpass_alpha: float = 0.0
    max_delta_accel: float = 1.20
    max_delta_steering_rate: float = 4.00
    enforce_one_step_safety: bool = True
    one_step_safety_clearance: float = 0.0


    low_noise_proposal_count: int = 0
    low_noise_proposal_scale: float = 0.15
    min_curve_speed: float = 0.16

    prior_screen_rollouts_per_mode: int = 16
    max_nearby_prior_modes: int = 3
    nearby_prior_distance_slack: float = 0.75
    nearby_prior_blocked_penalty: float = 1.25
    goal_acceptance_epsilon: float = 0.005

    def __post_init__(self) -> None:
        axle_sum = float(self.front_axle_distance + self.rear_axle_distance)
        if axle_sum <= 0.0:
            raise ValueError("Ackermann axle distances must sum to a positive wheelbase.")
        self.wheelbase = axle_sum
        if self.mass <= 0.0:
            raise ValueError("mass must be positive.")
        if self.yaw_inertia <= 0.0:
            raise ValueError("yaw_inertia must be positive.")
        if self.cornering_stiffness_front <= 0.0 or self.cornering_stiffness_rear <= 0.0:
            raise ValueError("cornering stiffnesses must be positive.")
        if self.tire_friction_coefficient <= 0.0:
            raise ValueError("tire_friction_coefficient must be positive.")
        if self.minimum_tire_speed <= 0.0:
            raise ValueError("minimum_tire_speed must be positive.")
        if self.vehicle_length <= 0.0 or self.vehicle_width <= 0.0:
            raise ValueError("vehicle footprint dimensions must be positive.")
        self.dynamics_substeps = max(1, int(self.dynamics_substeps))
        if self.v_min > self.v_max:
            raise ValueError("v_min must not exceed v_max.")
        if self.accel_min > self.accel_max:
            raise ValueError("accel_min must not exceed accel_max.")
        if self.steering_min > self.steering_max:
            raise ValueError("steering_min must not exceed steering_max.")
        if self.steering_rate_min > self.steering_rate_max:
            raise ValueError("steering_rate_min must not exceed steering_rate_max.")


def resample_path(path: Array, K: int) -> Array:
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"path must have shape (N,2), got {p.shape}")
    if p.shape[0] == 1:
        return np.repeat(p, K, axis=0)

    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.zeros(p.shape[0], dtype=np.float64)
    s[1:] = np.cumsum(d)
    if s[-1] <= 1e-12:
        return np.repeat(p[:1], K, axis=0)

    q = np.linspace(0.0, s[-1], K)
    return np.column_stack([
        np.interp(q, s, p[:, 0]),
        np.interp(q, s, p[:, 1]),
    ])


def snap_path_end_to_goal(
    path: Array,
    goal: Optional[Array],
    snap_radius: float = 0.2,
    straight_tail_points: int = 8,
) -> Array:


    p = np.asarray(path, dtype=np.float64)
    if goal is None:
        return p
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        return p

    g = np.asarray(goal, dtype=np.float64).reshape(2)
    d = np.linalg.norm(p - g[None, :], axis=1)
    inside = np.where(d <= float(snap_radius))[0]
    if len(inside) == 0:
        return p

    entry = int(inside[0])
    entry = min(entry, len(p) - 1)
    anchor = p[entry].copy()

    n_tail = max(2, int(straight_tail_points))
    tail = np.linspace(anchor, g, n_tail)

    if entry == 0:
        snapped = tail
    else:
        snapped = np.vstack([p[:entry], tail])


    keep = [0]
    for i in range(1, len(snapped)):
        if np.linalg.norm(snapped[i] - snapped[keep[-1]]) > 1e-10:
            keep.append(i)
    return snapped[keep]


def flatten_path(path_K: Array) -> Array:
    return np.asarray(path_K, dtype=np.float64).reshape(-1)


def unflatten_path(vec: Array) -> Array:
    return np.asarray(vec, dtype=np.float64).reshape(-1, 2)


def stable_softmax_from_cost(costs: Array, beta: float) -> Array:
    c = np.asarray(costs, dtype=np.float64)
    if c.size == 0:
        return c
    z = -float(beta) * (c - np.nanmin(c))
    z = np.clip(z, -80.0, 80.0)
    w = np.exp(z)
    s = np.sum(w)
    if s <= 1e-12:
        return np.ones_like(w) / len(w)
    return w / s


@dataclass
class GaussianTrajectoryMode:
    signature: Tuple[int, ...]
    probability: float
    mean: Array
    cov: Array
    samples: Array
    weights: Array
    mean_cost: float
    count: int

    @property
    def mean_path(self) -> Array:
        return unflatten_path(self.mean)


@dataclass
class TopologicalTrajectoryMixture:
    modes: Dict[Tuple[int, ...], GaussianTrajectoryMode]
    K: int
    beta: float


@dataclass
class MPPIHomotopyMode:
    signature: Tuple[int, ...]
    probability: float
    mean_path: Array
    cov_blocks: Array
    sample_paths: Optional[List[Array]] = None
    arc_length: Optional[Array] = None
    gaussian_variance: Optional[Array] = None


def prepare_mode_prior_cache(mode: MPPIHomotopyMode) -> MPPIHomotopyMode:
    """Precompute arc length and isotropic covariance once for a global mode."""
    mean_path = np.asarray(mode.mean_path, dtype=np.float64)
    cov_blocks = np.asarray(mode.cov_blocks, dtype=np.float64)
    count = len(mean_path)

    arc_length = np.zeros(count, dtype=np.float64)
    if count > 1:
        arc_length[1:] = np.cumsum(np.linalg.norm(np.diff(mean_path, axis=0), axis=1))

    symmetric_cov = 0.5 * (cov_blocks + np.swapaxes(cov_blocks, 1, 2))
    gaussian_variance = 0.5 * np.trace(symmetric_cov, axis1=1, axis2=2)

    mode.arc_length = np.ascontiguousarray(arc_length)
    mode.gaussian_variance = np.ascontiguousarray(np.maximum(gaussian_variance, 0.0))
    return mode


def fit_topological_trajectory_mixture(
    gen_out,
    obstacles,
    *,
    K: int = 50,
    beta: float = 1.0,
    min_mode_samples: int = 3,
    covariance_jitter: float = 2e-4,
    costmap=None,
    bounds=((0.0, 10.0), (0.0, 10.0)),
    goal: Optional[Array] = None,
    snap_to_goal_radius: float = 0.2,
    snap_straight_tail_points: int = 8,
) -> TopologicalTrajectoryMixture:
    raw_paths = list(gen_out.samples)
    if len(raw_paths) == 0:
        raise RuntimeError("Swarm planner produced zero trajectory samples.")


    all_paths = [
        snap_path_end_to_goal(
            p,
            goal=goal,
            snap_radius=snap_to_goal_radius,
            straight_tail_points=snap_straight_tail_points,
        )
        for p in raw_paths
    ]

    all_costs = np.array([
        trajectory_cost(p, costmap=costmap, bounds=bounds, w_len=1.0, w_smooth=0.05)
        for p in all_paths
    ], dtype=np.float64)

    all_weights = stable_softmax_from_cost(all_costs, beta=beta)

    snapped_by_raw_id = {id(raw): snapped for raw, snapped in zip(raw_paths, all_paths)}
    weight_by_raw_id = {id(raw): float(w) for raw, w in zip(raw_paths, all_weights)}
    cost_by_raw_id = {id(raw): float(c) for raw, c in zip(raw_paths, all_costs)}

    mode_raw = {}
    total_mode_weight = 0.0

    for sig, paths in gen_out.homotopy_groups.items():
        if len(paths) < min_mode_samples:
            continue

        snapped_paths = [snapped_by_raw_id.get(id(p), p) for p in paths]

        X = np.stack([flatten_path(resample_path(p, K)) for p in snapped_paths], axis=0)
        w = np.array([weight_by_raw_id.get(id(p), 1.0) for p in paths], dtype=np.float64)
        c = np.array([cost_by_raw_id.get(id(p), np.nan) for p in paths], dtype=np.float64)

        if np.sum(w) <= 1e-12:
            w = np.ones(len(paths), dtype=np.float64) / len(paths)
        else:
            w = w / np.sum(w)

        mu = np.sum(X * w[:, None], axis=0)
        Xc = X - mu[None, :]
        cov = (Xc * w[:, None]).T @ Xc
        cov = 0.5 * (cov + cov.T) + covariance_jitter * np.eye(cov.shape[0])

        mode_weight = float(np.sum([weight_by_raw_id.get(id(p), 0.0) for p in paths]))
        total_mode_weight += mode_weight

        mode_raw[sig] = dict(
            X=X,
            w=w,
            mu=mu,
            cov=cov,
            mode_weight=mode_weight,
            mean_cost=float(np.nanmean(c)),
        )

    if not mode_raw:
        raise RuntimeError("No homotopy group had enough samples.")

    if total_mode_weight <= 1e-12:
        total_mode_weight = float(len(mode_raw))
        for sig in mode_raw:
            mode_raw[sig]["mode_weight"] = 1.0

    modes = {}
    for sig, d in mode_raw.items():
        modes[sig] = GaussianTrajectoryMode(
            signature=sig,
            probability=float(d["mode_weight"] / total_mode_weight),
            mean=d["mu"],
            cov=d["cov"],
            samples=d["X"],
            weights=d["w"],
            mean_cost=d["mean_cost"],
            count=int(d["X"].shape[0]),
        )

    return TopologicalTrajectoryMixture(modes=modes, K=K, beta=beta)


def mixture_to_mppi_modes(mixture: TopologicalTrajectoryMixture) -> List[MPPIHomotopyMode]:
    modes = []
    for sig, mode in mixture.modes.items():
        mean_path = mode.mean_path
        K = mean_path.shape[0]
        cov_blocks = np.zeros((K, 2, 2), dtype=np.float64)
        for t in range(K):
            cov_blocks[t] = mode.cov[2*t:2*t+2, 2*t:2*t+2]

        sample_paths = [unflatten_path(v) for v in mode.samples]

        modes.append(prepare_mode_prior_cache(MPPIHomotopyMode(
            signature=sig,
            probability=mode.probability,
            mean_path=mean_path,
            cov_blocks=cov_blocks,
            sample_paths=sample_paths,
        )))

    modes.sort(key=lambda m: m.probability, reverse=True)
    return modes


def _poly_vertices(obs) -> Array:
    if hasattr(obs, "vertices"):
        return np.asarray(obs.vertices, dtype=np.float64)[:, :2]
    return np.asarray(obs, dtype=np.float64)[:, :2]


def normalize_plot_bounds(bounds):
    b0 = np.asarray(bounds[0], dtype=np.float64)
    b1 = np.asarray(bounds[1], dtype=np.float64)
    if b0.shape == (2,) and b1.shape == (2,) and b0[0] <= b1[0] and b0[1] <= b1[1]:
        xmin, ymin = b0
        xmax, ymax = b1
        return float(xmin), float(xmax), float(ymin), float(ymax)

    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    return float(xmin), float(xmax), float(ymin), float(ymax)


def point_segment_distance_and_normal(p: Array, a: Array, b: Array) -> Tuple[float, Array]:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        closest = a
    else:
        u = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
        closest = a + u * ab
    dvec = p - closest
    dist = float(np.linalg.norm(dvec))
    normal = np.array([1.0, 0.0]) if dist <= 1e-12 else dvec / dist
    return dist, normal


def point_in_poly(p: Array, poly: Array) -> bool:
    x, y = p
    inside = False
    n = poly.shape[0]
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i + 1) % n]
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / ((yj - yi) + 1e-18)
            if x < x_cross:
                inside = not inside
    return inside


def polygon_signed_distance_and_normal(p: Array, obs) -> Tuple[float, Array]:
    poly = _poly_vertices(obs)
    best_dist = float("inf")
    best_normal = np.array([1.0, 0.0])
    for i in range(poly.shape[0]):
        d, n = point_segment_distance_and_normal(p, poly[i], poly[(i + 1) % poly.shape[0]])
        if d < best_dist:
            best_dist = d
            best_normal = n

    if point_in_poly(p, poly):
        return -best_dist, best_normal
    return best_dist, best_normal


def obstacle_bounding_circles(
    obstacles: Sequence,
    *,
    elongated_aspect_ratio: float = 2.25,
    max_segment_length: float = 0.10,
    wall_max_segment_length: float = 0.15,
) -> List[Tuple[Array, float]]:


    circles: List[Tuple[Array, float]] = []

    for obs in obstacles:
        poly = _poly_vertices(obs)
        center = poly.mean(axis=0)

        if len(poly) < 4:
            radius = float(np.max(np.linalg.norm(poly - center[None, :], axis=1)))
            circles.append((center, radius))
            continue

        centered = poly - center[None, :]
        covariance = centered.T @ centered / max(1, len(poly))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float64)

        major_coord = centered @ major_axis
        minor_coord = centered @ minor_axis
        major_min = float(np.min(major_coord))
        major_max = float(np.max(major_coord))
        minor_min = float(np.min(minor_coord))
        minor_max = float(np.max(minor_coord))

        length = max(major_max - major_min, 1e-12)
        width = max(minor_max - minor_min, 1e-12)
        aspect_ratio = length / width

        if aspect_ratio < elongated_aspect_ratio:
            radius = float(np.max(np.linalg.norm(centered, axis=1)))
            circles.append((center, radius))
            continue


        target_segment_length = (
            wall_max_segment_length if len(poly) == 4 else max_segment_length
        )
        segment_count = max(2, int(math.ceil(length / target_segment_length)))
        segment_length = length / segment_count
        circle_radius = math.sqrt((0.5 * segment_length) ** 2 + (0.5 * width) ** 2)
        minor_mid = 0.5 * (minor_min + minor_max)

        for index in range(segment_count):
            major_mid = major_min + (index + 0.5) * segment_length
            circle_center = center + major_mid * major_axis + minor_mid * minor_axis
            circles.append((circle_center.astype(np.float64), float(circle_radius)))

    return circles


def ackermann_rectangle_corners(
    state: Array,
    vehicle_length: float,
    vehicle_width: float,
) -> Array:
    """Return the four world-frame corners of the centered Ackermann body."""
    x, y, heading = map(float, np.asarray(state, dtype=np.float64)[:3])
    half_length = 0.5 * float(vehicle_length)
    half_width = 0.5 * float(vehicle_width)
    local = np.array([
        [-half_length, -half_width],
        [ half_length, -half_width],
        [ half_length,  half_width],
        [-half_length,  half_width],
    ], dtype=np.float64)
    c = math.cos(heading)
    s = math.sin(heading)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + np.array([x, y], dtype=np.float64)


def rectangle_circle_clearance(
    state: Array,
    circle_center: Array,
    circle_radius: float,
    vehicle_length: float,
    vehicle_width: float,
) -> float:
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


def minimum_rectangle_circle_clearance(
    state: Array,
    centers: Array,
    radii: Array,
    vehicle_length: float,
    vehicle_width: float,
) -> float:
    state_array = np.asarray(state, dtype=np.float64)
    center_array = np.asarray(centers, dtype=np.float64)
    radius_array = np.asarray(radii, dtype=np.float64)
    if radius_array.size == 0:
        return float("inf")
    if minimum_rectangle_circle_clearance_nb is not None:
        return float(minimum_rectangle_circle_clearance_nb(
            state_array, center_array, radius_array,
            float(vehicle_length), float(vehicle_width),
        ))
    return float(min(
        rectangle_circle_clearance(
            state_array, center_array[j], radius_array[j],
            vehicle_length, vehicle_width,
        )
        for j in range(len(radius_array))
    ))


def _point_in_oriented_rectangle(
    point: Array,
    state: Array,
    vehicle_length: float,
    vehicle_width: float,
) -> bool:
    x, y, heading = map(float, np.asarray(state, dtype=np.float64)[:3])
    dx = float(point[0] - x)
    dy = float(point[1] - y)
    c = math.cos(heading)
    s = math.sin(heading)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return bool(
        abs(local_x) <= 0.5 * float(vehicle_length) + 1e-12
        and abs(local_y) <= 0.5 * float(vehicle_width) + 1e-12
    )


def _orientation_2d(a: Array, b: Array, c: Array) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _point_on_segment(point: Array, a: Array, b: Array, eps: float = 1e-12) -> bool:
    if abs(_orientation_2d(a, b, point)) > eps:
        return False
    return bool(
        min(a[0], b[0]) - eps <= point[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= point[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(a: Array, b: Array, c: Array, d: Array) -> bool:
    o1 = _orientation_2d(a, b, c)
    o2 = _orientation_2d(a, b, d)
    o3 = _orientation_2d(c, d, a)
    o4 = _orientation_2d(c, d, b)
    if ((o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0)) and (
        (o3 > 0.0 and o4 < 0.0) or (o3 < 0.0 and o4 > 0.0)
    ):
        return True
    return bool(
        (abs(o1) <= 1e-12 and _point_on_segment(c, a, b))
        or (abs(o2) <= 1e-12 and _point_on_segment(d, a, b))
        or (abs(o3) <= 1e-12 and _point_on_segment(a, c, d))
        or (abs(o4) <= 1e-12 and _point_on_segment(b, c, d))
    )


def _point_segment_distance(point: Array, a: Array, b: Array) -> float:
    ab = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    denom = float(ab @ ab)
    if denom <= 1e-16:
        return float(np.linalg.norm(np.asarray(point) - np.asarray(a)))
    t = float(np.clip(((np.asarray(point) - np.asarray(a)) @ ab) / denom, 0.0, 1.0))
    closest = np.asarray(a) + t * ab
    return float(np.linalg.norm(np.asarray(point) - closest))


def _segment_segment_distance(a: Array, b: Array, c: Array, d: Array) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def rectangle_polygon_clearance(
    state: Array,
    obstacle,
    vehicle_length: float,
    vehicle_width: float,
) -> float:
    """Signed rectangle--polygon clearance; negative means overlap."""
    rectangle = ackermann_rectangle_corners(state, vehicle_length, vehicle_width)
    polygon = _poly_vertices(obstacle)

    if any(point_in_poly(corner, polygon) for corner in rectangle):
        return -1e-9
    if any(
        _point_in_oriented_rectangle(vertex, state, vehicle_length, vehicle_width)
        for vertex in polygon
    ):
        return -1e-9

    minimum = float("inf")
    for i in range(len(rectangle)):
        a = rectangle[i]
        b = rectangle[(i + 1) % len(rectangle)]
        for j in range(len(polygon)):
            c = polygon[j]
            d = polygon[(j + 1) % len(polygon)]
            distance = _segment_segment_distance(a, b, c, d)
            if distance <= 1e-12:
                return -1e-9
            minimum = min(minimum, distance)
    return minimum


def min_clearance(
    states: Array,
    obstacles: Sequence,
    robot_radius: float = 0.18,
    vehicle_length: float = 0.81,
    vehicle_width: float = 0.36,
) -> float:
    """Minimum signed clearance using the oriented rectangular body."""
    del robot_radius  # Kept in the signature for compatibility with older callers.
    state_array = np.asarray(states, dtype=np.float64)
    if state_array.size == 0 or not obstacles:
        return float("inf")
    if min_clearance_nb is not None:
        padded, lengths = obstacles_to_padded_arrays(obstacles)
        return float(min_clearance_nb(
            state_array, padded, lengths,
            float(vehicle_length), float(vehicle_width),
        ))
    values = [
        rectangle_polygon_clearance(
            state, obstacle, vehicle_length, vehicle_width
        )
        for state in state_array
        for obstacle in obstacles
    ]
    return float(np.min(values)) if values else float("inf")


def path_collided(
    states: Array,
    obstacles: Sequence,
    robot_radius: float = 0.18,
    vehicle_length: float = 0.81,
    vehicle_width: float = 0.36,
) -> bool:
    return min_clearance(
        states,
        obstacles,
        robot_radius,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
    ) < 0.0


def path_length(states: Array) -> float:
    p = states[:, :2]
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def control_effort(controls: Array) -> float:

    if len(controls) == 0:
        return 0.0
    return float(np.sum(controls[:, 0] ** 2 + 0.15 * controls[:, 1] ** 2))


def control_smoothness(controls: Array) -> float:

    if len(controls) < 2:
        return 0.0
    dU = np.diff(controls, axis=0)
    return float(np.sum(dU[:, 0] ** 2 + 0.2 * dU[:, 1] ** 2))


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _dynamic_ackermann_derivatives(
    state: Array,
    control: Array,
    cfg: MPPIConfig,
) -> Array:


    px, py, psi, vx, vy, yaw_rate, steering = np.asarray(state, dtype=np.float64)
    accel, steering_rate = np.asarray(control, dtype=np.float64)

    accel = float(np.clip(accel, cfg.accel_min, cfg.accel_max))
    steering_rate = float(np.clip(
        steering_rate, cfg.steering_rate_min, cfg.steering_rate_max
    ))

    lf = float(cfg.front_axle_distance)
    lr = float(cfg.rear_axle_distance)
    wheelbase = max(lf + lr, 1e-9)
    mass = float(cfg.mass)
    inertia = float(cfg.yaw_inertia)


    slip_speed = max(abs(float(vx)), float(cfg.minimum_tire_speed))
    speed_scale = min(1.0, abs(float(vx)) / float(cfg.minimum_tire_speed))
    alpha_front = math.atan2(vy + lf * yaw_rate, slip_speed) - steering
    alpha_rear = math.atan2(vy - lr * yaw_rate, slip_speed)

    force_y_front = -float(cfg.cornering_stiffness_front) * alpha_front * speed_scale
    force_y_rear = -float(cfg.cornering_stiffness_rear) * alpha_rear * speed_scale

    normal_front = mass * float(cfg.gravity) * lr / wheelbase
    normal_rear = mass * float(cfg.gravity) * lf / wheelbase
    front_limit = float(cfg.tire_friction_coefficient) * normal_front
    rear_limit = float(cfg.tire_friction_coefficient) * normal_rear
    force_y_front = float(np.clip(force_y_front, -front_limit, front_limit))
    force_y_rear = float(np.clip(force_y_rear, -rear_limit, rear_limit))

    drag_force = float(cfg.aerodynamic_drag_coefficient) * vx * abs(vx)
    rolling_force = float(cfg.rolling_resistance_force) * math.tanh(vx / 0.10)
    force_x = mass * accel - drag_force - rolling_force

    cos_psi = math.cos(psi)
    sin_psi = math.sin(psi)
    cos_delta = math.cos(steering)
    sin_delta = math.sin(steering)

    return np.array([
        vx * cos_psi - vy * sin_psi,
        vx * sin_psi + vy * cos_psi,
        yaw_rate,
        (force_x - force_y_front * sin_delta) / mass + yaw_rate * vy,
        (force_y_front * cos_delta + force_y_rear) / mass - yaw_rate * vx,
        (
            lf * force_y_front * cos_delta
            - lr * force_y_rear
        ) / inertia,
        steering_rate,
    ], dtype=np.float64)


def ackermann_step(x: Array, u: Array, cfg: MPPIConfig) -> Array:


    state = np.asarray(x, dtype=np.float64).copy()
    control = np.asarray(u, dtype=np.float64)
    substeps = max(1, int(cfg.dynamics_substeps))
    h = float(cfg.dt) / substeps

    for _ in range(substeps):
        derivative = _dynamic_ackermann_derivatives(state, control, cfg)
        state += h * derivative
        state[2] = wrap_angle(state[2])
        state[3] = np.clip(state[3], cfg.v_min, cfg.v_max)
        state[4] = np.clip(
            state[4], -cfg.lateral_velocity_limit, cfg.lateral_velocity_limit
        )
        state[5] = np.clip(state[5], -cfg.yaw_rate_limit, cfg.yaw_rate_limit)
        state[6] = np.clip(state[6], cfg.steering_min, cfg.steering_max)

    return state


def segment_goal_entry_state(
    x0: Array,
    x1: Array,
    goal: Array,
    goal_tolerance: float,
) -> Tuple[bool, Array]:

    p0 = np.asarray(x0[:2], dtype=np.float64)
    p1 = np.asarray(x1[:2], dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    r = float(goal_tolerance)
    if np.linalg.norm(p0 - g) <= r:
        return True, np.asarray(x0, dtype=np.float64).copy()
    if np.linalg.norm(p1 - g) <= r:
        return True, np.asarray(x1, dtype=np.float64).copy()
    d = p1 - p0
    qa = float(d @ d)
    if qa <= 1e-16:
        return False, np.asarray(x1, dtype=np.float64).copy()
    f = p0 - g
    qb = 2.0 * float(f @ d)
    qc = float(f @ f) - r * r
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return False, np.asarray(x1, dtype=np.float64).copy()
    root = math.sqrt(max(0.0, disc))
    roots = [
        q for q in ((-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa))
        if 0.0 <= q <= 1.0
    ]
    if not roots:
        return False, np.asarray(x1, dtype=np.float64).copy()
    alpha = float(min(roots))
    hit = np.asarray(x0, dtype=np.float64).copy()
    hit[:2] = p0 + alpha * d
    hit[2] = wrap_angle(
        float(x0[2]) + alpha * wrap_angle(float(x1[2]) - float(x0[2]))
    )
    hit[3:] = np.asarray(x0[3:], dtype=np.float64) + alpha * (
        np.asarray(x1[3:], dtype=np.float64) - np.asarray(x0[3:], dtype=np.float64)
    )
    return True, hit


def goal_pose_satisfied(
    state: Array,
    goal: Array,
    goal_tolerance: float,
    cfg: MPPIConfig,
) -> bool:


    _ = cfg
    return bool(
        np.linalg.norm(np.asarray(state[:2]) - np.asarray(goal))
        <= goal_tolerance
    )


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
    )


def rollout_ackermann(x0: Array, U: Array, cfg: MPPIConfig) -> Array:
    if rollout_ackermann_single_nb is not None:
        return rollout_ackermann_single_nb(
            np.asarray(x0, dtype=np.float64),
            np.asarray(U, dtype=np.float64),
            *_dynamic_model_arguments(cfg),
        )

    X = np.zeros((len(U) + 1, 7), dtype=np.float64)
    X[0] = x0
    for t, u in enumerate(U):
        X[t + 1] = ackermann_step(X[t], u, cfg)
    return X


def rollout_ackermann_batch(x0: Array, U: Array, cfg: MPPIConfig) -> Array:
    if rollout_ackermann_batch_nb is not None:
        return rollout_ackermann_batch_nb(
            np.asarray(x0, dtype=np.float64),
            np.asarray(U, dtype=np.float64),
            *_dynamic_model_arguments(cfg),
        )

    N, H, _ = U.shape
    X = np.zeros((N, H + 1, 7), dtype=np.float64)
    X[:, 0, :] = x0[None, :]
    for n in range(N):
        for t in range(H):
            X[n, t + 1] = ackermann_step(X[n, t], U[n, t], cfg)
    return X


def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _ensure_mode_prior_cache(mode: MPPIHomotopyMode) -> MPPIHomotopyMode:
    if mode.arc_length is None or mode.gaussian_variance is None:
        return prepare_mode_prior_cache(mode)
    return mode


if njit is not None:
    @njit(cache=True)
    def localize_prior_horizon_nb(
        mean_path, cov_blocks, arc_length, gaussian_variance, start_index, horizon,
    ):
        count = mean_path.shape[0]
        H = max(1, int(horizon))
        local_mean = np.empty((H, 2), dtype=np.float64)
        local_cov = np.empty((H, 2, 2), dtype=np.float64)
        local_gaussian = np.empty(H, dtype=np.float64)

        start = min(max(int(start_index), 0), max(0, count - 1))
        s0 = arc_length[start]
        s1 = arc_length[count - 1]

        cursor = start
        for t in range(H):
            alpha_h = 0.0 if H == 1 else t / (H - 1.0)
            target = s0 + alpha_h * (s1 - s0)
            while cursor + 1 < count and arc_length[cursor + 1] < target:
                cursor += 1
            right = min(cursor + 1, count - 1)
            left = cursor
            denom = arc_length[right] - arc_length[left]
            alpha = 0.0 if denom <= 1e-12 else (target - arc_length[left]) / denom
            beta = 1.0 - alpha

            for j in range(2):
                local_mean[t, j] = beta * mean_path[left, j] + alpha * mean_path[right, j]
                for k in range(2):
                    local_cov[t, j, k] = (
                        beta * cov_blocks[left, j, k]
                        + alpha * cov_blocks[right, j, k]
                    )
            local_gaussian[t] = beta * gaussian_variance[left] + alpha * gaussian_variance[right]

        return local_mean, local_cov, local_gaussian

    @njit(cache=True)
    def apply_gaussian_prior_noise_nb(noise, variance, sigma_floor, covariance_scale):
        floor_var = sigma_floor * sigma_floor
        reference_std = max(sigma_floor, 1e-9)
        for t in range(noise.shape[1]):
            var = max(variance[t], floor_var)
            scale = covariance_scale * math.sqrt(var) / reference_std
            for n in range(noise.shape[0]):
                noise[n, t, 0] *= scale
                noise[n, t, 1] *= scale
        return noise

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
    def _dynamic_ackermann_step_nb(
        state, accel_cmd, steering_rate_cmd,
        dt, front_axle_distance, rear_axle_distance,
        mass, yaw_inertia, cornering_stiffness_front,
        cornering_stiffness_rear, tire_friction_coefficient,
        gravity, aerodynamic_drag_coefficient,
        rolling_resistance_force, minimum_tire_speed,
        dynamics_substeps, v_min, v_max,
        lateral_velocity_limit, yaw_rate_limit,
        accel_min, accel_max, steering_min, steering_max,
        steering_rate_min, steering_rate_max,
    ):
        px = state[0]
        py = state[1]
        psi = state[2]
        vx = state[3]
        vy = state[4]
        yaw_rate = state[5]
        steering = state[6]

        accel = min(max(accel_cmd, accel_min), accel_max)
        steering_rate = min(max(
            steering_rate_cmd, steering_rate_min
        ), steering_rate_max)
        substeps = max(1, int(dynamics_substeps))
        h = dt / substeps
        wheelbase = max(front_axle_distance + rear_axle_distance, 1e-9)
        normal_front = mass * gravity * rear_axle_distance / wheelbase
        normal_rear = mass * gravity * front_axle_distance / wheelbase
        front_limit = tire_friction_coefficient * normal_front
        rear_limit = tire_friction_coefficient * normal_rear

        for _ in range(substeps):
            slip_speed = max(abs(vx), minimum_tire_speed)
            speed_scale = min(1.0, abs(vx) / minimum_tire_speed)
            alpha_front = math.atan2(
                vy + front_axle_distance * yaw_rate, slip_speed
            ) - steering
            alpha_rear = math.atan2(
                vy - rear_axle_distance * yaw_rate, slip_speed
            )
            force_y_front = -cornering_stiffness_front * alpha_front * speed_scale
            force_y_rear = -cornering_stiffness_rear * alpha_rear * speed_scale
            force_y_front = min(max(force_y_front, -front_limit), front_limit)
            force_y_rear = min(max(force_y_rear, -rear_limit), rear_limit)

            drag_force = aerodynamic_drag_coefficient * vx * abs(vx)
            rolling_force = rolling_resistance_force * math.tanh(vx / 0.10)
            force_x = mass * accel - drag_force - rolling_force
            cos_psi = math.cos(psi)
            sin_psi = math.sin(psi)
            cos_delta = math.cos(steering)
            sin_delta = math.sin(steering)

            px_dot = vx * cos_psi - vy * sin_psi
            py_dot = vx * sin_psi + vy * cos_psi
            psi_dot = yaw_rate
            vx_dot = (
                (force_x - force_y_front * sin_delta) / mass
                + yaw_rate * vy
            )
            vy_dot = (
                (force_y_front * cos_delta + force_y_rear) / mass
                - yaw_rate * vx
            )
            yaw_accel = (
                front_axle_distance * force_y_front * cos_delta
                - rear_axle_distance * force_y_rear
            ) / yaw_inertia

            px += h * px_dot
            py += h * py_dot
            psi = _wrap_angle_nb(psi + h * psi_dot)
            vx = min(max(vx + h * vx_dot, v_min), v_max)
            vy = min(max(
                vy + h * vy_dot, -lateral_velocity_limit
            ), lateral_velocity_limit)
            yaw_rate = min(max(
                yaw_rate + h * yaw_accel, -yaw_rate_limit
            ), yaw_rate_limit)
            steering = min(max(
                steering + h * steering_rate, steering_min
            ), steering_max)

        return px, py, psi, vx, vy, yaw_rate, steering

    @njit(cache=True)
    def rollout_ackermann_batch_nb(
        x0, U, dt, front_axle_distance, rear_axle_distance,
        mass, yaw_inertia, cornering_stiffness_front,
        cornering_stiffness_rear, tire_friction_coefficient,
        gravity, aerodynamic_drag_coefficient,
        rolling_resistance_force, minimum_tire_speed,
        dynamics_substeps, v_min, v_max,
        lateral_velocity_limit, yaw_rate_limit,
        accel_min, accel_max, steering_min, steering_max,
        steering_rate_min, steering_rate_max,
    ):
        N = U.shape[0]
        H = U.shape[1]
        X = np.zeros((N, H + 1, 7), dtype=np.float64)
        for n in range(N):
            for j in range(7):
                X[n, 0, j] = x0[j]
            for t in range(H):
                values = _dynamic_ackermann_step_nb(
                    X[n, t], U[n, t, 0], U[n, t, 1],
                    dt, front_axle_distance, rear_axle_distance,
                    mass, yaw_inertia, cornering_stiffness_front,
                    cornering_stiffness_rear, tire_friction_coefficient,
                    gravity, aerodynamic_drag_coefficient,
                    rolling_resistance_force, minimum_tire_speed,
                    dynamics_substeps, v_min, v_max,
                    lateral_velocity_limit, yaw_rate_limit,
                    accel_min, accel_max, steering_min, steering_max,
                    steering_rate_min, steering_rate_max,
                )
                for j in range(7):
                    X[n, t + 1, j] = values[j]
        return X

    @njit(cache=True)
    def rollout_ackermann_single_nb(
        x0, U, dt, front_axle_distance, rear_axle_distance,
        mass, yaw_inertia, cornering_stiffness_front,
        cornering_stiffness_rear, tire_friction_coefficient,
        gravity, aerodynamic_drag_coefficient,
        rolling_resistance_force, minimum_tire_speed,
        dynamics_substeps, v_min, v_max,
        lateral_velocity_limit, yaw_rate_limit,
        accel_min, accel_max, steering_min, steering_max,
        steering_rate_min, steering_rate_max,
    ):
        H = U.shape[0]
        X = np.zeros((H + 1, 7), dtype=np.float64)
        for j in range(7):
            X[0, j] = x0[j]
        for t in range(H):
            values = _dynamic_ackermann_step_nb(
                X[t], U[t, 0], U[t, 1],
                dt, front_axle_distance, rear_axle_distance,
                mass, yaw_inertia, cornering_stiffness_front,
                cornering_stiffness_rear, tire_friction_coefficient,
                gravity, aerodynamic_drag_coefficient,
                rolling_resistance_force, minimum_tire_speed,
                dynamics_substeps, v_min, v_max,
                lateral_velocity_limit, yaw_rate_limit,
                accel_min, accel_max, steering_min, steering_max,
                steering_rate_min, steering_rate_max,
            )
            for j in range(7):
                X[t + 1, j] = values[j]
        return X

    @njit(cache=True)
    def nominal_controls_to_track_path_nb(
        x0, ref, horizon,
        dt, front_axle_distance, rear_axle_distance,
        mass, yaw_inertia, cornering_stiffness_front,
        cornering_stiffness_rear, tire_friction_coefficient,
        gravity, aerodynamic_drag_coefficient,
        rolling_resistance_force, minimum_tire_speed,
        dynamics_substeps, v_min, v_max,
        lateral_velocity_limit, yaw_rate_limit,
        accel_min, accel_max, steering_min, steering_max,
        steering_rate_min, steering_rate_max,
    ):
        U = np.zeros((horizon, 2), dtype=np.float64)
        x = np.zeros(7, dtype=np.float64)
        for j in range(7):
            x[j] = x0[j]
        ref_len = ref.shape[0]
        wheelbase = front_axle_distance + rear_axle_distance

        for t in range(horizon):
            target_idx = min(t + 3, ref_len - 1)
            dx = ref[target_idx, 0] - x[0]
            dy = ref[target_idx, 1] - x[1]
            dist = math.sqrt(dx * dx + dy * dy)
            desired_heading = math.atan2(dy, dx)
            heading_error = _wrap_angle_nb(desired_heading - x[2])
            heading_scale = max(0.0, math.cos(heading_error)) ** 2
            desired_speed = min(max(
                0.20 + 2.4 * dist * heading_scale, 0.0
            ), v_max)
            accel = min(max(
                3.0 * (desired_speed - x[3]), accel_min
            ), accel_max)

            lookahead = max(dist, 0.35)
            curvature = 2.0 * math.sin(heading_error) / lookahead
            desired_steering = math.atan(wheelbase * curvature)
            desired_steering = min(max(
                desired_steering, steering_min
            ), steering_max)
            steering_rate = 4.0 * (desired_steering - x[6]) - 0.15 * x[5]
            steering_rate = min(max(
                steering_rate, steering_rate_min
            ), steering_rate_max)
            U[t, 0] = accel
            U[t, 1] = steering_rate

            values = _dynamic_ackermann_step_nb(
                x, accel, steering_rate,
                dt, front_axle_distance, rear_axle_distance,
                mass, yaw_inertia, cornering_stiffness_front,
                cornering_stiffness_rear, tire_friction_coefficient,
                gravity, aerodynamic_drag_coefficient,
                rolling_resistance_force, minimum_tire_speed,
                dynamics_substeps, v_min, v_max,
                lateral_velocity_limit, yaw_rate_limit,
                accel_min, accel_max, steering_min, steering_max,
                steering_rate_min, steering_rate_max,
            )
            for j in range(7):
                x[j] = values[j]
        return U

    @njit(cache=True)
    def nominal_controls_to_goal_nb(
        x0, goal, horizon,
        dt, front_axle_distance, rear_axle_distance,
        mass, yaw_inertia, cornering_stiffness_front,
        cornering_stiffness_rear, tire_friction_coefficient,
        gravity, aerodynamic_drag_coefficient,
        rolling_resistance_force, minimum_tire_speed,
        dynamics_substeps, v_min, v_max,
        lateral_velocity_limit, yaw_rate_limit,
        accel_min, accel_max, steering_min, steering_max,
        steering_rate_min, steering_rate_max,
    ):
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
            desired_speed = min(max(
                0.20 + 2.2 * dist * heading_scale, 0.0
            ), v_max)
            accel = min(max(
                3.0 * (desired_speed - x[3]), accel_min
            ), accel_max)

            lookahead = max(dist, 0.35)
            curvature = 2.0 * math.sin(heading_error) / lookahead
            desired_steering = math.atan(wheelbase * curvature)
            desired_steering = min(max(
                desired_steering, steering_min
            ), steering_max)
            steering_rate = 4.0 * (desired_steering - x[6]) - 0.15 * x[5]
            steering_rate = min(max(
                steering_rate, steering_rate_min
            ), steering_rate_max)
            U[t, 0] = accel
            U[t, 1] = steering_rate

            values = _dynamic_ackermann_step_nb(
                x, accel, steering_rate,
                dt, front_axle_distance, rear_axle_distance,
                mass, yaw_inertia, cornering_stiffness_front,
                cornering_stiffness_rear, tire_friction_coefficient,
                gravity, aerodynamic_drag_coefficient,
                rolling_resistance_force, minimum_tire_speed,
                dynamics_substeps, v_min, v_max,
                lateral_velocity_limit, yaw_rate_limit,
                accel_min, accel_max, steering_min, steering_max,
                steering_rate_min, steering_rate_max,
            )
            for j in range(7):
                x[j] = values[j]
        return U

    @njit(cache=True)
    def temporal_smooth_noise_nb(noise, alpha):
        one_minus_alpha = 1.0 - alpha
        for n in range(noise.shape[0]):
            for t in range(1, noise.shape[1]):
                noise[n, t, 0] = (
                    alpha * noise[n, t - 1, 0]
                    + one_minus_alpha * noise[n, t, 0]
                )
                noise[n, t, 1] = (
                    alpha * noise[n, t - 1, 1]
                    + one_minus_alpha * noise[n, t, 1]
                )
        return noise

    @njit(cache=True)
    def rectangle_circle_clearance_nb(
        px, py, heading, circle_x, circle_y, circle_radius,
        vehicle_length, vehicle_width,
    ):
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
    def minimum_rectangle_circle_clearance_nb(
        state, circle_centers, circle_radii, vehicle_length, vehicle_width
    ):
        if circle_radii.shape[0] == 0:
            return 1e18
        px = state[0]
        py = state[1]
        heading = state[2]
        best = 1e18
        for j in range(circle_radii.shape[0]):
            clearance = rectangle_circle_clearance_nb(
                px, py, heading,
                circle_centers[j, 0], circle_centers[j, 1], circle_radii[j],
                vehicle_length, vehicle_width,
            )
            if clearance < best:
                best = clearance
        return best

    @njit(cache=True)
    def path_min_clearance_to_circles_nb(
        path, circle_centers, circle_radii, vehicle_length, vehicle_width, substeps
    ):
        count = path.shape[0]
        if count == 0 or circle_radii.shape[0] == 0:
            return 1e18

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
                headings[i] = headings[i - 1] + _wrap_angle_nb(
                    headings[i] - headings[i - 1]
                )

        best = 1e18
        for i in range(count):
            state = np.empty(3, dtype=np.float64)
            state[0] = path[i, 0]
            state[1] = path[i, 1]
            state[2] = headings[i]
            clearance = minimum_rectangle_circle_clearance_nb(
                state, circle_centers, circle_radii, vehicle_length, vehicle_width
            )
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
                    clearance = minimum_rectangle_circle_clearance_nb(
                        state, circle_centers, circle_radii,
                        vehicle_length, vehicle_width,
                    )
                    if clearance < best:
                        best = clearance
        return best

    @njit(cache=True)
    def rollout_collision_mask_nb(
        X, circle_centers, circle_radii, goal,
        vehicle_length, vehicle_width, hard_collision_clearance,
        rollout_goal_tolerance,
    ):
        N = X.shape[0]
        H = X.shape[1] - 1
        mask = np.zeros(N, dtype=np.bool_)
        goal_radius_sq = rollout_goal_tolerance * rollout_goal_tolerance

        for n in range(N):
            for t in range(H):
                state = X[n, t + 1]
                clearance = minimum_rectangle_circle_clearance_nb(
                    state, circle_centers, circle_radii,
                    vehicle_length, vehicle_width,
                )
                if clearance < hard_collision_clearance:
                    mask[n] = True
                    break

                gx = state[0] - goal[0]
                gy = state[1] - goal[1]
                if gx * gx + gy * gy <= goal_radius_sq:
                    break
        return mask

    @njit(cache=True)
    def apply_smooth_safe_control_nb(
        x_current, u, previous_control, has_previous_control,
        circle_centers, circle_radii,
        apply_control_lowpass, control_lowpass_alpha,
        max_delta_accel, max_delta_steering_rate,
        enforce_one_step_safety, one_step_safety_clearance,
        vehicle_length, vehicle_width,
        dt, front_axle_distance, rear_axle_distance,
        mass, yaw_inertia, cornering_stiffness_front,
        cornering_stiffness_rear, tire_friction_coefficient,
        gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
        minimum_tire_speed, dynamics_substeps, v_min, v_max,
        lateral_velocity_limit, yaw_rate_limit,
        accel_min, accel_max, steering_min, steering_max,
        steering_rate_min, steering_rate_max,
    ):
        cmd = np.empty(2, dtype=np.float64)
        cmd[0] = u[0]
        cmd[1] = u[1]

        if has_previous_control:
            if apply_control_lowpass:
                alpha = min(max(control_lowpass_alpha, 0.0), 1.0)
                cmd[0] = alpha * previous_control[0] + (1.0 - alpha) * cmd[0]
                cmd[1] = alpha * previous_control[1] + (1.0 - alpha) * cmd[1]

            delta_accel = cmd[0] - previous_control[0]
            delta_accel = min(max(delta_accel, -max_delta_accel), max_delta_accel)
            delta_steering_rate = cmd[1] - previous_control[1]
            delta_steering_rate = min(
                max(delta_steering_rate, -max_delta_steering_rate),
                max_delta_steering_rate,
            )
            cmd[0] = previous_control[0] + delta_accel
            cmd[1] = previous_control[1] + delta_steering_rate

        cmd[0] = min(max(cmd[0], accel_min), accel_max)
        cmd[1] = min(max(cmd[1], steering_rate_min), steering_rate_max)

        if enforce_one_step_safety and circle_radii.shape[0] > 0:
            values = _dynamic_ackermann_step_nb(
                x_current, cmd[0], cmd[1],
                dt, front_axle_distance, rear_axle_distance,
                mass, yaw_inertia, cornering_stiffness_front,
                cornering_stiffness_rear, tire_friction_coefficient,
                gravity, aerodynamic_drag_coefficient, rolling_resistance_force,
                minimum_tire_speed, dynamics_substeps, v_min, v_max,
                lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max,
                steering_min, steering_max, steering_rate_min, steering_rate_max,
            )
            x_next = np.empty(7, dtype=np.float64)
            for j in range(7):
                x_next[j] = values[j]

            current_clearance = minimum_rectangle_circle_clearance_nb(
                x_current, circle_centers, circle_radii,
                vehicle_length, vehicle_width,
            )
            next_clearance = minimum_rectangle_circle_clearance_nb(
                x_next, circle_centers, circle_radii,
                vehicle_length, vehicle_width,
            )
            moving_deeper = next_clearance < current_clearance - 1e-4
            below_required = next_clearance < one_step_safety_clearance
            if below_required and moving_deeper:
                if x_current[3] > 0.0:
                    cmd[0] = accel_min
                else:
                    cmd[0] = min(0.0, cmd[0])
        return cmd

    @njit(cache=True)
    def fast_swarm_prior_costs_nb(
        X,
        U,
        mean_path,
        cov_blocks,
        mode_probability,
        circle_centers,
        circle_radii,
        goal,
        horizon,
        vehicle_length,
        vehicle_width,
        base_safety_margin,
        uncertainty_margin_gain,
        w_goal,
        w_obstacle,
        w_control,
        w_control_smooth,
        w_heading,
        w_mode_prior,
        w_reference_tracking,
        smooth_accel_weight,
        smooth_steering_rate_weight,
        sigma_floor,
        max_precision,
        use_gaussian_tracking,
        use_uncertainty_margin,
        use_mode_prior,
        use_mean_reference,
    ):
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
                cost += (w_goal / H) * (gx * gx + gy * gy)


                s00 = sigma_floor * sigma_floor
                s01 = 0.0
                s11 = sigma_floor * sigma_floor

                if use_mean_reference:
                    mux = mean_path[t, 0]
                    muy = mean_path[t, 1]
                    ex = px - mux
                    ey = py - muy

                    s00 = cov_blocks[t, 0, 0] + sigma_floor * sigma_floor
                    s01 = cov_blocks[t, 0, 1]
                    s11 = cov_blocks[t, 1, 1] + sigma_floor * sigma_floor

                    if use_gaussian_tracking:


                        trace = s00 + s11
                        diff = s00 - s11
                        disc = math.sqrt(diff * diff + 4.0 * s01 * s01)
                        lam1 = 0.5 * (trace + disc)
                        lam2 = 0.5 * (trace - disc)

                        min_var = sigma_floor * sigma_floor
                        if lam1 < min_var:
                            lam1 = min_var
                        if lam2 < min_var:
                            lam2 = min_var

                        p1 = 1.0 / lam1
                        p2 = 1.0 / lam2
                        if p1 > max_precision:
                            p1 = max_precision
                        if p2 > max_precision:
                            p2 = max_precision


                        if abs(s01) < 1e-12 and abs(diff) < 1e-12:
                            inv00 = p1
                            inv01 = 0.0
                            inv11 = p2
                        else:

                            vx = s01
                            vy = lam1 - s00
                            norm_v = math.sqrt(vx * vx + vy * vy)
                            if norm_v < 1e-12:
                                vx = 1.0
                                vy = 0.0
                            else:
                                vx /= norm_v
                                vy /= norm_v


                            wx = -vy
                            wy = vx

                            inv00 = p1 * vx * vx + p2 * wx * wx
                            inv01 = p1 * vx * vy + p2 * wx * wy
                            inv11 = p1 * vy * vy + p2 * wy * wy

                        mahal = ex * (inv00 * ex + inv01 * ey) + ey * (inv01 * ex + inv11 * ey)

                    else:

                        pass

                    if t < H - 1:
                        tx = mean_path[t + 1, 0] - mean_path[t, 0]
                        ty = mean_path[t + 1, 1] - mean_path[t, 1]
                        if math.sqrt(tx * tx + ty * ty) > 1e-9:
                            ref_heading = math.atan2(ty, tx)
                            dh = _wrap_angle_nb(X[n, t + 1, 2] - ref_heading)


                heading = X[n, t + 1, 2]
                for j in range(M):
                    dx = px - circle_centers[j, 0]
                    dy = py - circle_centers[j, 1]
                    norm = math.sqrt(dx * dx + dy * dy) + 1e-12
                    nx = dx / norm
                    ny = dy / norm
                    clearance = rectangle_circle_clearance_nb(
                        px, py, heading,
                        circle_centers[j, 0], circle_centers[j, 1], circle_radii[j],
                        vehicle_length, vehicle_width,
                    )

                    margin = base_safety_margin

                    if use_uncertainty_margin and use_mean_reference:
                        sigma_n_sq = nx * (s00 * nx + s01 * ny) + ny * (s01 * nx + s11 * ny)
                        if sigma_n_sq < 0.0:
                            sigma_n_sq = 0.0
                        margin += uncertainty_margin_gain * math.sqrt(sigma_n_sq)

                    z = 8.0 * (margin - clearance)
                    sp = _softplus_scalar_nb(z)
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
                smooth_cost += smooth_accel_weight * dv * dv + smooth_steering_rate_weight * dom * dom
            cost += w_control_smooth * smooth_cost

            if use_mode_prior:
                cost += w_mode_prior * (-math.log(mode_probability + 1e-12))

            costs[n] = cost

        return costs


    @njit(cache=True)
    def standard_mppi_costs_batch_nb(
        X,
        U,
        circle_centers,
        circle_radii,
        goal,
        horizon,
        vehicle_length,
        vehicle_width,
        base_safety_margin,
        w_goal,
        w_time_to_goal,
        rollout_goal_tolerance,
        w_obstacle,
        w_control,
        w_control_smooth,
        w_steering_angle,
        w_yaw_rate,
    ):


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


                cost += w_time_to_goal / H
                cost += (w_goal / H) * goal_distance_sq


                yaw_rate = X[n, t + 1, 5]
                steering_angle = X[n, t + 1, 6]
                cost += w_steering_angle * steering_angle * steering_angle
                cost += w_yaw_rate * yaw_rate * yaw_rate

                heading = X[n, t + 1, 2]
                for j in range(M):
                    clearance = rectangle_circle_clearance_nb(
                        px, py, heading,
                        circle_centers[j, 0], circle_centers[j, 1], circle_radii[j],
                        vehicle_length, vehicle_width,
                    )
                    sp = _softplus_scalar_nb(8.0 * (base_safety_margin - clearance))
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
    def interpolated_obstacle_penalty_nb(
        X,
        circle_centers,
        circle_radii,
        vehicle_length,
        vehicle_width,
        base_safety_margin,
        w_obstacle,
        collision_substeps,
        hard_collision_clearance,
        hard_collision_penalty,
    ):
        N = X.shape[0]
        H = X.shape[1] - 1
        M = circle_radii.shape[0]
        extras = np.zeros(N, dtype=np.float64)
        substeps = max(0, int(collision_substeps))
        denominator = float(substeps + 1)

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
                    min_clearance = 1e18

                    for j in range(M):
                        clearance = rectangle_circle_clearance_nb(
                            px, py, heading,
                            circle_centers[j, 0], circle_centers[j, 1], circle_radii[j],
                            vehicle_length, vehicle_width,
                        )
                        if clearance < min_clearance:
                            min_clearance = clearance

                        if q <= substeps:
                            sp = _softplus_scalar_nb(
                                8.0 * (base_safety_margin - clearance)
                            )
                            cost += (w_obstacle / denominator) * sp * sp

                    if min_clearance < hard_collision_clearance:
                        penetration = hard_collision_clearance - min_clearance
                        cost += hard_collision_penalty * (1.0 + penetration * penetration)

            extras[n] = cost

        return extras

    @njit(cache=True)
    def boundary_penalty_nb(
        X,
        xmin,
        xmax,
        ymin,
        ymax,
        vehicle_length,
        vehicle_width,
        base_safety_margin,
        w_boundary,
        collision_substeps,
        hard_collision_clearance,
        hard_collision_penalty,
    ):
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
                    clearance = min(
                        px - xmin - extent_x,
                        xmax - px - extent_x,
                        py - ymin - extent_y,
                        ymax - py - extent_y,
                    )

                    if q <= substeps:
                        sp = _softplus_scalar_nb(
                            8.0 * (base_safety_margin - clearance)
                        )
                        cost += (w_boundary / denominator) * sp * sp

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
                x_cross = xi + (py - yi) * (xj - xi) / ((yj - yi) + 1e-18)
                if px < x_cross:
                    inside = not inside
        return inside

    @njit(cache=True)
    def point_in_oriented_rectangle_nb(
        px, py, cx, cy, heading, half_length, half_width
    ):
        dx = px - cx
        dy = py - cy
        c = math.cos(heading)
        s = math.sin(heading)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        return (
            abs(local_x) <= half_length + 1e-12
            and abs(local_y) <= half_width + 1e-12
        )

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
        return (
            min(ax, bx) - 1e-12 <= px <= max(ax, bx) + 1e-12
            and min(ay, by) - 1e-12 <= py <= max(ay, by) + 1e-12
        )

    @njit(cache=True)
    def segments_intersect_nb(ax, ay, bx, by, cx, cy, dx, dy):
        o1 = orientation_2d_nb(ax, ay, bx, by, cx, cy)
        o2 = orientation_2d_nb(ax, ay, bx, by, dx, dy)
        o3 = orientation_2d_nb(cx, cy, dx, dy, ax, ay)
        o4 = orientation_2d_nb(cx, cy, dx, dy, bx, by)
        if (((o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0)) and
                ((o3 > 0.0 and o4 < 0.0) or (o3 < 0.0 and o4 > 0.0))):
            return True
        return (
            (abs(o1) <= 1e-12 and point_on_segment_nb(cx, cy, ax, ay, bx, by))
            or (abs(o2) <= 1e-12 and point_on_segment_nb(dx, dy, ax, ay, bx, by))
            or (abs(o3) <= 1e-12 and point_on_segment_nb(ax, ay, cx, cy, dx, dy))
            or (abs(o4) <= 1e-12 and point_on_segment_nb(bx, by, cx, cy, dx, dy))
        )

    @njit(cache=True)
    def segment_segment_distance_nb(ax, ay, bx, by, cx, cy, dx, dy):
        if segments_intersect_nb(ax, ay, bx, by, cx, cy, dx, dy):
            return 0.0
        return min(
            point_segment_dist_nb(ax, ay, cx, cy, dx, dy),
            point_segment_dist_nb(bx, by, cx, cy, dx, dy),
            point_segment_dist_nb(cx, cy, ax, ay, bx, by),
            point_segment_dist_nb(dx, dy, ax, ay, bx, by),
        )

    @njit(cache=True)
    def rectangle_polygon_clearance_nb(
        state, poly, n, vehicle_length, vehicle_width
    ):
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
                return -1e-9

        for i in range(n):
            if point_in_oriented_rectangle_nb(
                poly[i, 0], poly[i, 1], cx, cy, heading,
                half_length, half_width,
            ):
                return -1e-9

        best = 1e18
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
                distance = segment_segment_distance_nb(
                    ax, ay, bx, by,
                    poly[j, 0], poly[j, 1], poly[nj, 0], poly[nj, 1],
                )
                if distance <= 1e-12:
                    return -1e-9
                if distance < best:
                    best = distance
        return best

    @njit(cache=True)
    def min_clearance_nb(
        states, polys_padded, poly_lengths, vehicle_length, vehicle_width
    ):
        best = 1e18
        for state_index in range(states.shape[0]):
            state = states[state_index]
            for polygon_index in range(poly_lengths.shape[0]):
                n = int(poly_lengths[polygon_index])
                clearance = rectangle_polygon_clearance_nb(
                    state, polys_padded[polygon_index], n,
                    vehicle_length, vehicle_width,
                )
                if clearance < best:
                    best = clearance
        return best

else:
    localize_prior_horizon_nb = None
    apply_gaussian_prior_noise_nb = None
    rollout_ackermann_batch_nb = None
    rollout_ackermann_single_nb = None
    nominal_controls_to_track_path_nb = None
    nominal_controls_to_goal_nb = None
    temporal_smooth_noise_nb = None
    fast_swarm_prior_costs_nb = None
    standard_mppi_costs_batch_nb = None
    interpolated_obstacle_penalty_nb = None
    boundary_penalty_nb = None
    minimum_rectangle_circle_clearance_nb = None
    path_min_clearance_to_circles_nb = None
    rollout_collision_mask_nb = None
    apply_smooth_safe_control_nb = None
    min_clearance_nb = None


def obstacle_circles_to_arrays(obstacle_circles: List[Tuple[Array, float]]) -> Tuple[Array, Array]:
    if not obstacle_circles:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    return centers, radii


def apply_smooth_safe_control(
    x_current: Array,
    u: Array,
    previous_control: Optional[Array],
    obstacle_circles: List[Tuple[Array, float]],
    cfg: MPPIConfig,
) -> Array:
    state = np.asarray(x_current, dtype=np.float64)
    command = np.asarray(u, dtype=np.float64)
    centers, radii = obstacle_circles_to_arrays(obstacle_circles)

    if apply_smooth_safe_control_nb is not None:
        has_previous = previous_control is not None
        previous = (
            np.asarray(previous_control, dtype=np.float64)
            if has_previous else np.zeros(2, dtype=np.float64)
        )
        return apply_smooth_safe_control_nb(
            state, command, previous, has_previous, centers, radii,
            bool(cfg.apply_control_lowpass), float(cfg.control_lowpass_alpha),
            float(cfg.max_delta_accel), float(cfg.max_delta_steering_rate),
            bool(cfg.enforce_one_step_safety),
            float(cfg.one_step_safety_clearance),
            float(cfg.vehicle_length), float(cfg.vehicle_width),
            *_dynamic_model_arguments(cfg),
        )

    cmd = command.copy()
    if previous_control is not None:
        previous = np.asarray(previous_control, dtype=np.float64)
        if cfg.apply_control_lowpass:
            alpha = float(np.clip(cfg.control_lowpass_alpha, 0.0, 1.0))
            cmd = alpha * previous + (1.0 - alpha) * cmd
        cmd[0] = previous[0] + float(np.clip(
            cmd[0] - previous[0], -cfg.max_delta_accel, cfg.max_delta_accel
        ))
        cmd[1] = previous[1] + float(np.clip(
            cmd[1] - previous[1],
            -cfg.max_delta_steering_rate, cfg.max_delta_steering_rate,
        ))

    cmd[0] = np.clip(cmd[0], cfg.accel_min, cfg.accel_max)
    cmd[1] = np.clip(cmd[1], cfg.steering_rate_min, cfg.steering_rate_max)
    if cfg.enforce_one_step_safety and len(radii) > 0:
        x_next = ackermann_step(state, cmd, cfg)
        current_clearance = minimum_rectangle_circle_clearance(
            state, centers, radii, cfg.vehicle_length, cfg.vehicle_width
        )
        next_clearance = minimum_rectangle_circle_clearance(
            x_next, centers, radii, cfg.vehicle_length, cfg.vehicle_width
        )
        if (
            next_clearance < cfg.one_step_safety_clearance
            and next_clearance < current_clearance - 1e-4
        ):
            cmd[0] = cfg.accel_min if state[3] > 0.0 else min(0.0, cmd[0])
    return cmd


def interpolated_obstacle_penalty(
    X: Array,
    obstacle_circles: List[Tuple[Array, float]],
    cfg: MPPIConfig,
) -> Array:
    N = int(X.shape[0])
    if not obstacle_circles:
        return np.zeros(N, dtype=np.float64)

    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    if interpolated_obstacle_penalty_nb is not None:
        return interpolated_obstacle_penalty_nb(
            np.asarray(X, dtype=np.float64),
            centers,
            radii,
            float(cfg.vehicle_length),
            float(cfg.vehicle_width),
            float(cfg.base_safety_margin),
            float(cfg.w_obstacle),
            int(cfg.collision_substeps),
            float(cfg.hard_collision_clearance),
            float(cfg.hard_collision_penalty),
        )

    extras = np.zeros(N, dtype=np.float64)
    substeps = max(0, int(cfg.collision_substeps))
    denominator = float(substeps + 1)
    for n in range(N):
        for t in range(X.shape[1] - 1):
            h0 = float(X[n, t, 2])
            dh = wrap_angle(float(X[n, t + 1, 2]) - h0)
            for q in range(1, substeps + 2):
                alpha = q / denominator
                state = np.asarray(X[n, t], dtype=np.float64).copy()
                state[:2] = X[n, t, :2] + alpha * (
                    X[n, t + 1, :2] - X[n, t, :2]
                )
                state[2] = wrap_angle(h0 + alpha * dh)
                clearances = np.array([
                    rectangle_circle_clearance(
                        state,
                        centers[j],
                        radii[j],
                        cfg.vehicle_length,
                        cfg.vehicle_width,
                    )
                    for j in range(len(radii))
                ], dtype=np.float64)
                if q <= substeps:
                    extras[n] += (cfg.w_obstacle / denominator) * np.sum(
                        softplus(8.0 * (cfg.base_safety_margin - clearances)) ** 2
                    )
                min_value = float(np.min(clearances))
                if min_value < cfg.hard_collision_clearance:
                    penetration = cfg.hard_collision_clearance - min_value
                    extras[n] += cfg.hard_collision_penalty * (
                        1.0 + penetration ** 2
                    )
    return extras


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


def path_min_clearance_to_circles(
    path: Array,
    obstacle_circles: List[Tuple[Array, float]],
    vehicle_length: float,
    vehicle_width: float,
    substeps: int = 2,
) -> float:
    p = np.asarray(path, dtype=np.float64)
    if len(p) == 0 or not obstacle_circles:
        return float("inf")

    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    if path_min_clearance_to_circles_nb is not None:
        return float(path_min_clearance_to_circles_nb(
            p, centers, radii, float(vehicle_length), float(vehicle_width),
            int(substeps),
        ))
    headings = _path_tangent_headings(p)
    states = [np.column_stack([p, headings])]
    if len(p) > 1:
        count = max(0, int(substeps))
        for q in range(1, count + 1):
            alpha = q / float(count + 1)
            positions = p[:-1] + alpha * (p[1:] - p[:-1])
            delta_heading = np.array([
                wrap_angle(headings[i + 1] - headings[i])
                for i in range(len(headings) - 1)
            ])
            interp_heading = headings[:-1] + alpha * delta_heading
            states.append(np.column_stack([positions, interp_heading]))

    minimum = float("inf")
    for state in np.vstack(states):
        minimum = min(
            minimum,
            minimum_rectangle_circle_clearance(
                state,
                centers,
                radii,
                vehicle_length,
                vehicle_width,
            ),
        )
    return minimum


def unblocked_mode_indices(
    local_modes: Sequence[MPPIHomotopyMode],
    obstacle_circles: List[Tuple[Array, float]],
    cfg: MPPIConfig,
) -> Tuple[List[int], Array]:


    if not cfg.suppress_blocked_modes or len(local_modes) <= 1:
        return (
            list(range(len(local_modes))),
            np.full(len(local_modes), np.nan, dtype=np.float64),
        )

    clearances = np.asarray([
        path_min_clearance_to_circles(
            mode.mean_path,
            obstacle_circles,
            cfg.vehicle_length,
            cfg.vehicle_width,
            substeps=cfg.mode_blocking_substeps,
        )
        for mode in local_modes
    ], dtype=np.float64)

    usable = np.where(clearances >= cfg.mode_blocking_clearance)[0].tolist()
    if not usable:
        usable = [int(np.argmax(clearances))]
    return usable, clearances


def obstacles_to_padded_arrays(obstacles: Sequence) -> Tuple[Array, Array]:
    polys = [_poly_vertices(o).astype(np.float64) for o in obstacles]
    if not polys:
        return np.zeros((0, 0, 2), dtype=np.float64), np.zeros(0, dtype=np.int64)
    max_n = max(p.shape[0] for p in polys)
    padded = np.zeros((len(polys), max_n, 2), dtype=np.float64)
    lengths = np.zeros(len(polys), dtype=np.int64)

    for i, p in enumerate(polys):
        padded[i, :p.shape[0], :] = p
        lengths[i] = p.shape[0]

    return padded, lengths


def localize_mode_for_state_with_index(
    mode: MPPIHomotopyMode,
    x_current: Array,
    H: int,
    previous_idx: Optional[int] = None,
    max_advance: Optional[int] = None,
) -> Tuple[MPPIHomotopyMode, int]:
    mode = _ensure_mode_prior_cache(mode)
    mu = np.asarray(mode.mean_path, dtype=np.float64)
    nearest_idx = int(np.argmin(np.sum((mu - x_current[:2]) ** 2, axis=1)))

    if previous_idx is None:
        idx = nearest_idx
    else:
        idx = max(int(previous_idx), nearest_idx)
        if max_advance is not None:
            idx = min(idx, int(previous_idx) + int(max_advance))
        idx = min(idx, len(mu) - 2)

    args = (
        np.asarray(mode.mean_path, dtype=np.float64),
        np.asarray(mode.cov_blocks, dtype=np.float64),
        np.asarray(mode.arc_length, dtype=np.float64),
        np.asarray(mode.gaussian_variance, dtype=np.float64),
        idx,
        H,
    )
    if localize_prior_horizon_nb is not None:
        local_mu, local_cov, local_gaussian = localize_prior_horizon_nb(*args)
    else:
        source_s = np.linspace(mode.arc_length[idx], mode.arc_length[-1], H)
        local_mu = np.column_stack([
            np.interp(source_s, mode.arc_length, mode.mean_path[:, 0]),
            np.interp(source_s, mode.arc_length, mode.mean_path[:, 1]),
        ])
        local_cov = np.empty((H, 2, 2), dtype=np.float64)
        for row in range(2):
            for col in range(2):
                local_cov[:, row, col] = np.interp(
                    source_s, mode.arc_length, mode.cov_blocks[:, row, col]
                )
        local_gaussian = np.interp(source_s, mode.arc_length, mode.gaussian_variance)

    local_arc = np.zeros(H, dtype=np.float64)
    if H > 1:
        local_arc[1:] = np.cumsum(np.linalg.norm(np.diff(local_mu, axis=0), axis=1))

    return MPPIHomotopyMode(
        signature=mode.signature,
        probability=mode.probability,
        mean_path=local_mu,
        cov_blocks=local_cov,
        sample_paths=None,
        arc_length=local_arc,
        gaussian_variance=local_gaussian,
    ), idx

def localize_mode_for_state(mode: MPPIHomotopyMode, x_current: Array, H: int) -> MPPIHomotopyMode:
    local_mode, _ = localize_mode_for_state_with_index(
        mode=mode,
        x_current=x_current,
        H=H,
        previous_idx=None,
        max_advance=None,
    )
    return local_mode


def localize_path_for_state_with_index(
    path: Array,
    x_current: Array,
    H: int,
    previous_idx: Optional[int] = None,
    max_advance: Optional[int] = None,
) -> Tuple[Array, int]:
    p = np.asarray(path, dtype=np.float64)
    d = np.linalg.norm(p - x_current[:2], axis=1)
    nearest_idx = int(np.argmin(d))

    if previous_idx is None:
        idx = nearest_idx
    else:
        idx = max(int(previous_idx), nearest_idx)
        if max_advance is not None:
            idx = min(idx, int(previous_idx) + int(max_advance))
        idx = min(idx, len(p) - 2)

    tail = p[idx:] if idx < len(p) - 1 else p[-2:]
    return resample_path(tail, H), idx


def localize_path_for_state(path: Array, x_current: Array, H: int) -> Array:
    local_path, _ = localize_path_for_state_with_index(
        path=path,
        x_current=x_current,
        H=H,
        previous_idx=None,
        max_advance=None,
    )
    return local_path


def nominal_controls_to_track_path(x0: Array, ref: Array, cfg: MPPIConfig) -> Array:
    if nominal_controls_to_track_path_nb is not None:
        return nominal_controls_to_track_path_nb(
            np.asarray(x0, dtype=np.float64),
            np.asarray(ref, dtype=np.float64),
            int(cfg.horizon),
            *_dynamic_model_arguments(cfg),
        )

    U = np.zeros((cfg.horizon, 2), dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64).copy()
    for t in range(cfg.horizon):
        target = ref[min(t + 3, len(ref) - 1)]
        delta = target - x[:2]
        distance = float(np.linalg.norm(delta))
        desired_heading = math.atan2(delta[1], delta[0])
        heading_error = wrap_angle(desired_heading - x[2])
        heading_scale = max(0.0, math.cos(heading_error)) ** 2
        desired_speed = np.clip(
            0.20 + 2.4 * distance * heading_scale, 0.0, cfg.v_max
        )
        accel = np.clip(
            3.0 * (desired_speed - x[3]), cfg.accel_min, cfg.accel_max
        )
        lookahead = max(distance, 0.35)
        desired_curvature = 2.0 * math.sin(heading_error) / lookahead
        desired_steering = np.clip(
            math.atan(cfg.wheelbase * desired_curvature),
            cfg.steering_min,
            cfg.steering_max,
        )
        steering_rate = np.clip(
            4.0 * (desired_steering - x[6]),
            cfg.steering_rate_min,
            cfg.steering_rate_max,
        )
        U[t] = [accel, steering_rate]
        x = ackermann_step(x, U[t], cfg)
    return U


def nominal_controls_to_goal(x0: Array, goal: Array, cfg: MPPIConfig) -> Array:
    if nominal_controls_to_goal_nb is not None:
        return nominal_controls_to_goal_nb(
            np.asarray(x0, dtype=np.float64),
            np.asarray(goal, dtype=np.float64),
            int(cfg.horizon),
            *_dynamic_model_arguments(cfg),
        )

    U = np.zeros((cfg.horizon, 2), dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64).copy()
    for t in range(cfg.horizon):
        delta = goal - x[:2]
        distance = float(np.linalg.norm(delta))
        desired_heading = math.atan2(delta[1], delta[0])
        heading_error = wrap_angle(desired_heading - x[2])
        heading_scale = max(0.0, math.cos(heading_error)) ** 2
        desired_speed = np.clip(
            0.20 + 2.2 * distance * heading_scale, 0.0, cfg.v_max
        )
        accel = np.clip(
            3.0 * (desired_speed - x[3]), cfg.accel_min, cfg.accel_max
        )
        lookahead = max(distance, 0.35)
        desired_curvature = 2.0 * math.sin(heading_error) / lookahead
        desired_steering = np.clip(
            math.atan(cfg.wheelbase * desired_curvature),
            cfg.steering_min,
            cfg.steering_max,
        )
        steering_rate = np.clip(
            4.0 * (desired_steering - x[6]),
            cfg.steering_rate_min,
            cfg.steering_rate_max,
        )
        U[t] = [accel, steering_rate]
        x = ackermann_step(x, U[t], cfg)
    return U


def build_empirical_nominal_bank(
    x_current: Array,
    global_mode: MPPIHomotopyMode,
    mean_nominal: Array,
    cfg: MPPIConfig,
    rng: np.random.Generator,
    previous_idx: Optional[int] = None,
) -> List[Array]:
    bank = [mean_nominal]
    if not global_mode.sample_paths:
        return bank

    n = min(cfg.max_empirical_nominals_per_mode, len(global_mode.sample_paths))
    ids = rng.choice(len(global_mode.sample_paths), size=n, replace=False)
    for sid in ids:
        local_sample_path, _ = localize_path_for_state_with_index(
            global_mode.sample_paths[int(sid)],
            x_current,
            cfg.horizon,
            previous_idx=previous_idx if cfg.use_monotonic_reference_progress else None,
            max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None,
        )
        bank.append(nominal_controls_to_track_path(x_current, local_sample_path, cfg))
    return bank


def capped_inverse_covariance(Sigma: Array, sigma_floor: float, max_precision: float) -> Array:


    S = 0.5 * (Sigma + Sigma.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, sigma_floor ** 2)
    precision = np.minimum(1.0 / vals, max_precision)
    return vecs @ np.diag(precision) @ vecs.T


def fast_swarm_prior_costs(
    X: Array,
    U: Array,
    mode: MPPIHomotopyMode,
    obstacle_circles: List[Tuple[Array, float]],
    goal: Array,
    cfg: MPPIConfig,
    *,
    use_gaussian_tracking: bool,
    use_uncertainty_margin: bool,
    use_mode_prior: bool,
    use_mean_reference: bool,
) -> Array:

    del mode, use_gaussian_tracking, use_uncertainty_margin, use_mode_prior, use_mean_reference
    return standard_mppi_costs_batch(X, U, obstacle_circles, goal, cfg)


def boundary_penalty(X: Array, cfg: MPPIConfig) -> Array:
    states = np.asarray(X, dtype=np.float64)
    if states.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    if boundary_penalty_nb is not None:
        return boundary_penalty_nb(
            states,
            float(cfg.boundary_xmin),
            float(cfg.boundary_xmax),
            float(cfg.boundary_ymin),
            float(cfg.boundary_ymax),
            float(cfg.vehicle_length),
            float(cfg.vehicle_width),
            float(cfg.base_safety_margin),
            float(cfg.w_boundary),
            int(cfg.collision_substeps),
            float(cfg.hard_collision_clearance),
            float(cfg.hard_collision_penalty),
        )

    N = states.shape[0]
    extras = np.zeros(N, dtype=np.float64)
    substeps = max(0, int(cfg.collision_substeps))
    denominator = float(substeps + 1)
    half_length = 0.5 * float(cfg.vehicle_length)
    half_width = 0.5 * float(cfg.vehicle_width)

    for n in range(N):
        for t in range(states.shape[1] - 1):
            h0 = float(states[n, t, 2])
            dh = wrap_angle(float(states[n, t + 1, 2]) - h0)
            for q in range(1, substeps + 2):
                alpha = q / denominator
                px, py = states[n, t, :2] + alpha * (
                    states[n, t + 1, :2] - states[n, t, :2]
                )
                heading = wrap_angle(h0 + alpha * dh)
                c = abs(math.cos(heading))
                s = abs(math.sin(heading))
                extent_x = half_length * c + half_width * s
                extent_y = half_length * s + half_width * c
                clearance = min(
                    px - cfg.boundary_xmin - extent_x,
                    cfg.boundary_xmax - px - extent_x,
                    py - cfg.boundary_ymin - extent_y,
                    cfg.boundary_ymax - py - extent_y,
                )
                if q <= substeps:
                    extras[n] += (cfg.w_boundary / denominator) * float(
                        softplus(8.0 * (cfg.base_safety_margin - clearance)) ** 2
                    )
                if clearance < cfg.hard_collision_clearance:
                    penetration = cfg.hard_collision_clearance - clearance
                    extras[n] += cfg.hard_collision_penalty * (
                        1.0 + penetration ** 2
                    )
    return extras


def standard_mppi_costs_batch(
    X: Array,
    U: Array,
    obstacle_circles: List[Tuple[Array, float]],
    goal: Array,
    cfg: MPPIConfig,
) -> Array:
    if standard_mppi_costs_batch_nb is not None:
        centers, radii = obstacle_circles_to_arrays(obstacle_circles)
        costs = standard_mppi_costs_batch_nb(
            np.asarray(X, dtype=np.float64),
            np.asarray(U, dtype=np.float64),
            centers,
            radii,
            np.asarray(goal, dtype=np.float64),
            int(cfg.horizon),
            float(cfg.vehicle_length),
            float(cfg.vehicle_width),
            float(cfg.base_safety_margin),
            float(cfg.w_goal),
            float(cfg.w_time_to_goal),
            float(cfg.rollout_goal_tolerance),
            float(cfg.w_obstacle),
            float(cfg.w_control),
            float(cfg.w_control_smooth),
            float(cfg.w_steering_angle),
            float(cfg.w_yaw_rate),
        )
        return costs + boundary_penalty(X, cfg)

    N, H, _ = U.shape
    costs = np.zeros(N, dtype=np.float64)
    goal_radius_sq = float(cfg.rollout_goal_tolerance) ** 2
    for n in range(N):
        arrival_index = H
        for t in range(H):
            state = X[n, t + 1]
            p = state[:2]
            goal_delta = p - goal
            goal_distance_sq = float(goal_delta @ goal_delta)
            costs[n] += cfg.w_time_to_goal / H
            costs[n] += (cfg.w_goal / H) * goal_distance_sq
            steering_angle = float(state[6])
            yaw_rate = float(state[5])
            costs[n] += cfg.w_steering_angle * steering_angle ** 2
            costs[n] += cfg.w_yaw_rate * yaw_rate ** 2

            for center, radius in obstacle_circles:
                clearance = rectangle_circle_clearance(
                    state,
                    center,
                    radius,
                    cfg.vehicle_length,
                    cfg.vehicle_width,
                )
                costs[n] += cfg.w_obstacle * float(
                    softplus(8.0 * (cfg.base_safety_margin - clearance)) ** 2
                )

            if goal_distance_sq <= goal_radius_sq:
                arrival_index = t + 1
                break

        prefix = U[n, :arrival_index]
        costs[n] += cfg.w_control * np.sum(
            prefix[:, 0] ** 2 + 0.15 * prefix[:, 1] ** 2
        )
        if arrival_index > 1:
            dU = np.diff(prefix, axis=0)
            costs[n] += cfg.w_control_smooth * np.sum(
                dU[:, 0] ** 2 + 0.2 * dU[:, 1] ** 2
            )

    return costs + boundary_penalty(X, cfg)


REP_GAUSSIAN = 1
REP_CORRIDOR = 2
REP_CONTROL_BANK = 3


def stable_representation_costs(
    X: Array,
    U: Array,
    mode: MPPIHomotopyMode,
    obstacle_circles: List[Tuple[Array, float]],
    goal: Array,
    cfg: MPPIConfig,
    *,
    rep_type: int,
    use_mode_prior: bool = False,
) -> Array:


    del mode, rep_type, use_mode_prior
    return standard_mppi_costs_batch(X, U, obstacle_circles, goal, cfg)


def softmin_score(costs: Array, cfg: MPPIConfig) -> float:
    values = np.asarray(costs, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("inf")
    finite_values = values[finite]
    rho = float(np.min(finite_values))
    z = np.exp(-(finite_values - rho) / cfg.lambda_temperature)
    return float(rho - cfg.lambda_temperature * math.log(np.sum(z) / len(values) + 1e-12))


def build_nominal_bank_for_mode(
    x_current: Array,
    local_mode: MPPIHomotopyMode,
    global_mode: MPPIHomotopyMode,
    goal: Array,
    cfg: MPPIConfig,
    rng: np.random.Generator,
    *,
    use_empirical_init: bool,
    use_mean_nominal: bool,
    previous_idx: Optional[int] = None,
) -> List[Array]:
    goal_nominal = nominal_controls_to_goal(x_current, goal, cfg)
    if use_mean_nominal:
        mean_nominal = nominal_controls_to_track_path(x_current, local_mode.mean_path, cfg)
    else:
        mean_nominal = goal_nominal

    if use_empirical_init:
        bank = build_empirical_nominal_bank(
            x_current=x_current,
            global_mode=global_mode,
            mean_nominal=mean_nominal,
            cfg=cfg,
            rng=rng,
            previous_idx=previous_idx,
        )
    else:
        bank = [mean_nominal]


    if not any(np.allclose(candidate, goal_nominal) for candidate in bank):
        bank.append(goal_nominal)
    return bank


def enforce_ackermann_control_bounds(U: Array, cfg: MPPIConfig) -> Array:

    U = np.asarray(U, dtype=np.float64)
    if U.size == 0:
        return U
    U[:, :, 0] = np.clip(U[:, :, 0], cfg.accel_min, cfg.accel_max)
    U[:, :, 1] = np.clip(
        U[:, :, 1], cfg.steering_rate_min, cfg.steering_rate_max
    )
    return U


def sample_controls_from_nominal_bank(
    nominal_bank: List[Array],
    n: int,
    cfg: MPPIConfig,
    rng: np.random.Generator,
    *,
    prefer_empirical: bool = True,
) -> Array:
    if len(nominal_bank) == 1:
        bank_ids = np.zeros(n, dtype=np.int64)
    else:
        probs = np.ones(len(nominal_bank), dtype=np.float64)
        if prefer_empirical:
            probs[0] = max(1e-6, 1.0 - cfg.swarm_init_probability)
            probs[1:] = cfg.swarm_init_probability / (len(nominal_bank) - 1)
        probs /= probs.sum()
        bank_ids = rng.choice(len(nominal_bank), size=n, p=probs)

    U = np.stack([nominal_bank[int(j)].copy() for j in bank_ids], axis=0)
    noise = make_temporally_correlated_noise(n, cfg.horizon, cfg, rng)
    U += noise


    bank_count = len(nominal_bank)
    exact_count = min(bank_count, n)
    for j in range(exact_count):
        U[j] = nominal_bank[j]

    cursor = exact_count
    low_noise_budget = min(max(0, int(cfg.low_noise_proposal_count)), n - cursor)
    for q in range(low_noise_budget):
        j = q % bank_count
        U[cursor] = (
            nominal_bank[j]
            + float(cfg.low_noise_proposal_scale) * noise[cursor]
        )
        cursor += 1

    return enforce_ackermann_control_bounds(U, cfg)


def sample_exact_control_bank(
    x_current: Array,
    global_mode: MPPIHomotopyMode,
    fallback_nominal: Array,
    n: int,
    cfg: MPPIConfig,
    rng: np.random.Generator,
    previous_idx: Optional[int] = None,
) -> Array:
    """Use inverse controls of empirical trajectories without added control noise."""
    if n <= 0:
        return np.zeros((0, cfg.horizon, 2), dtype=np.float64)

    candidates: List[Array] = []
    if global_mode.sample_paths:
        order = rng.permutation(len(global_mode.sample_paths))
        for sid in order:
            local_path, _ = localize_path_for_state_with_index(
                global_mode.sample_paths[int(sid)],
                x_current,
                cfg.horizon,
                previous_idx=(
                    previous_idx if cfg.use_monotonic_reference_progress else None
                ),
                max_advance=(
                    cfg.max_reference_index_advance
                    if cfg.use_monotonic_reference_progress else None
                ),
            )
            candidates.append(nominal_controls_to_track_path(x_current, local_path, cfg))

    if not candidates:
        candidates = [np.asarray(fallback_nominal, dtype=np.float64).copy()]

    U = np.stack([candidates[i % len(candidates)].copy() for i in range(n)], axis=0)
    return enforce_ackermann_control_bounds(U, cfg)


def sample_gaussian_controls(
    x_current: Array,
    local_mode: MPPIHomotopyMode,
    n: int,
    cfg: MPPIConfig,
    rng: np.random.Generator,
) -> Array:
    """Sample in control space with Numba-scaled isotropic prior uncertainty."""
    H = int(cfg.horizon)
    if n <= 0:
        return np.zeros((0, H, 2), dtype=np.float64)

    local_mode = _ensure_mode_prior_cache(local_mode)
    mean_path = np.asarray(local_mode.mean_path, dtype=np.float64)
    nominal = nominal_controls_to_track_path(x_current, mean_path, cfg)
    noise = make_temporally_correlated_noise(n, H, cfg, rng)

    variance = np.asarray(local_mode.gaussian_variance, dtype=np.float64)
    if apply_gaussian_prior_noise_nb is not None:
        noise = apply_gaussian_prior_noise_nb(
            noise,
            variance,
            float(cfg.sigma_floor),
            float(cfg.gaussian_covariance_scale),
        )
    else:
        floor_var = float(cfg.sigma_floor) ** 2
        scale = float(cfg.gaussian_covariance_scale) * np.sqrt(
            np.maximum(variance, floor_var)
        ) / max(float(cfg.sigma_floor), 1e-9)
        noise *= scale[None, :, None]

    U = nominal[None, :, :] + noise
    U[0] = nominal
    return enforce_ackermann_control_bounds(U, cfg)

def nearby_mode_indices(
    global_modes: Sequence[MPPIHomotopyMode],
    x_current: Array,
    cfg: MPPIConfig,
    obstacle_circles: Optional[List[Tuple[Array, float]]] = None,
) -> List[int]:


    if not global_modes:
        return []
    p = np.asarray(x_current[:2], dtype=np.float64)
    distances = np.asarray([
        float(np.min(np.linalg.norm(np.asarray(mode.mean_path) - p[None, :], axis=1)))
        for mode in global_modes
    ], dtype=np.float64)

    locality_order = np.argsort(distances)
    slack = max(0.0, float(cfg.nearby_prior_distance_slack))
    local_threshold = float(distances[locality_order[0]] + slack)
    local_pool = [int(i) for i in locality_order if distances[i] <= local_threshold]
    if not local_pool:
        local_pool = [int(locality_order[0])]


    scores = []
    for i in local_pool:
        blocked_term = 0.0
        if obstacle_circles:
            local_mode = localize_mode_for_state(global_modes[i], x_current, cfg.horizon)
            clearance = path_min_clearance_to_circles(
                local_mode.mean_path,
                obstacle_circles,
                cfg.vehicle_length,
                cfg.vehicle_width,
                substeps=0,
            )
            blocked_term = max(0.0, float(cfg.mode_blocking_clearance) - clearance)
        score = float(distances[i]) + float(cfg.nearby_prior_blocked_penalty) * blocked_term
        scores.append((score, float(distances[i]), i))

    scores.sort()
    limit = min(max(1, int(cfg.max_nearby_prior_modes)), len(scores))
    return [int(item[2]) for item in scores[:limit]]


def balanced_rollout_counts(total: int, groups: int) -> List[int]:

    total = max(1, int(total))
    groups = max(1, min(int(groups), total))
    base, remainder = divmod(total, groups)
    return [base + (1 if i < remainder else 0) for i in range(groups)]

def stable_swarm_mppi_step(
    x_current: Array,
    global_modes: List[MPPIHomotopyMode],
    obstacles: Sequence,
    goal: Array,
    cfg: MPPIConfig,
    rng: np.random.Generator,
    *,
    rep_type: int,
    use_empirical_init: bool,
    use_mean_nominal: bool,
    use_mode_prior: bool,
    progress_by_mode: Optional[Dict[str, int]] = None,
    obstacle_circles: Optional[List[Tuple[Array, float]]] = None,
    record_optimal_traj: bool = True,
) -> Tuple[Array, Dict[str, object], Dict[str, int]]:


    if rep_type not in {
        REP_GAUSSIAN, REP_CORRIDOR, REP_CONTROL_BANK
    }:
        raise ValueError(f"Unsupported pooled proposal representation: {rep_type}")

    progress_by_mode = {} if progress_by_mode is None else dict(progress_by_mode)
    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)

    nearby_indices = nearby_mode_indices(global_modes, x_current, cfg, obstacle_circles)
    candidate_global_modes = [global_modes[i] for i in nearby_indices]
    local_modes = []
    new_progress_by_mode = dict(progress_by_mode)
    for mode in candidate_global_modes:
        key = str(mode.signature)
        previous = progress_by_mode.get(key)
        local_mode, index = localize_mode_for_state_with_index(
            mode,
            x_current,
            cfg.horizon,
            previous_idx=previous if cfg.use_monotonic_reference_progress else None,
            max_advance=(
                cfg.max_reference_index_advance
                if cfg.use_monotonic_reference_progress else None
            ),
        )
        local_modes.append(local_mode)
        new_progress_by_mode[key] = index

    active_mode_indices, mode_clearances = unblocked_mode_indices(
        local_modes, obstacle_circles, cfg
    )
    total_budget = max(1, int(cfg.num_rollouts))
    active_mode_indices = active_mode_indices[:total_budget]
    active_local_modes = [local_modes[i] for i in active_mode_indices]
    active_global_modes = [candidate_global_modes[i] for i in active_mode_indices]

    counts = balanced_rollout_counts(total_budget, len(active_local_modes))
    mode_ids = np.concatenate([
        np.full(count, mode_index, dtype=np.int64)
        for mode_index, count in enumerate(counts)
    ])

    all_costs = np.zeros(total_budget, dtype=np.float64)
    all_U = np.zeros((total_budget, cfg.horizon, 2), dtype=np.float64)
    best_cost = float("inf")
    best_traj = None

    for mode_index, local_mode in enumerate(active_local_modes):
        ids = np.where(mode_ids == mode_index)[0]
        n = len(ids)
        if n == 0:
            continue

        global_mode = active_global_modes[mode_index]
        key = str(global_mode.signature)
        nominal_bank = build_nominal_bank_for_mode(
            x_current,
            local_mode,
            global_mode,
            goal,
            cfg,
            rng,
            use_empirical_init=use_empirical_init,
            use_mean_nominal=use_mean_nominal,
            previous_idx=progress_by_mode.get(key),
        )

        if rep_type == REP_GAUSSIAN:
            U = sample_gaussian_controls(x_current, local_mode, n, cfg, rng)
        elif rep_type == REP_CONTROL_BANK:
            U = sample_exact_control_bank(
                x_current, global_mode, nominal_bank[0], n, cfg, rng,
                previous_idx=progress_by_mode.get(key),
            )
        else:
            # Corridor prior: one mean-control center with ordinary MPPI noise.
            mean_nominal = nominal_controls_to_track_path(
                x_current, local_mode.mean_path, cfg
            )
            U = sample_controls_from_nominal_bank(
                [mean_nominal], n, cfg, rng, prefer_empirical=False
            )

        if rep_type != REP_CONTROL_BANK:
            U = ensure_direct_goal_prior(U, x_current, goal, cfg)
        X = rollout_ackermann_batch(x_current, U, cfg)
        costs = stable_representation_costs(
            X,
            U,
            local_mode,
            obstacle_circles,
            goal,
            cfg,
            rep_type=rep_type,
            use_mode_prior=use_mode_prior,
        )
        costs = reject_colliding_rollouts(costs, X, obstacle_circles, goal, cfg)
        all_costs[ids] = costs
        all_U[ids] = U

        if record_optimal_traj:
            local_best = int(np.argmin(costs))
            if float(costs[local_best]) < best_cost:
                best_cost = float(costs[local_best])
                best_traj = np.asarray(X[local_best], dtype=np.float64).copy()

    planned_sequence = mppi_weighted_control_sequence(all_costs, all_U, cfg)
    info = {
        "cost_min": float(np.min(all_costs)),
        "cost_mean": float(np.mean(all_costs)),
        "soft_value": float(softmin_score(all_costs, cfg)),
        "rep_type": int(rep_type),
        "rollout_budget_total": total_budget,
        "rollouts_by_mode": counts,
        "active_mode_count": int(len(active_mode_indices)),
        "suppressed_mode_count": int(len(global_modes) - len(active_mode_indices)),
        "nearby_mode_count": int(len(candidate_global_modes)),
        "mode_clearances": mode_clearances.tolist(),
        "optimal_traj": best_traj,
        "planned_control_sequence": planned_sequence,
    }
    return planned_sequence[0].copy(), info, new_progress_by_mode


def make_temporally_correlated_noise(
    n: int,
    H: int,
    cfg: MPPIConfig,
    rng: np.random.Generator,
    *,
    noise_accel: Optional[float] = None,
    noise_steering_rate: Optional[float] = None,
) -> Array:
    noise_scale = np.array([
        cfg.noise_accel if noise_accel is None else float(noise_accel),
        (
            cfg.noise_steering_rate
            if noise_steering_rate is None
            else float(noise_steering_rate)
        ),
    ], dtype=np.float64)
    noise = rng.normal(size=(n, H, 2)) * noise_scale[None, None, :]
    alpha = float(cfg.temporal_noise_smoothing)
    if temporal_smooth_noise_nb is not None:
        return temporal_smooth_noise_nb(noise, alpha)
    for t in range(1, H):
        noise[:, t, :] = (
            alpha * noise[:, t - 1, :] + (1.0 - alpha) * noise[:, t, :]
        )
    return noise


def rollout_collision_mask(
    X: Array,
    obstacle_circles: Sequence[Tuple[Array, float]],
    goal: Array,
    cfg: MPPIConfig,
) -> Array:
    state_batch = np.asarray(X, dtype=np.float64)
    if not obstacle_circles or state_batch.shape[0] == 0:
        return np.zeros(state_batch.shape[0], dtype=bool)
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    goal_array = np.asarray(goal, dtype=np.float64)
    if rollout_collision_mask_nb is not None:
        return rollout_collision_mask_nb(
            state_batch, centers, radii, goal_array,
            float(cfg.vehicle_length), float(cfg.vehicle_width),
            float(cfg.hard_collision_clearance),
            float(cfg.rollout_goal_tolerance),
        )
    points = np.asarray(state_batch[:, 1:, :2], dtype=np.float64)
    collision_by_step = np.zeros(points.shape[:2], dtype=bool)
    for n in range(points.shape[0]):
        for t in range(points.shape[1]):
            clearance = minimum_rectangle_circle_clearance(
                state_batch[n, t + 1],
                centers,
                radii,
                cfg.vehicle_length,
                cfg.vehicle_width,
            )
            collision_by_step[n, t] = (
                clearance < float(cfg.hard_collision_clearance)
            )

    goal_by_step = np.linalg.norm(
        points - np.asarray(goal, dtype=np.float64)[None, None, :], axis=2
    ) <= float(cfg.rollout_goal_tolerance)
    colliding = np.zeros(state_batch.shape[0], dtype=bool)
    for n in range(X.shape[0]):
        reached = np.flatnonzero(goal_by_step[n])
        stop = int(reached[0]) + 1 if len(reached) else points.shape[1]
        colliding[n] = bool(np.any(collision_by_step[n, :stop]))
    return colliding


def reject_colliding_rollouts(
    costs: Array,
    X: Array,
    obstacle_circles: Sequence[Tuple[Array, float]],
    goal: Array,
    cfg: MPPIConfig,
) -> Array:


    if not obstacle_circles or X.shape[0] == 0:
        return costs

    colliding = rollout_collision_mask(X, obstacle_circles, goal, cfg)

    if np.all(colliding):
        return costs

    rejected = np.asarray(costs, dtype=np.float64).copy()
    rejected[colliding] = np.inf
    return rejected


def mppi_weights(costs: Array, cfg: MPPIConfig) -> Array:
    costs = np.asarray(costs, dtype=np.float64)
    finite = np.isfinite(costs)
    if not np.any(finite):
        return np.ones(len(costs), dtype=np.float64) / max(1, len(costs))
    rho = float(np.min(costs[finite]))
    weights = np.zeros_like(costs)
    weights[finite] = np.exp(-(costs[finite] - rho) / cfg.lambda_temperature)
    total = float(weights.sum())
    if total <= 1e-12:
        weights[finite] = 1.0 / float(np.count_nonzero(finite))
    else:
        weights /= total
    return weights


def mppi_weighted_control(costs: Array, U0: Array, cfg: MPPIConfig) -> Array:
    weights = mppi_weights(costs, cfg)
    u = weights @ U0
    u[0] = np.clip(u[0], cfg.accel_min, cfg.accel_max)
    u[1] = np.clip(u[1], cfg.steering_rate_min, cfg.steering_rate_max)
    return u


def mppi_weighted_control_sequence(costs: Array, U: Array, cfg: MPPIConfig) -> Array:
    weights = mppi_weights(costs, cfg)
    sequence = np.tensordot(weights, U, axes=(0, 0))
    sequence[:, 0] = np.clip(sequence[:, 0], cfg.accel_min, cfg.accel_max)
    sequence[:, 1] = np.clip(
        sequence[:, 1], cfg.steering_rate_min, cfg.steering_rate_max
    )
    return np.asarray(sequence, dtype=np.float64)


def update_display_trajectory(
    info: Dict[str, object],
    x_current: Array,
    executed_u: Array,
    goal: Array,
    cfg: MPPIConfig,
) -> None:

    sequence = info.get("planned_control_sequence")
    if sequence is None:
        return
    display_u = np.asarray(sequence, dtype=np.float64).copy()
    if display_u.ndim != 2 or display_u.shape[1] != 2 or len(display_u) == 0:
        return
    display_u[0] = np.asarray(executed_u, dtype=np.float64)
    trajectory = rollout_ackermann(x_current, display_u, cfg)
    distances = np.linalg.norm(
        trajectory[:, :2] - np.asarray(goal, dtype=np.float64)[None, :],
        axis=1,
    )
    reached = np.flatnonzero(distances <= float(cfg.rollout_goal_tolerance))
    if len(reached):
        trajectory = trajectory[: int(reached[0]) + 1]
    info["optimal_traj"] = trajectory


def best_output_trajectory_from_costs(costs: Array, X: Array) -> Array:

    best_idx = int(np.argmin(costs))
    return np.asarray(X[best_idx], dtype=np.float64).copy()


def standard_mppi_step(
    x_current: Array,
    obstacles: Sequence,
    goal: Array,
    cfg: MPPIConfig,
    rng: np.random.Generator,
    *,
    obstacle_circles: Optional[List[Tuple[Array, float]]] = None,
    record_optimal_traj: bool = True,
) -> Tuple[Array, Dict[str, object]]:
    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)
    U_nom = nominal_controls_to_goal(x_current, goal, cfg)

    U = np.repeat(U_nom[None, :, :], cfg.num_rollouts, axis=0)
    noise = make_temporally_correlated_noise(cfg.num_rollouts, cfg.horizon, cfg, rng)
    U += noise
    k = min(max(1, int(cfg.low_noise_proposal_count)), cfg.num_rollouts)
    U[:k] = U_nom[None, :, :] + float(cfg.low_noise_proposal_scale) * noise[:k]
    U[0] = U_nom
    U = enforce_ackermann_control_bounds(U, cfg)

    X = rollout_ackermann_batch(x_current, U, cfg)
    costs = standard_mppi_costs_batch(X, U, obstacle_circles, goal, cfg)
    costs = reject_colliding_rollouts(costs, X, obstacle_circles, goal, cfg)

    planned_sequence = mppi_weighted_control_sequence(costs, U, cfg)
    u = planned_sequence[0].copy()
    return u, {
        "cost_min": float(costs.min()),
        "cost_mean": float(costs.mean()),
        "optimal_traj": (
            best_output_trajectory_from_costs(costs, X)
            if record_optimal_traj else None
        ),
        "planned_control_sequence": planned_sequence,
    }


def build_default_scene():
    scale = 4.0
    bounds_xy = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    bounds_ranges = ((0.0, 10.0), (0.0, 10.0))
    start = np.array([1.0, 1.0], dtype=np.float64)
    goal = np.array([9.0, 9.0], dtype=np.float64)

    obstacles = [
        PolyObstacle(round_obstacle(np.array([[3.0, 1.5], [5.2, 2.2], [4.7, 4.0], [2.8, 3.4]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[6.2, 6.0], [8.5, 6.3], [8.1, 8.4], [6.8, 8.9], [5.9, 7.4]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[1.8, 4.2], [2.7, 4.0], [3.0, 4.8], [2.3, 5.3], [1.7, 4.9]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[4.6, 5.1], [5.4, 5.0], [5.8, 5.7], [5.0, 6.2], [4.4, 5.7]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[7.9, 3.0], [9.0, 3.2], [8.8, 4.2], [7.7, 4.0]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[5.7, 1.0], [6.6, 1.2], [6.4, 2.3], [5.6, 2.1]]), n_iters=4, n_points=32)),
    ]


    return scale, bounds_xy, bounds_ranges, start, goal, obstacles


def run_swarm_planner(start, goal, obstacles, scale, bounds_xy, *, seed=3):
    segs = obstacles_to_segs(obstacles, scale=scale)
    base_action = pickle.load(open("save/policy.pkl", "rb"))["best_theta"]

    graph_goals, graph_W = build_full_graph(
        obstacles=obstacles,
        start=start,
        goal=goal,
        scale=scale,
        bounds=bounds_xy,
    )

    planner = HomotopyAwareGenerativePlanner(
        env_cls=FishGoalEnv2D,
        action=base_action,
        obstacles=obstacles,
        segs=segs,
        scale=scale,
        boid_count=1200,
        max_steps=700,
        dt=0.5,
    )

    return planner.sample(
        start_unscaled=start,
        goal_unscaled=goal,
        graph_goals=graph_goals,
        graph_W=graph_W,
        seed=seed,
    )


def initial_pose(start: Array, goal: Array) -> Array:

    direction = goal - start
    heading = math.atan2(direction[1], direction[0])
    return np.array([
        start[0], start[1], heading, 0.0, 0.0, 0.0, 0.0
    ], dtype=np.float64)


def ensure_direct_goal_prior(
    U: Array,
    x_current: Array,
    goal: Array,
    cfg: MPPIConfig,
) -> Array:


    proposals = np.asarray(U, dtype=np.float64)
    if proposals.ndim != 3 or proposals.shape[0] == 0:
        return proposals
    proposals[-1] = nominal_controls_to_goal(x_current, goal, cfg)
    return enforce_ackermann_control_bounds(proposals, cfg)


def run_controller_variant(
    variant: ControllerVariant,
    modes: List[MPPIHomotopyMode],
    obstacles: Sequence,
    start: Array,
    goal: Array,
    *,
    seed: int,
    max_steps: int = 200,
    goal_tolerance: float = 0.15,
    mppi_cfg: Optional[MPPIConfig] = None,
    record_infos: bool = True,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    mppi_cfg = MPPIConfig() if mppi_cfg is None else mppi_cfg
    effective_goal_tolerance = float(goal_tolerance) + max(
        0.0, float(getattr(mppi_cfg, "goal_acceptance_epsilon", 0.0))
    )

    mppi_cfg.rollout_goal_tolerance = effective_goal_tolerance

    x = initial_pose(start, goal)
    states = [x.copy()]
    controls = []
    infos = []
    previous_control = None
    reached_goal = goal_pose_satisfied(x, goal, effective_goal_tolerance, mppi_cfg)
    arrival_step = 0 if reached_goal else None

    swarm_progress = {}
    obstacle_circles = obstacle_bounding_circles(obstacles)

    t0 = time.perf_counter()

    for _ in range(max_steps):
        step_cfg = mppi_cfg
        if variant == ControllerVariant.GAUSSIAN_PRIOR_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, obstacles, goal, step_cfg, rng,
                rep_type=REP_GAUSSIAN,
                use_empirical_init=False,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
                obstacle_circles=obstacle_circles,
                record_optimal_traj=record_infos,
            )

        elif variant == ControllerVariant.CORRIDOR_PRIOR_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, obstacles, goal, step_cfg, rng,
                rep_type=REP_CORRIDOR,
                use_empirical_init=False,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
                obstacle_circles=obstacle_circles,
                record_optimal_traj=record_infos,
            )

        elif variant == ControllerVariant.CONTROL_BANK_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, obstacles, goal, step_cfg, rng,
                rep_type=REP_CONTROL_BANK,
                use_empirical_init=True,
                use_mean_nominal=False,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
                obstacle_circles=obstacle_circles,
                record_optimal_traj=record_infos,
            )

        elif variant == ControllerVariant.STANDARD_MPPI:
            u, info = standard_mppi_step(
                x, obstacles, goal, step_cfg, rng,
                obstacle_circles=obstacle_circles,
                record_optimal_traj=record_infos,
            )

        elif variant == ControllerVariant.STANDARD_MPPI_128:
            variant_cfg = replace(step_cfg, num_rollouts=128)
            u, info = standard_mppi_step(
                x, obstacles, goal, variant_cfg, rng,
                obstacle_circles=obstacle_circles,
                record_optimal_traj=record_infos,
            )

        else:
            raise ValueError(f"Unknown variant {variant}")

        if "mppi" in variant.value:


            u = apply_smooth_safe_control(
                x, u, previous_control, obstacle_circles, step_cfg
            )

        if record_infos and isinstance(info, dict):
            update_display_trajectory(info, x, u, goal, step_cfg)

        previous_control = u.copy()
        x_next = ackermann_step(x, u, step_cfg)

        arrived = goal_pose_satisfied(x_next, goal, effective_goal_tolerance, step_cfg)
        x = x_next

        states.append(x.copy())
        controls.append(u.copy())
        if record_infos:
            infos.append(info)

        if arrived:
            reached_goal = True
            arrival_step = len(states) - 1
            break

    runtime = time.perf_counter() - t0

    return {
        "variant": variant.value,
        "seed": seed,
        "states": np.asarray(states),
        "controls": np.asarray(controls),
        "infos": infos,
        "runtime": runtime,
        "reached_goal": bool(reached_goal),
        "arrival_step": arrival_step,
    }


def summarize_result(result: Dict[str, object], obstacles, goal, robot_radius: float, goal_tolerance: float = 0.35, vehicle_length: float = 0.81, vehicle_width: float = 0.36):
    states = result["states"]
    controls = result["controls"]
    final_dist = float(np.linalg.norm(states[-1, :2] - goal))
    collision = path_collided(states, obstacles, robot_radius, vehicle_length, vehicle_width)

    reached_goal = bool(result.get(
        "reached_goal", final_dist <= goal_tolerance + 1e-9
    ))
    return {
        "variant": result["variant"],
        "seed": result["seed"],
        "success": bool(reached_goal and not collision),
        "reached_goal": reached_goal,
        "collision": bool(collision),
        "final_dist": final_dist,
        "min_clearance": min_clearance(states, obstacles, robot_radius, vehicle_length, vehicle_width),
        "path_length": path_length(states),
        "control_effort": control_effort(controls),
        "control_smoothness": control_smoothness(controls),
        "steps": int(len(states) - 1),
        "runtime_sec": float(result["runtime"]),
        "runtime_per_step_sec": float(result["runtime"] / max(1, len(states) - 1)),
    }


def setup_workspace(ax, obstacles, start, goal, bounds, title=None):
    xmin, xmax, ymin, ymax = normalize_plot_bounds(bounds)

    for obs in obstacles:
        p = _poly_vertices(obs)
        ax.fill(p[:, 0], p[:, 1], alpha=0.25)
        ax.plot(np.r_[p[:, 0], p[0, 0]], np.r_[p[:, 1], p[0, 1]], linewidth=1.0)

    ax.scatter([start[0]], [start[1]], s=80, marker="o", label="start")
    ax.scatter([goal[0]], [goal[1]], s=140, marker="*", label="goal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)


def plot_paths(results: List[Dict[str, object]], obstacles, start, goal, bounds, save_path: str):
    fig, ax = plt.subplots(figsize=(9, 9))
    setup_workspace(ax, obstacles, start, goal, bounds, "Controller comparison paths")

    for res in results:
        states = res["states"]
        ax.plot(states[:, 0], states[:, 1], linewidth=2.0, label=f"{res['variant']} seed={res['seed']}")

    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")


def obstacle_center(obs) -> Array:

    return _poly_vertices(obs).mean(axis=0)


def translate_obstacle_to_center(obs, target_center: Array):

    vertices = _poly_vertices(obs).copy()
    shift = np.asarray(target_center, dtype=np.float64) - vertices.mean(axis=0)
    return PolyObstacle(vertices + shift[None, :])


def random_obstacle_center_swap(
    obstacles: Sequence,
    *,
    seed: int,
) -> Tuple[List[object], Tuple[int, ...]]:


    n = len(obstacles)
    if n < 2:
        return list(obstacles), tuple(range(n))

    rng = np.random.default_rng(int(seed))
    permutation = np.arange(n, dtype=np.int64)


    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))
        permutation[i], permutation[j] = permutation[j], permutation[i]

    original_centers = [obstacle_center(obs) for obs in obstacles]
    swapped = [
        translate_obstacle_to_center(obs, original_centers[int(permutation[i])])
        for i, obs in enumerate(obstacles)
    ]
    return swapped, tuple(int(v) for v in permutation)


def obstacle_center_permutation_text(permutation: Sequence[int]) -> str:

    return ";".join(f"{i}->{int(target)}" for i, target in enumerate(permutation))


def make_wall_between_points(p0: Array, p1: Array, width: float = 0.35, extension: float = 0.0):


    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)

    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L <= 1e-12:
        raise ValueError("Cannot create wall: endpoints are identical.")

    u = d / L
    n = np.array([-u[1], u[0]], dtype=np.float64)

    a = p0 - extension * u
    b = p1 + extension * u
    half = 0.5 * float(width)

    vertices = np.array([
        a + half * n,
        b + half * n,
        b - half * n,
        a - half * n,
    ], dtype=np.float64)

    return PolyObstacle(vertices)


def make_wall_between_obstacles(
    obstacles: Sequence,
    idx_a: int,
    idx_b: int,
    width: float = 0.35,
    extension: float = 0.15,
):


    c0 = obstacle_center(obstacles[idx_a])
    c1 = obstacle_center(obstacles[idx_b])
    return make_wall_between_points(c0, c1, width=width, extension=extension)


def make_wall_blockers_between_obstacles(
    obstacles: Sequence,
    pairs: Sequence[Tuple[int, int]],
    width: float = 0.35,
    extension: float = 0.15,
):


    centers = [obstacle_center(obs).copy() for obs in obstacles]
    return make_wall_blockers_between_centers(
        centers=centers,
        pairs=pairs,
        width=width,
        extension=extension,
    )


def make_wall_blockers_between_centers(
    centers: Sequence[Array],
    pairs: Sequence[Tuple[int, int]],
    width: float = 0.35,
    extension: float = 0.15,
):


    fixed_centers = [
        np.asarray(center, dtype=np.float64).reshape(2).copy()
        for center in centers
    ]

    blockers = []
    for i, j in pairs:
        if i == j:
            raise ValueError(f"Cannot create wall for degenerate center pair {(i, j)}.")
        if not (0 <= i < len(fixed_centers) and 0 <= j < len(fixed_centers)):
            raise IndexError(
                f"Center pair {(i, j)} is outside the valid index range "
                f"[0, {len(fixed_centers) - 1}]."
            )
        blockers.append(
            make_wall_between_points(
                fixed_centers[i],
                fixed_centers[j],
                width=width,
                extension=extension,
            )
        )
    return blockers


def as_blocker_list(blocker_or_blockers):
    if blocker_or_blockers is None:
        return []
    if isinstance(blocker_or_blockers, (list, tuple)):
        return list(blocker_or_blockers)
    return [blocker_or_blockers]


def active_obstacles_for_step(base_obstacles, blocker, step, block_step):

    if step >= block_step:
        return list(base_obstacles) + as_blocker_list(blocker)
    return list(base_obstacles)


def spatial_progress_along_start_goal(x: Array, start: Array, goal: Array) -> float:

    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    position = np.asarray(x[:2], dtype=np.float64)
    direction = goal - start
    denom = float(direction @ direction)
    if denom <= 1e-12:
        return 1.0
    return float(np.clip(((position - start) @ direction) / denom, 0.0, 1.0))


def active_obstacles_for_state(
    base_obstacles: Sequence,
    blocker,
    state_index: int,
    activation_step: Optional[int],
):
    if activation_step is not None and state_index >= activation_step:
        return list(base_obstacles) + as_blocker_list(blocker)
    return list(base_obstacles)


def run_dynamic_blockage_controller(
    variant: ControllerVariant,
    modes: List[MPPIHomotopyMode],
    base_obstacles: Sequence,
    blocker,
    start: Array,
    goal: Array,
    *,
    seed: int,
    trigger_progress: Optional[float] = 0.35,
    activation_preview_clearance: Optional[float] = None,
    blocker_active_from_start: bool = False,
    condition: str = "dynamic_wall",
    block_step: Optional[int] = None,
    max_steps: int = 200,
    goal_tolerance: float = 0.35,
    mppi_cfg: Optional[MPPIConfig] = None,
    record_infos: bool = True,
    record_obstacle_history: bool = True,
):


    rng = np.random.default_rng(seed)
    mppi_cfg = MPPIConfig() if mppi_cfg is None else mppi_cfg
    effective_goal_tolerance = float(goal_tolerance) + max(
        0.0, float(getattr(mppi_cfg, "goal_acceptance_epsilon", 0.0))
    )

    mppi_cfg.rollout_goal_tolerance = effective_goal_tolerance

    x = initial_pose(start, goal)
    states = [x.copy()]
    controls = []
    infos = []
    obstacle_history = []
    previous_control = None
    reached_goal = goal_pose_satisfied(x, goal, effective_goal_tolerance, mppi_cfg)
    arrival_step = 0 if reached_goal else None
    swarm_progress = {}
    blockers = as_blocker_list(blocker)

    activation_step: Optional[int] = 0 if blocker_active_from_start else None
    activation_progress: Optional[float] = 0.0 if blocker_active_from_start else None
    activation_reason: Optional[str] = "from_start" if blocker_active_from_start else None
    activation_clearance: Optional[float] = None
    if blocker_active_from_start and blockers:
        activation_clearance = min_clearance(
            x[None, :],
            blockers,
            mppi_cfg.robot_radius,
            mppi_cfg.vehicle_length,
            mppi_cfg.vehicle_width,
        )


    base_obstacle_circles = obstacle_bounding_circles(base_obstacles)
    blocked_obstacles = list(base_obstacles) + blockers
    blocked_obstacle_circles = (
        obstacle_bounding_circles(blocked_obstacles)
        if blockers else base_obstacle_circles
    )

    t0 = time.perf_counter()

    for step in range(max_steps):
        step_cfg = mppi_cfg
        current_progress = spatial_progress_along_start_goal(x, start, goal)

        if activation_step is None and blockers:
            blocker_clearance = min_clearance(
                x[None, :],
                blockers,
                step_cfg.robot_radius,
                step_cfg.vehicle_length,
                step_cfg.vehicle_width,
            )
            progress_ready = bool(
                trigger_progress is not None
                and current_progress >= float(trigger_progress)
            )
            clearance_ready = bool(
                activation_preview_clearance is not None
                and blocker_clearance <= float(activation_preview_clearance)
            )
            legacy_step_ready = bool(
                trigger_progress is None
                and block_step is not None
                and step >= int(block_step)
            )

            if progress_ready or clearance_ready or legacy_step_ready:
                activation_step = step
                activation_progress = current_progress
                activation_clearance = blocker_clearance
                if clearance_ready and progress_ready:
                    activation_reason = "progress_and_clearance"
                elif clearance_ready:
                    activation_reason = "clearance_preview"
                elif progress_ready:
                    activation_reason = "progress"
                else:
                    activation_reason = "legacy_step"

        active_obstacles = active_obstacles_for_state(
            base_obstacles,
            blocker,
            step,
            activation_step,
        )
        active_obstacle_circles = (
            blocked_obstacle_circles
            if activation_step is not None and step >= activation_step
            else base_obstacle_circles
        )
        if record_obstacle_history:
            obstacle_history.append(active_obstacles)

        if variant == ControllerVariant.GAUSSIAN_PRIOR_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, active_obstacles, goal, step_cfg, rng,
                rep_type=REP_GAUSSIAN,
                use_empirical_init=False,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
                obstacle_circles=active_obstacle_circles,
                record_optimal_traj=record_infos,
            )

        elif variant == ControllerVariant.CORRIDOR_PRIOR_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, active_obstacles, goal, step_cfg, rng,
                rep_type=REP_CORRIDOR,
                use_empirical_init=False,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
                obstacle_circles=active_obstacle_circles,
                record_optimal_traj=record_infos,
            )

        elif variant == ControllerVariant.CONTROL_BANK_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, active_obstacles, goal, step_cfg, rng,
                rep_type=REP_CONTROL_BANK,
                use_empirical_init=True,
                use_mean_nominal=False,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
                obstacle_circles=active_obstacle_circles,
                record_optimal_traj=record_infos,
            )

        elif variant == ControllerVariant.STANDARD_MPPI:
            u, info = standard_mppi_step(
                x, active_obstacles, goal, step_cfg, rng,
                obstacle_circles=active_obstacle_circles,
                record_optimal_traj=record_infos,
            )

        elif variant == ControllerVariant.STANDARD_MPPI_128:
            variant_cfg = replace(step_cfg, num_rollouts=128)
            u, info = standard_mppi_step(
                x, active_obstacles, goal, variant_cfg, rng,
                obstacle_circles=active_obstacle_circles,
                record_optimal_traj=record_infos,
            )

        else:
            raise ValueError(f"Unsupported variant: {variant}")

        if "mppi" in variant.value:


            u = apply_smooth_safe_control(
                x, u, previous_control, active_obstacle_circles, step_cfg
            )

        if record_infos and isinstance(info, dict):
            update_display_trajectory(info, x, u, goal, step_cfg)

        previous_control = u.copy()
        x_next = ackermann_step(x, u, step_cfg)

        arrived = goal_pose_satisfied(x_next, goal, effective_goal_tolerance, step_cfg)
        x = x_next
        states.append(x.copy())
        controls.append(u.copy())


        if record_infos:
            infos.append(info)

        if arrived:
            reached_goal = True
            arrival_step = len(states) - 1
            break

    runtime = time.perf_counter() - t0

    if record_obstacle_history:
        obstacle_history.append(active_obstacles_for_state(
            base_obstacles,
            blocker,
            len(states) - 1,
            activation_step,
        ))

    legacy_block_step = activation_step if activation_step is not None else max_steps + 1
    return {
        "variant": variant.value,
        "seed": seed,
        "condition": condition,
        "states": np.asarray(states),
        "controls": np.asarray(controls),
        "infos": infos,
        "runtime": runtime,
        "block_step": legacy_block_step,
        "activation_step": activation_step,
        "activation_progress": activation_progress,
        "activation_reason": activation_reason,
        "activation_clearance": activation_clearance,
        "trigger_progress": trigger_progress,
        "activation_preview_clearance": activation_preview_clearance,
        "blocker": blocker,
        "obstacle_history": obstacle_history,
        "reached_goal": bool(reached_goal),
        "arrival_step": arrival_step,
    }


def summarize_dynamic_result(result, base_obstacles, blocker, goal, robot_radius, goal_tolerance=0.15, vehicle_length: float = 0.81, vehicle_width: float = 0.36):

    states = result["states"]
    controls = result["controls"]
    condition = str(result.get("condition", "dynamic_wall"))
    activation_step = result.get("activation_step")
    if activation_step is not None:
        activation_step = int(activation_step)


    if condition == "static_wall":
        metric_start_step: Optional[int] = 0
    elif condition == "dynamic_wall":
        metric_start_step = activation_step
    else:
        metric_start_step = None

    min_vals = []
    min_vals_after_block = []
    collision = False
    first_collision_step = None

    collision_substeps = 5
    for step in range(len(states)):
        active_obs = active_obstacles_for_state(
            base_obstacles,
            blocker,
            step,
            activation_step,
        )
        clearance = min_clearance(states[step:step + 1], active_obs, robot_radius, vehicle_length, vehicle_width)
        min_vals.append(clearance)
        if metric_start_step is not None and step >= metric_start_step:
            min_vals_after_block.append(clearance)
        if clearance < 0.0 and first_collision_step is None:
            collision = True
            first_collision_step = step

        if step + 1 < len(states):
            segment_obs = active_obstacles_for_state(
                base_obstacles,
                blocker,
                step,
                activation_step,
            )
            alpha = np.linspace(0.0, 1.0, collision_substeps + 2)[1:-1, None]
            segment_states = states[step][None, :] + alpha * (
                states[step + 1][None, :] - states[step][None, :]
            )
            segment_clearance = min_clearance(segment_states, segment_obs, robot_radius, vehicle_length, vehicle_width)
            min_vals.append(segment_clearance)
            if metric_start_step is not None and step >= metric_start_step:
                min_vals_after_block.append(segment_clearance)
            if segment_clearance < 0.0 and first_collision_step is None:
                collision = True
                first_collision_step = step

    final_dist = float(np.linalg.norm(states[-1, :2] - goal))
    reached_goal = bool(result.get(
        "reached_goal", final_dist <= goal_tolerance + 1e-9
    ))
    success = bool(reached_goal and not collision)

    if success:
        failure_reason = ""
    elif collision:
        failure_reason = "collision"
    else:
        failure_reason = "not_reaching"


    if metric_start_step is None:
        after_block_state_start = len(states)
        after_block_control_start = len(controls)
        steps_after_block = 0
    else:
        after_block_state_start = min(metric_start_step, len(states) - 1)
        after_block_control_start = min(metric_start_step, len(controls))
        steps_after_block = int(max(0, len(states) - 1 - metric_start_step))

    exposed_to_blocker = bool(
        condition == "static_wall"
        or (
            condition == "dynamic_wall"
            and activation_step is not None
            and len(states) - 1 >= activation_step
        )
    )

    return {
        "variant": result["variant"],
        "seed": result["seed"],
        "condition": condition,
        "success": success,
        "failure_reason": failure_reason,
        "reached_goal": reached_goal,
        "collision": bool(collision),
        "not_reaching": bool(not reached_goal and not collision),
        "first_collision_step": first_collision_step,
        "collision_after_block": bool(
            metric_start_step is not None
            and first_collision_step is not None
            and first_collision_step >= metric_start_step
        ),
        "exposed_to_blocker": exposed_to_blocker,
        "goal_reached_before_block": bool(
            condition == "dynamic_wall"
            and reached_goal
            and activation_step is None
        ),
        "activation_step": activation_step,
        "activation_progress": result.get("activation_progress"),
        "activation_reason": result.get("activation_reason"),
        "activation_clearance": result.get("activation_clearance"),
        "trigger_progress": result.get("trigger_progress"),
        "activation_preview_clearance": result.get("activation_preview_clearance"),
        "final_dist": final_dist,
        "min_clearance_dynamic": float(np.min(min_vals)) if min_vals else float("inf"),
        "min_clearance_after_block": (
            float(np.min(min_vals_after_block)) if min_vals_after_block else float("nan")
        ),
        "path_length": path_length(states),
        "path_length_after_block": (
            path_length(states[after_block_state_start:])
            if metric_start_step is not None else float("nan")
        ),
        "control_effort": control_effort(controls),
        "control_effort_after_block": (
            control_effort(controls[after_block_control_start:])
            if metric_start_step is not None else float("nan")
        ),
        "control_smoothness": control_smoothness(controls),
        "control_smoothness_after_block": (
            control_smoothness(controls[after_block_control_start:])
            if metric_start_step is not None else float("nan")
        ),
        "steps": int(len(states) - 1),
        "steps_after_block": steps_after_block,
        "runtime_sec": float(result["runtime"]),
        "runtime_per_step_sec": float(result["runtime"] / max(1, len(states) - 1)),
        "block_step": activation_step,
        "goal_tolerance": float(goal_tolerance),
    }


def make_variant_color_map(results):


    variant_names = sorted({res["variant"] for res in results})
    n = len(variant_names)

    if n <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(n)]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / max(1, n)) for i in range(n)]

    return {
        name: colors[i]
        for i, name in enumerate(variant_names)
    }


def plot_dynamic_paths(results, base_obstacles, blocker, start, goal, bounds, save_path):
    fig, ax = plt.subplots(figsize=(11, 9))
    setup_workspace(ax, base_obstacles, start, goal, bounds, "Dynamic blockage: paths after sudden wall closure")

    color_by_variant = make_variant_color_map(results)


    for bi, bobs in enumerate(as_blocker_list(blocker)):
        p = _poly_vertices(bobs)
        label = "dynamic wall blocker" if bi == 0 else None
        ax.fill(p[:, 0], p[:, 1], alpha=0.55, label=label)
        ax.plot(
            np.r_[p[:, 0], p[0, 0]],
            np.r_[p[:, 1], p[0, 1]],
            linewidth=2.0,
        )

    for res in results:
        states = res["states"]
        b = min(res["block_step"], len(states) - 1)
        color = color_by_variant[res["variant"]]


        ax.plot(
            states[:b+1, 0],
            states[:b+1, 1],
            linewidth=2.2,
            linestyle="--",
            color=color,
            label=res["variant"],
        )


        ax.plot(
            states[b:, 0],
            states[b:, 1],
            linewidth=3.0,
            linestyle="-",
            color=color,
        )


        if b < len(states):
            ax.scatter(
                [states[b, 0]],
                [states[b, 1]],
                s=60,
                marker="x",
                color=color,
            )

    ax.legend(
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig, ax


def safe_variant_filename(name: str) -> str:

    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def tracked_reference_for_frame(
    result: Dict[str, object],
    frame: int,
    modes: List[MPPIHomotopyMode],
    goal: Array,
    *,
    mppi_cfg: Optional[MPPIConfig] = None,
) -> Optional[Array]:


    if not modes:
        return None

    mppi_cfg = MPPIConfig() if mppi_cfg is None else mppi_cfg

    states = result["states"]
    j = min(frame, len(states) - 1)
    x = states[j]
    variant_name = str(result.get("variant", ""))

    if variant_name in {ControllerVariant.STANDARD_MPPI.value, ControllerVariant.STANDARD_MPPI_128.value}:
        return np.vstack([x[:2], np.asarray(goal, dtype=np.float64)])

    d = [float(np.min(np.linalg.norm(mode.mean_path - x[:2], axis=1))) for mode in modes]
    idx = int(np.argmin(d))

    H = mppi_cfg.horizon
    return localize_mode_for_state(modes[idx], x, H).mean_path


def optimal_trajectory_for_frame(result: Dict[str, object], frame: int) -> Optional[Array]:

    infos = result.get("infos", [])
    if not infos:
        return None
    info = infos[min(frame, len(infos) - 1)]
    if not isinstance(info, dict):
        return None
    traj = info.get("optimal_traj")
    if traj is None:
        return None
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[0] < 2 or traj.shape[1] < 2:
        return None
    return traj


def draw_control_input_arrows(
    ax,
    states: Array,
    controls: Array,
    upto_step: int,
    color,
    *,
    dt: float,
    stride: int = 1,
    min_length: float = 0.08,
    max_length: float = 0.35,
):

    if controls is None or len(controls) == 0:
        return
    last = min(int(upto_step), len(controls) - 1, len(states) - 1)
    if last < 0:
        return
    for k in range(0, last + 1, max(1, int(stride))):
        x = states[k]
        speed = float(x[3]) if len(x) > 3 else 0.0
        if abs(speed) < 1e-9:
            length = min_length
            direction_sign = 1.0
        else:
            length = float(np.clip(abs(speed) * dt, min_length, max_length))
            direction_sign = 1.0 if speed >= 0.0 else -1.0
        dx = direction_sign * length * math.cos(float(x[2]))
        dy = direction_sign * length * math.sin(float(x[2]))
        ax.arrow(
            float(x[0]), float(x[1]), dx, dy,
            head_width=0.07, head_length=0.10,
            length_includes_head=True, linewidth=0.8,
            color=color, alpha=0.45,
        )


def animate_dynamic_blockage_one_variant(
    result,
    all_results,
    modes,
    base_obstacles,
    blocker,
    start,
    goal,
    bounds,
    save_path,
    *,
    mppi_cfg: Optional[MPPIConfig] = None,
    fps: int = 8,
):


    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as exc:
        print(f"Could not import animation tools: {exc}")
        return

    mppi_cfg = MPPIConfig() if mppi_cfg is None else mppi_cfg

    xmin, xmax, ymin, ymax = normalize_plot_bounds(bounds)
    color_by_variant = make_variant_color_map(all_results)
    color = color_by_variant.get(result["variant"], "C0")
    states = result["states"]
    block_step = int(result["block_step"])

    fig, ax = plt.subplots(figsize=(9, 8))

    def update(frame):
        ax.clear()

        for obs in base_obstacles:
            p = _poly_vertices(obs)
            ax.fill(p[:, 0], p[:, 1], alpha=0.25)
            ax.plot(
                np.r_[p[:, 0], p[0, 0]],
                np.r_[p[:, 1], p[0, 1]],
                linewidth=1.0,
            )

        if frame >= block_step:
            for bi, bobs in enumerate(as_blocker_list(blocker)):
                p = _poly_vertices(bobs)
                label = "dynamic wall blocker" if bi == 0 else None
                ax.fill(p[:, 0], p[:, 1], alpha=0.60, label=label)
                ax.plot(
                    np.r_[p[:, 0], p[0, 0]],
                    np.r_[p[:, 1], p[0, 1]],
                    linewidth=2.0,
                )

        j = min(frame, len(states) - 1)
        ref = tracked_reference_for_frame(
            result,
            j,
            modes,
            goal,
            mppi_cfg=mppi_cfg,
        )
        if ref is not None and len(ref) >= 2:
            ax.plot(
                ref[:, 0],
                ref[:, 1],
                linestyle=":",
                linewidth=2.5,
                color=color,
                alpha=0.85,
                label="tracked reference",
            )
            ax.scatter(
                [ref[-1, 0]],
                [ref[-1, 1]],
                s=35,
                marker="^",
                color=color,
                alpha=0.85,
            )

        opt = optimal_trajectory_for_frame(result, j)
        if opt is not None and len(opt) >= 2:
            ax.plot(
                opt[:, 0],
                opt[:, 1],
                linestyle="-.",
                linewidth=2.0,
                color=color,
                alpha=0.70,
                label="optimal horizon trajectory",
            )

        dt_for_arrows = mppi_cfg.dt
        draw_control_input_arrows(
            ax,
            states,
            result.get("controls", np.zeros((0, 2), dtype=np.float64)),
            j,
            color,
            dt=0.12,
            stride=1,
        )

        ax.plot(
            states[:j+1, 0],
            states[:j+1, 1],
            linewidth=3.0,
            color=color,
            label=result["variant"],
        )
        ax.scatter([states[j, 0]], [states[j, 1]], s=55, color=color)

        if block_step < len(states):
            ax.scatter(
                [states[block_step, 0]],
                [states[block_step, 1]],
                s=55,
                marker="x",
                color=color,
                label="blockage-time state",
            )

        ax.scatter([start[0]], [start[1]], s=80, marker="o", label="start")
        ax.scatter([goal[0]], [goal[1]], s=140, marker="*", label="goal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{result['variant']} with tracked reference, step {j}")
        ax.legend(
            fontsize=7,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=True,
        )

    anim = FuncAnimation(fig, update, frames=len(states), interval=int(1000 / fps), repeat=False)

    try:
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print(f"Saved per-variant animation: {save_path}")
    except Exception as exc:
        print(f"Could not save per-variant GIF animation {save_path}: {exc}")
    finally:
        plt.close(fig)


def animate_dynamic_blockage_by_variant(
    results,
    modes,
    base_obstacles,
    blocker,
    start,
    goal,
    bounds,
    output_dir="dynamic_block_soft",
    *,
    mppi_cfg: Optional[MPPIConfig] = None,
):

    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for res in results:
        filename = f"{safe_variant_filename(res['variant'])}.gif"
        animate_dynamic_blockage_one_variant(
            res,
            results,
            modes,
            base_obstacles,
            blocker,
            start,
            goal,
            bounds,
            str(out_dir / filename),
            mppi_cfg=mppi_cfg,
        )

def animate_dynamic_blockage(results, base_obstacles, blocker, start, goal, bounds, save_path):


    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as exc:
        print(f"Could not import animation tools: {exc}")
        return

    max_len = max(len(r["states"]) for r in results)
    xmin, xmax, ymin, ymax = normalize_plot_bounds(bounds)

    color_by_variant = make_variant_color_map(results)

    fig, ax = plt.subplots(figsize=(9, 8))

    def update(frame):
        ax.clear()

        for obs in base_obstacles:
            p = _poly_vertices(obs)
            ax.fill(p[:, 0], p[:, 1], alpha=0.25)
            ax.plot(
                np.r_[p[:, 0], p[0, 0]],
                np.r_[p[:, 1], p[0, 1]],
                linewidth=1.0,
            )


        block_step = min(r["block_step"] for r in results)
        if frame >= block_step:
            for bi, bobs in enumerate(as_blocker_list(blocker)):
                p = _poly_vertices(bobs)
                label = "dynamic wall blocker" if bi == 0 else None
                ax.fill(p[:, 0], p[:, 1], alpha=0.60, label=label)
                ax.plot(
                    np.r_[p[:, 0], p[0, 0]],
                    np.r_[p[:, 1], p[0, 1]],
                    linewidth=2.0,
                )

        ax.scatter([start[0]], [start[1]], s=80, marker="o", label="start")
        ax.scatter([goal[0]], [goal[1]], s=140, marker="*", label="goal")

        for res in results:
            states = res["states"]
            j = min(frame, len(states) - 1)
            color = color_by_variant[res["variant"]]

            ax.plot(
                states[:j+1, 0],
                states[:j+1, 1],
                linewidth=2.5,
                color=color,
                label=res["variant"],
            )
            ax.scatter(
                [states[j, 0]],
                [states[j, 1]],
                s=45,
                color=color,
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"Dynamic wall-blockage simulation, step {frame}")
        ax.legend(
            fontsize=7,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=True,
        )

    anim = FuncAnimation(fig, update, frames=max_len, interval=120, repeat=False)

    try:
        anim.save(save_path, writer=PillowWriter(fps=8))
        print(f"Saved animation: {save_path}")
    except Exception as exc:
        print(f"Could not save GIF animation: {exc}")
    finally:
        plt.close(fig)

def main_dynamic_blockage():
    print("dynamic_block_soft")
    print(f"Numba enabled: {njit is not None}")

    scale, bounds_xy, bounds_ranges, start, goal, base_obstacles = build_default_scene()

    print("Running initial swarm planner without dynamic blocker...")
    gen_out = run_swarm_planner(
        start=start,
        goal=goal,
        obstacles=base_obstacles,
        scale=scale,
        bounds_xy=bounds_xy,
        seed=RUN_SWARM_SEED,
    )

    print(f"Generated swarm trajectories: {len(gen_out.samples)}")
    print(f"Homotopy groups: {len(gen_out.homotopy_groups)}")

    print("Fitting Gaussian homotopy mixture...")
    mixture = fit_topological_trajectory_mixture(
        gen_out,
        base_obstacles,
        K=50,
        beta=1.0,
        min_mode_samples=3,
        covariance_jitter=2e-4,
        bounds=bounds_ranges,
        goal=goal,
        snap_to_goal_radius=0.2,
        snap_straight_tail_points=8,
    )
    modes = mixture_to_mppi_modes(mixture)

    print("Modes:")
    for i, m in enumerate(modes):
        print(f"  h{i}: sig={m.signature}, pi={m.probability:.3f}, samples={len(m.sample_paths or [])}")


    wall_pairs = []
    wall_width = 0.40
    wall_extension = 0.20
    block_step = 150
    seed = 2


    blocker = make_wall_blockers_between_obstacles(
        obstacles=base_obstacles,
        pairs=wall_pairs,
        width=wall_width,
        extension=wall_extension,
    )

    cfg = MPPIConfig(
        horizon=50,
        num_rollouts=64,
        dt=0.12,
        v_min=-1.0,
        v_max=2.8,
        lambda_temperature=2.2,
        noise_accel=0.90,
        noise_steering_rate=1.00,
        temporal_noise_smoothing=0.72,
        w_goal=110.0,
        w_obstacle=500.0,
        w_control=0.004,
        max_empirical_nominals_per_mode=16,
        swarm_init_probability=0.60,
        sigma_floor=0.25,
        max_precision=10.0,
        w_reference_tracking=1.20,
        w_control_smooth=0.40,
        smooth_accel_weight=0.5,
        smooth_steering_rate_weight=2.0,
        w_heading=0.0,
        w_mode_prior=0.25,
        uncertainty_margin_gain=0.25,
        apply_control_lowpass=False,
        control_lowpass_alpha=0.0,
        max_delta_accel=1.20,
        max_delta_steering_rate=4.00,
        suppress_blocked_modes=True,
        mode_blocking_clearance=0.02,
        hard_collision_clearance=0.01,
        hard_collision_penalty=800_000.0,

    )


    print(f"Dynamic wall pairs: {wall_pairs}")
    print(f"Dynamic wall width: {wall_width}")
    print(f"Dynamic wall extension: {wall_extension}")
    print(f"Block step: {block_step}")

    variants = [
        ControllerVariant.GAUSSIAN_PRIOR_MPPI,
        ControllerVariant.CORRIDOR_PRIOR_MPPI,
        ControllerVariant.CONTROL_BANK_MPPI,
        ControllerVariant.STANDARD_MPPI,
        ControllerVariant.STANDARD_MPPI_128,
    ]

    results = []
    rows = []

    for variant in variants:
        print(f"Running {variant.value} with dynamic blocker at step {block_step}...")
        res = run_dynamic_blockage_controller(
            variant=variant,
            modes=modes,
            base_obstacles=base_obstacles,
            blocker=blocker,
            start=start,
            goal=goal,
            seed=seed,
            block_step=block_step,
            max_steps=150,
            goal_tolerance=0.3,
            mppi_cfg=cfg,
        )
        row = summarize_dynamic_result(
            res, base_obstacles, blocker, goal, cfg.robot_radius,
            vehicle_length=cfg.vehicle_length,
            vehicle_width=cfg.vehicle_width,
        )
        print(
            f"  success={row['success']} collision={row['collision']} "
            f"final_dist={row['final_dist']:.3f} smooth={row['control_smoothness']:.3f} "
            f"runtime/step={row['runtime_per_step_sec']:.3f}s"
        )
        results.append(res)
        rows.append(row)


    from pathlib import Path
    output_dir = Path("dynamic_block_soft")
    output_dir.mkdir(parents=True, exist_ok=True)

    animate_dynamic_blockage(
        results,
        base_obstacles,
        blocker,
        start,
        goal,
        bounds_xy,
        str(output_dir / "all_paths.gif"),
    )

    animate_dynamic_blockage_by_variant(
        results,
        modes,
        base_obstacles,
        blocker,
        start,
        goal,
        bounds_xy,
        output_dir=str(output_dir),
        mppi_cfg=cfg,
    )

    plt.show()


def main():
    print("Variant file: dynamic_block_soft")
    print(f"Numba enabled: {njit is not None}")
    scale, bounds_xy, bounds_ranges, start, goal, obstacles = build_default_scene()

    print("Running swarm planner once...")
    gen_out = run_swarm_planner(
        start=start,
        goal=goal,
        obstacles=obstacles,
        scale=scale,
        bounds_xy=bounds_xy,
        seed=RUN_SWARM_SEED,
    )
    print(f"Generated swarm trajectories: {len(gen_out.samples)}")
    print(f"Homotopy groups: {len(gen_out.homotopy_groups)}")

    print("Fitting Gaussian homotopy mixture...")
    mixture = fit_topological_trajectory_mixture(
        gen_out,
        obstacles,
        K=50,
        beta=1.0,
        min_mode_samples=3,
        covariance_jitter=2e-4,
        bounds=bounds_ranges,
        goal=goal,
        snap_to_goal_radius=0.2,
        snap_straight_tail_points=8,
    )
    modes = mixture_to_mppi_modes(mixture)

    print("Modes:")
    for i, m in enumerate(modes):
        print(f"  h{i}: sig={m.signature}, pi={m.probability:.3f}, samples={len(m.sample_paths or [])}")


    mppi_cfg = MPPIConfig(
        horizon=50,
        num_rollouts=64,
        dt=0.12,
        v_min=-1.0,
        v_max=2.8,
        lambda_temperature=2.2,
        noise_accel=0.90,
        noise_steering_rate=1.00,
        temporal_noise_smoothing=0.72,
        w_goal=110.0,
        w_obstacle=500.0,
        w_control=0.004,
        max_empirical_nominals_per_mode=16,
        swarm_init_probability=0.60,
        sigma_floor=0.25,
        max_precision=10.0,
        w_reference_tracking=1.20,
        w_control_smooth=0.40,
        smooth_accel_weight=0.5,
        smooth_steering_rate_weight=2.0,
        w_heading=0.0,
        w_mode_prior=0.25,
        uncertainty_margin_gain=0.25,
        apply_control_lowpass=False,
        control_lowpass_alpha=0.0,
        max_delta_accel=1.20,
        max_delta_steering_rate=4.00,
        suppress_blocked_modes=True,
        mode_blocking_clearance=0.02,
        hard_collision_clearance=0.01,
        hard_collision_penalty=800_000.0,
    )

    variants = [
        ControllerVariant.GAUSSIAN_PRIOR_MPPI,
        ControllerVariant.CORRIDOR_PRIOR_MPPI,
        ControllerVariant.CONTROL_BANK_MPPI,
        ControllerVariant.STANDARD_MPPI,
        ControllerVariant.STANDARD_MPPI_128,
    ]

    all_results = []
    rows = []

    for seed in RUN_SEEDS:
        print(f"\nSeed {seed}")
        for variant in variants:
            print(f"  Running {variant.value}...")
            try:
                res = run_controller_variant(
                    variant,
                    modes,
                    obstacles,
                    start,
                    goal,
                    seed=seed,
                    max_steps=150,
                    goal_tolerance=0.15,
                    mppi_cfg=mppi_cfg,
                        )
                row = summarize_result(
                    res, obstacles, goal,
                    robot_radius=mppi_cfg.robot_radius,
                    vehicle_length=mppi_cfg.vehicle_length,
                    vehicle_width=mppi_cfg.vehicle_width,
                )
                print(
                    f"    success={row['success']} collision={row['collision']} "
                    f"final_dist={row['final_dist']:.3f} runtime/step={row['runtime_per_step_sec']:.3f}s"
                )
                all_results.append(res)
                rows.append(row)
            except Exception as exc:
                print(f"    FAILED: {exc}")
                rows.append({
                    "variant": variant.value,
                    "seed": seed,
                    "success": False,
                    "reached_goal": False,
                    "collision": False,
                    "final_dist": np.nan,
                    "min_clearance": np.nan,
                    "path_length": np.nan,
                    "control_effort": np.nan,
                    "control_smoothness": np.nan,
                    "steps": np.nan,
                    "runtime_sec": np.nan,
                    "runtime_per_step_sec": np.nan,
                    "error": str(exc),
                })

    if pd is not None:
        df = pd.DataFrame(rows)
        metrics_path = f"{OUTPUT_PREFIX}_metrics.csv"
        df.to_csv(metrics_path, index=False)
        print(f"\nSaved metrics: {metrics_path}")

        summary = df.groupby("variant").agg({
            "success": "mean",
            "reached_goal": "mean",
            "collision": "mean",
            "final_dist": ["mean", "std"],
            "min_clearance": ["mean", "std"],
            "path_length": ["mean", "std"],
            "control_effort": ["mean", "std"],
            "control_smoothness": ["mean", "std"],
            "steps": ["mean", "std"],
            "runtime_per_step_sec": ["mean", "std"],
        })
        summary_path = f"{OUTPUT_PREFIX}_summary.csv"
        summary.to_csv(summary_path)
        print(f"Saved summary: {summary_path}")
        print("\nSummary:")
        print(summary)
    else:
        print("pandas unavailable; printing rows only.")
        for row in rows:
            print(row)


@dataclass(frozen=True)
class DynamicWallScenario:
    scenario_id: str
    wall_pairs: Tuple[Tuple[int, int], ...]
    trigger_progress: float = 0.35
    wall_width: float = 0.40
    wall_extension: float = 0.20


def default_dynamic_wall_scenarios() -> List[DynamicWallScenario]:

    return [
        DynamicWallScenario("wall_0_1", ((0, 1),), trigger_progress=0.25),
        DynamicWallScenario("wall_1_2", ((1, 2),), trigger_progress=0.25),
        DynamicWallScenario(
            "walls_0_1__1_2",
            ((0, 1), (1, 2)),
            trigger_progress=0.25,
        ),
    ]


def validate_dynamic_wall_scenario(scenario: DynamicWallScenario, obstacle_count: int):
    if not (0.0 <= scenario.trigger_progress <= 1.0):
        raise ValueError(
            f"Scenario {scenario.scenario_id}: trigger_progress must be in [0, 1]."
        )
    for i, j in scenario.wall_pairs:
        if i == j:
            raise ValueError(f"Scenario {scenario.scenario_id}: wall pair {(i, j)} is degenerate.")
        if not (0 <= i < obstacle_count and 0 <= j < obstacle_count):
            raise IndexError(
                f"Scenario {scenario.scenario_id}: wall pair {(i, j)} is outside "
                f"the obstacle index range [0, {obstacle_count - 1}]."
            )


def append_csv_row(path, row, fieldnames):

    path = str(path)
    write_header = not Path(path).exists() or Path(path).stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denom
    return center - half, center + half


def build_homotopy_modes_for_obstacles(
    start: Array,
    goal: Array,
    obstacles: Sequence,
    scale: float,
    bounds_xy,
    bounds_ranges,
    swarm_seed: int,
) -> List[MPPIHomotopyMode]:
    gen_out = run_swarm_planner(
        start=start,
        goal=goal,
        obstacles=obstacles,
        scale=scale,
        bounds_xy=bounds_xy,
        seed=swarm_seed,
    )
    mixture = fit_topological_trajectory_mixture(
        gen_out,
        obstacles,
        K=50,
        beta=1.0,
        min_mode_samples=3,
        covariance_jitter=2e-4,
        bounds=bounds_ranges,
        goal=goal,
        snap_to_goal_radius=0.2,
        snap_straight_tail_points=8,
    )
    return mixture_to_mppi_modes(mixture)


def save_robustness_summaries(
    detail_csv: str,
    summary_csv: str,
    scenario_summary_csv: str,
    success_per_scenario_csv: str,
):
    if pd is None:
        print("pandas unavailable; detailed CSV was saved, summary CSVs were skipped.")
        return

    df = pd.read_csv(detail_csv)
    completed = df[df["failure_reason"] != "controller_error"].copy()

    rows = []
    for (condition, variant), all_group in df.groupby(["condition", "variant"], sort=True):
        group = all_group[all_group["failure_reason"] != "controller_error"]
        n = int(len(group))
        successes = int(group["success"].sum()) if n else 0
        exposed = group[group["exposed_to_blocker"] == True]
        exposed_n = int(len(exposed))
        exposed_successes = int(exposed["success"].sum()) if exposed_n else 0
        lo, hi = wilson_interval(successes, n)
        rows.append({
            "condition": condition,
            "variant": variant,
            "trials": n,
            "successes": successes,
            "success_rate": successes / n if n else np.nan,
            "success_ci95_low": lo,
            "success_ci95_high": hi,
            "exposed_trials": exposed_n,
            "exposed_success_rate": (
                exposed_successes / exposed_n if exposed_n else np.nan
            ),
            "collision_rate": float(group["collision"].mean()) if n else np.nan,
            "not_reaching_rate": float(group["not_reaching"].mean()) if n else np.nan,
            "mean_min_clearance_after_block": (
                float(group["min_clearance_after_block"].mean()) if n else np.nan
            ),
            "mean_final_dist": float(group["final_dist"].mean()) if n else np.nan,
            "mean_steps_after_block": float(group["steps_after_block"].mean()) if n else np.nan,
            "mean_runtime_per_step_sec": (
                float(group["runtime_per_step_sec"].mean()) if n else np.nan
            ),
            "controller_errors": int((all_group["failure_reason"] == "controller_error").sum()),
        })
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    by_scenario = completed.groupby(
        ["condition", "scenario_id", "variant"], sort=True
    ).agg(
        trials=("success", "size"),
        successes=("success", "sum"),
        success_rate=("success", "mean"),
        collision_rate=("collision", "mean"),
        not_reaching_rate=("not_reaching", "mean"),
        exposed_rate=("exposed_to_blocker", "mean"),
        mean_activation_step=("activation_step", "mean"),
        mean_activation_progress=("activation_progress", "mean"),
        mean_min_clearance_after_block=("min_clearance_after_block", "mean"),
        mean_final_dist=("final_dist", "mean"),
    ).reset_index()
    by_scenario.to_csv(scenario_summary_csv, index=False)


    success_matrix = by_scenario.pivot_table(
        index=["condition", "scenario_id"],
        columns="variant",
        values="success_rate",
        aggfunc="first",
    )
    success_matrix.to_csv(success_per_scenario_csv)


def main():


    controller_seeds = RUN_SEEDS
    swarm_seeds = [RUN_SWARM_SEED]
    scenarios = default_dynamic_wall_scenarios()
    max_steps = 150
    goal_tolerance = 0.30
    activation_preview_clearance = None

    variants = [
        ControllerVariant.GAUSSIAN_PRIOR_MPPI,
        ControllerVariant.CORRIDOR_PRIOR_MPPI,
        ControllerVariant.CONTROL_BANK_MPPI,
        ControllerVariant.STANDARD_MPPI,
        ControllerVariant.STANDARD_MPPI_128,
    ]


    cfg = MPPIConfig(
        horizon=50,
        num_rollouts=64,
        dt=0.12,
        base_safety_margin=0.0,
        collision_substeps=5,
        hard_collision_clearance=0.01,
        hard_collision_penalty=800_000.0,
        suppress_blocked_modes=True,
        mode_blocking_clearance=0.02,
        mode_blocking_substeps=2,
        max_empirical_nominals_per_mode=16,
        swarm_init_probability=0.60,
        sigma_floor=0.25,
        max_precision=10.0,
        w_reference_tracking=1.20,
        w_control_smooth=0.40,
        smooth_accel_weight=0.5,
        smooth_steering_rate_weight=2.0,
        w_heading=0.0,
        w_mode_prior=0.25,
        uncertainty_margin_gain=0.25,
        apply_control_lowpass=False,
        control_lowpass_alpha=0.0,
        max_delta_accel=1.20,
        max_delta_steering_rate=4.00,
    )

    scale, bounds_xy, bounds_ranges, start, goal, original_obstacles = build_default_scene()
    for scenario in scenarios:
        validate_dynamic_wall_scenario(scenario, len(original_obstacles))


    fixed_wall_centers = tuple(
        obstacle_center(obs).copy()
        for obs in original_obstacles
    )


    fixed_blockers = {
        scenario.scenario_id: make_wall_blockers_between_centers(
            centers=fixed_wall_centers,
            pairs=scenario.wall_pairs,
            width=scenario.wall_width,
            extension=scenario.wall_extension,
        )
        for scenario in scenarios
    }

    detail_csv = "dynamic_block_long_robustness_trials.csv"
    summary_csv = "dynamic_block_long_robustness_summary.csv"
    scenario_summary_csv = "dynamic_block_long_robustness_by_scenario.csv"
    success_per_scenario_csv = "dynamic_block_long_success_per_scenario.csv"

    for output_path in (
        detail_csv,
        summary_csv,
        scenario_summary_csv,
        success_per_scenario_csv,
    ):
        output = Path(output_path)
        if output.exists():
            output.unlink()

    fieldnames = [
        "condition", "variant", "swarm_seed", "controller_seed", "seed",
        "obstacle_layout_seed", "obstacle_center_permutation",
        "scenario_id", "wall_pairs", "wall_count", "wall_width",
        "wall_extension", "trigger_progress", "activation_preview_clearance",
        "activation_step", "activation_progress", "activation_reason",
        "activation_clearance", "block_step", "success", "failure_reason",
        "reached_goal", "collision", "not_reaching", "first_collision_step",
        "collision_after_block", "exposed_to_blocker",
        "goal_reached_before_block", "final_dist", "min_clearance_dynamic",
        "min_clearance_after_block", "path_length", "path_length_after_block",
        "control_effort", "control_effort_after_block", "control_smoothness",
        "control_smoothness_after_block", "steps", "steps_after_block",
        "runtime_sec", "runtime_per_step_sec", "goal_tolerance", "error",
    ]

    trials_per_layout = (1 + 2 * len(scenarios)) * len(variants)
    total = len(swarm_seeds) * len(controller_seeds) * trials_per_layout
    trial_index = 0

    def execute_condition_trials(
        *,
        condition: str,
        scenario_id: str,
        wall_pairs: Tuple[Tuple[int, int], ...],
        wall_width: float,
        wall_extension: float,
        trigger_progress: Optional[float],
        modes: Optional[List[MPPIHomotopyMode]],
        controller_base_obstacles: Sequence,
        blocker,
        blocker_active_from_start: bool,
        swarm_seed: int,
        controller_seed: int,
        obstacle_layout_seed: int,
        obstacle_center_permutation: Tuple[int, ...],
        setup_error: str = "",
    ) -> None:
        nonlocal trial_index
        wall_pairs_text = ";".join(f"{i}-{j}" for i, j in wall_pairs)
        permutation_text = obstacle_center_permutation_text(
            obstacle_center_permutation
        )

        for variant in variants:
            trial_index += 1
            print(
                f"[{trial_index}/{total}] {condition}/{scenario_id} "
                f"layout={obstacle_layout_seed} seed={controller_seed} "
                f"variant={variant.value}"
            )
            base_row = {
                "condition": condition,
                "variant": variant.value,
                "swarm_seed": swarm_seed,
                "controller_seed": controller_seed,
                "seed": controller_seed,
                "obstacle_layout_seed": obstacle_layout_seed,
                "obstacle_center_permutation": permutation_text,
                "scenario_id": scenario_id,
                "wall_pairs": wall_pairs_text,
                "wall_count": len(wall_pairs),
                "wall_width": wall_width,
                "wall_extension": wall_extension,
                "trigger_progress": trigger_progress,
                "activation_preview_clearance": activation_preview_clearance,
                "goal_tolerance": goal_tolerance,
                "error": setup_error,
            }

            if setup_error or modes is None:
                row = dict(base_row)
                row.update({
                    "success": False,
                    "failure_reason": "controller_error",
                    "reached_goal": False,
                    "collision": False,
                    "not_reaching": False,
                })
            else:
                try:
                    result = run_dynamic_blockage_controller(
                        variant=variant,
                        modes=modes,
                        base_obstacles=controller_base_obstacles,
                        blocker=blocker,
                        start=start,
                        goal=goal,
                        seed=controller_seed,
                        trigger_progress=trigger_progress,
                        activation_preview_clearance=activation_preview_clearance,
                        blocker_active_from_start=blocker_active_from_start,
                        condition=condition,
                        max_steps=max_steps,
                        goal_tolerance=goal_tolerance,
                        mppi_cfg=cfg,
                        record_infos=False,
                        record_obstacle_history=False,
                    )
                except Exception as exc:
                    row = dict(base_row)
                    row.update({
                        "success": False,
                        "failure_reason": "controller_error",
                        "reached_goal": False,
                        "collision": False,
                        "not_reaching": False,
                        "error": repr(exc),
                    })
                else:
                    try:
                        row = summarize_dynamic_result(
                            result,
                            controller_base_obstacles,
                            blocker,
                            goal,
                            cfg.robot_radius,
                            goal_tolerance=goal_tolerance,
                            vehicle_length=cfg.vehicle_length,
                            vehicle_width=cfg.vehicle_width,
                        )
                        row.update(base_row)
                    except Exception as exc:
                        reached_goal = bool(result.get("reached_goal", False))
                        row = dict(base_row)
                        row.update({
                            "success": reached_goal,
                            "failure_reason": "summary_error",
                            "reached_goal": reached_goal,
                            "collision": False,
                            "not_reaching": not reached_goal,
                            "error": repr(exc),
                        })

            append_csv_row(detail_csv, row, fieldnames)
            print(
                f"  success={row.get('success')} "
                f"failure={row.get('failure_reason') or '-'} "
                f"error={row.get('error') or '-'}"
            )

    for swarm_seed in swarm_seeds:
        for controller_seed in controller_seeds:


            obstacle_layout_seed = int(controller_seed)
            swapped_obstacles, center_permutation = random_obstacle_center_swap(
                original_obstacles,
                seed=obstacle_layout_seed,
            )
            print(
                "Obstacle-center layout "
                f"seed={obstacle_layout_seed}: "
                f"{obstacle_center_permutation_text(center_permutation)}"
            )

            try:
                print(
                    "Building no-wall/dynamic prior for randomized obstacle "
                    f"layout {obstacle_layout_seed}, swarm seed {swarm_seed}..."
                )
                base_modes = build_homotopy_modes_for_obstacles(
                    start,
                    goal,
                    swapped_obstacles,
                    scale,
                    bounds_xy,
                    bounds_ranges,
                    swarm_seed,
                )
                base_setup_error = ""
            except Exception as exc:
                base_modes = None
                base_setup_error = repr(exc)
                print(f"  Random-layout prior failed: {base_setup_error}")


            execute_condition_trials(
                condition="no_wall",
                scenario_id="no_wall",
                wall_pairs=tuple(),
                wall_width=0.0,
                wall_extension=0.0,
                trigger_progress=None,
                modes=base_modes,
                controller_base_obstacles=swapped_obstacles,
                blocker=[],
                blocker_active_from_start=False,
                swarm_seed=swarm_seed,
                controller_seed=controller_seed,
                obstacle_layout_seed=obstacle_layout_seed,
                obstacle_center_permutation=center_permutation,
                setup_error=base_setup_error,
            )

            for scenario in scenarios:


                blocker = fixed_blockers[scenario.scenario_id]


                execute_condition_trials(
                    condition="dynamic_wall",
                    scenario_id=scenario.scenario_id,
                    wall_pairs=scenario.wall_pairs,
                    wall_width=scenario.wall_width,
                    wall_extension=scenario.wall_extension,
                    trigger_progress=scenario.trigger_progress,
                    modes=base_modes,
                    controller_base_obstacles=swapped_obstacles,
                    blocker=blocker,
                    blocker_active_from_start=False,
                    swarm_seed=swarm_seed,
                    controller_seed=controller_seed,
                    obstacle_layout_seed=obstacle_layout_seed,
                    obstacle_center_permutation=center_permutation,
                    setup_error=base_setup_error,
                )


                static_obstacles = list(swapped_obstacles) + list(blocker)
                if base_setup_error:
                    static_modes = None
                    static_setup_error = base_setup_error
                else:
                    try:
                        print(
                            f"Building static-wall oracle prior for "
                            f"{scenario.scenario_id}, layout "
                            f"{obstacle_layout_seed}, swarm seed {swarm_seed}..."
                        )
                        static_modes = build_homotopy_modes_for_obstacles(
                            start,
                            goal,
                            static_obstacles,
                            scale,
                            bounds_xy,
                            bounds_ranges,
                            swarm_seed,
                        )
                        static_setup_error = ""
                    except Exception as exc:
                        static_modes = None
                        static_setup_error = repr(exc)
                        print(f"  Static-wall prior failed: {static_setup_error}")

                execute_condition_trials(
                    condition="static_wall",
                    scenario_id=scenario.scenario_id,
                    wall_pairs=scenario.wall_pairs,
                    wall_width=scenario.wall_width,
                    wall_extension=scenario.wall_extension,
                    trigger_progress=None,
                    modes=static_modes,
                    controller_base_obstacles=static_obstacles,
                    blocker=[],
                    blocker_active_from_start=True,
                    swarm_seed=swarm_seed,
                    controller_seed=controller_seed,
                    obstacle_layout_seed=obstacle_layout_seed,
                    obstacle_center_permutation=center_permutation,
                    setup_error=static_setup_error,
                )

    save_robustness_summaries(
        detail_csv,
        summary_csv,
        scenario_summary_csv,
        success_per_scenario_csv,
    )
    print(f"Saved detailed trials: {detail_csv}")
    if Path(summary_csv).exists():
        print(f"Saved condition/variant summary: {summary_csv}")
    if Path(scenario_summary_csv).exists():
        print(f"Saved scenario summary: {scenario_summary_csv}")
    if Path(success_per_scenario_csv).exists():
        print(f"Saved success matrix: {success_per_scenario_csv}")


if __name__ == "__main__":
    main()