#!/usr/bin/env python3
"""
Batch robustness experiment for MPPI controller variants with hard collision constraints.

The default entry point runs paired controller trials across several dynamic-wall
scenarios and random seeds. Obstacle avoidance is removed from the objective;
rollouts are constrained using the exact polygon geometry. The script writes CSV
metrics only: no GIFs, plots, or images.

Run from the project root:
    python runs_hard_exact.py
"""

from __future__ import annotations

import csv
import math
import pickle
import time
from dataclasses import dataclass
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


# -----------------------------------------------------------------------------
# Your project imports
# -----------------------------------------------------------------------------

try:
    from geometry.utils import round_obstacle, PolyObstacle, obstacles_to_segs
    from RL.env2Ddiverse import FishGoalEnv2D
    from graph.graph import build_full_graph
    from planner import HomotopyAwareGenerativePlanner, trajectory_cost
except Exception as exc:
    raise ImportError(
        "Could not import your project modules. Run from the root of your project, "
        "where geometry/, RL/, graph/, planner.py, and save/ exist.\n"
        f"Original import error: {exc}"
    )


Array = np.ndarray


# =============================================================================
# Config
# =============================================================================

RUN_SEEDS = list(range(1))          # Increase to 20-50 for real experiments.
RUN_SWARM_SEED = 5
OUTPUT_PREFIX = "dynamic_block_hard"


class ControllerVariant(str, Enum):
    FULL_SWARM_PRIOR_MPPI = "full_swarm_prior_mppi"
    GAUSSIAN_PRIOR_MPPI = "gaussian_prior_mppi"
    EMPIRICAL_INIT_MPPI = "empirical_init_mppi"

    # Stable representation variants.
    HOMOTOPY_SEEDED_MPPI = "homotopy_seeded_mppi"
    CORRIDOR_PRIOR_MPPI = "corridor_prior_mppi"
    FRENET_CORRIDOR_MPPI = "frenet_corridor_mppi"
    HEATMAP_PRIOR_MPPI = "heatmap_prior_mppi"
    CONTROL_BANK_MPPI = "control_bank_mppi"
    MODE_SELECTING_HOMOTOPY_MPPI = "mode_selecting_homotopy_mppi"
    MODE_SELECTING_CORRIDOR_MPPI = "mode_selecting_corridor_mppi"

    STANDARD_MPPI = "standard_mppi"


@dataclass
class MPPIConfig:
    dt: float = 0.12
    horizon: int = 28
    num_rollouts: int = 650
    lambda_temperature: float = 7.0

    v_min: float = -1.6
    v_max: float = 1.6
    omega_min: float = -4.5
    omega_max: float = 4.5

    noise_v: float = 0.38
    noise_omega: float = 0.95
    temporal_noise_smoothing: float = 0.65

    swarm_init_probability: float = 0.45
    max_empirical_nominals_per_mode: int = 16

    robot_radius: float = 0.18
    base_safety_margin: float = 0.12
    uncertainty_margin_gain: float = 0.25

    # Obstacle handling. In the hard experiment the soft obstacle term is
    # disabled and exact polygonal collision avoidance is imposed as a rollout
    # feasibility constraint. A nonzero buffer enlarges the robot footprint.
    use_soft_obstacle_cost: bool = False
    enable_hard_collision_filter: bool = True
    hard_collision_clearance_buffer: float = 0.0

    # Legacy soft-mode settings retained only so the same cost routines can also
    # be used with use_soft_obstacle_cost=True. They are inactive in hard runs.
    collision_substeps: int = 5
    hard_collision_clearance: float = 0.02
    hard_collision_penalty: float = 200_000.0
    suppress_blocked_modes: bool = False
    mode_blocking_clearance: float = 0.02
    mode_blocking_substeps: int = 2

    w_goal: float = 30.0
    w_obstacle: float = 500.0
    w_control: float = 0.025
    # Smoother-prior settings.
    # The earlier version was too aggressive because small covariance blocks
    # created huge Mahalanobis tracking penalties. These defaults make the
    # Gaussian prior useful without letting it dominate control smoothness.
    w_control_smooth: float = 0.40
    w_heading: float = 0.0
    w_mode_prior: float = 0.15
    sigma_floor: float = 0.25

    # Soft-corridor tracking weight.
    # Earlier versions effectively used weight 1.0 on Mahalanobis tracking.
    # That made the robot chase the geometric swarm tube too aggressively.
    # This turns the Gaussian into a corridor prior rather than a hard reference.
    w_reference_tracking: float = 0.20

    # Penalize yaw-rate changes more than velocity changes. For a unicycle,
    # most visible/control jerk comes from abrupt omega changes.
    smooth_v_weight: float = 0.5
    smooth_omega_weight: float = 2.0

    # Cap eigenvalues of the covariance inverse. This prevents narrow Gaussian
    # tubes from producing arbitrarily large tracking gains.
    max_precision: float = 10.0

    # Monotonic progress tracking for local reference extraction.
    # Prevents the local reference window from jumping backward/forward abruptly.
    use_monotonic_reference_progress: bool = True
    max_reference_index_advance: int = 4

    # Optional smoothing on the actually applied command for MPPI variants.
    # This reduces high-frequency command jitter caused by stochastic sampling.
    apply_control_lowpass: bool = True
    control_lowpass_alpha: float = 0.55

    # Stable representation settings.
    # Corridor/Frenet variants penalize leaving a topological tube rather than
    # tracking a Cartesian Gaussian mean at every time index.
    w_corridor: float = 8.0
    corridor_radius_base: float = 0.35
    corridor_radius_scale: float = 1.25
    corridor_radius_min: float = 0.30
    corridor_radius_max: float = 1.20

    # Heatmap-like prior implemented as a smooth radial potential around the
    # homotopy centerline, without building a grid.
    w_heatmap: float = 3.0
    heatmap_sigma_scale: float = 1.4

    # Per-homotopy mode-selection settings.
    mode_select_top_k: int = 4
    mode_select_min_rollouts_per_mode: int = 64



# =============================================================================
# Trajectory mixture extraction
# =============================================================================

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
    """
    Snap the terminal part of a swarm trajectory to the exact goal.

    If the path enters a radius-snap_radius ball around the goal, keep the path
    until the first entry point and replace the remaining terminal segment by a
    straight line to the exact goal.

    This is applied before fitting the homotopy mixture, so Gaussian means,
    corridors, and empirical control banks all terminate cleanly.
    """
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

    # Remove duplicate / near-duplicate points.
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

    # Snap only the terminal segment of paths that already enter the goal ball.
    # Quality costs are evaluated on the snapped paths, but homotopy groups are
    # preserved via original object IDs below.
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

        modes.append(MPPIHomotopyMode(
            signature=sig,
            probability=mode.probability,
            mean_path=mean_path,
            cov_blocks=cov_blocks,
            sample_paths=sample_paths,
        ))

    modes.sort(key=lambda m: m.probability, reverse=True)
    return modes


# =============================================================================
# Geometry
# =============================================================================

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
    max_segment_length: float = 0.20,
    wall_max_segment_length: float = 0.10,
) -> List[Tuple[Array, float]]:
    """Build conservative circle covers for fast MPPI obstacle costs.

    Compact polygons retain the original single bounding-circle approximation.
    Elongated polygons, including the inserted rectangular walls, are covered by
    a chain of overlapping circles along their principal axis. This avoids
    replacing a thin wall with one very large circular forbidden region while
    keeping the existing Numba circle-cost kernels unchanged.

    The chain conservatively covers the polygon's PCA-aligned bounding box.
    Exact collision classification still uses polygon signed distance.
    """
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

        # Divide the major-axis interval into short strips. A circle centered in
        # each strip has a radius large enough to cover the strip's far corner.
        # Inserted walls are four-vertex elongated rectangles. Use a denser
        # circle chain for them while retaining a slightly coarser cover for
        # other elongated polygons. The radius still covers each strip's far
        # corner, so the complete wall remains conservatively covered.
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


def min_clearance(states: Array, obstacles: Sequence, robot_radius: float) -> float:
    if min_clearance_nb is not None:
        padded, lengths = obstacles_to_padded_arrays(obstacles)
        return float(min_clearance_nb(
            np.asarray(states, dtype=np.float64),
            padded,
            lengths,
            float(robot_radius),
        ))

    vals = []
    for x in states:
        p = x[:2]
        for obs in obstacles:
            d, _ = polygon_signed_distance_and_normal(p, obs)
            vals.append(d - robot_radius)
    return float(np.min(vals)) if vals else float("inf")


def path_collided(states: Array, obstacles: Sequence, robot_radius: float) -> bool:
    return min_clearance(states, obstacles, robot_radius) < 0.0


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


# =============================================================================
# Dynamics and reference localization
# =============================================================================

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def unicycle_step(x: Array, u: Array, dt: float) -> Array:
    px, py, th = x
    v, om = u
    return np.array([
        px + v * math.cos(th) * dt,
        py + v * math.sin(th) * dt,
        wrap_angle(th + om * dt),
    ], dtype=np.float64)


def rollout_unicycle(x0: Array, U: Array, dt: float) -> Array:
    if rollout_unicycle_single_nb is not None:
        return rollout_unicycle_single_nb(
            np.asarray(x0, dtype=np.float64),
            np.asarray(U, dtype=np.float64),
            float(dt),
        )

    X = np.zeros((len(U) + 1, 3), dtype=np.float64)
    X[0] = x0
    for t, u in enumerate(U):
        X[t + 1] = unicycle_step(X[t], u, dt)
    return X


def rollout_unicycle_batch(x0: Array, U: Array, dt: float) -> Array:
    if rollout_unicycle_batch_nb is not None:
        return rollout_unicycle_batch_nb(
            np.asarray(x0, dtype=np.float64),
            np.asarray(U, dtype=np.float64),
            float(dt),
        )

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


# =============================================================================
# Numba-accelerated kernels
# =============================================================================

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

    @njit(cache=True)
    def rollout_unicycle_batch_nb(x0, U, dt):
        N = U.shape[0]
        H = U.shape[1]
        X = np.zeros((N, H + 1, 3), dtype=np.float64)

        for n in range(N):
            X[n, 0, 0] = x0[0]
            X[n, 0, 1] = x0[1]
            X[n, 0, 2] = x0[2]

        for n in range(N):
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
        robot_radius,
        base_safety_margin,
        uncertainty_margin_gain,
        w_goal,
        w_obstacle,
        w_control,
        w_control_smooth,
        w_heading,
        w_mode_prior,
        w_reference_tracking,
        smooth_v_weight,
        smooth_omega_weight,
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

                # Default Sigma is sigma_floor^2 I.
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
                        # Closed-form eigenvalue clamp for a 2x2 covariance.
                        # This caps the precision matrix so narrow covariance
                        # tubes do not create non-smooth control reactions.
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

                        # Eigenvectors for symmetric 2x2. If nearly diagonal,
                        # use axis-aligned basis.
                        if abs(s01) < 1e-12 and abs(diff) < 1e-12:
                            inv00 = p1
                            inv01 = 0.0
                            inv11 = p2
                        else:
                            # Eigenvector for lam1.
                            vx = s01
                            vy = lam1 - s00
                            norm_v = math.sqrt(vx * vx + vy * vy)
                            if norm_v < 1e-12:
                                vx = 1.0
                                vy = 0.0
                            else:
                                vx /= norm_v
                                vy /= norm_v

                            # Orthogonal eigenvector for lam2.
                            wx = -vy
                            wy = vx

                            inv00 = p1 * vx * vx + p2 * wx * wx
                            inv01 = p1 * vx * vy + p2 * wx * wy
                            inv11 = p1 * vy * vy + p2 * wy * wy

                        mahal = ex * (inv00 * ex + inv01 * ey) + ey * (inv01 * ex + inv11 * ey)
                        cost += w_reference_tracking * mahal
                    else:
                        cost += w_reference_tracking * 4.0 * (ex * ex + ey * ey)

                    if t < H - 1:
                        tx = mean_path[t + 1, 0] - mean_path[t, 0]
                        ty = mean_path[t + 1, 1] - mean_path[t, 1]
                        if math.sqrt(tx * tx + ty * ty) > 1e-9:
                            ref_heading = math.atan2(ty, tx)
                            dh = _wrap_angle_nb(X[n, t + 1, 2] - ref_heading)
                            cost += w_heading * dh * dh

                for j in range(M):
                    dx = px - circle_centers[j, 0]
                    dy = py - circle_centers[j, 1]
                    norm = math.sqrt(dx * dx + dy * dy) + 1e-12
                    nx = dx / norm
                    ny = dy / norm
                    d = norm - circle_radii[j]

                    margin = robot_radius + base_safety_margin

                    if use_uncertainty_margin and use_mean_reference:
                        sigma_n_sq = nx * (s00 * nx + s01 * ny) + ny * (s01 * nx + s11 * ny)
                        if sigma_n_sq < 0.0:
                            sigma_n_sq = 0.0
                        margin += uncertainty_margin_gain * math.sqrt(sigma_n_sq)

                    z = 8.0 * (margin - d)
                    sp = _softplus_scalar_nb(z)
                    cost += w_obstacle * sp * sp

            gx = X[n, H, 0] - goal[0]
            gy = X[n, H, 1] - goal[1]
            cost += w_goal * (gx * gx + gy * gy)

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
                smooth_cost += smooth_v_weight * dv * dv + smooth_omega_weight * dom * dom
            cost += w_control_smooth * smooth_cost

            if use_mode_prior:
                cost += w_mode_prior * (-math.log(mode_probability + 1e-12))

            costs[n] = cost

        return costs


    @njit(cache=True)
    def stable_representation_costs_nb(
        X,
        U,
        mean_path,
        corridor_radius,
        mode_probability,
        circle_centers,
        circle_radii,
        goal,
        horizon,
        robot_radius,
        base_safety_margin,
        w_goal,
        w_obstacle,
        w_control,
        w_control_smooth,
        smooth_v_weight,
        smooth_omega_weight,
        w_corridor,
        w_heatmap,
        heatmap_sigma_scale,
        w_mode_prior,
        rep_type,
        use_mode_prior,
    ):
        """
        Fast cost for stable homotopy representations.

        rep_type:
            0 = no representation cost, only current goal/obstacle/control cost
            1 = soft corridor around centerline
            2 = Frenet/lateral corridor around centerline
            3 = heatmap-like radial prior around centerline
            4 = control-bank/proposal only, no representation cost
        """
        N = U.shape[0]
        H = horizon
        M = circle_radii.shape[0]
        costs = np.zeros(N, dtype=np.float64)

        for n in range(N):
            cost = 0.0

            for t in range(H):
                px = X[n, t + 1, 0]
                py = X[n, t + 1, 1]

                mux = mean_path[t, 0]
                muy = mean_path[t, 1]
                ex = px - mux
                ey = py - muy
                r = corridor_radius[t]

                if rep_type == 1:
                    dcen = math.sqrt(ex * ex + ey * ey)
                    outside = dcen - r
                    sp = _softplus_scalar_nb(6.0 * outside)
                    cost += w_corridor * sp * sp

                elif rep_type == 2:
                    if t < H - 1:
                        tx = mean_path[t + 1, 0] - mean_path[t, 0]
                        ty = mean_path[t + 1, 1] - mean_path[t, 1]
                    else:
                        tx = mean_path[t, 0] - mean_path[t - 1, 0]
                        ty = mean_path[t, 1] - mean_path[t - 1, 1]

                    norm_t = math.sqrt(tx * tx + ty * ty) + 1e-12
                    nx = -ty / norm_t
                    ny = tx / norm_t
                    lateral = abs(ex * nx + ey * ny)
                    outside = lateral - r
                    sp = _softplus_scalar_nb(6.0 * outside)
                    cost += w_corridor * sp * sp

                elif rep_type == 3:
                    dcen = math.sqrt(ex * ex + ey * ey)
                    sigma = heatmap_sigma_scale * (r + 1e-9)
                    q = dcen / sigma
                    heat_cost = 1.0 - math.exp(-0.5 * q * q)
                    cost += w_heatmap * heat_cost

                for j in range(M):
                    dx = px - circle_centers[j, 0]
                    dy = py - circle_centers[j, 1]
                    d = math.sqrt(dx * dx + dy * dy) - circle_radii[j]
                    margin = robot_radius + base_safety_margin
                    sp = _softplus_scalar_nb(8.0 * (margin - d))
                    cost += w_obstacle * sp * sp

            gx = X[n, H, 0] - goal[0]
            gy = X[n, H, 1] - goal[1]
            cost += w_goal * (gx * gx + gy * gy)

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
                smooth_cost += smooth_v_weight * dv * dv + smooth_omega_weight * dom * dom
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
        robot_radius,
        base_safety_margin,
        w_goal,
        w_obstacle,
        w_control,
        w_control_smooth,
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

                for j in range(M):
                    dx = px - circle_centers[j, 0]
                    dy = py - circle_centers[j, 1]
                    d = math.sqrt(dx * dx + dy * dy) - circle_radii[j]
                    margin = robot_radius + base_safety_margin
                    sp = _softplus_scalar_nb(8.0 * (margin - d))
                    cost += w_obstacle * sp * sp

            gx = X[n, H, 0] - goal[0]
            gy = X[n, H, 1] - goal[1]
            cost += w_goal * (gx * gx + gy * gy)

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
    def interpolated_obstacle_penalty_nb(
        X,
        circle_centers,
        circle_radii,
        robot_radius,
        base_safety_margin,
        w_obstacle,
        collision_substeps,
        hard_collision_clearance,
        hard_collision_penalty,
    ):
        """Extra obstacle cost at interior rollout points plus a near-hard term."""
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
                x1 = X[n, t + 1, 0]
                y1 = X[n, t + 1, 1]

                # q == substeps + 1 is the endpoint. Endpoints already receive
                # the normal soft obstacle cost, but still need the hard check.
                for q in range(1, substeps + 2):
                    alpha = q / denominator
                    px = x0 + alpha * (x1 - x0)
                    py = y0 + alpha * (y1 - y0)
                    min_clearance = 1e18

                    for j in range(M):
                        dx = px - circle_centers[j, 0]
                        dy = py - circle_centers[j, 1]
                        d = math.sqrt(dx * dx + dy * dy) - circle_radii[j]
                        clearance = d - robot_radius
                        if clearance < min_clearance:
                            min_clearance = clearance

                        if q <= substeps:
                            margin = robot_radius + base_safety_margin
                            sp = _softplus_scalar_nb(8.0 * (margin - d))
                            cost += (w_obstacle / denominator) * sp * sp

                    if min_clearance < hard_collision_clearance:
                        penetration = hard_collision_clearance - min_clearance
                        cost += hard_collision_penalty * (1.0 + penetration * penetration)

            extras[n] = cost

        return extras

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
                x_cross = xi + (py - yi) * (xj - xi) / ((yj - yi) + 1e-18)
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
        best = 1e18
        S = states.shape[0]
        M = poly_lengths.shape[0]

        for s in range(S):
            px = states[s, 0]
            py = states[s, 1]

            for m in range(M):
                n = poly_lengths[m]
                poly = polys_padded[m]

                min_d = 1e18
                for i in range(n):
                    j = i + 1
                    if j == n:
                        j = 0
                    d = point_segment_dist_nb(
                        px, py,
                        poly[i, 0], poly[i, 1],
                        poly[j, 0], poly[j, 1],
                    )
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
    def _cross2_nb(ax, ay, bx, by):
        return ax * by - ay * bx

    @njit(cache=True)
    def _on_segment_nb(ax, ay, bx, by, px, py, eps=1e-12):
        return (
            min(ax, bx) - eps <= px <= max(ax, bx) + eps
            and min(ay, by) - eps <= py <= max(ay, by) + eps
        )

    @njit(cache=True)
    def _segments_intersect_nb(ax, ay, bx, by, cx, cy, dx, dy):
        eps = 1e-12
        o1 = _cross2_nb(bx - ax, by - ay, cx - ax, cy - ay)
        o2 = _cross2_nb(bx - ax, by - ay, dx - ax, dy - ay)
        o3 = _cross2_nb(dx - cx, dy - cy, ax - cx, ay - cy)
        o4 = _cross2_nb(dx - cx, dy - cy, bx - cx, by - cy)

        if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
            (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
        ):
            return True

        if abs(o1) <= eps and _on_segment_nb(ax, ay, bx, by, cx, cy, eps):
            return True
        if abs(o2) <= eps and _on_segment_nb(ax, ay, bx, by, dx, dy, eps):
            return True
        if abs(o3) <= eps and _on_segment_nb(cx, cy, dx, dy, ax, ay, eps):
            return True
        if abs(o4) <= eps and _on_segment_nb(cx, cy, dx, dy, bx, by, eps):
            return True
        return False

    @njit(cache=True)
    def _segment_segment_distance_nb(ax, ay, bx, by, cx, cy, dx, dy):
        if _segments_intersect_nb(ax, ay, bx, by, cx, cy, dx, dy):
            return 0.0
        d1 = point_segment_dist_nb(ax, ay, cx, cy, dx, dy)
        d2 = point_segment_dist_nb(bx, by, cx, cy, dx, dy)
        d3 = point_segment_dist_nb(cx, cy, ax, ay, bx, by)
        d4 = point_segment_dist_nb(dx, dy, ax, ay, bx, by)
        return min(d1, d2, d3, d4)

    @njit(cache=True)
    def hard_feasible_mask_nb(
        X,
        polys_padded,
        poly_lengths,
        poly_bboxes,
        robot_radius,
        clearance_buffer,
    ):
        """Exact swept-disc feasibility for the piecewise-linear rollout path."""
        N = X.shape[0]
        H = X.shape[1] - 1
        M = poly_lengths.shape[0]
        feasible = np.ones(N, dtype=np.bool_)
        required = robot_radius + clearance_buffer

        for rollout in range(N):
            for t in range(H):
                ax = X[rollout, t, 0]
                ay = X[rollout, t, 1]
                bx = X[rollout, t + 1, 0]
                by = X[rollout, t + 1, 1]

                seg_min_x = min(ax, bx) - required
                seg_max_x = max(ax, bx) + required
                seg_min_y = min(ay, by) - required
                seg_max_y = max(ay, by) + required

                for obstacle in range(M):
                    if (
                        seg_max_x < poly_bboxes[obstacle, 0]
                        or seg_min_x > poly_bboxes[obstacle, 1]
                        or seg_max_y < poly_bboxes[obstacle, 2]
                        or seg_min_y > poly_bboxes[obstacle, 3]
                    ):
                        continue

                    n_vertices = poly_lengths[obstacle]
                    poly = polys_padded[obstacle]
                    if point_in_poly_nb(ax, ay, poly[:n_vertices]) or point_in_poly_nb(
                        bx, by, poly[:n_vertices]
                    ):
                        feasible[rollout] = False
                        break

                    for edge in range(n_vertices):
                        next_edge = edge + 1
                        if next_edge == n_vertices:
                            next_edge = 0
                        distance = _segment_segment_distance_nb(
                            ax,
                            ay,
                            bx,
                            by,
                            poly[edge, 0],
                            poly[edge, 1],
                            poly[next_edge, 0],
                            poly[next_edge, 1],
                        )
                        if distance < required:
                            feasible[rollout] = False
                            break

                    if not feasible[rollout]:
                        break

                if not feasible[rollout]:
                    break

        return feasible

else:
    rollout_unicycle_batch_nb = None
    rollout_unicycle_single_nb = None
    fast_swarm_prior_costs_nb = None
    stable_representation_costs_nb = None
    standard_mppi_costs_batch_nb = None
    interpolated_obstacle_penalty_nb = None
    min_clearance_nb = None
    hard_feasible_mask_nb = None


def obstacle_circles_to_arrays(obstacle_circles: List[Tuple[Array, float]]) -> Tuple[Array, Array]:
    if not obstacle_circles:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    return centers, radii


def interpolated_obstacle_penalty(
    X: Array,
    obstacle_circles: List[Tuple[Array, float]],
    cfg: MPPIConfig,
) -> Array:
    """Evaluate interior rollout points and apply a near-hard collision cost."""
    N = int(X.shape[0])
    if not obstacle_circles:
        return np.zeros(N, dtype=np.float64)

    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    if interpolated_obstacle_penalty_nb is not None:
        return interpolated_obstacle_penalty_nb(
            np.asarray(X, dtype=np.float64),
            centers,
            radii,
            float(cfg.robot_radius),
            float(cfg.base_safety_margin),
            float(cfg.w_obstacle),
            int(cfg.collision_substeps),
            float(cfg.hard_collision_clearance),
            float(cfg.hard_collision_penalty),
        )

    extras = np.zeros(N, dtype=np.float64)
    substeps = max(0, int(cfg.collision_substeps))
    denominator = float(substeps + 1)

    for t in range(X.shape[1] - 1):
        p0 = X[:, t, :2]
        p1 = X[:, t + 1, :2]
        for q in range(1, substeps + 2):
            alpha = q / denominator
            p = p0 + alpha * (p1 - p0)
            d = np.linalg.norm(
                p[:, None, :] - centers[None, :, :], axis=2
            ) - radii[None, :]
            clearance = d - cfg.robot_radius

            if q <= substeps:
                margin = cfg.robot_radius + cfg.base_safety_margin
                extras += (cfg.w_obstacle / denominator) * np.sum(
                    softplus(8.0 * (margin - d)) ** 2,
                    axis=1,
                )

            min_clearance = np.min(clearance, axis=1)
            penetration = np.maximum(
                0.0,
                cfg.hard_collision_clearance - min_clearance,
            )
            extras += cfg.hard_collision_penalty * (
                penetration > 0.0
            ) * (1.0 + penetration ** 2)

    return extras


def path_min_clearance_to_circles(
    path: Array,
    obstacle_circles: List[Tuple[Array, float]],
    robot_radius: float,
    substeps: int = 2,
) -> float:
    """Minimum robot clearance along a polyline under the circle approximation."""
    p = np.asarray(path, dtype=np.float64)
    if len(p) == 0 or not obstacle_circles:
        return float("inf")

    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    samples = [p]
    if len(p) > 1:
        for q in range(1, max(0, int(substeps)) + 1):
            alpha = q / float(max(0, int(substeps)) + 1)
            samples.append(p[:-1] + alpha * (p[1:] - p[:-1]))
    points = np.vstack(samples)
    clearance = (
        np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        - radii[None, :]
        - float(robot_radius)
    )
    return float(np.min(clearance))


def unblocked_mode_indices(
    local_modes: Sequence[MPPIHomotopyMode],
    obstacle_circles: List[Tuple[Array, float]],
    cfg: MPPIConfig,
) -> Tuple[List[int], Array]:
    """Return usable mode indices and their local centerline clearances."""
    clearances = np.asarray([
        path_min_clearance_to_circles(
            mode.mean_path,
            obstacle_circles,
            cfg.robot_radius,
            substeps=cfg.mode_blocking_substeps,
        )
        for mode in local_modes
    ], dtype=np.float64)

    if not cfg.suppress_blocked_modes or len(local_modes) <= 1:
        return list(range(len(local_modes))), clearances

    usable = np.where(clearances >= cfg.mode_blocking_clearance)[0].tolist()
    if not usable:
        usable = [int(np.argmax(clearances))]
    return usable, clearances


def obstacles_to_padded_arrays(obstacles: Sequence) -> Tuple[Array, Array]:
    polys = [_poly_vertices(o).astype(np.float64) for o in obstacles]
    max_n = max(p.shape[0] for p in polys)
    padded = np.zeros((len(polys), max_n, 2), dtype=np.float64)
    lengths = np.zeros(len(polys), dtype=np.int64)

    for i, p in enumerate(polys):
        padded[i, :p.shape[0], :] = p
        lengths[i] = p.shape[0]

    return padded, lengths



def obstacles_to_hard_arrays(obstacles: Sequence) -> Tuple[Array, Array, Array]:
    """Pack exact polygon vertices, lengths, and axis-aligned bounding boxes."""
    if not obstacles:
        return (
            np.zeros((0, 0, 2), dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 4), dtype=np.float64),
        )
    padded, lengths = obstacles_to_padded_arrays(obstacles)
    bboxes = np.zeros((len(obstacles), 4), dtype=np.float64)
    for index, obstacle in enumerate(obstacles):
        poly = _poly_vertices(obstacle)
        bboxes[index] = [
            float(np.min(poly[:, 0])),
            float(np.max(poly[:, 0])),
            float(np.min(poly[:, 1])),
            float(np.max(poly[:, 1])),
        ]
    return padded, lengths, bboxes


def _orientation_xy(a: Array, b: Array, c: Array) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment_xy(a: Array, b: Array, p: Array, eps: float = 1e-12) -> bool:
    return bool(
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect_xy(a: Array, b: Array, c: Array, d: Array) -> bool:
    eps = 1e-12
    o1 = _orientation_xy(a, b, c)
    o2 = _orientation_xy(a, b, d)
    o3 = _orientation_xy(c, d, a)
    o4 = _orientation_xy(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    return bool(
        (abs(o1) <= eps and _on_segment_xy(a, b, c, eps))
        or (abs(o2) <= eps and _on_segment_xy(a, b, d, eps))
        or (abs(o3) <= eps and _on_segment_xy(c, d, a, eps))
        or (abs(o4) <= eps and _on_segment_xy(c, d, b, eps))
    )


def _segment_segment_distance_xy(a: Array, b: Array, c: Array, d: Array) -> float:
    if _segments_intersect_xy(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance_and_normal(a, c, d)[0],
        point_segment_distance_and_normal(b, c, d)[0],
        point_segment_distance_and_normal(c, a, b)[0],
        point_segment_distance_and_normal(d, a, b)[0],
    )


def hard_feasible_rollout_mask(
    X: Array,
    obstacles: Sequence,
    robot_radius: float,
    clearance_buffer: float = 0.0,
    enabled: bool = True,
) -> Array:
    """Return the exact polygonal feasibility mask for a rollout batch.

    The unicycle discretization produces a straight position segment during each
    integration interval. The minimum distance between every rollout segment and
    every polygon edge is therefore checked directly, with the polygon inflated
    by the robot radius and optional clearance buffer.
    """
    X = np.asarray(X, dtype=np.float64)
    if not enabled or len(obstacles) == 0:
        return np.ones(X.shape[0], dtype=bool)

    padded, lengths, bboxes = obstacles_to_hard_arrays(obstacles)
    if hard_feasible_mask_nb is not None:
        return np.asarray(
            hard_feasible_mask_nb(
                X,
                padded,
                lengths,
                bboxes,
                float(robot_radius),
                float(clearance_buffer),
            ),
            dtype=bool,
        )

    required = float(robot_radius + clearance_buffer)
    feasible = np.ones(X.shape[0], dtype=bool)
    for rollout in range(X.shape[0]):
        for t in range(X.shape[1] - 1):
            a = X[rollout, t, :2]
            b = X[rollout, t + 1, :2]
            for obstacle in obstacles:
                poly = _poly_vertices(obstacle)
                if point_in_poly(a, poly) or point_in_poly(b, poly):
                    feasible[rollout] = False
                    break
                for edge in range(len(poly)):
                    c = poly[edge]
                    d = poly[(edge + 1) % len(poly)]
                    if _segment_segment_distance_xy(a, b, c, d) < required:
                        feasible[rollout] = False
                        break
                if not feasible[rollout]:
                    break
            if not feasible[rollout]:
                break
    return feasible


def impose_hard_collision_constraints(
    costs: Array,
    X: Array,
    obstacles: Sequence,
    cfg: MPPIConfig,
) -> Tuple[Array, Array]:
    """Set the objective of infeasible rollouts to infinity."""
    feasible = hard_feasible_rollout_mask(
        X,
        obstacles,
        cfg.robot_radius,
        clearance_buffer=cfg.hard_collision_clearance_buffer,
        enabled=cfg.enable_hard_collision_filter,
    )
    constrained = np.asarray(costs, dtype=np.float64).copy()
    constrained[~feasible] = np.inf
    return constrained, feasible


def finite_cost_statistics(costs: Array) -> Tuple[float, float]:
    finite = np.asarray(costs, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float('inf'), float('inf')
    return float(np.min(finite)), float(np.mean(finite))


def emergency_safe_control(
    x_current: Array,
    obstacles: Sequence,
    goal: Array,
    cfg: MPPIConfig,
) -> Tuple[Array, Dict[str, object]]:
    """Select a collision-free one-step fallback command when no rollout is feasible."""
    candidates = np.asarray([
        (0.0, 0.0),
        (-0.25, 0.0), (-0.50, 0.0), (-0.75, 0.0),
        (0.0, 1.5), (0.0, -1.5),
        (-0.30, 1.5), (-0.30, -1.5),
        (0.25, 1.5), (0.25, -1.5),
        (-0.60, 2.5), (-0.60, -2.5),
        (0.40, 2.5), (0.40, -2.5),
    ], dtype=np.float64)
    candidates[:, 0] = np.clip(candidates[:, 0], cfg.v_min, cfg.v_max)
    candidates[:, 1] = np.clip(candidates[:, 1], cfg.omega_min, cfg.omega_max)
    U = candidates[:, None, :]
    X = rollout_unicycle_batch(x_current, U, cfg.dt)
    feasible = hard_feasible_rollout_mask(
        X,
        obstacles,
        cfg.robot_radius,
        clearance_buffer=cfg.hard_collision_clearance_buffer,
        enabled=cfg.enable_hard_collision_filter,
    )
    if np.any(feasible):
        ids = np.flatnonzero(feasible)
        terminal = X[ids, -1, :2]
        score = np.linalg.norm(terminal - goal[None, :], axis=1)
        score += 0.05 * (
            candidates[ids, 0] ** 2 + 0.15 * candidates[ids, 1] ** 2
        )
        selected = int(ids[int(np.argmin(score))])
        return candidates[selected].copy(), {
            'mppi_action_rule': 'emergency_safe_control',
            'num_feasible_rollouts': 0,
            'optimal_traj': X[selected].copy(),
        }
    return np.zeros(2, dtype=np.float64), {
        'mppi_action_rule': 'no_safe_one_step_control_found',
        'num_feasible_rollouts': 0,
        'optimal_traj': rollout_unicycle(x_current, np.zeros((1, 2)), cfg.dt),
    }


def select_hard_feasible_action(
    costs: Array,
    U: Array,
    X: Array,
    x_current: Array,
    obstacles: Sequence,
    goal: Array,
    cfg: MPPIConfig,
) -> Tuple[Array, Dict[str, object], Array]:
    """Apply the first command of the lowest-cost feasible rollout."""
    constrained, feasible = impose_hard_collision_constraints(costs, X, obstacles, cfg)
    finite_feasible = feasible & np.isfinite(costs)
    if np.any(finite_feasible):
        candidate_costs = constrained.copy()
        candidate_costs[~finite_feasible] = np.inf
        selected = int(np.argmin(candidate_costs))
        return U[selected, 0, :].copy(), {
            'mppi_action_rule': 'best_feasible_rollout_first_control',
            'num_feasible_rollouts': int(np.sum(finite_feasible)),
            'optimal_traj': X[selected].copy(),
        }, constrained

    control, info = emergency_safe_control(x_current, obstacles, goal, cfg)
    return control, info, constrained


def shield_control_if_unsafe(
    x_current: Array,
    control: Array,
    obstacles: Sequence,
    goal: Array,
    cfg: MPPIConfig,
) -> Tuple[Array, Optional[Dict[str, object]]]:
    """Recheck the actually applied command after optional filtering."""
    U = np.asarray(control, dtype=np.float64).reshape(1, 1, 2)
    X = rollout_unicycle_batch(x_current, U, cfg.dt)
    feasible = hard_feasible_rollout_mask(
        X,
        obstacles,
        cfg.robot_radius,
        clearance_buffer=cfg.hard_collision_clearance_buffer,
        enabled=cfg.enable_hard_collision_filter,
    )
    if bool(feasible[0]):
        return np.asarray(control, dtype=np.float64), None
    safe_control, info = emergency_safe_control(x_current, obstacles, goal, cfg)
    info['unsafe_requested_control'] = np.asarray(control, dtype=np.float64).copy()
    return safe_control, info


def localize_mode_for_state_with_index(
    mode: MPPIHomotopyMode,
    x_current: Array,
    H: int,
    previous_idx: Optional[int] = None,
    max_advance: Optional[int] = None,
) -> Tuple[MPPIHomotopyMode, int]:
    """
    Localize a global homotopy path into a receding-horizon reference.

    If previous_idx is provided, the selected path index is forced to move
    monotonically. This avoids reference-window jumps, which were a major source
    of non-smooth controls in the full swarm-prior variant.
    """
    mu = mode.mean_path
    d = np.linalg.norm(mu - x_current[:2], axis=1)
    nearest_idx = int(np.argmin(d))

    if previous_idx is None:
        idx = nearest_idx
    else:
        idx = max(int(previous_idx), nearest_idx)
        if max_advance is not None:
            idx = min(idx, int(previous_idx) + int(max_advance))
        idx = min(idx, len(mu) - 2)

    tail = mu[idx:] if idx < len(mu) - 1 else mu[-2:]
    local_mu = resample_path(tail, H)

    source_ids = np.linspace(idx, len(mu) - 1, H)
    local_cov = np.zeros((H, 2, 2), dtype=np.float64)
    for t, j in enumerate(source_ids):
        local_cov[t] = mode.cov_blocks[int(round(j))]

    return MPPIHomotopyMode(
        signature=mode.signature,
        probability=mode.probability,
        mean_path=local_mu,
        cov_blocks=local_cov,
        sample_paths=None,
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


def nominal_controls_to_track_path(x0: Array, ref: Array, cfg) -> Array:
    H = cfg.horizon
    U = np.zeros((H, 2), dtype=np.float64)
    x = x0.copy()

    for t in range(H):
        target = ref[min(t + 3, len(ref) - 1)]
        delta = target - x[:2]
        dist = float(np.linalg.norm(delta))
        desired_heading = math.atan2(delta[1], delta[0])
        err = wrap_angle(desired_heading - x[2])

        v = np.clip(0.45 + 1.8 * dist * max(0.15, math.cos(err)), cfg.v_min, cfg.v_max)
        omega = np.clip(3.2 * err, cfg.omega_min, cfg.omega_max)

        U[t] = [v, omega]
        x = unicycle_step(x, U[t], cfg.dt)

    return U


def nominal_controls_to_goal(x0: Array, goal: Array, cfg) -> Array:
    H = cfg.horizon
    U = np.zeros((H, 2), dtype=np.float64)
    x = x0.copy()

    for t in range(H):
        delta = goal - x[:2]
        dist = float(np.linalg.norm(delta))
        desired_heading = math.atan2(delta[1], delta[0])
        err = wrap_angle(desired_heading - x[2])

        v = np.clip(0.4 + 1.5 * dist * max(0.1, math.cos(err)), cfg.v_min, cfg.v_max)
        omega = np.clip(3.0 * err, cfg.omega_min, cfg.omega_max)

        U[t] = [v, omega]
        x = unicycle_step(x, U[t], cfg.dt)

    return U


# =============================================================================
# MPPI costs and steps
# =============================================================================

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
    """
    Invert a 2x2 covariance with eigenvalue/precision clipping.

    This is the non-numba fallback. The numba kernel implements the same idea
    inline for speed.
    """
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
    if fast_swarm_prior_costs_nb is not None:
        centers, radii = obstacle_circles_to_arrays(obstacle_circles)
        costs = fast_swarm_prior_costs_nb(
            np.asarray(X, dtype=np.float64),
            np.asarray(U, dtype=np.float64),
            np.asarray(mode.mean_path, dtype=np.float64),
            np.asarray(mode.cov_blocks, dtype=np.float64),
            float(mode.probability),
            centers,
            radii,
            np.asarray(goal, dtype=np.float64),
            int(cfg.horizon),
            float(cfg.robot_radius),
            float(cfg.base_safety_margin),
            float(cfg.uncertainty_margin_gain),
            float(cfg.w_goal),
            float(cfg.w_obstacle),
            float(cfg.w_control),
            float(cfg.w_control_smooth),
            float(cfg.w_heading),
            float(cfg.w_mode_prior),
            float(cfg.w_reference_tracking),
            float(cfg.smooth_v_weight),
            float(cfg.smooth_omega_weight),
            float(cfg.sigma_floor),
            float(cfg.max_precision),
            bool(use_gaussian_tracking),
            bool(use_uncertainty_margin),
            bool(use_mode_prior),
            bool(use_mean_reference),
        )
        return costs + interpolated_obstacle_penalty(X, obstacle_circles, cfg)

    N = U.shape[0]
    H = cfg.horizon
    costs = np.zeros(N, dtype=np.float64)
    P = X[:, 1:H+1, :2]

    for t in range(H):
        p = P[:, t, :]

        if use_mean_reference:
            mu = mode.mean_path[t]
            Sigma = mode.cov_blocks[t] + (cfg.sigma_floor ** 2) * np.eye(2)

            E = p - mu[None, :]
            if use_gaussian_tracking:
                inv_Sigma = capped_inverse_covariance(Sigma, cfg.sigma_floor, cfg.max_precision)
                costs += cfg.w_reference_tracking * np.einsum("ni,ij,nj->n", E, inv_Sigma, E)
            else:
                costs += cfg.w_reference_tracking * 4.0 * np.sum(E ** 2, axis=1)

            if t < H - 1:
                tangent = mode.mean_path[t + 1] - mode.mean_path[t]
                if np.linalg.norm(tangent) > 1e-9:
                    ref_heading = math.atan2(tangent[1], tangent[0])
                    costs += cfg.w_heading * wrap_angle(X[:, t + 1, 2] - ref_heading) ** 2
        else:
            Sigma = (cfg.sigma_floor ** 2) * np.eye(2)

        for center, radius in obstacle_circles:
            dvec = p - center[None, :]
            norm = np.linalg.norm(dvec, axis=1) + 1e-12
            normal = dvec / norm[:, None]
            d = norm - radius

            margin = cfg.robot_radius + cfg.base_safety_margin
            if use_uncertainty_margin and use_mean_reference:
                sigma_n_sq = np.einsum("ni,ij,nj->n", normal, Sigma, normal)
                sigma_n = np.sqrt(np.maximum(0.0, sigma_n_sq))
                margin = margin + cfg.uncertainty_margin_gain * sigma_n

            costs += cfg.w_obstacle * softplus(8.0 * (margin - d)) ** 2

    costs += cfg.w_goal * np.sum((X[:, -1, :2] - goal[None, :]) ** 2, axis=1)
    costs += cfg.w_control * np.sum(U[:, :, 0] ** 2 + 0.15 * U[:, :, 1] ** 2, axis=1)

    dU = np.diff(U, axis=1)
    costs += cfg.w_control_smooth * np.sum(
        cfg.smooth_v_weight * dU[:, :, 0] ** 2
        + cfg.smooth_omega_weight * dU[:, :, 1] ** 2,
        axis=1,
    )

    if use_mode_prior:
        costs += cfg.w_mode_prior * (-math.log(mode.probability + 1e-12))

    return costs + interpolated_obstacle_penalty(X, obstacle_circles, cfg)


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
            float(cfg.robot_radius),
            float(cfg.base_safety_margin),
            float(cfg.w_goal),
            float(cfg.w_obstacle),
            float(cfg.w_control),
            float(cfg.w_control_smooth),
        )
        return costs + interpolated_obstacle_penalty(X, obstacle_circles, cfg)

    N, H, _ = U.shape
    costs = np.zeros(N, dtype=np.float64)
    P = X[:, 1:H+1, :2]

    for t in range(H):
        p = P[:, t, :]
        for center, radius in obstacle_circles:
            d = np.linalg.norm(p - center[None, :], axis=1) - radius
            margin = cfg.robot_radius + cfg.base_safety_margin
            costs += cfg.w_obstacle * softplus(8.0 * (margin - d)) ** 2

    costs += cfg.w_goal * np.sum((X[:, -1, :2] - goal[None, :]) ** 2, axis=1)
    costs += cfg.w_control * np.sum(U[:, :, 0] ** 2 + 0.15 * U[:, :, 1] ** 2, axis=1)

    dU = np.diff(U, axis=1)
    costs += cfg.w_control_smooth * np.sum(dU[:, :, 0] ** 2 + 0.2 * dU[:, :, 1] ** 2, axis=1)

    return costs + interpolated_obstacle_penalty(X, obstacle_circles, cfg)




# =============================================================================
# Stable representation MPPI variants
# =============================================================================

REP_NONE = 0
REP_CORRIDOR = 1
REP_FRENET = 2
REP_HEATMAP = 3
REP_CONTROL_BANK = 4


def corridor_radius_from_mode(mode: MPPIHomotopyMode, cfg: MPPIConfig) -> Array:
    """
    Convert per-time covariance blocks into a scalar corridor radius.

    This uses covariance as corridor width, not inverse precision. It is
    therefore much more stable than Mahalanobis trajectory tracking.
    """
    K = mode.cov_blocks.shape[0]
    r = np.zeros(K, dtype=np.float64)
    for t in range(K):
        S = 0.5 * (mode.cov_blocks[t] + mode.cov_blocks[t].T)
        vals = np.linalg.eigvalsh(S)
        spread = math.sqrt(max(float(np.max(vals)), 0.0) + cfg.sigma_floor ** 2)
        r[t] = np.clip(
            cfg.corridor_radius_base + cfg.corridor_radius_scale * spread,
            cfg.corridor_radius_min,
            cfg.corridor_radius_max,
        )
    return r


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
    radius = corridor_radius_from_mode(mode, cfg)

    if stable_representation_costs_nb is not None:
        centers, radii = obstacle_circles_to_arrays(obstacle_circles)
        costs = stable_representation_costs_nb(
            np.asarray(X, dtype=np.float64),
            np.asarray(U, dtype=np.float64),
            np.asarray(mode.mean_path, dtype=np.float64),
            np.asarray(radius, dtype=np.float64),
            float(mode.probability),
            centers,
            radii,
            np.asarray(goal, dtype=np.float64),
            int(cfg.horizon),
            float(cfg.robot_radius),
            float(cfg.base_safety_margin),
            float(cfg.w_goal),
            float(cfg.w_obstacle),
            float(cfg.w_control),
            float(cfg.w_control_smooth),
            float(cfg.smooth_v_weight),
            float(cfg.smooth_omega_weight),
            float(cfg.w_corridor),
            float(cfg.w_heatmap),
            float(cfg.heatmap_sigma_scale),
            float(cfg.w_mode_prior),
            int(rep_type),
            bool(use_mode_prior),
        )
        return costs + interpolated_obstacle_penalty(X, obstacle_circles, cfg)

    N, H, _ = U.shape
    costs = np.zeros(N, dtype=np.float64)
    P = X[:, 1:H+1, :2]

    for t in range(H):
        p = P[:, t, :]
        mu = mode.mean_path[t]
        e = p - mu[None, :]
        r = float(radius[t])

        if rep_type == REP_CORRIDOR:
            d = np.linalg.norm(e, axis=1)
            costs += cfg.w_corridor * softplus(6.0 * (d - r)) ** 2

        elif rep_type == REP_FRENET:
            if t < H - 1:
                tangent = mode.mean_path[t + 1] - mode.mean_path[t]
            else:
                tangent = mode.mean_path[t] - mode.mean_path[t - 1]
            nt = np.linalg.norm(tangent) + 1e-12
            normal = np.array([-tangent[1] / nt, tangent[0] / nt], dtype=np.float64)
            lateral = np.abs(e @ normal)
            costs += cfg.w_corridor * softplus(6.0 * (lateral - r)) ** 2

        elif rep_type == REP_HEATMAP:
            d = np.linalg.norm(e, axis=1)
            sigma = cfg.heatmap_sigma_scale * (r + 1e-9)
            costs += cfg.w_heatmap * (1.0 - np.exp(-0.5 * (d / sigma) ** 2))

        for center, radius_obs in obstacle_circles:
            d_obs = np.linalg.norm(p - center[None, :], axis=1) - radius_obs
            margin = cfg.robot_radius + cfg.base_safety_margin
            costs += cfg.w_obstacle * softplus(8.0 * (margin - d_obs)) ** 2

    costs += cfg.w_goal * np.sum((X[:, -1, :2] - goal[None, :]) ** 2, axis=1)
    costs += cfg.w_control * np.sum(U[:, :, 0] ** 2 + 0.15 * U[:, :, 1] ** 2, axis=1)

    dU = np.diff(U, axis=1)
    costs += cfg.w_control_smooth * np.sum(
        cfg.smooth_v_weight * dU[:, :, 0] ** 2
        + cfg.smooth_omega_weight * dU[:, :, 1] ** 2,
        axis=1,
    )

    if use_mode_prior:
        costs += cfg.w_mode_prior * (-math.log(mode.probability + 1e-12))

    return costs + interpolated_obstacle_penalty(X, obstacle_circles, cfg)



def softmin_score(costs: Array, cfg: MPPIConfig) -> float:
    finite = np.asarray(costs, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float('inf')
    rho = float(np.min(finite))
    z = np.exp(-(finite - rho) / cfg.lambda_temperature)
    return float(rho - cfg.lambda_temperature * math.log(np.mean(z) + 1e-12))

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
    if use_mean_nominal:
        mean_nominal = nominal_controls_to_track_path(x_current, local_mode.mean_path, cfg)
    else:
        mean_nominal = nominal_controls_to_goal(x_current, goal, cfg)

    if not use_empirical_init:
        return [mean_nominal]

    return build_empirical_nominal_bank(
        x_current=x_current,
        global_mode=global_mode,
        mean_nominal=mean_nominal,
        cfg=cfg,
        rng=rng,
        previous_idx=previous_idx,
    )


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
    U += make_temporally_correlated_noise(n, cfg.horizon, cfg, rng)
    U[:, :, 0] = np.clip(U[:, :, 0], cfg.v_min, cfg.v_max)
    U[:, :, 1] = np.clip(U[:, :, 1], cfg.omega_min, cfg.omega_max)
    return U



def stable_swarm_mppi_step(
    x_current: Array,
    global_modes: List[MPPIHomotopyMode],
    obstacles: Sequence,
    goal: Array,
    cfg: MPPIConfig,
    rng: np.random.Generator,
    *,
    rep_type: int,
    use_pi_sampling: bool,
    use_empirical_init: bool,
    use_mean_nominal: bool,
    use_mode_prior: bool,
    progress_by_mode: Optional[Dict[str, int]] = None,
) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    progress_by_mode = {} if progress_by_mode is None else dict(progress_by_mode)

    local_modes = []
    new_progress_by_mode = dict(progress_by_mode)
    for mode in global_modes:
        key = str(mode.signature)
        previous = progress_by_mode.get(key)
        local_mode, index = localize_mode_for_state_with_index(
            mode,
            x_current,
            cfg.horizon,
            previous_idx=previous if cfg.use_monotonic_reference_progress else None,
            max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None,
        )
        local_modes.append(local_mode)
        new_progress_by_mode[key] = index

    # Hard runs do not use a circular obstacle approximation in the objective.
    objective_obstacles = obstacle_bounding_circles(obstacles) if cfg.use_soft_obstacle_cost else []
    active_mode_indices = list(range(len(local_modes)))
    active_local_modes = local_modes
    active_global_modes = global_modes

    if use_pi_sampling:
        probabilities = np.asarray([m.probability for m in active_local_modes], dtype=np.float64)
        probabilities /= probabilities.sum() + 1e-12
    else:
        probabilities = np.ones(len(active_local_modes), dtype=np.float64) / len(active_local_modes)

    mode_ids = rng.choice(len(active_local_modes), size=cfg.num_rollouts, p=probabilities)
    all_costs = np.full(cfg.num_rollouts, np.inf, dtype=np.float64)
    all_U = np.zeros((cfg.num_rollouts, cfg.horizon, 2), dtype=np.float64)
    all_X = np.zeros((cfg.num_rollouts, cfg.horizon + 1, 3), dtype=np.float64)

    for mode_id, local_mode in enumerate(active_local_modes):
        ids = np.where(mode_ids == mode_id)[0]
        if ids.size == 0:
            continue
        key = str(active_global_modes[mode_id].signature)
        nominal_bank = build_nominal_bank_for_mode(
            x_current,
            local_mode,
            active_global_modes[mode_id],
            goal,
            cfg,
            rng,
            use_empirical_init=use_empirical_init,
            use_mean_nominal=use_mean_nominal,
            previous_idx=progress_by_mode.get(key),
        )
        U = sample_controls_from_nominal_bank(
            nominal_bank,
            int(ids.size),
            cfg,
            rng,
            prefer_empirical=use_empirical_init,
        )
        X = rollout_unicycle_batch(x_current, U, cfg.dt)
        costs = stable_representation_costs(
            X,
            U,
            local_mode,
            objective_obstacles,
            goal,
            cfg,
            rep_type=rep_type,
            use_mode_prior=use_mode_prior,
        )
        all_costs[ids] = costs
        all_U[ids] = U
        all_X[ids] = X

    control, action_info, constrained = select_hard_feasible_action(
        all_costs, all_U, all_X, x_current, obstacles, goal, cfg
    )
    cost_min, cost_mean = finite_cost_statistics(constrained)
    info = {
        'cost_min': cost_min,
        'cost_mean': cost_mean,
        'soft_value': softmin_score(constrained, cfg),
        'rep_type': int(rep_type),
        'mode_selection': False,
        'active_mode_count': int(len(active_mode_indices)),
        'suppressed_mode_count': 0,
        'mode_clearances': [],
        **action_info,
    }
    return control, info, new_progress_by_mode


def mode_selecting_stable_mppi_step(
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
) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    """Optimize retained homotopy modes independently and select a feasible mode."""
    progress_by_mode = {} if progress_by_mode is None else dict(progress_by_mode)
    top_k = min(max(1, int(cfg.mode_select_top_k)), len(global_modes))
    candidate_modes = global_modes[:top_k]
    n_per_mode = max(
        int(cfg.mode_select_min_rollouts_per_mode),
        int(math.ceil(cfg.num_rollouts / float(top_k))),
    )
    objective_obstacles = obstacle_bounding_circles(obstacles) if cfg.use_soft_obstacle_cost else []
    new_progress_by_mode = dict(progress_by_mode)
    best = None

    for mode_index, global_mode in enumerate(candidate_modes):
        key = str(global_mode.signature)
        previous = progress_by_mode.get(key)
        local_mode, index = localize_mode_for_state_with_index(
            global_mode,
            x_current,
            cfg.horizon,
            previous_idx=previous if cfg.use_monotonic_reference_progress else None,
            max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None,
        )
        new_progress_by_mode[key] = index
        nominal_bank = build_nominal_bank_for_mode(
            x_current,
            local_mode,
            global_mode,
            goal,
            cfg,
            rng,
            use_empirical_init=use_empirical_init,
            use_mean_nominal=use_mean_nominal,
            previous_idx=previous,
        )
        U = sample_controls_from_nominal_bank(
            nominal_bank,
            n_per_mode,
            cfg,
            rng,
            prefer_empirical=use_empirical_init,
        )
        X = rollout_unicycle_batch(x_current, U, cfg.dt)
        costs = stable_representation_costs(
            X,
            U,
            local_mode,
            objective_obstacles,
            goal,
            cfg,
            rep_type=rep_type,
            use_mode_prior=use_mode_prior,
        )
        control, action_info, constrained = select_hard_feasible_action(
            costs, U, X, x_current, obstacles, goal, cfg
        )
        score = softmin_score(constrained, cfg)
        cost_min, cost_mean = finite_cost_statistics(constrained)
        record = {
            'score': score,
            'u': control,
            'mode_index': int(mode_index),
            'signature': str(global_mode.signature),
            'probability': float(global_mode.probability),
            'cost_min': cost_min,
            'cost_mean': cost_mean,
            **action_info,
        }
        if best is None or score < best['score']:
            best = record

    assert best is not None
    info = {
        'cost_min': best['cost_min'],
        'cost_mean': best['cost_mean'],
        'soft_value': best['score'],
        'selected_mode_index': best['mode_index'],
        'selected_mode_signature': best['signature'],
        'selected_mode_probability': best['probability'],
        'rep_type': int(rep_type),
        'mode_selection': True,
        'active_mode_count': int(top_k),
        'suppressed_mode_count': 0,
        'mode_clearances': [],
        'mppi_action_rule': best.get('mppi_action_rule'),
        'num_feasible_rollouts': best.get('num_feasible_rollouts', 0),
        'optimal_traj': best.get('optimal_traj'),
    }
    return best['u'], info, new_progress_by_mode

def make_temporally_correlated_noise(n: int, H: int, cfg: MPPIConfig, rng: np.random.Generator) -> Array:
    noise_scale = np.array([cfg.noise_v, cfg.noise_omega], dtype=np.float64)
    noise = rng.normal(size=(n, H, 2)) * noise_scale[None, None, :]
    alpha = cfg.temporal_noise_smoothing
    for t in range(1, H):
        noise[:, t, :] = alpha * noise[:, t - 1, :] + (1.0 - alpha) * noise[:, t, :]
    return noise


def mppi_weighted_control(costs: Array, U0: Array, cfg: MPPIConfig) -> Array:
    rho = float(costs.min())
    weights = np.exp(-(costs - rho) / cfg.lambda_temperature)
    weights = weights / (weights.sum() + 1e-12)
    u = weights @ U0
    u[0] = np.clip(u[0], cfg.v_min, cfg.v_max)
    u[1] = np.clip(u[1], cfg.omega_min, cfg.omega_max)
    return u


def best_output_trajectory_from_costs(costs: Array, X: Array) -> Array:
    """Return the lowest-cost predicted output trajectory from an MPPI batch."""
    best_idx = int(np.argmin(costs))
    return np.asarray(X[best_idx], dtype=np.float64).copy()



def swarm_mppi_step(
    x_current: Array,
    global_modes: List[MPPIHomotopyMode],
    obstacles: Sequence,
    goal: Array,
    cfg: MPPIConfig,
    rng: np.random.Generator,
    *,
    use_pi_sampling: bool,
    use_empirical_init: bool,
    use_mean_reference: bool,
    use_gaussian_tracking: bool,
    use_uncertainty_margin: bool,
    use_mode_prior: bool,
    progress_by_mode: Optional[Dict[str, int]] = None,
) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    progress_by_mode = {} if progress_by_mode is None else dict(progress_by_mode)
    local_modes = []
    new_progress_by_mode = dict(progress_by_mode)
    for mode in global_modes:
        key = str(mode.signature)
        previous = progress_by_mode.get(key)
        local_mode, index = localize_mode_for_state_with_index(
            mode,
            x_current,
            cfg.horizon,
            previous_idx=previous if cfg.use_monotonic_reference_progress else None,
            max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None,
        )
        local_modes.append(local_mode)
        new_progress_by_mode[key] = index

    objective_obstacles = obstacle_bounding_circles(obstacles) if cfg.use_soft_obstacle_cost else []
    if use_pi_sampling:
        probabilities = np.asarray([m.probability for m in local_modes], dtype=np.float64)
        probabilities /= probabilities.sum() + 1e-12
    else:
        probabilities = np.ones(len(local_modes), dtype=np.float64) / len(local_modes)

    mode_ids = rng.choice(len(local_modes), size=cfg.num_rollouts, p=probabilities)
    all_costs = np.full(cfg.num_rollouts, np.inf, dtype=np.float64)
    all_U = np.zeros((cfg.num_rollouts, cfg.horizon, 2), dtype=np.float64)
    all_X = np.zeros((cfg.num_rollouts, cfg.horizon + 1, 3), dtype=np.float64)

    for mode_id, mode in enumerate(local_modes):
        ids = np.where(mode_ids == mode_id)[0]
        if ids.size == 0:
            continue
        if use_mean_reference:
            mean_nominal = nominal_controls_to_track_path(x_current, mode.mean_path, cfg)
        else:
            mean_nominal = nominal_controls_to_goal(x_current, goal, cfg)
        if use_empirical_init:
            nominal_bank = build_empirical_nominal_bank(
                x_current=x_current,
                global_mode=global_modes[mode_id],
                mean_nominal=mean_nominal,
                cfg=cfg,
                rng=rng,
                previous_idx=progress_by_mode.get(str(global_modes[mode_id].signature)),
            )
        else:
            nominal_bank = [mean_nominal]

        U = sample_controls_from_nominal_bank(
            nominal_bank,
            int(ids.size),
            cfg,
            rng,
            prefer_empirical=use_empirical_init,
        )
        X = rollout_unicycle_batch(x_current, U, cfg.dt)
        costs = fast_swarm_prior_costs(
            X,
            U,
            mode,
            objective_obstacles,
            goal,
            cfg,
            use_gaussian_tracking=use_gaussian_tracking,
            use_uncertainty_margin=use_uncertainty_margin,
            use_mode_prior=use_mode_prior,
            use_mean_reference=use_mean_reference,
        )
        all_costs[ids] = costs
        all_U[ids] = U
        all_X[ids] = X

    control, action_info, constrained = select_hard_feasible_action(
        all_costs, all_U, all_X, x_current, obstacles, goal, cfg
    )
    cost_min, cost_mean = finite_cost_statistics(constrained)
    info = {
        'cost_min': cost_min,
        'cost_mean': cost_mean,
        'active_mode_count': int(len(local_modes)),
        'suppressed_mode_count': 0,
        'mode_clearances': [],
        **action_info,
    }
    return control, info, new_progress_by_mode


def standard_mppi_step(
    x_current: Array,
    obstacles: Sequence,
    goal: Array,
    cfg: MPPIConfig,
    rng: np.random.Generator,
) -> Tuple[Array, Dict[str, object]]:
    objective_obstacles = obstacle_bounding_circles(obstacles) if cfg.use_soft_obstacle_cost else []
    nominal = nominal_controls_to_goal(x_current, goal, cfg)
    U = np.repeat(nominal[None, :, :], cfg.num_rollouts, axis=0)
    U += make_temporally_correlated_noise(cfg.num_rollouts, cfg.horizon, cfg, rng)
    U[:, :, 0] = np.clip(U[:, :, 0], cfg.v_min, cfg.v_max)
    U[:, :, 1] = np.clip(U[:, :, 1], cfg.omega_min, cfg.omega_max)
    X = rollout_unicycle_batch(x_current, U, cfg.dt)
    costs = standard_mppi_costs_batch(X, U, objective_obstacles, goal, cfg)
    control, action_info, constrained = select_hard_feasible_action(
        costs, U, X, x_current, obstacles, goal, cfg
    )
    cost_min, cost_mean = finite_cost_statistics(constrained)
    return control, {
        'cost_min': cost_min,
        'cost_mean': cost_mean,
        **action_info,
    }

# =============================================================================
# Scene and swarm planner
# =============================================================================

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
    base_action = pickle.load(open("save/best_policy.pkl", "rb"))["best_theta"]

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


# =============================================================================
# Controller runners and metrics
# =============================================================================

def initial_pose(start: Array, goal: Array) -> Array:
    direction = goal - start
    heading = math.atan2(direction[1], direction[0])
    return np.array([start[0], start[1], heading], dtype=np.float64)


def run_controller_variant(
    variant: ControllerVariant,
    modes: List[MPPIHomotopyMode],
    obstacles: Sequence,
    start: Array,
    goal: Array,
    *,
    seed: int,
    max_steps: int = 120,
    goal_tolerance: float = 0.15,
    mppi_cfg: Optional[MPPIConfig] = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    mppi_cfg = MPPIConfig() if mppi_cfg is None else mppi_cfg

    x = initial_pose(start, goal)
    states = [x.copy()]
    controls = []
    infos = []
    previous_control = None

    swarm_progress = {}
    obstacle_circles = obstacle_bounding_circles(obstacles)

    t0 = time.perf_counter()

    for _ in range(max_steps):
        if variant == ControllerVariant.FULL_SWARM_PRIOR_MPPI:
            u, info, swarm_progress = swarm_mppi_step(
                x, modes, obstacles, goal, mppi_cfg, rng,
                use_pi_sampling=True,
                use_empirical_init=True,
                use_mean_reference=True,
                use_gaussian_tracking=True,
                use_uncertainty_margin=True,
                use_mode_prior=True,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.GAUSSIAN_PRIOR_MPPI:
            u, info, swarm_progress = swarm_mppi_step(
                x, modes, obstacles, goal, mppi_cfg, rng,
                use_pi_sampling=True,
                use_empirical_init=False,
                use_mean_reference=True,
                use_gaussian_tracking=True,
                use_uncertainty_margin=True,
                use_mode_prior=True,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.EMPIRICAL_INIT_MPPI:
            u, info, swarm_progress = swarm_mppi_step(
                x, modes, obstacles, goal, mppi_cfg, rng,
                use_pi_sampling=False,
                use_empirical_init=True,
                use_mean_reference=False,
                use_gaussian_tracking=False,
                use_uncertainty_margin=False,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.HOMOTOPY_SEEDED_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, obstacles, goal, mppi_cfg, rng,
                rep_type=REP_NONE,
                use_pi_sampling=True,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.CORRIDOR_PRIOR_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, obstacles, goal, mppi_cfg, rng,
                rep_type=REP_CORRIDOR,
                use_pi_sampling=True,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.FRENET_CORRIDOR_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, obstacles, goal, mppi_cfg, rng,
                rep_type=REP_FRENET,
                use_pi_sampling=True,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.HEATMAP_PRIOR_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, obstacles, goal, mppi_cfg, rng,
                rep_type=REP_HEATMAP,
                use_pi_sampling=True,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.CONTROL_BANK_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, obstacles, goal, mppi_cfg, rng,
                rep_type=REP_CONTROL_BANK,
                use_pi_sampling=False,
                use_empirical_init=True,
                use_mean_nominal=False,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.MODE_SELECTING_HOMOTOPY_MPPI:
            u, info, swarm_progress = mode_selecting_stable_mppi_step(
                x, modes, obstacles, goal, mppi_cfg, rng,
                rep_type=REP_NONE,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.MODE_SELECTING_CORRIDOR_MPPI:
            u, info, swarm_progress = mode_selecting_stable_mppi_step(
                x, modes, obstacles, goal, mppi_cfg, rng,
                rep_type=REP_CORRIDOR,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.STANDARD_MPPI:
            u, info = standard_mppi_step(x, obstacles, goal, mppi_cfg, rng)

        else:
            raise ValueError(f"Unknown variant {variant}")

        if "mppi" in variant.value and mppi_cfg.apply_control_lowpass and previous_control is not None:
            alpha = float(mppi_cfg.control_lowpass_alpha)
            u = alpha * previous_control + (1.0 - alpha) * u
            u[0] = np.clip(u[0], mppi_cfg.v_min, mppi_cfg.v_max)
            u[1] = np.clip(u[1], mppi_cfg.omega_min, mppi_cfg.omega_max)

        shielded_control, shield_info = shield_control_if_unsafe(
            x, u, obstacles, goal, mppi_cfg
        )
        u = shielded_control
        if shield_info is not None and isinstance(info, dict):
            info.update(shield_info)

        previous_control = u.copy()

        x = unicycle_step(x, u, mppi_cfg.dt)

        states.append(x.copy())
        controls.append(u.copy())
        infos.append(info)

        if np.linalg.norm(x[:2] - goal) <= goal_tolerance:
            break

    runtime = time.perf_counter() - t0

    return {
        "variant": variant.value,
        "seed": seed,
        "states": np.asarray(states),
        "controls": np.asarray(controls),
        "infos": infos,
        "runtime": runtime,
    }


def summarize_result(result: Dict[str, object], obstacles, goal, robot_radius: float, goal_tolerance: float = 0.35):
    states = result["states"]
    controls = result["controls"]
    final_dist = float(np.linalg.norm(states[-1, :2] - goal))
    collision = path_collided(states, obstacles, robot_radius)

    return {
        "variant": result["variant"],
        "seed": result["seed"],
        "success": bool(final_dist <= goal_tolerance and not collision),
        "reached_goal": bool(final_dist <= goal_tolerance),
        "collision": bool(collision),
        "final_dist": final_dist,
        "min_clearance": min_clearance(states, obstacles, robot_radius),
        "path_length": path_length(states),
        "control_effort": control_effort(controls),
        "control_smoothness": control_smoothness(controls),
        "steps": int(len(states) - 1),
        "runtime_sec": float(result["runtime"]),
        "runtime_per_step_sec": float(result["runtime"] / max(1, len(states) - 1)),
    }


# =============================================================================
# Plotting
# =============================================================================

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


# =============================================================================
# Dynamic blockage experiment
# =============================================================================

def obstacle_center(obs) -> Array:
    """Return the geometric center of a polygon obstacle."""
    return _poly_vertices(obs).mean(axis=0)


def translate_obstacle_to_center(obs, target_center: Array):
    """Return a copy of ``obs`` translated so its center is ``target_center``."""
    vertices = _poly_vertices(obs).copy()
    shift = np.asarray(target_center, dtype=np.float64) - vertices.mean(axis=0)
    return PolyObstacle(vertices + shift[None, :])


def random_obstacle_center_swap(
    obstacles: Sequence,
    *,
    seed: int,
) -> Tuple[List[object], Tuple[int, ...]]:
    """Randomly permute obstacle centers while preserving obstacle shapes.

    Obstacle ``i`` keeps its original polygon shape but is translated to the
    original center of obstacle ``permutation[i]``. A Sattolo shuffle is used,
    so every obstacle moves to a different center when at least two obstacles
    are present. The returned permutation makes each trial layout reproducible.
    """
    n = len(obstacles)
    if n < 2:
        return list(obstacles), tuple(range(n))

    rng = np.random.default_rng(int(seed))
    permutation = np.arange(n, dtype=np.int64)

    # Sattolo's algorithm: one random cycle, hence no fixed points.
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
    """Serialize an obstacle-to-original-center assignment for the trial CSV."""
    return ";".join(f"{i}->{int(target)}" for i, target in enumerate(permutation))


def make_wall_between_points(p0: Array, p1: Array, width: float = 0.35, extension: float = 0.0):
    """
    Create a rectangular wall obstacle between two points.

    Args:
        p0, p1:
            Endpoints of the wall centerline.
        width:
            Wall thickness, perpendicular to the segment p0-p1.
        extension:
            Optional extension at both ends along the segment direction. This is
            useful if the wall should overlap slightly with the two obstacles it
            connects, preventing small gaps.

    Returns:
        PolyObstacle with four vertices.
    """
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
    """
    Create a wall from the center of obstacle idx_a to the center of obstacle idx_b.

    Indices are zero-based. For example:
        (0, 1) connects obstacle 1 and obstacle 2 in human counting.
        (1, 2) connects obstacle 2 and obstacle 3 in human counting.
    """
    c0 = obstacle_center(obstacles[idx_a])
    c1 = obstacle_center(obstacles[idx_b])
    return make_wall_between_points(c0, c1, width=width, extension=extension)


def make_wall_blockers_between_obstacles(
    obstacles: Sequence,
    pairs: Sequence[Tuple[int, int]],
    width: float = 0.35,
    extension: float = 0.15,
):
    """
    Create one or more wall blockers between obstacle-center pairs.

    Example:
        pairs=[(0, 1), (1, 2)]
    creates:
        wall from obstacle 1 center to obstacle 2 center
        wall from obstacle 2 center to obstacle 3 center
    """
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
    """Create wall blockers between fixed spatial center anchors.

    Unlike ``make_wall_blockers_between_obstacles``, this helper does not inspect
    the current obstacle layout. It is therefore the preferred constructor when
    obstacle shapes are later permuted between center locations but the walls
    must remain on the same original center-to-center segments.
    """
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
    """Legacy step-based obstacle activation helper used by plotting utilities."""
    if step >= block_step:
        return list(base_obstacles) + as_blocker_list(blocker)
    return list(base_obstacles)


def spatial_progress_along_start_goal(x: Array, start: Array, goal: Array) -> float:
    """Return normalized projection progress along the start-to-goal direction."""
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
    trigger_progress: Optional[float] = 0.25,
    activation_preview_clearance: Optional[float] = 0.75,
    blocker_active_from_start: bool = False,
    condition: str = "dynamic_wall",
    block_step: Optional[int] = None,
    max_steps: int = 120,
    goal_tolerance: float = 0.35,
    mppi_cfg: Optional[MPPIConfig] = None,
    record_infos: bool = True,
    record_obstacle_history: bool = True,
):
    """Simulate one controller under no-wall, static-wall, or dynamic-wall conditions.

    Dynamic walls activate at the earlier of the spatial progress trigger or a
    clearance-preview trigger. The latter inserts the wall before the robot gets
    closer than the configurable reaction distance. ``block_step`` is retained
    only for backwards compatibility.
    """
    rng = np.random.default_rng(seed)
    mppi_cfg = MPPIConfig() if mppi_cfg is None else mppi_cfg

    x = initial_pose(start, goal)
    states = [x.copy()]
    controls = []
    infos = []
    obstacle_history = []
    previous_control = None
    swarm_progress = {}
    selected_mode_switches = 0
    last_selected_mode = None
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
        )

    t0 = time.perf_counter()

    for step in range(max_steps):
        current_progress = spatial_progress_along_start_goal(x, start, goal)

        if activation_step is None and blockers:
            blocker_clearance = min_clearance(
                x[None, :],
                blockers,
                mppi_cfg.robot_radius,
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
        if record_obstacle_history:
            obstacle_history.append(active_obstacles)

        if variant == ControllerVariant.FULL_SWARM_PRIOR_MPPI:
            u, info, swarm_progress = swarm_mppi_step(
                x, modes, active_obstacles, goal, mppi_cfg, rng,
                use_pi_sampling=True,
                use_empirical_init=True,
                use_mean_reference=True,
                use_gaussian_tracking=True,
                use_uncertainty_margin=True,
                use_mode_prior=True,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.GAUSSIAN_PRIOR_MPPI:
            u, info, swarm_progress = swarm_mppi_step(
                x, modes, active_obstacles, goal, mppi_cfg, rng,
                use_pi_sampling=True,
                use_empirical_init=False,
                use_mean_reference=True,
                use_gaussian_tracking=True,
                use_uncertainty_margin=True,
                use_mode_prior=True,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.EMPIRICAL_INIT_MPPI:
            u, info, swarm_progress = swarm_mppi_step(
                x, modes, active_obstacles, goal, mppi_cfg, rng,
                use_pi_sampling=False,
                use_empirical_init=True,
                use_mean_reference=False,
                use_gaussian_tracking=False,
                use_uncertainty_margin=False,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.HOMOTOPY_SEEDED_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, active_obstacles, goal, mppi_cfg, rng,
                rep_type=REP_NONE,
                use_pi_sampling=True,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.CORRIDOR_PRIOR_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, active_obstacles, goal, mppi_cfg, rng,
                rep_type=REP_CORRIDOR,
                use_pi_sampling=True,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.FRENET_CORRIDOR_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, active_obstacles, goal, mppi_cfg, rng,
                rep_type=REP_FRENET,
                use_pi_sampling=True,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.HEATMAP_PRIOR_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, active_obstacles, goal, mppi_cfg, rng,
                rep_type=REP_HEATMAP,
                use_pi_sampling=True,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.CONTROL_BANK_MPPI:
            u, info, swarm_progress = stable_swarm_mppi_step(
                x, modes, active_obstacles, goal, mppi_cfg, rng,
                rep_type=REP_CONTROL_BANK,
                use_pi_sampling=False,
                use_empirical_init=True,
                use_mean_nominal=False,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.MODE_SELECTING_HOMOTOPY_MPPI:
            u, info, swarm_progress = mode_selecting_stable_mppi_step(
                x, modes, active_obstacles, goal, mppi_cfg, rng,
                rep_type=REP_NONE,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.MODE_SELECTING_CORRIDOR_MPPI:
            u, info, swarm_progress = mode_selecting_stable_mppi_step(
                x, modes, active_obstacles, goal, mppi_cfg, rng,
                rep_type=REP_CORRIDOR,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.STANDARD_MPPI:
            u, info = standard_mppi_step(x, active_obstacles, goal, mppi_cfg, rng)

        else:
            raise ValueError(f"Unsupported variant: {variant}")

        if "mppi" in variant.value and mppi_cfg.apply_control_lowpass and previous_control is not None:
            alpha = float(mppi_cfg.control_lowpass_alpha)
            u = alpha * previous_control + (1.0 - alpha) * u
            u[0] = np.clip(u[0], mppi_cfg.v_min, mppi_cfg.v_max)
            u[1] = np.clip(u[1], mppi_cfg.omega_min, mppi_cfg.omega_max)

        shielded_control, shield_info = shield_control_if_unsafe(
            x, u, active_obstacles, goal, mppi_cfg
        )
        u = shielded_control
        if shield_info is not None and isinstance(info, dict):
            info.update(shield_info)

        previous_control = u.copy()
        x = unicycle_step(x, u, mppi_cfg.dt)
        states.append(x.copy())
        controls.append(u.copy())

        selected_mode = info.get("selected_mode_index") if isinstance(info, dict) else None
        if selected_mode is not None:
            if last_selected_mode is not None and selected_mode != last_selected_mode:
                selected_mode_switches += 1
            last_selected_mode = selected_mode

        if record_infos:
            infos.append(info)

        if np.linalg.norm(x[:2] - goal) <= goal_tolerance:
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
        "selected_mode_switches": int(selected_mode_switches),
        "last_selected_mode": last_selected_mode,
    }


def summarize_dynamic_result(result, base_obstacles, blocker, goal, robot_radius, goal_tolerance=0.15):
    """Summarize a trial and classify every unsuccessful run."""
    states = result["states"]
    controls = result["controls"]
    condition = str(result.get("condition", "dynamic_wall"))
    activation_step = result.get("activation_step")
    if activation_step is not None:
        activation_step = int(activation_step)

    # For static-wall baselines, the complete trajectory is considered the
    # post-wall interval. No-wall baselines have no post-wall interval.
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
        clearance = min_clearance(states[step:step + 1], active_obs, robot_radius)
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
            segment_clearance = min_clearance(segment_states, segment_obs, robot_radius)
            min_vals.append(segment_clearance)
            if metric_start_step is not None and step >= metric_start_step:
                min_vals_after_block.append(segment_clearance)
            if segment_clearance < 0.0 and first_collision_step is None:
                collision = True
                first_collision_step = step

    final_dist = float(np.linalg.norm(states[-1, :2] - goal))
    reached_goal = bool(final_dist <= goal_tolerance)
    success = bool(reached_goal and not collision)

    if success:
        failure_reason = ""
    elif collision:
        failure_reason = "collision"
    else:
        failure_reason = "not_reaching"

    selected_modes = []
    for info in result.get("infos", []):
        if isinstance(info, dict) and "selected_mode_index" in info:
            selected_modes.append(info.get("selected_mode_index"))

    if selected_modes:
        selected_mode_switches = sum(a != b for a, b in zip(selected_modes[:-1], selected_modes[1:]))
        last_selected_mode = selected_modes[-1]
    else:
        selected_mode_switches = int(result.get("selected_mode_switches", 0))
        last_selected_mode = result.get("last_selected_mode")

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
        "selected_mode_switches": int(selected_mode_switches),
        "last_selected_mode": last_selected_mode,
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
    """
    Assign one stable color per variant.

    Uses tab20 for up to 20 variants. If there are more than 20, falls back to
    evenly spaced HSV colors.
    """
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

    # Draw dynamic blocker walls distinctly.
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

        # Before blockage: dashed, same color.
        ax.plot(
            states[:b+1, 0],
            states[:b+1, 1],
            linewidth=2.2,
            linestyle="--",
            color=color,
            label=res["variant"],
        )

        # After blockage: solid, same color.
        ax.plot(
            states[b:, 0],
            states[b:, 1],
            linewidth=3.0,
            linestyle="-",
            color=color,
        )

        # Mark the blockage-time state.
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
    """Return a filesystem-safe variant name for image/GIF outputs."""
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def _mode_index_from_info(info: Dict[str, object], modes: List[MPPIHomotopyMode]) -> Optional[int]:
    """
    Recover the homotopy mode index recorded by mode-selecting controllers.

    """
    if not isinstance(info, dict):
        return None

    if "selected_mode_index" in info and info["selected_mode_index"] is not None:
        idx = int(info["selected_mode_index"])
        if 0 <= idx < len(modes):
            return idx

    if "mode" in info and info["mode"] is not None:
        mode_sig = str(info["mode"])
        for i, mode in enumerate(modes):
            if str(mode.signature) == mode_sig:
                return i

    return None


def tracked_reference_for_frame(
    result: Dict[str, object],
    frame: int,
    modes: List[MPPIHomotopyMode],
    goal: Array,
    *,
    mppi_cfg: Optional[MPPIConfig] = None,
) -> Optional[Array]:
    """
    Return the reference path to overlay in the per-variant animation.

    The simulation logic is intentionally unchanged. This function reconstructs
    the visual reference from the recorded state and controller info:
      - mode-selecting controllers use the recorded selected mode;
      - other homotopy/swarm controllers use the nearest homotopy mean at the
        current state, which is the clearest single reference to display when
        MPPI samples multiple homotopies at a step;
      - standard goal-directed controllers show a short current-to-goal guide.
    """
    if not modes:
        return None

    mppi_cfg = MPPIConfig() if mppi_cfg is None else mppi_cfg

    states = result["states"]
    j = min(frame, len(states) - 1)
    x = states[j]
    variant_name = str(result.get("variant", ""))

    if variant_name in {
        ControllerVariant.STANDARD_MPPI.value,
    }:
        return np.vstack([x[:2], np.asarray(goal, dtype=np.float64)])

    info = None
    infos = result.get("infos", [])
    if infos:
        info = infos[min(j, len(infos) - 1)]

    idx = _mode_index_from_info(info, modes)

    if idx is None:
        # Non-mode-selecting MPPI may sample all modes. For a one-at-a-time GIF,
        # the nearest homotopy mean is the most readable single reference.
        d = [float(np.min(np.linalg.norm(mode.mean_path - x[:2], axis=1))) for mode in modes]
        idx = int(np.argmin(d))

    H = mppi_cfg.horizon
    return localize_mode_for_state(modes[idx], x, H).mean_path


def optimal_trajectory_for_frame(result: Dict[str, object], frame: int) -> Optional[Array]:
    """Return the recorded optimal predicted horizon trajectory for a frame."""
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
    """Draw the applied linear control direction at each recorded time step.

    The arrow direction is the robot heading at that step and the arrow length is
    proportional to |v| * dt. Yaw rate is not drawn geometrically; the arrow
    visualizes the translational part of the applied unicycle input.
    """
    if controls is None or len(controls) == 0:
        return

    last = min(int(upto_step), len(controls) - 1, len(states) - 1)
    if last < 0:
        return

    for k in range(0, last + 1, max(1, int(stride))):
        x = states[k]
        u = controls[k]
        v = float(u[0])
        if abs(v) < 1e-9:
            length = min_length
            direction_sign = 1.0
        else:
            length = float(np.clip(abs(v) * dt, min_length, max_length))
            direction_sign = 1.0 if v >= 0.0 else -1.0

        dx = direction_sign * length * math.cos(float(x[2]))
        dy = direction_sign * length * math.sin(float(x[2]))
        ax.arrow(
            float(x[0]),
            float(x[1]),
            dx,
            dy,
            head_width=0.07,
            head_length=0.10,
            length_includes_head=True,
            linewidth=0.8,
            color=color,
            alpha=0.45,
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
    """
    Save one GIF per controller variant.

    The robot path and the tracked reference use the exact same variant color as
    plot_dynamic_paths. The reference is dashed and updated at every frame.
    """
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
            dt=float(dt_for_arrows),
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
    """Save one tracked-reference GIF for each result/variant."""
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
    """
    Save a simple GIF animation of the robot moving and the blocker appearing.
    Requires pillow. If unavailable, this function prints a warning and returns.
    """
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

        # Draw blocker after the common block step.
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
    print("dynamic_block_hard")
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

    # Dynamic wall-blocker configuration.
    # Indices are zero-based:
    #   (0, 1) means "wall from obstacle 1 center to obstacle 2 center"
    #   (1, 2) means "wall from obstacle 2 center to obstacle 3 center"
    #
    # Change these values to choose which corridors close.
    wall_pairs = []
    wall_width = 0.40
    wall_extension = 0.20
    block_step = 150
    seed = 2

    # wall_pairs = [(0, 1), (1, 2)]
    # wall_width = 0.40
    # wall_extension = 0.20
    # block_step = 25
    # seed = 2

    # wall_pairs = [(0, 2), (1, 2)]
    # wall_width = 0.40
    # wall_extension = 0.20
    # block_step = 30
    # seed = 2

    blocker = make_wall_blockers_between_obstacles(
        obstacles=base_obstacles,
        pairs=wall_pairs,
        width=wall_width,
        extension=wall_extension,
    )

    cfg = MPPIConfig(
        horizon=28,
        num_rollouts=650,
        dt=0.12,
        max_empirical_nominals_per_mode=16,
        swarm_init_probability=0.45,
        sigma_floor=0.25,
        max_precision=10.0,
        w_reference_tracking=0.80,
        w_control_smooth=0.40,
        smooth_v_weight=0.5,
        smooth_omega_weight=2.0,
        w_heading=0.0,
        w_mode_prior=0.15,
        uncertainty_margin_gain=0.25,
        use_soft_obstacle_cost=False,
        enable_hard_collision_filter=True,
        hard_collision_clearance_buffer=0.0,
        apply_control_lowpass=True,
        control_lowpass_alpha=0.55,

        # Stable representation settings.
        w_corridor=8.0,
        corridor_radius_base=0.35,
        corridor_radius_scale=1.25,
        corridor_radius_min=0.30,
        corridor_radius_max=1.20,
        w_heatmap=3.0,
        heatmap_sigma_scale=1.4,
        mode_select_top_k=4,
        mode_select_min_rollouts_per_mode=64,
    )


    print(f"Dynamic wall pairs: {wall_pairs}")
    print(f"Dynamic wall width: {wall_width}")
    print(f"Dynamic wall extension: {wall_extension}")
    print(f"Block step: {block_step}")

    variants = [
        ControllerVariant.FULL_SWARM_PRIOR_MPPI,
        ControllerVariant.GAUSSIAN_PRIOR_MPPI,
        ControllerVariant.EMPIRICAL_INIT_MPPI,

        ControllerVariant.HOMOTOPY_SEEDED_MPPI,
        ControllerVariant.CORRIDOR_PRIOR_MPPI,
        ControllerVariant.FRENET_CORRIDOR_MPPI,
        ControllerVariant.HEATMAP_PRIOR_MPPI,
        ControllerVariant.CONTROL_BANK_MPPI,
        ControllerVariant.MODE_SELECTING_HOMOTOPY_MPPI,
        ControllerVariant.MODE_SELECTING_CORRIDOR_MPPI,

        ControllerVariant.STANDARD_MPPI,
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
            max_steps=130,
            goal_tolerance=0.3,
            mppi_cfg=cfg,
        )
        row = summarize_dynamic_result(res, base_obstacles, blocker, goal, cfg.robot_radius)
        print(
            f"  success={row['success']} collision={row['collision']} "
            f"final_dist={row['final_dist']:.3f} smooth={row['control_smoothness']:.3f} "
            f"runtime/step={row['runtime_per_step_sec']:.3f}s"
        )
        results.append(res)
        rows.append(row)

    # if pd is not None:
    #     df = pd.DataFrame(rows)
    #     df.to_csv("dynamic_block_hard_metrics.csv", index=False)
    #     print("Saved metrics: dynamic_block_soft_metrics.csv")
    #     print(df)
    # else:
    #     for row in rows:
    #         print(row)

    from pathlib import Path
    output_dir = Path("dynamic_block_hard")
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


# =============================================================================
# Main experiment
# =============================================================================

def main():
    print("Variant file: dynamic_block_hard")
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

    # Keep settings identical where possible.
    mppi_cfg = MPPIConfig(
        horizon=28,
        num_rollouts=650,
        dt=0.12,
        max_empirical_nominals_per_mode=12,
        swarm_init_probability=0.45,
        sigma_floor=0.25,
        max_precision=10.0,
        w_reference_tracking=0.20,
        w_control_smooth=0.40,
        smooth_v_weight=0.5,
        smooth_omega_weight=2.0,
        w_heading=0.0,
        w_mode_prior=0.15,
        uncertainty_margin_gain=0.25,
        use_soft_obstacle_cost=False,
        enable_hard_collision_filter=True,
        hard_collision_clearance_buffer=0.0,
        apply_control_lowpass=True,
        control_lowpass_alpha=0.55,
    )

    variants = [
        ControllerVariant.FULL_SWARM_PRIOR_MPPI,
        ControllerVariant.GAUSSIAN_PRIOR_MPPI,
        ControllerVariant.EMPIRICAL_INIT_MPPI,

        ControllerVariant.HOMOTOPY_SEEDED_MPPI,
        ControllerVariant.CORRIDOR_PRIOR_MPPI,
        ControllerVariant.FRENET_CORRIDOR_MPPI,
        ControllerVariant.HEATMAP_PRIOR_MPPI,
        ControllerVariant.CONTROL_BANK_MPPI,
        ControllerVariant.MODE_SELECTING_HOMOTOPY_MPPI,
        ControllerVariant.MODE_SELECTING_CORRIDOR_MPPI,

        ControllerVariant.STANDARD_MPPI,
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
                    max_steps=120,
                    goal_tolerance=0.15,
                    mppi_cfg=mppi_cfg,
                        )
                row = summarize_result(res, obstacles, goal, robot_radius=mppi_cfg.robot_radius)
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

    # # Plot one seed's trajectories for all variants.
    # one_seed_results = [r for r in all_results if r["seed"] == RUN_SEEDS[0]]
    # if one_seed_results:
    #     plot_path = f"{OUTPUT_PREFIX}_paths.png"
    #     plot_paths(one_seed_results, obstacles, start, goal, bounds_xy, plot_path)
    #     print(f"Saved path plot: {plot_path}")

    # plt.show()


# =============================================================================
# Lightweight repeated robustness experiment
# =============================================================================

@dataclass(frozen=True)
class DynamicWallScenario:
    scenario_id: str
    wall_pairs: Tuple[Tuple[int, int], ...]
    trigger_progress: float = 0.25
    wall_width: float = 0.40
    wall_extension: float = 0.20


def default_dynamic_wall_scenarios() -> List[DynamicWallScenario]:
    """Smoke-test scenarios using a common spatial activation threshold."""
    return [
        DynamicWallScenario("wall_0_1", ((0, 1),), trigger_progress=0.25),
        DynamicWallScenario("wall_0_2", ((0, 2),), trigger_progress=0.25),
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
    """Append one row immediately so long experiments preserve partial results."""
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
        exposed = group[group["exposed_to_blocker"] == True]  # noqa: E712
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

    # Compact matrix requested for quickly comparing success by scenario.
    success_matrix = by_scenario.pivot_table(
        index=["condition", "scenario_id"],
        columns="variant",
        values="success_rate",
        aggfunc="first",
    )
    success_matrix.to_csv(success_per_scenario_csv)


def main_dynamic_robustness():
    """Run paired robustness trials with randomized obstacle-center layouts.

    For each controller seed, obstacle polygon shapes are reassigned to the
    original obstacle centers using a seeded derangement. The same permuted
    layout is shared by every controller variant and every condition for that
    seed. Dynamic and static walls are built once from the original scene, so
    their geometry remains fixed even though obstacle shapes exchange centers.
    """
    controller_seeds = RUN_SEEDS
    swarm_seeds = [RUN_SWARM_SEED]
    scenarios = default_dynamic_wall_scenarios()
    max_steps = 130
    goal_tolerance = 0.30
    activation_preview_clearance = 0.75

    variants = [
        ControllerVariant.FULL_SWARM_PRIOR_MPPI,
        ControllerVariant.GAUSSIAN_PRIOR_MPPI,
        ControllerVariant.EMPIRICAL_INIT_MPPI,
        ControllerVariant.HOMOTOPY_SEEDED_MPPI,
        ControllerVariant.CORRIDOR_PRIOR_MPPI,
        ControllerVariant.FRENET_CORRIDOR_MPPI,
        ControllerVariant.HEATMAP_PRIOR_MPPI,
        ControllerVariant.CONTROL_BANK_MPPI,
        ControllerVariant.MODE_SELECTING_HOMOTOPY_MPPI,
        ControllerVariant.MODE_SELECTING_CORRIDOR_MPPI,
        ControllerVariant.STANDARD_MPPI,
    ]

    # Stronger topology/reference priors than the previous smoke-test setup.
    # Applied-control low-pass filtering is disabled for immediate reaction.
    cfg = MPPIConfig(
        horizon=28,
        num_rollouts=650,
        dt=0.12,
        base_safety_margin=0.0,
        use_soft_obstacle_cost=False,
        enable_hard_collision_filter=True,
        hard_collision_clearance_buffer=0.0,
        suppress_blocked_modes=False,
        max_empirical_nominals_per_mode=16,
        swarm_init_probability=0.60,
        sigma_floor=0.25,
        max_precision=10.0,
        w_reference_tracking=1.20,
        w_control_smooth=0.40,
        smooth_v_weight=0.5,
        smooth_omega_weight=2.0,
        w_heading=0.0,
        w_mode_prior=0.25,
        uncertainty_margin_gain=0.25,
        apply_control_lowpass=False,
        control_lowpass_alpha=0.0,
        w_corridor=12.0,
        corridor_radius_base=0.35,
        corridor_radius_scale=1.25,
        corridor_radius_min=0.30,
        corridor_radius_max=1.20,
        w_heatmap=4.5,
        heatmap_sigma_scale=1.4,
        mode_select_top_k=4,
        mode_select_min_rollouts_per_mode=64,
    )

    scale, bounds_xy, bounds_ranges, start, goal, original_obstacles = build_default_scene()
    for scenario in scenarios:
        validate_dynamic_wall_scenario(scenario, len(original_obstacles))

    # Freeze the original spatial center anchors before any obstacle swapping.
    # Wall-pair indices refer to these center slots, not to obstacle identities.
    fixed_wall_centers = tuple(
        obstacle_center(obs).copy()
        for obs in original_obstacles
    )

    # Build walls from the frozen anchors exactly once. These objects are reused
    # for every randomized obstacle layout, so wall positions never move.
    fixed_blockers = {
        scenario.scenario_id: make_wall_blockers_between_centers(
            centers=fixed_wall_centers,
            pairs=scenario.wall_pairs,
            width=scenario.wall_width,
            extension=scenario.wall_extension,
        )
        for scenario in scenarios
    }

    detail_csv = "dynamic_block_hard_robustness_trials.csv"
    summary_csv = "dynamic_block_hard_robustness_summary.csv"
    scenario_summary_csv = "dynamic_block_hard_robustness_by_scenario.csv"
    success_per_scenario_csv = "dynamic_block_hard_success_per_scenario.csv"

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
        "goal_reached_before_block", "selected_mode_switches",
        "last_selected_mode", "final_dist", "min_clearance_dynamic",
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
                    row = summarize_dynamic_result(
                        result,
                        controller_base_obstacles,
                        blocker,
                        goal,
                        cfg.robot_radius,
                        goal_tolerance=goal_tolerance,
                    )
                    row.update(base_row)
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

            append_csv_row(detail_csv, row, fieldnames)
            print(
                f"  success={row.get('success')} "
                f"failure={row.get('failure_reason') or '-'} "
                f"activation={row.get('activation_step')}"
            )

    for swarm_seed in swarm_seeds:
        for controller_seed in controller_seeds:
            # Use controller_seed as the layout seed so all swarm seeds, variants,
            # and wall conditions can be compared on the same obstacle layouts.
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

            # Baseline 1: randomized obstacles, no wall.
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
                # This blocker was built from fixed_wall_centers, not from the
                # swapped layout, and therefore stays on the original segment.
                blocker = fixed_blockers[scenario.scenario_id]

                # Main test: prior matches randomized obstacles but not the wall
                # that appears later at the fixed original-scene position.
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

                # Baseline 2: oracle prior knows the fixed wall from t=0 while
                # using the same randomized obstacle layout.
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
    main_dynamic_robustness()