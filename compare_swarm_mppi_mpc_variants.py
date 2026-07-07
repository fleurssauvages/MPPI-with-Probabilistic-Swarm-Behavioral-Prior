#!/usr/bin/env python3
"""
Compare topology-aware swarm-prior MPPI variants against standard MPPI and deterministic MPC baselines.

This script evaluates a family of receding-horizon controllers for a nonholonomic unicycle robot navigating a 2D workspace with polygonal obstacles. A homotopy-aware generative swarm planner is first used to produce multiple feasible trajectory samples from the start to the goal. These samples are grouped by homotopy signature and converted into a topological trajectory mixture. Each homotopy mode stores a probability, a mean path, covariance blocks, and the original empirical sample paths. The controllers then use this information in different ways to test which representation of the swarm prior is most useful for local control.

The compared variants are:

1. `full_swarm_prior_mppi`

   MPPI using the complete Gaussian swarm prior. Rollouts are sampled from homotopy-conditioned empirical and mean-path proposals. The cost includes the homotopy probability prior, Mahalanobis tracking to the Gaussian mean path, covariance-aware obstacle-margin inflation, control effort, control smoothness, and terminal goal cost.

   This is the most direct probabilistic interpretation of the swarm output, but it can become too stiff when covariance is small or when the fitted Gaussian path is not dynamically convenient for the unicycle.

2. `gaussian_prior_mppi`

   MPPI using the Gaussian homotopy prior without empirical swarm rollout initialization. It uses homotopy probabilities, mean paths, covariance blocks, Mahalanobis tracking, and uncertainty-inflated obstacle margins, but it does not initialize rollouts from the raw swarm sample paths.

   This isolates the effect of the fitted Gaussian trajectory mixture itself.

3. `empirical_init_mppi`

   MPPI using empirical swarm trajectories only as proposal initializations. It does not use the homotopy probability prior, Gaussian tracking cost, or covariance-based obstacle-margin inflation.

   This tests whether raw swarm rollouts are useful as control-sequence seeds even without a probabilistic Gaussian tracking objective.

4. `homotopy_seeded_mppi`

   MPPI using homotopy-conditioned proposal sampling but no additional trajectory-prior cost. For each homotopy, the controller builds nominal control sequences from the localized mean path and empirical samples, then evaluates rollouts using only the current goal, obstacle, control-effort, and smoothness costs.

   This variant treats the swarm as a proposal generator rather than as a reference to be tracked.

5. `corridor_prior_mppi`

   MPPI using a soft Euclidean corridor around each homotopy mean path. The covariance is converted into a corridor radius, not inverted as a precision matrix. Rollouts inside the corridor are nearly free with respect to the prior, while rollouts outside the corridor receive a soft penalty.

   This avoids brittle Mahalanobis tracking and uses the swarm to define a topological tube rather than a hard reference trajectory.

6. `frenet_corridor_mppi`

   MPPI using a Frenet-style lateral corridor around the homotopy mean path. Instead of penalizing full Euclidean distance to the path at each time index, it mainly penalizes lateral deviation from the local path direction.

   This allows more freedom in progress along the path while still encouraging the robot to remain in the selected topological channel.

7. `heatmap_prior_mppi`

   MPPI using a smooth heatmap-like radial prior around each homotopy mean path. The prior cost is low near the centerline and saturates farther away.

   This is a softer alternative to corridor penalties. It gives spatial preference to swarm-supported regions without imposing a hard tube boundary or Gaussian precision.

8. `control_bank_mppi`

   MPPI using the empirical swarm paths only to build a bank of candidate nominal control sequences. It does not apply a mean-path, corridor, heatmap, covariance, or homotopy-prior cost.

   This tests the pure proposal-distribution benefit of the swarm: the prior influences what controls are sampled, but not how rollouts are scored.

9. `mode_selecting_homotopy_mppi`

   Per-homotopy MPPI with hard mode selection and no additional representation cost. The controller runs a separate MPPI rollout batch for each candidate homotopy mode, computes a soft MPPI value for each mode, selects the lowest-cost mode, and applies only that mode’s weighted control.

   This avoids destructive averaging between incompatible homotopies, such as one mode turning left and another turning right.

10. `mode_selecting_corridor_mppi`

Per-homotopy MPPI with hard mode selection and a soft corridor prior. Each homotopy receives its own rollout batch and corridor cost. The controller scores each homotopy independently, selects the best mode, and applies only the selected mode’s control.

This is the main topology-aware stable-representation variant: it combines homotopy-conditioned proposal sampling, soft corridor guidance, and explicit online mode selection.

11. `standard_mppi`

A baseline MPPI controller with no swarm prior and no homotopy information. It samples noisy control sequences around a simple goal-directed unicycle nominal policy and scores them using obstacle, goal, control-effort, and smoothness costs.

This baseline tests whether the swarm/topology information improves over ordinary local stochastic control.

12. `standard_mpc`

A deterministic nonlinear MPC baseline with no swarm prior and no homotopy modes. It optimizes a single control sequence using `scipy.optimize.minimize`, with warm starting from the previous MPC solution.

This baseline tests how well a conventional local optimizer performs when initialized only by a goal-directed nominal sequence.

13. `homotopy_mean_mpc`

A deterministic MPC variant that uses the homotopy mean paths as references. It solves a separate MPC problem for the top-k homotopy mean paths and applies the control from the lowest-cost solution.

This tests whether deterministic MPC benefits from the same homotopy information, without using stochastic MPPI sampling.

Endpoint handling:

Before fitting the homotopy trajectory mixture, swarm trajectories that enter a radius of `0.2` around the goal have their terminal segment replaced by a straight line to the exact goal. This removes endpoint curls and ensures that fitted mean paths, covariance blocks, corridors, and empirical control banks terminate cleanly at the desired goal.

Main comparison objective:

The experiment is designed to separate four effects:

1. whether homotopy-aware swarm information improves local receding-horizon control;
2. whether Gaussian trajectory tracking is too stiff compared with softer corridor or proposal representations;
3. whether empirical swarm samples are more useful as proposal distributions than as hard tracking references;
4. whether hard per-homotopy mode selection avoids destructive averaging between incompatible topological alternatives.

The output metrics include success, collision, final distance to goal, minimum clearance, path length, control effort, control smoothness, number of executed steps, total runtime, and runtime per step.

"""

from __future__ import annotations

import math
import pickle
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

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
RUN_SWARM_SEED = 3
OUTPUT_PREFIX = "comparison"


class ControllerVariant(str, Enum):
    FULL_SWARM_PRIOR_MPPI = "full_swarm_prior_mppi"
    GAUSSIAN_PRIOR_MPPI = "gaussian_prior_mppi"
    EMPIRICAL_INIT_MPPI = "empirical_init_mppi"

    # Stable representation variants.
    # These use the swarm/topology as a proposal or corridor, not as a stiff
    # Cartesian Gaussian trajectory tracker.
    HOMOTOPY_SEEDED_MPPI = "homotopy_seeded_mppi"
    CORRIDOR_PRIOR_MPPI = "corridor_prior_mppi"
    FRENET_CORRIDOR_MPPI = "frenet_corridor_mppi"
    HEATMAP_PRIOR_MPPI = "heatmap_prior_mppi"
    CONTROL_BANK_MPPI = "control_bank_mppi"
    MODE_SELECTING_HOMOTOPY_MPPI = "mode_selecting_homotopy_mppi"
    MODE_SELECTING_CORRIDOR_MPPI = "mode_selecting_corridor_mppi"

    STANDARD_MPPI = "standard_mppi"
    STANDARD_MPC = "standard_mpc"
    HOMOTOPY_MEAN_MPC = "homotopy_mean_mpc"


@dataclass
class MPPIConfig:
    dt: float = 0.12
    horizon: int = 28
    num_rollouts: int = 500
    lambda_temperature: float = 7.0

    v_min: float = -1.6
    v_max: float = 1.6
    omega_min: float = -2.8
    omega_max: float = 2.8

    noise_v: float = 0.38
    noise_omega: float = 0.95
    temporal_noise_smoothing: float = 0.65

    swarm_init_probability: float = 0.45
    max_empirical_nominals_per_mode: int = 16

    robot_radius: float = 0.18
    base_safety_margin: float = 0.07
    uncertainty_margin_gain: float = 0.25

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

    # Heatmap-like prior implemented as a fast smooth radial potential around
    # the mode centerline. This avoids building a grid and remains numba-friendly.
    w_heatmap: float = 3.0
    heatmap_sigma_scale: float = 1.4

    # Per-mode mode-selection settings.
    mode_select_top_k: int = 4
    mode_select_min_rollouts_per_mode: int = 64


@dataclass
class MPCConfig:
    dt: float = 0.12
    horizon: int = 20

    v_min: float = 0.0
    v_max: float = 1.6
    omega_min: float = -2.8
    omega_max: float = 2.8

    robot_radius: float = 0.18
    base_safety_margin: float = 0.07

    w_goal: float = 30.0
    w_obstacle: float = 500.0
    w_control: float = 0.03
    w_control_smooth: float = 0.15
    w_reference: float = 4.0
    w_heading: float = 0.08

    maxiter: int = 12

    # Speed settings for deterministic MPC baselines.
    # Exact polygon signed distance is expensive inside scipy.optimize.
    # For fair fast ablations, use the same bounding-circle obstacle proxy as
    # the vectorized MPPI cost, then evaluate final metrics with exact polygons.
    use_fast_obstacle_circles: bool = True

    # Homotopy-mean MPC is especially slow if it solves one optimization per
    # homotopy. Use only the top-k most likely modes for the deterministic
    # baseline.
    top_k_homotopy_mpc: int = 2

    # Reuse the previous optimal control sequence as the next initial guess.
    warm_start: bool = True


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
    Replace the terminal part of a swarm trajectory by a straight segment to goal.

    If the path enters a ball of radius snap_radius around the goal, the function
    keeps the trajectory until the first entry point and replaces the rest by a
    straight line from that entry point to the exact goal.

    This removes small noisy endpoint curls before mixture fitting, so the fitted
    mean/covariance and empirical control banks end exactly at the desired goal.

    If the path never enters the snap ball, the path is returned unchanged.
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

    # If the trajectory starts inside the goal ball, just make a short straight
    # terminal path to avoid degeneracy.
    entry = min(entry, len(p) - 1)
    anchor_pt = p[entry].copy()

    n_tail = max(2, int(straight_tail_points))
    tail = np.linspace(anchor_pt, g, n_tail)

    if entry == 0:
        snapped = tail
    else:
        snapped = np.vstack([p[:entry], tail])

    # Remove zero-length duplicate points, which can confuse arc-length resampling.
    keep = [0]
    for i in range(1, len(snapped)):
        if np.linalg.norm(snapped[i] - snapped[keep[-1]]) > 1e-10:
            keep.append(i)
    snapped = snapped[keep]

    return snapped


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

    # Optional endpoint snapping for the demo:
    # if a swarm trajectory enters snap_to_goal_radius around goal, replace its
    # noisy terminal segment by a straight line to the exact goal before fitting.
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

    # Map original path IDs to the snapped path and its quality information.
    # gen_out.homotopy_groups stores original path objects, so this preserves
    # the homotopy grouping while fitting on snapped trajectories.
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


def obstacle_bounding_circles(obstacles: Sequence) -> List[Tuple[Array, float]]:
    circles = []
    for obs in obstacles:
        poly = _poly_vertices(obs)
        center = poly.mean(axis=0)
        radius = float(np.max(np.linalg.norm(poly - center[None, :], axis=1)))
        circles.append((center, radius))
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

        This deliberately avoids Mahalanobis tracking. The prior guides sampling
        and/or corridor membership, while current obstacle and goal costs decide.
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
                    # Soft Euclidean corridor. Free inside the tube, penalized outside.
                    dcen = math.sqrt(ex * ex + ey * ey)
                    outside = dcen - r
                    sp = _softplus_scalar_nb(6.0 * outside)
                    cost += w_corridor * sp * sp

                elif rep_type == 2:
                    # Frenet/lateral corridor: penalize lateral deviation only.
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
                    # Heatmap-like prior without an actual grid:
                    # low cost near centerline, saturating cost far away.
                    dcen = math.sqrt(ex * ex + ey * ey)
                    sigma = heatmap_sigma_scale * (r + 1e-9)
                    q = dcen / sigma
                    heat_cost = 1.0 - math.exp(-0.5 * q * q)
                    cost += w_heatmap * heat_cost

                # Current obstacle cost.
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
    def mpc_stage_cost_fast_circles_nb(
        X,
        U,
        circle_centers,
        circle_radii,
        goal,
        ref_path,
        has_ref,
        horizon,
        robot_radius,
        base_safety_margin,
        w_goal,
        w_obstacle,
        w_control,
        w_control_smooth,
        w_reference,
        w_heading,
    ):
        H = horizon
        M = circle_radii.shape[0]
        cost = 0.0

        for t in range(H):
            px = X[t + 1, 0]
            py = X[t + 1, 1]

            for j in range(M):
                dx = px - circle_centers[j, 0]
                dy = py - circle_centers[j, 1]
                d = math.sqrt(dx * dx + dy * dy) - circle_radii[j]
                margin = robot_radius + base_safety_margin
                sp = _softplus_scalar_nb(8.0 * (margin - d))
                cost += w_obstacle * sp * sp

            if has_ref:
                ex = px - ref_path[t, 0]
                ey = py - ref_path[t, 1]
                cost += w_reference * (ex * ex + ey * ey)

                if t < H - 1:
                    tx = ref_path[t + 1, 0] - ref_path[t, 0]
                    ty = ref_path[t + 1, 1] - ref_path[t, 1]
                    if math.sqrt(tx * tx + ty * ty) > 1e-9:
                        ref_heading = math.atan2(ty, tx)
                        dh = _wrap_angle_nb(X[t + 1, 2] - ref_heading)
                        cost += w_heading * dh * dh

        gx = X[H, 0] - goal[0]
        gy = X[H, 1] - goal[1]
        cost += w_goal * (gx * gx + gy * gy)

        ctrl = 0.0
        for t in range(H):
            v = U[t, 0]
            om = U[t, 1]
            ctrl += v * v + 0.15 * om * om
        cost += w_control * ctrl

        smooth = 0.0
        for t in range(H - 1):
            dv = U[t + 1, 0] - U[t, 0]
            dom = U[t + 1, 1] - U[t, 1]
            smooth += dv * dv + 0.2 * dom * dom
        cost += w_control_smooth * smooth

        return cost

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

else:
    rollout_unicycle_batch_nb = None
    rollout_unicycle_single_nb = None
    fast_swarm_prior_costs_nb = None
    stable_representation_costs_nb = None
    standard_mppi_costs_batch_nb = None
    mpc_stage_cost_fast_circles_nb = None
    min_clearance_nb = None


def obstacle_circles_to_arrays(obstacle_circles: List[Tuple[Array, float]]) -> Tuple[Array, Array]:
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    return centers, radii


def obstacles_to_padded_arrays(obstacles: Sequence) -> Tuple[Array, Array]:
    polys = [_poly_vertices(o).astype(np.float64) for o in obstacles]
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
        return fast_swarm_prior_costs_nb(
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

    return costs


def standard_mppi_costs_batch(
    X: Array,
    U: Array,
    obstacle_circles: List[Tuple[Array, float]],
    goal: Array,
    cfg: MPPIConfig,
) -> Array:
    if standard_mppi_costs_batch_nb is not None:
        centers, radii = obstacle_circles_to_arrays(obstacle_circles)
        return standard_mppi_costs_batch_nb(
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

    return costs




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

    This is intentionally softer and more stable than inverting covariance.
    The radius says "the robot is free inside this topological tube."
    """
    K = mode.cov_blocks.shape[0]
    r = np.zeros(K, dtype=np.float64)
    for t in range(K):
        S = 0.5 * (mode.cov_blocks[t] + mode.cov_blocks[t].T)
        vals = np.linalg.eigvalsh(S)
        # Use the larger spatial spread, not the inverse precision.
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
    """
    Costs for stable swarm/topology representations.

    Unlike fast_swarm_prior_costs(), this does not use Mahalanobis tracking.
    """
    radius = corridor_radius_from_mode(mode, cfg)

    if stable_representation_costs_nb is not None:
        centers, radii = obstacle_circles_to_arrays(obstacle_circles)
        return stable_representation_costs_nb(
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

    return costs


def softmin_score(costs: Array, cfg: MPPIConfig) -> float:
    """
    Numerically stable MPPI soft value. Lower is better.
    """
    rho = float(np.min(costs))
    z = np.exp(-(costs - rho) / cfg.lambda_temperature)
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
    """
    Mixed rollout MPPI using stable representations.

    This still computes one weighted MPPI control across all sampled modes.
    Use mode_selecting_stable_mppi_step() to avoid cross-homotopy action averaging.
    """
    progress_by_mode = {} if progress_by_mode is None else dict(progress_by_mode)

    local_modes = []
    new_progress_by_mode = dict(progress_by_mode)
    for m in global_modes:
        key = str(m.signature)
        prev = progress_by_mode.get(key)
        local_m, idx = localize_mode_for_state_with_index(
            m,
            x_current,
            cfg.horizon,
            previous_idx=prev if cfg.use_monotonic_reference_progress else None,
            max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None,
        )
        local_modes.append(local_m)
        new_progress_by_mode[key] = idx

    if use_pi_sampling:
        pi = np.array([m.probability for m in local_modes], dtype=np.float64)
        pi = pi / (pi.sum() + 1e-12)
    else:
        pi = np.ones(len(local_modes), dtype=np.float64) / len(local_modes)

    mode_ids = rng.choice(len(local_modes), size=cfg.num_rollouts, p=pi)
    obstacle_circles = obstacle_bounding_circles(obstacles)

    all_costs = np.zeros(cfg.num_rollouts, dtype=np.float64)
    all_U0 = np.zeros((cfg.num_rollouts, 2), dtype=np.float64)

    for mid, local_mode in enumerate(local_modes):
        ids = np.where(mode_ids == mid)[0]
        n = len(ids)
        if n == 0:
            continue

        key = str(global_modes[mid].signature)
        nominal_bank = build_nominal_bank_for_mode(
            x_current,
            local_mode,
            global_modes[mid],
            goal,
            cfg,
            rng,
            use_empirical_init=use_empirical_init,
            use_mean_nominal=use_mean_nominal,
            previous_idx=progress_by_mode.get(key),
        )

        U = sample_controls_from_nominal_bank(
            nominal_bank,
            n,
            cfg,
            rng,
            prefer_empirical=use_empirical_init,
        )
        X = rollout_unicycle_batch(x_current, U, cfg.dt)

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

        all_costs[ids] = costs
        all_U0[ids] = U[:, 0, :]

    u = mppi_weighted_control(all_costs, all_U0, cfg)
    info = {
        "cost_min": float(all_costs.min()),
        "cost_mean": float(all_costs.mean()),
        "soft_value": softmin_score(all_costs, cfg),
        "rep_type": int(rep_type),
        "mode_selection": False,
    }
    return u, info, new_progress_by_mode


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
    """
    Per-homotopy MPPI with hard mode selection.

    This avoids destructive averaging between incompatible homotopies.
    Each homotopy proposes its own u_h. The controller applies the u_h from
    the lowest soft-value homotopy under the current obstacle/goal cost.
    """
    progress_by_mode = {} if progress_by_mode is None else dict(progress_by_mode)

    top_k = min(max(1, int(cfg.mode_select_top_k)), len(global_modes))
    candidate_modes = global_modes[:top_k]
    n_per_mode = max(
        int(cfg.mode_select_min_rollouts_per_mode),
        int(math.ceil(cfg.num_rollouts / float(top_k))),
    )

    obstacle_circles = obstacle_bounding_circles(obstacles)
    best = None
    new_progress_by_mode = dict(progress_by_mode)

    for mid, global_mode in enumerate(candidate_modes):
        key = str(global_mode.signature)
        prev = progress_by_mode.get(key)

        local_mode, idx = localize_mode_for_state_with_index(
            global_mode,
            x_current,
            cfg.horizon,
            previous_idx=prev if cfg.use_monotonic_reference_progress else None,
            max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None,
        )
        new_progress_by_mode[key] = idx

        nominal_bank = build_nominal_bank_for_mode(
            x_current,
            local_mode,
            global_mode,
            goal,
            cfg,
            rng,
            use_empirical_init=use_empirical_init,
            use_mean_nominal=use_mean_nominal,
            previous_idx=prev,
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
            obstacle_circles,
            goal,
            cfg,
            rep_type=rep_type,
            use_mode_prior=use_mode_prior,
        )

        u_h = mppi_weighted_control(costs, U[:, 0, :], cfg)
        J_h = softmin_score(costs, cfg)

        if best is None or J_h < best["score"]:
            best = {
                "score": float(J_h),
                "u": u_h,
                "mode_index": int(mid),
                "signature": str(global_mode.signature),
                "probability": float(global_mode.probability),
                "cost_min": float(costs.min()),
                "cost_mean": float(costs.mean()),
            }

    assert best is not None
    info = {
        "cost_min": best["cost_min"],
        "cost_mean": best["cost_mean"],
        "soft_value": best["score"],
        "selected_mode_index": best["mode_index"],
        "selected_mode_signature": best["signature"],
        "selected_mode_probability": best["probability"],
        "rep_type": int(rep_type),
        "mode_selection": True,
    }
    return best["u"], info, new_progress_by_mode


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
    for m in global_modes:
        key = str(m.signature)
        prev = progress_by_mode.get(key)
        local_m, idx = localize_mode_for_state_with_index(
            m,
            x_current,
            cfg.horizon,
            previous_idx=prev if cfg.use_monotonic_reference_progress else None,
            max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None,
        )
        local_modes.append(local_m)
        new_progress_by_mode[key] = idx

    if use_pi_sampling:
        pi = np.array([m.probability for m in local_modes], dtype=np.float64)
        pi = pi / (pi.sum() + 1e-12)
    else:
        pi = np.ones(len(local_modes), dtype=np.float64) / len(local_modes)

    mode_ids = rng.choice(len(local_modes), size=cfg.num_rollouts, p=pi)
    obstacle_circles = obstacle_bounding_circles(obstacles)

    all_costs = np.zeros(cfg.num_rollouts, dtype=np.float64)
    all_U0 = np.zeros((cfg.num_rollouts, 2), dtype=np.float64)

    for mid, mode in enumerate(local_modes):
        ids = np.where(mode_ids == mid)[0]
        n = len(ids)
        if n == 0:
            continue

        if use_mean_reference:
            mean_nominal = nominal_controls_to_track_path(x_current, mode.mean_path, cfg)
        else:
            mean_nominal = nominal_controls_to_goal(x_current, goal, cfg)

        if use_empirical_init:
            nominal_bank = build_empirical_nominal_bank(
                x_current=x_current,
                global_mode=global_modes[mid],
                mean_nominal=mean_nominal,
                cfg=cfg,
                rng=rng,
                previous_idx=progress_by_mode.get(str(global_modes[mid].signature)),
            )
        else:
            nominal_bank = [mean_nominal]

        if len(nominal_bank) == 1:
            bank_ids = np.zeros(n, dtype=int)
        else:
            probs = np.ones(len(nominal_bank), dtype=np.float64)
            probs[0] = max(1e-6, 1.0 - cfg.swarm_init_probability)
            probs[1:] = cfg.swarm_init_probability / (len(nominal_bank) - 1)
            probs /= probs.sum()
            bank_ids = rng.choice(len(nominal_bank), size=n, p=probs)

        U = np.stack([nominal_bank[j].copy() for j in bank_ids], axis=0)
        U += make_temporally_correlated_noise(n, cfg.horizon, cfg, rng)

        U[:, :, 0] = np.clip(U[:, :, 0], cfg.v_min, cfg.v_max)
        U[:, :, 1] = np.clip(U[:, :, 1], cfg.omega_min, cfg.omega_max)

        X = rollout_unicycle_batch(x_current, U, cfg.dt)

        costs = fast_swarm_prior_costs(
            X, U, mode, obstacle_circles, goal, cfg,
            use_gaussian_tracking=use_gaussian_tracking,
            use_uncertainty_margin=use_uncertainty_margin,
            use_mode_prior=use_mode_prior,
            use_mean_reference=use_mean_reference,
        )

        all_costs[ids] = costs
        all_U0[ids] = U[:, 0, :]

    u = mppi_weighted_control(all_costs, all_U0, cfg)
    info = {
        "cost_min": float(all_costs.min()),
        "cost_mean": float(all_costs.mean()),
    }
    return u, info, new_progress_by_mode


def standard_mppi_step(
    x_current: Array,
    obstacles: Sequence,
    goal: Array,
    cfg: MPPIConfig,
    rng: np.random.Generator,
) -> Tuple[Array, Dict[str, object]]:
    obstacle_circles = obstacle_bounding_circles(obstacles)
    U_nom = nominal_controls_to_goal(x_current, goal, cfg)

    U = np.repeat(U_nom[None, :, :], cfg.num_rollouts, axis=0)
    U += make_temporally_correlated_noise(cfg.num_rollouts, cfg.horizon, cfg, rng)

    U[:, :, 0] = np.clip(U[:, :, 0], cfg.v_min, cfg.v_max)
    U[:, :, 1] = np.clip(U[:, :, 1], cfg.omega_min, cfg.omega_max)

    X = rollout_unicycle_batch(x_current, U, cfg.dt)
    costs = standard_mppi_costs_batch(X, U, obstacle_circles, goal, cfg)

    u = mppi_weighted_control(costs, U[:, 0, :], cfg)
    return u, {"cost_min": float(costs.min()), "cost_mean": float(costs.mean())}


# =============================================================================
# Standard MPC baselines
# =============================================================================

def shift_warm_start(U_prev: Optional[Array], U_nominal: Array) -> Array:
    """
    Shift previous MPC solution by one step and append the last control.

    This is the standard speed trick for receding-horizon MPC.
    """
    if U_prev is None or len(U_prev) != len(U_nominal):
        return U_nominal.copy()

    U = np.zeros_like(U_prev)
    U[:-1] = U_prev[1:]
    U[-1] = U_prev[-1]
    return U


def mpc_stage_cost_fast_circles(
    X: Array,
    U: Array,
    obstacle_circles: List[Tuple[Array, float]],
    goal: Array,
    cfg: MPCConfig,
    ref_path: Optional[Array] = None,
) -> float:
    """
    Faster deterministic MPC cost using bounding-circle obstacle proxy.

    Final evaluation metrics still use exact polygon signed distances.
    """
    if mpc_stage_cost_fast_circles_nb is not None:
        centers, radii = obstacle_circles_to_arrays(obstacle_circles)
        has_ref = ref_path is not None
        ref = np.zeros((cfg.horizon, 2), dtype=np.float64) if ref_path is None else np.asarray(ref_path, dtype=np.float64)
        return float(mpc_stage_cost_fast_circles_nb(
            np.asarray(X, dtype=np.float64),
            np.asarray(U, dtype=np.float64),
            centers,
            radii,
            np.asarray(goal, dtype=np.float64),
            ref,
            bool(has_ref),
            int(cfg.horizon),
            float(cfg.robot_radius),
            float(cfg.base_safety_margin),
            float(cfg.w_goal),
            float(cfg.w_obstacle),
            float(cfg.w_control),
            float(cfg.w_control_smooth),
            float(cfg.w_reference),
            float(cfg.w_heading),
        ))

    cost = 0.0
    H = cfg.horizon

    for t in range(H):
        p = X[t + 1, :2]

        for center, radius in obstacle_circles:
            d = float(np.linalg.norm(p - center) - radius)
            margin = cfg.robot_radius + cfg.base_safety_margin
            cost += cfg.w_obstacle * float(softplus(8.0 * (margin - d)) ** 2)

        if ref_path is not None:
            ref = ref_path[t]
            cost += cfg.w_reference * float(np.sum((p - ref) ** 2))

            if t < H - 1:
                tangent = ref_path[t + 1] - ref_path[t]
                if np.linalg.norm(tangent) > 1e-9:
                    ref_heading = math.atan2(tangent[1], tangent[0])
                    cost += cfg.w_heading * float(wrap_angle(X[t + 1, 2] - ref_heading) ** 2)

    cost += cfg.w_goal * float(np.sum((X[-1, :2] - goal) ** 2))
    cost += cfg.w_control * float(np.sum(U[:, 0] ** 2 + 0.15 * U[:, 1] ** 2))

    dU = np.diff(U, axis=0)
    cost += cfg.w_control_smooth * float(np.sum(dU[:, 0] ** 2 + 0.2 * dU[:, 1] ** 2))

    return float(cost)


def mpc_stage_cost(
    X: Array,
    U: Array,
    obstacles: Sequence,
    goal: Array,
    cfg: MPCConfig,
    ref_path: Optional[Array] = None,
) -> float:
    cost = 0.0
    H = cfg.horizon

    for t in range(H):
        p = X[t + 1, :2]

        for obs in obstacles:
            d, _ = polygon_signed_distance_and_normal(p, obs)
            margin = cfg.robot_radius + cfg.base_safety_margin
            cost += cfg.w_obstacle * float(softplus(8.0 * (margin - d)) ** 2)

        if ref_path is not None:
            ref = ref_path[t]
            cost += cfg.w_reference * float(np.sum((p - ref) ** 2))

            if t < H - 1:
                tangent = ref_path[t + 1] - ref_path[t]
                if np.linalg.norm(tangent) > 1e-9:
                    ref_heading = math.atan2(tangent[1], tangent[0])
                    cost += cfg.w_heading * float(wrap_angle(X[t + 1, 2] - ref_heading) ** 2)

    cost += cfg.w_goal * float(np.sum((X[-1, :2] - goal) ** 2))
    cost += cfg.w_control * float(np.sum(U[:, 0] ** 2 + 0.15 * U[:, 1] ** 2))

    dU = np.diff(U, axis=0)
    cost += cfg.w_control_smooth * float(np.sum(dU[:, 0] ** 2 + 0.2 * dU[:, 1] ** 2))

    return float(cost)


def solve_standard_mpc_once(
    x_current: Array,
    obstacles: Sequence,
    goal: Array,
    cfg: MPCConfig,
    ref_path: Optional[Array] = None,
    U_init: Optional[Array] = None,
    U_warm: Optional[Array] = None,
    obstacle_circles: Optional[List[Tuple[Array, float]]] = None,
) -> Tuple[Array, Array, Dict[str, object]]:
    if minimize is None:
        raise ImportError("scipy is required for MPC variants: pip install scipy")

    if U_init is None:
        if ref_path is None:
            U_init = nominal_controls_to_goal(x_current, goal, cfg)
        else:
            U_init = nominal_controls_to_track_path(x_current, ref_path, cfg)

    if cfg.warm_start:
        U_init = shift_warm_start(U_warm, U_init)

    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)

    bounds = [(cfg.v_min, cfg.v_max), (cfg.omega_min, cfg.omega_max)] * cfg.horizon

    def objective(z):
        U = np.asarray(z, dtype=np.float64).reshape(cfg.horizon, 2)
        X = rollout_unicycle(x_current, U, cfg.dt)
        if cfg.use_fast_obstacle_circles:
            return mpc_stage_cost_fast_circles(X, U, obstacle_circles, goal, cfg, ref_path=ref_path)
        return mpc_stage_cost(X, U, obstacles, goal, cfg, ref_path=ref_path)

    res = minimize(
        objective,
        U_init.reshape(-1),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": cfg.maxiter, "ftol": 1e-4},
    )

    U_star = np.asarray(res.x, dtype=np.float64).reshape(cfg.horizon, 2)
    u = U_star[0].copy()
    u[0] = np.clip(u[0], cfg.v_min, cfg.v_max)
    u[1] = np.clip(u[1], cfg.omega_min, cfg.omega_max)

    return u, U_star, {
        "success": bool(res.success),
        "fun": float(res.fun),
        "nit": int(getattr(res, "nit", -1)),
    }


def homotopy_mean_mpc_step(
    x_current: Array,
    modes: List[MPPIHomotopyMode],
    obstacles: Sequence,
    goal: Array,
    cfg: MPCConfig,
    warm_starts_by_mode: Optional[Dict[str, Array]] = None,
    obstacle_circles: Optional[List[Tuple[Array, float]]] = None,
) -> Tuple[Array, Dict[str, object], Dict[str, Array]]:
    best = None
    new_warm_starts = {} if warm_starts_by_mode is None else dict(warm_starts_by_mode)

    candidate_modes = modes[:max(1, int(cfg.top_k_homotopy_mpc))]

    for mode in candidate_modes:
        local_mode = localize_mode_for_state(mode, x_current, cfg.horizon)
        ref_path = local_mode.mean_path
        U_init = nominal_controls_to_track_path(x_current, ref_path, cfg)

        key = str(mode.signature)
        u, U_star, info = solve_standard_mpc_once(
            x_current=x_current,
            obstacles=obstacles,
            goal=goal,
            cfg=cfg,
            ref_path=ref_path,
            U_init=U_init,
            U_warm=(warm_starts_by_mode or {}).get(key),
            obstacle_circles=obstacle_circles,
        )
        new_warm_starts[key] = U_star

        score = info["fun"] - 0.5 * math.log(mode.probability + 1e-12)

        if best is None or score < best[0]:
            best = (score, u, info, mode.signature)

    assert best is not None
    return best[1], {"score": float(best[0]), "mode": str(best[3]), **best[2]}, new_warm_starts


def standard_mpc_step(
    x_current: Array,
    obstacles: Sequence,
    goal: Array,
    cfg: MPCConfig,
    U_warm: Optional[Array] = None,
    obstacle_circles: Optional[List[Tuple[Array, float]]] = None,
) -> Tuple[Array, Dict[str, object], Array]:
    u, U_star, info = solve_standard_mpc_once(
        x_current=x_current,
        obstacles=obstacles,
        goal=goal,
        cfg=cfg,
        ref_path=None,
        U_init=nominal_controls_to_goal(x_current, goal, cfg),
        U_warm=U_warm,
        obstacle_circles=obstacle_circles,
    )
    return u, info, U_star


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
        PolyObstacle(round_obstacle(
            np.array([[3.0, 1.5], [5.2, 2.2], [4.7, 4.0], [2.8, 3.4]]),
            n_iters=4, n_points=32,
        )),
        PolyObstacle(round_obstacle(
            np.array([[6.2, 6.0], [8.5, 6.3], [8.1, 8.4], [6.8, 8.9], [5.9, 7.4]]),
            n_iters=4, n_points=32,
        )),
        PolyObstacle(round_obstacle(
            np.array([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]]),
            n_iters=4, n_points=32,
        )),
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
    mpc_cfg: Optional[MPCConfig] = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    mppi_cfg = MPPIConfig() if mppi_cfg is None else mppi_cfg
    mpc_cfg = MPCConfig() if mpc_cfg is None else mpc_cfg

    x = initial_pose(start, goal)
    states = [x.copy()]
    controls = []
    infos = []
    previous_control = None

    # Warm starts are only used by deterministic MPC variants.
    standard_mpc_warm = None
    homotopy_mpc_warm = {}
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
            # Proposal-only: use homotopy/swarm samples to seed MPPI, but no
            # Gaussian/corridor/heatmap cost. Current obstacle/goal costs decide.
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
            # Control-bank/proposal-only variant. It samples empirical
            # homotopy-conditioned control sequences, then evaluates only the
            # current MPPI objective. No trajectory prior cost is added.
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
            # Per-mode MPPI, no representation cost. This tests whether
            # avoiding cross-homotopy averaging is enough.
            u, info, swarm_progress = mode_selecting_stable_mppi_step(
                x, modes, obstacles, goal, mppi_cfg, rng,
                rep_type=REP_NONE,
                use_empirical_init=True,
                use_mean_nominal=True,
                use_mode_prior=False,
                progress_by_mode=swarm_progress,
            )

        elif variant == ControllerVariant.MODE_SELECTING_CORRIDOR_MPPI:
            # Recommended stable variant: one MPPI bank per homotopy plus a
            # soft corridor cost, then hard-select the best homotopy online.
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

        elif variant == ControllerVariant.STANDARD_MPC:
            u, info, standard_mpc_warm = standard_mpc_step(
                x,
                obstacles,
                goal,
                mpc_cfg,
                U_warm=standard_mpc_warm,
                obstacle_circles=obstacle_circles,
            )

        elif variant == ControllerVariant.HOMOTOPY_MEAN_MPC:
            u, info, homotopy_mpc_warm = homotopy_mean_mpc_step(
                x,
                modes,
                obstacles,
                goal,
                mpc_cfg,
                warm_starts_by_mode=homotopy_mpc_warm,
                obstacle_circles=obstacle_circles,
            )

        else:
            raise ValueError(f"Unknown variant {variant}")

        if "mppi" in variant.value and mppi_cfg.apply_control_lowpass and previous_control is not None:
            alpha = float(mppi_cfg.control_lowpass_alpha)
            u = alpha * previous_control + (1.0 - alpha) * u
            u[0] = np.clip(u[0], mppi_cfg.v_min, mppi_cfg.v_max)
            u[1] = np.clip(u[1], mppi_cfg.omega_min, mppi_cfg.omega_max)

        previous_control = u.copy()

        x = unicycle_step(x, u, mppi_cfg.dt if "mppi" in variant.value else mpc_cfg.dt)

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


def summarize_result(result: Dict[str, object], obstacles, goal, robot_radius: float, goal_tolerance: float = 0.15):
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
    fig, ax = plt.subplots(figsize=(10, 9))
    setup_workspace(ax, obstacles, start, goal, bounds, "Controller comparison paths")

    # Enough distinct colors for many variants.
    # tab20 gives 20 categorical colors; fallback to hsv if more are needed.
    variant_names = sorted({res["variant"] for res in results})
    n = len(variant_names)

    if n <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(n)]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / max(1, n)) for i in range(n)]

    color_by_variant = {
        name: colors[i]
        for i, name in enumerate(variant_names)
    }

    # Different line styles for repeated seeds of the same variant.
    seed_values = sorted({res["seed"] for res in results})
    linestyles = ["-", "--", "-.", ":"]
    linestyle_by_seed = {
        seed: linestyles[i % len(linestyles)]
        for i, seed in enumerate(seed_values)
    }

    for res in results:
        states = res["states"]
        variant = res["variant"]
        seed = res["seed"]

        ax.plot(
            states[:, 0],
            states[:, 1],
            linewidth=2.4,
            linestyle=linestyle_by_seed[seed],
            color=color_by_variant[variant],
            label=f"{variant} seed={seed}",
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


# =============================================================================
# Main experiment
# =============================================================================

def main():
    print("Variant file: v7_goal_snap")
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
        num_rollouts=500,
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
    mpc_cfg = MPCConfig(
        horizon=12,
        dt=0.12,
        maxiter=12,
        use_fast_obstacle_circles=True,
        top_k_homotopy_mpc=2,
        warm_start=True,
    )

    variants = [
        # Original Gaussian/swarm ablations.
        ControllerVariant.FULL_SWARM_PRIOR_MPPI,
        ControllerVariant.GAUSSIAN_PRIOR_MPPI,
        ControllerVariant.EMPIRICAL_INIT_MPPI,

        # New stable representation variants.
        ControllerVariant.HOMOTOPY_SEEDED_MPPI,
        ControllerVariant.CORRIDOR_PRIOR_MPPI,
        ControllerVariant.FRENET_CORRIDOR_MPPI,
        ControllerVariant.HEATMAP_PRIOR_MPPI,
        ControllerVariant.CONTROL_BANK_MPPI,
        ControllerVariant.MODE_SELECTING_HOMOTOPY_MPPI,
        ControllerVariant.MODE_SELECTING_CORRIDOR_MPPI,

        # Baselines.
        ControllerVariant.STANDARD_MPPI,
        ControllerVariant.STANDARD_MPC,
        ControllerVariant.HOMOTOPY_MEAN_MPC,
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
                    mpc_cfg=mpc_cfg,
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

    # Plot one seed's trajectories for all variants.
    one_seed_results = [r for r in all_results if r["seed"] == RUN_SEEDS[0]]
    if one_seed_results:
        plot_path = f"{OUTPUT_PREFIX}_paths.png"
        plot_paths(one_seed_results, obstacles, start, goal, bounds_xy, plot_path)
        print(f"Saved path plot: {plot_path}")

    plt.show()


if __name__ == "__main__":
    main()