#!/usr/bin/env python3
"""
Combined swarm -> topological Gaussian mixture -> homotopy-conditioned MPPI.

This file combines two parts:

1. Your swarm / generative homotopy planner:
       - runs the fish/swarm rollout generator
       - groups rollouts by homotopy
       - fits a topological Gaussian trajectory mixture

           p(tau | E, xs, xg) = sum_h pi_h N(tau; mu_h, Sigma_h)

2. A probabilistic MPPI controller for a nonholonomic unicycle robot:
       - samples homotopy modes using pi_h
       - uses empirical swarm rollouts and mu_h as local receding-horizon references
       - uses Sigma_h in Mahalanobis tracking and uncertainty-aware obstacle cost
       - rolls out unicycle dynamics and applies the MPPI exponential update

The intended use is with the same local modules as your original script:

    geometry.utils
    RL.env2Ddiverse
    graph.graph
    planner

Run:

    python swarm_prior_mppi_unicycle.py

Expected local files/modules:
    save/best_policy.pkl
    geometry/utils.py
    RL/env2Ddiverse.py
    graph/graph.py
    planner.py

Outputs:
    swarm_prior_mppi_unicycle.png

If your planner returns the same object structure as in probabilistic(3).py, this file should plug in directly.
"""

from __future__ import annotations

import math
import pickle
import colorsys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time

# -----------------------------------------------------------------------------
# Imports from your existing project
# -----------------------------------------------------------------------------

try:
    from geometry.utils import round_obstacle, PolyObstacle, obstacles_to_segs
    from RL.env2Ddiverse import FishGoalEnv2D
    from graph.graph import build_full_graph
    from planner import HomotopyAwareGenerativePlanner, trajectory_cost
except Exception as exc:
    raise ImportError(
        "Could not import your project modules. Run this file from the root of "
        "your project, where geometry/, RL/, graph/, planner.py, and save/ exist.\n"
        f"Original import error: {exc}"
    )


Array = np.ndarray


# =============================================================================
# Part A. Trajectory utilities and probabilistic mixture fitting
# =============================================================================

def resample_path(path: Array, K: int) -> Array:
    """Arc-length resample a 2D path to K points."""
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"path must have shape (N,2), got {p.shape}")

    if p.shape[0] == 1:
        return np.repeat(p, K, axis=0)

    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.zeros(p.shape[0], dtype=np.float64)
    s[1:] = np.cumsum(d)
    total = s[-1]

    if total <= 1e-12:
        return np.repeat(p[:1], K, axis=0)

    s_new = np.linspace(0.0, total, K)
    x = np.interp(s_new, s, p[:, 0])
    y = np.interp(s_new, s, p[:, 1])
    return np.column_stack([x, y])


def flatten_path(path_K: Array) -> Array:
    return np.asarray(path_K, dtype=np.float64).reshape(-1)


def unflatten_path(vec: Array) -> Array:
    return np.asarray(vec, dtype=np.float64).reshape(-1, 2)


def stable_softmax_from_cost(costs: Array, beta: float) -> Array:
    """Convert costs to normalized exp(-beta cost) weights."""
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
    mean: Array          # D = 2K
    cov: Array           # D x D
    samples: Array       # M x D
    weights: Array       # M
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

    @property
    def signatures(self) -> List[Tuple[int, ...]]:
        return list(self.modes.keys())

    def prior(self) -> Dict[Tuple[int, ...], float]:
        return {h: m.probability for h, m in self.modes.items()}


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
) -> TopologicalTrajectoryMixture:
    """
    Fit p(tau,h) = pi_h N(tau; mu_h, Sigma_h) from swarm/fish samples.

    Assumes gen_out has:
        gen_out.samples
        gen_out.homotopy_groups

    The mixture weights are quality-weighted:
        w_i proportional to exp(-beta C(tau_i))
        pi_h = sum_{i in h} w_i / sum_i w_i
    """
    all_paths = list(gen_out.samples)
    if len(all_paths) == 0:
        raise RuntimeError("Swarm planner produced zero trajectory samples.")

    all_costs = np.array([
        trajectory_cost(p, costmap=costmap, bounds=bounds, w_len=1.0, w_smooth=0.05)
        for p in all_paths
    ], dtype=np.float64)
    all_weights = stable_softmax_from_cost(all_costs, beta=beta)

    weight_by_id = {id(p): float(w) for p, w in zip(all_paths, all_weights)}
    cost_by_id = {id(p): float(c) for p, c in zip(all_paths, all_costs)}

    mode_raw = {}
    total_mode_weight = 0.0

    for sig, paths in gen_out.homotopy_groups.items():
        if len(paths) < min_mode_samples:
            continue

        X = np.stack([flatten_path(resample_path(p, K)) for p in paths], axis=0)
        w = np.array([weight_by_id.get(id(p), 1.0) for p in paths], dtype=np.float64)
        c = np.array([cost_by_id.get(id(p), np.nan) for p in paths], dtype=np.float64)

        if np.sum(w) <= 1e-12:
            w = np.ones(len(paths), dtype=np.float64) / len(paths)
        else:
            w = w / np.sum(w)

        mu = np.sum(X * w[:, None], axis=0)
        Xc = X - mu[None, :]
        cov = (Xc * w[:, None]).T @ Xc
        cov = 0.5 * (cov + cov.T) + covariance_jitter * np.eye(cov.shape[0])

        mode_weight = float(np.sum([weight_by_id.get(id(p), 0.0) for p in paths]))
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
        raise RuntimeError(
            "No homotopy mode had enough samples. "
            "Lower min_mode_samples or increase swarm rollout count."
        )

    if total_mode_weight <= 1e-12:
        total_mode_weight = float(len(mode_raw))
        for sig in mode_raw:
            mode_raw[sig]["mode_weight"] = 1.0

    modes = {}
    for sig, d in mode_raw.items():
        pi = float(d["mode_weight"] / total_mode_weight)
        modes[sig] = GaussianTrajectoryMode(
            signature=sig,
            probability=pi,
            mean=d["mu"],
            cov=d["cov"],
            samples=d["X"],
            weights=d["w"],
            mean_cost=d["mean_cost"],
            count=int(d["X"].shape[0]),
        )

    return TopologicalTrajectoryMixture(modes=modes, K=K, beta=beta)


# =============================================================================
# Part B. Geometry / collision utilities for MPC
# =============================================================================

def normalize_plot_bounds(bounds):
    """
    Accept either:
      - bounds = (lower_xy, upper_xy), e.g. (array([0,0]), array([10,10]))
      - bounds = ((xmin, xmax), (ymin, ymax))
    Return xmin, xmax, ymin, ymax.
    """
    b0 = np.asarray(bounds[0], dtype=np.float64)
    b1 = np.asarray(bounds[1], dtype=np.float64)

    if b0.shape == (2,) and b1.shape == (2,) and b0[0] <= b1[0] and b0[1] <= b1[1]:
        xmin, ymin = b0
        xmax, ymax = b1
        return float(xmin), float(xmax), float(ymin), float(ymax)

    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    return float(xmin), float(xmax), float(ymin), float(ymax)


def _poly_vertices(obs) -> Array:
    if hasattr(obs, "vertices"):
        return np.asarray(obs.vertices, dtype=np.float64)[:, :2]
    return np.asarray(obs, dtype=np.float64)[:, :2]


def point_segment_distance_and_normal(p: Array, a: Array, b: Array) -> Tuple[float, Array]:
    """
    Distance from p to segment a-b and normal from the segment toward p.
    """
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        closest = a
    else:
        u = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
        closest = a + u * ab

    dvec = p - closest
    dist = float(np.linalg.norm(dvec))
    if dist <= 1e-12:
        normal = np.array([1.0, 0.0], dtype=np.float64)
    else:
        normal = dvec / dist
    return dist, normal


def point_in_poly(p: Array, poly: Array, eps: float = 1e-10) -> bool:
    """Ray-casting point-in-polygon test."""
    p = np.asarray(p, dtype=np.float64)
    poly = np.asarray(poly, dtype=np.float64)

    inside = False
    x, y = p
    n = poly.shape[0]

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        xi, yi = a
        xj, yj = b

        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / ((yj - yi) + 1e-18)
            if x < x_cross:
                inside = not inside

    return inside


def polygon_signed_distance_and_normal(p: Array, obs) -> Tuple[float, Array]:
    """
    Approximate signed distance from point p to polygon obstacle.

    Positive outside, negative inside.
    Normal points from obstacle boundary toward p.
    """
    poly = _poly_vertices(obs)
    best_dist = float("inf")
    best_normal = np.array([1.0, 0.0], dtype=np.float64)

    for i in range(poly.shape[0]):
        a = poly[i]
        b = poly[(i + 1) % poly.shape[0]]
        d, n = point_segment_distance_and_normal(p, a, b)
        if d < best_dist:
            best_dist = d
            best_normal = n

    if point_in_poly(p, poly):
        return -best_dist, best_normal
    return best_dist, best_normal


# =============================================================================
# Part C. MPPI modes extracted from the topological Gaussian mixture
# =============================================================================

@dataclass
class MPPIHomotopyMode:
    signature: Tuple[int, ...]
    probability: float
    mean_path: Array                 # K x 2
    cov_blocks: Array                # K x 2 x 2
    sample_paths: Optional[List[Array]] = None  # empirical swarm rollouts in this homotopy


def mixture_to_mppi_modes(mixture: TopologicalTrajectoryMixture) -> List[MPPIHomotopyMode]:
    """
    Convert the full trajectory covariance into per-time positional covariance blocks.

    Full covariance:
        Sigma_h has shape (2K, 2K)

    MPPI marginal block at time t:
        Sigma_h,t = Sigma_h[2t:2t+2, 2t:2t+2]
    """
    modes = []

    for sig, mode in mixture.modes.items():
        mean_path = mode.mean_path
        K = mean_path.shape[0]
        cov_blocks = np.zeros((K, 2, 2), dtype=np.float64)

        for t in range(K):
            cov_blocks[t] = mode.cov[2*t:2*t+2, 2*t:2*t+2]

        # Keep empirical swarm rollout samples for this homotopy.
        # These are used by MPPI as actual proposal/initialization trajectories,
        # not only to compute the Gaussian mean and covariance.
        sample_paths = [unflatten_path(v) for v in mode.samples]

        modes.append(
            MPPIHomotopyMode(
                signature=sig,
                probability=mode.probability,
                mean_path=mean_path,
                cov_blocks=cov_blocks,
                sample_paths=sample_paths,
            )
        )

    modes.sort(key=lambda m: m.probability, reverse=True)
    return modes


def localize_mode_for_state(mode: MPPIHomotopyMode, x_current: Array, H: int) -> MPPIHomotopyMode:
    """
    Convert a global homotopy path into a local receding-horizon reference.

    Important:
        Without this, the MPC repeatedly tracks the beginning of the global path.
    """
    mu = mode.mean_path
    d = np.linalg.norm(mu - x_current[:2], axis=1)
    idx = int(np.argmin(d))

    tail = mu[idx:]
    if len(tail) < 2:
        tail = mu[-2:]

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
    )


# =============================================================================
# Part D. Nonholonomic unicycle MPPI controller
# =============================================================================

@dataclass
class MPPIConfig:
    dt: float = 0.12
    horizon: int = 32
    num_rollouts: int = 900
    lambda_temperature: float = 7.0

    v_min: float = 0.0
    v_max: float = 1.6
    omega_min: float = -2.8
    omega_max: float = 2.8

    noise_v: float = 0.38
    noise_omega: float = 0.95
    temporal_noise_smoothing: float = 0.65

    # Probability that a rollout is initialized from an empirical swarm rollout
    # from the selected homotopy, rather than from the homotopy mean.
    # This is the main difference from the first version.
    swarm_init_probability: float = 0.70

    # Fast mode:
    # - vectorizes the expensive MPPI rollout simulation and cost evaluation
    # - approximates polygon obstacle distance by precomputed circular bounds
    #   inside the MPPI loop
    use_fast_vectorized_mppi: bool = True
    max_empirical_nominals_per_mode: int = 24

    robot_radius: float = 0.18
    base_safety_margin: float = 0.07
    uncertainty_margin_gain: float = 1.3

    w_goal: float = 30.0
    w_obstacle: float = 65.0
    w_control: float = 0.025
    w_control_smooth: float = 0.08
    w_heading: float = 0.08
    w_mode_prior: float = 0.8

    sigma_floor: float = 0.06


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
    X = np.zeros((len(U) + 1, 3), dtype=np.float64)
    X[0] = x0

    for t, u in enumerate(U):
        X[t + 1] = unicycle_step(X[t], u, dt)

    return X


def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def nominal_controls_to_track_path(x0: Array, ref: Array, cfg: MPPIConfig) -> Array:
    """
    Simple pure-pursuit-like nominal controller.

    MPPI samples around this nominal sequence.
    """
    U = np.zeros((cfg.horizon, 2), dtype=np.float64)
    x = x0.copy()

    for t in range(cfg.horizon):
        target = ref[min(t + 3, len(ref) - 1)]
        delta = target - x[:2]
        dist = float(np.linalg.norm(delta))

        desired_heading = math.atan2(delta[1], delta[0])
        err = wrap_angle(desired_heading - x[2])

        v = np.clip(
            0.45 + 1.8 * dist * max(0.15, math.cos(err)),
            cfg.v_min,
            cfg.v_max,
        )
        omega = np.clip(3.2 * err, cfg.omega_min, cfg.omega_max)

        U[t] = [v, omega]
        x = unicycle_step(x, U[t], cfg.dt)

    return U


def probabilistic_homotopy_cost(
    X: Array,
    U: Array,
    mode: MPPIHomotopyMode,
    obstacles: Sequence,
    goal: Array,
    cfg: MPPIConfig,
) -> float:
    """
    Rollout cost using the swarm-derived Gaussian representation.

    Uses:
        mu_h(t)      for reference tracking
        Sigma_h(t)   for Mahalanobis tracking
        Sigma_h(t)   for uncertainty-aware obstacle margin
        pi_h         as a homotopy prior penalty
    """
    cost = 0.0

    for t in range(cfg.horizon):
        p = X[t + 1, :2]
        mu = mode.mean_path[t]

        Sigma = mode.cov_blocks[t] + (cfg.sigma_floor ** 2) * np.eye(2)
        inv_Sigma = np.linalg.inv(Sigma)

        # 1. Covariance-aware tracking to the swarm mean.
        e = p - mu
        cost += float(e.T @ inv_Sigma @ e)

        # 2. Optional heading alignment to the local mean tangent.
        if t < cfg.horizon - 1:
            tangent = mode.mean_path[t + 1] - mode.mean_path[t]
            if np.linalg.norm(tangent) > 1e-9:
                ref_heading = math.atan2(tangent[1], tangent[0])
                cost += cfg.w_heading * float(wrap_angle(X[t + 1, 2] - ref_heading) ** 2)

        # 3. Uncertainty-aware obstacle risk.
        for obs in obstacles:
            d, normal = polygon_signed_distance_and_normal(p, obs)
            sigma_n = math.sqrt(max(0.0, float(normal.T @ Sigma @ normal)))

            # A larger tube toward the obstacle increases the effective margin.
            margin = (
                cfg.robot_radius
                + cfg.base_safety_margin
                + cfg.uncertainty_margin_gain * sigma_n
            )

            # Penalize being inside or too near the uncertainty-inflated obstacle.
            cost += cfg.w_obstacle * float(softplus(8.0 * (margin - d)) ** 2)

    # 4. Goal convergence.
    cost += cfg.w_goal * float(np.sum((X[-1, :2] - goal) ** 2))

    # 5. Control effort and control smoothness.
    cost += cfg.w_control * float(np.sum(U[:, 0] ** 2 + 0.15 * U[:, 1] ** 2))
    dU = np.diff(U, axis=0)
    cost += cfg.w_control_smooth * float(np.sum(dU[:, 0] ** 2 + 0.2 * dU[:, 1] ** 2))

    # 6. Homotopy prior from swarm probability.
    cost += cfg.w_mode_prior * (-math.log(mode.probability + 1e-12))

    return float(cost)


def localize_path_for_state(path: Array, x_current: Array, H: int) -> Array:
    """
    Localize an empirical swarm rollout to the current robot state.

    This lets MPPI initialize a rollout from an actual swarm trajectory sample,
    rather than always from the Gaussian mean.

    Returns:
        local path with shape H x 2.
    """
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        raise ValueError(f"Expected path with shape (N,2), got {p.shape}")

    d = np.linalg.norm(p - x_current[:2], axis=1)
    idx = int(np.argmin(d))

    tail = p[idx:]
    if len(tail) < 2:
        tail = p[-2:]

    return resample_path(tail, H)


def empirical_swarm_nominal_controls(
    x_current: Array,
    global_mode: MPPIHomotopyMode,
    cfg: MPPIConfig,
    rng: np.random.Generator,
) -> Optional[Array]:
    """
    Build an MPPI nominal control sequence from one empirical swarm rollout.

    This is the part that uses the swarm rollout as an actual initialization.
    """
    if not global_mode.sample_paths:
        return None

    sample_id = int(rng.integers(0, len(global_mode.sample_paths)))
    sample_path = global_mode.sample_paths[sample_id]
    local_sample_path = localize_path_for_state(sample_path, x_current, cfg.horizon)
    return nominal_controls_to_track_path(x_current, local_sample_path, cfg)


# =============================================================================
# Fast vectorized MPPI utilities
# =============================================================================

def obstacle_bounding_circles(obstacles: Sequence) -> List[Tuple[Array, float]]:
    """
    Approximate each polygon obstacle by a bounding circle for fast MPPI scoring.

    The exact polygon distance in probabilistic_homotopy_cost is expensive
    because it loops over polygon edges. For MPPI, this approximation is often
    acceptable because it is used as a soft risk cost, while the global swarm
    planner has already generated collision-free topological proposals.

    For final safety-critical use, keep a separate exact collision check.
    """
    circles = []
    for obs in obstacles:
        poly = _poly_vertices(obs)
        center = poly.mean(axis=0)
        radius = float(np.max(np.linalg.norm(poly - center[None, :], axis=1)))
        circles.append((center, radius))
    return circles


def build_empirical_nominal_bank(
    x_current: Array,
    global_mode: MPPIHomotopyMode,
    mean_nominal: Array,
    cfg: MPPIConfig,
    rng: np.random.Generator,
) -> List[Array]:
    """
    Build a small bank of nominal controls for one homotopy mode.

    Rather than generating a fresh empirical nominal for every rollout, which is
    expensive, we precompute a small bank once per MPC step and let rollouts
    sample from it.
    """
    bank = [mean_nominal]

    if not global_mode.sample_paths:
        return bank

    n = min(cfg.max_empirical_nominals_per_mode, len(global_mode.sample_paths))
    ids = rng.choice(len(global_mode.sample_paths), size=n, replace=False)

    for sid in ids:
        sample_path = global_mode.sample_paths[int(sid)]
        local_sample_path = localize_path_for_state(sample_path, x_current, cfg.horizon)
        U_emp = nominal_controls_to_track_path(x_current, local_sample_path, cfg)
        bank.append(U_emp)

    return bank


def rollout_unicycle_batch(x0: Array, U: Array, dt: float) -> Array:
    """
    Vectorized unicycle rollout.

    Args:
        x0: shape (3,)
        U:  shape (N, H, 2)

    Returns:
        X: shape (N, H+1, 3)
    """
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


def fast_mode_rollout_costs(
    X: Array,
    U: Array,
    mode: MPPIHomotopyMode,
    obstacle_circles: List[Tuple[Array, float]],
    goal: Array,
    cfg: MPPIConfig,
) -> Array:
    """
    Vectorized cost for all rollouts assigned to one homotopy mode.

    Args:
        X: shape (N, H+1, 3)
        U: shape (N, H, 2)

    Returns:
        costs: shape (N,)
    """
    N = U.shape[0]
    H = cfg.horizon
    costs = np.zeros(N, dtype=np.float64)

    P = X[:, 1:H+1, :2]  # N x H x 2

    for t in range(H):
        p = P[:, t, :]              # N x 2
        mu = mode.mean_path[t]      # 2
        Sigma = mode.cov_blocks[t] + (cfg.sigma_floor ** 2) * np.eye(2)
        inv_Sigma = np.linalg.inv(Sigma)

        E = p - mu[None, :]
        costs += np.einsum("ni,ij,nj->n", E, inv_Sigma, E)

        if t < H - 1:
            tangent = mode.mean_path[t + 1] - mode.mean_path[t]
            if np.linalg.norm(tangent) > 1e-9:
                ref_heading = math.atan2(tangent[1], tangent[0])
                costs += cfg.w_heading * wrap_angle(X[:, t + 1, 2] - ref_heading) ** 2

        for center, radius in obstacle_circles:
            dvec = p - center[None, :]
            norm = np.linalg.norm(dvec, axis=1) + 1e-12
            normal = dvec / norm[:, None]
            d = norm - radius

            sigma_n_sq = np.einsum("ni,ij,nj->n", normal, Sigma, normal)
            sigma_n = np.sqrt(np.maximum(0.0, sigma_n_sq))

            margin = (
                cfg.robot_radius
                + cfg.base_safety_margin
                + cfg.uncertainty_margin_gain * sigma_n
            )
            costs += cfg.w_obstacle * softplus(8.0 * (margin - d)) ** 2

    costs += cfg.w_goal * np.sum((X[:, -1, :2] - goal[None, :]) ** 2, axis=1)

    costs += cfg.w_control * np.sum(U[:, :, 0] ** 2 + 0.15 * U[:, :, 1] ** 2, axis=1)

    dU = np.diff(U, axis=1)
    costs += cfg.w_control_smooth * np.sum(dU[:, :, 0] ** 2 + 0.2 * dU[:, :, 1] ** 2, axis=1)

    costs += cfg.w_mode_prior * (-math.log(mode.probability + 1e-12))

    return costs


def homotopy_swarm_mppi_step_fast(
    x_current: Array,
    global_modes: List[MPPIHomotopyMode],
    obstacles: Sequence,
    goal: Array,
    cfg: MPPIConfig,
    rng: np.random.Generator,
) -> Tuple[Array, Dict[str, object]]:
    """
    Fast vectorized MPPI step.

    Main speedups:
        - precompute a small bank of empirical swarm nominals per homotopy
        - vectorize rollout simulation per homotopy
        - vectorize cost evaluation per homotopy
        - approximate polygon obstacles by bounding circles inside MPPI
    """
    local_modes = [localize_mode_for_state(m, x_current, cfg.horizon) for m in global_modes]

    pi = np.array([m.probability for m in local_modes], dtype=np.float64)
    pi = pi / (pi.sum() + 1e-12)

    mode_ids = rng.choice(len(local_modes), size=cfg.num_rollouts, p=pi)

    obstacle_circles = obstacle_bounding_circles(obstacles)

    all_costs = np.zeros(cfg.num_rollouts, dtype=np.float64)
    all_U0 = np.zeros((cfg.num_rollouts, 2), dtype=np.float64)

    noise_scale = np.array([cfg.noise_v, cfg.noise_omega], dtype=np.float64)

    for mid, mode in enumerate(local_modes):
        ids = np.where(mode_ids == mid)[0]
        n = len(ids)
        if n == 0:
            continue

        mean_nominal = nominal_controls_to_track_path(x_current, mode.mean_path, cfg)
        nominal_bank = build_empirical_nominal_bank(
            x_current=x_current,
            global_mode=global_modes[mid],
            mean_nominal=mean_nominal,
            cfg=cfg,
            rng=rng,
        )

        # Choose which nominal each rollout uses.
        if len(nominal_bank) == 1:
            bank_ids = np.zeros(n, dtype=int)
        else:
            # Empirical bank entries get most probability, mean nominal remains available.
            probs = np.ones(len(nominal_bank), dtype=np.float64)
            probs[0] = max(1e-6, 1.0 - cfg.swarm_init_probability)
            if len(nominal_bank) > 1:
                probs[1:] = cfg.swarm_init_probability / (len(nominal_bank) - 1)
            probs /= probs.sum()
            bank_ids = rng.choice(len(nominal_bank), size=n, p=probs)

        U = np.stack([nominal_bank[j].copy() for j in bank_ids], axis=0)

        noise = rng.normal(size=(n, cfg.horizon, 2)) * noise_scale[None, None, :]

        alpha = cfg.temporal_noise_smoothing
        for t in range(1, cfg.horizon):
            noise[:, t, :] = alpha * noise[:, t - 1, :] + (1.0 - alpha) * noise[:, t, :]

        U += noise
        U[:, :, 0] = np.clip(U[:, :, 0], cfg.v_min, cfg.v_max)
        U[:, :, 1] = np.clip(U[:, :, 1], cfg.omega_min, cfg.omega_max)

        X = rollout_unicycle_batch(x_current, U, cfg.dt)
        costs = fast_mode_rollout_costs(X, U, mode, obstacle_circles, goal, cfg)

        all_costs[ids] = costs
        all_U0[ids] = U[:, 0, :]

    rho = float(all_costs.min())
    weights = np.exp(-(all_costs - rho) / cfg.lambda_temperature)
    weights = weights / (weights.sum() + 1e-12)

    u_apply = weights @ all_U0
    u_apply[0] = np.clip(u_apply[0], cfg.v_min, cfg.v_max)
    u_apply[1] = np.clip(u_apply[1], cfg.omega_min, cfg.omega_max)

    mode_weight = {
        str(m.signature): float(weights[mode_ids == i].sum())
        for i, m in enumerate(local_modes)
    }

    info = {
        "cost_min": float(all_costs.min()),
        "cost_mean": float(all_costs.mean()),
        "mode_weight": mode_weight,
        "fast": True,
    }

    return u_apply, info


def homotopy_swarm_mppi_step(
    x_current: Array,
    global_modes: List[MPPIHomotopyMode],
    obstacles: Sequence,
    goal: Array,
    cfg: MPPIConfig,
    rng: np.random.Generator,
) -> Tuple[Array, Dict[str, object]]:
    """
    One MPPI step.

    Each rollout:
        1. samples homotopy h ~ pi_h
        2. localizes mu_h and Sigma_h to the current robot position
        3. builds nominal controls toward the local mean
        4. adds Gaussian control noise
        5. rolls out unicycle dynamics
        6. scores using mean/covariance/prior/obstacles
    """
    local_modes = [localize_mode_for_state(m, x_current, cfg.horizon) for m in global_modes]

    pi = np.array([m.probability for m in local_modes], dtype=np.float64)
    pi = pi / (pi.sum() + 1e-12)

    nominal_by_mode = [
        nominal_controls_to_track_path(x_current, m.mean_path, cfg)
        for m in local_modes
    ]

    mode_ids = rng.choice(len(local_modes), size=cfg.num_rollouts, p=pi)

    costs = np.zeros(cfg.num_rollouts, dtype=np.float64)
    U0 = np.zeros((cfg.num_rollouts, 2), dtype=np.float64)

    noise_scale = np.array([cfg.noise_v, cfg.noise_omega], dtype=np.float64)

    for k in range(cfg.num_rollouts):
        mid = int(mode_ids[k])
        mode = local_modes[mid]

        # Initialization/proposal center for this rollout.
        # With probability cfg.swarm_init_probability, use one empirical swarm
        # rollout from the selected homotopy as the nominal path. Otherwise use
        # the Gaussian mean path. The cost is still evaluated against the
        # Gaussian mode, so mu_h/Sigma_h remain active priors.
        U_empirical = None
        if rng.random() < cfg.swarm_init_probability:
            U_empirical = empirical_swarm_nominal_controls(
                x_current=x_current,
                global_mode=global_modes[mid],
                cfg=cfg,
                rng=rng,
            )

        if U_empirical is not None:
            U = U_empirical
        else:
            U = nominal_by_mode[mid].copy()

        noise = rng.normal(size=(cfg.horizon, 2)) * noise_scale[None, :]

        # Temporally correlated noise gives smoother sampled controls.
        alpha = cfg.temporal_noise_smoothing
        for t in range(1, cfg.horizon):
            noise[t] = alpha * noise[t - 1] + (1.0 - alpha) * noise[t]

        U += noise
        U[:, 0] = np.clip(U[:, 0], cfg.v_min, cfg.v_max)
        U[:, 1] = np.clip(U[:, 1], cfg.omega_min, cfg.omega_max)

        X = rollout_unicycle(x_current, U, cfg.dt)
        costs[k] = probabilistic_homotopy_cost(X, U, mode, obstacles, goal, cfg)
        U0[k] = U[0]

    # Standard MPPI exponential transform.
    rho = float(costs.min())
    weights = np.exp(-(costs - rho) / cfg.lambda_temperature)
    weights = weights / (weights.sum() + 1e-12)

    u_apply = weights @ U0
    u_apply[0] = np.clip(u_apply[0], cfg.v_min, cfg.v_max)
    u_apply[1] = np.clip(u_apply[1], cfg.omega_min, cfg.omega_max)

    mode_weight = {
        str(m.signature): float(weights[mode_ids == i].sum())
        for i, m in enumerate(local_modes)
    }

    info = {
        "cost_min": float(costs.min()),
        "cost_mean": float(costs.mean()),
        "mode_weight": mode_weight,
    }

    return u_apply, info


# =============================================================================
# Part E. Build your original obstacle scene and run the combined pipeline
# =============================================================================

def build_default_scene():
    """
    Same style as your original probabilistic script.
    """
    scale = 4.0

    bounds_xy = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    bounds_ranges = ((0.0, 10.0), (0.0, 10.0))

    start = np.array([1.0, 1.0], dtype=np.float64)
    goal = np.array([9.0, 9.0], dtype=np.float64)

    obstacles = [
        PolyObstacle(round_obstacle(
            np.array([[3.0, 1.5], [5.2, 2.2], [4.7, 4.0], [2.8, 3.4]]),
            n_iters=4,
            n_points=32,
        )),
        PolyObstacle(round_obstacle(
            np.array([[6.2, 6.0], [8.5, 6.3], [8.1, 8.4], [6.8, 8.9], [5.9, 7.4]]),
            n_iters=4,
            n_points=32,
        )),
        PolyObstacle(round_obstacle(
            np.array([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]]),
            n_iters=4,
            n_points=32,
        )),
    ]

    return scale, bounds_xy, bounds_ranges, start, goal, obstacles


def run_swarm_planner(
    start: Array,
    goal: Array,
    obstacles: Sequence,
    scale: float,
    bounds_xy,
    *,
    boid_count: int = 1200,
    max_steps: int = 700,
    dt: float = 0.5,
    seed: int = 3,
):
    """
    Run your homotopy-aware generative/swarm planner.
    """
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
        boid_count=boid_count,
        max_steps=max_steps,
        dt=dt,
    )

    gen_out = planner.sample(
        start_unscaled=start,
        goal_unscaled=goal,
        graph_goals=graph_goals,
        graph_W=graph_W,
        seed=seed,
    )

    return gen_out


def run_mppi_controller(
    modes: List[MPPIHomotopyMode],
    obstacles: Sequence,
    start: Array,
    goal: Array,
    *,
    start_heading: Optional[float] = None,
    cfg: Optional[MPPIConfig] = None,
    seed: int = 7,
    max_mpc_steps: int = 120,
    goal_tolerance: float = 0.35,
):
    """
    Run receding-horizon MPPI using swarm-derived homotopy Gaussian priors.
    """
    if cfg is None:
        cfg = MPPIConfig()

    if start_heading is None:
        direction = goal - start
        start_heading = math.atan2(direction[1], direction[0])

    x = np.array([start[0], start[1], start_heading], dtype=np.float64)

    rng = np.random.default_rng(seed)

    states = [x.copy()]
    controls = []
    infos = []

    for step in range(max_mpc_steps):
        if cfg.use_fast_vectorized_mppi:
            u, info = homotopy_swarm_mppi_step_fast(
                x_current=x,
                global_modes=modes,
                obstacles=obstacles,
                goal=goal,
                cfg=cfg,
                rng=rng,
            )
        else:
            u, info = homotopy_swarm_mppi_step(
                x_current=x,
                global_modes=modes,
                obstacles=obstacles,
                goal=goal,
                cfg=cfg,
                rng=rng,
            )

        x = unicycle_step(x, u, cfg.dt)

        states.append(x.copy())
        controls.append(u.copy())
        infos.append(info)

        if np.linalg.norm(x[:2] - goal) <= goal_tolerance:
            break

    return {
        "states": np.asarray(states),
        "controls": np.asarray(controls),
        "infos": infos,
        "cfg": cfg,
    }


# =============================================================================
# Part F. Visualization
# =============================================================================

def setup_workspace(ax, obstacles, start, goal, bounds, title=None):
    xmin, xmax, ymin, ymax = normalize_plot_bounds(bounds)

    for obs in obstacles:
        p = _poly_vertices(obs)
        ax.fill(p[:, 0], p[:, 1], alpha=0.25)
        ax.plot(
            np.r_[p[:, 0], p[0, 0]],
            np.r_[p[:, 1], p[0, 1]],
            linewidth=1.0,
        )

    ax.scatter([start[0]], [start[1]], s=80, marker="o", label="start")
    ax.scatter([goal[0]], [goal[1]], s=140, marker="*", label="goal")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    if title:
        ax.set_title(title)


def add_covariance_ellipse(ax, mean_xy, cov_xy, nsig=2.0, **kwargs):
    cov_xy = 0.5 * (cov_xy + cov_xy.T)
    vals, vecs = np.linalg.eigh(cov_xy)
    vals = np.maximum(vals, 1e-12)

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    width = 2.0 * nsig * math.sqrt(vals[0])
    height = 2.0 * nsig * math.sqrt(vals[1])

    ell = Ellipse(
        xy=mean_xy,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        **kwargs,
    )
    ax.add_patch(ell)
    return ell


def plot_combined_result(
    gen_out,
    mixture: TopologicalTrajectoryMixture,
    modes: List[MPPIHomotopyMode],
    mppi_result,
    obstacles,
    start,
    goal,
    bounds,
    save_path: str = "swarm_prior_mppi_unicycle.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ax0, ax1 = axes

    # Panel 1: raw swarm rollouts grouped implicitly + mixture means.
    setup_workspace(ax0, obstacles, start, goal, bounds, "Swarm rollouts and extracted Gaussian homotopy modes")

    raw = list(gen_out.samples)
    if len(raw) > 250:
        ids = np.linspace(0, len(raw) - 1, 250).astype(int)
        raw = [raw[i] for i in ids]

    for p in raw:
        ax0.plot(p[:, 0], p[:, 1], linewidth=0.5, alpha=0.13)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for k, mode in enumerate(modes):
        color = colors[k % len(colors)]
        mu = mode.mean_path
        ax0.plot(
            mu[:, 0],
            mu[:, 1],
            color=color,
            linewidth=2.4,
            label=f"h{k}: pi={mode.probability:.2f}, sig={mode.signature}",
        )

        for t in np.linspace(4, len(mu) - 5, 5).astype(int):
            add_covariance_ellipse(
                ax0,
                mu[t],
                mode.cov_blocks[t],
                nsig=2.0,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.45,
            )

    ax0.legend(fontsize=7, loc="lower right")

    # Panel 2: MPPI result.
    setup_workspace(ax1, obstacles, start, goal, bounds, "MPPI execution using swarm Gaussian modes as priors")

    for k, mode in enumerate(modes):
        color = colors[k % len(colors)]
        mu = mode.mean_path
        ax1.plot(mu[:, 0], mu[:, 1], color=color, linewidth=1.2, alpha=0.45)

    states = mppi_result["states"]
    ax1.plot(states[:, 0], states[:, 1], linewidth=3.0, label="executed MPPI trajectory")
    ax1.quiver(
        states[::8, 0],
        states[::8, 1],
        np.cos(states[::8, 2]),
        np.sin(states[::8, 2]),
        angles="xy",
        scale_units="xy",
        scale=4.0,
        width=0.004,
        alpha=0.65,
    )

    final_dist = float(np.linalg.norm(states[-1, :2] - goal))
    ax1.text(
        0.02,
        0.98,
        f"steps={len(states)-1}\nfinal dist={final_dist:.3f}",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )
    ax1.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig, axes


# =============================================================================
# Main
# =============================================================================

def main():
    scale, bounds_xy, bounds_ranges, start, goal, obstacles = build_default_scene()

    print("Running swarm / homotopy-aware generative planner...")
    gen_out = run_swarm_planner(
        start=start,
        goal=goal,
        obstacles=obstacles,
        scale=scale,
        bounds_xy=bounds_xy,
        boid_count=1200,
        max_steps=700,
        dt=0.5,
        seed=3,
    )

    print(f"Generated swarm trajectories: {len(gen_out.samples)}")
    print(f"Empirical homotopy groups: {len(gen_out.homotopy_groups)}")

    if hasattr(gen_out, "probabilities"):
        print("Raw empirical homotopy probabilities:")
        for sig, prob in sorted(gen_out.probabilities.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {sig}: {prob:.3f}")

    print("\nFitting topological Gaussian trajectory mixture...")
    mixture = fit_topological_trajectory_mixture(
        gen_out,
        obstacles,
        K=50,
        beta=1.0,
        min_mode_samples=3,
        covariance_jitter=2e-4,
        costmap=None,
        bounds=bounds_ranges,
    )

    print("Quality-weighted Gaussian modes:")
    for i, (sig, mode) in enumerate(sorted(mixture.modes.items(), key=lambda kv: kv[1].probability, reverse=True)):
        print(
            f"  h{i}: sig={sig}, pi={mode.probability:.3f}, "
            f"count={mode.count}, mean_cost={mode.mean_cost:.3f}"
        )

    modes = mixture_to_mppi_modes(mixture)

    print("\nRunning homotopy-conditioned MPPI with unicycle dynamics...")
    cfg = MPPIConfig(
        horizon=28,
        num_rollouts=500,
        dt=0.12,
        use_fast_vectorized_mppi=True,
        max_empirical_nominals_per_mode=16,
    )

    t0 = time.perf_counter()
    mppi_result = run_mppi_controller(
        modes=modes,
        obstacles=obstacles,
        start=start,
        goal=goal,
        start_heading=None,
        cfg=cfg,
        seed=7,
        max_mpc_steps=120,
        goal_tolerance=0.35,
    )
    t1 = time.perf_counter()
    print(f"MPPI took {t1-t0:.3f} seconds")

    states = mppi_result["states"]
    controls = mppi_result["controls"]
    final_dist = float(np.linalg.norm(states[-1, :2] - goal))

    print(f"MPPI steps: {len(states)-1}")
    print(f"Final position: {states[-1, :2]}")
    print(f"Distance to goal: {final_dist:.3f}")
    if len(controls):
        print(f"Mean v: {controls[:, 0].mean():.3f}")
        print(f"Mean omega: {controls[:, 1].mean():.3f}")

    out_path = "swarm_prior_mppi_unicycle.png"
    plot_combined_result(
        gen_out=gen_out,
        mixture=mixture,
        modes=modes,
        mppi_result=mppi_result,
        obstacles=obstacles,
        start=start,
        goal=goal,
        bounds=bounds_xy,
        save_path=out_path,
    )
    print(f"Saved plot: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()