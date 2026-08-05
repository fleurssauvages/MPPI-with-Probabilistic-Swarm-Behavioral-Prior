from __future__ import annotations
import math
import pickle
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np
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
    raise ImportError(f'Could not import your project modules. Run from the root of your project, where geometry/, planner/, graph/, planner.py, and save/ exist.\nOriginal import error: {exc}')
NUMBA_AVAILABLE = njit is not None
__all__ = ['ControllerVariant', 'MPPIConfig', 'Scene', 'SimulationResult', 'DynamicWallScenario', 'build_default_scene', 'default_dynamic_wall_scenarios', 'obstacle_center', 'make_wall_blockers_between_centers', 'build_homotopy_modes', 'run_controller', 'min_clearance', 'minimum_clearance', 'obstacle_bounding_circles', 'localize_mode_for_state', 'localize_path_for_state', 'NUMBA_AVAILABLE']
Array = np.ndarray

class ControllerVariant(str, Enum):
    SENSITIVITY_PROJECTED_GAUSSIAN_MPPI = 'sensitivity_projected_gaussian_prior_mppi'
    GAUSSIAN_PRIOR_MPPI = 'gaussian_prior_mppi'
    CORRIDOR_PRIOR_MPPI = 'corridor_prior_mppi'
    CONTROL_BANK_MPPI = 'control_bank_mppi'
    MODE_SELECTING_GAUSSIAN_MPPI = 'mode_selecting_gaussian_mppi'
    MODE_SELECTING_CORRIDOR_MPPI = 'mode_selecting_corridor_mppi'
    STANDARD_MPPI = 'standard_mppi'
    STANDARD_MPPI_128 = 'standard_mppi_128_rollouts'

@dataclass
class MPPIConfig:
    dt: float = 0.12
    horizon: int = 50
    num_rollouts: int = 64
    lambda_temperature: float = 2.2
    v_min: float = -1.0
    v_max: float = 2.8
    omega_min: float = -4.5
    omega_max: float = 4.5
    noise_v: float = 0.5
    noise_omega: float = 0.9
    temporal_noise_smoothing: float = 0.72
    gaussian_covariance_scale: float = 4.0
    spg_lookahead_steps: int = 10
    spg_fd_accel: float = 0.05
    spg_fd_steering_rate: float = 0.05
    spg_pseudoinverse_damping: float = 0.001
    spg_covariance_jitter: float = 1e-08
    swarm_init_probability: float = 0.6
    max_empirical_nominals_per_mode: int = 16
    robot_radius: float = 0.18
    hard_collision_clearance: float = 0.01
    suppress_blocked_modes: bool = True
    mode_blocking_clearance: float = 0.02
    mode_blocking_substeps: int = 2
    w_goal: float = 110.0
    w_obstacle: float = 500.0
    w_control: float = 0.004
    w_control_smooth: float = 0.4
    sigma_floor: float = 0.25
    goal_tolerance: float = 0.305
    use_monotonic_reference_progress: bool = True
    max_reference_index_advance: int = 4
    max_delta_v: float = 0.7
    max_delta_omega: float = 1.4
    enforce_one_step_safety: bool = True
    one_step_safety_clearance: float = 0.0
    terminal_slowdown_radius: float = 0.75
    terminal_max_speed: float = 0.55
    terminal_distance_gain: float = 1.8
    terminal_heading_gain: float = 1.2
    terminal_heading_deadzone: float = 0.45
    terminal_blend_power: float = 1.5
    low_noise_proposal_count: int = 1
    low_noise_proposal_scale: float = 0.15
    mode_select_top_k: int = 4
    mode_select_rollouts_per_mode: int = 0
    max_nearby_prior_modes: int = 3
    nearby_prior_distance_slack: float = 0.75
    nearby_prior_blocked_penalty: float = 1.25

    def __post_init__(self) -> None:
        self.spg_lookahead_steps = max(1, int(self.spg_lookahead_steps))
        if self.v_min > self.v_max or self.omega_min > self.omega_max:
            raise ValueError('Invalid unicycle control bounds.')
        if self.spg_fd_accel <= 0.0 or self.spg_fd_steering_rate <= 0.0:
            raise ValueError('SPG finite-difference steps must be positive.')
        if self.spg_pseudoinverse_damping < 0.0 or self.spg_covariance_jitter < 0.0:
            raise ValueError('SPG damping and covariance jitter must be nonnegative.')

@dataclass(frozen=True)
class Scene:
    scale: float
    bounds_xy: tuple[Array, Array]
    planner_bounds: tuple[tuple[float, float], tuple[float, float]]
    start: Array
    goal: Array
    obstacles: tuple[object, ...]

@dataclass
class SimulationResult:
    states: Array
    controls: Array
    infos: list[dict[str, object]]
    runtime: float
    activation_step: Optional[int]
    obstacle_history: list[list[object]]
    reached_goal: bool

def resample_path(path: Array, K: int) -> Array:
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f'path must have shape (N,2), got {p.shape}')
    if p.shape[0] == 1:
        return np.repeat(p, K, axis=0)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.zeros(p.shape[0], dtype=np.float64)
    s[1:] = np.cumsum(d)
    if s[-1] <= 1e-12:
        return np.repeat(p[:1], K, axis=0)
    q = np.linspace(0.0, s[-1], K)
    return np.column_stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])])

def snap_path_end_to_goal(path: Array, goal: Optional[Array], snap_radius: float=0.2, straight_tail_points: int=8) -> Array:
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

def fit_topological_trajectory_mixture(gen_out, *, K: int=50, beta: float=1.0, min_mode_samples: int=3, covariance_jitter: float=0.0002, costmap=None, bounds=((0.0, 10.0), (0.0, 10.0)), goal: Optional[Array]=None, snap_to_goal_radius: float=0.2, snap_straight_tail_points: int=8) -> TopologicalTrajectoryMixture:
    raw_paths = list(gen_out.samples)
    if len(raw_paths) == 0:
        raise RuntimeError('Swarm planner produced zero trajectory samples.')
    all_paths = [snap_path_end_to_goal(p, goal=goal, snap_radius=snap_to_goal_radius, straight_tail_points=snap_straight_tail_points) for p in raw_paths]
    all_costs = np.array([trajectory_cost(p, costmap=costmap, bounds=bounds, w_len=1.0, w_smooth=0.05) for p in all_paths], dtype=np.float64)
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
        mode_raw[sig] = dict(X=X, w=w, mu=mu, cov=cov, mode_weight=mode_weight, mean_cost=float(np.nanmean(c)))
    if not mode_raw:
        raise RuntimeError('No homotopy group had enough samples.')
    if total_mode_weight <= 1e-12:
        total_mode_weight = float(len(mode_raw))
        for sig in mode_raw:
            mode_raw[sig]['mode_weight'] = 1.0
    modes = {}
    for sig, d in mode_raw.items():
        modes[sig] = GaussianTrajectoryMode(signature=sig, probability=float(d['mode_weight'] / total_mode_weight), mean=d['mu'], cov=d['cov'], samples=d['X'], weights=d['w'], mean_cost=d['mean_cost'], count=int(d['X'].shape[0]))
    return TopologicalTrajectoryMixture(modes=modes, K=K, beta=beta)

def mixture_to_mppi_modes(mixture: TopologicalTrajectoryMixture) -> List[MPPIHomotopyMode]:
    modes = []
    for sig, mode in mixture.modes.items():
        mean_path = mode.mean_path
        K = mean_path.shape[0]
        cov_blocks = np.zeros((K, 2, 2), dtype=np.float64)
        for t in range(K):
            cov_blocks[t] = mode.cov[2 * t:2 * t + 2, 2 * t:2 * t + 2]
        sample_paths = [unflatten_path(v) for v in mode.samples]
        modes.append(prepare_mode_prior_cache(MPPIHomotopyMode(signature=sig, probability=mode.probability, mean_path=mean_path, cov_blocks=cov_blocks, sample_paths=sample_paths)))
    modes.sort(key=lambda m: m.probability, reverse=True)
    return modes

def _poly_vertices(obs) -> Array:
    if hasattr(obs, 'vertices'):
        return np.asarray(obs.vertices, dtype=np.float64)[:, :2]
    return np.asarray(obs, dtype=np.float64)[:, :2]

def point_segment_distance_and_normal(p: Array, a: Array, b: Array) -> Tuple[float, Array]:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        closest = a
    else:
        u = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
        closest = a + u * ab
    dvec = p - closest
    dist = float(np.linalg.norm(dvec))
    normal = np.array([1.0, 0.0]) if dist <= 1e-12 else dvec / dist
    return (dist, normal)

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

def polygon_signed_distance_and_normal(p: Array, obs) -> Tuple[float, Array]:
    poly = _poly_vertices(obs)
    best_dist = 1e309
    best_normal = np.array([1.0, 0.0])
    for i in range(poly.shape[0]):
        d, n = point_segment_distance_and_normal(p, poly[i], poly[(i + 1) % poly.shape[0]])
        if d < best_dist:
            best_dist = d
            best_normal = n
    if point_in_poly(p, poly):
        return (-best_dist, best_normal)
    return (best_dist, best_normal)

def obstacle_bounding_circles(obstacles: Sequence, *, elongated_aspect_ratio: float=2.25, max_segment_length: float=0.1, wall_max_segment_length: float=0.15) -> List[Tuple[Array, float]]:
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
        target_segment_length = wall_max_segment_length if len(poly) == 4 else max_segment_length
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
        return float(min_clearance_nb(np.asarray(states, dtype=np.float64), padded, lengths, float(robot_radius)))
    vals = []
    for x in states:
        p = x[:2]
        for obs in obstacles:
            d, _ = polygon_signed_distance_and_normal(p, obs)
            vals.append(d - robot_radius)
    return float(np.min(vals)) if vals else 1e309

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def unicycle_step(x: Array, u: Array, dt: float) -> Array:
    px, py, th = x
    v, om = u
    return np.array([px + v * math.cos(th) * dt, py + v * math.sin(th) * dt, wrap_angle(th + om * dt)], dtype=np.float64)

def segment_goal_entry_state(x0: Array, x1: Array, goal: Array, goal_tolerance: float) -> Tuple[bool, Array]:
    p0 = np.asarray(x0[:2], dtype=np.float64)
    p1 = np.asarray(x1[:2], dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    r = float(goal_tolerance)
    if np.linalg.norm(p0 - g) <= r:
        return (True, np.asarray(x0, dtype=np.float64).copy())
    if np.linalg.norm(p1 - g) <= r:
        return (True, np.asarray(x1, dtype=np.float64).copy())
    d = p1 - p0
    a = float(d @ d)
    if a <= 1e-16:
        return (False, np.asarray(x1, dtype=np.float64).copy())
    f = p0 - g
    b = 2.0 * float(f @ d)
    c = float(f @ f) - r * r
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return (False, np.asarray(x1, dtype=np.float64).copy())
    root = math.sqrt(max(0.0, disc))
    roots = [q for q in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if 0.0 <= q <= 1.0]
    if not roots:
        return (False, np.asarray(x1, dtype=np.float64).copy())
    alpha = float(min(roots))
    hit = np.asarray(x0, dtype=np.float64).copy()
    hit[:2] = p0 + alpha * d
    hit[2] = wrap_angle(float(x0[2]) + alpha * wrap_angle(float(x1[2]) - float(x0[2])))
    return (True, hit)

def rollout_unicycle(x0: Array, U: Array, dt: float) -> Array:
    if rollout_unicycle_single_nb is not None:
        return rollout_unicycle_single_nb(np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), float(dt))
    X = np.zeros((len(U) + 1, 3), dtype=np.float64)
    X[0] = x0
    for t, u in enumerate(U):
        X[t + 1] = unicycle_step(X[t], u, dt)
    return X

def rollout_unicycle_batch(x0: Array, U: Array, dt: float) -> Array:
    if rollout_unicycle_batch_nb is not None:
        return rollout_unicycle_batch_nb(np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), float(dt))
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

def _ensure_mode_prior_cache(mode: MPPIHomotopyMode) -> MPPIHomotopyMode:
    if mode.arc_length is None or mode.gaussian_variance is None:
        return prepare_mode_prior_cache(mode)
    return mode
if njit is not None:

    @njit(cache=True)
    def localize_prior_horizon_nb(mean_path, cov_blocks, arc_length, gaussian_variance, start_index, horizon):
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
                    local_cov[t, j, k] = beta * cov_blocks[left, j, k] + alpha * cov_blocks[right, j, k]
            local_gaussian[t] = beta * gaussian_variance[left] + alpha * gaussian_variance[right]
        return (local_mean, local_cov, local_gaussian)

    @njit(cache=True)
    def apply_gaussian_prior_noise_nb(noise, variance, sigma_floor, covariance_scale):
        floor_var = sigma_floor * sigma_floor
        reference_std = max(sigma_floor, 1e-09)
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
    def nominal_controls_to_track_path_nb(x0, ref, horizon, dt, v_min, v_max, omega_min, omega_max):
        U = np.zeros((horizon, 2), dtype=np.float64)
        px = x0[0]
        py = x0[1]
        theta = x0[2]
        ref_len = ref.shape[0]
        for t in range(horizon):
            target_idx = t + 3
            if target_idx >= ref_len:
                target_idx = ref_len - 1
            dx = ref[target_idx, 0] - px
            dy = ref[target_idx, 1] - py
            dist = math.sqrt(dx * dx + dy * dy)
            desired_heading = math.atan2(dy, dx)
            err = _wrap_angle_nb(desired_heading - theta)
            forward = math.cos(err)
            if forward < 0.0:
                forward = 0.0
            heading_scale = forward * forward
            v = 0.2 + 2.4 * dist * heading_scale
            if v < v_min:
                v = v_min
            elif v > v_max:
                v = v_max
            omega = 3.2 * err
            if omega < omega_min:
                omega = omega_min
            elif omega > omega_max:
                omega = omega_max
            U[t, 0] = v
            U[t, 1] = omega
            px += v * math.cos(theta) * dt
            py += v * math.sin(theta) * dt
            theta = _wrap_angle_nb(theta + omega * dt)
        return U

    @njit(cache=True)
    def nominal_controls_to_goal_nb(x0, goal, horizon, dt, v_min, v_max, omega_min, omega_max):
        U = np.zeros((horizon, 2), dtype=np.float64)
        px = x0[0]
        py = x0[1]
        theta = x0[2]
        for t in range(horizon):
            dx = goal[0] - px
            dy = goal[1] - py
            dist = math.sqrt(dx * dx + dy * dy)
            desired_heading = math.atan2(dy, dx)
            err = _wrap_angle_nb(desired_heading - theta)
            forward = math.cos(err)
            if forward < 0.0:
                forward = 0.0
            heading_scale = forward * forward
            v = 0.2 + 2.2 * dist * heading_scale
            if v < v_min:
                v = v_min
            elif v > v_max:
                v = v_max
            omega = 3.0 * err
            if omega < omega_min:
                omega = omega_min
            elif omega > omega_max:
                omega = omega_max
            U[t, 0] = v
            U[t, 1] = omega
            px += v * math.cos(theta) * dt
            py += v * math.sin(theta) * dt
            theta = _wrap_angle_nb(theta + omega * dt)
        return U

    @njit(cache=True)
    def temporal_smooth_noise_nb(noise, alpha):
        one_minus_alpha = 1.0 - alpha
        for n in range(noise.shape[0]):
            for t in range(1, noise.shape[1]):
                noise[n, t, 0] = alpha * noise[n, t - 1, 0] + one_minus_alpha * noise[n, t, 0]
                noise[n, t, 1] = alpha * noise[n, t - 1, 1] + one_minus_alpha * noise[n, t, 1]
        return noise

    @njit(cache=True)
    def standard_mppi_costs_batch_nb(X, U, circle_centers, circle_radii, goal, horizon, robot_radius, w_goal, w_obstacle, w_control, w_control_smooth):
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
                cost += w_goal / H * (gx * gx + gy * gy)
                for j in range(M):
                    dx = px - circle_centers[j, 0]
                    dy = py - circle_centers[j, 1]
                    d = math.sqrt(dx * dx + dy * dy) - circle_radii[j]
                    margin = robot_radius
                    sp = _softplus_scalar_nb(8.0 * (margin - d))
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
                smooth_cost += dv * dv + 0.2 * dom * dom
            cost += w_control_smooth * smooth_cost
            costs[n] = cost
        return costs

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
                x_cross = xi + (py - yi) * (xj - xi) / (yj - yi + 1e-18)
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
        best = 1e+18
        S = states.shape[0]
        M = poly_lengths.shape[0]
        for s in range(S):
            px = states[s, 0]
            py = states[s, 1]
            for m in range(M):
                n = poly_lengths[m]
                poly = polys_padded[m]
                min_d = 1e+18
                for i in range(n):
                    j = i + 1
                    if j == n:
                        j = 0
                    d = point_segment_dist_nb(px, py, poly[i, 0], poly[i, 1], poly[j, 0], poly[j, 1])
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
    def sensitivity_projected_covariances_nb(x0, nominal_controls, position_covariances, lookahead_steps, fd_v, fd_omega, pseudoinverse_damping, covariance_jitter, dt, v_min, v_max, omega_min, omega_max):
        """Project planar trajectory covariance into unicycle control space."""
        horizon = nominal_controls.shape[0]
        nominal_states = rollout_unicycle_single_nb(x0, nominal_controls, dt)
        projected = np.zeros((horizon, 2, 2), dtype=np.float64)
        damping_sq = pseudoinverse_damping * pseudoinverse_damping
        for t in range(horizon):
            interval = min(max(1, int(lookahead_steps)), horizon - t)
            jacobian = np.zeros((2, 2), dtype=np.float64)
            for control_index in range(2):
                delta = fd_v if control_index == 0 else fd_omega
                lower = v_min if control_index == 0 else omega_min
                upper = v_max if control_index == 0 else omega_max
                center = nominal_controls[t, control_index]
                plus_value = min(center + delta, upper)
                minus_value = max(center - delta, lower)
                denominator = plus_value - minus_value
                if denominator <= 1e-12:
                    continue
                plus_state = nominal_states[t].copy()
                minus_state = nominal_states[t].copy()
                for k in range(interval):
                    v_plus = nominal_controls[t + k, 0]
                    omega_plus = nominal_controls[t + k, 1]
                    v_minus = v_plus
                    omega_minus = omega_plus
                    if k == 0:
                        if control_index == 0:
                            v_plus = plus_value
                            v_minus = minus_value
                        else:
                            omega_plus = plus_value
                            omega_minus = minus_value
                    plus_theta = plus_state[2]
                    plus_state[0] += v_plus * math.cos(plus_theta) * dt
                    plus_state[1] += v_plus * math.sin(plus_theta) * dt
                    plus_state[2] = _wrap_angle_nb(plus_theta + omega_plus * dt)
                    minus_theta = minus_state[2]
                    minus_state[0] += v_minus * math.cos(minus_theta) * dt
                    minus_state[1] += v_minus * math.sin(minus_theta) * dt
                    minus_state[2] = _wrap_angle_nb(minus_theta + omega_minus * dt)
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
else:
    localize_prior_horizon_nb = None
    apply_gaussian_prior_noise_nb = None
    rollout_unicycle_batch_nb = None
    rollout_unicycle_single_nb = None
    nominal_controls_to_track_path_nb = None
    nominal_controls_to_goal_nb = None
    temporal_smooth_noise_nb = None
    standard_mppi_costs_batch_nb = None
    min_clearance_nb = None
    sensitivity_projected_covariances_nb = None

def obstacle_circles_to_arrays(obstacle_circles: List[Tuple[Array, float]]) -> Tuple[Array, Array]:
    if not obstacle_circles:
        return (np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64))
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    return (centers, radii)

def apply_terminal_goal_approach(x_current: Array, u: Array, goal: Array, goal_tolerance: float, cfg: MPPIConfig) -> Array:
    cmd = np.asarray(u, dtype=np.float64).copy()
    dx = float(goal[0] - x_current[0])
    dy = float(goal[1] - x_current[1])
    distance = math.hypot(dx, dy)
    radius = max(float(cfg.terminal_slowdown_radius), goal_tolerance + 1e-06)
    if distance >= radius:
        return cmd
    desired_heading = math.atan2(dy, dx)
    heading_error = wrap_angle(desired_heading - float(x_current[2]))
    remaining = max(0.0, distance)
    heading_scale = max(0.0, math.cos(heading_error)) ** 2
    terminal_v = min(float(cfg.terminal_max_speed), float(cfg.terminal_distance_gain) * remaining) * heading_scale
    heading_fade = float(np.clip((distance - goal_tolerance) / max(float(cfg.terminal_heading_deadzone), 1e-06), 0.0, 1.0))
    terminal_omega = heading_fade * float(np.clip(float(cfg.terminal_heading_gain) * heading_error, cfg.omega_min, cfg.omega_max))
    normalized = np.clip((radius - distance) / max(radius - goal_tolerance, 1e-06), 0.0, 1.0)
    blend = float(normalized ** max(float(cfg.terminal_blend_power), 1e-06))
    cmd[0] = (1.0 - blend) * cmd[0] + blend * terminal_v
    cmd[1] = (1.0 - blend) * cmd[1] + blend * terminal_omega
    cmd[0] = max(0.0, cmd[0])
    return cmd

def apply_smooth_safe_control(x_current: Array, u: Array, previous_control: Optional[Array], obstacle_circles: List[Tuple[Array, float]], cfg: MPPIConfig) -> Array:
    cmd = np.asarray(u, dtype=np.float64).copy()
    if previous_control is not None:
        dv = float(np.clip(cmd[0] - previous_control[0], -cfg.max_delta_v, cfg.max_delta_v))
        domega = float(np.clip(cmd[1] - previous_control[1], -cfg.max_delta_omega, cfg.max_delta_omega))
        cmd[0] = previous_control[0] + dv
        cmd[1] = previous_control[1] + domega
    cmd[0] = np.clip(cmd[0], cfg.v_min, cfg.v_max)
    cmd[1] = np.clip(cmd[1], cfg.omega_min, cfg.omega_max)
    if cfg.enforce_one_step_safety and obstacle_circles:
        x_next = unicycle_step(x_current, cmd, cfg.dt)
        centers, radii = obstacle_circles_to_arrays(obstacle_circles)
        current_clearance = float(np.min(np.linalg.norm(x_current[None, :2] - centers, axis=1) - radii - cfg.robot_radius))
        next_clearance = float(np.min(np.linalg.norm(x_next[None, :2] - centers, axis=1) - radii - cfg.robot_radius))
        moving_deeper = next_clearance < current_clearance - 0.0001
        below_required_clearance = next_clearance < cfg.one_step_safety_clearance
        if below_required_clearance and moving_deeper:
            cmd[0] = 0.0
    return cmd

def path_min_clearance_to_circles(path: Array, obstacle_circles: List[Tuple[Array, float]], robot_radius: float, substeps: int=2) -> float:
    p = np.asarray(path, dtype=np.float64)
    if len(p) == 0 or not obstacle_circles:
        return 1e309
    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    samples = [p]
    if len(p) > 1:
        for q in range(1, max(0, int(substeps)) + 1):
            alpha = q / float(max(0, int(substeps)) + 1)
            samples.append(p[:-1] + alpha * (p[1:] - p[:-1]))
    points = np.vstack(samples)
    clearance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2) - radii[None, :] - float(robot_radius)
    return float(np.min(clearance))

def unblocked_mode_indices(local_modes: Sequence[MPPIHomotopyMode], obstacle_circles: List[Tuple[Array, float]], cfg: MPPIConfig) -> Tuple[List[int], Array]:
    if not cfg.suppress_blocked_modes or len(local_modes) <= 1:
        return (list(range(len(local_modes))), np.full(len(local_modes), np.nan, dtype=np.float64))
    clearances = np.asarray([path_min_clearance_to_circles(mode.mean_path, obstacle_circles, cfg.robot_radius, substeps=cfg.mode_blocking_substeps) for mode in local_modes], dtype=np.float64)
    usable = np.where(clearances >= cfg.mode_blocking_clearance)[0].tolist()
    if not usable:
        usable = [int(np.argmax(clearances))]
    return (usable, clearances)

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

def localize_mode_for_state_with_index(mode: MPPIHomotopyMode, x_current: Array, H: int, previous_idx: Optional[int]=None, max_advance: Optional[int]=None) -> Tuple[MPPIHomotopyMode, int]:
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
    args = (np.asarray(mode.mean_path, dtype=np.float64), np.asarray(mode.cov_blocks, dtype=np.float64), np.asarray(mode.arc_length, dtype=np.float64), np.asarray(mode.gaussian_variance, dtype=np.float64), idx, H)
    if localize_prior_horizon_nb is not None:
        local_mu, local_cov, local_gaussian = localize_prior_horizon_nb(*args)
    else:
        source_s = np.linspace(mode.arc_length[idx], mode.arc_length[-1], H)
        local_mu = np.column_stack([np.interp(source_s, mode.arc_length, mode.mean_path[:, 0]), np.interp(source_s, mode.arc_length, mode.mean_path[:, 1])])
        local_cov = np.empty((H, 2, 2), dtype=np.float64)
        for row in range(2):
            for col in range(2):
                local_cov[:, row, col] = np.interp(source_s, mode.arc_length, mode.cov_blocks[:, row, col])
        local_gaussian = np.interp(source_s, mode.arc_length, mode.gaussian_variance)
    local_arc = np.zeros(H, dtype=np.float64)
    if H > 1:
        local_arc[1:] = np.cumsum(np.linalg.norm(np.diff(local_mu, axis=0), axis=1))
    return (MPPIHomotopyMode(signature=mode.signature, probability=mode.probability, mean_path=local_mu, cov_blocks=local_cov, sample_paths=None, arc_length=local_arc, gaussian_variance=local_gaussian), idx)

def localize_mode_for_state(mode: MPPIHomotopyMode, x_current: Array, H: int) -> MPPIHomotopyMode:
    local_mode, _ = localize_mode_for_state_with_index(mode=mode, x_current=x_current, H=H, previous_idx=None, max_advance=None)
    return local_mode

def localize_path_for_state_with_index(path: Array, x_current: Array, H: int, previous_idx: Optional[int]=None, max_advance: Optional[int]=None) -> Tuple[Array, int]:
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
    return (resample_path(tail, H), idx)

def localize_path_for_state(path: Array, x_current: Array, H: int) -> Array:
    local_path, _ = localize_path_for_state_with_index(path=path, x_current=x_current, H=H, previous_idx=None, max_advance=None)
    return local_path

def nominal_controls_to_track_path(x0: Array, ref: Array, cfg) -> Array:
    if nominal_controls_to_track_path_nb is not None:
        return nominal_controls_to_track_path_nb(np.asarray(x0, dtype=np.float64), np.asarray(ref, dtype=np.float64), int(cfg.horizon), float(cfg.dt), float(cfg.v_min), float(cfg.v_max), float(cfg.omega_min), float(cfg.omega_max))
    H = cfg.horizon
    U = np.zeros((H, 2), dtype=np.float64)
    x = x0.copy()
    for t in range(H):
        target = ref[min(t + 3, len(ref) - 1)]
        delta = target - x[:2]
        dist = float(np.linalg.norm(delta))
        desired_heading = math.atan2(delta[1], delta[0])
        err = wrap_angle(desired_heading - x[2])
        heading_scale = max(0.0, math.cos(err)) ** 2
        v = np.clip(0.2 + 2.4 * dist * heading_scale, 0.0, cfg.v_max)
        omega = np.clip(3.2 * err, cfg.omega_min, cfg.omega_max)
        U[t] = [v, omega]
        x = unicycle_step(x, U[t], cfg.dt)
    return U

def nominal_controls_to_goal(x0: Array, goal: Array, cfg) -> Array:
    if nominal_controls_to_goal_nb is not None:
        return nominal_controls_to_goal_nb(np.asarray(x0, dtype=np.float64), np.asarray(goal, dtype=np.float64), int(cfg.horizon), float(cfg.dt), float(cfg.v_min), float(cfg.v_max), float(cfg.omega_min), float(cfg.omega_max))
    H = cfg.horizon
    U = np.zeros((H, 2), dtype=np.float64)
    x = x0.copy()
    for t in range(H):
        delta = goal - x[:2]
        dist = float(np.linalg.norm(delta))
        desired_heading = math.atan2(delta[1], delta[0])
        err = wrap_angle(desired_heading - x[2])
        heading_scale = max(0.0, math.cos(err)) ** 2
        v = np.clip(0.2 + 2.2 * dist * heading_scale, 0.0, cfg.v_max)
        omega = np.clip(3.0 * err, cfg.omega_min, cfg.omega_max)
        U[t] = [v, omega]
        x = unicycle_step(x, U[t], cfg.dt)
    return U

def build_empirical_nominal_bank(x_current: Array, global_mode: MPPIHomotopyMode, mean_nominal: Array, cfg: MPPIConfig, rng: np.random.Generator, previous_idx: Optional[int]=None) -> List[Array]:
    bank = [mean_nominal]
    if not global_mode.sample_paths:
        return bank
    n = min(cfg.max_empirical_nominals_per_mode, len(global_mode.sample_paths))
    ids = rng.choice(len(global_mode.sample_paths), size=n, replace=False)
    for sid in ids:
        local_sample_path, _ = localize_path_for_state_with_index(global_mode.sample_paths[int(sid)], x_current, cfg.horizon, previous_idx=previous_idx if cfg.use_monotonic_reference_progress else None, max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None)
        bank.append(nominal_controls_to_track_path(x_current, local_sample_path, cfg))
    return bank

def standard_mppi_costs_batch(X: Array, U: Array, obstacle_circles: List[Tuple[Array, float]], goal: Array, cfg: MPPIConfig) -> Array:
    if standard_mppi_costs_batch_nb is not None:
        centers, radii = obstacle_circles_to_arrays(obstacle_circles)
        costs = standard_mppi_costs_batch_nb(np.asarray(X, dtype=np.float64), np.asarray(U, dtype=np.float64), centers, radii, np.asarray(goal, dtype=np.float64), int(cfg.horizon), float(cfg.robot_radius), float(cfg.w_goal), float(cfg.w_obstacle), float(cfg.w_control), float(cfg.w_control_smooth))
        return costs
    N, H, _ = U.shape
    costs = np.zeros(N, dtype=np.float64)
    P = X[:, 1:H + 1, :2]
    for t in range(H):
        p = P[:, t, :]
        goal_error = p - goal[None, :]
        costs += cfg.w_goal / H * np.sum(goal_error ** 2, axis=1)
        for center, radius in obstacle_circles:
            d = np.linalg.norm(p - center[None, :], axis=1) - radius
            margin = cfg.robot_radius
            costs += cfg.w_obstacle * softplus(8.0 * (margin - d)) ** 2
    costs += cfg.w_control * np.sum(U[:, :, 0] ** 2 + 0.15 * U[:, :, 1] ** 2, axis=1)
    dU = np.diff(U, axis=1)
    costs += cfg.w_control_smooth * np.sum(dU[:, :, 0] ** 2 + 0.2 * dU[:, :, 1] ** 2, axis=1)
    return costs
REP_GAUSSIAN = 1
REP_CORRIDOR = 2
REP_CONTROL_BANK = 3
REP_SENSITIVITY_PROJECTED_GAUSSIAN = 4

def stable_representation_costs(X: Array, U: Array, obstacle_circles: List[Tuple[Array, float]], goal: Array, cfg: MPPIConfig) -> Array:
    return standard_mppi_costs_batch(X, U, obstacle_circles, goal, cfg)

def softmin_score(costs: Array, cfg: MPPIConfig) -> float:
    values = np.asarray(costs, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return 1e309
    finite_values = values[finite]
    rho = float(np.min(finite_values))
    z = np.exp(-(finite_values - rho) / cfg.lambda_temperature)
    return float(rho - cfg.lambda_temperature * math.log(np.sum(z) / len(values) + 1e-12))

def build_nominal_bank_for_mode(x_current: Array, local_mode: MPPIHomotopyMode, global_mode: MPPIHomotopyMode, goal: Array, cfg: MPPIConfig, rng: np.random.Generator, *, use_empirical_init: bool, use_mean_nominal: bool, previous_idx: Optional[int]=None) -> List[Array]:
    goal_nominal = nominal_controls_to_goal(x_current, goal, cfg)
    if use_mean_nominal:
        mean_nominal = nominal_controls_to_track_path(x_current, local_mode.mean_path, cfg)
    else:
        mean_nominal = goal_nominal
    if use_empirical_init:
        bank = build_empirical_nominal_bank(x_current=x_current, global_mode=global_mode, mean_nominal=mean_nominal, cfg=cfg, rng=rng, previous_idx=previous_idx)
    else:
        bank = [mean_nominal]
    if not any((np.allclose(candidate, goal_nominal) for candidate in bank)):
        bank.append(goal_nominal)
    return bank

def enforce_forward_curve_proposals(U: Array, cfg: MPPIConfig) -> Array:
    U = np.asarray(U, dtype=np.float64)
    if U.size == 0:
        return U
    U[:, :, 0] = np.clip(U[:, :, 0], cfg.v_min, cfg.v_max)
    U[:, :, 1] = np.clip(U[:, :, 1], cfg.omega_min, cfg.omega_max)
    return U

def sample_controls_from_nominal_bank(nominal_bank: List[Array], n: int, cfg: MPPIConfig, rng: np.random.Generator, *, prefer_empirical: bool=True) -> Array:
    if len(nominal_bank) == 1:
        bank_ids = np.zeros(n, dtype=np.int64)
    else:
        probs = np.ones(len(nominal_bank), dtype=np.float64)
        if prefer_empirical:
            probs[0] = max(1e-06, 1.0 - cfg.swarm_init_probability)
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
        U[cursor] = nominal_bank[j] + float(cfg.low_noise_proposal_scale) * noise[cursor]
        cursor += 1
    return enforce_forward_curve_proposals(U, cfg)

def sample_exact_control_bank(x_current: Array, global_mode: MPPIHomotopyMode, fallback_nominal: Array, n: int, cfg: MPPIConfig, rng: np.random.Generator, previous_idx: Optional[int]=None) -> Array:
    """Use inverse controls of empirical trajectories without added control noise."""
    if n <= 0:
        return np.zeros((0, cfg.horizon, 2), dtype=np.float64)
    candidates: List[Array] = []
    if global_mode.sample_paths:
        order = rng.permutation(len(global_mode.sample_paths))
        for sid in order:
            local_path, _ = localize_path_for_state_with_index(global_mode.sample_paths[int(sid)], x_current, cfg.horizon, previous_idx=previous_idx if cfg.use_monotonic_reference_progress else None, max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None)
            candidates.append(nominal_controls_to_track_path(x_current, local_path, cfg))
    if not candidates:
        candidates = [np.asarray(fallback_nominal, dtype=np.float64).copy()]
    U = np.stack([candidates[i % len(candidates)].copy() for i in range(n)], axis=0)
    return enforce_forward_curve_proposals(U, cfg)

def sample_gaussian_controls(x_current: Array, local_mode: MPPIHomotopyMode, n: int, cfg: MPPIConfig, rng: np.random.Generator) -> Array:
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
        noise = apply_gaussian_prior_noise_nb(noise, variance, float(cfg.sigma_floor), float(cfg.gaussian_covariance_scale))
    else:
        floor_var = float(cfg.sigma_floor) ** 2
        scale = float(cfg.gaussian_covariance_scale) * np.sqrt(np.maximum(variance, floor_var)) / max(float(cfg.sigma_floor), 1e-09)
        noise *= scale[None, :, None]
    U = nominal[None, :, :] + noise
    U[0] = nominal
    return enforce_forward_curve_proposals(U, cfg)

def sensitivity_projected_control_covariances(x0: Array, nominal_controls: Array, position_covariances: Array, cfg: MPPIConfig) -> Array:
    """Compute Eq. (26) for the unicycle controls (v, omega)."""
    nominal = np.asarray(nominal_controls, dtype=np.float64)
    covariances = np.asarray(position_covariances, dtype=np.float64)
    horizon = int(nominal.shape[0])
    if nominal.shape != (horizon, 2):
        raise ValueError(f'nominal_controls must have shape (H,2), got {nominal.shape}')
    if covariances.shape != (horizon, 2, 2):
        raise ValueError(f'position_covariances must have shape (H,2,2), got {covariances.shape}')
    if sensitivity_projected_covariances_nb is not None:
        return sensitivity_projected_covariances_nb(np.asarray(x0, dtype=np.float64), nominal, covariances, int(cfg.spg_lookahead_steps), float(cfg.spg_fd_accel), float(cfg.spg_fd_steering_rate), float(cfg.spg_pseudoinverse_damping), float(cfg.spg_covariance_jitter), float(cfg.dt), float(cfg.v_min), float(cfg.v_max), float(cfg.omega_min), float(cfg.omega_max))
    nominal_states = rollout_unicycle(x0, nominal, cfg.dt)
    projected = np.zeros((horizon, 2, 2), dtype=np.float64)
    damping = float(cfg.spg_pseudoinverse_damping)
    covariance_scale = 1.0
    eye2 = np.eye(2, dtype=np.float64)
    for t in range(horizon):
        interval = min(max(1, int(cfg.spg_lookahead_steps)), horizon - t)
        jacobian = np.zeros((2, 2), dtype=np.float64)
        for control_index, delta in enumerate((float(cfg.spg_fd_accel), float(cfg.spg_fd_steering_rate))):
            lower = cfg.v_min if control_index == 0 else cfg.omega_min
            upper = cfg.v_max if control_index == 0 else cfg.omega_max
            plus_value = float(np.clip(nominal[t, control_index] + delta, lower, upper))
            minus_value = float(np.clip(nominal[t, control_index] - delta, lower, upper))
            denominator = plus_value - minus_value
            if denominator <= 1e-12:
                continue
            plus_controls = nominal[t:t + interval].copy()
            minus_controls = plus_controls.copy()
            plus_controls[0, control_index] = plus_value
            minus_controls[0, control_index] = minus_value
            plus_position = rollout_unicycle(nominal_states[t], plus_controls, cfg.dt)[-1, :2]
            minus_position = rollout_unicycle(nominal_states[t], minus_controls, cfg.dt)[-1, :2]
            jacobian[:, control_index] = (plus_position - minus_position) / denominator
        regularized = jacobian @ jacobian.T + (damping * damping + 1e-18) * eye2
        pseudoinverse = np.linalg.solve(regularized, jacobian).T
        position_covariance = 0.5 * (covariances[t] + covariances[t].T)
        control_covariance = pseudoinverse @ position_covariance @ pseudoinverse.T
        projected[t] = covariance_scale * 0.5 * (control_covariance + control_covariance.T) + float(cfg.spg_covariance_jitter) * eye2
    return projected

def sample_sensitivity_projected_gaussian_controls(x_current: Array, local_mode: MPPIHomotopyMode, n: int, cfg: MPPIConfig, rng: np.random.Generator) -> Array:
    """Sample the unicycle SPG proposal without baseline control covariance."""
    horizon = int(cfg.horizon)
    if n <= 0:
        return np.zeros((0, horizon, 2), dtype=np.float64)
    mean_path = np.asarray(local_mode.mean_path, dtype=np.float64)
    nominal = nominal_controls_to_track_path(x_current, mean_path, cfg)
    projected_covariances = sensitivity_projected_control_covariances(x_current, nominal, np.asarray(local_mode.cov_blocks, dtype=np.float64), cfg)
    standard_noise = make_temporally_correlated_noise(n, horizon, cfg, rng, noise_v=1.0, noise_omega=1.0)
    noise = np.zeros_like(standard_noise)
    for t in range(horizon):
        covariance = 0.5 * (projected_covariances[t] + projected_covariances[t].T)
        if not np.all(np.isfinite(covariance)):
            covariance = np.zeros((2, 2), dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        covariance_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        noise[:, t, :] = standard_noise[:, t, :] @ covariance_sqrt.T
    controls = nominal[None, :, :] + noise
    controls[0] = nominal
    return enforce_forward_curve_proposals(controls, cfg)

def nearby_mode_indices(global_modes: Sequence[MPPIHomotopyMode], x_current: Array, cfg: MPPIConfig, obstacle_circles: Optional[List[Tuple[Array, float]]]=None) -> List[int]:
    if not global_modes:
        return []
    p = np.asarray(x_current[:2], dtype=np.float64)
    distances = np.asarray([float(np.min(np.linalg.norm(np.asarray(mode.mean_path) - p[None, :], axis=1))) for mode in global_modes], dtype=np.float64)
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
            clearance = path_min_clearance_to_circles(local_mode.mean_path, obstacle_circles, cfg.robot_radius, substeps=0)
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

def stable_swarm_mppi_step(x_current: Array, global_modes: List[MPPIHomotopyMode], obstacles: Sequence, goal: Array, cfg: MPPIConfig, rng: np.random.Generator, *, rep_type: int, use_empirical_init: bool, use_mean_nominal: bool, progress_by_mode: Optional[Dict[str, int]], obstacle_circles: Optional[List[Tuple[Array, float]]]=None, record_optimal_traj: bool=None) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    if rep_type not in {REP_GAUSSIAN, REP_CORRIDOR, REP_CONTROL_BANK, REP_SENSITIVITY_PROJECTED_GAUSSIAN}:
        raise ValueError(f'Unsupported pooled proposal representation: {rep_type}')
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
        local_mode, index = localize_mode_for_state_with_index(mode, x_current, cfg.horizon, previous_idx=previous if cfg.use_monotonic_reference_progress else None, max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None)
        local_modes.append(local_mode)
        new_progress_by_mode[key] = index
    active_mode_indices, mode_clearances = unblocked_mode_indices(local_modes, obstacle_circles, cfg)
    total_budget = max(1, int(cfg.num_rollouts))
    active_mode_indices = active_mode_indices[:total_budget]
    active_local_modes = [local_modes[i] for i in active_mode_indices]
    active_global_modes = [candidate_global_modes[i] for i in active_mode_indices]
    counts = balanced_rollout_counts(total_budget, len(active_local_modes))
    mode_ids = np.concatenate([np.full(count, mode_index, dtype=np.int64) for mode_index, count in enumerate(counts)])
    all_costs = np.zeros(total_budget, dtype=np.float64)
    all_U = np.zeros((total_budget, cfg.horizon, 2), dtype=np.float64)
    best_cost = 1e309
    best_traj = None
    for mode_index, local_mode in enumerate(active_local_modes):
        ids = np.where(mode_ids == mode_index)[0]
        n = len(ids)
        if n == 0:
            continue
        global_mode = active_global_modes[mode_index]
        key = str(global_mode.signature)
        nominal_bank = build_nominal_bank_for_mode(x_current, local_mode, global_mode, goal, cfg, rng, use_empirical_init=use_empirical_init, use_mean_nominal=use_mean_nominal, previous_idx=progress_by_mode.get(key))
        if rep_type == REP_GAUSSIAN:
            U = sample_gaussian_controls(x_current, local_mode, n, cfg, rng)
        elif rep_type == REP_SENSITIVITY_PROJECTED_GAUSSIAN:
            U = sample_sensitivity_projected_gaussian_controls(x_current, local_mode, n, cfg, rng)
        elif rep_type == REP_CONTROL_BANK:
            U = sample_exact_control_bank(x_current, global_mode, nominal_bank[0], n, cfg, rng, previous_idx=progress_by_mode.get(key))
        else:
            mean_nominal = nominal_controls_to_track_path(x_current, local_mode.mean_path, cfg)
            U = sample_controls_from_nominal_bank([mean_nominal], n, cfg, rng, prefer_empirical=False)
        if rep_type != REP_CONTROL_BANK:
            U = ensure_direct_goal_prior(U, x_current, goal, cfg)
        X = rollout_unicycle_batch(x_current, U, cfg.dt)
        costs = stable_representation_costs(X, U, obstacle_circles, goal, cfg)
        costs = reject_colliding_rollouts(costs, X, obstacle_circles, cfg)
        all_costs[ids] = costs
        all_U[ids] = U
        if record_optimal_traj:
            local_best = int(np.argmin(costs))
            if float(costs[local_best]) < best_cost:
                best_cost = float(costs[local_best])
                best_traj = np.asarray(X[local_best], dtype=np.float64).copy()
    planned_sequence = mppi_weighted_control_sequence(all_costs, all_U, cfg)
    info = {'cost_min': float(np.min(all_costs)), 'cost_mean': float(np.mean(all_costs)), 'soft_value': float(softmin_score(all_costs, cfg)), 'rep_type': int(rep_type), 'mode_selection': False, 'selected_mode_index': None, 'rollout_budget_total': total_budget, 'rollouts_by_mode': counts, 'active_mode_count': int(len(active_mode_indices)), 'suppressed_mode_count': int(len(global_modes) - len(active_mode_indices)), 'nearby_mode_count': int(len(candidate_global_modes)), 'mode_clearances': mode_clearances.tolist(), 'optimal_traj': best_traj, 'planned_control_sequence': planned_sequence}
    return (planned_sequence[0].copy(), info, new_progress_by_mode)

def mode_selecting_stable_mppi_step(x_current: Array, global_modes: List[MPPIHomotopyMode], obstacles: Sequence, goal: Array, cfg: MPPIConfig, rng: np.random.Generator, *, rep_type: int, progress_by_mode: Optional[Dict[str, int]]=None, obstacle_circles: Optional[List[Tuple[Array, float]]]=None, record_optimal_traj: bool=True) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    if rep_type not in {REP_GAUSSIAN, REP_CORRIDOR}:
        raise ValueError('Mode-selecting MPPI supports only Gaussian or corridor proposals.')
    progress_by_mode = {} if progress_by_mode is None else dict(progress_by_mode)
    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)
    nearby_indices = nearby_mode_indices(global_modes, x_current, cfg, obstacle_circles)
    top_k = min(max(1, int(cfg.mode_select_top_k)), len(nearby_indices))
    candidate_indices = nearby_indices[:top_k]
    records = []
    new_progress_by_mode = dict(progress_by_mode)
    for original_index in candidate_indices:
        global_mode = global_modes[original_index]
        key = str(global_mode.signature)
        previous = progress_by_mode.get(key)
        local_mode, index = localize_mode_for_state_with_index(global_mode, x_current, cfg.horizon, previous_idx=previous if cfg.use_monotonic_reference_progress else None, max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None)
        new_progress_by_mode[key] = index
        records.append({'original_mid': int(original_index), 'global_mode': global_mode, 'local_mode': local_mode})
    active_positions, mode_clearances = unblocked_mode_indices([record['local_mode'] for record in records], obstacle_circles, cfg)
    active_records = [records[i] for i in active_positions]
    configured_per_mode = int(cfg.mode_select_rollouts_per_mode)
    rollouts_per_mode = configured_per_mode if configured_per_mode > 0 else max(1, int(cfg.num_rollouts))
    completed = []
    for record in active_records:
        global_mode = record['global_mode']
        local_mode = record['local_mode']
        if rep_type == REP_GAUSSIAN:
            U = sample_gaussian_controls(x_current, local_mode, rollouts_per_mode, cfg, rng)
        else:
            mean_nominal = nominal_controls_to_track_path(x_current, local_mode.mean_path, cfg)
            U = sample_controls_from_nominal_bank([mean_nominal], rollouts_per_mode, cfg, rng, prefer_empirical=False)
        X = rollout_unicycle_batch(x_current, U, cfg.dt)
        costs = stable_representation_costs(X, U, obstacle_circles, goal, cfg)
        collision_mask = rollout_collision_mask(X, obstacle_circles, cfg)
        feasible_count = int(np.count_nonzero(~collision_mask))
        costs = reject_colliding_rollouts(costs, X, obstacle_circles, cfg)
        planned_sequence = mppi_weighted_control_sequence(costs, U, cfg)
        completed.append({'score': float(softmin_score(costs, cfg)), 'mode_index': int(record['original_mid']), 'signature': str(global_mode.signature), 'probability': float(global_mode.probability), 'feasible_count': feasible_count, 'cost_min': float(np.min(costs)), 'cost_mean': float(np.mean(costs)), 'optimal_traj': np.asarray(X[int(np.argmin(costs))], dtype=np.float64).copy() if record_optimal_traj else None, 'planned_control_sequence': planned_sequence})
    if not completed:
        raise RuntimeError('No homotopy mode was available for mode-selecting MPPI.')
    feasible = [record for record in completed if record['feasible_count'] > 0]
    best = min(feasible if feasible else completed, key=lambda record: record['score'])
    counts = [rollouts_per_mode] * len(completed)
    total_budget = rollouts_per_mode * len(completed)
    info = {'cost_min': best['cost_min'], 'cost_mean': best['cost_mean'], 'soft_value': best['score'], 'selected_mode_index': best['mode_index'], 'selected_mode_signature': best['signature'], 'selected_mode_probability': best['probability'], 'rep_type': int(rep_type), 'mode_selection': True, 'rollout_budget_per_mode': rollouts_per_mode, 'rollout_budget_total': total_budget, 'rollouts_by_mode': counts, 'active_mode_count': int(len(completed)), 'suppressed_mode_count': int(len(records) - len(completed)), 'mode_clearances': mode_clearances.tolist(), 'optimal_traj': best['optimal_traj'], 'planned_control_sequence': best['planned_control_sequence']}
    return (best['planned_control_sequence'][0].copy(), info, new_progress_by_mode)

def make_temporally_correlated_noise(n: int, H: int, cfg: MPPIConfig, rng: np.random.Generator, *, noise_v: Optional[float]=None, noise_omega: Optional[float]=None) -> Array:
    noise_scale = np.array([cfg.noise_v if noise_v is None else float(noise_v), cfg.noise_omega if noise_omega is None else float(noise_omega)], dtype=np.float64)
    noise = rng.normal(size=(n, H, 2)) * noise_scale[None, None, :]
    alpha = float(cfg.temporal_noise_smoothing)
    if temporal_smooth_noise_nb is not None:
        return temporal_smooth_noise_nb(noise, alpha)
    for t in range(1, H):
        noise[:, t, :] = alpha * noise[:, t - 1, :] + (1.0 - alpha) * noise[:, t, :]
    return noise

def rollout_collision_mask(X: Array, obstacle_circles: Sequence[Tuple[Array, float]], cfg: MPPIConfig) -> Array:
    if not obstacle_circles or X.shape[0] == 0:
        return np.zeros(X.shape[0], dtype=bool)
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    points = np.asarray(X[:, 1:, :2], dtype=np.float64)
    delta = points[:, :, None, :] - centers[None, None, :, :]
    clearance = np.linalg.norm(delta, axis=-1) - radii[None, None, :] - float(cfg.robot_radius)
    return np.any(clearance < float(cfg.hard_collision_clearance), axis=(1, 2))

def reject_colliding_rollouts(costs: Array, X: Array, obstacle_circles: Sequence[Tuple[Array, float]], cfg: MPPIConfig) -> Array:
    if not obstacle_circles or X.shape[0] == 0:
        return costs
    colliding = rollout_collision_mask(X, obstacle_circles, cfg)
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

def mppi_weighted_control_sequence(costs: Array, U: Array, cfg: MPPIConfig) -> Array:
    weights = mppi_weights(costs, cfg)
    sequence = np.tensordot(weights, U, axes=(0, 0))
    sequence[:, 0] = np.clip(sequence[:, 0], cfg.v_min, cfg.v_max)
    sequence[:, 1] = np.clip(sequence[:, 1], cfg.omega_min, cfg.omega_max)
    return np.asarray(sequence, dtype=np.float64)

def update_display_trajectory(info: Dict[str, object], x_current: Array, executed_u: Array, cfg: MPPIConfig) -> None:
    sequence = info.get('planned_control_sequence')
    if sequence is None:
        return
    display_u = np.asarray(sequence, dtype=np.float64).copy()
    if display_u.ndim != 2 or display_u.shape[1] != 2 or len(display_u) == 0:
        return
    display_u[0] = np.asarray(executed_u, dtype=np.float64)
    info['optimal_traj'] = rollout_unicycle(x_current, display_u, cfg.dt)

def best_output_trajectory_from_costs(costs: Array, X: Array) -> Array:
    best_idx = int(np.argmin(costs))
    return np.asarray(X[best_idx], dtype=np.float64).copy()

def standard_mppi_step(x_current: Array, obstacles: Sequence, goal: Array, cfg: MPPIConfig, rng: np.random.Generator, *, obstacle_circles: Optional[List[Tuple[Array, float]]]=None, record_optimal_traj: bool=True) -> Tuple[Array, Dict[str, object]]:
    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)
    U_nom = nominal_controls_to_goal(x_current, goal, cfg)
    U = np.repeat(U_nom[None, :, :], cfg.num_rollouts, axis=0)
    noise = make_temporally_correlated_noise(cfg.num_rollouts, cfg.horizon, cfg, rng)
    U += noise
    k = min(max(1, int(cfg.low_noise_proposal_count)), cfg.num_rollouts)
    U[:k] = U_nom[None, :, :] + float(cfg.low_noise_proposal_scale) * noise[:k]
    U[0] = U_nom
    U = enforce_forward_curve_proposals(U, cfg)
    X = rollout_unicycle_batch(x_current, U, cfg.dt)
    costs = standard_mppi_costs_batch(X, U, obstacle_circles, goal, cfg)
    costs = reject_colliding_rollouts(costs, X, obstacle_circles, cfg)
    planned_sequence = mppi_weighted_control_sequence(costs, U, cfg)
    u = planned_sequence[0].copy()
    return (u, {'cost_min': float(costs.min()), 'cost_mean': float(costs.mean()), 'optimal_traj': best_output_trajectory_from_costs(costs, X) if record_optimal_traj else None, 'planned_control_sequence': planned_sequence})

def build_default_scene() -> Scene:
    bounds_xy = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    obstacles = (PolyObstacle(round_obstacle(np.array([[3.0, 1.5], [5.2, 2.2], [4.7, 4.0], [2.8, 3.4]]), n_iters=4, n_points=32)), PolyObstacle(round_obstacle(np.array([[6.2, 6.0], [8.5, 6.3], [8.1, 8.4], [6.8, 8.9], [5.9, 7.4]]), n_iters=4, n_points=32)), PolyObstacle(round_obstacle(np.array([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]]), n_iters=4, n_points=32)), PolyObstacle(round_obstacle(np.array([[1.8, 4.2], [2.7, 4.0], [3.0, 4.8], [2.3, 5.3], [1.7, 4.9]]), n_iters=4, n_points=32)), PolyObstacle(round_obstacle(np.array([[4.6, 5.1], [5.4, 5.0], [5.8, 5.7], [5.0, 6.2], [4.4, 5.7]]), n_iters=4, n_points=32)), PolyObstacle(round_obstacle(np.array([[7.9, 3.0], [9.0, 3.2], [8.8, 4.2], [7.7, 4.0]]), n_iters=4, n_points=32)), PolyObstacle(round_obstacle(np.array([[5.7, 1.0], [6.6, 1.2], [6.4, 2.3], [5.6, 2.1]]), n_iters=4, n_points=32)))
    return Scene(scale=4.0, bounds_xy=bounds_xy, planner_bounds=((0.0, 10.0), (0.0, 10.0)), start=np.array([1.0, 1.0]), goal=np.array([9.0, 9.0]), obstacles=obstacles)

def run_swarm_planner(start, goal, obstacles, scale, bounds_xy, *, seed: int):
    segments = obstacles_to_segs(obstacles, scale=scale)
    with open('save/policy.pkl', 'rb') as policy_file:
        action = pickle.load(policy_file)['best_theta']
    graph_goals, graph_weights = build_full_graph(obstacles=obstacles, start=start, goal=goal, scale=scale, bounds=bounds_xy)
    planner = HomotopyAwareGenerativePlanner(env_cls=FishGoalEnv2D, action=action, obstacles=obstacles, segs=segments, scale=scale, boid_count=1200, max_steps=700, dt=0.5)
    return planner.sample(start_unscaled=start, goal_unscaled=goal, graph_goals=graph_goals, graph_W=graph_weights, seed=seed)

def initial_pose(start: Array, goal: Array) -> Array:
    direction = goal - start
    heading = math.atan2(direction[1], direction[0])
    return np.array([start[0], start[1], heading], dtype=np.float64)

def ensure_direct_goal_prior(U: Array, x_current: Array, goal: Array, cfg: MPPIConfig) -> Array:
    proposals = np.asarray(U, dtype=np.float64)
    if proposals.ndim != 3 or proposals.shape[0] == 0:
        return proposals
    proposals[-1] = nominal_controls_to_goal(x_current, goal, cfg)
    return enforce_forward_curve_proposals(proposals, cfg)

def obstacle_center(obs) -> Array:
    return _poly_vertices(obs).mean(axis=0)

def make_wall_between_points(p0: Array, p1: Array, width: float=0.35, extension: float=0.0):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L <= 1e-12:
        raise ValueError('Cannot create wall: endpoints are identical.')
    u = d / L
    n = np.array([-u[1], u[0]], dtype=np.float64)
    a = p0 - extension * u
    b = p1 + extension * u
    half = 0.5 * float(width)
    vertices = np.array([a + half * n, b + half * n, b - half * n, a - half * n], dtype=np.float64)
    return PolyObstacle(vertices)

def make_wall_blockers_between_centers(centers: Sequence[Array], pairs: Sequence[Tuple[int, int]], width: float=0.35, extension: float=0.15):
    fixed_centers = [np.asarray(center, dtype=np.float64).reshape(2).copy() for center in centers]
    blockers = []
    for i, j in pairs:
        if i == j:
            raise ValueError(f'Cannot create wall for degenerate center pair {(i, j)}.')
        if not (0 <= i < len(fixed_centers) and 0 <= j < len(fixed_centers)):
            raise IndexError(f'Center pair {(i, j)} is outside the valid index range [0, {len(fixed_centers) - 1}].')
        blockers.append(make_wall_between_points(fixed_centers[i], fixed_centers[j], width=width, extension=extension))
    return blockers

def as_blocker_list(blocker_or_blockers):
    if blocker_or_blockers is None:
        return []
    if isinstance(blocker_or_blockers, (list, tuple)):
        return list(blocker_or_blockers)
    return [blocker_or_blockers]

def spatial_progress_along_start_goal(x: Array, start: Array, goal: Array) -> float:
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    position = np.asarray(x[:2], dtype=np.float64)
    direction = goal - start
    denom = float(direction @ direction)
    if denom <= 1e-12:
        return 1.0
    return float(np.clip((position - start) @ direction / denom, 0.0, 1.0))

def active_obstacles_for_state(base_obstacles: Sequence, blocker, state_index: int, activation_step: Optional[int]):
    if activation_step is not None and state_index >= activation_step:
        return list(base_obstacles) + as_blocker_list(blocker)
    return list(base_obstacles)

def run_controller(variant: ControllerVariant, modes: list[MPPIHomotopyMode], base_obstacles: Sequence, blockers: Sequence, scene: Scene, *, seed: int, trigger_progress: Optional[float], blocker_active_from_start: bool, max_steps: int, cfg: MPPIConfig, record: bool=True) -> SimulationResult:
    rng = np.random.default_rng(seed)
    x = initial_pose(scene.start, scene.goal)
    states = [x.copy()]
    controls: list[Array] = []
    infos: list[dict[str, object]] = []
    obstacle_history: list[list[object]] = []
    previous_control: Optional[Array] = None
    reached_goal = bool(np.linalg.norm(x[:2] - scene.goal) <= cfg.goal_tolerance)
    progress_by_mode: dict[str, int] = {}
    blockers = list(blockers)
    activation_step: Optional[int] = 0 if blocker_active_from_start else None
    base_circles = obstacle_bounding_circles(base_obstacles)
    blocked_obstacles = list(base_obstacles) + blockers
    blocked_circles = obstacle_bounding_circles(blocked_obstacles) if blockers else base_circles
    started = time.perf_counter()
    for step in range(max_steps):
        if activation_step is None and blockers and (trigger_progress is not None) and (spatial_progress_along_start_goal(x, scene.start, scene.goal) >= trigger_progress):
            activation_step = step
        wall_active = activation_step is not None and step >= activation_step
        active_obstacles = blocked_obstacles if wall_active else list(base_obstacles)
        active_circles = blocked_circles if wall_active else base_circles
        if record:
            obstacle_history.append(list(active_obstacles))
        if variant == ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI:
            u, info, progress_by_mode = stable_swarm_mppi_step(x, modes, active_obstacles, scene.goal, cfg, rng, rep_type=REP_SENSITIVITY_PROJECTED_GAUSSIAN, use_empirical_init=False, use_mean_nominal=True, progress_by_mode=progress_by_mode, obstacle_circles=active_circles, record_optimal_traj=record)
        elif variant == ControllerVariant.GAUSSIAN_PRIOR_MPPI:
            u, info, progress_by_mode = stable_swarm_mppi_step(x, modes, active_obstacles, scene.goal, cfg, rng, rep_type=REP_GAUSSIAN, use_empirical_init=False, use_mean_nominal=True, progress_by_mode=progress_by_mode, obstacle_circles=active_circles, record_optimal_traj=record)
        elif variant == ControllerVariant.CORRIDOR_PRIOR_MPPI:
            u, info, progress_by_mode = stable_swarm_mppi_step(x, modes, active_obstacles, scene.goal, cfg, rng, rep_type=REP_CORRIDOR, use_empirical_init=False, use_mean_nominal=True, progress_by_mode=progress_by_mode, obstacle_circles=active_circles, record_optimal_traj=record)
        elif variant == ControllerVariant.CONTROL_BANK_MPPI:
            u, info, progress_by_mode = stable_swarm_mppi_step(x, modes, active_obstacles, scene.goal, cfg, rng, rep_type=REP_CONTROL_BANK, use_empirical_init=True, use_mean_nominal=False, progress_by_mode=progress_by_mode, obstacle_circles=active_circles, record_optimal_traj=record)
        elif variant == ControllerVariant.MODE_SELECTING_GAUSSIAN_MPPI:
            u, info, progress_by_mode = mode_selecting_stable_mppi_step(x, modes, active_obstacles, scene.goal, cfg, rng, rep_type=REP_GAUSSIAN, progress_by_mode=progress_by_mode, obstacle_circles=active_circles, record_optimal_traj=record)
        elif variant == ControllerVariant.MODE_SELECTING_CORRIDOR_MPPI:
            u, info, progress_by_mode = mode_selecting_stable_mppi_step(x, modes, active_obstacles, scene.goal, cfg, rng, rep_type=REP_CORRIDOR, progress_by_mode=progress_by_mode, obstacle_circles=active_circles, record_optimal_traj=record)
        elif variant in (ControllerVariant.STANDARD_MPPI, ControllerVariant.STANDARD_MPPI_128):
            step_cfg = replace(cfg, num_rollouts=128) if variant == ControllerVariant.STANDARD_MPPI_128 else cfg
            u, info = standard_mppi_step(x, active_obstacles, scene.goal, step_cfg, rng, obstacle_circles=active_circles, record_optimal_traj=record)
        else:
            raise ValueError(f'Unsupported variant: {variant}')
        u = apply_terminal_goal_approach(x, u, scene.goal, cfg.goal_tolerance, cfg)
        u = apply_smooth_safe_control(x, u, previous_control, active_circles, cfg)
        if record:
            update_display_trajectory(info, x, u, cfg)
            infos.append(info)
        previous_control = u.copy()
        x_next = unicycle_step(x, u, cfg.dt)
        arrived, x_at_goal = segment_goal_entry_state(x, x_next, scene.goal, cfg.goal_tolerance)
        x = x_at_goal if arrived else x_next
        states.append(x.copy())
        controls.append(u.copy())
        if arrived:
            reached_goal = True
            break
    if record:
        wall_active = activation_step is not None and len(states) - 1 >= activation_step
        obstacle_history.append(list(blocked_obstacles if wall_active else base_obstacles))
    return SimulationResult(states=np.asarray(states), controls=np.asarray(controls), infos=infos, runtime=time.perf_counter() - started, activation_step=activation_step, obstacle_history=obstacle_history, reached_goal=bool(reached_goal))

@dataclass(frozen=True)
class DynamicWallScenario:
    scenario_id: str
    wall_pairs: Tuple[Tuple[int, int], ...]
    trigger_progress: float = 0.25
    wall_width: float = 0.4
    wall_extension: float = 0.2

def default_dynamic_wall_scenarios() -> List[DynamicWallScenario]:
    return [DynamicWallScenario('wall_0_1', ((0, 1),), trigger_progress=0.25), DynamicWallScenario('wall_1_2', ((1, 2),), trigger_progress=0.25), DynamicWallScenario('walls_0_1__1_2', ((0, 1), (1, 2)), trigger_progress=0.25)]

def build_homotopy_modes(scene: Scene, obstacles: Sequence, swarm_seed: int) -> list[MPPIHomotopyMode]:
    generated = run_swarm_planner(start=scene.start, goal=scene.goal, obstacles=obstacles, scale=scene.scale, bounds_xy=scene.bounds_xy, seed=swarm_seed)
    mixture = fit_topological_trajectory_mixture(generated, K=50, beta=1.0, min_mode_samples=3, covariance_jitter=0.0002, bounds=scene.planner_bounds, goal=scene.goal, snap_to_goal_radius=0.2, snap_straight_tail_points=8)
    return mixture_to_mppi_modes(mixture)

def minimum_clearance(states: Array, obstacles: Sequence, cfg: MPPIConfig) -> float:
    return min_clearance(states, obstacles, cfg.robot_radius)