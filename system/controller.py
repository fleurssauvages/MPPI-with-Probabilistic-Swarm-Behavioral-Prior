from __future__ import annotations

import math
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit, prange
except Exception:
    njit = None
    prange = range

Array = np.ndarray
NUMBA_AVAILABLE = njit is not None

# iLQR mode solves are independent and the Numba entry kernels release the GIL.
# Keep one persistent pool to avoid per-control-step thread creation overhead.
_ILQR_MAX_WORKERS = max(1, min(8, int(os.cpu_count() or 1)))
_ILQR_EXECUTOR = ThreadPoolExecutor(max_workers=_ILQR_MAX_WORKERS, thread_name_prefix="ilqr")
_ILQR_WARM_KEYS: set[tuple[str, bool]] = set()
_ILQR_WARM_LOCK = Lock()


class ControllerVariant(str, Enum):
    PLANNER_ILQR = "planner_ilqr"
    SENSITIVITY_PROJECTED_GAUSSIAN_MPPI = "sensitivity_projected_gaussian_prior_mppi"
    GAUSSIAN_PRIOR_MPPI = "gaussian_prior_mppi"
    CORRIDOR_PRIOR_MPPI = "corridor_prior_mppi"
    CONTROL_BANK_MPPI = "control_bank_mppi"
    MODE_SELECTING_GAUSSIAN_MPPI = "mode_selecting_gaussian_mppi"
    MODE_SELECTING_CORRIDOR_MPPI = "mode_selecting_corridor_mppi"
    STANDARD_MPPI = "standard_mppi"

@dataclass
class ControllerConfig:
    dt: float = 0.10
    horizon: int = 50
    num_rollouts: int = 64
    mppi_iterations: int = 3
    lambda_temperature: float = 16

    temporal_noise_smoothing: float = 0.5

    sigma_ref: float = 1.0

    spg_lookahead_steps: int = 10
    spg_fd_accel: float = 0.05
    spg_fd_steering_rate: float = 0.05
    spg_pseudoinverse_damping: float = 0.05
    spg_covariance_jitter: float = 1e-8

    robot_radius: float = 0.18
    hard_collision_clearance: float = 0.01
    suppress_blocked_modes: bool = True
    mode_blocking_clearance: float = 0.02
    mode_blocking_substeps: int = 2

    w_goal: float = 10.0
    w_obstacle: float = 50.0
    w_terminal_position: float = 100.0
    w_terminal_velocity: float = 100.0
    terminal_velocity_tolerance: float = 0.1
    goal_tolerance: float = 0.1

    mode_select_rollouts_per_mode: int = 0
    max_nearby_prior_modes: int = 32
    max_centerline_distance: float = 1.0
    centerline_history_points: int = 10

    def __post_init__(self) -> None:
        self.horizon = max(1, int(self.horizon))
        self.num_rollouts = max(1, int(self.num_rollouts))
        self.mppi_iterations = max(1, int(self.mppi_iterations))
        self.spg_lookahead_steps = max(1, int(self.spg_lookahead_steps))
        self.max_nearby_prior_modes = max(1, int(self.max_nearby_prior_modes))
        self.centerline_history_points = max(1, int(self.centerline_history_points))
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.lambda_temperature <= 0.0:
            raise ValueError("lambda_temperature must be positive.")
        if self.spg_fd_accel <= 0.0 or self.spg_fd_steering_rate <= 0.0:
            raise ValueError("SPG finite-difference steps must be positive.")
        if self.spg_pseudoinverse_damping < 0.0 or self.spg_covariance_jitter < 0.0:
            raise ValueError("SPG damping and covariance jitter must be nonnegative.")
        if self.max_centerline_distance < 0.0:
            raise ValueError("max_centerline_distance must be nonnegative.")
        if self.w_terminal_position < 0.0:
            raise ValueError("w_terminal_position must be nonnegative.")
        if self.w_terminal_velocity < 0.0:
            raise ValueError("w_terminal_velocity must be nonnegative.")
        if self.terminal_velocity_tolerance < 0.0:
            raise ValueError("terminal_velocity_tolerance must be nonnegative.")


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


@dataclass(frozen=True)
class PackedModeBank:
    """Dense mode geometry packed once for allocation-free Numba localization."""
    mean_paths: Array
    arc_lengths: Array
    lengths: Array


@dataclass(frozen=True)
class DynamicWallScenario:
    scenario_id: str
    wall_pairs: tuple[tuple[int, int], ...]
    trigger_progress: float = 0.3
    wall_width: float = 0.35
    wall_extension: float = 0.0


REP_GAUSSIAN = 1
REP_CORRIDOR = 2
REP_CONTROL_BANK = 3
REP_SENSITIVITY_PROJECTED_GAUSSIAN = 4

def prior_preview_step_distance(cfg: Any) -> float:
    max_speed = max(
        0.0,
        float(getattr(cfg, "max_translational_speed", getattr(cfg, "v_max", 2.8))),
    )
    dt = float(getattr(cfg, "dt", 0.12))
    return max(1e-6, max_speed * dt)


if njit is not None:
    @njit(cache=True)
    def _temporal_smooth_noise_nb(noise, alpha):
        one_minus_alpha = 1.0 - alpha
        for sample in range(noise.shape[0]):
            for t in range(1, noise.shape[1]):
                noise[sample, t, 0] = alpha * noise[sample, t - 1, 0] + one_minus_alpha * noise[sample, t, 0]
                noise[sample, t, 1] = alpha * noise[sample, t - 1, 1] + one_minus_alpha * noise[sample, t, 1]
        return noise

    @njit(cache=True)
    def _apply_projected_covariance_nb(standard_noise, projected):
        output = np.zeros_like(standard_noise)
        for t in range(projected.shape[0]):
            a = projected[t, 0, 0]
            b = 0.5 * (projected[t, 0, 1] + projected[t, 1, 0])
            d = projected[t, 1, 1]
            if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(d)):
                continue
            trace = a + d
            discriminant = math.sqrt(max(0.0, (a - d) * (a - d) + 4.0 * b * b))
            lambda_1 = max(0.0, 0.5 * (trace + discriminant))
            lambda_2 = max(0.0, 0.5 * (trace - discriminant))
            angle = 0.5 * math.atan2(2.0 * b, a - d)
            c = math.cos(angle)
            s = math.sin(angle)
            root_1 = math.sqrt(lambda_1)
            root_2 = math.sqrt(lambda_2)
            r00 = c * c * root_1 + s * s * root_2
            r01 = c * s * (root_1 - root_2)
            r11 = s * s * root_1 + c * c * root_2
            for sample in range(standard_noise.shape[0]):
                z0 = standard_noise[sample, t, 0]
                z1 = standard_noise[sample, t, 1]
                output[sample, t, 0] = z0 * r00 + z1 * r01
                output[sample, t, 1] = z0 * r01 + z1 * r11
        return output
else:
    _temporal_smooth_noise_nb = None
    _apply_projected_covariance_nb = None

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
    return np.column_stack((np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])))


def snap_path_end_to_goal(
    path: Array,
    goal: Optional[Array],
    snap_radius: float = 0.2,
    straight_tail_points: int = 8,
) -> Array:
    p = np.asarray(path, dtype=np.float64)
    if goal is None or p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        return p
    g = np.asarray(goal, dtype=np.float64).reshape(2)
    inside = np.flatnonzero(np.linalg.norm(p - g[None, :], axis=1) <= float(snap_radius))
    if len(inside) == 0:
        return p
    entry = min(int(inside[0]), len(p) - 1)
    tail = np.linspace(p[entry].copy(), g, max(2, int(straight_tail_points)))
    snapped = tail if entry == 0 else np.vstack((p[:entry], tail))
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
    w = np.exp(np.clip(z, -80.0, 80.0))
    total = float(np.sum(w))
    return np.ones_like(w) / len(w) if total <= 1e-12 else w / total


def prepare_mode_prior_cache(mode: MPPIHomotopyMode) -> MPPIHomotopyMode:
    mean_path = np.asarray(mode.mean_path, dtype=np.float64)
    cov_blocks = np.asarray(mode.cov_blocks, dtype=np.float64)
    arc_length = np.zeros(len(mean_path), dtype=np.float64)
    if len(mean_path) > 1:
        arc_length[1:] = np.cumsum(np.linalg.norm(np.diff(mean_path, axis=0), axis=1))
    symmetric_cov = 0.5 * (cov_blocks + np.swapaxes(cov_blocks, 1, 2))
    gaussian_variance = 0.5 * np.trace(symmetric_cov, axis1=1, axis2=2)
    mode.arc_length = np.ascontiguousarray(arc_length)
    mode.gaussian_variance = np.ascontiguousarray(np.maximum(gaussian_variance, 0.0))
    return mode


def _ensure_mode_prior_cache(mode: MPPIHomotopyMode) -> MPPIHomotopyMode:
    if mode.arc_length is None or mode.gaussian_variance is None:
        return prepare_mode_prior_cache(mode)
    return mode


def pack_mode_bank(modes: Sequence[MPPIHomotopyMode]) -> PackedModeBank:
    """Pack mean paths and arc lengths once for repeated controller localization."""
    if not modes:
        return PackedModeBank(
            np.zeros((0, 0, 2), dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )
    cached = [_ensure_mode_prior_cache(mode) for mode in modes]
    lengths = np.asarray([len(mode.mean_path) for mode in cached], dtype=np.int64)
    max_len = int(np.max(lengths)) if lengths.size else 0
    paths = np.zeros((len(cached), max_len, 2), dtype=np.float64)
    arcs = np.zeros((len(cached), max_len), dtype=np.float64)
    for i, mode in enumerate(cached):
        n = int(lengths[i])
        if n <= 0:
            continue
        paths[i, :n] = np.asarray(mode.mean_path, dtype=np.float64)[:, :2]
        arcs[i, :n] = np.asarray(mode.arc_length, dtype=np.float64)[:n]
    return PackedModeBank(np.ascontiguousarray(paths), np.ascontiguousarray(arcs), np.ascontiguousarray(lengths))


def _planner_symbols() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from geometry.utils import PolyObstacle, obstacles_to_segs, round_obstacle
        from graph.graph import build_full_graph
        from planner.env import FishGoalEnv2D
        from planner.planner import HomotopyAwareGenerativePlanner, trajectory_cost
    except Exception as exc: 
        raise ImportError(
            "Could not import the project planner modules. Run from the project root "
            "where geometry/, graph/, planner/, and save/ exist.\n"
            f"Original import error: {exc}"
        ) from exc
    return (
        PolyObstacle,
        obstacles_to_segs,
        round_obstacle,
        build_full_graph,
        FishGoalEnv2D,
        (HomotopyAwareGenerativePlanner, trajectory_cost),
    )


def fit_topological_trajectory_mixture(
    gen_out: Any,
    *,
    K: int = 50,
    beta: float = 1.0,
    min_mode_samples: int = 3,
    covariance_jitter: float = 0.0002,
    costmap: Any = None,
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 10.0), (0.0, 10.0)),
    goal: Optional[Array] = None,
    snap_to_goal_radius: float = 0.2,
    snap_straight_tail_points: int = 8,
) -> TopologicalTrajectoryMixture:
    *_, planner_pair = _planner_symbols()
    _, trajectory_cost = planner_pair
    raw_paths = list(gen_out.samples)
    if not raw_paths:
        raise RuntimeError("Swarm planner produced zero trajectory samples.")
    all_paths = [
        snap_path_end_to_goal(
            path,
            goal=goal,
            snap_radius=snap_to_goal_radius,
            straight_tail_points=snap_straight_tail_points,
        )
        for path in raw_paths
    ]
    all_costs = np.asarray(
        [trajectory_cost(path, costmap=costmap, bounds=bounds, w_len=1.0, w_smooth=0.0) for path in all_paths],
        dtype=np.float64,
    )
    all_weights = stable_softmax_from_cost(all_costs, beta=beta)
    snapped_by_raw_id = {id(raw): snapped for raw, snapped in zip(raw_paths, all_paths)}
    weight_by_raw_id = {id(raw): float(w) for raw, w in zip(raw_paths, all_weights)}
    cost_by_raw_id = {id(raw): float(c) for raw, c in zip(raw_paths, all_costs)}

    mode_raw: dict[Tuple[int, ...], dict[str, Any]] = {}
    total_mode_weight = 0.0
    for signature, paths in gen_out.homotopy_groups.items():
        if len(paths) < min_mode_samples:
            continue
        snapped_paths = [snapped_by_raw_id.get(id(path), path) for path in paths]
        X = np.stack([flatten_path(resample_path(path, K)) for path in snapped_paths], axis=0)
        weights = np.asarray([weight_by_raw_id.get(id(path), 1.0) for path in paths], dtype=np.float64)
        costs = np.asarray([cost_by_raw_id.get(id(path), np.nan) for path in paths], dtype=np.float64)
        weights = np.ones(len(paths), dtype=np.float64) / len(paths) if np.sum(weights) <= 1e-12 else weights / np.sum(weights)
        mean = np.sum(X * weights[:, None], axis=0)
        centered = X - mean[None, :]
        covariance = (centered * weights[:, None]).T @ centered
        covariance = 0.5 * (covariance + covariance.T) + covariance_jitter * np.eye(covariance.shape[0])
        mode_weight = float(sum(weight_by_raw_id.get(id(path), 0.0) for path in paths))
        total_mode_weight += mode_weight
        mode_raw[signature] = {
            "X": X,
            "weights": weights,
            "mean": mean,
            "covariance": covariance,
            "mode_weight": mode_weight,
            "mean_cost": float(np.nanmean(costs)),
        }

    if not mode_raw:
        raise RuntimeError("No homotopy group had enough samples.")
    if total_mode_weight <= 1e-12:
        total_mode_weight = float(len(mode_raw))
        for data in mode_raw.values():
            data["mode_weight"] = 1.0

    modes: Dict[Tuple[int, ...], GaussianTrajectoryMode] = {}
    for signature, data in mode_raw.items():
        modes[signature] = GaussianTrajectoryMode(
            signature=signature,
            probability=float(data["mode_weight"] / total_mode_weight),
            mean=data["mean"],
            cov=data["covariance"],
            samples=data["X"],
            weights=data["weights"],
            mean_cost=data["mean_cost"],
            count=int(data["X"].shape[0]),
        )
    return TopologicalTrajectoryMixture(modes=modes, K=K, beta=beta)


def mixture_to_mppi_modes(mixture: TopologicalTrajectoryMixture) -> List[MPPIHomotopyMode]:
    modes: List[MPPIHomotopyMode] = []
    for signature, mode in mixture.modes.items():
        mean_path = mode.mean_path
        K = mean_path.shape[0]
        cov_blocks = np.zeros((K, 2, 2), dtype=np.float64)
        for t in range(K):
            cov_blocks[t] = mode.cov[2 * t : 2 * t + 2, 2 * t : 2 * t + 2]
        sample_paths = [unflatten_path(vector) for vector in mode.samples]
        modes.append(
            prepare_mode_prior_cache(
                MPPIHomotopyMode(
                    signature=signature,
                    probability=mode.probability,
                    mean_path=mean_path,
                    cov_blocks=cov_blocks,
                    sample_paths=sample_paths,
                )
            )
        )
    modes.sort(key=lambda item: item.probability, reverse=True)
    return modes


def localize_mode_for_state_with_index(
    mode: MPPIHomotopyMode,
    x_current: Array,
    H: int,
    step_distance: Optional[float] = None,
) -> Tuple[MPPIHomotopyMode, int]:
    mode = _ensure_mode_prior_cache(mode)
    mean_path = np.asarray(mode.mean_path, dtype=np.float64)
    nearest_idx = int(np.argmin(np.sum((mean_path - np.asarray(x_current[:2])) ** 2, axis=1)))
    index = min(nearest_idx, len(mean_path) - 2)

    ds = float(step_distance)
    ds = max(ds, 1e-6)
    H = max(1, int(H))
    arc = np.asarray(mode.arc_length, dtype=np.float64)
    s0 = float(arc[index])
    preview_end = min(s0 + max(0, H - 1) * ds, float(arc[-1]))

    end = int(np.searchsorted(arc, preview_end, side="left"))
    end = min(max(end, index + 1), len(mean_path) - 1)

    local_mean = np.ascontiguousarray(mean_path[index : end + 1])
    local_cov = np.ascontiguousarray(np.asarray(mode.cov_blocks, dtype=np.float64)[index : end + 1])
    local_gaussian = np.ascontiguousarray(np.asarray(mode.gaussian_variance, dtype=np.float64)[index : end + 1])
    local_arc = np.ascontiguousarray(arc[index : end + 1] - s0)

    return (
        MPPIHomotopyMode(
            signature=mode.signature,
            probability=mode.probability,
            mean_path=local_mean,
            cov_blocks=local_cov,
            sample_paths=None,
            arc_length=local_arc,
            gaussian_variance=local_gaussian,
        ),
        index,
    )


def localize_mode_for_state(
    mode: MPPIHomotopyMode,
    x_current: Array,
    H: int,
    step_distance: Optional[float] = None,
) -> MPPIHomotopyMode:
    return localize_mode_for_state_with_index(mode, x_current, H, step_distance=step_distance)[0]


def localize_path_for_state_with_index(
    path: Array,
    x_current: Array,
    H: int,
    step_distance: Optional[float] = None,
) -> Tuple[Array, int]:
    """Return the dense empirical-path suffix spanning the control preview."""
    p = np.asarray(path, dtype=np.float64)
    nearest_idx = int(np.argmin(np.linalg.norm(p - np.asarray(x_current[:2]), axis=1)))
    index = min(nearest_idx, len(p) - 2)

    ds = float(step_distance)
    ds = max(ds, 1e-6)
    H = max(1, int(H))
    arc = np.zeros(len(p), dtype=np.float64)
    if len(p) > 1:
        arc[1:] = np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    s0 = float(arc[index])
    preview_end = min(s0 + max(0, H - 1) * ds, float(arc[-1]))
    end = int(np.searchsorted(arc, preview_end, side="left"))
    end = min(max(end, index + 1), len(p) - 1)
    local = np.ascontiguousarray(p[index : end + 1].copy())
    return local, index


def localize_path_for_state(
    path: Array,
    x_current: Array,
    H: int,
    step_distance: Optional[float] = None,
) -> Array:
    return localize_path_for_state_with_index(path, x_current, H, step_distance=step_distance)[0]

def _poly_vertices(obstacle: Any) -> Array:
    if hasattr(obstacle, "vertices"):
        return np.asarray(obstacle.vertices, dtype=np.float64)[:, :2]
    return np.asarray(obstacle, dtype=np.float64)[:, :2]


def obstacle_bounding_circles(
    obstacles: Sequence[Any],
    *,
    elongated_aspect_ratio: float = 2.25,
    max_segment_length: float = 0.1,
    wall_max_segment_length: float = 0.15,
) -> List[Tuple[Array, float]]:
    circles: List[Tuple[Array, float]] = []
    for obstacle in obstacles:
        polygon = _poly_vertices(obstacle)
        center = polygon.mean(axis=0)
        if len(polygon) < 4:
            circles.append((center, float(np.max(np.linalg.norm(polygon - center[None, :], axis=1)))))
            continue
        centered = polygon - center[None, :]
        covariance = centered.T @ centered / max(1, len(polygon))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        minor_axis = np.asarray([-major_axis[1], major_axis[0]], dtype=np.float64)
        major_coord = centered @ major_axis
        minor_coord = centered @ minor_axis
        major_min, major_max = float(np.min(major_coord)), float(np.max(major_coord))
        minor_min, minor_max = float(np.min(minor_coord)), float(np.max(minor_coord))
        length = max(major_max - major_min, 1e-12)
        width = max(minor_max - minor_min, 1e-12)
        if length / width < elongated_aspect_ratio:
            circles.append((center, float(np.max(np.linalg.norm(centered, axis=1)))))
            continue
        target = wall_max_segment_length if len(polygon) == 4 else max_segment_length
        segment_count = max(2, int(math.ceil(length / target)))
        segment_length = length / segment_count
        circle_radius = math.sqrt((0.5 * segment_length) ** 2 + (0.5 * width) ** 2)
        minor_mid = 0.5 * (minor_min + minor_max)
        for index in range(segment_count):
            major_mid = major_min + (index + 0.5) * segment_length
            circle_center = center + major_mid * major_axis + minor_mid * minor_axis
            circles.append((circle_center.astype(np.float64), float(circle_radius)))
    return circles


def _point_in_polygon_geometric(point: Array, polygon: Array) -> bool:
    """Return True when a planar point lies inside a polygon."""
    x = float(point[0])
    y = float(point[1])
    poly = np.asarray(polygon, dtype=np.float64)[:, :2]
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = float(poly[i, 0]), float(poly[i, 1])
        x1, y1 = float(poly[(i + 1) % n, 0]), float(poly[(i + 1) % n, 1])
        if (y0 > y) != (y1 > y):
            x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0 + 1e-18)
            if x < x_cross:
                inside = not inside
    return inside


def _orientation_geometric(a: Array, b: Array, c: Array) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment_geometric(p: Array, a: Array, b: Array, eps: float = 1e-12) -> bool:
    if abs(_orientation_geometric(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect_geometric(a: Array, b: Array, c: Array, d: Array, eps: float = 1e-12) -> bool:
    o1 = _orientation_geometric(a, b, c)
    o2 = _orientation_geometric(a, b, d)
    o3 = _orientation_geometric(c, d, a)
    o4 = _orientation_geometric(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    if abs(o1) <= eps and _on_segment_geometric(c, a, b, eps):
        return True
    if abs(o2) <= eps and _on_segment_geometric(d, a, b, eps):
        return True
    if abs(o3) <= eps and _on_segment_geometric(a, c, d, eps):
        return True
    if abs(o4) <= eps and _on_segment_geometric(b, c, d, eps):
        return True
    return False


def _point_segment_distance_geometric(p: Array, a: Array, b: Array) -> float:
    ab = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    denom = float(ab @ ab)
    if denom <= 1e-18:
        return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - np.asarray(a, dtype=np.float64)))
    u = float(np.clip((np.asarray(p, dtype=np.float64) - np.asarray(a, dtype=np.float64)) @ ab / denom, 0.0, 1.0))
    q = np.asarray(a, dtype=np.float64) + u * ab
    return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - q))


def obstacle_polygons_to_padded_arrays(obstacles: Sequence[Any]) -> Tuple[Array, Array]:
    """Pack true obstacle polygons once for fast geometric prior filtering."""
    polygons = [np.asarray(_poly_vertices(obstacle), dtype=np.float64)[:, :2] for obstacle in obstacles]
    if not polygons:
        return (np.zeros((0, 0, 2), dtype=np.float64), np.zeros(0, dtype=np.int64))
    max_vertices = max(len(poly) for poly in polygons)
    padded = np.zeros((len(polygons), max_vertices, 2), dtype=np.float64)
    lengths = np.zeros(len(polygons), dtype=np.int64)
    for i, poly in enumerate(polygons):
        padded[i, : len(poly)] = poly
        lengths[i] = len(poly)
    return np.ascontiguousarray(padded), np.ascontiguousarray(lengths)


if njit is not None:
    @njit(cache=True)
    def _point_in_polygon_padded_nb(px, py, poly, n):
        inside = False
        for i in range(n):
            j = i + 1
            if j == n:
                j = 0
            x0 = poly[i, 0]
            y0 = poly[i, 1]
            x1 = poly[j, 0]
            y1 = poly[j, 1]
            if (y0 > py) != (y1 > py):
                x_cross = x0 + (py - y0) * (x1 - x0) / (y1 - y0 + 1e-18)
                if px < x_cross:
                    inside = not inside
        return inside


    @njit(cache=True)
    def _orientation_geometric_nb(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


    @njit(cache=True)
    def _on_segment_geometric_nb(px, py, ax, ay, bx, by, eps):
        if abs(_orientation_geometric_nb(ax, ay, bx, by, px, py)) > eps:
            return False
        return (
            min(ax, bx) - eps <= px <= max(ax, bx) + eps
            and min(ay, by) - eps <= py <= max(ay, by) + eps
        )


    @njit(cache=True)
    def _segments_intersect_geometric_nb(ax, ay, bx, by, cx, cy, dx, dy):
        eps = 1e-12
        o1 = _orientation_geometric_nb(ax, ay, bx, by, cx, cy)
        o2 = _orientation_geometric_nb(ax, ay, bx, by, dx, dy)
        o3 = _orientation_geometric_nb(cx, cy, dx, dy, ax, ay)
        o4 = _orientation_geometric_nb(cx, cy, dx, dy, bx, by)
        if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
            return True
        if abs(o1) <= eps and _on_segment_geometric_nb(cx, cy, ax, ay, bx, by, eps):
            return True
        if abs(o2) <= eps and _on_segment_geometric_nb(dx, dy, ax, ay, bx, by, eps):
            return True
        if abs(o3) <= eps and _on_segment_geometric_nb(ax, ay, cx, cy, dx, dy, eps):
            return True
        if abs(o4) <= eps and _on_segment_geometric_nb(bx, by, cx, cy, dx, dy, eps):
            return True
        return False


    @njit(cache=True)
    def _point_segment_distance_geometric_nb(px, py, ax, ay, bx, by):
        abx = bx - ax
        aby = by - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-18:
            dx = px - ax
            dy = py - ay
            return math.sqrt(dx * dx + dy * dy)
        u = ((px - ax) * abx + (py - ay) * aby) / denom
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        qx = ax + u * abx
        qy = ay + u * aby
        dx = px - qx
        dy = py - qy
        return math.sqrt(dx * dx + dy * dy)


    @njit(cache=True)
    def _geometric_mean_path_clearance_nb(path, polygons_padded, polygon_lengths):
        """Exact centerline-vs-polygon test with the old signed-clearance semantics."""
        if path.shape[0] == 0 or polygon_lengths.shape[0] == 0:
            return 1e18
        best = 1e18
        path_count = path.shape[0]
        for m in range(polygon_lengths.shape[0]):
            n = int(polygon_lengths[m])
            if n < 2:
                continue
            poly = polygons_padded[m]

            for i in range(path_count):
                if _point_in_polygon_padded_nb(path[i, 0], path[i, 1], poly, n):
                    return -1e-9

            for i in range(path_count - 1):
                ax = path[i, 0]
                ay = path[i, 1]
                bx = path[i + 1, 0]
                by = path[i + 1, 1]
                for j in range(n):
                    k = j + 1
                    if k == n:
                        k = 0
                    if _segments_intersect_geometric_nb(
                        ax, ay, bx, by,
                        poly[j, 0], poly[j, 1], poly[k, 0], poly[k, 1],
                    ):
                        return -1e-9

            for i in range(path_count):
                px = path[i, 0]
                py = path[i, 1]
                for j in range(n):
                    k = j + 1
                    if k == n:
                        k = 0
                    distance = _point_segment_distance_geometric_nb(
                        px, py, poly[j, 0], poly[j, 1], poly[k, 0], poly[k, 1]
                    )
                    if distance < best:
                        best = distance
        return best
else:
    _geometric_mean_path_clearance_nb = None


if njit is not None:
    @njit(cache=True)
    def _geometric_mean_path_clearance_range_nb(path, start, end, polygons_padded, polygon_lengths):
        if end < start or polygon_lengths.shape[0] == 0:
            return 1e18
        best = 1e18
        for m in range(polygon_lengths.shape[0]):
            n = int(polygon_lengths[m])
            if n < 2:
                continue
            poly = polygons_padded[m]
            for i in range(start, end + 1):
                if _point_in_polygon_padded_nb(path[i, 0], path[i, 1], poly, n):
                    return -1e-9
            for i in range(start, end):
                ax = path[i, 0]; ay = path[i, 1]
                bx = path[i + 1, 0]; by = path[i + 1, 1]
                for j in range(n):
                    k = j + 1
                    if k == n:
                        k = 0
                    if _segments_intersect_geometric_nb(
                        ax, ay, bx, by, poly[j, 0], poly[j, 1], poly[k, 0], poly[k, 1]
                    ):
                        return -1e-9
            for i in range(start, end + 1):
                px = path[i, 0]; py = path[i, 1]
                for j in range(n):
                    k = j + 1
                    if k == n:
                        k = 0
                    distance = _point_segment_distance_geometric_nb(
                        px, py, poly[j, 0], poly[j, 1], poly[k, 0], poly[k, 1]
                    )
                    if distance < best:
                        best = distance
        return best

    @njit(cache=True, parallel=True)
    def _localize_mode_bank_nb(mean_paths, arc_lengths, lengths, px, py, recent_positions, H, ds,
                               polygons_padded, polygon_lengths, compute_clearance):
        M = lengths.shape[0]
        starts = np.zeros(M, dtype=np.int64)
        ends = np.zeros(M, dtype=np.int64)
        distances = np.empty(M, dtype=np.float64)
        clearances = np.empty(M, dtype=np.float64)
        preview_span = max(0, H - 1) * ds
        for m in prange(M):
            n = int(lengths[m])
            if n < 2:
                starts[m] = 0
                ends[m] = max(0, n - 1)
                distances[m] = math.inf
                clearances[m] = math.inf
                continue
            best_d2 = math.inf
            nearest = 0
            for i in range(n):
                dx = mean_paths[m, i, 0] - px
                dy = mean_paths[m, i, 1] - py
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    nearest = i
            start = min(nearest, n - 2)
            s0 = arc_lengths[m, start]
            preview_end = min(s0 + preview_span, arc_lengths[m, n - 1])
            end = start + 1
            while end < n - 1 and arc_lengths[m, end] < preview_end:
                end += 1
            starts[m] = start
            ends[m] = end

            local_best = math.inf
            for i in range(start, end + 1):
                dx = mean_paths[m, i, 0] - px
                dy = mean_paths[m, i, 1] - py
                d2 = dx * dx + dy * dy
                if d2 < local_best:
                    local_best = d2
            centerline_distance = math.sqrt(local_best)
            for h in range(recent_positions.shape[0]):
                hist_best = math.inf
                hx = recent_positions[h, 0]
                hy = recent_positions[h, 1]
                for i in range(n):
                    dx = mean_paths[m, i, 0] - hx
                    dy = mean_paths[m, i, 1] - hy
                    d2 = dx * dx + dy * dy
                    if d2 < hist_best:
                        hist_best = d2
                hd = math.sqrt(hist_best)
                if hd > centerline_distance:
                    centerline_distance = hd
            distances[m] = centerline_distance
            if compute_clearance:
                clearances[m] = _geometric_mean_path_clearance_range_nb(
                    mean_paths[m], start, end, polygons_padded, polygon_lengths
                )
            else:
                clearances[m] = math.inf
        return starts, ends, distances, clearances

    @njit(cache=True, parallel=True)
    def _mode_nearest_distances_nb(mean_paths, lengths, px, py):
        M = lengths.shape[0]
        distances = np.empty(M, dtype=np.float64)
        for m in prange(M):
            n = int(lengths[m])
            best = math.inf
            for i in range(n):
                dx = mean_paths[m, i, 0] - px
                dy = mean_paths[m, i, 1] - py
                d2 = dx * dx + dy * dy
                if d2 < best:
                    best = d2
            distances[m] = math.sqrt(best)
        return distances
else:
    _localize_mode_bank_nb = None
    _mode_nearest_distances_nb = None


def geometric_mean_path_clearance_packed(path: Array, polygons_padded: Array, polygon_lengths: Array) -> float:
    p = np.ascontiguousarray(np.asarray(path, dtype=np.float64))
    if p.ndim != 2 or p.shape[0] == 0 or polygon_lengths.size == 0:
        return float("inf")
    return float(_geometric_mean_path_clearance_nb(p, polygons_padded, polygon_lengths))


def geometric_mean_path_clearance(path: Array, obstacles: Sequence[Any]) -> float:
    """Signed dense-centerline clearance to true polygons, accelerated with Numba."""
    if not obstacles:
        return float("inf")
    polygons_padded, polygon_lengths = obstacle_polygons_to_padded_arrays(obstacles)
    return geometric_mean_path_clearance_packed(path, polygons_padded, polygon_lengths)


def obstacle_configuration_key(
    obstacle_circles: Sequence[Tuple[Array, float]],
) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(
        (float(np.asarray(center)[0]), float(np.asarray(center)[1]), float(radius))
        for center, radius in obstacle_circles
    )


def obstacle_center(obstacle: Any) -> Array:
    return _poly_vertices(obstacle).mean(axis=0)


def make_wall_between_points(p0: Array, p1: Array, width: float = 0.35, extension: float = 0.0) -> Any:
    PolyObstacle, *_ = _planner_symbols()
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length <= 1e-12:
        raise ValueError("Cannot create wall: endpoints are identical.")
    tangent = delta / length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    a = p0 - extension * tangent
    b = p1 + extension * tangent
    half = 0.5 * float(width)
    return PolyObstacle(np.asarray([a + half * normal, b + half * normal, b - half * normal, a - half * normal]))


def make_wall_blockers_between_centers(
    centers: Sequence[Array],
    pairs: Sequence[Tuple[int, int]],
    width: float = 0.35,
    extension: float = 0.0,
) -> List[Any]:
    """Build dynamic blockers.

    The two default blocker IDs are now fixed world-frame walls rather than
    walls joining obstacle centers:
      (0, 1): horizontal wall from (2, 8) to (8, 8)
      (1, 2): vertical wall from (8, 2) to (8, 8)

    Any other pair retains the original obstacle-center behavior for backward
    compatibility with custom scenarios.
    """
    fixed_default_segments = {
        (0, 1): (np.asarray([4.0, 8.0]), np.asarray([8.0, 8.0])),
        (1, 2): (np.asarray([8.0, 4.0]), np.asarray([8.0, 8.0])),
    }

    fixed = [np.asarray(center, dtype=np.float64).reshape(2).copy() for center in centers]
    blockers: List[Any] = []
    for i, j in pairs:
        key = (int(i), int(j))
        reverse_key = (int(j), int(i))
        segment = fixed_default_segments.get(key, fixed_default_segments.get(reverse_key))
        if segment is not None:
            p0, p1 = segment
            blockers.append(make_wall_between_points(p0, p1, width=width, extension=extension))
            continue

        if i == j:
            raise ValueError(f"Cannot create wall for degenerate center pair {(i, j)}.")
        if not (0 <= i < len(fixed) and 0 <= j < len(fixed)):
            raise IndexError(f"Center pair {(i, j)} is outside [0, {len(fixed) - 1}].")
        blockers.append(make_wall_between_points(fixed[i], fixed[j], width=width, extension=extension))
    return blockers


def spatial_progress_along_start_goal(x: Array, start: Array, goal: Array) -> float:
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    direction = goal - start
    denominator = float(direction @ direction)
    if denominator <= 1e-12:
        return 1.0
    return float(np.clip((np.asarray(x[:2]) - start) @ direction / denominator, 0.0, 1.0))


def default_dynamic_wall_scenarios() -> tuple[DynamicWallScenario, ...]:
    return (
        DynamicWallScenario("wall_0_1", ((0, 1),)),
        DynamicWallScenario("wall_1_2", ((1, 2),)),
        DynamicWallScenario("walls_0_1__1_2", ((0, 1), (1, 2))),
    )


def build_default_scene() -> Scene:
    PolyObstacle, _, round_obstacle, *_ = _planner_symbols()
    bounds_xy = (np.asarray([0.0, 0.0]), np.asarray([10.0, 10.0]))
    polygons = (
        np.asarray([[3.0, 1.5], [5.2, 2.2], [4.7, 4.0], [2.8, 3.4]]),
        np.asarray([[6.2, 6.0], [8.5, 6.3], [8.1, 8.2], [6.8, 8.], [5.9, 7.4]]),
        np.asarray([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]]),
        np.asarray([[1.2, 4.2], [2.1, 4.0], [2.4, 4.8], [1.7, 5.3], [1.1, 4.9]]),
        np.asarray([[4.6, 5.1], [5.4, 5.0], [5.8, 5.7], [5.0, 6.2], [4.4, 5.7]]),
        np.asarray([[7.9, 3.0], [9.0, 3.2], [8.8, 4.2], [7.7, 4.0]]),
        np.asarray([[5.7, 1.0], [6.6, 1.2], [6.4, 2.3], [5.6, 2.1]]),
    )
    obstacles = tuple(PolyObstacle(round_obstacle(poly, n_iters=4, n_points=32)) for poly in polygons)
    return Scene(
        scale=4.0,
        bounds_xy=bounds_xy,
        planner_bounds=((0.0, 10.0), (0.0, 10.0)),
        start=np.asarray([1.0, 1.0]),
        goal=np.asarray([9.0, 9.0]),
        obstacles=obstacles,
    )


def run_swarm_planner(
    start: Array,
    goal: Array,
    obstacles: Sequence[Any],
    scale: float,
    bounds_xy: Any,
    *,
    seed: int,
) -> Any:
    _, obstacles_to_segs, _, build_full_graph, FishGoalEnv2D, planner_pair = _planner_symbols()
    HomotopyAwareGenerativePlanner, _ = planner_pair
    segments = obstacles_to_segs(obstacles, scale=scale)
    with open("save/policy.pkl", "rb") as policy_file:
        action = pickle.load(policy_file)["best_theta"]
    graph_goals, graph_weights = build_full_graph(
        obstacles=obstacles,
        start=start,
        goal=goal,
        scale=scale,
        bounds=bounds_xy,
    )
    planner = HomotopyAwareGenerativePlanner(
        env_cls=FishGoalEnv2D,
        action=action,
        obstacles=obstacles,
        segs=segments,
        scale=scale,
        boid_count=1200,
        max_steps=700,
        dt=0.5,
    )
    return planner.sample(
        start_unscaled=start,
        goal_unscaled=goal,
        graph_goals=graph_goals,
        graph_W=graph_weights,
        seed=seed,
    )


def build_homotopy_modes(scene: Scene, obstacles: Sequence[Any], seed: int) -> List[MPPIHomotopyMode]:
    generated = run_swarm_planner(
        scene.start,
        scene.goal,
        obstacles,
        scene.scale,
        scene.bounds_xy,
        seed=seed,
    )
    mixture = fit_topological_trajectory_mixture(
        generated,
        K=200,
        beta=1.0,
        min_mode_samples=1,
        bounds=scene.planner_bounds,
        goal=scene.goal,
    )
    return mixture_to_mppi_modes(mixture)


def sample_dense_scalar_at_arc_positions(
    values: Array, arc_length: Array, positions: Array
) -> Array:
    """Sample a dense geometric scalar field at model-defined control positions."""
    values = np.asarray(values, dtype=np.float64)
    arc = np.asarray(arc_length, dtype=np.float64)
    targets = np.asarray(positions, dtype=np.float64)
    if len(values) == 0:
        return np.zeros(len(targets), dtype=np.float64)
    if len(values) == 1 or len(arc) <= 1 or float(arc[-1]) <= 1e-12:
        return np.full(len(targets), float(values[0]), dtype=np.float64)
    return np.interp(np.clip(targets, 0.0, float(arc[-1])), arc, values)


def sample_dense_covariance_at_arc_positions(
    covariances: Array, arc_length: Array, positions: Array
) -> Array:
    """Sample dense 2x2 covariance blocks at model-defined control positions."""
    cov = np.asarray(covariances, dtype=np.float64)
    arc = np.asarray(arc_length, dtype=np.float64)
    targets = np.asarray(positions, dtype=np.float64)
    if len(cov) == 0:
        return np.zeros((len(targets), 2, 2), dtype=np.float64)
    if len(cov) == 1 or len(arc) <= 1 or float(arc[-1]) <= 1e-12:
        return np.repeat(cov[:1], len(targets), axis=0)
    targets = np.clip(targets, 0.0, float(arc[-1]))
    out = np.empty((len(targets), 2, 2), dtype=np.float64)
    for row in range(2):
        for col in range(2):
            out[:, row, col] = np.interp(targets, arc, cov[:, row, col])
    return out

def make_temporally_correlated_noise(
    model: Any,
    n: int,
    H: int,
    cfg: Any,
    rng: np.random.Generator,
    *,
    scale_override: Optional[Array] = None,
) -> Array:
    scale = np.asarray(model.control_noise_scale(cfg), dtype=np.float64)
    if scale_override is not None:
        scale = np.asarray(scale_override, dtype=np.float64)
    if scale.shape != (2,):
        raise ValueError("model.control_noise_scale(cfg) must return shape (2,).")
    noise = rng.normal(size=(n, H, 2))
    noise *= scale[None, None, :]
    alpha = float(cfg.temporal_noise_smoothing)
    if _temporal_smooth_noise_nb is not None:
        return _temporal_smooth_noise_nb(noise, alpha)
    for t in range(1, H):
        noise[:, t, :] = alpha * noise[:, t - 1, :] + (1.0 - alpha) * noise[:, t, :]
    return noise


def sample_controls_around_nominal(
    model: Any,
    nominal: Array,
    n: int,
    cfg: Any,
    rng: np.random.Generator,
) -> Array:
    if n <= 0:
        return np.zeros((0, cfg.horizon, 2), dtype=np.float64)
    center = np.asarray(nominal, dtype=np.float64)
    controls = make_temporally_correlated_noise(model, n, cfg.horizon, cfg, rng)
    controls += center[None, :, :]
    clip_inplace = getattr(model, "clip_control_batch_inplace", None)
    return clip_inplace(controls, cfg) if clip_inplace is not None else model.clip_control_batch(controls, cfg)


def sample_exact_control_bank(
    model: Any,
    x_current: Array,
    global_mode: MPPIHomotopyMode,
    n: int,
    cfg: Any,
) -> Array:
    """Convert exactly ``n`` unique empirical trajectories from one mode.

    The closest trajectories to the current position are selected, localized,
    and independently mapped to dynamically feasible controls with iLQR.
    No trajectories are repeated and no control perturbations are added.
    """
    if n <= 0:
        return np.zeros((0, cfg.horizon, 2), dtype=np.float64)

    paths = list(global_mode.sample_paths or [])
    if len(paths) < int(n):
        raise ValueError(
            f"Control-bank mode {global_mode.signature!s} contains only "
            f"{len(paths)} empirical trajectories but {int(n)} were requested."
        )

    x_xy = np.asarray(x_current[:2], dtype=np.float64)
    distances = np.empty(len(paths), dtype=np.float64)
    for sample_id, path in enumerate(paths):
        p = np.asarray(path, dtype=np.float64)
        if len(p) == 0:
            distances[sample_id] = np.inf
        else:
            delta = p[:, :2] - x_xy[None, :]
            distances[sample_id] = float(np.min(np.sum(delta * delta, axis=1)))

    selected_ids = np.argsort(distances, kind="stable")[: int(n)]
    localized: List[Array] = []
    for sample_id in selected_ids:
        local_path, _ = localize_path_for_state_with_index(
            paths[int(sample_id)],
            x_current,
            cfg.horizon,
            step_distance=prior_preview_step_distance(cfg),
        )
        localized.append(local_path)

    if len(localized) <= 1 or _ILQR_MAX_WORKERS <= 1:
        controls = np.asarray(
            [model.nominal_controls_to_track_path(x_current, path, cfg) for path in localized],
            dtype=np.float64,
        )
    else:
        futures = [
            _ILQR_EXECUTOR.submit(model.nominal_controls_to_track_path, x_current, path, cfg)
            for path in localized
        ]
        controls = np.asarray([future.result() for future in futures], dtype=np.float64)

    return model.clip_control_batch(controls, cfg)


def balanced_unique_control_bank_counts(
    total_budget: int,
    modes: Sequence[MPPIHomotopyMode],
) -> Array:
    """Allocate an exact bank budget as evenly as possible without repeats."""
    total_budget = int(total_budget)
    capacities = np.asarray([len(mode.sample_paths or []) for mode in modes], dtype=np.int64)
    if total_budget < 0:
        raise ValueError("Control-bank rollout budget must be non-negative.")
    if int(np.sum(capacities)) < total_budget:
        raise ValueError(
            f"Control bank contains {int(np.sum(capacities))} unique trajectories, "
            f"but cfg.num_rollouts={total_budget}."
        )

    counts = np.zeros(len(modes), dtype=np.int64)
    remaining = total_budget
    active = [index for index, capacity in enumerate(capacities) if capacity > 0]
    while remaining > 0:
        progressed = False
        for index in active:
            if remaining <= 0:
                break
            if counts[index] < capacities[index]:
                counts[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("Unable to allocate the requested unique control-bank budget.")
    return counts

def nominal_controls_and_arc_positions(
    model: Any,
    x_current: Array,
    path: Array,
    cfg: Any,
    cov_blocks: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Compute one cold spatially covariance-conditioned iLQR nominal."""
    combined = getattr(model, "nominal_controls_and_arc_positions", None)
    if combined is not None:
        controls, positions = combined(x_current, path, cfg, cov_blocks)
        return np.asarray(controls, dtype=np.float64), np.asarray(positions, dtype=np.float64)
    controls = model.nominal_controls_to_track_path(x_current, path, cfg, cov_blocks)
    positions = model.prior_control_arc_positions(x_current, path, cfg, cov_blocks)
    return np.asarray(controls, dtype=np.float64), np.asarray(positions, dtype=np.float64)


def _solve_mode_nominal_worker(model: Any, x_current: Array, mode: MPPIHomotopyMode, cfg: Any, need_jacobians: bool):
    if need_jacobians:
        combined = getattr(model, "nominal_controls_and_arc_positions_with_jacobians", None)
        project = getattr(model, "project_control_covariances_from_jacobians", None)
        if combined is not None and project is not None:
            controls, positions, A, B = combined(
                x_current, mode.mean_path, cfg, mode.cov_blocks, None
            )
            return (
                np.asarray(controls, dtype=np.float64),
                np.asarray(positions, dtype=np.float64),
                np.asarray(A, dtype=np.float64),
                np.asarray(B, dtype=np.float64),
            )
    controls, positions = nominal_controls_and_arc_positions(
        model, x_current, mode.mean_path, cfg, mode.cov_blocks
    )
    return controls, positions, None, None


def parallel_mode_nominals(
    model: Any,
    x_current: Array,
    local_modes: Sequence[MPPIHomotopyMode],
    cfg: Any,
    *,
    need_jacobians: bool = False,
) -> List[Tuple[Array, Array, Optional[Array], Optional[Array]]]:
    """Solve independent cold iLQR mode priors concurrently, preserving mode order."""
    if not local_modes:
        return []
    if len(local_modes) == 1 or _ILQR_MAX_WORKERS <= 1:
        return [_solve_mode_nominal_worker(model, x_current, local_modes[0], cfg, need_jacobians)]

    # Do not send multiple threads into the same Numba dispatcher while its first
    # specialization is compiling.  That can multiply cold-start compile work.
    # Warm exactly one solve per model/path type, then use the persistent pool.
    model_key = str(getattr(model, "MODEL_NAME", getattr(model, "__name__", type(model).__name__)))
    warm_key = (model_key, bool(need_jacobians))
    first_solution = None
    with _ILQR_WARM_LOCK:
        if warm_key not in _ILQR_WARM_KEYS:
            first_solution = _solve_mode_nominal_worker(
                model, x_current, local_modes[0], cfg, need_jacobians
            )
            _ILQR_WARM_KEYS.add(warm_key)

    start_index = 1 if first_solution is not None else 0
    futures = [
        _ILQR_EXECUTOR.submit(_solve_mode_nominal_worker, model, x_current, mode, cfg, need_jacobians)
        for mode in local_modes[start_index:]
    ]
    tail = [future.result() for future in futures]
    return ([first_solution] + tail) if first_solution is not None else tail


def _sample_gaussian_from_nominal(
    model: Any, local_mode: MPPIHomotopyMode, ilqr_nominal: Array, control_positions: Array,
    n: int, cfg: Any, rng: np.random.Generator,
) -> Array:
    H = int(cfg.horizon)
    noise = make_temporally_correlated_noise(model, n, H, cfg, rng)
    variance = sample_dense_scalar_at_arc_positions(
        np.asarray(local_mode.gaussian_variance, dtype=np.float64),
        np.asarray(local_mode.arc_length, dtype=np.float64),
        control_positions,
    )
    sigma_ref = max(float(cfg.sigma_ref), 1e-9)
    noise *= (np.sqrt(np.maximum(variance, 0.0)) / sigma_ref)[None, :, None]
    noise += np.asarray(ilqr_nominal, dtype=np.float64)[None, :, :]
    clip_inplace = getattr(model, "clip_control_batch_inplace", None)
    return clip_inplace(noise, cfg) if clip_inplace is not None else model.clip_control_batch(noise, cfg)


def _sample_spg_from_nominal(
    model: Any, x_current: Array, local_mode: MPPIHomotopyMode, ilqr_nominal: Array,
    control_positions: Array, A: Optional[Array], B: Optional[Array], n: int, cfg: Any,
    rng: np.random.Generator,
) -> Array:
    H = int(cfg.horizon)
    cov_at_controls = sample_dense_covariance_at_arc_positions(
        np.asarray(local_mode.cov_blocks, dtype=np.float64),
        np.asarray(local_mode.arc_length, dtype=np.float64),
        control_positions,
    )
    project_from_jacobians = getattr(model, "project_control_covariances_from_jacobians", None)
    if A is not None and B is not None and project_from_jacobians is not None:
        projected = project_from_jacobians(A, B, cov_at_controls, cfg)
    else:
        projected = model.project_control_covariances(x_current, ilqr_nominal, cov_at_controls, cfg)
    standard_noise = make_temporally_correlated_noise(
        model, n, H, cfg, rng, scale_override=np.ones(2, dtype=np.float64)
    )
    if _apply_projected_covariance_nb is not None:
        noise = _apply_projected_covariance_nb(standard_noise, np.asarray(projected, dtype=np.float64))
    else:
        noise = np.zeros_like(standard_noise)
        for t in range(H):
            covariance = 0.5 * (projected[t] + projected[t].T)
            if not np.all(np.isfinite(covariance)):
                covariance = np.zeros((2, 2), dtype=np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            square_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
            noise[:, t, :] = standard_noise[:, t, :] @ square_root.T
    noise += np.asarray(ilqr_nominal, dtype=np.float64)[None, :, :]
    clip_inplace = getattr(model, "clip_control_batch_inplace", None)
    return clip_inplace(noise, cfg) if clip_inplace is not None else model.clip_control_batch(noise, cfg)


def sample_gaussian_controls_with_nominal(
    model: Any,
    x_current: Array,
    local_mode: MPPIHomotopyMode,
    n: int,
    cfg: Any,
    rng: np.random.Generator,
    *,
    goal: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Gaussian proposal centered exactly on the path-tracking iLQR solution."""
    del goal
    H = int(cfg.horizon)
    if n <= 0:
        return np.zeros((0, H, 2), dtype=np.float64), np.zeros((H, 2), dtype=np.float64)

    local_mode = _ensure_mode_prior_cache(local_mode)
    ilqr_nominal, control_positions = nominal_controls_and_arc_positions(
        model, x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
    )
    noise = make_temporally_correlated_noise(model, n, H, cfg, rng)
    variance = sample_dense_scalar_at_arc_positions(
        np.asarray(local_mode.gaussian_variance, dtype=np.float64),
        np.asarray(local_mode.arc_length, dtype=np.float64),
        control_positions,
    )
    sigma_ref = max(float(cfg.sigma_ref), 1e-9)
    scale = np.sqrt(np.maximum(variance, 0.0)) / sigma_ref
    noise *= scale[None, :, None]
    noise += np.asarray(ilqr_nominal, dtype=np.float64)[None, :, :]
    clip_inplace = getattr(model, "clip_control_batch_inplace", None)
    controls = clip_inplace(noise, cfg) if clip_inplace is not None else model.clip_control_batch(noise, cfg)
    return controls, np.asarray(ilqr_nominal, dtype=np.float64).copy()

def sample_gaussian_controls(
    model: Any,
    x_current: Array,
    local_mode: MPPIHomotopyMode,
    n: int,
    cfg: Any,
    rng: np.random.Generator,
    *,
    goal: Optional[Array] = None,
) -> Array:
    controls, _ = sample_gaussian_controls_with_nominal(
        model, x_current, local_mode, n, cfg, rng, goal=goal
    )
    return controls


def sample_sensitivity_projected_gaussian_controls_with_nominal(
    model: Any,
    x_current: Array,
    local_mode: MPPIHomotopyMode,
    n: int,
    cfg: Any,
    rng: np.random.Generator,
    *,
    goal: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """SPG proposal centered on iLQR and reusing its dynamics Jacobians when available."""
    del goal
    H = int(cfg.horizon)
    if n <= 0:
        return np.zeros((0, H, 2), dtype=np.float64), np.zeros((H, 2), dtype=np.float64)
    local_mode = _ensure_mode_prior_cache(local_mode)
    cov_at_controls = None
    with_jacobians = getattr(model, "nominal_controls_and_arc_positions_with_jacobians", None)
    project_from_jacobians = getattr(model, "project_control_covariances_from_jacobians", None)
    if with_jacobians is not None and project_from_jacobians is not None:
        ilqr_nominal, control_positions, A, B = with_jacobians(
            x_current, local_mode.mean_path, cfg, local_mode.cov_blocks, None
        )
        cov_at_controls = sample_dense_covariance_at_arc_positions(
            np.asarray(local_mode.cov_blocks, dtype=np.float64),
            np.asarray(local_mode.arc_length, dtype=np.float64),
            control_positions,
        )
        projected = project_from_jacobians(A, B, cov_at_controls, cfg)
    else:
        ilqr_nominal, control_positions = nominal_controls_and_arc_positions(
            model, x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
        )
        cov_at_controls = sample_dense_covariance_at_arc_positions(
            np.asarray(local_mode.cov_blocks, dtype=np.float64),
            np.asarray(local_mode.arc_length, dtype=np.float64),
            control_positions,
        )
        projected = model.project_control_covariances(x_current, ilqr_nominal, cov_at_controls, cfg)
    standard_noise = make_temporally_correlated_noise(
        model, n, H, cfg, rng, scale_override=np.ones(2, dtype=np.float64)
    )
    if _apply_projected_covariance_nb is not None:
        noise = _apply_projected_covariance_nb(standard_noise, np.asarray(projected, dtype=np.float64))
    else:
        noise = np.zeros_like(standard_noise)
        for t in range(H):
            covariance = 0.5 * (projected[t] + projected[t].T)
            if not np.all(np.isfinite(covariance)):
                covariance = np.zeros((2, 2), dtype=np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            square_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
            noise[:, t, :] = standard_noise[:, t, :] @ square_root.T
    noise += np.asarray(ilqr_nominal, dtype=np.float64)[None, :, :]
    clip_inplace = getattr(model, "clip_control_batch_inplace", None)
    controls = clip_inplace(noise, cfg) if clip_inplace is not None else model.clip_control_batch(noise, cfg)
    return controls, np.asarray(ilqr_nominal, dtype=np.float64).copy()

def sample_sensitivity_projected_gaussian_controls(
    model: Any,
    x_current: Array,
    local_mode: MPPIHomotopyMode,
    n: int,
    cfg: Any,
    rng: np.random.Generator,
    *,
    goal: Optional[Array] = None,
) -> Array:
    controls, _ = sample_sensitivity_projected_gaussian_controls_with_nominal(
        model, x_current, local_mode, n, cfg, rng, goal=goal
    )
    return controls


def cached_mode_mean_clearances(
    model: Any,
    global_modes: Sequence[MPPIHomotopyMode],
    obstacle_circles: Sequence[Tuple[Array, float]],
    cfg: Any,
) -> Array:
    if not global_modes:
        return np.zeros(0, dtype=np.float64)
    if not cfg.suppress_blocked_modes or not obstacle_circles:
        return np.full(len(global_modes), np.inf, dtype=np.float64)
    return np.asarray(
        [model.mean_path_clearance(mode.mean_path, obstacle_circles, cfg) for mode in global_modes],
        dtype=np.float64,
    )


def nearby_mode_indices(
    global_modes: Sequence[MPPIHomotopyMode],
    x_current: Array,
    cfg: Any,
    cached_clearances: Optional[Array] = None,
    *,
    packed_mode_bank: Optional[PackedModeBank] = None,
) -> List[int]:
    """Sequentially keep the nearest feasible modes, using packed Numba distances."""
    if not global_modes:
        return []
    position = np.asarray(x_current[:2], dtype=np.float64)
    bank = packed_mode_bank if packed_mode_bank is not None else pack_mode_bank(global_modes)
    if _mode_nearest_distances_nb is not None:
        distances = _mode_nearest_distances_nb(
            bank.mean_paths, bank.lengths, float(position[0]), float(position[1])
        )
    else:
        distances = np.asarray(
            [float(np.min(np.linalg.norm(np.asarray(mode.mean_path) - position[None, :], axis=1))) for mode in global_modes],
            dtype=np.float64,
        )
    order = np.argsort(distances)
    if cached_clearances is None or not cfg.suppress_blocked_modes:
        feasible = np.ones(len(global_modes), dtype=bool)
    else:
        clearances = np.asarray(cached_clearances, dtype=np.float64)
        if clearances.shape != (len(global_modes),):
            raise ValueError("cached_clearances must contain one value per global mode.")
        feasible = clearances >= float(cfg.mode_blocking_clearance)

    selected: List[int] = []
    limit = min(max(1, int(cfg.max_nearby_prior_modes)), len(global_modes))
    max_distance = float(cfg.max_centerline_distance)
    for index in order:
        i = int(index)
        if distances[i] > max_distance:
            break
        if not feasible[i]:
            continue
        selected.append(i)
        if len(selected) >= limit:
            break
    return selected


def balanced_rollout_counts(total: int, groups: int) -> List[int]:
    total = max(1, int(total))
    groups = max(1, min(int(groups), total))
    base, remainder = divmod(total, groups)
    return [base + (1 if index < remainder else 0) for index in range(groups)]


def renormalized_mode_probabilities(modes: Sequence[MPPIHomotopyMode]) -> Array:
    """Return nonnegative homotopy probabilities normalized over the active set."""
    if not modes:
        return np.zeros(0, dtype=np.float64)
    probabilities = np.asarray(
        [max(0.0, float(mode.probability)) for mode in modes],
        dtype=np.float64,
    )
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 1e-12:
        return np.full(len(modes), 1.0 / float(len(modes)), dtype=np.float64)
    return probabilities / total


def probability_proportional_rollout_counts(total: int, probabilities: Array) -> List[int]:
    """Allocate a fixed rollout budget according to renormalized mode weights.

    Largest-remainder rounding keeps the integer allocation as close as possible
    to ``total * probabilities``.  When the budget can cover every active mode,
    a zero-count mode is given one rollout by transferring a rollout from the
    mode with the largest positive allocation surplus.  This guarantees that
    every feasible prior mean is explicitly represented without changing the
    total MPPI budget.
    """
    total = max(1, int(total))
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return []
    p = np.maximum(p, 0.0)
    mass = float(np.sum(p))
    if not np.isfinite(mass) or mass <= 1e-12:
        p = np.full(p.size, 1.0 / float(p.size), dtype=np.float64)
    else:
        p = p / mass

    ideal = float(total) * p
    counts = np.floor(ideal).astype(np.int64)
    remainder = int(total - int(np.sum(counts)))
    if remainder > 0:
        fractions = ideal - counts
        order = np.argsort(-fractions, kind="stable")
        counts[order[:remainder]] += 1

    if total >= p.size:
        zero_ids = np.flatnonzero(counts == 0)
        for zero_id in zero_ids:
            donor_ids = np.flatnonzero(counts > 1)
            if donor_ids.size == 0:
                break
            surplus = counts[donor_ids].astype(np.float64) - ideal[donor_ids]
            donor = int(donor_ids[int(np.argmax(surplus))])
            counts[donor] -= 1
            counts[int(zero_id)] += 1

    return counts.astype(int).tolist()


def localize_all_feasible_mean_modes(
    global_modes: Sequence[MPPIHomotopyMode],
    x_current: Array,
    obstacles: Sequence[Any],
    cfg: Any,
    progress: Dict[str, int],
    state_history: Optional[Sequence[Array]] = None,
    *,
    packed_mode_bank: Optional[PackedModeBank] = None,
    packed_polygons: Optional[Tuple[Array, Array]] = None,
) -> Tuple[List[int], List[MPPIHomotopyMode], List[float], List[float], Dict[str, int]]:
    """Localize and filter all prior means using one packed Numba geometry pass."""
    active_indices: List[int] = []
    active_modes: List[MPPIHomotopyMode] = []
    active_clearances: List[float] = []
    new_progress = dict(progress)
    initial_prior_pass = len(progress) == 0
    if not global_modes:
        return active_indices, active_modes, active_clearances, [], new_progress

    bank = packed_mode_bank if packed_mode_bank is not None else pack_mode_bank(global_modes)
    position = np.asarray(x_current[:2], dtype=np.float64)
    history_count = max(1, int(cfg.centerline_history_points))
    if state_history:
        recent_positions = np.empty((min(history_count, len(state_history)), 2), dtype=np.float64)
        selected_history = state_history[-recent_positions.shape[0]:]
        for i, state in enumerate(selected_history):
            recent_positions[i, 0] = float(state[0])
            recent_positions[i, 1] = float(state[1])
    else:
        recent_positions = np.zeros((0, 2), dtype=np.float64)

    compute_clearance = bool((not initial_prior_pass) and cfg.suppress_blocked_modes and obstacles)
    if compute_clearance:
        if packed_polygons is None:
            polygons_padded, polygon_lengths = obstacle_polygons_to_padded_arrays(obstacles)
        else:
            polygons_padded, polygon_lengths = packed_polygons
    else:
        polygons_padded = np.zeros((0, 0, 2), dtype=np.float64)
        polygon_lengths = np.zeros(0, dtype=np.int64)

    if _localize_mode_bank_nb is not None:
        starts, ends, distances, clearances = _localize_mode_bank_nb(
            bank.mean_paths, bank.arc_lengths, bank.lengths,
            float(position[0]), float(position[1]), np.ascontiguousarray(recent_positions),
            max(1, int(cfg.horizon)), prior_preview_step_distance(cfg),
            np.ascontiguousarray(polygons_padded), np.ascontiguousarray(polygon_lengths),
            compute_clearance,
        )
    else:
        starts = np.zeros(len(global_modes), dtype=np.int64)
        ends = np.zeros(len(global_modes), dtype=np.int64)
        distances = np.empty(len(global_modes), dtype=np.float64)
        clearances = np.full(len(global_modes), np.inf, dtype=np.float64)
        for i, mode in enumerate(global_modes):
            local, start_idx = localize_mode_for_state_with_index(
                mode, x_current, cfg.horizon, step_distance=prior_preview_step_distance(cfg)
            )
            starts[i] = start_idx
            ends[i] = start_idx + len(local.mean_path) - 1
            delta = np.asarray(local.mean_path)[:, :2] - position[None, :]
            distances[i] = float(np.sqrt(np.min(np.sum(delta * delta, axis=1))))
            if compute_clearance:
                clearances[i] = geometric_mean_path_clearance_packed(
                    local.mean_path, polygons_padded, polygon_lengths
                )

    all_clearances = [float(value) for value in clearances]
    max_centerline_distance = float(cfg.max_centerline_distance)
    for global_index, mode in enumerate(global_modes):
        mode = _ensure_mode_prior_cache(mode)
        index = int(starts[global_index])
        end_index = int(ends[global_index])
        new_progress[str(mode.signature)] = index
        if not initial_prior_pass:
            if float(distances[global_index]) > max_centerline_distance:
                continue
            if compute_clearance and float(clearances[global_index]) <= 0.0:
                continue

        mean_path = np.asarray(mode.mean_path, dtype=np.float64)
        arc = np.asarray(mode.arc_length, dtype=np.float64)
        s0 = float(arc[index])
        local_mode = MPPIHomotopyMode(
            signature=mode.signature,
            probability=mode.probability,
            mean_path=np.ascontiguousarray(mean_path[index:end_index + 1]),
            cov_blocks=np.ascontiguousarray(np.asarray(mode.cov_blocks, dtype=np.float64)[index:end_index + 1]),
            sample_paths=None,
            arc_length=np.ascontiguousarray(arc[index:end_index + 1] - s0),
            gaussian_variance=np.ascontiguousarray(
                np.asarray(mode.gaussian_variance, dtype=np.float64)[index:end_index + 1]
            ),
        )
        active_indices.append(int(global_index))
        active_modes.append(local_mode)
        active_clearances.append(float(clearances[global_index]))

    return active_indices, active_modes, active_clearances, all_clearances, new_progress


def softmin_score(costs: Array, cfg: Any) -> float:
    values = np.asarray(costs, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return 1e309
    finite_values = values[finite]
    rho = float(np.min(finite_values))
    z = np.exp(-(finite_values - rho) / float(cfg.lambda_temperature))
    return float(rho - float(cfg.lambda_temperature) * math.log(np.sum(z) / len(values) + 1e-12))


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _mppi_weighted_sequence_nb(costs, controls, temperature):
        n = costs.shape[0]
        H = controls.shape[1]
        d = controls.shape[2]
        out = np.zeros((H, d), dtype=np.float64)
        finite_count = 0
        rho = math.inf
        for i in range(n):
            if math.isfinite(costs[i]):
                finite_count += 1
                if costs[i] < rho:
                    rho = costs[i]
        if finite_count == 0:
            if n == 0:
                return out
            inv_n = 1.0 / n
            for i in range(n):
                for h in range(H):
                    for j in range(d):
                        out[h, j] += inv_n * controls[i, h, j]
            return out
        total = 0.0
        for i in range(n):
            if math.isfinite(costs[i]):
                w = math.exp(-(costs[i] - rho) / temperature)
                total += w
                for h in range(H):
                    for j in range(d):
                        out[h, j] += w * controls[i, h, j]
        if total <= 1e-12:
            inv_count = 1.0 / finite_count
            for h in range(H):
                for j in range(d):
                    out[h, j] = 0.0
            for i in range(n):
                if math.isfinite(costs[i]):
                    for h in range(H):
                        for j in range(d):
                            out[h, j] += inv_count * controls[i, h, j]
        else:
            inv_total = 1.0 / total
            for h in range(H):
                for j in range(d):
                    out[h, j] *= inv_total
        return out
else:
    _mppi_weighted_sequence_nb = None


def _prepare_model_obstacles(model: Any, obstacle_circles):
    pack = getattr(model, "pack_obstacle_circles", None)
    return pack(obstacle_circles) if pack is not None else None


def _rollout_costs_and_collisions(
    model: Any,
    x_current: Array,
    controls: Array,
    obstacle_circles,
    goal: Array,
    cfg: Any,
    packed_obstacles=None,
) -> Tuple[Array, Array, Optional[Array]]:
    fused = getattr(model, "rollout_costs_and_collisions", None)
    if fused is not None:
        costs, colliding = fused(
            x_current, controls, obstacle_circles, goal, cfg,
            packed_obstacles=packed_obstacles,
        )
        return np.asarray(costs, dtype=np.float64), np.asarray(colliding, dtype=bool), None
    states = model.rollout_batch(x_current, controls, cfg)
    raw_costs = model.trajectory_costs(states, controls, obstacle_circles, goal, cfg)
    colliding = (
        model.collision_mask(states, obstacle_circles, goal, cfg)
        if obstacle_circles else np.zeros(len(controls), dtype=bool)
    )
    return np.asarray(raw_costs, dtype=np.float64), np.asarray(colliding, dtype=bool), states


def mppi_weights(costs: Array, cfg: Any) -> Array:
    costs = np.asarray(costs, dtype=np.float64)
    finite = np.isfinite(costs)
    if not np.any(finite):
        return np.ones(len(costs), dtype=np.float64) / max(1, len(costs))
    rho = float(np.min(costs[finite]))
    weights = np.zeros_like(costs)
    weights[finite] = np.exp(-(costs[finite] - rho) / float(cfg.lambda_temperature))
    total = float(weights.sum())
    if total <= 1e-12:
        weights[finite] = 1.0 / float(np.count_nonzero(finite))
    else:
        weights /= total
    return weights


def mppi_weighted_control_sequence(model: Any, costs: Array, controls: Array, cfg: Any) -> Array:
    costs_arr = np.ascontiguousarray(np.asarray(costs, dtype=np.float64))
    controls_arr = np.ascontiguousarray(np.asarray(controls, dtype=np.float64))
    if _mppi_weighted_sequence_nb is not None:
        sequence = _mppi_weighted_sequence_nb(costs_arr, controls_arr, float(cfg.lambda_temperature))
    else:
        sequence = np.tensordot(mppi_weights(costs_arr, cfg), controls_arr, axes=(0, 0))
    clip_sequence = getattr(model, "clip_control_sequence", None)
    if clip_sequence is not None:
        return np.asarray(clip_sequence(sequence, cfg), dtype=np.float64)
    return model.clip_control_batch(np.asarray(sequence, dtype=np.float64)[None, :, :], cfg)[0]


def _single_sequence_evaluation(
    model: Any,
    x_current: Array,
    sequence: Array,
    obstacle_circles: Sequence[Tuple[Array, float]],
    goal: Array,
    cfg: Any,
    packed_obstacles=None,
) -> Tuple[float, Array, bool]:
    clip_sequence = getattr(model, "clip_control_sequence", None)
    if clip_sequence is not None:
        controls = np.asarray(clip_sequence(sequence, cfg), dtype=np.float64)
    else:
        controls = model.clip_control_batch(np.asarray(sequence, dtype=np.float64)[None, :, :], cfg)[0]
    evaluate = getattr(model, "evaluate_sequence", None)
    if evaluate is not None:
        raw_cost, states, colliding = evaluate(
            x_current, controls, obstacle_circles, goal, cfg,
            packed_obstacles=packed_obstacles,
        )
        return (np.inf if colliding else float(raw_cost)), np.asarray(states), bool(colliding)
    states = model.rollout_single(x_current, controls, cfg)
    batch_states = states[None, :, :]
    batch_controls = controls[None, :, :]
    raw_cost = float(model.trajectory_costs(batch_states, batch_controls, obstacle_circles, goal, cfg)[0])
    colliding = bool(model.collision_mask(batch_states, obstacle_circles, goal, cfg)[0]) if obstacle_circles else False
    return (np.inf if colliding else raw_cost), np.asarray(states), colliding

def _finite_cost_mean(costs: Array) -> float:
    values = np.asarray(costs, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("inf")


def _accept_improving_candidate(
    model: Any,
    x_current: Array,
    current: Array,
    current_cost: float,
    current_traj: Array,
    candidate: Array,
    obstacle_circles: Sequence[Tuple[Array, float]],
    goal: Array,
    cfg: Any,
    packed_obstacles=None,
) -> Tuple[Array, float, Array, bool]:
    candidate_cost, candidate_traj, candidate_collision = _single_sequence_evaluation(
        model, x_current, candidate, obstacle_circles, goal, cfg, packed_obstacles
    )
    improves = (
        not candidate_collision
        and np.isfinite(candidate_cost)
        and (not np.isfinite(current_cost) or candidate_cost < current_cost - 1e-9)
    )
    if improves:
        return np.asarray(candidate, dtype=np.float64), float(candidate_cost), candidate_traj, True
    return np.asarray(current, dtype=np.float64), float(current_cost), np.asarray(current_traj), False


def _standard_mppi_refinement_pass(
    model: Any,
    x_current: Array,
    center: Array,
    rollout_count: int,
    obstacle_circles: Sequence[Tuple[Array, float]],
    goal: Array,
    cfg: Any,
    rng: np.random.Generator,
    packed_obstacles=None,
) -> Tuple[Array, Array, Optional[Array]]:
    controls = sample_controls_around_nominal(model, center, rollout_count, cfg, rng)
    raw_costs, colliding, states = _rollout_costs_and_collisions(
        model, x_current, controls, obstacle_circles, goal, cfg, packed_obstacles
    )
    raw_costs[colliding] = np.inf
    candidate = mppi_weighted_control_sequence(model, raw_costs, controls, cfg)
    return candidate, raw_costs, states

def _run_additional_mppi_refinements(
    model: Any,
    x_current: Array,
    current: Array,
    current_cost: float,
    current_traj: Optional[Array],
    additional_iterations: int,
    rollout_count: int,
    obstacle_circles: Sequence[Tuple[Array, float]],
    goal: Array,
    cfg: Any,
    rng: np.random.Generator,
    packed_obstacles=None,
) -> Dict[str, object]:
    current = np.asarray(current, dtype=np.float64)
    if current_traj is None or not np.isfinite(current_cost):
        current_cost, current_traj, _ = _single_sequence_evaluation(
            model, x_current, current, obstacle_circles, goal, cfg, packed_obstacles
        )
    history = [float(current_cost)]
    accepted = 0
    last_costs = np.asarray([current_cost], dtype=np.float64)
    for _ in range(max(0, int(additional_iterations))):
        candidate, costs, _ = _standard_mppi_refinement_pass(
            model, x_current, current, rollout_count, obstacle_circles, goal, cfg, rng, packed_obstacles
        )
        last_costs = costs
        current, current_cost, current_traj, improved = _accept_improving_candidate(
            model, x_current, current, current_cost, current_traj, candidate,
            obstacle_circles, goal, cfg, packed_obstacles
        )
        accepted += int(improved)
        history.append(float(current_cost))
    return {
        "sequence": current,
        "cost": float(current_cost),
        "trajectory": np.asarray(current_traj),
        "accepted": int(accepted),
        "history": history,
        "last_costs": last_costs,
    }


def standard_mppi_step(
    model: Any,
    x_current: Array,
    obstacles: Sequence[Any],
    goal: Array,
    cfg: Any,
    rng: np.random.Generator,
    *,
    obstacle_circles: Optional[List[Tuple[Array, float]]] = None,
    record_optimal_traj: bool = True,
) -> Tuple[Array, Dict[str, object]]:
    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)
    packed_obstacles = _prepare_model_obstacles(model, obstacle_circles)
    nominal = model.clip_control_batch(
        np.asarray(model.nominal_controls_to_goal(x_current, goal, cfg), dtype=np.float64)[None, :, :], cfg
    )[0]
    current_cost, current_traj, _ = _single_sequence_evaluation(
        model, x_current, nominal, obstacle_circles, goal, cfg, packed_obstacles
    )
    history = [float(current_cost)]
    accepted = 0
    last_costs = np.asarray([current_cost], dtype=np.float64)

    current = nominal
    for _ in range(max(1, int(cfg.mppi_iterations))):
        candidate, costs, _ = _standard_mppi_refinement_pass(
            model, x_current, current, int(cfg.num_rollouts), obstacle_circles, goal, cfg, rng, packed_obstacles
        )
        last_costs = costs
        current, current_cost, current_traj, improved = _accept_improving_candidate(
            model, x_current, current, current_cost, current_traj, candidate, obstacle_circles, goal, cfg, packed_obstacles
        )
        accepted += int(improved)
        history.append(float(current_cost))

    info: Dict[str, object] = {
        "cost_min": float(np.min(last_costs)) if last_costs.size else float("inf"),
        "cost_mean": _finite_cost_mean(last_costs),
        "optimal_traj": np.asarray(current_traj).copy() if record_optimal_traj else None,
        "planned_control_sequence": np.asarray(current).copy(),
        "selected_rollout_mode_index": None,
        "mppi_iterations": int(cfg.mppi_iterations),
        "mppi_accepted_iterations": int(accepted),
        "mppi_cost_history": history,
        "mppi_initial_cost": float(history[0]),
        "mppi_final_cost": float(current_cost),
    }
    return np.asarray(current[0]).copy(), info

def planner_ilqr_step(
    model: Any,
    x_current: Array,
    global_modes: List[MPPIHomotopyMode],
    obstacles: Sequence[Any],
    goal: Array,
    cfg: Any,
    *,
    progress_by_mode: Optional[Dict[str, int]] = None,
    state_history: Optional[Sequence[Array]] = None,
    record_optimal_traj: bool = True,
    packed_mode_bank: Optional[PackedModeBank] = None,
    packed_polygons: Optional[Tuple[Array, Array]] = None,
) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    """Planner-prior + iLQR ablation with no MPPI sampling or weighting."""
    if not global_modes:
        raise ValueError("Planner iLQR requires at least one planner homotopy mode.")

    progress = {} if progress_by_mode is None else dict(progress_by_mode)
    (
        active_indices,
        local_modes,
        active_clearances,
        all_local_clearances,
        new_progress,
    ) = localize_all_feasible_mean_modes(
        global_modes,
        x_current,
        obstacles,
        cfg,
        progress,
        state_history=state_history,
        packed_mode_bank=packed_mode_bank,
        packed_polygons=packed_polygons,
    )

    fallback = None
    if not active_indices:
        position = np.asarray(x_current[:2], dtype=np.float64)
        distances = np.asarray(
            [
                float(np.min(np.linalg.norm(np.asarray(mode.mean_path, dtype=np.float64)[:, :2] - position[None, :], axis=1)))
                for mode in global_modes
            ],
            dtype=np.float64,
        )
        selected_global_index = int(np.argmin(distances))
        selected_local_mode, selected_index = localize_mode_for_state_with_index(
            global_modes[selected_global_index],
            x_current,
            cfg.horizon,
            step_distance=prior_preview_step_distance(cfg),
        )
        new_progress[str(global_modes[selected_global_index].signature)] = selected_index
        retained_indices = [selected_global_index]
        retained_clearances: List[float] = []
        fallback = "nearest_prior_ilqr"
    else:
        active_global_modes = [global_modes[index] for index in active_indices]
        probabilities = renormalized_mode_probabilities(active_global_modes)
        selected_local_index = int(np.argmax(probabilities))
        selected_global_index = int(active_indices[selected_local_index])
        selected_local_mode = local_modes[selected_local_index]
        retained_indices = [int(index) for index in active_indices]
        retained_clearances = [float(value) for value in active_clearances]

    # Recompute the localized-path iLQR nominal from its deterministic path-tracking
    # initialization every control step. Shifted cached nominals alter the standalone
    # iLQR closed-loop behavior, especially during terminal braking near the goal.
    nominal, _ = nominal_controls_and_arc_positions(
        model, x_current, selected_local_mode.mean_path, cfg, selected_local_mode.cov_blocks
    )
    clip_sequence = getattr(model, "clip_control_sequence", None)
    if clip_sequence is not None:
        nominal = np.asarray(clip_sequence(nominal, cfg), dtype=np.float64)
    else:
        nominal = model.clip_control_batch(np.asarray(nominal, dtype=np.float64)[None, :, :], cfg)[0]
    nominal_states = model.rollout_single(x_current, nominal, cfg)

    info: Dict[str, object] = {
        "rep_type": None,
        "mode_selection": True,
        "mode_selection_policy": "highest_prior_probability_ilqr_only",
        "selected_mode_index": selected_global_index,
        "selected_rollout_mode_index": selected_global_index,
        "active_mode_count": int(len(active_indices)),
        "suppressed_mode_count": int(len(global_modes) - len(active_indices)),
        "candidate_mode_count": int(len(global_modes)),
        "nearby_mode_count": int(len(active_indices)),
        "mode_clearances": [float(value) for value in all_local_clearances],
        "retained_mode_indices": retained_indices,
        "retained_mode_clearances": retained_clearances,
        "active_mode_probabilities": [
            float(global_modes[index].probability) for index in active_indices
        ],
        "renormalized_mode_probabilities": (
            renormalized_mode_probabilities([global_modes[index] for index in active_indices]).tolist()
            if active_indices else []
        ),
        "rollouts_by_mode": [],
        "optimal_traj": np.asarray(nominal_states, dtype=np.float64).copy() if record_optimal_traj else None,
        "nominal_ilqr_traj": np.asarray(nominal_states, dtype=np.float64).copy() if record_optimal_traj else None,
        "planned_control_sequence": nominal,
    }
    if fallback is not None:
        info["mode_filter_fallback"] = fallback
    return nominal[0].copy(), info, new_progress



def control_bank_step(
    model: Any,
    x_current: Array,
    global_modes: List[MPPIHomotopyMode],
    obstacles: Sequence[Any],
    goal: Array,
    cfg: Any,
    *,
    progress_by_mode: Optional[Dict[str, int]] = None,
    state_history: Optional[Sequence[Array]] = None,
    obstacle_circles: Optional[List[Tuple[Array, float]]] = None,
    cached_mode_clearances: Optional[Array] = None,
    record_optimal_traj: bool = True,
    packed_mode_bank: Optional[PackedModeBank] = None,
) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    """Pure empirical control-bank ablation.

    Up to ``cfg.num_rollouts`` unique empirical trajectories are converted to
    controls with iLQR and evaluated once. If fewer unique trajectories are
    available, all available trajectories are evaluated. The minimum-cost
    collision-free sequence is executed directly. There is no MPPI weighting,
    stochastic perturbation, or additional MPPI refinement.
    """
    progress = {} if progress_by_mode is None else dict(progress_by_mode)
    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)
    packed_obstacles = _prepare_model_obstacles(model, obstacle_circles)
    total_budget = max(1, int(cfg.num_rollouts))

    if cached_mode_clearances is None:
        cached_mode_clearances = cached_mode_mean_clearances(
            model, global_modes, obstacle_circles, cfg
        )
    clearances = np.asarray(cached_mode_clearances, dtype=np.float64)

    nearby = nearby_mode_indices(
        global_modes, x_current, cfg, clearances, packed_mode_bank=packed_mode_bank
    )
    # Keep the usual local mode preference, but expand the bank if necessary.
    # If fewer unique trajectories exist than the requested rollout budget, use
    # every available empirical trajectory rather than repeating samples.
    active_indices = [
        int(index) for index in nearby if len(global_modes[int(index)].sample_paths or []) > 0
    ]
    available = sum(len(global_modes[index].sample_paths or []) for index in active_indices)
    if available < total_budget:
        for index, mode in enumerate(global_modes):
            if index in active_indices or not (mode.sample_paths or []):
                continue
            active_indices.append(index)
            available += len(mode.sample_paths or [])
            if available >= total_budget:
                break
    if available <= 0:
        raise ValueError("Control bank contains no empirical trajectories to evaluate.")

    eval_budget = min(total_budget, available)

    active_modes = [global_modes[index] for index in active_indices]
    counts = balanced_unique_control_bank_counts(eval_budget, active_modes)

    offsets = np.zeros(len(active_modes) + 1, dtype=np.int64)
    for mode_index, count in enumerate(counts):
        offsets[mode_index + 1] = offsets[mode_index] + int(count)

    all_controls = np.empty((eval_budget, cfg.horizon, 2), dtype=np.float64)
    for mode_index, mode in enumerate(active_modes):
        count = int(counts[mode_index])
        if count <= 0:
            continue
        start_id = int(offsets[mode_index])
        end_id = int(offsets[mode_index + 1])
        all_controls[start_id:end_id] = sample_exact_control_bank(
            model, x_current, mode, count, cfg
        )

        # Progress is diagnostic for the bank, but keeping it updated preserves
        # the same receding-horizon bookkeeping used by the other prior variants.
        _, progress_index = localize_mode_for_state_with_index(
            mode, x_current, cfg.horizon, step_distance=prior_preview_step_distance(cfg)
        )
        progress[str(mode.signature)] = int(progress_index)

    raw_costs, colliding, _ = _rollout_costs_and_collisions(
        model, x_current, all_controls, obstacle_circles, goal, cfg, packed_obstacles
    )
    costs = np.asarray(raw_costs, dtype=np.float64)
    costs[np.asarray(colliding, dtype=bool)] = np.inf

    finite_ids = np.flatnonzero(np.isfinite(costs))
    selected_mode_global_index: Optional[int] = None
    if finite_ids.size:
        best_id = int(finite_ids[np.argmin(costs[finite_ids])])
        planned_sequence = np.asarray(all_controls[best_id], dtype=np.float64)
        best_cost, best_traj, _ = _single_sequence_evaluation(
            model, x_current, planned_sequence, obstacle_circles, goal, cfg, packed_obstacles
        )
        local_mode_index = int(np.searchsorted(offsets[1:], best_id, side="right"))
        selected_mode_global_index = int(active_indices[local_mode_index])
        fallback = None
    else:
        # No bank member is feasible. Do not fall back to Standard MPPI, since
        # that would contaminate the Control-bank ablation. Return a neutral
        # sequence and let the trial fail naturally if the bank cannot recover.
        planned_sequence = np.zeros((cfg.horizon, 2), dtype=np.float64)
        clip_sequence = getattr(model, "clip_control_sequence", None)
        if clip_sequence is not None:
            planned_sequence = np.asarray(clip_sequence(planned_sequence, cfg), dtype=np.float64)
        else:
            planned_sequence = model.clip_control_batch(planned_sequence[None, :, :], cfg)[0]
        best_cost, best_traj, _ = _single_sequence_evaluation(
            model, x_current, planned_sequence, obstacle_circles, goal, cfg, packed_obstacles
        )
        fallback = "no_feasible_control_bank_candidate"

    probabilities = renormalized_mode_probabilities(active_modes)
    info = {
        "cost_min": float(np.min(costs)),
        "cost_mean": _finite_cost_mean(costs),
        "soft_value": float(softmin_score(costs, cfg)),
        "rep_type": int(REP_CONTROL_BANK),
        "mode_selection": False,
        "mode_selection_policy": "empirical_control_bank_minimum_cost",
        "selected_mode_index": None,
        "selected_rollout_mode_index": selected_mode_global_index,
        "rollout_budget_requested": int(total_budget),
        "rollout_budget_total": int(eval_budget),
        "control_bank_available": int(available),
        "rollouts_by_mode": [int(value) for value in counts],
        "active_mode_count": int(len(active_modes)),
        "suppressed_mode_count": int(len(global_modes) - len(active_modes)),
        "candidate_mode_count": int(len(active_modes)),
        "nearby_mode_count": int(len(nearby)),
        "mode_clearances": clearances.tolist(),
        "retained_mode_indices": [int(index) for index in active_indices],
        "retained_mode_clearances": [float(clearances[index]) for index in active_indices],
        "active_mode_probabilities": [float(global_modes[index].probability) for index in active_indices],
        "renormalized_mode_probabilities": probabilities.tolist(),
        "optimal_traj": np.asarray(best_traj).copy() if record_optimal_traj else None,
        "nominal_ilqr_traj": None,
        "planned_control_sequence": planned_sequence,
        "mppi_iterations": 0,
        "mppi_accepted_iterations": 0,
        "mppi_cost_history": [float(best_cost)],
        "mppi_initial_cost": float(best_cost),
        "mppi_final_cost": float(best_cost),
    }
    if fallback is not None:
        info["mode_filter_fallback"] = fallback
    return planned_sequence[0].copy(), info, progress

def stable_swarm_mppi_step(
    model: Any,
    x_current: Array,
    global_modes: List[MPPIHomotopyMode],
    obstacles: Sequence[Any],
    goal: Array,
    cfg: Any,
    rng: np.random.Generator,
    *,
    rep_type: int,
    progress_by_mode: Optional[Dict[str, int]],
    state_history: Optional[Sequence[Array]] = None,
    obstacle_circles: Optional[List[Tuple[Array, float]]] = None,
    cached_mode_clearances: Optional[Array] = None,
    record_optimal_traj: Optional[bool] = None,
    packed_mode_bank: Optional[PackedModeBank] = None,
    packed_polygons: Optional[Tuple[Array, Array]] = None,
) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    if rep_type not in {REP_GAUSSIAN, REP_CORRIDOR, REP_CONTROL_BANK, REP_SENSITIVITY_PROJECTED_GAUSSIAN}:
        raise ValueError(f"Unsupported pooled proposal representation: {rep_type}")

    progress = {} if progress_by_mode is None else dict(progress_by_mode)
    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)

    if rep_type == REP_CONTROL_BANK:
        return control_bank_step(
            model,
            x_current,
            global_modes,
            obstacles,
            goal,
            cfg,
            progress_by_mode=progress,
            state_history=state_history,
            obstacle_circles=obstacle_circles,
            cached_mode_clearances=cached_mode_clearances,
            record_optimal_traj=bool(record_optimal_traj),
            packed_mode_bank=packed_mode_bank,
        )

    packed_obstacles = _prepare_model_obstacles(model, obstacle_circles)
    # MPPI priors are intentionally recomputed from the localized path each step.
    # Shifted iLQR warm starts changed closed-loop behavior for the quadrotor models.
    total_budget = max(1, int(cfg.num_rollouts))
    compressed_mean_rep = rep_type in {REP_GAUSSIAN, REP_CORRIDOR, REP_SENSITIVITY_PROJECTED_GAUSSIAN}

    if compressed_mean_rep:
        (active_global_indices, active_local_modes, active_clearances, all_local_clearances, new_progress) = localize_all_feasible_mean_modes(
            global_modes, x_current, obstacles, cfg, progress, state_history=state_history,
            packed_mode_bank=packed_mode_bank, packed_polygons=packed_polygons
        )
        if not active_global_indices:
            control, info = standard_mppi_step(
                model, x_current, obstacles, goal, cfg, rng,
                obstacle_circles=obstacle_circles, record_optimal_traj=bool(record_optimal_traj)
            )
            info.update({
                "active_mode_count": 0, "suppressed_mode_count": int(len(global_modes)),
                "candidate_mode_count": int(len(global_modes)), "nearby_mode_count": int(len(global_modes)),
                "mode_clearances": list(all_local_clearances), "mode_filter_fallback": "standard_mppi",
                "retained_mode_indices": [], "retained_mode_clearances": [], "active_mode_probabilities": [],
                "renormalized_mode_probabilities": [], "rollouts_by_mode": [], "selected_rollout_mode_index": None,
            })
            return control, info, new_progress
        active_global_modes = [global_modes[index] for index in active_global_indices]
        probabilities = renormalized_mode_probabilities(active_global_modes)
        counts = probability_proportional_rollout_counts(total_budget, probabilities)
        selection_policy = "all_localized_collision_free_pi_weighted_iterative"
    else:
        if cached_mode_clearances is None:
            cached_mode_clearances = cached_mode_mean_clearances(model, global_modes, obstacle_circles, cfg)
        nearby_indices = nearby_mode_indices(global_modes, x_current, cfg, cached_mode_clearances, packed_mode_bank=packed_mode_bank)
        if not nearby_indices:
            control, info = standard_mppi_step(
                model, x_current, obstacles, goal, cfg, rng,
                obstacle_circles=obstacle_circles, record_optimal_traj=bool(record_optimal_traj)
            )
            info.update({
                "active_mode_count": 0, "suppressed_mode_count": int(len(global_modes)), "nearby_mode_count": 0,
                "candidate_mode_count": 0, "mode_clearances": np.asarray(cached_mode_clearances).tolist(),
                "mode_filter_fallback": "standard_mppi", "retained_mode_indices": [], "retained_mode_clearances": [],
                "active_mode_probabilities": [], "renormalized_mode_probabilities": [], "rollouts_by_mode": [],
                "selected_rollout_mode_index": None,
            })
            return control, info, progress
        candidate_global_modes = [global_modes[index] for index in nearby_indices]
        local_modes: List[MPPIHomotopyMode] = []
        new_progress = dict(progress)
        for mode in candidate_global_modes:
            key = str(mode.signature)
            local_mode, index = localize_mode_for_state_with_index(
                mode, x_current, cfg.horizon, step_distance=prior_preview_step_distance(cfg)
            )
            local_modes.append(local_mode); new_progress[key] = index
        active_count = min(len(local_modes), total_budget)
        active_local_modes = local_modes[:active_count]
        active_global_modes = candidate_global_modes[:active_count]
        active_global_indices = nearby_indices[:active_count]
        active_clearances = np.asarray(cached_mode_clearances, dtype=np.float64)[active_global_indices].tolist()
        all_local_clearances = np.asarray(cached_mode_clearances, dtype=np.float64).tolist()
        probabilities = renormalized_mode_probabilities(active_global_modes)
        counts = balanced_rollout_counts(total_budget, active_count)
        selection_policy = "nearby_balanced_control_bank_iterative"

    active_count = len(active_global_modes)
    if sum(int(count) for count in counts) != total_budget:
        raise RuntimeError(
            f"Internal rollout allocation error: allocated {sum(int(count) for count in counts)} of {total_budget} rollouts."
        )

    all_controls = np.empty((total_budget, cfg.horizon, 2), dtype=np.float64)
    nominal_controls_by_mode: List[Array] = [np.zeros((cfg.horizon, 2), dtype=np.float64) for _ in range(active_count)]

    # A: cold iLQR mode priors are independent. Solve all of them concurrently,
    # then draw proposal noise serially in the original mode order so RNG behavior
    # remains deterministic. SPG-capable models also return the final A/B sequence.
    nominal_solutions = parallel_mode_nominals(
        model, x_current, active_local_modes, cfg,
        need_jacobians=(rep_type == REP_SENSITIVITY_PROJECTED_GAUSSIAN),
    )

    offsets = np.empty(active_count + 1, dtype=np.int64)
    offsets[0] = 0
    for mode_index in range(active_count):
        offsets[mode_index + 1] = offsets[mode_index] + int(counts[mode_index])

    for mode_index, local_mode in enumerate(active_local_modes):
        start_id = int(offsets[mode_index])
        end_id = int(offsets[mode_index + 1])
        count = end_id - start_id
        if count <= 0:
            continue
        global_mode = active_global_modes[mode_index]
        ilqr_nominal, control_positions, A, B = nominal_solutions[mode_index]
        if rep_type == REP_GAUSSIAN:
            controls = _sample_gaussian_from_nominal(
                model, local_mode, ilqr_nominal, control_positions, count, cfg, rng
            )
        elif rep_type == REP_SENSITIVITY_PROJECTED_GAUSSIAN:
            controls = _sample_spg_from_nominal(
                model, x_current, local_mode, ilqr_nominal, control_positions, A, B, count, cfg, rng
            )
        elif rep_type == REP_CONTROL_BANK:
            controls = sample_exact_control_bank(
                model, x_current, global_mode, ilqr_nominal, count, cfg
            )
        else:
            controls = sample_controls_around_nominal(model, ilqr_nominal, count, cfg, rng)
        nominal_controls_by_mode[mode_index] = np.asarray(ilqr_nominal, dtype=np.float64)
        all_controls[start_id:end_id] = controls

    # B: evaluate the complete rollout budget in one Numba call. This avoids one
    # parallel-kernel launch per mode and gives prange the full batch to distribute.
    all_costs, collision_mask, _ = _rollout_costs_and_collisions(
        model, x_current, all_controls, obstacle_circles, goal, cfg, packed_obstacles
    )
    all_costs[collision_mask] = np.inf

    best_cost = 1e309
    best_mode_global_index: Optional[int] = None
    for mode_index in range(active_count):
        start_id = int(offsets[mode_index])
        end_id = int(offsets[mode_index + 1])
        if end_id <= start_id:
            continue
        costs = all_costs[start_id:end_id]
        local_best = int(np.argmin(costs))
        local_best_cost = float(costs[local_best])
        if np.isfinite(local_best_cost) and local_best_cost < best_cost:
            best_cost = local_best_cost
            best_mode_global_index = int(active_global_indices[mode_index])

    first_candidate = mppi_weighted_control_sequence(model, all_costs, all_controls, cfg)
    if best_mode_global_index is None:
        baseline_local_index = int(np.argmax(probabilities)) if len(probabilities) else 0
        best_mode_global_index = int(active_global_indices[baseline_local_index])
    else:
        baseline_local_index = next(
            (i for i,g in enumerate(active_global_indices) if int(g)==int(best_mode_global_index)), 0
        )
    baseline = np.asarray(nominal_controls_by_mode[baseline_local_index], dtype=np.float64)
    baseline_cost, baseline_traj, _ = _single_sequence_evaluation(
        model, x_current, baseline, obstacle_circles, goal, cfg, packed_obstacles
    )
    current, current_cost, current_traj, first_improved = _accept_improving_candidate(
        model, x_current, baseline, baseline_cost, baseline_traj, first_candidate, obstacle_circles, goal, cfg, packed_obstacles
    )
    history = [float(baseline_cost), float(current_cost)]
    accepted = int(first_improved)

    extra = _run_additional_mppi_refinements(
        model, x_current, current, current_cost, current_traj, max(0, int(cfg.mppi_iterations)-1), total_budget,
        obstacle_circles, goal, cfg, rng, packed_obstacles
    )
    if len(extra["history"]) > 1:
        history.extend(list(extra["history"])[1:])
    accepted += int(extra["accepted"])
    planned_sequence = np.asarray(extra["sequence"], dtype=np.float64)
    final_cost = float(extra["cost"])
    final_traj = np.asarray(extra["trajectory"]).copy()

    nominal_ilqr_traj = None
    if record_optimal_traj:
        nominal_ilqr_traj = np.asarray(baseline_traj).copy()

    info = {
        "cost_min": float(np.min(all_costs)), "cost_mean": _finite_cost_mean(all_costs),
        "soft_value": float(softmin_score(all_costs, cfg)), "rep_type": int(rep_type),
        "mode_selection": False, "mode_selection_policy": selection_policy, "selected_mode_index": None,
        "rollout_budget_total": int(total_budget * int(cfg.mppi_iterations)),
        "rollouts_by_mode": [int(count) for count in counts], "active_mode_count": active_count,
        "suppressed_mode_count": int(len(global_modes)-active_count),
        "candidate_mode_count": int(len(global_modes) if compressed_mean_rep else active_count),
        "nearby_mode_count": int(len(global_modes) if compressed_mean_rep else active_count),
        "mode_clearances": [float(value) for value in all_local_clearances],
        "retained_mode_indices": [int(index) for index in active_global_indices],
        "retained_mode_clearances": [float(value) for value in active_clearances],
        "active_mode_probabilities": [float(global_modes[index].probability) for index in active_global_indices],
        "renormalized_mode_probabilities": probabilities.tolist(), "selected_rollout_mode_index": best_mode_global_index,
        "optimal_traj": final_traj if record_optimal_traj else None, "nominal_ilqr_traj": nominal_ilqr_traj,
        "planned_control_sequence": planned_sequence, "mppi_iterations": int(cfg.mppi_iterations),
        "mppi_accepted_iterations": int(accepted), "mppi_cost_history": history,
        "mppi_initial_cost": float(baseline_cost), "mppi_final_cost": float(final_cost),
    }
    return planned_sequence[0].copy(), info, new_progress

def mode_selecting_stable_mppi_step(
    model: Any,
    x_current: Array,
    global_modes: List[MPPIHomotopyMode],
    obstacles: Sequence[Any],
    goal: Array,
    cfg: Any,
    rng: np.random.Generator,
    *,
    rep_type: int,
    progress_by_mode: Optional[Dict[str, int]] = None,
    state_history: Optional[Sequence[Array]] = None,
    obstacle_circles: Optional[List[Tuple[Array, float]]] = None,
    record_optimal_traj: bool = True,
    packed_mode_bank: Optional[PackedModeBank] = None,
    packed_polygons: Optional[Tuple[Array, Array]] = None,
) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    if rep_type not in {REP_GAUSSIAN, REP_CORRIDOR}:
        raise ValueError("Mode-selecting MPPI supports only Gaussian or corridor proposals.")

    progress = {} if progress_by_mode is None else dict(progress_by_mode)
    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)
    packed_obstacles = _prepare_model_obstacles(model, obstacle_circles)
    (
        active_indices,
        local_modes,
        active_clearances,
        all_local_clearances,
        new_progress,
    ) = localize_all_feasible_mean_modes(
        global_modes,
        x_current,
        obstacles,
        cfg,
        progress,
        state_history=state_history,
        packed_mode_bank=packed_mode_bank,
        packed_polygons=packed_polygons,
    )

    if not active_indices:
        control, info = standard_mppi_step(
            model,
            x_current,
            obstacles,
            goal,
            cfg,
            rng,
            obstacle_circles=obstacle_circles,
            record_optimal_traj=record_optimal_traj,
        )
        info.update(
            {
                "active_mode_count": 0,
                "suppressed_mode_count": int(len(global_modes)),
                "candidate_mode_count": int(len(global_modes)),
                "mode_filter_fallback": "standard_mppi",
                "mode_clearances": [float(value) for value in all_local_clearances],
                "retained_mode_indices": [],
                "retained_mode_clearances": [],
                "active_mode_probabilities": [],
                "renormalized_mode_probabilities": [],
                "selected_mode_index": None,
                "selected_rollout_mode_index": None,
            }
        )
        return control, info, new_progress

    active_global_modes = [global_modes[index] for index in active_indices]
    probabilities = renormalized_mode_probabilities(active_global_modes)
    configured = int(cfg.mode_select_rollouts_per_mode)
    rollouts_per_mode = configured if configured > 0 else max(1, int(cfg.num_rollouts))
    completed: List[dict[str, Any]] = []
    mode_count = len(local_modes)
    nominal_solutions = parallel_mode_nominals(model, x_current, local_modes, cfg)
    pooled_controls = np.empty((mode_count * rollouts_per_mode, cfg.horizon, 2), dtype=np.float64)

    # Generate noise in mode order to preserve deterministic RNG semantics.
    for local_index, (original_index, local_mode) in enumerate(zip(active_indices, local_modes)):
        ilqr_nominal, control_positions, _, _ = nominal_solutions[local_index]
        start_id = local_index * rollouts_per_mode
        end_id = start_id + rollouts_per_mode
        if rep_type == REP_GAUSSIAN:
            controls = _sample_gaussian_from_nominal(
                model, local_mode, ilqr_nominal, control_positions, rollouts_per_mode, cfg, rng
            )
        else:
            controls = sample_controls_around_nominal(
                model, ilqr_nominal, rollouts_per_mode, cfg, rng
            )
        pooled_controls[start_id:end_id] = controls

    pooled_costs, pooled_collision, _ = _rollout_costs_and_collisions(
        model, x_current, pooled_controls, obstacle_circles, goal, cfg, packed_obstacles
    )
    pooled_costs[pooled_collision] = np.inf

    for local_index, original_index in enumerate(active_indices):
        global_mode = global_modes[int(original_index)]
        ilqr_nominal = nominal_solutions[local_index][0]
        start_id = local_index * rollouts_per_mode
        end_id = start_id + rollouts_per_mode
        controls = pooled_controls[start_id:end_id]
        costs = pooled_costs[start_id:end_id]
        collisions = pooled_collision[start_id:end_id]
        feasible_count = int(np.count_nonzero(~collisions))
        first_candidate = mppi_weighted_control_sequence(model, costs, controls, cfg)
        completed.append(
            {
                "score": float(softmin_score(costs, cfg)),
                "mode_index": int(original_index),
                "signature": str(global_mode.signature),
                "probability": float(global_mode.probability),
                "feasible_count": feasible_count,
                "cost_min": float(np.min(costs)),
                "cost_mean": _finite_cost_mean(costs),
                "first_candidate": np.asarray(first_candidate, dtype=np.float64).copy(),
                "nominal_controls": np.asarray(ilqr_nominal, dtype=np.float64),
            }
        )

    feasible = [record for record in completed if record["feasible_count"] > 0]
    best = min(feasible if feasible else completed, key=lambda record: record["score"])

    baseline = np.asarray(best["nominal_controls"], dtype=np.float64)
    baseline_cost, baseline_traj, _ = _single_sequence_evaluation(
        model, x_current, baseline, obstacle_circles, goal, cfg, packed_obstacles
    )
    current, current_cost, current_traj, first_improved = _accept_improving_candidate(
        model,
        x_current,
        baseline,
        baseline_cost,
        baseline_traj,
        np.asarray(best["first_candidate"], dtype=np.float64),
        obstacle_circles,
        goal,
        cfg,
        packed_obstacles,
    )
    history = [float(baseline_cost), float(current_cost)]
    accepted = int(first_improved)

    extra = _run_additional_mppi_refinements(
        model,
        x_current,
        current,
        current_cost,
        current_traj,
        max(0, int(cfg.mppi_iterations) - 1),
        rollouts_per_mode,
        obstacle_circles,
        goal,
        cfg,
        rng,
        packed_obstacles,
    )
    if len(extra["history"]) > 1:
        history.extend(list(extra["history"])[1:])
    accepted += int(extra["accepted"])
    planned_sequence = np.asarray(extra["sequence"], dtype=np.float64)
    final_cost = float(extra["cost"])
    final_traj = np.asarray(extra["trajectory"]).copy()

    nominal_ilqr_traj: Optional[Array] = None
    if record_optimal_traj:
        nominal_ilqr_traj = np.asarray(baseline_traj, dtype=np.float64).copy()

    info = {
        "cost_min": best["cost_min"],
        "cost_mean": best["cost_mean"],
        "soft_value": best["score"],
        "selected_mode_index": best["mode_index"],
        "selected_mode_signature": best["signature"],
        "selected_mode_probability": best["probability"],
        "selected_rollout_mode_index": best["mode_index"],
        "rep_type": int(rep_type),
        "mode_selection": True,
        "mode_selection_policy": "one_pass_mode_selection_then_iterative_improving_mppi",
        "rollout_budget_per_mode": int(rollouts_per_mode),
        "rollout_budget_total": int(
            rollouts_per_mode * len(completed)
            + max(0, int(cfg.mppi_iterations) - 1) * rollouts_per_mode
        ),
        "rollouts_by_mode": [
            int(
                rollouts_per_mode
                + (max(0, int(cfg.mppi_iterations) - 1) * rollouts_per_mode
                   if int(record["mode_index"]) == int(best["mode_index"]) else 0)
            )
            for record in completed
        ],
        "active_mode_count": len(completed),
        "suppressed_mode_count": int(len(global_modes) - len(completed)),
        "candidate_mode_count": int(len(global_modes)),
        "mode_clearances": [float(value) for value in all_local_clearances],
        "retained_mode_indices": [int(index) for index in active_indices],
        "retained_mode_clearances": [float(value) for value in active_clearances],
        "active_mode_probabilities": [
            float(global_modes[index].probability) for index in active_indices
        ],
        "renormalized_mode_probabilities": probabilities.tolist(),
        "optimal_traj": final_traj if record_optimal_traj else None,
        "nominal_ilqr_traj": nominal_ilqr_traj,
        "planned_control_sequence": planned_sequence,
        "mppi_iterations": int(cfg.mppi_iterations),
        "mppi_accepted_iterations": int(accepted),
        "mppi_cost_history": history,
        "mppi_initial_cost": float(baseline_cost),
        "mppi_final_cost": float(final_cost),
    }
    return planned_sequence[0].copy(), info, new_progress

def run_controller(
    model: Any,
    variant: ControllerVariant,
    modes: list[MPPIHomotopyMode],
    base_obstacles: Sequence[Any],
    blockers: Sequence[Any],
    scene: Scene,
    *,
    seed: int,
    trigger_progress: Optional[float],
    blocker_active_from_start: bool,
    max_steps: int,
    cfg: Any,
    record: bool = True,
) -> SimulationResult:
    supported = getattr(model, "SUPPORTED_VARIANTS", None)
    if supported is not None and variant not in supported:
        raise ValueError(f"{getattr(model, 'MODEL_NAME', 'model')} does not support {variant.value}.")

    rng = np.random.default_rng(seed)
    if getattr(model, "INITIAL_POSE_USES_CONFIG", False):
        state = model.initial_pose(scene.start, scene.goal, cfg)
    else:
        state = model.initial_pose(scene.start, scene.goal)
    states = [state.copy()]
    controls: list[Array] = []
    infos: list[dict[str, object]] = []
    obstacle_history: list[list[Any]] = []
    previous_control: Optional[Array] = None
    reached_goal = bool(model.goal_reached(state, scene.goal, cfg))
    progress_by_mode: Dict[str, int] = {}

    blockers = list(blockers)
    activation_step: Optional[int] = 0 if blocker_active_from_start else None
    base_circles = obstacle_bounding_circles(base_obstacles)
    blocked_obstacles = list(base_obstacles) + blockers
    blocked_circles = obstacle_bounding_circles(blocked_obstacles) if blockers else base_circles
    packed_mode_bank = pack_mode_bank(modes)
    base_polygons_packed = obstacle_polygons_to_padded_arrays(base_obstacles)
    blocked_polygons_packed = obstacle_polygons_to_padded_arrays(blocked_obstacles) if blockers else base_polygons_packed
    control_bank_clearance_cache: Dict[Tuple[Tuple[float, float, float], ...], Array] = {}

    started = time.perf_counter()
    for step in range(max_steps):
        if (
            activation_step is None
            and blockers
            and trigger_progress is not None
            and spatial_progress_along_start_goal(state, scene.start, scene.goal) >= trigger_progress
        ):
            activation_step = step
        wall_active = activation_step is not None and step >= activation_step
        active_obstacles = blocked_obstacles if wall_active else list(base_obstacles)
        active_circles = blocked_circles if wall_active else base_circles
        active_packed_polygons = blocked_polygons_packed if wall_active else base_polygons_packed

        if record:
            obstacle_history.append(list(active_obstacles))

        if variant == ControllerVariant.PLANNER_ILQR:
            control, info, progress_by_mode = planner_ilqr_step(
                model,
                state,
                modes,
                active_obstacles,
                scene.goal,
                cfg,
                progress_by_mode=progress_by_mode,
                state_history=states,
                record_optimal_traj=record,
                packed_mode_bank=packed_mode_bank,
                packed_polygons=active_packed_polygons,
            )
        elif variant == ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI:
            control, info, progress_by_mode = stable_swarm_mppi_step(
                model,
                state,
                modes,
                active_obstacles,
                scene.goal,
                cfg,
                rng,
                rep_type=REP_SENSITIVITY_PROJECTED_GAUSSIAN,
                progress_by_mode=progress_by_mode,
                state_history=states,
                obstacle_circles=active_circles,
                record_optimal_traj=record,
                packed_mode_bank=packed_mode_bank,
                packed_polygons=active_packed_polygons,
            )
        elif variant == ControllerVariant.GAUSSIAN_PRIOR_MPPI:
            control, info, progress_by_mode = stable_swarm_mppi_step(
                model,
                state,
                modes,
                active_obstacles,
                scene.goal,
                cfg,
                rng,
                rep_type=REP_GAUSSIAN,
                progress_by_mode=progress_by_mode,
                state_history=states,
                obstacle_circles=active_circles,
                record_optimal_traj=record,
                packed_mode_bank=packed_mode_bank,
                packed_polygons=active_packed_polygons,
            )
        elif variant == ControllerVariant.CORRIDOR_PRIOR_MPPI:
            control, info, progress_by_mode = stable_swarm_mppi_step(
                model,
                state,
                modes,
                active_obstacles,
                scene.goal,
                cfg,
                rng,
                rep_type=REP_CORRIDOR,
                progress_by_mode=progress_by_mode,
                state_history=states,
                obstacle_circles=active_circles,
                record_optimal_traj=record,
                packed_mode_bank=packed_mode_bank,
                packed_polygons=active_packed_polygons,
            )
        elif variant == ControllerVariant.CONTROL_BANK_MPPI:
            obstacle_key = obstacle_configuration_key(active_circles)
            if obstacle_key not in control_bank_clearance_cache:
                control_bank_clearance_cache[obstacle_key] = cached_mode_mean_clearances(
                    model, modes, active_circles, cfg
                )
            mode_clearances = control_bank_clearance_cache[obstacle_key]
            control, info, progress_by_mode = stable_swarm_mppi_step(
                model,
                state,
                modes,
                active_obstacles,
                scene.goal,
                cfg,
                rng,
                rep_type=REP_CONTROL_BANK,
                progress_by_mode=progress_by_mode,
                state_history=states,
                obstacle_circles=active_circles,
                cached_mode_clearances=mode_clearances,
                record_optimal_traj=record,
                packed_mode_bank=packed_mode_bank,
                packed_polygons=active_packed_polygons,
            )
        elif variant == ControllerVariant.MODE_SELECTING_GAUSSIAN_MPPI:
            control, info, progress_by_mode = mode_selecting_stable_mppi_step(
                model,
                state,
                modes,
                active_obstacles,
                scene.goal,
                cfg,
                rng,
                rep_type=REP_GAUSSIAN,
                progress_by_mode=progress_by_mode,
                state_history=states,
                obstacle_circles=active_circles,
                record_optimal_traj=record,
                packed_mode_bank=packed_mode_bank,
                packed_polygons=active_packed_polygons,
            )
        elif variant == ControllerVariant.MODE_SELECTING_CORRIDOR_MPPI:
            control, info, progress_by_mode = mode_selecting_stable_mppi_step(
                model,
                state,
                modes,
                active_obstacles,
                scene.goal,
                cfg,
                rng,
                rep_type=REP_CORRIDOR,
                progress_by_mode=progress_by_mode,
                state_history=states,
                obstacle_circles=active_circles,
                record_optimal_traj=record,
                packed_mode_bank=packed_mode_bank,
                packed_polygons=active_packed_polygons,
            )
        elif variant in (ControllerVariant.STANDARD_MPPI):
            control, info = standard_mppi_step(
                model,
                state,
                active_obstacles,
                scene.goal,
                cfg,
                rng,
                obstacle_circles=active_circles,
                record_optimal_traj=record,
            )
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        executed_control = model.apply_final_output(
            state,
            control,
            previous_control,
            active_circles,
            scene.goal,
            cfg,
        )
        if record:
            model.render_output_trajectory(info, state, executed_control, scene.goal, cfg)
            infos.append(info)
        previous_control = executed_control.copy()
        state, arrived = model.advance_state(state, executed_control, scene.goal, cfg)
        states.append(state.copy())
        controls.append(executed_control.copy())
        if arrived:
            reached_goal = True
            break

    if record:
        wall_active = activation_step is not None and len(states) - 1 >= activation_step
        obstacle_history.append(list(blocked_obstacles if wall_active else base_obstacles))

    return SimulationResult(
        states=np.asarray(states),
        controls=np.asarray(controls),
        infos=infos,
        runtime=time.perf_counter() - started,
        activation_step=activation_step,
        obstacle_history=obstacle_history,
        reached_goal=bool(reached_goal),
    )