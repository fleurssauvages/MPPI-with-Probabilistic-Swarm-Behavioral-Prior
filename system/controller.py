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

from numba import njit, prange

Array = np.ndarray
NUMBA_AVAILABLE = True
_ILQR_MAX_WORKERS = max(1, min(8, int(os.cpu_count() or 1)))
_ILQR_EXECUTOR = ThreadPoolExecutor(max_workers=_ILQR_MAX_WORKERS, thread_name_prefix="ilqr")
_ILQR_WARM_KEYS: set[tuple[str, bool, bool]] = set()
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
    lambda_temperature: float = 4096
    adaptive_temperature_lbps: bool = True
    lbps_delta: float = 0.9
    lbps_optimizer_iterations: int = 32

    temporal_noise_smoothing: float = 0.1

    sigma_ref: float = 1.0

    spg_lookahead_steps: int = 10
    spg_pseudoinverse_damping: float = 1.0e-8
    spg_covariance_jitter: float = 1e-8

    robot_radius: float = 0.18
    hard_collision_clearance: float = 0.01
    suppress_blocked_modes: bool = True
    mode_blocking_clearance: float = 0.02
    mode_blocking_substeps: int = 2

    w_goal: float = 1000.0
    goal_tolerance: float = 0.1
    terminal_velocity_tolerance: float = 0.5

    mode_select_rollouts_per_mode: int = 0
    max_nearby_prior_modes: int = 32
    max_centerline_distance: float = 2.0
    centerline_history_points: int = 50

    def __post_init__(self) -> None:
        self.horizon = max(1, int(self.horizon))
        self.num_rollouts = max(1, int(self.num_rollouts))
        self.spg_lookahead_steps = max(1, int(self.spg_lookahead_steps))
        self.max_nearby_prior_modes = max(1, int(self.max_nearby_prior_modes))
        self.centerline_history_points = max(1, int(self.centerline_history_points))
        self.lbps_optimizer_iterations = max(8, int(self.lbps_optimizer_iterations))
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.lambda_temperature <= 0.0:
            raise ValueError("lambda_temperature must be positive.")
        if not (0.0 < float(self.lbps_delta) < 1.0):
            raise ValueError("lbps_delta must lie strictly between 0 and 1.")
        if self.spg_pseudoinverse_damping < 0.0 or self.spg_covariance_jitter < 0.0:
            raise ValueError("SPG damping and covariance jitter must be nonnegative.")
        if self.max_centerline_distance < 0.0:
            raise ValueError("max_centerline_distance must be nonnegative.")
        if self.w_goal < 0.0:
            raise ValueError("w_goal must be nonnegative.")
        if self.goal_tolerance < 0.0:
            raise ValueError("goal_tolerance must be nonnegative.")
        if self.terminal_velocity_tolerance < 0.0:
            raise ValueError("terminal_velocity_tolerance must be nonnegative.")

    @property
    def w_terminal_position(self) -> float:
        return 0.0

    @property
    def w_terminal_velocity(self) -> float:
        return 0.0


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
        beta = math.sqrt(max(0.0, 1.0 - alpha * alpha))

        for sample in range(noise.shape[0]):
            for t in range(1, noise.shape[1]):
                noise[sample, t, 0] = (
                    alpha * noise[sample, t - 1, 0]
                    + beta * noise[sample, t, 0]
                )
                noise[sample, t, 1] = (
                    alpha * noise[sample, t - 1, 1]
                    + beta * noise[sample, t, 1]
                )

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
        np.asarray([[3.5, 1.5], [5.7, 2.2], [5.2, 4.0], [3.3, 3.4]]),
        np.asarray([[6.2, 6.0], [8.5, 6.3], [8.1, 8.2], [6.8, 8.], [5.9, 7.4]]),
        np.asarray([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]]),
        np.asarray([[1.2, 4.2], [2.1, 4.0], [2.4, 4.8], [1.7, 5.3], [1.1, 4.9]]),
        np.asarray([[4.6, 5.1], [5.4, 5.0], [5.8, 5.7], [5.0, 6.2], [4.4, 5.7]]),
        np.asarray([[7.9, 3.0], [9.0, 3.2], [8.8, 4.2], [7.7, 4.0]]),
        np.asarray([[5.7, 0.5], [6.6, 0.7], [6.4, 1.8], [5.6, 1.6]]),
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


if njit is not None:
    @njit(cache=True)
    def _prior_second_moment_about_ilqr_nb(mean_path, cov_blocks, arc, positions, ilqr_positions):
        """Numba kernel for center-corrected local prior second moments.

        The added center-shift term is the full displacement outer product
        d d^T, where d is the difference between the geometric prior mean and
        the iLQR rollout position at the same path progress. This is the exact
        second moment of the geometric prior about the iLQR proposal center.
        """
        H = positions.shape[0]
        n = mean_path.shape[0]
        corrected = np.zeros((H, 2, 2), dtype=np.float64)
        displacement = np.zeros((H, 2), dtype=np.float64)
        scalar_variance = np.zeros(H, dtype=np.float64)

        if n == 0:
            return corrected, displacement, scalar_variance

        for t in range(H):
            s = positions[t]
            if n == 1 or arc.shape[0] <= 1 or arc[-1] <= 1e-12:
                lo = 0
                hi = 0
                w = 0.0
            elif s <= arc[0]:
                lo = 0
                hi = 1
                w = 0.0
            elif s >= arc[n - 1]:
                lo = n - 2
                hi = n - 1
                w = 1.0
            else:
                left = 0
                right = n - 1
                while right - left > 1:
                    mid = (left + right) // 2
                    if arc[mid] <= s:
                        left = mid
                    else:
                        right = mid
                lo = left
                hi = right
                denom = arc[hi] - arc[lo]
                if denom <= 1e-12:
                    w = 0.0
                else:
                    w = (s - arc[lo]) / denom

            one_minus_w = 1.0 - w
            mux = one_minus_w * mean_path[lo, 0] + w * mean_path[hi, 0]
            muy = one_minus_w * mean_path[lo, 1] + w * mean_path[hi, 1]

            c00 = one_minus_w * cov_blocks[lo, 0, 0] + w * cov_blocks[hi, 0, 0]
            c01a = one_minus_w * cov_blocks[lo, 0, 1] + w * cov_blocks[hi, 0, 1]
            c01b = one_minus_w * cov_blocks[lo, 1, 0] + w * cov_blocks[hi, 1, 0]
            c11 = one_minus_w * cov_blocks[lo, 1, 1] + w * cov_blocks[hi, 1, 1]
            c01 = 0.5 * (c01a + c01b)

            dx = mux - ilqr_positions[t, 0]
            dy = muy - ilqr_positions[t, 1]

            displacement[t, 0] = dx
            displacement[t, 1] = dy
            corrected[t, 0, 0] = c00 + dx * dx
            corrected[t, 0, 1] = c01 + dx * dy
            corrected[t, 1, 0] = c01 + dx * dy
            corrected[t, 1, 1] = c11 + dy * dy
            scalar_variance[t] = 0.5 * (corrected[t, 0, 0] + corrected[t, 1, 1])

        return corrected, displacement, scalar_variance
else:
    _prior_second_moment_about_ilqr_nb = None


def _prior_second_moment_about_ilqr(
    model: Any,
    x_current: Array,
    local_mode: MPPIHomotopyMode,
    ilqr_nominal: Array,
    control_positions: Array,
    cfg: Any,
    *,
    ilqr_positions: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
    """Return center-corrected spatial spread about the iLQR proposal.

    The geometric covariance is defined about the path mean. MPPI is centered
    on the dynamically feasible iLQR trajectory, so the spatial second moment
    gains the full center-shift term d d^T. No directional component is removed.

    The interpolation, outer product, symmetrization and Gaussian scalarization
    all run in one Numba kernel.
    """
    positions = np.ascontiguousarray(np.asarray(control_positions, dtype=np.float64).reshape(-1))

    if ilqr_positions is None:
        nominal_states = np.asarray(
            model.rollout_single(x_current, np.asarray(ilqr_nominal, dtype=np.float64), cfg),
            dtype=np.float64,
        )
        if nominal_states.ndim != 2 or nominal_states.shape[1] < 2:
            raise ValueError("model.rollout_single must return a 2-D state trajectory with planar position in columns 0:2.")
        if nominal_states.shape[0] < len(positions):
            raise ValueError(
                "iLQR nominal rollout is shorter than the control-position sequence: "
                f"{nominal_states.shape[0]} < {len(positions)}."
            )
        ilqr_xy = np.ascontiguousarray(nominal_states[: len(positions), :2])
    else:
        ilqr_xy = np.ascontiguousarray(np.asarray(ilqr_positions, dtype=np.float64))
        if ilqr_xy.ndim != 2 or ilqr_xy.shape[0] < len(positions) or ilqr_xy.shape[1] < 2:
            raise ValueError("ilqr_positions must have shape (>=H, >=2).")
        ilqr_xy = np.ascontiguousarray(ilqr_xy[: len(positions), :2])

    mean_path = np.ascontiguousarray(np.asarray(local_mode.mean_path, dtype=np.float64)[:, :2])
    cov_blocks = np.ascontiguousarray(np.asarray(local_mode.cov_blocks, dtype=np.float64))
    arc = np.ascontiguousarray(np.asarray(local_mode.arc_length, dtype=np.float64))

    corrected, displacement, scalar_variance = _prior_second_moment_about_ilqr_nb(
        mean_path, cov_blocks, arc, positions, ilqr_xy
    )
    return corrected, displacement, scalar_variance

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
    return _temporal_smooth_noise_nb(noise, alpha)


def _clip_controls(model: Any, raw_controls: Array, cfg: Any) -> Array:
    """Clip sampled controls to the model actuator limits."""
    clipped = np.ascontiguousarray(np.asarray(raw_controls, dtype=np.float64)).copy()
    clip_inplace = getattr(model, "clip_control_batch_inplace", None)
    if clip_inplace is not None:
        return np.asarray(clip_inplace(clipped, cfg), dtype=np.float64)
    return np.asarray(model.clip_control_batch(clipped, cfg), dtype=np.float64)


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
    raw_controls = make_temporally_correlated_noise(model, n, cfg.horizon, cfg, rng)
    raw_controls += center[None, :, :]
    return _clip_controls(model, raw_controls, cfg)


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


def _solve_mode_nominal_worker(
    model: Any,
    x_current: Array,
    mode: MPPIHomotopyMode,
    cfg: Any,
    need_jacobians: bool,
    need_trajectory: bool,
):
    """Solve one iLQR prior and return only the data required by the proposal.

    Recent center-correction logic needs the final iLQR planar trajectory for
    Gaussian/SPG, but Corridor does not. Avoiding an unconditional extra nonlinear
    rollout keeps the non-covariance ablations on their original fast path.

    Updated model files expose the final iLQR planar trajectory directly from the
    Numba iLQR solve, so Gaussian/SPG do not need to propagate the same nominal a
    second time. Older models remain supported via a rollout_single fallback.
    """
    if need_jacobians:
        combined5 = getattr(
            model,
            "nominal_controls_and_arc_positions_with_jacobians_and_trajectory",
            None,
        )
        project = getattr(model, "project_control_covariances_from_jacobians", None)
        if combined5 is not None and project is not None:
            controls, positions, A, B, ilqr_positions = combined5(
                x_current, mode.mean_path, cfg, mode.cov_blocks, None
            )
            return (
                np.asarray(controls, dtype=np.float64),
                np.asarray(positions, dtype=np.float64),
                np.asarray(A, dtype=np.float64),
                np.asarray(B, dtype=np.float64),
                np.ascontiguousarray(np.asarray(ilqr_positions, dtype=np.float64)[:, :2]),
            )

        combined = getattr(model, "nominal_controls_and_arc_positions_with_jacobians", None)
        if combined is not None and project is not None:
            controls, positions, A, B = combined(
                x_current, mode.mean_path, cfg, mode.cov_blocks, None
            )
            controls = np.asarray(controls, dtype=np.float64)
            positions = np.asarray(positions, dtype=np.float64)
            ilqr_positions = None
            if need_trajectory:
                states = np.asarray(model.rollout_single(x_current, controls, cfg), dtype=np.float64)
                ilqr_positions = np.ascontiguousarray(states[: len(positions), :2])
            return controls, positions, np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64), ilqr_positions

    if need_trajectory:
        combined3 = getattr(model, "nominal_controls_and_arc_positions_with_trajectory", None)
        if combined3 is not None:
            controls, positions, ilqr_positions = combined3(
                x_current, mode.mean_path, cfg, mode.cov_blocks
            )
            return (
                np.asarray(controls, dtype=np.float64),
                np.asarray(positions, dtype=np.float64),
                None,
                None,
                np.ascontiguousarray(np.asarray(ilqr_positions, dtype=np.float64)[:, :2]),
            )

    controls, positions = nominal_controls_and_arc_positions(
        model, x_current, mode.mean_path, cfg, mode.cov_blocks
    )
    controls = np.asarray(controls, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    ilqr_positions = None
    if need_trajectory:
        states = np.asarray(model.rollout_single(x_current, controls, cfg), dtype=np.float64)
        ilqr_positions = np.ascontiguousarray(states[: len(positions), :2])
    return controls, positions, None, None, ilqr_positions


def parallel_mode_nominals(
    model: Any,
    x_current: Array,
    local_modes: Sequence[MPPIHomotopyMode],
    cfg: Any,
    *,
    need_jacobians: bool = False,
    need_trajectory: bool = False,
) -> List[Tuple[Array, Array, Optional[Array], Optional[Array], Optional[Array]]]:
    """Solve independent cold iLQR mode priors concurrently, preserving mode order."""
    if not local_modes:
        return []
    if len(local_modes) == 1 or _ILQR_MAX_WORKERS <= 1:
        return [_solve_mode_nominal_worker(model, x_current, local_modes[0], cfg, need_jacobians, need_trajectory)]

    model_key = str(getattr(model, "MODEL_NAME", getattr(model, "__name__", type(model).__name__)))
    warm_key = (model_key, bool(need_jacobians), bool(need_trajectory))
    first_solution = None
    with _ILQR_WARM_LOCK:
        if warm_key not in _ILQR_WARM_KEYS:
            first_solution = _solve_mode_nominal_worker(
                model, x_current, local_modes[0], cfg, need_jacobians, need_trajectory
            )
            _ILQR_WARM_KEYS.add(warm_key)

    start_index = 1 if first_solution is not None else 0
    futures = [
        _ILQR_EXECUTOR.submit(_solve_mode_nominal_worker, model, x_current, mode, cfg, need_jacobians, need_trajectory)
        for mode in local_modes[start_index:]
    ]
    tail = [future.result() for future in futures]
    return ([first_solution] + tail) if first_solution is not None else tail


def _sample_gaussian_from_nominal(
    model: Any, x_current: Array, local_mode: MPPIHomotopyMode, ilqr_nominal: Array,
    control_positions: Array, n: int, cfg: Any, rng: np.random.Generator,
    *, ilqr_positions: Optional[Array] = None,
) -> Array:
    H = int(cfg.horizon)
    noise = make_temporally_correlated_noise(model, n, H, cfg, rng)
    corrected_cov, _, variance = _prior_second_moment_about_ilqr(
        model, x_current, local_mode, ilqr_nominal, control_positions, cfg,
        ilqr_positions=ilqr_positions,
    )

    sigma_ref = max(float(cfg.sigma_ref), 1e-9)
    noise *= (np.sqrt(np.maximum(variance, 0.0)) / sigma_ref)[None, :, None]
    noise += np.asarray(ilqr_nominal, dtype=np.float64)[None, :, :]
    return _clip_controls(model, noise, cfg)


def _project_spg_covariances(
    model: Any, A: Optional[Array], B: Optional[Array], cov_at_controls: Array, cfg: Any,
    *, x_current: Optional[Array] = None, ilqr_nominal: Optional[Array] = None,
) -> Array:
    """Project spatial covariance through the raw local control-to-position Jacobian."""
    project_from_jacobians = getattr(model, "project_control_covariances_from_jacobians", None)
    if A is not None and B is not None and project_from_jacobians is not None:
        return np.asarray(
            project_from_jacobians(A, B, cov_at_controls, cfg),
            dtype=np.float64,
        )

    if x_current is None or ilqr_nominal is None:
        raise ValueError("SPG fallback projection requires x_current and ilqr_nominal.")
    return np.asarray(
        model.project_control_covariances(x_current, ilqr_nominal, cov_at_controls, cfg),
        dtype=np.float64,
    )


def _sample_spg_from_nominal(
    model: Any, x_current: Array, local_mode: MPPIHomotopyMode, ilqr_nominal: Array,
    control_positions: Array, A: Optional[Array], B: Optional[Array], n: int, cfg: Any,
    rng: np.random.Generator, *, ilqr_positions: Optional[Array] = None,
) -> Array:
    H = int(cfg.horizon)
    cov_at_controls, _, _ = _prior_second_moment_about_ilqr(
        model, x_current, local_mode, ilqr_nominal, control_positions, cfg,
        ilqr_positions=ilqr_positions,
    )
    projected = _project_spg_covariances(
        model, A, B, cov_at_controls, cfg,
        x_current=x_current, ilqr_nominal=ilqr_nominal,
    )
    standard_noise = make_temporally_correlated_noise(
        model, n, H, cfg, rng, scale_override=np.ones(2, dtype=np.float64)
    )
    noise = _apply_projected_covariance_nb(standard_noise, np.asarray(projected, dtype=np.float64))
    noise += np.asarray(ilqr_nominal, dtype=np.float64)[None, :, :]
    return _clip_controls(model, noise, cfg)


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
    with_trajectory = getattr(model, "nominal_controls_and_arc_positions_with_trajectory", None)
    if with_trajectory is not None:
        ilqr_nominal, control_positions, ilqr_positions = with_trajectory(
            x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
        )
    else:
        ilqr_nominal, control_positions = nominal_controls_and_arc_positions(
            model, x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
        )
        ilqr_positions = None
    noise = make_temporally_correlated_noise(model, n, H, cfg, rng)
    corrected_cov, _, variance = _prior_second_moment_about_ilqr(
        model, x_current, local_mode, ilqr_nominal, control_positions, cfg,
        ilqr_positions=ilqr_positions,
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
    with_jacobians_and_trajectory = getattr(
        model, "nominal_controls_and_arc_positions_with_jacobians_and_trajectory", None
    )
    with_jacobians = getattr(model, "nominal_controls_and_arc_positions_with_jacobians", None)
    project_from_jacobians = getattr(model, "project_control_covariances_from_jacobians", None)
    if with_jacobians_and_trajectory is not None and project_from_jacobians is not None:
        ilqr_nominal, control_positions, A, B, ilqr_positions = with_jacobians_and_trajectory(
            x_current, local_mode.mean_path, cfg, local_mode.cov_blocks, None
        )
        cov_at_controls, _, _ = _prior_second_moment_about_ilqr(
            model, x_current, local_mode, ilqr_nominal, control_positions, cfg,
            ilqr_positions=ilqr_positions,
        )
        projected = _project_spg_covariances(
            model, A, B, cov_at_controls, cfg,
            x_current=x_current, ilqr_nominal=ilqr_nominal,
        )
    elif with_jacobians is not None and project_from_jacobians is not None:
        ilqr_nominal, control_positions, A, B = with_jacobians(
            x_current, local_mode.mean_path, cfg, local_mode.cov_blocks, None
        )
        cov_at_controls, _, _ = _prior_second_moment_about_ilqr(
            model, x_current, local_mode, ilqr_nominal, control_positions, cfg
        )
        projected = _project_spg_covariances(
            model, A, B, cov_at_controls, cfg,
            x_current=x_current, ilqr_nominal=ilqr_nominal,
        )
    else:
        with_trajectory = getattr(model, "nominal_controls_and_arc_positions_with_trajectory", None)
        if with_trajectory is not None:
            ilqr_nominal, control_positions, ilqr_positions = with_trajectory(
                x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
            )
        else:
            ilqr_nominal, control_positions = nominal_controls_and_arc_positions(
                model, x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
            )
            ilqr_positions = None
        cov_at_controls, _, _ = _prior_second_moment_about_ilqr(
            model, x_current, local_mode, ilqr_nominal, control_positions, cfg,
            ilqr_positions=ilqr_positions,
        )
        projected = _project_spg_covariances(
            model, None, None, cov_at_controls, cfg,
            x_current=x_current, ilqr_nominal=ilqr_nominal,
        )
    standard_noise = make_temporally_correlated_noise(
        model, n, H, cfg, rng, scale_override=np.ones(2, dtype=np.float64)
    )
    noise = _apply_projected_covariance_nb(standard_noise, np.asarray(projected, dtype=np.float64))
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
    distances = _mode_nearest_distances_nb(
        bank.mean_paths, bank.lengths, float(position[0]), float(position[1])
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

    compute_clearance = bool(cfg.suppress_blocked_modes and obstacles)
    if compute_clearance:
        if packed_polygons is None:
            polygons_padded, polygon_lengths = obstacle_polygons_to_padded_arrays(obstacles)
        else:
            polygons_padded, polygon_lengths = packed_polygons
    else:
        polygons_padded = np.zeros((0, 0, 2), dtype=np.float64)
        polygon_lengths = np.zeros(0, dtype=np.int64)

    starts, ends, distances, clearances = _localize_mode_bank_nb(
        bank.mean_paths, bank.arc_lengths, bank.lengths,
        float(position[0]), float(position[1]), np.ascontiguousarray(recent_positions),
        max(1, int(cfg.horizon)), prior_preview_step_distance(cfg),
        np.ascontiguousarray(polygons_padded), np.ascontiguousarray(polygon_lengths),
        compute_clearance,
    )

    all_clearances = [float(value) for value in clearances]
    max_centerline_distance = float(cfg.max_centerline_distance)
    for global_index, mode in enumerate(global_modes):
        mode = _ensure_mode_prior_cache(mode)
        index = int(starts[global_index])
        end_index = int(ends[global_index])
        new_progress[str(mode.signature)] = index
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


def softmin_score(
    costs: Array, cfg: Any, *, temperature: Optional[float] = None
) -> float:
    values = np.ascontiguousarray(np.asarray(costs, dtype=np.float64))
    temp = float(cfg.lambda_temperature) if temperature is None else float(temperature)
    return float(_softmin_score_nb(values, temp))


@njit(cache=True, parallel=True)
def _goal_costs_nb(states, goal, w_goal):
    """Original MPPI running goal cost: mean squared planar distance to goal."""
    n = states.shape[0]
    T = states.shape[1]
    H = max(1, T - 1)
    inv_h = 1.0 / H
    out = np.empty(n, dtype=np.float64)

    for i in prange(n):
        cost = 0.0
        for k in range(1, T):
            dx = states[i, k, 0] - goal[0]
            dy = states[i, k, 1] - goal[1]
            cost += w_goal * inv_h * (dx * dx + dy * dy)
        out[i] = cost

    return out

@njit(cache=True)
def _mppi_weights_nb(costs, temperature):
    n = costs.shape[0]
    weights = np.zeros(n, dtype=np.float64)
    finite_count = 0
    rho = math.inf
    for i in range(n):
        if math.isfinite(costs[i]):
            finite_count += 1
            if costs[i] < rho:
                rho = costs[i]
    if n == 0:
        return weights
    if finite_count == 0:
        inv_n = 1.0 / n
        for i in range(n):
            weights[i] = inv_n
        return weights
    total = 0.0
    for i in range(n):
        if math.isfinite(costs[i]):
            w = math.exp(-(costs[i] - rho) / temperature)
            weights[i] = w
            total += w
    if total <= 1e-12:
        inv = 1.0 / finite_count
        for i in range(n):
            if math.isfinite(costs[i]):
                weights[i] = inv
    else:
        inv = 1.0 / total
        for i in range(n):
            weights[i] *= inv
    return weights


@njit(cache=True)
def _softmin_score_nb(costs, temperature):
    finite_count = 0
    rho = math.inf
    for i in range(costs.shape[0]):
        if math.isfinite(costs[i]):
            finite_count += 1
            if costs[i] < rho:
                rho = costs[i]
    if finite_count == 0:
        return 1e309
    total = 0.0
    for i in range(costs.shape[0]):
        if math.isfinite(costs[i]):
            total += math.exp(-(costs[i] - rho) / temperature)
    return rho - temperature * math.log(total / max(1, costs.shape[0]) + 1e-12)


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _lbps_score_nb(costs, alpha, delta, rho, reward_norm):
        """Watson-Peters LBPS Eq. (5) evaluated with SNIS weights.

        The controller minimizes costs J, so the paper's return is R=-J and
        alpha is the inverse temperature (alpha = 1/lambda). Infinite costs
        are hard-rejected rollouts and receive zero importance weight.
        """
        sum_w = 0.0
        sum_w2 = 0.0
        sum_wr = 0.0
        for i in range(costs.shape[0]):
            c = costs[i]
            if not math.isfinite(c):
                continue
            w = math.exp(-alpha * (c - rho))
            sum_w += w
            sum_w2 += w * w
            sum_wr += w * (-c)

        if sum_w <= 0.0 or sum_w2 <= 0.0:
            return -math.inf, 0.0, -math.inf

        ess = (sum_w * sum_w) / sum_w2
        expected_return = sum_wr / sum_w
        penalty = reward_norm * math.sqrt((1.0 - delta) / (delta * max(ess, 1e-300)))
        return expected_return - penalty, ess, expected_return

    @njit(cache=True)
    def _lbps_temperature_nb(costs, fallback_temperature, delta, optimizer_iterations):
        """Optimize Watson-Peters LBPS over alpha>=0 using a Numba golden search.

        The paper uses Brent/scipy. Its Appendix F characterizes the objective
        as quasi-concave. Here we first bracket its maximum by doubling alpha,
        then apply golden-section search entirely inside Numba.
        """
        finite_count = 0
        rho = math.inf
        max_cost = -math.inf
        reward_norm = 0.0
        for i in range(costs.shape[0]):
            c = costs[i]
            if math.isfinite(c):
                finite_count += 1
                if c < rho:
                    rho = c
                if c > max_cost:
                    max_cost = c
                ac = abs(c) 
                if ac > reward_norm:
                    reward_norm = ac

        fallback_alpha = 1.0 / max(fallback_temperature, 1e-300)
        if finite_count == 0:
            return fallback_temperature, fallback_alpha, 0.0, -math.inf, 0, 0.0, -math.inf

        spread = max_cost - rho
        if finite_count <= 1 or spread <= 1e-12 or reward_norm <= 1e-15:
            score, ess, expected_return = _lbps_score_nb(
                costs, fallback_alpha, delta, rho, reward_norm
            )
            return (
                fallback_temperature, fallback_alpha, ess, score,
                finite_count, reward_norm, expected_return,
            )

        alpha0 = 0.0
        score0, _, _ = _lbps_score_nb(costs, alpha0, delta, rho, reward_norm)
        alpha1 = 1.0 / spread
        score1, _, _ = _lbps_score_nb(costs, alpha1, delta, rho, reward_norm)

        left = 0.0
        right = alpha1
        if score1 > score0:
            prevprev = alpha0
            prev = alpha1
            prev_score = score1
            bracketed = False
            for _ in range(40):
                nxt = prev * 2.0
                nxt_score, _, _ = _lbps_score_nb(costs, nxt, delta, rho, reward_norm)
                if nxt_score <= prev_score:
                    left = prevprev
                    right = nxt
                    bracketed = True
                    break
                prevprev = prev
                prev = nxt
                prev_score = nxt_score
            if not bracketed:
                score, ess, expected_return = _lbps_score_nb(
                    costs, prev, delta, rho, reward_norm
                )
                temperature = 1.0 / max(prev, 1e-300)
                return temperature, prev, ess, score, finite_count, reward_norm, expected_return

        golden = 0.6180339887498949
        a = left
        b = right
        c = b - golden * (b - a)
        d = a + golden * (b - a)
        fc, _, _ = _lbps_score_nb(costs, c, delta, rho, reward_norm)
        fd, _, _ = _lbps_score_nb(costs, d, delta, rho, reward_norm)
        for _ in range(max(8, int(optimizer_iterations))):
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - golden * (b - a)
                fc, _, _ = _lbps_score_nb(costs, c, delta, rho, reward_norm)
            else:
                a = c
                c = d
                fc = fd
                d = a + golden * (b - a)
                fd, _, _ = _lbps_score_nb(costs, d, delta, rho, reward_norm)

        alpha = 0.5 * (a + b)
        score, ess, expected_return = _lbps_score_nb(costs, alpha, delta, rho, reward_norm)

        if score0 >= score:
            alpha = 0.0
            score, ess, expected_return = _lbps_score_nb(costs, alpha, delta, rho, reward_norm)

        if alpha <= 1e-14 / spread:
            alpha = 1e-14 / spread
            score, ess, expected_return = _lbps_score_nb(costs, alpha, delta, rho, reward_norm)

        temperature = 1.0 / alpha
        return temperature, alpha, ess, score, finite_count, reward_norm, expected_return

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

    @njit(cache=True)
    def _mppi_mode_diagnostics_nb(costs, controls, offsets, nominal_controls, temperature):
        """Compute pooled MPPI weight mass and conditional statistics per mode.

        The global MPPI weights are used for the mode masses.  Each mode's ESS
        and weighted sequence are then computed from the weights renormalized
        within that mode.  This is diagnostic only and does not alter the MPPI
        update.
        """
        n = costs.shape[0]
        mode_count = offsets.shape[0] - 1
        H = controls.shape[1]
        d = controls.shape[2]

        mode_mass = np.zeros(mode_count, dtype=np.float64)
        mode_ess = np.zeros(mode_count, dtype=np.float64)
        mode_sequence = np.zeros((mode_count, H, d), dtype=np.float64)
        weighted_nominal = np.zeros((H, d), dtype=np.float64)

        if n == 0 or mode_count <= 0:
            return mode_mass, mode_ess, mode_sequence, weighted_nominal

        finite_count = 0
        rho = math.inf
        for i in range(n):
            if math.isfinite(costs[i]):
                finite_count += 1
                if costs[i] < rho:
                    rho = costs[i]

        if finite_count == 0:
            inv_n = 1.0 / float(n)
            for mode in range(mode_count):
                start = int(offsets[mode])
                end = int(offsets[mode + 1])
                count = end - start
                if count <= 0:
                    continue
                mass = float(count) * inv_n
                mode_mass[mode] = mass
                mode_ess[mode] = float(count)
                inv_count = 1.0 / float(count)
                for i in range(start, end):
                    for h in range(H):
                        for j in range(d):
                            mode_sequence[mode, h, j] += inv_count * controls[i, h, j]
                for h in range(H):
                    for j in range(d):
                        weighted_nominal[h, j] += mass * nominal_controls[mode, h, j]
            return mode_mass, mode_ess, mode_sequence, weighted_nominal

        total_weight = 0.0
        for i in range(n):
            if math.isfinite(costs[i]):
                total_weight += math.exp(-(costs[i] - rho) / temperature)

        if total_weight <= 1e-12:
            inv_finite = 1.0 / float(finite_count)
            for mode in range(mode_count):
                start = int(offsets[mode])
                end = int(offsets[mode + 1])
                local_count = 0
                for i in range(start, end):
                    if math.isfinite(costs[i]):
                        local_count += 1
                if local_count <= 0:
                    continue
                mass = float(local_count) * inv_finite
                mode_mass[mode] = mass
                mode_ess[mode] = float(local_count)
                inv_local = 1.0 / float(local_count)
                for i in range(start, end):
                    if math.isfinite(costs[i]):
                        for h in range(H):
                            for j in range(d):
                                mode_sequence[mode, h, j] += inv_local * controls[i, h, j]
                for h in range(H):
                    for j in range(d):
                        weighted_nominal[h, j] += mass * nominal_controls[mode, h, j]
            return mode_mass, mode_ess, mode_sequence, weighted_nominal

        for mode in range(mode_count):
            start = int(offsets[mode])
            end = int(offsets[mode + 1])
            local_sum = 0.0
            local_sum_sq = 0.0

            for i in range(start, end):
                if not math.isfinite(costs[i]):
                    continue
                unnormalized = math.exp(-(costs[i] - rho) / temperature)
                local_sum += unnormalized
                local_sum_sq += unnormalized * unnormalized
                for h in range(H):
                    for j in range(d):
                        mode_sequence[mode, h, j] += unnormalized * controls[i, h, j]

            if local_sum <= 1e-12:
                continue

            mass = local_sum / total_weight
            mode_mass[mode] = mass
            if local_sum_sq > 0.0:
                mode_ess[mode] = (local_sum * local_sum) / local_sum_sq

            inv_local = 1.0 / local_sum
            for h in range(H):
                for j in range(d):
                    mode_sequence[mode, h, j] *= inv_local
                    weighted_nominal[h, j] += mass * nominal_controls[mode, h, j]

        return mode_mass, mode_ess, mode_sequence, weighted_nominal
else:
    _lbps_score_nb = None
    _lbps_temperature_nb = None
    _mppi_weighted_sequence_nb = None
    _mppi_mode_diagnostics_nb = None


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
    del packed_obstacles
    states = np.ascontiguousarray(np.asarray(model.rollout_batch(x_current, controls, cfg), dtype=np.float64))
    costs = _goal_costs_nb(
        states,
        np.ascontiguousarray(np.asarray(goal, dtype=np.float64)),
        float(cfg.w_goal),
    )
    colliding = (
        np.asarray(model.collision_mask(states, obstacle_circles, goal, cfg), dtype=bool)
        if obstacle_circles else np.zeros(len(controls), dtype=bool)
    )
    return np.asarray(costs, dtype=np.float64), colliding, states


def resolve_mppi_temperature(
    costs: Array, cfg: Any
) -> Tuple[float, float, float, float, int, float, float]:
    """Return (lambda, alpha, ESS, LBPS score, finite N, ||R||inf, E[R]).

    Watson & Peters parameterize the Gibbs posterior by inverse temperature
    alpha. This code uses MPPI temperature lambda, so lambda = 1/alpha.
    """
    fallback = float(cfg.lambda_temperature)
    if not bool(getattr(cfg, "adaptive_temperature_lbps", False)):
        values = np.asarray(costs, dtype=np.float64)
        finite_count = int(np.count_nonzero(np.isfinite(values)))
        return fallback, 1.0 / fallback, float("nan"), float("nan"), finite_count, float("nan"), float("nan")

    values = np.ascontiguousarray(np.asarray(costs, dtype=np.float64))
    delta = float(getattr(cfg, "lbps_delta", 0.5))
    iterations = int(getattr(cfg, "lbps_optimizer_iterations", 32))
    return _lbps_temperature_nb(values, fallback, delta, iterations)


def mppi_weights(
    costs: Array, cfg: Any, *, temperature: Optional[float] = None
) -> Array:
    values = np.ascontiguousarray(np.asarray(costs, dtype=np.float64))
    temp = float(cfg.lambda_temperature) if temperature is None else float(temperature)
    return _mppi_weights_nb(values, temp)


def mppi_weighted_control_sequence(
    model: Any, costs: Array, controls: Array, cfg: Any, *, temperature: Optional[float] = None
) -> Array:
    costs_arr = np.ascontiguousarray(np.asarray(costs, dtype=np.float64))
    controls_arr = np.ascontiguousarray(np.asarray(controls, dtype=np.float64))
    temp = float(cfg.lambda_temperature) if temperature is None else float(temperature)
    sequence = _mppi_weighted_sequence_nb(costs_arr, controls_arr, temp)
    clip_sequence = getattr(model, "clip_control_sequence", None)
    if clip_sequence is not None:
        return np.asarray(clip_sequence(sequence, cfg), dtype=np.float64)
    return model.clip_control_batch(np.asarray(sequence, dtype=np.float64)[None, :, :], cfg)[0]


def mppi_mode_diagnostics(
    costs: Array,
    controls: Array,
    offsets: Array,
    nominal_controls_by_mode: Sequence[Array],
    cfg: Any,
    *,
    temperature: Optional[float] = None,
) -> Tuple[Array, Array, Array, Array]:
    """Return mode weight mass, within-mode ESS, weighted sequences, and weighted nominal.

    This is a read-only diagnostic of the same global MPPI weights used by Eq. (6).
    Rows correspond to the active-mode ordering used by ``offsets``.
    """
    costs_arr = np.ascontiguousarray(np.asarray(costs, dtype=np.float64))
    controls_arr = np.ascontiguousarray(np.asarray(controls, dtype=np.float64))
    offsets_arr = np.ascontiguousarray(np.asarray(offsets, dtype=np.int64))
    nominals_arr = np.ascontiguousarray(np.asarray(nominal_controls_by_mode, dtype=np.float64))

    temp = float(cfg.lambda_temperature) if temperature is None else float(temperature)
    return _mppi_mode_diagnostics_nb(
        costs_arr, controls_arr, offsets_arr, nominals_arr, temp
    )


def _single_sequence_evaluation(
    model: Any,
    x_current: Array,
    sequence: Array,
    obstacle_circles: Sequence[Tuple[Array, float]],
    goal: Array,
    cfg: Any,
    packed_obstacles=None,
    packed_polygons=None,
) -> Tuple[float, Array, bool]:
    del packed_obstacles, packed_polygons
    clip_sequence = getattr(model, "clip_control_sequence", None)
    if clip_sequence is not None:
        controls = np.asarray(clip_sequence(sequence, cfg), dtype=np.float64)
    else:
        controls = model.clip_control_batch(np.asarray(sequence, dtype=np.float64)[None, :, :], cfg)[0]
    states = np.ascontiguousarray(np.asarray(model.rollout_single(x_current, controls, cfg), dtype=np.float64))
    batch_states = np.ascontiguousarray(states[None, :, :])
    raw_cost = float(_goal_costs_nb(
        batch_states,
        np.ascontiguousarray(np.asarray(goal, dtype=np.float64)),
        float(cfg.w_goal),
    )[0])
    colliding = bool(model.collision_mask(batch_states, obstacle_circles, goal, cfg)[0]) if obstacle_circles else False
    return raw_cost, states, colliding


def _finite_cost_mean(costs: Array) -> float:
    values = np.asarray(costs, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("inf")


def _select_feasible_mppi_output(
    model: Any,
    x_current: Array,
    nominal: Array,
    nominal_cost: float,
    nominal_traj: Array,
    weighted_candidate: Array,
    rollout_controls: Array,
    rollout_costs: Array,
    obstacle_circles: Sequence[Tuple[Array, float]],
    goal: Array,
    cfg: Any,
    packed_obstacles=None,
    packed_polygons=None,
) -> Tuple[Array, float, Array, bool, str, float, bool, float]:
    """Return the MPPI weighted sequence directly.

    Sampled rollouts may already have been hard-rejected before Eq. (6), but the
    weighted sequence itself is not replaced by an argmin rollout or by the iLQR
    nominal.  Collision of the weighted sequence is retained as a diagnostic so
    temperature/proposal failures remain visible instead of being hidden by a
    temperature-independent fallback.
    """
    del nominal, nominal_cost, nominal_traj, rollout_controls, packed_polygons
    candidate_cost, candidate_traj, candidate_collision = _single_sequence_evaluation(
        model, x_current, weighted_candidate, obstacle_circles, goal, cfg, packed_obstacles, None
    )
    costs = np.asarray(rollout_costs, dtype=np.float64)
    finite_ids = np.flatnonzero(np.isfinite(costs))
    best_rollout_cost = (
        float(np.min(costs[finite_ids])) if finite_ids.size else float("inf")
    )
    return (
        np.asarray(weighted_candidate, dtype=np.float64),
        float(candidate_cost),
        np.asarray(candidate_traj),
        not bool(candidate_collision),
        "weighted_candidate" if not candidate_collision else "weighted_candidate_colliding",
        float(candidate_cost),
        bool(candidate_collision),
        best_rollout_cost,
    )


def _standard_mppi_pass(
    model: Any,
    x_current: Array,
    center: Array,
    rollout_count: int,
    obstacle_circles: Sequence[Tuple[Array, float]],
    goal: Array,
    cfg: Any,
    rng: np.random.Generator,
    packed_obstacles=None,
) -> Tuple[Array, Array, Array, Optional[Array], Tuple[float, float, float, float, int, float, float]]:
    controls = sample_controls_around_nominal(model, center, rollout_count, cfg, rng)
    raw_costs, colliding, states = _rollout_costs_and_collisions(
        model, x_current, controls, obstacle_circles, goal, cfg, packed_obstacles
    )
    raw_costs = np.asarray(raw_costs, dtype=np.float64)
    raw_costs[np.asarray(colliding, dtype=bool)] = np.inf
    temperature_info = resolve_mppi_temperature(raw_costs, cfg)
    temperature = float(temperature_info[0])
    candidate = mppi_weighted_control_sequence(
        model, raw_costs, controls, cfg, temperature=temperature
    )
    return candidate, controls, raw_costs, states, temperature_info

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
    packed_polygons = obstacle_polygons_to_padded_arrays(obstacles)
    nominal = model.clip_control_batch(
        np.asarray(model.nominal_controls_to_goal(x_current, goal, cfg), dtype=np.float64)[None, :, :], cfg
    )[0]
    current_cost, current_traj, _ = _single_sequence_evaluation(
        model, x_current, nominal, obstacle_circles, goal, cfg, packed_obstacles, packed_polygons
    )
    candidate, rollout_controls, costs, _, temperature_info = _standard_mppi_pass(
        model, x_current, nominal, int(cfg.num_rollouts), obstacle_circles, goal, cfg, rng, packed_obstacles
    )
    temperature, inverse_temperature, lbps_ess, lbps_objective, lbps_finite, lbps_reward_norm, lbps_expected_return = temperature_info
    (
        planned_sequence, final_cost, final_traj, candidate_accepted, output_source,
        weighted_candidate_cost, weighted_candidate_collision, best_rollout_cost,
    ) = _select_feasible_mppi_output(
        model, x_current, nominal, current_cost, current_traj, candidate,
        rollout_controls, costs, obstacle_circles, goal, cfg, packed_obstacles, packed_polygons
    )
    history = [float(current_cost), float(final_cost)]
    weights = mppi_weights(costs, cfg, temperature=float(temperature))
    effective_sample_size = float(1.0 / np.sum(weights * weights)) if weights.size else 0.0

    info: Dict[str, object] = {
        "cost_min": float(np.min(costs)) if costs.size else float("inf"),
        "cost_mean": _finite_cost_mean(costs),
        "optimal_traj": np.asarray(final_traj).copy() if record_optimal_traj else None,
        "planned_control_sequence": np.asarray(planned_sequence).copy(),
        "selected_rollout_mode_index": None,
        "rollout_budget_total": int(cfg.num_rollouts),
        "mppi_passes": 1,
        "mppi_candidate_accepted": int(candidate_accepted),
        "mppi_output_source": output_source,
        "mppi_weighted_candidate_cost": float(weighted_candidate_cost),
        "mppi_weighted_candidate_collision": bool(weighted_candidate_collision),
        "mppi_best_rollout_cost": float(best_rollout_cost),
        "mppi_effective_sample_size": float(effective_sample_size),
        "mppi_temperature_strategy": "lbps" if bool(getattr(cfg, "adaptive_temperature_lbps", False)) else "fixed",
        "mppi_temperature": float(temperature),
        "mppi_inverse_temperature": float(inverse_temperature),
        "mppi_lbps_delta": float(getattr(cfg, "lbps_delta", 0.5)),
        "mppi_lbps_objective": float(lbps_objective),
        "mppi_lbps_effective_sample_size": float(lbps_ess),
        "mppi_lbps_finite_rollouts": int(lbps_finite),
        "mppi_lbps_reward_norm": float(lbps_reward_norm),
        "mppi_lbps_expected_return": float(lbps_expected_return),
        "mppi_nominal_delta_norm": float(np.linalg.norm(np.asarray(planned_sequence) - np.asarray(nominal))),
        "mppi_cost_history": history,
        "mppi_initial_cost": float(current_cost),
        "mppi_final_cost": float(final_cost),
    }
    return np.asarray(planned_sequence[0]).copy(), info

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
        "mppi_passes": 0,
        "mppi_candidate_accepted": 0,
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
    if packed_polygons is None:
        packed_polygons = obstacle_polygons_to_padded_arrays(obstacles)
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
        selection_policy = "all_localized_pi_weighted_soft_cost_single_pass"
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
        selection_policy = "nearby_balanced_control_bank_single_pass"

    active_count = len(active_global_modes)
    if sum(int(count) for count in counts) != total_budget:
        raise RuntimeError(
            f"Internal rollout allocation error: allocated {sum(int(count) for count in counts)} of {total_budget} rollouts."
        )

    all_controls = np.empty((total_budget, cfg.horizon, 2), dtype=np.float64)
    nominal_controls_by_mode: List[Array] = [np.zeros((cfg.horizon, 2), dtype=np.float64) for _ in range(active_count)]

    nominal_solutions = parallel_mode_nominals(
        model, x_current, active_local_modes, cfg,
        need_jacobians=(rep_type == REP_SENSITIVITY_PROJECTED_GAUSSIAN),
        need_trajectory=(rep_type in {REP_GAUSSIAN, REP_SENSITIVITY_PROJECTED_GAUSSIAN}),
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
        ilqr_nominal, control_positions, A, B, ilqr_positions = nominal_solutions[mode_index]
        if rep_type == REP_GAUSSIAN:
            controls = _sample_gaussian_from_nominal(
                model, x_current, local_mode, ilqr_nominal, control_positions, count, cfg, rng,
                ilqr_positions=ilqr_positions,
            )
        elif rep_type == REP_SENSITIVITY_PROJECTED_GAUSSIAN:
            controls = _sample_spg_from_nominal(
                model, x_current, local_mode, ilqr_nominal, control_positions, A, B, count, cfg, rng,
                ilqr_positions=ilqr_positions,
            )
        elif rep_type == REP_CONTROL_BANK:
            controls = sample_exact_control_bank(
                model, x_current, global_mode, ilqr_nominal, count, cfg
            )
        else:
            controls = sample_controls_around_nominal(model, ilqr_nominal, count, cfg, rng)
        nominal_controls_by_mode[mode_index] = np.asarray(ilqr_nominal, dtype=np.float64)
        all_controls[start_id:end_id] = controls

    all_costs, collision_mask, _ = _rollout_costs_and_collisions(
        model, x_current, all_controls, obstacle_circles, goal, cfg, packed_obstacles
    )
    all_costs = np.asarray(all_costs, dtype=np.float64)
    all_costs[np.asarray(collision_mask, dtype=bool)] = np.inf

    selection_temperature_info = resolve_mppi_temperature(all_costs, cfg)
    (
        selection_temperature,
        selection_inverse_temperature,
        selection_lbps_ess,
        selection_lbps_objective,
        selection_lbps_finite,
        selection_lbps_reward_norm,
        selection_lbps_expected_return,
    ) = selection_temperature_info

    pooled_candidate = mppi_weighted_control_sequence(
        model, all_costs, all_controls, cfg, temperature=float(selection_temperature)
    )
    (
        mode_weight_mass,
        mode_effective_sample_size,
        mode_weighted_sequences,
        pooled_weighted_nominal,
    ) = mppi_mode_diagnostics(
        all_costs, all_controls, offsets, nominal_controls_by_mode, cfg,
        temperature=float(selection_temperature),
    )
    mode_weighted_control_delta = mode_weighted_sequences - np.asarray(
        nominal_controls_by_mode, dtype=np.float64
    )
    global_weighted_control_delta = np.asarray(pooled_candidate, dtype=np.float64) - pooled_weighted_nominal

    mode_scores = np.full(active_count, np.inf, dtype=np.float64)
    for mode_index in range(active_count):
        start_id = int(offsets[mode_index])
        end_id = int(offsets[mode_index + 1])
        if end_id <= start_id:
            continue
        local_costs = all_costs[start_id:end_id]
        if not np.any(np.isfinite(local_costs)):
            continue
        prior_probability = max(float(probabilities[mode_index]), 1e-12)
        mode_scores[mode_index] = (
            float(softmin_score(local_costs, cfg, temperature=float(selection_temperature)))
            - float(selection_temperature) * math.log(prior_probability)
        )

    finite_mode_ids = np.flatnonzero(np.isfinite(mode_scores))
    if finite_mode_ids.size:
        baseline_local_index = int(finite_mode_ids[np.argmin(mode_scores[finite_mode_ids])])
    else:
        baseline_local_index = int(np.argmax(probabilities)) if len(probabilities) else 0

    best_mode_global_index = int(active_global_indices[baseline_local_index])
    baseline = np.asarray(nominal_controls_by_mode[baseline_local_index], dtype=np.float64)
    baseline_cost, baseline_traj, _ = _single_sequence_evaluation(
        model, x_current, baseline, obstacle_circles, goal, cfg, packed_obstacles, packed_polygons
    )

    selected_start = int(offsets[baseline_local_index])
    selected_end = int(offsets[baseline_local_index + 1])
    selected_controls = all_controls[selected_start:selected_end]
    selected_costs = all_costs[selected_start:selected_end]

    temperature_info = resolve_mppi_temperature(selected_costs, cfg)
    (
        temperature, inverse_temperature, lbps_ess, lbps_objective,
        lbps_finite, lbps_reward_norm, lbps_expected_return,
    ) = temperature_info
    selected_candidate = mppi_weighted_control_sequence(
        model, selected_costs, selected_controls, cfg, temperature=float(temperature)
    )

    (
        current, current_cost, current_traj, candidate_feasible, output_source,
        weighted_candidate_cost, weighted_candidate_collision, best_rollout_cost,
    ) = _select_feasible_mppi_output(
        model, x_current, baseline, baseline_cost, baseline_traj, selected_candidate,
        selected_controls, selected_costs, obstacle_circles, goal, cfg, packed_obstacles, packed_polygons
    )
    history = [float(baseline_cost), float(current_cost)]
    accepted = int(candidate_feasible)
    planned_sequence = np.asarray(current, dtype=np.float64)
    final_cost = float(current_cost)
    final_traj = np.asarray(current_traj).copy()
    weights = mppi_weights(selected_costs, cfg, temperature=float(temperature))
    effective_sample_size = float(1.0 / np.sum(weights * weights)) if weights.size else 0.0

    nominal_ilqr_traj = None
    if record_optimal_traj:
        nominal_ilqr_traj = np.asarray(baseline_traj).copy()

    info = {
        "cost_min": float(np.min(selected_costs)) if selected_costs.size else float("inf"),
        "cost_mean": _finite_cost_mean(selected_costs),
        "soft_value": float(mode_scores[baseline_local_index]), "rep_type": int(rep_type),
        "mode_selection": True,
        "mode_selection_policy": "single_homotopy_softmin_with_prior",
        "selected_mode_index": best_mode_global_index,
        "rollout_budget_total": int(total_budget),
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
        "planned_control_sequence": planned_sequence, "mppi_passes": 1,
        "mppi_candidate_accepted": int(accepted), "mppi_output_source": output_source,
        "mppi_weighted_candidate_cost": float(weighted_candidate_cost),
        "mppi_weighted_candidate_collision": bool(weighted_candidate_collision),
        "mppi_best_rollout_cost": float(best_rollout_cost),
        "mppi_effective_sample_size": float(effective_sample_size),
        "mppi_temperature_strategy": "lbps" if bool(getattr(cfg, "adaptive_temperature_lbps", False)) else "fixed",
        "mppi_temperature": float(temperature),
        "mppi_inverse_temperature": float(inverse_temperature),
        "mppi_lbps_delta": float(getattr(cfg, "lbps_delta", 0.5)),
        "mppi_lbps_objective": float(lbps_objective),
        "mppi_lbps_effective_sample_size": float(lbps_ess),
        "mppi_lbps_finite_rollouts": int(lbps_finite),
        "mppi_lbps_reward_norm": float(lbps_reward_norm),
        "mppi_lbps_expected_return": float(lbps_expected_return),
        "mppi_mode_selection_temperature": float(selection_temperature),
        "mppi_mode_selection_inverse_temperature": float(selection_inverse_temperature),
        "mppi_mode_selection_lbps_effective_sample_size": float(selection_lbps_ess),
        "mppi_mode_selection_lbps_objective": float(selection_lbps_objective),
        "mppi_mode_selection_lbps_finite_rollouts": int(selection_lbps_finite),
        "mppi_mode_selection_lbps_reward_norm": float(selection_lbps_reward_norm),
        "mppi_mode_selection_lbps_expected_return": float(selection_lbps_expected_return),
        "mppi_mode_selection_scores": np.asarray(mode_scores, dtype=np.float64).copy(),
        "mppi_mode_weight_mass": np.asarray(mode_weight_mass, dtype=np.float64).copy(),
        "mppi_mode_ess": np.asarray(mode_effective_sample_size, dtype=np.float64).copy(),
        "mppi_mode_weighted_control_sequence": np.asarray(mode_weighted_sequences, dtype=np.float64).copy(),
        "mppi_mode_weighted_control_delta": np.asarray(mode_weighted_control_delta, dtype=np.float64).copy(),
        "mppi_mode_weighted_control_delta_norm": [
            float(np.linalg.norm(mode_weighted_control_delta[i])) for i in range(active_count)
        ],
        "mppi_mode_weighted_first_control": np.asarray(mode_weighted_sequences[:, 0, :], dtype=np.float64).copy(),
        "mppi_mode_weighted_first_control_delta": np.asarray(mode_weighted_control_delta[:, 0, :], dtype=np.float64).copy(),
        "mppi_pooled_weighted_nominal": np.asarray(pooled_weighted_nominal, dtype=np.float64).copy(),
        "mppi_global_weighted_control_delta": np.asarray(global_weighted_control_delta, dtype=np.float64).copy(),
        "mppi_global_weighted_control_delta_norm": float(np.linalg.norm(global_weighted_control_delta)),
        "mppi_global_weighted_first_control_delta": np.asarray(global_weighted_control_delta[0], dtype=np.float64).copy(),
        "mppi_collision_count_diagnostic": int(np.count_nonzero(collision_mask)),
        "mppi_nominal_delta_norm": float(np.linalg.norm(planned_sequence - baseline)),
        "mppi_cost_history": history,
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
    if packed_polygons is None:
        packed_polygons = obstacle_polygons_to_padded_arrays(obstacles)
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
    nominal_solutions = parallel_mode_nominals(
        model, x_current, local_modes, cfg,
        need_trajectory=(rep_type == REP_GAUSSIAN),
    )
    pooled_controls = np.empty((mode_count * rollouts_per_mode, cfg.horizon, 2), dtype=np.float64)

    for local_index, (original_index, local_mode) in enumerate(zip(active_indices, local_modes)):
        ilqr_nominal, control_positions, _, _, ilqr_positions = nominal_solutions[local_index]
        start_id = local_index * rollouts_per_mode
        end_id = start_id + rollouts_per_mode
        if rep_type == REP_GAUSSIAN:
            controls = _sample_gaussian_from_nominal(
                model, x_current, local_mode, ilqr_nominal, control_positions, rollouts_per_mode, cfg, rng,
                ilqr_positions=ilqr_positions,
            )
        else:
            controls = sample_controls_around_nominal(model, ilqr_nominal, rollouts_per_mode, cfg, rng)
        pooled_controls[start_id:end_id] = controls

    pooled_costs, pooled_collision, _ = _rollout_costs_and_collisions(
        model, x_current, pooled_controls, obstacle_circles, goal, cfg, packed_obstacles
    )
    pooled_costs = np.asarray(pooled_costs, dtype=np.float64)
    pooled_costs[np.asarray(pooled_collision, dtype=bool)] = np.inf

    selection_temperature_info = resolve_mppi_temperature(pooled_costs, cfg)
    (
        selection_temperature,
        selection_inverse_temperature,
        selection_lbps_ess,
        selection_lbps_objective,
        selection_lbps_finite,
        selection_lbps_reward_norm,
        selection_lbps_expected_return,
    ) = selection_temperature_info

    for local_index, original_index in enumerate(active_indices):
        global_mode = global_modes[int(original_index)]
        ilqr_nominal = nominal_solutions[local_index][0]
        start_id = local_index * rollouts_per_mode
        end_id = start_id + rollouts_per_mode
        controls = pooled_controls[start_id:end_id]
        costs = pooled_costs[start_id:end_id]
        collisions = pooled_collision[start_id:end_id]
        collision_count = int(np.count_nonzero(collisions))
        completed.append(
            {
                "score": float(softmin_score(costs, cfg, temperature=float(selection_temperature)))
                - float(selection_temperature) * math.log(max(float(probabilities[local_index]), 1e-12)),
                "mode_index": int(original_index),
                "signature": str(global_mode.signature),
                "probability": float(global_mode.probability),
                "collision_count_diagnostic": collision_count,
                "cost_min": float(np.min(costs)),
                "cost_mean": _finite_cost_mean(costs),
                "rollout_controls": np.asarray(controls, dtype=np.float64),
                "rollout_costs": np.asarray(costs, dtype=np.float64),
                "nominal_controls": np.asarray(ilqr_nominal, dtype=np.float64),
            }
        )

    best = min(completed, key=lambda record: record["score"])

    baseline = np.asarray(best["nominal_controls"], dtype=np.float64)
    baseline_cost, baseline_traj, _ = _single_sequence_evaluation(
        model, x_current, baseline, obstacle_circles, goal, cfg, packed_obstacles, packed_polygons
    )

    selected_costs = np.asarray(best["rollout_costs"], dtype=np.float64)
    selected_controls = np.asarray(best["rollout_controls"], dtype=np.float64)
    temperature_info = resolve_mppi_temperature(selected_costs, cfg)
    (
        temperature, inverse_temperature, lbps_ess, lbps_objective,
        lbps_finite, lbps_reward_norm, lbps_expected_return,
    ) = temperature_info
    selected_candidate = mppi_weighted_control_sequence(
        model, selected_costs, selected_controls, cfg, temperature=float(temperature)
    )

    (
        current, current_cost, current_traj, candidate_feasible, output_source,
        weighted_candidate_cost, weighted_candidate_collision, best_rollout_cost,
    ) = _select_feasible_mppi_output(
        model,
        x_current,
        baseline,
        baseline_cost,
        baseline_traj,
        selected_candidate,
        selected_controls,
        selected_costs,
        obstacle_circles,
        goal,
        cfg,
        packed_obstacles,
        packed_polygons,
    )
    history = [float(baseline_cost), float(current_cost)]
    accepted = int(candidate_feasible)
    planned_sequence = np.asarray(current, dtype=np.float64)
    final_cost = float(current_cost)
    final_traj = np.asarray(current_traj).copy()
    weights = mppi_weights(selected_costs, cfg, temperature=float(temperature))
    effective_sample_size = float(1.0 / np.sum(weights * weights)) if weights.size else 0.0

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
        "mode_selection_policy": "single_pass_mode_selection_mppi",
        "rollout_budget_per_mode": int(rollouts_per_mode),
        "rollout_budget_total": int(rollouts_per_mode * len(completed)),
        "rollouts_by_mode": [int(rollouts_per_mode) for _ in completed],
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
        "mppi_passes": 1,
        "mppi_candidate_accepted": int(accepted),
        "mppi_output_source": output_source,
        "mppi_weighted_candidate_cost": float(weighted_candidate_cost),
        "mppi_weighted_candidate_collision": bool(weighted_candidate_collision),
        "mppi_best_rollout_cost": float(best_rollout_cost),
        "mppi_effective_sample_size": float(effective_sample_size),
        "mppi_temperature_strategy": "lbps" if bool(getattr(cfg, "adaptive_temperature_lbps", False)) else "fixed",
        "mppi_temperature": float(temperature),
        "mppi_inverse_temperature": float(inverse_temperature),
        "mppi_lbps_delta": float(getattr(cfg, "lbps_delta", 0.5)),
        "mppi_lbps_objective": float(lbps_objective),
        "mppi_lbps_effective_sample_size": float(lbps_ess),
        "mppi_lbps_finite_rollouts": int(lbps_finite),
        "mppi_lbps_reward_norm": float(lbps_reward_norm),
        "mppi_lbps_expected_return": float(lbps_expected_return),
        "mppi_mode_selection_temperature": float(selection_temperature),
        "mppi_mode_selection_inverse_temperature": float(selection_inverse_temperature),
        "mppi_mode_selection_lbps_effective_sample_size": float(selection_lbps_ess),
        "mppi_mode_selection_lbps_objective": float(selection_lbps_objective),
        "mppi_mode_selection_lbps_finite_rollouts": int(selection_lbps_finite),
        "mppi_mode_selection_lbps_reward_norm": float(selection_lbps_reward_norm),
        "mppi_mode_selection_lbps_expected_return": float(selection_lbps_expected_return),
        "mppi_collision_count_diagnostic": int(np.count_nonzero(pooled_collision)),
        "mppi_nominal_delta_norm": float(np.linalg.norm(planned_sequence - baseline)),
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