from __future__ import annotations

import math
import pickle
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit
except Exception:
    njit = None

Array = np.ndarray
NUMBA_AVAILABLE = njit is not None


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
    dt: float = 0.12
    horizon: int = 50
    num_rollouts: int = 64
    lambda_temperature: float = 16

    temporal_noise_smoothing: float = 0.3
    prior_nominal_ilqr_ratio: float = 0.5

    sigma_ref: float = 1.0 # Gaussian

    spg_lookahead_steps: int = 10 # SPG
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
    w_control: float = 0.0
    w_control_smooth: float = 0.0
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
        self.spg_lookahead_steps = max(1, int(self.spg_lookahead_steps))
        self.max_nearby_prior_modes = max(1, int(self.max_nearby_prior_modes))
        self.centerline_history_points = max(1, int(self.centerline_history_points))
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.lambda_temperature <= 0.0:
            raise ValueError("lambda_temperature must be positive.")
        if not 0.0 <= float(self.prior_nominal_ilqr_ratio) <= 1.0:
            raise ValueError("prior_nominal_ilqr_ratio must be between 0 and 1.")
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
class DynamicWallScenario:
    scenario_id: str
    wall_pairs: tuple[tuple[int, int], ...]
    trigger_progress: float = 0.28
    wall_width: float = 0.35
    wall_extension: float = 0.15


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
        [trajectory_cost(path, costmap=costmap, bounds=bounds, w_len=1.0, w_smooth=0.05) for path in all_paths],
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

    local_mean = np.ascontiguousarray(mean_path[index : end + 1].copy())
    local_cov = np.ascontiguousarray(np.asarray(mode.cov_blocks, dtype=np.float64)[index : end + 1].copy())
    local_gaussian = np.ascontiguousarray(np.asarray(mode.gaussian_variance, dtype=np.float64)[index : end + 1].copy())
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

    ds = float(step_distance) if step_distance is not None else DEFAULT_PRIOR_REFERENCE_SPEED * 0.12
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
    extension: float = 0.15,
) -> List[Any]:
    fixed = [np.asarray(center, dtype=np.float64).reshape(2).copy() for center in centers]
    blockers: List[Any] = []
    for i, j in pairs:
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
    noise = rng.normal(size=(n, H, 2)) * scale[None, None, :]
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
    controls = np.repeat(center[None, :, :], n, axis=0)
    noise = make_temporally_correlated_noise(model, n, cfg.horizon, cfg, rng)
    controls += noise
    return model.clip_control_batch(controls, cfg)


def sample_exact_control_bank(
    model: Any,
    x_current: Array,
    global_mode: MPPIHomotopyMode,
    fallback_nominal: Array,
    n: int,
    cfg: Any,
) -> Array:
    """Return controls from the closest empirical trajectories needed by this mode.

    At most ``n`` unique sample paths are converted. Closeness is the minimum
    Euclidean distance from the current position to each complete empirical
    trajectory. The selected paths are localized first and converted to
    dynamically feasible iLQR controls.
    """
    if n <= 0:
        return np.zeros((0, cfg.horizon, 2), dtype=np.float64)

    paths = list(global_mode.sample_paths or [])
    if not paths:
        fallback = np.asarray(fallback_nominal, dtype=np.float64)[None, :, :]
        return model.clip_control_batch(np.repeat(fallback, n, axis=0), cfg)

    x_xy = np.asarray(x_current[:2], dtype=np.float64)
    path_count = len(paths)
    selected_count = min(int(n), path_count)
    distances = np.empty(path_count, dtype=np.float64)

    for sample_id, path in enumerate(paths):
        p = np.asarray(path, dtype=np.float64)
        if len(p) == 0:
            distances[sample_id] = np.inf
        else:
            delta = p[:, :2] - x_xy[None, :]
            distances[sample_id] = float(np.min(np.sum(delta * delta, axis=1)))

    selected_ids = np.argsort(distances, kind="stable")[:selected_count]
    localized: List[Array] = []
    for sample_id in selected_ids:
        local_path, _ = localize_path_for_state_with_index(
            paths[int(sample_id)],
            x_current,
            cfg.horizon,
            step_distance=prior_preview_step_distance(cfg),
        )
        localized.append(local_path)

    candidate_array = np.asarray(
        [model.nominal_controls_to_track_path(x_current, path, cfg) for path in localized],
        dtype=np.float64,
    )

    controls = candidate_array[np.arange(n, dtype=np.int64) % selected_count].copy()
    return model.clip_control_batch(controls, cfg)


def nominal_controls_and_arc_positions(
    model: Any,
    x_current: Array,
    path: Array,
    cfg: Any,
    cov_blocks: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Compute a spatially covariance-conditioned iLQR nominal once.

    ``cov_blocks`` is indexed by geometric path position, never by control time.
    The model projects every predicted state onto ``path`` and interpolates the
    covariance at that projected arc position.  The returned positions are the
    physical progress of the optimized rollout and are reused by Gaussian/SPG.
    """
    combined = getattr(model, "nominal_controls_and_arc_positions", None)
    if combined is not None:
        controls, positions = combined(x_current, path, cfg, cov_blocks)
        return np.asarray(controls, dtype=np.float64), np.asarray(positions, dtype=np.float64)
    controls = model.nominal_controls_to_track_path(x_current, path, cfg, cov_blocks)
    positions = model.prior_control_arc_positions(x_current, path, cfg, cov_blocks)
    return np.asarray(controls, dtype=np.float64), np.asarray(positions, dtype=np.float64)


def blended_prior_nominal_controls(
    model: Any,
    x_current: Array,
    goal: Optional[Array],
    ilqr_nominal: Array,
    cfg: Any,
) -> Array:
    """Blend the path-tracking iLQR nominal with the goal-centered MPPI nominal.

    ``prior_nominal_ilqr_ratio = 1`` gives pure iLQR centering, while ``0``
    gives the same goal-centered nominal used by standard MPPI.
    """
    ratio = float(np.clip(getattr(cfg, "prior_nominal_ilqr_ratio", 0.5), 0.0, 1.0))
    ilqr = np.asarray(ilqr_nominal, dtype=np.float64)
    if goal is None or ratio >= 1.0:
        return model.clip_control_batch(ilqr[None, :, :], cfg)[0]
    goal_nominal = np.asarray(model.nominal_controls_to_goal(x_current, goal, cfg), dtype=np.float64)
    if goal_nominal.shape != ilqr.shape:
        raise ValueError(
            f"Goal-centered nominal shape {goal_nominal.shape} does not match iLQR nominal shape {ilqr.shape}."
        )
    blended = ratio * ilqr + (1.0 - ratio) * goal_nominal
    return model.clip_control_batch(blended[None, :, :], cfg)[0]


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
    H = int(cfg.horizon)
    if n <= 0:
        empty_controls = np.zeros((0, H, 2), dtype=np.float64)
        empty_nominal = np.zeros((H, 2), dtype=np.float64)
        return empty_controls, empty_nominal

    local_mode = _ensure_mode_prior_cache(local_mode)
    ilqr_nominal, control_positions = nominal_controls_and_arc_positions(
        model,
        x_current,
        local_mode.mean_path,
        cfg,
        local_mode.cov_blocks,
    )
    nominal = blended_prior_nominal_controls(model, x_current, goal, ilqr_nominal, cfg)
    noise = make_temporally_correlated_noise(
        model, n, H, cfg, rng
    )
    variance = sample_dense_scalar_at_arc_positions(
        np.asarray(local_mode.gaussian_variance, dtype=np.float64),
        np.asarray(local_mode.arc_length, dtype=np.float64),
        control_positions,
    )
    sigma_ref = max(float(cfg.sigma_ref), 1e-9)
    scale = np.sqrt(np.maximum(variance, 0.0)) / sigma_ref
    noise *= scale[None, :, None]
    controls = nominal[None, :, :] + noise
    controls = model.clip_control_batch(controls, cfg)
    return controls, np.asarray(nominal, dtype=np.float64).copy()


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
    H = int(cfg.horizon)
    if n <= 0:
        empty_controls = np.zeros((0, H, 2), dtype=np.float64)
        empty_nominal = np.zeros((H, 2), dtype=np.float64)
        return empty_controls, empty_nominal
    ilqr_nominal, control_positions = nominal_controls_and_arc_positions(
        model, x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
    )
    nominal = blended_prior_nominal_controls(model, x_current, goal, ilqr_nominal, cfg)
    projected = model.project_control_covariances(
        x_current,
        nominal,
        sample_dense_covariance_at_arc_positions(
            np.asarray(local_mode.cov_blocks, dtype=np.float64),
            np.asarray(local_mode.arc_length, dtype=np.float64),
            control_positions,
        ),
        cfg,
    )
    standard_noise = make_temporally_correlated_noise(
        model,
        n,
        H,
        cfg,
        rng,
        scale_override=np.ones(2, dtype=np.float64),
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
    controls = nominal[None, :, :] + noise
    controls = model.clip_control_batch(controls, cfg)
    return controls, np.asarray(nominal, dtype=np.float64).copy()


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
) -> List[int]:
    """Sequentially keep the nearest feasible modes, up to the configured limit."""
    if not global_modes:
        return []
    position = np.asarray(x_current[:2], dtype=np.float64)
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
) -> Tuple[List[int], List[MPPIHomotopyMode], List[float], List[float], Dict[str, int]]:
    """Localize every prior mean and retain nearby, collision-free means.

    The centerline-distance gate is evaluated on the localized usable forward
    horizon for the current point and on the global centerline for the most recent
    ``cfg.centerline_history_points`` states.  No top-k pruning is applied: every mode within
    ``cfg.max_centerline_distance`` is considered, then blocked means are removed.
    Collision filtering is evaluated on the localized forward horizon, not on the
    complete global mean path.  Since the prior is geometric, this validity test
    uses the dense mean centerline against the true obstacle polygons, with no
    vehicle-footprint inflation.  Footprint-aware collision checking remains in
    the MPPI rollout model.
    """
    active_indices: List[int] = []
    active_modes: List[MPPIHomotopyMode] = []
    active_clearances: List[float] = []
    all_clearances: List[float] = []
    new_progress = dict(progress)
    initial_prior_pass = len(progress) == 0

    if (not initial_prior_pass) and cfg.suppress_blocked_modes and obstacles:
        polygons_padded, polygon_lengths = obstacle_polygons_to_padded_arrays(obstacles)
    else:
        polygons_padded = np.zeros((0, 0, 2), dtype=np.float64)
        polygon_lengths = np.zeros(0, dtype=np.int64)

    for global_index, mode in enumerate(global_modes):
        key = str(mode.signature)
        local_mode, index = localize_mode_for_state_with_index(
            mode,
            x_current,
            cfg.horizon,
            step_distance=prior_preview_step_distance(cfg),
        )
        new_progress[key] = index

        position = np.asarray(x_current[:2], dtype=np.float64)
        local_centerline = np.asarray(local_mode.mean_path, dtype=np.float64)
        if local_centerline.size == 0:
            centerline_distance = float("inf")
        else:
            delta = local_centerline[:, :2] - position[None, :]
            centerline_distance = float(np.sqrt(np.min(np.sum(delta * delta, axis=1))))

        # Preserve the existing current-point check against the localized forward
        # suffix, and additionally require the recent executed trajectory to stay
        # close to this mode's global centerline.
        if state_history:
            history_count = max(1, int(cfg.centerline_history_points))
            recent_positions = np.asarray(
                [np.asarray(state, dtype=np.float64)[:2] for state in state_history[-history_count:]],
                dtype=np.float64,
            )
            global_centerline = np.asarray(mode.mean_path, dtype=np.float64)[:, :2]
            if recent_positions.size == 0 or global_centerline.size == 0:
                history_centerline_distance = float("inf")
            else:
                history_delta = (
                    recent_positions[:, None, :] - global_centerline[None, :, :]
                )
                history_distances = np.sqrt(np.min(np.sum(history_delta * history_delta, axis=2), axis=1))
                history_centerline_distance = float(np.max(history_distances))
            centerline_distance = max(centerline_distance, history_centerline_distance)

        if initial_prior_pass:
            clearance = float("inf")
        elif cfg.suppress_blocked_modes and obstacles:
            clearance = geometric_mean_path_clearance_packed(local_mode.mean_path, polygons_padded, polygon_lengths)
        else:
            clearance = float("inf")

        all_clearances.append(clearance)

        if not initial_prior_pass:
            if centerline_distance > float(cfg.max_centerline_distance):
                continue
            if cfg.suppress_blocked_modes and obstacles and clearance <= 0.0:
                continue

        active_indices.append(int(global_index))
        active_modes.append(local_mode)
        active_clearances.append(clearance)

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


def reject_colliding_rollouts(
    model: Any,
    costs: Array,
    states: Array,
    obstacle_circles: Sequence[Tuple[Array, float]],
    goal: Array,
    cfg: Any,
) -> Array:
    if not obstacle_circles or states.shape[0] == 0:
        return costs
    colliding = model.collision_mask(states, obstacle_circles, goal, cfg)
    if np.all(colliding):
        return costs
    rejected = np.asarray(costs, dtype=np.float64).copy()
    rejected[colliding] = np.inf
    return rejected


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
    sequence = np.tensordot(mppi_weights(costs, cfg), controls, axes=(0, 0))
    return model.clip_control_batch(np.asarray(sequence, dtype=np.float64)[None, :, :], cfg)[0]


def best_output_trajectory_from_costs(costs: Array, states: Array) -> Array:
    return np.asarray(states[int(np.argmin(costs))], dtype=np.float64).copy()


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
    nominal = model.nominal_controls_to_goal(x_current, goal, cfg)
    controls = np.repeat(nominal[None, :, :], int(cfg.num_rollouts), axis=0)
    noise = make_temporally_correlated_noise(model, cfg.num_rollouts, cfg.horizon, cfg, rng)
    controls += noise
    controls = model.clip_control_batch(controls, cfg)
    states = model.rollout_batch(x_current, controls, cfg)
    costs = model.trajectory_costs(states, controls, obstacle_circles, goal, cfg)
    costs = reject_colliding_rollouts(model, costs, states, obstacle_circles, goal, cfg)
    planned_sequence = mppi_weighted_control_sequence(model, costs, controls, cfg)
    info: Dict[str, object] = {
        "cost_min": float(np.min(costs)),
        "cost_mean": float(np.mean(costs)),
        "optimal_traj": best_output_trajectory_from_costs(costs, states) if record_optimal_traj else None,
        "planned_control_sequence": planned_sequence,
        "selected_rollout_mode_index": None,
    }
    return planned_sequence[0].copy(), info


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

    nominal = model.nominal_controls_to_track_path(
        x_current,
        selected_local_mode.mean_path,
        cfg,
        selected_local_mode.cov_blocks,
    )
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
) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    if rep_type not in {REP_GAUSSIAN, REP_CORRIDOR, REP_CONTROL_BANK, REP_SENSITIVITY_PROJECTED_GAUSSIAN}:
        raise ValueError(f"Unsupported pooled proposal representation: {rep_type}")

    progress = {} if progress_by_mode is None else dict(progress_by_mode)
    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)

    total_budget = max(1, int(cfg.num_rollouts))
    compressed_mean_rep = rep_type in {
        REP_GAUSSIAN,
        REP_CORRIDOR,
        REP_SENSITIVITY_PROJECTED_GAUSSIAN,
    }

    if compressed_mean_rep:
        (
            active_global_indices,
            active_local_modes,
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
        )

        if not active_global_indices:
            control, info = standard_mppi_step(
                model,
                x_current,
                obstacles,
                goal,
                cfg,
                rng,
                obstacle_circles=obstacle_circles,
                record_optimal_traj=bool(record_optimal_traj),
            )
            info.update(
                {
                    "active_mode_count": 0,
                    "suppressed_mode_count": int(len(global_modes)),
                    "candidate_mode_count": int(len(global_modes)),
                    "nearby_mode_count": int(len(global_modes)),
                    "mode_clearances": list(all_local_clearances),
                    "mode_filter_fallback": "standard_mppi",
                    "retained_mode_indices": [],
                    "retained_mode_clearances": [],
                    "active_mode_probabilities": [],
                    "renormalized_mode_probabilities": [],
                    "rollouts_by_mode": [],
                    "selected_rollout_mode_index": None,
                }
            )
            return control, info, new_progress

        active_global_modes = [global_modes[index] for index in active_global_indices]
        probabilities = renormalized_mode_probabilities(active_global_modes)
        counts = probability_proportional_rollout_counts(total_budget, probabilities)
        selection_policy = "all_localized_collision_free_pi_weighted"

    else:
        if cached_mode_clearances is None:
            cached_mode_clearances = cached_mode_mean_clearances(
                model, global_modes, obstacle_circles, cfg
            )
        nearby_indices = nearby_mode_indices(
            global_modes, x_current, cfg, cached_mode_clearances
        )
        if not nearby_indices:
            control, info = standard_mppi_step(
                model,
                x_current,
                obstacles,
                goal,
                cfg,
                rng,
                obstacle_circles=obstacle_circles,
                record_optimal_traj=bool(record_optimal_traj),
            )
            info.update(
                {
                    "active_mode_count": 0,
                    "suppressed_mode_count": int(len(global_modes)),
                    "nearby_mode_count": 0,
                    "candidate_mode_count": 0,
                    "mode_clearances": np.asarray(cached_mode_clearances).tolist(),
                    "mode_filter_fallback": "standard_mppi",
                    "retained_mode_indices": [],
                    "retained_mode_clearances": [],
                    "active_mode_probabilities": [],
                    "renormalized_mode_probabilities": [],
                    "rollouts_by_mode": [],
                    "selected_rollout_mode_index": None,
                }
            )
            return control, info, progress

        candidate_global_modes = [global_modes[index] for index in nearby_indices]
        local_modes: List[MPPIHomotopyMode] = []
        new_progress = dict(progress)
        for mode in candidate_global_modes:
            key = str(mode.signature)
            local_mode, index = localize_mode_for_state_with_index(
                mode,
                x_current,
                cfg.horizon,
                step_distance=prior_preview_step_distance(cfg),
            )
            local_modes.append(local_mode)
            new_progress[key] = index

        active_count = min(len(local_modes), total_budget)
        active_local_modes = local_modes[:active_count]
        active_global_modes = candidate_global_modes[:active_count]
        active_global_indices = nearby_indices[:active_count]
        active_clearances = np.asarray(cached_mode_clearances, dtype=np.float64)[
            active_global_indices
        ].tolist()
        all_local_clearances = np.asarray(cached_mode_clearances, dtype=np.float64).tolist()
        probabilities = renormalized_mode_probabilities(active_global_modes)
        counts = balanced_rollout_counts(total_budget, active_count)
        selection_policy = "nearby_balanced_control_bank"

    active_count = len(active_global_modes)
    mode_ids = np.concatenate(
        [
            np.full(count, mode_index, dtype=np.int64)
            for mode_index, count in enumerate(counts)
            if count > 0
        ]
    ) if any(count > 0 for count in counts) else np.zeros(0, dtype=np.int64)

    if mode_ids.size != total_budget:
        raise RuntimeError(
            f"Internal rollout allocation error: allocated {mode_ids.size} of {total_budget} rollouts."
        )

    all_costs = np.zeros(total_budget, dtype=np.float64)
    all_controls = np.zeros((total_budget, cfg.horizon, 2), dtype=np.float64)
    best_cost = 1e309
    best_traj: Optional[Array] = None
    best_mode_global_index: Optional[int] = None
    nominal_controls_by_mode: Dict[int, Array] = {}

    for mode_index, local_mode in enumerate(active_local_modes):
        ids = np.flatnonzero(mode_ids == mode_index)
        count = len(ids)
        if count == 0:
            continue

        global_mode = active_global_modes[mode_index]

        if rep_type == REP_GAUSSIAN:
            controls, nominal = sample_gaussian_controls_with_nominal(
                model, x_current, local_mode, count, cfg, rng, goal=goal
            )
            nominal_controls_by_mode[mode_index] = model.nominal_controls_to_track_path(
                x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
            )
        elif rep_type == REP_SENSITIVITY_PROJECTED_GAUSSIAN:
            controls, nominal = sample_sensitivity_projected_gaussian_controls_with_nominal(
                model, x_current, local_mode, count, cfg, rng, goal=goal
            )
            nominal_controls_by_mode[mode_index] = model.nominal_controls_to_track_path(
                x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
            )
        elif rep_type == REP_CONTROL_BANK:
            fallback_nominal = model.nominal_controls_to_track_path(
                x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
            )
            controls = sample_exact_control_bank(
                model,
                x_current,
                global_mode,
                fallback_nominal,
                count,
                cfg,
            )
        else:
            ilqr_nominal = model.nominal_controls_to_track_path(
                x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
            )
            mean_nominal = blended_prior_nominal_controls(model, x_current, goal, ilqr_nominal, cfg)
            nominal_controls_by_mode[mode_index] = np.asarray(ilqr_nominal, dtype=np.float64).copy()
            controls = sample_controls_around_nominal(
                model,
                mean_nominal,
                count,
                cfg,
                rng,
            )

        states = model.rollout_batch(x_current, controls, cfg)
        costs = model.trajectory_costs(states, controls, obstacle_circles, goal, cfg)
        costs = reject_colliding_rollouts(
            model, costs, states, obstacle_circles, goal, cfg
        )
        all_costs[ids] = costs
        all_controls[ids] = controls

        local_best = int(np.argmin(costs))
        local_best_cost = float(costs[local_best])
        if np.isfinite(local_best_cost) and local_best_cost < best_cost:
            best_cost = local_best_cost
            best_mode_global_index = int(active_global_indices[mode_index])
            if record_optimal_traj:
                best_traj = np.asarray(states[local_best], dtype=np.float64).copy()

    planned_sequence = mppi_weighted_control_sequence(model, all_costs, all_controls, cfg)

    nominal_ilqr_traj: Optional[Array] = None
    if record_optimal_traj and best_mode_global_index is not None:
        selected_local_mode_index = next(
            (
                local_index
                for local_index, global_index in enumerate(active_global_indices)
                if int(global_index) == int(best_mode_global_index)
            ),
            None,
        )
        if selected_local_mode_index is not None:
            nominal_controls = nominal_controls_by_mode.get(int(selected_local_mode_index))
            if nominal_controls is not None:
                nominal_states = model.rollout_batch(
                    x_current,
                    np.asarray(nominal_controls, dtype=np.float64)[None, :, :],
                    cfg,
                )
                if nominal_states.shape[0] > 0:
                    nominal_ilqr_traj = np.asarray(nominal_states[0], dtype=np.float64).copy()

    info = {
        "cost_min": float(np.min(all_costs)),
        "cost_mean": float(np.mean(all_costs)),
        "soft_value": float(softmin_score(all_costs, cfg)),
        "rep_type": int(rep_type),
        "mode_selection": False,
        "mode_selection_policy": selection_policy,
        "selected_mode_index": None,
        "rollout_budget_total": total_budget,
        "rollouts_by_mode": [int(count) for count in counts],
        "active_mode_count": active_count,
        "suppressed_mode_count": int(len(global_modes) - active_count),
        "candidate_mode_count": int(len(global_modes) if compressed_mean_rep else active_count),
        "nearby_mode_count": int(len(global_modes) if compressed_mean_rep else active_count),
        "mode_clearances": [float(value) for value in all_local_clearances],
        "retained_mode_indices": [int(index) for index in active_global_indices],
        "retained_mode_clearances": [float(value) for value in active_clearances],
        "active_mode_probabilities": [
            float(global_modes[index].probability) for index in active_global_indices
        ],
        "renormalized_mode_probabilities": probabilities.tolist(),
        "selected_rollout_mode_index": best_mode_global_index,
        "optimal_traj": best_traj,
        "nominal_ilqr_traj": nominal_ilqr_traj,
        "planned_control_sequence": planned_sequence,
        "prior_nominal_ilqr_ratio": float(getattr(cfg, "prior_nominal_ilqr_ratio", 0.5)),
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
) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    if rep_type not in {REP_GAUSSIAN, REP_CORRIDOR}:
        raise ValueError("Mode-selecting MPPI supports only Gaussian or corridor proposals.")

    progress = {} if progress_by_mode is None else dict(progress_by_mode)
    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)

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
    records: List[dict[str, Any]] = [
        {
            "original_mid": int(original_index),
            "global_mode": global_modes[int(original_index)],
            "local_mode": local_mode,
        }
        for original_index, local_mode in zip(active_indices, local_modes)
    ]

    configured = int(cfg.mode_select_rollouts_per_mode)
    rollouts_per_mode = configured if configured > 0 else max(1, int(cfg.num_rollouts))
    completed: List[dict[str, Any]] = []

    for record in records:
        local_mode = record["local_mode"]
        if rep_type == REP_GAUSSIAN:
            controls, nominal = sample_gaussian_controls_with_nominal(
                model, x_current, local_mode, rollouts_per_mode, cfg, rng, goal=goal
            )
        else:
            ilqr_nominal = model.nominal_controls_to_track_path(
                x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
            )
            nominal = blended_prior_nominal_controls(model, x_current, goal, ilqr_nominal, cfg)
            controls = sample_controls_around_nominal(
                model,
                nominal,
                rollouts_per_mode,
                cfg,
                rng,
            )

        states = model.rollout_batch(x_current, controls, cfg)
        costs = model.trajectory_costs(states, controls, obstacle_circles, goal, cfg)
        collision_mask = model.collision_mask(states, obstacle_circles, goal, cfg)
        feasible_count = int(np.count_nonzero(~collision_mask))
        costs = reject_colliding_rollouts(
            model, costs, states, obstacle_circles, goal, cfg
        )
        planned_sequence = mppi_weighted_control_sequence(model, costs, controls, cfg)
        global_mode = record["global_mode"]
        display_ilqr_nominal = model.nominal_controls_to_track_path(
            x_current, local_mode.mean_path, cfg, local_mode.cov_blocks
        )
        completed.append(
            {
                "score": float(softmin_score(costs, cfg)),
                "mode_index": int(record["original_mid"]),
                "signature": str(global_mode.signature),
                "probability": float(global_mode.probability),
                "feasible_count": feasible_count,
                "cost_min": float(np.min(costs)),
                "cost_mean": float(np.mean(costs)),
                "optimal_traj": np.asarray(states[int(np.argmin(costs))]).copy()
                if record_optimal_traj
                else None,
                "planned_control_sequence": planned_sequence,
                "nominal_controls": np.asarray(display_ilqr_nominal, dtype=np.float64).copy(),
            }
        )

    feasible = [record for record in completed if record["feasible_count"] > 0]
    best = min(feasible if feasible else completed, key=lambda record: record["score"])

    nominal_ilqr_traj: Optional[Array] = None
    if record_optimal_traj:
        nominal_controls = best.get("nominal_controls")
        if nominal_controls is not None:
            nominal_states = model.rollout_batch(
                x_current,
                np.asarray(nominal_controls, dtype=np.float64)[None, :, :],
                cfg,
            )
            if nominal_states.shape[0] > 0:
                nominal_ilqr_traj = np.asarray(nominal_states[0], dtype=np.float64).copy()

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
        "mode_selection_policy": "all_localized_collision_free_separate_eval",
        "rollout_budget_per_mode": rollouts_per_mode,
        "rollout_budget_total": rollouts_per_mode * len(completed),
        "rollouts_by_mode": [rollouts_per_mode] * len(completed),
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
        "optimal_traj": best["optimal_traj"],
        "nominal_ilqr_traj": nominal_ilqr_traj,
        "planned_control_sequence": best["planned_control_sequence"],
        "prior_nominal_ilqr_ratio": float(getattr(cfg, "prior_nominal_ilqr_ratio", 0.5)),
    }
    return best["planned_control_sequence"][0].copy(), info, new_progress

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