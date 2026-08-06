from __future__ import annotations

import math
import pickle
import time
from dataclasses import dataclass, replace
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
    SENSITIVITY_PROJECTED_GAUSSIAN_MPPI = "sensitivity_projected_gaussian_prior_mppi"
    GAUSSIAN_PRIOR_MPPI = "gaussian_prior_mppi"
    CORRIDOR_PRIOR_MPPI = "corridor_prior_mppi"
    CONTROL_BANK_MPPI = "control_bank_mppi"
    MODE_SELECTING_GAUSSIAN_MPPI = "mode_selecting_gaussian_mppi"
    MODE_SELECTING_CORRIDOR_MPPI = "mode_selecting_corridor_mppi"
    STANDARD_MPPI = "standard_mppi"

@dataclass
class ControllerConfig:
    """Controller-only parameters shared by both vehicle models."""

    dt: float = 0.12
    horizon: int = 50
    num_rollouts: int = 64
    lambda_temperature: float = 2.2

    temporal_noise_smoothing: float = 0.72
    gaussian_covariance_scale: float = 2.0
    spg_lookahead_steps: int = 10
    spg_fd_accel: float = 0.05
    spg_fd_steering_rate: float = 0.05
    spg_pseudoinverse_damping: float = 0.001
    spg_covariance_jitter: float = 1e-8

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
    low_noise_proposal_count: int = 1
    low_noise_proposal_scale: float = 0.15

    mode_select_top_k: int = 4
    mode_select_rollouts_per_mode: int = 0
    max_nearby_prior_modes: int = 3
    max_centerline_distance: float = 4.0

    # Kept for configuration-file compatibility. Selection is now strict and
    # sequential, so these values are not used by nearby_mode_indices().
    nearby_prior_distance_slack: float = 0.75
    nearby_prior_blocked_penalty: float = 1.25

    def __post_init__(self) -> None:
        self.horizon = max(1, int(self.horizon))
        self.num_rollouts = max(1, int(self.num_rollouts))
        self.spg_lookahead_steps = max(1, int(self.spg_lookahead_steps))
        self.max_reference_index_advance = max(0, int(self.max_reference_index_advance))
        self.max_nearby_prior_modes = max(1, int(self.max_nearby_prior_modes))
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


if njit is not None:
    @njit(cache=True)
    def _localize_prior_horizon_nb(mean_path, cov_blocks, arc_length, gaussian_variance, start_index, horizon):
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
            fraction = 0.0 if H == 1 else t / (H - 1.0)
            target = s0 + fraction * (s1 - s0)
            while cursor + 1 < count and arc_length[cursor + 1] < target:
                cursor += 1
            right = min(cursor + 1, count - 1)
            left = cursor
            denominator = arc_length[right] - arc_length[left]
            alpha = 0.0 if denominator <= 1e-12 else (target - arc_length[left]) / denominator
            beta = 1.0 - alpha
            for row in range(2):
                local_mean[t, row] = beta * mean_path[left, row] + alpha * mean_path[right, row]
                for col in range(2):
                    local_cov[t, row, col] = beta * cov_blocks[left, row, col] + alpha * cov_blocks[right, row, col]
            local_gaussian[t] = beta * gaussian_variance[left] + alpha * gaussian_variance[right]
        return local_mean, local_cov, local_gaussian

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
    _localize_prior_horizon_nb = None
    _temporal_smooth_noise_nb = None
    _apply_projected_covariance_nb = None


# ---------------------------------------------------------------------------
# Prior construction and localization
# ---------------------------------------------------------------------------

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
    except Exception as exc:  # pragma: no cover - depends on the project tree
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
    previous_idx: Optional[int] = None,
    max_advance: Optional[int] = None,
) -> Tuple[MPPIHomotopyMode, int]:
    mode = _ensure_mode_prior_cache(mode)
    mean_path = np.asarray(mode.mean_path, dtype=np.float64)
    nearest_idx = int(np.argmin(np.sum((mean_path - np.asarray(x_current[:2])) ** 2, axis=1)))
    if previous_idx is None:
        index = nearest_idx
    else:
        index = max(int(previous_idx), nearest_idx)
        if max_advance is not None:
            index = min(index, int(previous_idx) + int(max_advance))
        index = min(index, len(mean_path) - 2)

    if _localize_prior_horizon_nb is not None:
        local_mean, local_cov, local_gaussian = _localize_prior_horizon_nb(
            np.asarray(mode.mean_path, dtype=np.float64),
            np.asarray(mode.cov_blocks, dtype=np.float64),
            np.asarray(mode.arc_length, dtype=np.float64),
            np.asarray(mode.gaussian_variance, dtype=np.float64),
            index,
            int(H),
        )
    else:
        source_s = np.linspace(float(mode.arc_length[index]), float(mode.arc_length[-1]), int(H))
        local_mean = np.column_stack(
            (
                np.interp(source_s, mode.arc_length, mode.mean_path[:, 0]),
                np.interp(source_s, mode.arc_length, mode.mean_path[:, 1]),
            )
        )
        local_cov = np.empty((H, 2, 2), dtype=np.float64)
        for row in range(2):
            for col in range(2):
                local_cov[:, row, col] = np.interp(source_s, mode.arc_length, mode.cov_blocks[:, row, col])
        local_gaussian = np.interp(source_s, mode.arc_length, mode.gaussian_variance)
    local_arc = np.zeros(H, dtype=np.float64)
    if H > 1:
        local_arc[1:] = np.cumsum(np.linalg.norm(np.diff(local_mean, axis=0), axis=1))
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


def localize_mode_for_state(mode: MPPIHomotopyMode, x_current: Array, H: int) -> MPPIHomotopyMode:
    return localize_mode_for_state_with_index(mode, x_current, H)[0]


def localize_path_for_state_with_index(
    path: Array,
    x_current: Array,
    H: int,
    previous_idx: Optional[int] = None,
    max_advance: Optional[int] = None,
) -> Tuple[Array, int]:
    p = np.asarray(path, dtype=np.float64)
    nearest_idx = int(np.argmin(np.linalg.norm(p - np.asarray(x_current[:2]), axis=1)))
    if previous_idx is None:
        index = nearest_idx
    else:
        index = max(int(previous_idx), nearest_idx)
        if max_advance is not None:
            index = min(index, int(previous_idx) + int(max_advance))
        index = min(index, len(p) - 2)
    tail = p[index:] if index < len(p) - 1 else p[-2:]
    return resample_path(tail, H), index


def localize_path_for_state(path: Array, x_current: Array, H: int) -> Array:
    return localize_path_for_state_with_index(path, x_current, H)[0]


# ---------------------------------------------------------------------------
# Obstacle and scenario utilities
# ---------------------------------------------------------------------------

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


def as_blocker_list(blocker_or_blockers: Any) -> List[Any]:
    if blocker_or_blockers is None:
        return []
    if isinstance(blocker_or_blockers, (list, tuple)):
        return list(blocker_or_blockers)
    return [blocker_or_blockers]


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
        np.asarray([[6.2, 6.0], [8.5, 6.3], [8.1, 8.4], [6.8, 8.9], [5.9, 7.4]]),
        np.asarray([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]]),
        np.asarray([[1.8, 4.2], [2.7, 4.0], [3.0, 4.8], [2.3, 5.3], [1.7, 4.9]]),
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
        K=50,
        beta=1.0,
        bounds=scene.planner_bounds,
        goal=scene.goal,
    )
    return mixture_to_mppi_modes(mixture)


# ---------------------------------------------------------------------------
# Proposal generation
# ---------------------------------------------------------------------------

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


def build_empirical_nominal_bank(
    model: Any,
    x_current: Array,
    global_mode: MPPIHomotopyMode,
    mean_nominal: Array,
    cfg: Any,
    rng: np.random.Generator,
    previous_idx: Optional[int] = None,
) -> List[Array]:
    bank = [mean_nominal]
    if not global_mode.sample_paths:
        return bank
    count = min(int(cfg.max_empirical_nominals_per_mode), len(global_mode.sample_paths))
    ids = rng.choice(len(global_mode.sample_paths), size=count, replace=False)
    for sample_id in ids:
        local_path, _ = localize_path_for_state_with_index(
            global_mode.sample_paths[int(sample_id)],
            x_current,
            cfg.horizon,
            previous_idx=previous_idx if cfg.use_monotonic_reference_progress else None,
            max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None,
        )
        bank.append(model.nominal_controls_to_track_path(x_current, local_path, cfg))
    return bank


def build_nominal_bank_for_mode(
    model: Any,
    x_current: Array,
    local_mode: MPPIHomotopyMode,
    global_mode: MPPIHomotopyMode,
    goal: Array,
    cfg: Any,
    rng: np.random.Generator,
    *,
    use_empirical_init: bool,
    use_mean_nominal: bool,
    previous_idx: Optional[int] = None,
) -> List[Array]:
    goal_nominal = model.nominal_controls_to_goal(x_current, goal, cfg)
    mean_nominal = (
        model.nominal_controls_to_track_path(x_current, local_mode.mean_path, cfg)
        if use_mean_nominal
        else goal_nominal
    )
    bank = (
        build_empirical_nominal_bank(
            model,
            x_current,
            global_mode,
            mean_nominal,
            cfg,
            rng,
            previous_idx=previous_idx,
        )
        if use_empirical_init
        else [mean_nominal]
    )
    if not any(np.allclose(candidate, goal_nominal) for candidate in bank):
        bank.append(goal_nominal)
    return bank


def sample_controls_from_nominal_bank(
    model: Any,
    nominal_bank: List[Array],
    n: int,
    cfg: Any,
    rng: np.random.Generator,
    *,
    prefer_empirical: bool = True,
) -> Array:
    if n <= 0:
        return np.zeros((0, cfg.horizon, 2), dtype=np.float64)
    if len(nominal_bank) == 1:
        bank_ids = np.zeros(n, dtype=np.int64)
    else:
        probabilities = np.ones(len(nominal_bank), dtype=np.float64)
        if prefer_empirical:
            probabilities[0] = max(1e-6, 1.0 - float(cfg.swarm_init_probability))
            probabilities[1:] = float(cfg.swarm_init_probability) / (len(nominal_bank) - 1)
        probabilities /= probabilities.sum()
        bank_ids = rng.choice(len(nominal_bank), size=n, p=probabilities)
    bank_array = np.asarray(nominal_bank, dtype=np.float64)
    controls = bank_array[bank_ids].copy()
    noise = make_temporally_correlated_noise(model, n, cfg.horizon, cfg, rng)
    controls += noise
    exact_count = min(len(nominal_bank), n)
    if exact_count:
        controls[:exact_count] = bank_array[:exact_count]
    cursor = exact_count
    low_noise_budget = min(max(0, int(cfg.low_noise_proposal_count)), n - cursor)
    for offset in range(low_noise_budget):
        nominal_index = offset % len(nominal_bank)
        controls[cursor] = nominal_bank[nominal_index] + float(cfg.low_noise_proposal_scale) * noise[cursor]
        cursor += 1
    return model.clip_control_batch(controls, cfg)


def sample_exact_control_bank(
    model: Any,
    x_current: Array,
    global_mode: MPPIHomotopyMode,
    fallback_nominal: Array,
    n: int,
    cfg: Any,
    rng: np.random.Generator,
    previous_idx: Optional[int] = None,
) -> Array:
    if n <= 0:
        return np.zeros((0, cfg.horizon, 2), dtype=np.float64)
    candidates: List[Array] = []
    if global_mode.sample_paths:
        for sample_id in rng.permutation(len(global_mode.sample_paths)):
            local_path, _ = localize_path_for_state_with_index(
                global_mode.sample_paths[int(sample_id)],
                x_current,
                cfg.horizon,
                previous_idx=previous_idx if cfg.use_monotonic_reference_progress else None,
                max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None,
            )
            candidates.append(model.nominal_controls_to_track_path(x_current, local_path, cfg))
    if not candidates:
        candidates = [np.asarray(fallback_nominal, dtype=np.float64).copy()]
    candidate_array = np.asarray(candidates, dtype=np.float64)
    controls = candidate_array[np.arange(n, dtype=np.int64) % len(candidate_array)].copy()
    return model.clip_control_batch(controls, cfg)


def sample_gaussian_controls(
    model: Any,
    x_current: Array,
    local_mode: MPPIHomotopyMode,
    n: int,
    cfg: Any,
    rng: np.random.Generator,
) -> Array:
    H = int(cfg.horizon)
    if n <= 0:
        return np.zeros((0, H, 2), dtype=np.float64)
    local_mode = _ensure_mode_prior_cache(local_mode)
    nominal = model.nominal_controls_to_track_path(x_current, local_mode.mean_path, cfg)
    noise = make_temporally_correlated_noise(model, n, H, cfg, rng)
    variance = np.asarray(local_mode.gaussian_variance, dtype=np.float64)
    floor_variance = float(cfg.sigma_floor) ** 2
    scale = float(cfg.gaussian_covariance_scale) * np.sqrt(np.maximum(variance, floor_variance)) / max(
        float(cfg.sigma_floor), 1e-9
    )
    noise *= scale[None, :, None]
    controls = nominal[None, :, :] + noise
    controls[0] = nominal
    return model.clip_control_batch(controls, cfg)


def sample_sensitivity_projected_gaussian_controls(
    model: Any,
    x_current: Array,
    local_mode: MPPIHomotopyMode,
    n: int,
    cfg: Any,
    rng: np.random.Generator,
) -> Array:
    H = int(cfg.horizon)
    if n <= 0:
        return np.zeros((0, H, 2), dtype=np.float64)
    nominal = model.nominal_controls_to_track_path(x_current, local_mode.mean_path, cfg)
    projected = model.project_control_covariances(
        x_current,
        nominal,
        np.asarray(local_mode.cov_blocks, dtype=np.float64),
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
    controls[0] = nominal
    return model.clip_control_batch(controls, cfg)


def ensure_direct_goal_prior(model: Any, controls: Array, x_current: Array, goal: Array, cfg: Any) -> Array:
    proposals = np.asarray(controls, dtype=np.float64).copy()
    if proposals.ndim != 3 or proposals.shape[0] == 0:
        return proposals
    proposals[-1] = model.nominal_controls_to_goal(x_current, goal, cfg)
    return model.clip_control_batch(proposals, cfg)


# ---------------------------------------------------------------------------
# Mode selection, MPPI, and execution
# ---------------------------------------------------------------------------

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
    # Preserve the prior behavior: when all samples collide, retain their finite
    # costs so MPPI still emits a command instead of producing an empty update.
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
    low_count = min(max(1, int(cfg.low_noise_proposal_count)), int(cfg.num_rollouts))
    controls[:low_count] = nominal[None, :, :] + float(cfg.low_noise_proposal_scale) * noise[:low_count]
    controls[0] = nominal
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
    use_empirical_init: bool,
    use_mean_nominal: bool,
    progress_by_mode: Optional[Dict[str, int]],
    obstacle_circles: Optional[List[Tuple[Array, float]]] = None,
    cached_mode_clearances: Optional[Array] = None,
    record_optimal_traj: Optional[bool] = None,
) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    if rep_type not in {REP_GAUSSIAN, REP_CORRIDOR, REP_CONTROL_BANK, REP_SENSITIVITY_PROJECTED_GAUSSIAN}:
        raise ValueError(f"Unsupported pooled proposal representation: {rep_type}")
    progress = {} if progress_by_mode is None else dict(progress_by_mode)
    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)
    if cached_mode_clearances is None:
        cached_mode_clearances = cached_mode_mean_clearances(model, global_modes, obstacle_circles, cfg)
    nearby_indices = nearby_mode_indices(global_modes, x_current, cfg, cached_mode_clearances)
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
                "mode_clearances": np.asarray(cached_mode_clearances).tolist(),
                "mode_filter_fallback": "standard_mppi",
                "retained_mode_indices": [],
                "retained_mode_clearances": [],
                "selected_rollout_mode_index": None,
            }
        )
        return control, info, progress

    candidate_global_modes = [global_modes[index] for index in nearby_indices]
    local_modes: List[MPPIHomotopyMode] = []
    new_progress = dict(progress)
    for mode in candidate_global_modes:
        key = str(mode.signature)
        previous = progress.get(key)
        local_mode, index = localize_mode_for_state_with_index(
            mode,
            x_current,
            cfg.horizon,
            previous_idx=previous if cfg.use_monotonic_reference_progress else None,
            max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None,
        )
        local_modes.append(local_mode)
        new_progress[key] = index

    mode_clearances = np.asarray(cached_mode_clearances, dtype=np.float64)[nearby_indices]
    total_budget = max(1, int(cfg.num_rollouts))
    active_count = min(len(local_modes), total_budget)
    active_local_modes = local_modes[:active_count]
    active_global_modes = candidate_global_modes[:active_count]
    active_global_indices = nearby_indices[:active_count]
    counts = balanced_rollout_counts(total_budget, active_count)
    mode_ids = np.concatenate(
        [np.full(count, mode_index, dtype=np.int64) for mode_index, count in enumerate(counts)]
    )
    all_costs = np.zeros(total_budget, dtype=np.float64)
    all_controls = np.zeros((total_budget, cfg.horizon, 2), dtype=np.float64)
    best_cost = 1e309
    best_traj: Optional[Array] = None
    best_mode_global_index: Optional[int] = None

    for mode_index, local_mode in enumerate(active_local_modes):
        ids = np.flatnonzero(mode_ids == mode_index)
        count = len(ids)
        if count == 0:
            continue
        global_mode = active_global_modes[mode_index]
        key = str(global_mode.signature)
        nominal_bank = build_nominal_bank_for_mode(
            model,
            x_current,
            local_mode,
            global_mode,
            goal,
            cfg,
            rng,
            use_empirical_init=use_empirical_init,
            use_mean_nominal=use_mean_nominal,
            previous_idx=progress.get(key),
        )
        if rep_type == REP_GAUSSIAN:
            controls = sample_gaussian_controls(model, x_current, local_mode, count, cfg, rng)
        elif rep_type == REP_SENSITIVITY_PROJECTED_GAUSSIAN:
            controls = sample_sensitivity_projected_gaussian_controls(model, x_current, local_mode, count, cfg, rng)
        elif rep_type == REP_CONTROL_BANK:
            controls = sample_exact_control_bank(
                model,
                x_current,
                global_mode,
                nominal_bank[0],
                count,
                cfg,
                rng,
                previous_idx=progress.get(key),
            )
        else:
            mean_nominal = model.nominal_controls_to_track_path(x_current, local_mode.mean_path, cfg)
            controls = sample_controls_from_nominal_bank(
                model,
                [mean_nominal],
                count,
                cfg,
                rng,
                prefer_empirical=False,
            )
        if rep_type != REP_CONTROL_BANK:
            controls = ensure_direct_goal_prior(model, controls, x_current, goal, cfg)
        states = model.rollout_batch(x_current, controls, cfg)
        costs = model.trajectory_costs(states, controls, obstacle_circles, goal, cfg)
        costs = reject_colliding_rollouts(model, costs, states, obstacle_circles, goal, cfg)
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
    info = {
        "cost_min": float(np.min(all_costs)),
        "cost_mean": float(np.mean(all_costs)),
        "soft_value": float(softmin_score(all_costs, cfg)),
        "rep_type": int(rep_type),
        "mode_selection": False,
        "selected_mode_index": None,
        "rollout_budget_total": total_budget,
        "rollouts_by_mode": counts,
        "active_mode_count": active_count,
        "suppressed_mode_count": int(len(global_modes) - active_count),
        "nearby_mode_count": len(candidate_global_modes),
        "mode_clearances": mode_clearances.tolist(),
        "retained_mode_indices": [int(index) for index in active_global_indices],
        "retained_mode_clearances": mode_clearances[:active_count].tolist(),
        "selected_rollout_mode_index": best_mode_global_index,
        "optimal_traj": best_traj,
        "planned_control_sequence": planned_sequence,
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
    obstacle_circles: Optional[List[Tuple[Array, float]]] = None,
    cached_mode_clearances: Optional[Array] = None,
    record_optimal_traj: bool = True,
) -> Tuple[Array, Dict[str, object], Dict[str, int]]:
    if rep_type not in {REP_GAUSSIAN, REP_CORRIDOR}:
        raise ValueError("Mode-selecting MPPI supports only Gaussian or corridor proposals.")
    progress = {} if progress_by_mode is None else dict(progress_by_mode)
    if obstacle_circles is None:
        obstacle_circles = obstacle_bounding_circles(obstacles)
    if cached_mode_clearances is None:
        cached_mode_clearances = cached_mode_mean_clearances(model, global_modes, obstacle_circles, cfg)
    nearby_indices = nearby_mode_indices(global_modes, x_current, cfg, cached_mode_clearances)
    if not nearby_indices:
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
                "mode_filter_fallback": "standard_mppi",
                "retained_mode_indices": [],
                "retained_mode_clearances": [],
                "selected_mode_index": None,
                "selected_rollout_mode_index": None,
            }
        )
        return control, info, progress

    top_k = min(max(1, int(cfg.mode_select_top_k)), len(nearby_indices))
    candidate_indices = nearby_indices[:top_k]
    new_progress = dict(progress)
    records: List[dict[str, Any]] = []
    for original_index in candidate_indices:
        global_mode = global_modes[original_index]
        key = str(global_mode.signature)
        previous = progress.get(key)
        local_mode, index = localize_mode_for_state_with_index(
            global_mode,
            x_current,
            cfg.horizon,
            previous_idx=previous if cfg.use_monotonic_reference_progress else None,
            max_advance=cfg.max_reference_index_advance if cfg.use_monotonic_reference_progress else None,
        )
        new_progress[key] = index
        records.append(
            {"original_mid": int(original_index), "global_mode": global_mode, "local_mode": local_mode}
        )

    configured = int(cfg.mode_select_rollouts_per_mode)
    rollouts_per_mode = configured if configured > 0 else max(1, int(cfg.num_rollouts))
    completed: List[dict[str, Any]] = []
    for record in records:
        local_mode = record["local_mode"]
        if rep_type == REP_GAUSSIAN:
            controls = sample_gaussian_controls(model, x_current, local_mode, rollouts_per_mode, cfg, rng)
        else:
            nominal = model.nominal_controls_to_track_path(x_current, local_mode.mean_path, cfg)
            controls = sample_controls_from_nominal_bank(
                model,
                [nominal],
                rollouts_per_mode,
                cfg,
                rng,
                prefer_empirical=False,
            )
        states = model.rollout_batch(x_current, controls, cfg)
        costs = model.trajectory_costs(states, controls, obstacle_circles, goal, cfg)
        collision_mask = model.collision_mask(states, obstacle_circles, goal, cfg)
        feasible_count = int(np.count_nonzero(~collision_mask))
        costs = reject_colliding_rollouts(model, costs, states, obstacle_circles, goal, cfg)
        planned_sequence = mppi_weighted_control_sequence(model, costs, controls, cfg)
        global_mode = record["global_mode"]
        completed.append(
            {
                "score": float(softmin_score(costs, cfg)),
                "mode_index": int(record["original_mid"]),
                "signature": str(global_mode.signature),
                "probability": float(global_mode.probability),
                "feasible_count": feasible_count,
                "cost_min": float(np.min(costs)),
                "cost_mean": float(np.mean(costs)),
                "optimal_traj": np.asarray(states[int(np.argmin(costs))]).copy() if record_optimal_traj else None,
                "planned_control_sequence": planned_sequence,
            }
        )

    feasible = [record for record in completed if record["feasible_count"] > 0]
    best = min(feasible if feasible else completed, key=lambda record: record["score"])
    mode_clearances = np.asarray(cached_mode_clearances, dtype=np.float64)[candidate_indices]
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
        "rollout_budget_per_mode": rollouts_per_mode,
        "rollout_budget_total": rollouts_per_mode * len(completed),
        "rollouts_by_mode": [rollouts_per_mode] * len(completed),
        "active_mode_count": len(completed),
        "suppressed_mode_count": int(len(global_modes) - len(completed)),
        "mode_clearances": mode_clearances.tolist(),
        "retained_mode_indices": [int(index) for index in candidate_indices],
        "retained_mode_clearances": mode_clearances.tolist(),
        "optimal_traj": best["optimal_traj"],
        "planned_control_sequence": best["planned_control_sequence"],
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
    mode_clearance_cache: Dict[Tuple[Tuple[float, float, float], ...], Array] = {}

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

        obstacle_key = obstacle_configuration_key(active_circles)
        if obstacle_key not in mode_clearance_cache:
            mode_clearance_cache[obstacle_key] = cached_mode_mean_clearances(model, modes, active_circles, cfg)
        mode_clearances = mode_clearance_cache[obstacle_key]

        if record:
            obstacle_history.append(list(active_obstacles))

        if variant == ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI:
            control, info, progress_by_mode = stable_swarm_mppi_step(
                model,
                state,
                modes,
                active_obstacles,
                scene.goal,
                cfg,
                rng,
                rep_type=REP_SENSITIVITY_PROJECTED_GAUSSIAN,
                use_empirical_init=False,
                use_mean_nominal=True,
                progress_by_mode=progress_by_mode,
                obstacle_circles=active_circles,
                cached_mode_clearances=mode_clearances,
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
                use_empirical_init=False,
                use_mean_nominal=True,
                progress_by_mode=progress_by_mode,
                obstacle_circles=active_circles,
                cached_mode_clearances=mode_clearances,
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
                use_empirical_init=False,
                use_mean_nominal=True,
                progress_by_mode=progress_by_mode,
                obstacle_circles=active_circles,
                cached_mode_clearances=mode_clearances,
                record_optimal_traj=record,
            )
        elif variant == ControllerVariant.CONTROL_BANK_MPPI:
            control, info, progress_by_mode = stable_swarm_mppi_step(
                model,
                state,
                modes,
                active_obstacles,
                scene.goal,
                cfg,
                rng,
                rep_type=REP_CONTROL_BANK,
                use_empirical_init=True,
                use_mean_nominal=False,
                progress_by_mode=progress_by_mode,
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
                obstacle_circles=active_circles,
                cached_mode_clearances=mode_clearances,
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
                obstacle_circles=active_circles,
                cached_mode_clearances=mode_clearances,
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


__all__ = [
    "ControllerConfig",
    "ControllerVariant",
    "DynamicWallScenario",
    "GaussianTrajectoryMode",
    "MPPIHomotopyMode",
    "Scene",
    "SimulationResult",
    "TopologicalTrajectoryMixture",
    "build_default_scene",
    "build_homotopy_modes",
    "default_dynamic_wall_scenarios",
    "localize_mode_for_state",
    "localize_path_for_state",
    "make_wall_blockers_between_centers",
    "obstacle_bounding_circles",
    "obstacle_center",
    "run_controller",
]
