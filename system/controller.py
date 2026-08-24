from __future__ import annotations
import math
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from numba import njit
Array = np.ndarray
NUMBA_AVAILABLE = True
VEHICLE_SYSTEMS = (('Ackermann', 'ackermann'), ('Four-wheel', 'four_wheel'))

def resolve_vehicle_model(model_name: str):
    key = str(model_name).strip().lower()
    if key == 'ackermann':
        from . import ackermann
        return ackermann
    if key == 'four_wheel':
        from . import four_wheel
        return four_wheel
    raise ValueError(f'Unsupported vehicle model: {model_name}')


class ControllerVariant(str, Enum):
    PLANNER_ILQR = 'planner_ilqr'
    SENSITIVITY_PROJECTED_GAUSSIAN_MPPI = 'sensitivity_projected_gaussian_prior_mppi'
    GAUSSIAN_PRIOR_MPPI = 'gaussian_prior_mppi'
    CORRIDOR_PRIOR_MPPI = 'corridor_prior_mppi'
    CONTROL_BANK_MPPI = 'control_bank_mppi'
    STANDARD_MPPI = 'standard_mppi'

@dataclass
class ControllerConfig:
    dt: float = 0.1
    horizon: int = 50
    num_rollouts: int = 64
    lambda_temperature: float = 4096
    adaptive_temperature_lbps: bool = True
    lbps_delta: float = 0.9
    lbps_optimizer_iterations: int = 32
    temporal_noise_smoothing: float = 0.3
    sigma_ref: float = 1.0
    spg_lookahead_steps: int = 10
    spg_pseudoinverse_damping: float = 1e-08
    spg_covariance_jitter: float = 1e-08
    robot_radius: float = 0.18
    hard_collision_clearance: float = 0.01
    suppress_blocked_modes: bool = True
    mode_blocking_clearance: float = 0.02
    mode_blocking_substeps: int = 2
    w_failure_terminal_distance: float = 0.1
    goal_tolerance: float = 0.1
    terminal_velocity_tolerance: float = 0.1
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
            raise ValueError('dt must be positive.')
        if self.lambda_temperature <= 0.0:
            raise ValueError('lambda_temperature must be positive.')
        if not 0.0 < float(self.lbps_delta) < 1.0:
            raise ValueError('lbps_delta must lie strictly between 0 and 1.')
        if self.spg_pseudoinverse_damping < 0.0 or self.spg_covariance_jitter < 0.0:
            raise ValueError('SPG damping and covariance jitter must be nonnegative.')
        if self.max_centerline_distance < 0.0:
            raise ValueError('max_centerline_distance must be nonnegative.')
        if self.w_failure_terminal_distance < 0.0:
            raise ValueError('w_failure_terminal_distance must be nonnegative.')
        if self.goal_tolerance < 0.0:
            raise ValueError('goal_tolerance must be nonnegative.')
        if self.terminal_velocity_tolerance < 0.0:
            raise ValueError('terminal_velocity_tolerance must be nonnegative.')

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

def prior_preview_step_distance(cfg: Any) -> float:
    max_speed = max(0.0, float(getattr(cfg, 'max_translational_speed', getattr(cfg, 'v_max', 2.8))))
    dt = float(getattr(cfg, 'dt', 0.12))
    return max(1e-06, max_speed * dt)
if njit is not None:

    @njit(cache=True)
    def _temporal_smooth_noise_nb(noise, alpha):
        beta = math.sqrt(max(0.0, 1.0 - alpha * alpha))
        for sample in range(noise.shape[0]):
            for t in range(1, noise.shape[1]):
                noise[sample, t, 0] = alpha * noise[sample, t - 1, 0] + beta * noise[sample, t, 0]
                noise[sample, t, 1] = alpha * noise[sample, t - 1, 1] + beta * noise[sample, t, 1]
        return noise

    @njit(cache=True)
    def _apply_projected_covariance_into_nb(standard_noise, projected, output):
        for t in range(projected.shape[0]):
            a = projected[t, 0, 0]
            b = 0.5 * (projected[t, 0, 1] + projected[t, 1, 0])
            d = projected[t, 1, 1]
            if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(d)):
                for sample in range(standard_noise.shape[0]):
                    output[sample, t, 0] = 0.0
                    output[sample, t, 1] = 0.0
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

    @njit(cache=True)
    def _apply_projected_covariance_nb(standard_noise, projected):
        output = np.empty_like(standard_noise)
        return _apply_projected_covariance_into_nb(standard_noise, projected, output)
else:
    _temporal_smooth_noise_nb = None
    _apply_projected_covariance_into_nb = None
    _apply_projected_covariance_nb = None

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
    return np.column_stack((np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])))

def snap_path_end_to_goal(path: Array, goal: Optional[Array], snap_radius: float=0.2, straight_tail_points: int=8) -> Array:
    p = np.asarray(path, dtype=np.float64)
    if goal is None or p.ndim != 2 or p.shape[1] != 2 or (len(p) < 2):
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

def _planner_symbols() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from geometry.utils import PolyObstacle, obstacles_to_segs, round_obstacle
        from graph.graph import build_full_graph
        from planner.env import FishGoalEnv2D
        from planner.planner import HomotopyAwareGenerativePlanner, trajectory_cost
    except Exception as exc:
        raise ImportError(f'Could not import the project planner modules. Run from the project root where geometry/, graph/, planner/, and save/ exist.\nOriginal import error: {exc}') from exc
    return (PolyObstacle, obstacles_to_segs, round_obstacle, build_full_graph, FishGoalEnv2D, (HomotopyAwareGenerativePlanner, trajectory_cost))

def fit_topological_trajectory_mixture(gen_out: Any, *, K: int=50, beta: float=1.0, min_mode_samples: int=3, covariance_jitter: float=0.0002, costmap: Any=None, bounds: tuple[tuple[float, float], tuple[float, float]]=((0.0, 10.0), (0.0, 10.0)), goal: Optional[Array]=None, snap_to_goal_radius: float=0.2, snap_straight_tail_points: int=8) -> TopologicalTrajectoryMixture:
    *_, planner_pair = _planner_symbols()
    _, trajectory_cost = planner_pair
    raw_paths = list(gen_out.samples)
    if not raw_paths:
        raise RuntimeError('Swarm planner produced zero trajectory samples.')
    all_paths = [snap_path_end_to_goal(path, goal=goal, snap_radius=snap_to_goal_radius, straight_tail_points=snap_straight_tail_points) for path in raw_paths]
    all_costs = np.asarray([trajectory_cost(path, costmap=costmap, bounds=bounds, w_len=1.0, w_smooth=0.0) for path in all_paths], dtype=np.float64)
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
        mode_weight = float(sum((weight_by_raw_id.get(id(path), 0.0) for path in paths)))
        total_mode_weight += mode_weight
        mode_raw[signature] = {'X': X, 'weights': weights, 'mean': mean, 'covariance': covariance, 'mode_weight': mode_weight, 'mean_cost': float(np.nanmean(costs))}
    if not mode_raw:
        raise RuntimeError('No homotopy group had enough samples.')
    if total_mode_weight <= 1e-12:
        total_mode_weight = float(len(mode_raw))
        for data in mode_raw.values():
            data['mode_weight'] = 1.0
    modes: Dict[Tuple[int, ...], GaussianTrajectoryMode] = {}
    for signature, data in mode_raw.items():
        modes[signature] = GaussianTrajectoryMode(signature=signature, probability=float(data['mode_weight'] / total_mode_weight), mean=data['mean'], cov=data['covariance'], samples=data['X'], weights=data['weights'], mean_cost=data['mean_cost'], count=int(data['X'].shape[0]))
    return TopologicalTrajectoryMixture(modes=modes, K=K, beta=beta)

def mixture_to_mppi_modes(mixture: TopologicalTrajectoryMixture) -> List[MPPIHomotopyMode]:
    modes: List[MPPIHomotopyMode] = []
    for signature, mode in mixture.modes.items():
        mean_path = mode.mean_path
        K = mean_path.shape[0]
        cov_blocks = np.zeros((K, 2, 2), dtype=np.float64)
        for t in range(K):
            cov_blocks[t] = mode.cov[2 * t:2 * t + 2, 2 * t:2 * t + 2]
        sample_paths = [unflatten_path(vector) for vector in mode.samples]
        modes.append(prepare_mode_prior_cache(MPPIHomotopyMode(signature=signature, probability=mode.probability, mean_path=mean_path, cov_blocks=cov_blocks, sample_paths=sample_paths)))
    modes.sort(key=lambda item: item.probability, reverse=True)
    return modes

def _poly_vertices(obstacle: Any) -> Array:
    if hasattr(obstacle, 'vertices'):
        return np.asarray(obstacle.vertices, dtype=np.float64)[:, :2]
    return np.asarray(obstacle, dtype=np.float64)[:, :2]

def obstacle_bounding_circles(obstacles: Sequence[Any], *, elongated_aspect_ratio: float=2.25, max_segment_length: float=0.1, wall_max_segment_length: float=0.15) -> List[Tuple[Array, float]]:
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
        major_min, major_max = (float(np.min(major_coord)), float(np.max(major_coord)))
        minor_min, minor_max = (float(np.min(minor_coord)), float(np.max(minor_coord)))
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
    max_vertices = max((len(poly) for poly in polygons))
    padded = np.zeros((len(polygons), max_vertices, 2), dtype=np.float64)
    lengths = np.zeros(len(polygons), dtype=np.int64)
    for i, poly in enumerate(polygons):
        padded[i, :len(poly)] = poly
        lengths[i] = len(poly)
    return (np.ascontiguousarray(padded), np.ascontiguousarray(lengths))

def make_wall_between_points(p0: Array, p1: Array, width: float=0.35, extension: float=0.0) -> Any:
    PolyObstacle, *_ = _planner_symbols()
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length <= 1e-12:
        raise ValueError('Cannot create wall: endpoints are identical.')
    tangent = delta / length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    a = p0 - extension * tangent
    b = p1 + extension * tangent
    half = 0.5 * float(width)
    return PolyObstacle(np.asarray([a + half * normal, b + half * normal, b - half * normal, a - half * normal]))

def build_default_scene() -> Scene:
    PolyObstacle, _, round_obstacle, *_ = _planner_symbols()
    bounds_xy = (np.asarray([0.0, 0.0]), np.asarray([10.0, 10.0]))
    polygons = (np.asarray([[3.5, 1.5], [5.7, 2.2], [5.2, 4.0], [3.3, 3.4]]), np.asarray([[6.2, 6.0], [8.5, 6.3], [8.1, 8.2], [6.8, 8.0], [5.9, 7.4]]), np.asarray([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]]), np.asarray([[1.2, 4.2], [2.1, 4.0], [2.4, 4.8], [1.7, 5.3], [1.1, 4.9]]), np.asarray([[4.6, 5.1], [5.4, 5.0], [5.8, 5.7], [5.0, 6.2], [4.4, 5.7]]), np.asarray([[7.9, 3.0], [9.0, 3.2], [8.8, 4.2], [7.7, 4.0]]), np.asarray([[5.7, 0.5], [6.6, 0.7], [6.4, 1.8], [5.6, 1.6]]))
    obstacles = tuple((PolyObstacle(round_obstacle(poly, n_iters=4, n_points=32)) for poly in polygons))
    return Scene(scale=4.0, bounds_xy=bounds_xy, planner_bounds=((0.0, 10.0), (0.0, 10.0)), start=np.asarray([1.0, 1.0]), goal=np.asarray([9.0, 9.0]), obstacles=obstacles)

def run_swarm_planner(start: Array, goal: Array, obstacles: Sequence[Any], scale: float, bounds_xy: Any, *, seed: int) -> Any:
    _, obstacles_to_segs, _, build_full_graph, FishGoalEnv2D, planner_pair = _planner_symbols()
    HomotopyAwareGenerativePlanner, _ = planner_pair
    segments = obstacles_to_segs(obstacles, scale=scale)
    with open('planner/policy.pkl', 'rb') as policy_file:
        action = pickle.load(policy_file)['best_theta']
    graph_goals, graph_weights = build_full_graph(obstacles=obstacles, start=start, goal=goal, scale=scale, bounds=bounds_xy)
    planner = HomotopyAwareGenerativePlanner(env_cls=FishGoalEnv2D, action=action, obstacles=obstacles, segs=segments, scale=scale, boid_count=1200, max_steps=700, dt=0.5)
    return planner.sample(start_unscaled=start, goal_unscaled=goal, graph_goals=graph_goals, graph_W=graph_weights, seed=seed)

def build_homotopy_modes(scene: Scene, obstacles: Sequence[Any], seed: int) -> List[MPPIHomotopyMode]:
    generated = run_swarm_planner(scene.start, scene.goal, obstacles, scene.scale, scene.bounds_xy, seed=seed)
    mixture = fit_topological_trajectory_mixture(generated, K=200, beta=1.0, min_mode_samples=1, bounds=scene.planner_bounds, goal=scene.goal)
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
            return (corrected, displacement, scalar_variance)
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
        return (corrected, displacement, scalar_variance)
else:
    _prior_second_moment_about_ilqr_nb = None

def _prior_second_moment_about_ilqr(model: Any, x_current: Array, local_mode: MPPIHomotopyMode, ilqr_nominal: Array, control_positions: Array, cfg: Any, *, ilqr_positions: Optional[Array]=None) -> Tuple[Array, Array, Array]:
    """Return center-corrected spatial spread about the iLQR proposal.

    The geometric covariance is defined about the path mean. MPPI is centered
    on the dynamically feasible iLQR trajectory, so the spatial second moment
    gains the full center-shift term d d^T. No directional component is removed.

    The interpolation, outer product, symmetrization and Gaussian scalarization
    all run in one Numba kernel.
    """
    positions = np.ascontiguousarray(np.asarray(control_positions, dtype=np.float64).reshape(-1))
    if ilqr_positions is None:
        nominal_states = np.asarray(model.rollout_single(x_current, np.asarray(ilqr_nominal, dtype=np.float64), cfg), dtype=np.float64)
        if nominal_states.ndim != 2 or nominal_states.shape[1] < 2:
            raise ValueError('model.rollout_single must return a 2-D state trajectory with planar position in columns 0:2.')
        if nominal_states.shape[0] < len(positions):
            raise ValueError(f'iLQR nominal rollout is shorter than the control-position sequence: {nominal_states.shape[0]} < {len(positions)}.')
        ilqr_xy = np.ascontiguousarray(nominal_states[:len(positions), :2])
    else:
        ilqr_xy = np.ascontiguousarray(np.asarray(ilqr_positions, dtype=np.float64))
        if ilqr_xy.ndim != 2 or ilqr_xy.shape[0] < len(positions) or ilqr_xy.shape[1] < 2:
            raise ValueError('ilqr_positions must have shape (>=H, >=2).')
        ilqr_xy = np.ascontiguousarray(ilqr_xy[:len(positions), :2])
    mean_path = np.ascontiguousarray(np.asarray(local_mode.mean_path, dtype=np.float64)[:, :2])
    cov_blocks = np.ascontiguousarray(np.asarray(local_mode.cov_blocks, dtype=np.float64))
    arc = np.ascontiguousarray(np.asarray(local_mode.arc_length, dtype=np.float64))
    corrected, displacement, scalar_variance = _prior_second_moment_about_ilqr_nb(mean_path, cov_blocks, arc, positions, ilqr_xy)
    return (corrected, displacement, scalar_variance)

def fill_temporally_correlated_noise(model: Any, n: int, H: int, cfg: Any, rng: np.random.Generator, out: Array, *, scale_override: Optional[Array]=None) -> Array:
    count = max(0, int(n))
    horizon = max(0, int(H))
    target = np.asarray(out, dtype=np.float64)
    if target.ndim != 3 or target.shape[0] < count or target.shape[1] < horizon or target.shape[2] != 2:
        raise ValueError('out must have shape (>=n, >=H, 2).')
    view = target[:count, :horizon]
    scale = np.asarray(model.control_noise_scale(cfg), dtype=np.float64)
    if scale_override is not None:
        scale = np.asarray(scale_override, dtype=np.float64)
    if scale.shape != (2,):
        raise ValueError('model.control_noise_scale(cfg) must return shape (2,).')
    if count == 0 or horizon == 0:
        return view
    rng.standard_normal(size=view.shape, out=view)
    view[:, :, 0] *= float(scale[0])
    view[:, :, 1] *= float(scale[1])
    return _temporal_smooth_noise_nb(view, float(cfg.temporal_noise_smoothing))

def make_temporally_correlated_noise(model: Any, n: int, H: int, cfg: Any, rng: np.random.Generator, *, scale_override: Optional[Array]=None) -> Array:
    noise = np.empty((int(n), int(H), 2), dtype=np.float64)
    return fill_temporally_correlated_noise(model, n, H, cfg, rng, noise, scale_override=scale_override)

def clip_controls_inplace(model: Any, controls: Array, cfg: Any) -> Array:
    target = np.asarray(controls, dtype=np.float64)
    clip_inplace = getattr(model, 'clip_control_batch_inplace', None)
    if clip_inplace is not None:
        return np.asarray(clip_inplace(target, cfg), dtype=np.float64)
    target[...] = np.asarray(model.clip_control_batch(target, cfg), dtype=np.float64)
    return target

def sample_controls_around_nominal_into(model: Any, nominal: Array, n: int, cfg: Any, rng: np.random.Generator, out: Array) -> Array:
    count = max(0, int(n))
    horizon = int(cfg.horizon)
    target = np.asarray(out, dtype=np.float64)
    if target.ndim != 3 or target.shape[0] < count or target.shape[1] < horizon or target.shape[2] != 2:
        raise ValueError('out must have shape (>=n, >=horizon, 2).')
    view = target[:count, :horizon]
    if count == 0:
        return view
    fill_temporally_correlated_noise(model, count, horizon, cfg, rng, view)
    center = np.asarray(nominal, dtype=np.float64)
    view += center[None, :, :]
    return clip_controls_inplace(model, view, cfg)

def _clip_controls(model: Any, raw_controls: Array, cfg: Any) -> Array:
    """Clip sampled controls to the model actuator limits."""
    clipped = np.ascontiguousarray(np.asarray(raw_controls, dtype=np.float64)).copy()
    clip_inplace = getattr(model, 'clip_control_batch_inplace', None)
    if clip_inplace is not None:
        return np.asarray(clip_inplace(clipped, cfg), dtype=np.float64)
    return np.asarray(model.clip_control_batch(clipped, cfg), dtype=np.float64)

def sample_controls_around_nominal(model: Any, nominal: Array, n: int, cfg: Any, rng: np.random.Generator) -> Array:
    if n <= 0:
        return np.zeros((0, cfg.horizon, 2), dtype=np.float64)
    center = np.asarray(nominal, dtype=np.float64)
    raw_controls = make_temporally_correlated_noise(model, n, cfg.horizon, cfg, rng)
    raw_controls += center[None, :, :]
    return _clip_controls(model, raw_controls, cfg)

def nominal_controls_and_arc_positions(model: Any, x_current: Array, path: Array, cfg: Any, cov_blocks: Optional[Array]=None) -> Tuple[Array, Array]:
    """Compute one cold spatially covariance-conditioned iLQR nominal."""
    combined = getattr(model, 'nominal_controls_and_arc_positions', None)
    if combined is not None:
        controls, positions = combined(x_current, path, cfg, cov_blocks)
        return (np.asarray(controls, dtype=np.float64), np.asarray(positions, dtype=np.float64))
    controls = model.nominal_controls_to_track_path(x_current, path, cfg, cov_blocks)
    positions = model.prior_control_arc_positions(x_current, path, cfg, cov_blocks)
    return (np.asarray(controls, dtype=np.float64), np.asarray(positions, dtype=np.float64))

def _solve_mode_nominal_worker(model: Any, x_current: Array, mode: MPPIHomotopyMode, cfg: Any, need_jacobians: bool, need_trajectory: bool):
    """Solve one iLQR prior and return only the data required by the proposal.

    Recent center-correction logic needs the final iLQR planar trajectory for
    Gaussian/SPG, but Corridor does not. Avoiding an unconditional extra nonlinear
    rollout keeps the non-covariance ablations on their original fast path.

    Updated model files expose the final iLQR planar trajectory directly from the
    Numba iLQR solve, so Gaussian/SPG do not need to propagate the same nominal a
    second time. Older models remain supported via a rollout_single fallback.
    """
    if need_jacobians:
        combined5 = getattr(model, 'nominal_controls_and_arc_positions_with_jacobians_and_trajectory', None)
        project = getattr(model, 'project_control_covariances_from_jacobians', None)
        if combined5 is not None and project is not None:
            controls, positions, A, B, ilqr_positions = combined5(x_current, mode.mean_path, cfg, mode.cov_blocks, None)
            return (np.asarray(controls, dtype=np.float64), np.asarray(positions, dtype=np.float64), np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64), np.ascontiguousarray(np.asarray(ilqr_positions, dtype=np.float64)[:, :2]))
        combined = getattr(model, 'nominal_controls_and_arc_positions_with_jacobians', None)
        if combined is not None and project is not None:
            controls, positions, A, B = combined(x_current, mode.mean_path, cfg, mode.cov_blocks, None)
            controls = np.asarray(controls, dtype=np.float64)
            positions = np.asarray(positions, dtype=np.float64)
            ilqr_positions = None
            if need_trajectory:
                states = np.asarray(model.rollout_single(x_current, controls, cfg), dtype=np.float64)
                ilqr_positions = np.ascontiguousarray(states[:len(positions), :2])
            return (controls, positions, np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64), ilqr_positions)
    if need_trajectory:
        combined3 = getattr(model, 'nominal_controls_and_arc_positions_with_trajectory', None)
        if combined3 is not None:
            controls, positions, ilqr_positions = combined3(x_current, mode.mean_path, cfg, mode.cov_blocks)
            return (np.asarray(controls, dtype=np.float64), np.asarray(positions, dtype=np.float64), None, None, np.ascontiguousarray(np.asarray(ilqr_positions, dtype=np.float64)[:, :2]))
    controls, positions = nominal_controls_and_arc_positions(model, x_current, mode.mean_path, cfg, mode.cov_blocks)
    controls = np.asarray(controls, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    ilqr_positions = None
    if need_trajectory:
        states = np.asarray(model.rollout_single(x_current, controls, cfg), dtype=np.float64)
        ilqr_positions = np.ascontiguousarray(states[:len(positions), :2])
    return (controls, positions, None, None, ilqr_positions)

def parallel_mode_nominals(model: Any, x_current: Array, local_modes: Sequence[MPPIHomotopyMode], cfg: Any, *, need_jacobians: bool=False, need_trajectory: bool=False) -> List[Tuple[Array, Array, Optional[Array], Optional[Array], Optional[Array]]]:
    if not local_modes:
        return []
    batch_solver = getattr(model, 'batch_nominal_solutions', None)
    if batch_solver is not None and len(local_modes) > 1:
        lengths = np.asarray([len(mode.mean_path) for mode in local_modes], dtype=np.int64)
        max_len = int(np.max(lengths))
        count = len(local_modes)
        refs = np.zeros((count, max_len, 2), dtype=np.float64)
        covs = np.zeros((count, max_len, 2, 2), dtype=np.float64)
        for m, mode in enumerate(local_modes):
            n = int(lengths[m])
            refs[m, :n] = np.asarray(mode.mean_path, dtype=np.float64)[:n, :2]
            covs[m, :n] = np.asarray(mode.cov_blocks, dtype=np.float64)[:n]
        controls, positions, As, Bs, trajectories = batch_solver(
            x_current,
            np.ascontiguousarray(refs),
            np.ascontiguousarray(lengths),
            cfg,
            np.ascontiguousarray(covs),
        )
        result = []
        for m in range(count):
            result.append((
                np.asarray(controls[m], dtype=np.float64),
                np.asarray(positions[m], dtype=np.float64),
                np.asarray(As[m], dtype=np.float64) if need_jacobians else None,
                np.asarray(Bs[m], dtype=np.float64) if need_jacobians else None,
                np.ascontiguousarray(np.asarray(trajectories[m], dtype=np.float64)) if need_trajectory else None,
            ))
        return result
    return [
        _solve_mode_nominal_worker(model, x_current, mode, cfg, need_jacobians, need_trajectory)
        for mode in local_modes
    ]

def _sample_gaussian_from_nominal(model: Any, x_current: Array, local_mode: MPPIHomotopyMode, ilqr_nominal: Array, control_positions: Array, n: int, cfg: Any, rng: np.random.Generator, *, ilqr_positions: Optional[Array]=None) -> Array:
    H = int(cfg.horizon)
    noise = make_temporally_correlated_noise(model, n, H, cfg, rng)
    corrected_cov, _, variance = _prior_second_moment_about_ilqr(model, x_current, local_mode, ilqr_nominal, control_positions, cfg, ilqr_positions=ilqr_positions)
    sigma_ref = max(float(cfg.sigma_ref), 1e-09)
    noise *= (np.sqrt(np.maximum(variance, 0.0)) / sigma_ref)[None, :, None]
    noise += np.asarray(ilqr_nominal, dtype=np.float64)[None, :, :]
    return _clip_controls(model, noise, cfg)

def _project_spg_covariances(model: Any, A: Optional[Array], B: Optional[Array], cov_at_controls: Array, cfg: Any, *, x_current: Optional[Array]=None, ilqr_nominal: Optional[Array]=None) -> Array:
    """Project spatial covariance through the raw local control-to-position Jacobian."""
    project_from_jacobians = getattr(model, 'project_control_covariances_from_jacobians', None)
    if A is not None and B is not None and (project_from_jacobians is not None):
        return np.asarray(project_from_jacobians(A, B, cov_at_controls, cfg), dtype=np.float64)
    if x_current is None or ilqr_nominal is None:
        raise ValueError('SPG fallback projection requires x_current and ilqr_nominal.')
    return np.asarray(model.project_control_covariances(x_current, ilqr_nominal, cov_at_controls, cfg), dtype=np.float64)

def _sample_spg_from_nominal(model: Any, x_current: Array, local_mode: MPPIHomotopyMode, ilqr_nominal: Array, control_positions: Array, A: Optional[Array], B: Optional[Array], n: int, cfg: Any, rng: np.random.Generator, *, ilqr_positions: Optional[Array]=None) -> Array:
    H = int(cfg.horizon)
    cov_at_controls, _, _ = _prior_second_moment_about_ilqr(model, x_current, local_mode, ilqr_nominal, control_positions, cfg, ilqr_positions=ilqr_positions)
    projected = _project_spg_covariances(model, A, B, cov_at_controls, cfg, x_current=x_current, ilqr_nominal=ilqr_nominal)
    standard_noise = make_temporally_correlated_noise(model, n, H, cfg, rng, scale_override=np.ones(2, dtype=np.float64))
    noise = _apply_projected_covariance_nb(standard_noise, np.asarray(projected, dtype=np.float64))
    noise += np.asarray(ilqr_nominal, dtype=np.float64)[None, :, :]
    return _clip_controls(model, noise, cfg)

def sample_gaussian_controls_with_nominal(model: Any, x_current: Array, local_mode: MPPIHomotopyMode, n: int, cfg: Any, rng: np.random.Generator, *, goal: Optional[Array]=None) -> Tuple[Array, Array]:
    del goal
    H = int(cfg.horizon)
    if n <= 0:
        return np.zeros((0, H, 2), dtype=np.float64), np.zeros((H, 2), dtype=np.float64)
    mode = prepare_mode_prior_cache(local_mode)
    nominal, positions, _, _, ilqr_positions = parallel_mode_nominals(model, x_current, [mode], cfg, need_trajectory=True)[0]
    controls = _sample_gaussian_from_nominal(model, x_current, mode, nominal, positions, n, cfg, rng, ilqr_positions=ilqr_positions)
    return controls, np.asarray(nominal, dtype=np.float64).copy()

def sample_sensitivity_projected_gaussian_controls_with_nominal(model: Any, x_current: Array, local_mode: MPPIHomotopyMode, n: int, cfg: Any, rng: np.random.Generator, *, goal: Optional[Array]=None) -> Tuple[Array, Array]:
    del goal
    H = int(cfg.horizon)
    if n <= 0:
        return np.zeros((0, H, 2), dtype=np.float64), np.zeros((H, 2), dtype=np.float64)
    mode = prepare_mode_prior_cache(local_mode)
    nominal, positions, A, B, ilqr_positions = parallel_mode_nominals(model, x_current, [mode], cfg, need_jacobians=True, need_trajectory=True)[0]
    controls = _sample_spg_from_nominal(model, x_current, mode, nominal, positions, A, B, n, cfg, rng, ilqr_positions=ilqr_positions)
    return controls, np.asarray(nominal, dtype=np.float64).copy()

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
            sum_wr += w * -c
        if sum_w <= 0.0 or sum_w2 <= 0.0:
            return (-math.inf, 0.0, -math.inf)
        ess = sum_w * sum_w / sum_w2
        expected_return = sum_wr / sum_w
        penalty = reward_norm * math.sqrt((1.0 - delta) / (delta * max(ess, 1e-300)))
        return (expected_return - penalty, ess, expected_return)

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
            return (fallback_temperature, fallback_alpha, 0.0, -math.inf, 0, 0.0, -math.inf)
        spread = max_cost - rho
        if finite_count <= 1 or spread <= 1e-12 or reward_norm <= 1e-15:
            score, ess, expected_return = _lbps_score_nb(costs, fallback_alpha, delta, rho, reward_norm)
            return (fallback_temperature, fallback_alpha, ess, score, finite_count, reward_norm, expected_return)
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
                score, ess, expected_return = _lbps_score_nb(costs, prev, delta, rho, reward_norm)
                temperature = 1.0 / max(prev, 1e-300)
                return (temperature, prev, ess, score, finite_count, reward_norm, expected_return)
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
        return (temperature, alpha, ess, score, finite_count, reward_norm, expected_return)

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
    _lbps_score_nb = None
    _lbps_temperature_nb = None
    _mppi_weighted_sequence_nb = None

def resolve_mppi_temperature(costs: Array, cfg: Any) -> Tuple[float, float, float, float, int, float, float]:
    """Return (lambda, alpha, ESS, LBPS score, finite N, ||R||inf, E[R]).

    Watson & Peters parameterize the Gibbs posterior by inverse temperature
    alpha. This code uses MPPI temperature lambda, so lambda = 1/alpha.
    """
    fallback = float(cfg.lambda_temperature)
    if not bool(getattr(cfg, 'adaptive_temperature_lbps', False)):
        values = np.asarray(costs, dtype=np.float64)
        finite_count = int(np.count_nonzero(np.isfinite(values)))
        return (fallback, 1.0 / fallback, float('nan'), float('nan'), finite_count, float('nan'), float('nan'))
    values = np.ascontiguousarray(np.asarray(costs, dtype=np.float64))
    delta = float(getattr(cfg, 'lbps_delta', 0.5))
    iterations = int(getattr(cfg, 'lbps_optimizer_iterations', 32))
    return _lbps_temperature_nb(values, fallback, delta, iterations)

def mppi_weighted_control_sequence(model: Any, costs: Array, controls: Array, cfg: Any, *, temperature: Optional[float]=None) -> Array:
    costs_arr = np.ascontiguousarray(np.asarray(costs, dtype=np.float64))
    controls_arr = np.ascontiguousarray(np.asarray(controls, dtype=np.float64))
    temp = float(cfg.lambda_temperature) if temperature is None else float(temperature)
    sequence = _mppi_weighted_sequence_nb(costs_arr, controls_arr, temp)
    clip_sequence = getattr(model, 'clip_control_sequence', None)
    if clip_sequence is not None:
        return np.asarray(clip_sequence(sequence, cfg), dtype=np.float64)
    return model.clip_control_batch(np.asarray(sequence, dtype=np.float64)[None, :, :], cfg)[0]
