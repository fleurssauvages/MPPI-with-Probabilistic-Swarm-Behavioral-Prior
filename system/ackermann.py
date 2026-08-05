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
    raise ImportError(f'Could not import your project modules. Run from the root of your project, where geometry/, RL/, graph/, planner.py, and save/ exist.\nOriginal import error: {exc}')
NUMBA_AVAILABLE = njit is not None
__all__ = ['ControllerVariant', 'MPPIConfig', 'Scene', 'SimulationResult', 'DynamicWallScenario', 'build_default_scene', 'default_dynamic_wall_scenarios', 'obstacle_center', 'make_wall_blockers_between_centers', 'build_homotopy_modes', 'run_controller', 'min_clearance', 'minimum_clearance', 'obstacle_bounding_circles', 'localize_mode_for_state', 'localize_path_for_state', 'NUMBA_AVAILABLE']
Array = np.ndarray

class ControllerVariant(str, Enum):
    SENSITIVITY_PROJECTED_GAUSSIAN_MPPI = 'sensitivity_projected_gaussian_prior_mppi'
    GAUSSIAN_PRIOR_MPPI = 'gaussian_prior_mppi'
    CORRIDOR_PRIOR_MPPI = 'corridor_prior_mppi'
    CONTROL_BANK_MPPI = 'control_bank_mppi'
    STANDARD_MPPI = 'standard_mppi'
    STANDARD_MPPI_128 = 'standard_mppi_128_rollouts'

@dataclass
class MPPIConfig:
    dt: float = 0.12
    horizon: int = 50
    num_rollouts: int = 64
    lambda_temperature: float = 2.2
    rear_axle_distance: float = 0.275
    front_axle_distance: float = 0.275
    mass: float = 18.0
    yaw_inertia: float = 1.2
    cornering_stiffness_front: float = 85.0
    cornering_stiffness_rear: float = 95.0
    tire_friction_coefficient: float = 0.95
    gravity: float = 9.81
    aerodynamic_drag_coefficient: float = 0.12
    rolling_resistance_force: float = 0.7
    minimum_tire_speed: float = 0.4
    dynamics_substeps: int = 4
    lateral_velocity_limit: float = 3.0
    yaw_rate_limit: float = 7.0
    v_min: float = -2.8
    v_max: float = 2.8
    accel_min: float = -3.5
    accel_max: float = 5.0
    steering_min: float = -1.2
    steering_max: float = 1.2
    steering_rate_min: float = -50.0
    steering_rate_max: float = 50.0
    noise_accel: float = 0.8
    noise_steering_rate: float = 1.0
    temporal_noise_smoothing: float = 0.72
    gaussian_covariance_scale: float = 2.0
    spg_lookahead_steps: int = 10
    spg_fd_accel: float = 0.05
    spg_fd_steering_rate: float = 0.05
    spg_pseudoinverse_damping: float = 0.001
    spg_covariance_jitter: float = 1e-08
    swarm_init_probability: float = 0.6
    max_empirical_nominals_per_mode: int = 16
    vehicle_length: float = 0.81
    vehicle_width: float = 0.36
    robot_radius: float = 0.18
    collision_substeps: int = 5
    hard_collision_clearance: float = 0.01
    hard_collision_penalty: float = 800000.0
    suppress_blocked_modes: bool = True
    mode_blocking_clearance: float = 0.02
    mode_blocking_substeps: int = 2
    w_goal: float = 110.0
    rollout_goal_tolerance: float = 0.305
    w_obstacle: float = 500.0
    w_boundary: float = 500.0
    boundary_xmin: float = 0.0
    boundary_xmax: float = 10.0
    boundary_ymin: float = 0.0
    boundary_ymax: float = 10.0
    w_control: float = 0.004
    w_control_smooth: float = 0.4
    sigma_floor: float = 0.25
    use_monotonic_reference_progress: bool = True
    max_reference_index_advance: int = 4
    max_delta_accel: float = 1.2
    max_delta_steering_rate: float = 5.2
    enforce_one_step_safety: bool = True
    one_step_safety_clearance: float = 0.0
    low_noise_proposal_count: int = 1
    low_noise_proposal_scale: float = 0.15
    max_nearby_prior_modes: int = 3
    nearby_prior_distance_slack: float = 0.75
    nearby_prior_blocked_penalty: float = 1.25

    @property
    def wheelbase(self) -> float:
        return self.front_axle_distance + self.rear_axle_distance

    def __post_init__(self) -> None:
        if self.wheelbase <= 0.0:
            raise ValueError('Ackermann axle distances must sum to a positive wheelbase.')
        positive = {'mass': self.mass, 'yaw_inertia': self.yaw_inertia, 'cornering_stiffness_front': self.cornering_stiffness_front, 'cornering_stiffness_rear': self.cornering_stiffness_rear, 'tire_friction_coefficient': self.tire_friction_coefficient, 'minimum_tire_speed': self.minimum_tire_speed, 'vehicle_length': self.vehicle_length, 'vehicle_width': self.vehicle_width}
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"These values must be positive: {', '.join(invalid)}")
        self.dynamics_substeps = max(1, int(self.dynamics_substeps))
        self.spg_lookahead_steps = max(1, int(self.spg_lookahead_steps))
        if self.spg_fd_accel <= 0.0 or self.spg_fd_steering_rate <= 0.0:
            raise ValueError('SPG finite-difference steps must be positive.')
        if self.spg_pseudoinverse_damping < 0.0 or self.spg_covariance_jitter < 0.0:
            raise ValueError('SPG damping and covariance jitter must be nonnegative.')
        for lower, upper, name in ((self.v_min, self.v_max, 'velocity'), (self.accel_min, self.accel_max, 'acceleration'), (self.steering_min, self.steering_max, 'steering'), (self.steering_rate_min, self.steering_rate_max, 'steering rate')):
            if lower > upper:
                raise ValueError(f'Invalid {name} bounds.')

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

def ackermann_rectangle_corners(state: Array, vehicle_length: float, vehicle_width: float) -> Array:
    """Return the four world-frame corners of the centered Ackermann body."""
    x, y, heading = map(float, np.asarray(state, dtype=np.float64)[:3])
    half_length = 0.5 * float(vehicle_length)
    half_width = 0.5 * float(vehicle_width)
    local = np.array([[-half_length, -half_width], [half_length, -half_width], [half_length, half_width], [-half_length, half_width]], dtype=np.float64)
    c = math.cos(heading)
    s = math.sin(heading)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + np.array([x, y], dtype=np.float64)

def rectangle_circle_clearance(state: Array, circle_center: Array, circle_radius: float, vehicle_length: float, vehicle_width: float) -> float:
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

def minimum_rectangle_circle_clearance(state: Array, centers: Array, radii: Array, vehicle_length: float, vehicle_width: float) -> float:
    state_array = np.asarray(state, dtype=np.float64)
    center_array = np.asarray(centers, dtype=np.float64)
    radius_array = np.asarray(radii, dtype=np.float64)
    if radius_array.size == 0:
        return 1e309
    if minimum_rectangle_circle_clearance_nb is not None:
        return float(minimum_rectangle_circle_clearance_nb(state_array, center_array, radius_array, float(vehicle_length), float(vehicle_width)))
    return float(min((rectangle_circle_clearance(state_array, center_array[j], radius_array[j], vehicle_length, vehicle_width) for j in range(len(radius_array)))))

def _point_in_oriented_rectangle(point: Array, state: Array, vehicle_length: float, vehicle_width: float) -> bool:
    x, y, heading = map(float, np.asarray(state, dtype=np.float64)[:3])
    dx = float(point[0] - x)
    dy = float(point[1] - y)
    c = math.cos(heading)
    s = math.sin(heading)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return bool(abs(local_x) <= 0.5 * float(vehicle_length) + 1e-12 and abs(local_y) <= 0.5 * float(vehicle_width) + 1e-12)

def _orientation_2d(a: Array, b: Array, c: Array) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

def _point_on_segment(point: Array, a: Array, b: Array, eps: float=1e-12) -> bool:
    if abs(_orientation_2d(a, b, point)) > eps:
        return False
    return bool(min(a[0], b[0]) - eps <= point[0] <= max(a[0], b[0]) + eps and min(a[1], b[1]) - eps <= point[1] <= max(a[1], b[1]) + eps)

def _segments_intersect(a: Array, b: Array, c: Array, d: Array) -> bool:
    o1 = _orientation_2d(a, b, c)
    o2 = _orientation_2d(a, b, d)
    o3 = _orientation_2d(c, d, a)
    o4 = _orientation_2d(c, d, b)
    if (o1 > 0.0 and o2 < 0.0 or (o1 < 0.0 and o2 > 0.0)) and (o3 > 0.0 and o4 < 0.0 or (o3 < 0.0 and o4 > 0.0)):
        return True
    return bool(abs(o1) <= 1e-12 and _point_on_segment(c, a, b) or (abs(o2) <= 1e-12 and _point_on_segment(d, a, b)) or (abs(o3) <= 1e-12 and _point_on_segment(a, c, d)) or (abs(o4) <= 1e-12 and _point_on_segment(b, c, d)))

def _point_segment_distance(point: Array, a: Array, b: Array) -> float:
    ab = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    denom = float(ab @ ab)
    if denom <= 1e-16:
        return float(np.linalg.norm(np.asarray(point) - np.asarray(a)))
    t = float(np.clip((np.asarray(point) - np.asarray(a)) @ ab / denom, 0.0, 1.0))
    closest = np.asarray(a) + t * ab
    return float(np.linalg.norm(np.asarray(point) - closest))

def _segment_segment_distance(a: Array, b: Array, c: Array, d: Array) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d), _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))

def rectangle_polygon_clearance(state: Array, obstacle, vehicle_length: float, vehicle_width: float) -> float:
    """Signed rectangle--polygon clearance; negative means overlap."""
    rectangle = ackermann_rectangle_corners(state, vehicle_length, vehicle_width)
    polygon = _poly_vertices(obstacle)
    if any((point_in_poly(corner, polygon) for corner in rectangle)):
        return -1e-09
    if any((_point_in_oriented_rectangle(vertex, state, vehicle_length, vehicle_width) for vertex in polygon)):
        return -1e-09
    minimum = 1e309
    for i in range(len(rectangle)):
        a = rectangle[i]
        b = rectangle[(i + 1) % len(rectangle)]
        for j in range(len(polygon)):
            c = polygon[j]
            d = polygon[(j + 1) % len(polygon)]
            distance = _segment_segment_distance(a, b, c, d)
            if distance <= 1e-12:
                return -1e-09
            minimum = min(minimum, distance)
    return minimum

def min_clearance(states: Array, obstacles: Sequence=0.18, vehicle_length: float=0.81, vehicle_width: float=0.36) -> float:
    """Minimum signed clearance using the oriented rectangular body."""
    state_array = np.asarray(states, dtype=np.float64)
    if state_array.size == 0 or not obstacles:
        return 1e309
    if min_clearance_nb is not None:
        padded, lengths = obstacles_to_padded_arrays(obstacles)
        return float(min_clearance_nb(state_array, padded, lengths, float(vehicle_length), float(vehicle_width)))
    values = [rectangle_polygon_clearance(state, obstacle, vehicle_length, vehicle_width) for state in state_array for obstacle in obstacles]
    return float(np.min(values)) if values else 1e309

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def _dynamic_ackermann_derivatives(state: Array, control: Array, cfg: MPPIConfig) -> Array:
    px, py, psi, vx, vy, yaw_rate, steering = np.asarray(state, dtype=np.float64)
    accel, steering_rate = np.asarray(control, dtype=np.float64)
    accel = float(np.clip(accel, cfg.accel_min, cfg.accel_max))
    steering_rate = float(np.clip(steering_rate, cfg.steering_rate_min, cfg.steering_rate_max))
    lf = float(cfg.front_axle_distance)
    lr = float(cfg.rear_axle_distance)
    wheelbase = max(lf + lr, 1e-09)
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
    rolling_force = float(cfg.rolling_resistance_force) * math.tanh(vx / 0.1)
    force_x = mass * accel - drag_force - rolling_force
    cos_psi = math.cos(psi)
    sin_psi = math.sin(psi)
    cos_delta = math.cos(steering)
    sin_delta = math.sin(steering)
    return np.array([vx * cos_psi - vy * sin_psi, vx * sin_psi + vy * cos_psi, yaw_rate, (force_x - force_y_front * sin_delta) / mass + yaw_rate * vy, (force_y_front * cos_delta + force_y_rear) / mass - yaw_rate * vx, (lf * force_y_front * cos_delta - lr * force_y_rear) / inertia, steering_rate], dtype=np.float64)

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
        state[4] = np.clip(state[4], -cfg.lateral_velocity_limit, cfg.lateral_velocity_limit)
        state[5] = np.clip(state[5], -cfg.yaw_rate_limit, cfg.yaw_rate_limit)
        state[6] = np.clip(state[6], cfg.steering_min, cfg.steering_max)
    return state

def goal_pose_satisfied(state: Array, goal: Array, goal_tolerance: float, cfg: MPPIConfig) -> bool:
    _ = cfg
    return bool(np.linalg.norm(np.asarray(state[:2]) - np.asarray(goal)) <= goal_tolerance)

def _dynamic_model_arguments(cfg: MPPIConfig) -> Tuple[float, ...]:
    return (float(cfg.dt), float(cfg.front_axle_distance), float(cfg.rear_axle_distance), float(cfg.mass), float(cfg.yaw_inertia), float(cfg.cornering_stiffness_front), float(cfg.cornering_stiffness_rear), float(cfg.tire_friction_coefficient), float(cfg.gravity), float(cfg.aerodynamic_drag_coefficient), float(cfg.rolling_resistance_force), float(cfg.minimum_tire_speed), int(cfg.dynamics_substeps), float(cfg.v_min), float(cfg.v_max), float(cfg.lateral_velocity_limit), float(cfg.yaw_rate_limit), float(cfg.accel_min), float(cfg.accel_max), float(cfg.steering_min), float(cfg.steering_max), float(cfg.steering_rate_min), float(cfg.steering_rate_max))

def rollout_ackermann(x0: Array, U: Array, cfg: MPPIConfig) -> Array:
    if rollout_ackermann_single_nb is not None:
        return rollout_ackermann_single_nb(np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), *_dynamic_model_arguments(cfg))
    X = np.zeros((len(U) + 1, 7), dtype=np.float64)
    X[0] = x0
    for t, u in enumerate(U):
        X[t + 1] = ackermann_step(X[t], u, cfg)
    return X

def rollout_ackermann_batch(x0: Array, U: Array, cfg: MPPIConfig) -> Array:
    if rollout_ackermann_batch_nb is not None:
        return rollout_ackermann_batch_nb(np.asarray(x0, dtype=np.float64), np.asarray(U, dtype=np.float64), *_dynamic_model_arguments(cfg))
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
    def _dynamic_ackermann_step_nb(state, accel_cmd, steering_rate_cmd, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        px = state[0]
        py = state[1]
        psi = state[2]
        vx = state[3]
        vy = state[4]
        yaw_rate = state[5]
        steering = state[6]
        accel = min(max(accel_cmd, accel_min), accel_max)
        steering_rate = min(max(steering_rate_cmd, steering_rate_min), steering_rate_max)
        substeps = max(1, int(dynamics_substeps))
        h = dt / substeps
        wheelbase = max(front_axle_distance + rear_axle_distance, 1e-09)
        normal_front = mass * gravity * rear_axle_distance / wheelbase
        normal_rear = mass * gravity * front_axle_distance / wheelbase
        front_limit = tire_friction_coefficient * normal_front
        rear_limit = tire_friction_coefficient * normal_rear
        for _ in range(substeps):
            slip_speed = max(abs(vx), minimum_tire_speed)
            speed_scale = min(1.0, abs(vx) / minimum_tire_speed)
            alpha_front = math.atan2(vy + front_axle_distance * yaw_rate, slip_speed) - steering
            alpha_rear = math.atan2(vy - rear_axle_distance * yaw_rate, slip_speed)
            force_y_front = -cornering_stiffness_front * alpha_front * speed_scale
            force_y_rear = -cornering_stiffness_rear * alpha_rear * speed_scale
            force_y_front = min(max(force_y_front, -front_limit), front_limit)
            force_y_rear = min(max(force_y_rear, -rear_limit), rear_limit)
            drag_force = aerodynamic_drag_coefficient * vx * abs(vx)
            rolling_force = rolling_resistance_force * math.tanh(vx / 0.1)
            force_x = mass * accel - drag_force - rolling_force
            cos_psi = math.cos(psi)
            sin_psi = math.sin(psi)
            cos_delta = math.cos(steering)
            sin_delta = math.sin(steering)
            px_dot = vx * cos_psi - vy * sin_psi
            py_dot = vx * sin_psi + vy * cos_psi
            psi_dot = yaw_rate
            vx_dot = (force_x - force_y_front * sin_delta) / mass + yaw_rate * vy
            vy_dot = (force_y_front * cos_delta + force_y_rear) / mass - yaw_rate * vx
            yaw_accel = (front_axle_distance * force_y_front * cos_delta - rear_axle_distance * force_y_rear) / yaw_inertia
            px += h * px_dot
            py += h * py_dot
            psi = _wrap_angle_nb(psi + h * psi_dot)
            vx = min(max(vx + h * vx_dot, v_min), v_max)
            vy = min(max(vy + h * vy_dot, -lateral_velocity_limit), lateral_velocity_limit)
            yaw_rate = min(max(yaw_rate + h * yaw_accel, -yaw_rate_limit), yaw_rate_limit)
            steering = min(max(steering + h * steering_rate, steering_min), steering_max)
        return (px, py, psi, vx, vy, yaw_rate, steering)

    @njit(cache=True)
    def rollout_ackermann_batch_nb(x0, U, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        N = U.shape[0]
        H = U.shape[1]
        X = np.zeros((N, H + 1, 7), dtype=np.float64)
        for n in range(N):
            for j in range(7):
                X[n, 0, j] = x0[j]
            for t in range(H):
                values = _dynamic_ackermann_step_nb(X[n, t], U[n, t, 0], U[n, t, 1], dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
                for j in range(7):
                    X[n, t + 1, j] = values[j]
        return X

    @njit(cache=True)
    def rollout_ackermann_single_nb(x0, U, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        H = U.shape[0]
        X = np.zeros((H + 1, 7), dtype=np.float64)
        for j in range(7):
            X[0, j] = x0[j]
        for t in range(H):
            values = _dynamic_ackermann_step_nb(X[t], U[t, 0], U[t, 1], dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
            for j in range(7):
                X[t + 1, j] = values[j]
        return X

    @njit(cache=True)
    def nominal_controls_to_track_path_nb(x0, ref, horizon, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
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
            desired_speed = min(max(0.2 + 2.4 * dist * heading_scale, 0.0), v_max)
            accel = min(max(3.0 * (desired_speed - x[3]), accel_min), accel_max)
            lookahead = max(dist, 0.35)
            curvature = 2.0 * math.sin(heading_error) / lookahead
            desired_steering = math.atan(wheelbase * curvature)
            desired_steering = min(max(desired_steering, steering_min), steering_max)
            steering_rate = 4.0 * (desired_steering - x[6]) - 0.15 * x[5]
            steering_rate = min(max(steering_rate, steering_rate_min), steering_rate_max)
            U[t, 0] = accel
            U[t, 1] = steering_rate
            values = _dynamic_ackermann_step_nb(x, accel, steering_rate, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
            for j in range(7):
                x[j] = values[j]
        return U

    @njit(cache=True)
    def nominal_controls_to_goal_nb(x0, goal, horizon, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
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
            desired_speed = min(max(0.2 + 2.2 * dist * heading_scale, 0.0), v_max)
            accel = min(max(3.0 * (desired_speed - x[3]), accel_min), accel_max)
            lookahead = max(dist, 0.35)
            curvature = 2.0 * math.sin(heading_error) / lookahead
            desired_steering = math.atan(wheelbase * curvature)
            desired_steering = min(max(desired_steering, steering_min), steering_max)
            steering_rate = 4.0 * (desired_steering - x[6]) - 0.15 * x[5]
            steering_rate = min(max(steering_rate, steering_rate_min), steering_rate_max)
            U[t, 0] = accel
            U[t, 1] = steering_rate
            values = _dynamic_ackermann_step_nb(x, accel, steering_rate, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
            for j in range(7):
                x[j] = values[j]
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
    def rectangle_circle_clearance_nb(px, py, heading, circle_x, circle_y, circle_radius, vehicle_length, vehicle_width):
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
    def minimum_rectangle_circle_clearance_nb(state, circle_centers, circle_radii, vehicle_length, vehicle_width):
        if circle_radii.shape[0] == 0:
            return 1e+18
        px = state[0]
        py = state[1]
        heading = state[2]
        best = 1e+18
        for j in range(circle_radii.shape[0]):
            clearance = rectangle_circle_clearance_nb(px, py, heading, circle_centers[j, 0], circle_centers[j, 1], circle_radii[j], vehicle_length, vehicle_width)
            if clearance < best:
                best = clearance
        return best

    @njit(cache=True)
    def path_min_clearance_to_circles_nb(path, circle_centers, circle_radii, vehicle_length, vehicle_width, substeps):
        count = path.shape[0]
        if count == 0 or circle_radii.shape[0] == 0:
            return 1e+18
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
                headings[i] = headings[i - 1] + _wrap_angle_nb(headings[i] - headings[i - 1])
        best = 1e+18
        for i in range(count):
            state = np.empty(3, dtype=np.float64)
            state[0] = path[i, 0]
            state[1] = path[i, 1]
            state[2] = headings[i]
            clearance = minimum_rectangle_circle_clearance_nb(state, circle_centers, circle_radii, vehicle_length, vehicle_width)
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
                    clearance = minimum_rectangle_circle_clearance_nb(state, circle_centers, circle_radii, vehicle_length, vehicle_width)
                    if clearance < best:
                        best = clearance
        return best

    @njit(cache=True)
    def rollout_collision_mask_nb(X, circle_centers, circle_radii, goal, vehicle_length, vehicle_width, hard_collision_clearance, rollout_goal_tolerance):
        N = X.shape[0]
        H = X.shape[1] - 1
        mask = np.zeros(N, dtype=np.bool_)
        goal_radius_sq = rollout_goal_tolerance * rollout_goal_tolerance
        for n in range(N):
            for t in range(H):
                state = X[n, t + 1]
                clearance = minimum_rectangle_circle_clearance_nb(state, circle_centers, circle_radii, vehicle_length, vehicle_width)
                if clearance < hard_collision_clearance:
                    mask[n] = True
                    break
                gx = state[0] - goal[0]
                gy = state[1] - goal[1]
                if gx * gx + gy * gy <= goal_radius_sq:
                    break
        return mask

    @njit(cache=True)
    def apply_smooth_safe_control_nb(x_current, u, previous_control, has_previous_control, circle_centers, circle_radii, max_delta_accel, max_delta_steering_rate, enforce_one_step_safety, one_step_safety_clearance, vehicle_length, vehicle_width, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        cmd = np.empty(2, dtype=np.float64)
        cmd[0] = u[0]
        cmd[1] = u[1]
        if has_previous_control:
            delta_accel = cmd[0] - previous_control[0]
            delta_accel = min(max(delta_accel, -max_delta_accel), max_delta_accel)
            delta_steering_rate = cmd[1] - previous_control[1]
            delta_steering_rate = min(max(delta_steering_rate, -max_delta_steering_rate), max_delta_steering_rate)
            cmd[0] = previous_control[0] + delta_accel
            cmd[1] = previous_control[1] + delta_steering_rate
        cmd[0] = min(max(cmd[0], accel_min), accel_max)
        cmd[1] = min(max(cmd[1], steering_rate_min), steering_rate_max)
        if enforce_one_step_safety and circle_radii.shape[0] > 0:
            values = _dynamic_ackermann_step_nb(x_current, cmd[0], cmd[1], dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
            x_next = np.empty(7, dtype=np.float64)
            for j in range(7):
                x_next[j] = values[j]
            current_clearance = minimum_rectangle_circle_clearance_nb(x_current, circle_centers, circle_radii, vehicle_length, vehicle_width)
            next_clearance = minimum_rectangle_circle_clearance_nb(x_next, circle_centers, circle_radii, vehicle_length, vehicle_width)
            moving_deeper = next_clearance < current_clearance - 0.0001
            below_required = next_clearance < one_step_safety_clearance
            if below_required and moving_deeper:
                if x_current[3] > 0.0:
                    cmd[0] = accel_min
                else:
                    cmd[0] = min(0.0, cmd[0])
        return cmd

    @njit(cache=True)
    def standard_mppi_costs_batch_nb(X, U, circle_centers, circle_radii, goal, horizon, vehicle_length, vehicle_width, w_goal, rollout_goal_tolerance, w_obstacle, w_control, w_control_smooth):
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
                cost += w_goal / H * goal_distance_sq
                yaw_rate = X[n, t + 1, 5]
                steering_angle = X[n, t + 1, 6]
                heading = X[n, t + 1, 2]
                for j in range(M):
                    clearance = rectangle_circle_clearance_nb(px, py, heading, circle_centers[j, 0], circle_centers[j, 1], circle_radii[j], vehicle_length, vehicle_width)
                    sp = _softplus_scalar_nb(8.0 * (0.0 - clearance))
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
    def boundary_penalty_nb(X, xmin, xmax, ymin, ymax, vehicle_length, vehicle_width, w_boundary, collision_substeps, hard_collision_clearance, hard_collision_penalty):
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
                    clearance = min(px - xmin - extent_x, xmax - px - extent_x, py - ymin - extent_y, ymax - py - extent_y)
                    if q <= substeps:
                        sp = _softplus_scalar_nb(8.0 * (0.0 - clearance))
                        cost += w_boundary / denominator * sp * sp
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
                x_cross = xi + (py - yi) * (xj - xi) / (yj - yi + 1e-18)
                if px < x_cross:
                    inside = not inside
        return inside

    @njit(cache=True)
    def point_in_oriented_rectangle_nb(px, py, cx, cy, heading, half_length, half_width):
        dx = px - cx
        dy = py - cy
        c = math.cos(heading)
        s = math.sin(heading)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        return abs(local_x) <= half_length + 1e-12 and abs(local_y) <= half_width + 1e-12

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
        return min(ax, bx) - 1e-12 <= px <= max(ax, bx) + 1e-12 and min(ay, by) - 1e-12 <= py <= max(ay, by) + 1e-12

    @njit(cache=True)
    def segments_intersect_nb(ax, ay, bx, by, cx, cy, dx, dy):
        o1 = orientation_2d_nb(ax, ay, bx, by, cx, cy)
        o2 = orientation_2d_nb(ax, ay, bx, by, dx, dy)
        o3 = orientation_2d_nb(cx, cy, dx, dy, ax, ay)
        o4 = orientation_2d_nb(cx, cy, dx, dy, bx, by)
        if (o1 > 0.0 and o2 < 0.0 or (o1 < 0.0 and o2 > 0.0)) and (o3 > 0.0 and o4 < 0.0 or (o3 < 0.0 and o4 > 0.0)):
            return True
        return abs(o1) <= 1e-12 and point_on_segment_nb(cx, cy, ax, ay, bx, by) or (abs(o2) <= 1e-12 and point_on_segment_nb(dx, dy, ax, ay, bx, by)) or (abs(o3) <= 1e-12 and point_on_segment_nb(ax, ay, cx, cy, dx, dy)) or (abs(o4) <= 1e-12 and point_on_segment_nb(bx, by, cx, cy, dx, dy))

    @njit(cache=True)
    def segment_segment_distance_nb(ax, ay, bx, by, cx, cy, dx, dy):
        if segments_intersect_nb(ax, ay, bx, by, cx, cy, dx, dy):
            return 0.0
        return min(point_segment_dist_nb(ax, ay, cx, cy, dx, dy), point_segment_dist_nb(bx, by, cx, cy, dx, dy), point_segment_dist_nb(cx, cy, ax, ay, bx, by), point_segment_dist_nb(dx, dy, ax, ay, bx, by))

    @njit(cache=True)
    def rectangle_polygon_clearance_nb(state, poly, n, vehicle_length, vehicle_width):
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
                return -1e-09
        for i in range(n):
            if point_in_oriented_rectangle_nb(poly[i, 0], poly[i, 1], cx, cy, heading, half_length, half_width):
                return -1e-09
        best = 1e+18
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
                distance = segment_segment_distance_nb(ax, ay, bx, by, poly[j, 0], poly[j, 1], poly[nj, 0], poly[nj, 1])
                if distance <= 1e-12:
                    return -1e-09
                if distance < best:
                    best = distance
        return best

    @njit(cache=True)
    def min_clearance_nb(states, polys_padded, poly_lengths, vehicle_length, vehicle_width):
        best = 1e+18
        for state_index in range(states.shape[0]):
            state = states[state_index]
            for polygon_index in range(poly_lengths.shape[0]):
                n = int(poly_lengths[polygon_index])
                clearance = rectangle_polygon_clearance_nb(state, polys_padded[polygon_index], n, vehicle_length, vehicle_width)
                if clearance < best:
                    best = clearance
        return best

    @njit(cache=True)
    def sensitivity_projected_covariances_nb(x0, nominal_controls, position_covariances, lookahead_steps, fd_accel, fd_steering_rate, pseudoinverse_damping, covariance_jitter, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max):
        """Project position covariance into control space with Eq. (26)."""
        horizon = nominal_controls.shape[0]
        nominal_states = rollout_ackermann_single_nb(x0, nominal_controls, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
        projected = np.zeros((horizon, 2, 2), dtype=np.float64)
        damping_sq = pseudoinverse_damping * pseudoinverse_damping
        for t in range(horizon):
            interval = min(max(1, int(lookahead_steps)), horizon - t)
            jacobian = np.zeros((2, 2), dtype=np.float64)
            for control_index in range(2):
                delta = fd_accel if control_index == 0 else fd_steering_rate
                lower = accel_min if control_index == 0 else steering_rate_min
                upper = accel_max if control_index == 0 else steering_rate_max
                center = nominal_controls[t, control_index]
                plus_value = min(center + delta, upper)
                minus_value = max(center - delta, lower)
                denominator = plus_value - minus_value
                if denominator <= 1e-12:
                    continue
                plus_state = nominal_states[t].copy()
                minus_state = nominal_states[t].copy()
                for k in range(interval):
                    accel_plus = nominal_controls[t + k, 0]
                    steer_plus = nominal_controls[t + k, 1]
                    accel_minus = accel_plus
                    steer_minus = steer_plus
                    if k == 0:
                        if control_index == 0:
                            accel_plus = plus_value
                            accel_minus = minus_value
                        else:
                            steer_plus = plus_value
                            steer_minus = minus_value
                    plus_values = _dynamic_ackermann_step_nb(plus_state, accel_plus, steer_plus, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
                    minus_values = _dynamic_ackermann_step_nb(minus_state, accel_minus, steer_minus, dt, front_axle_distance, rear_axle_distance, mass, yaw_inertia, cornering_stiffness_front, cornering_stiffness_rear, tire_friction_coefficient, gravity, aerodynamic_drag_coefficient, rolling_resistance_force, minimum_tire_speed, dynamics_substeps, v_min, v_max, lateral_velocity_limit, yaw_rate_limit, accel_min, accel_max, steering_min, steering_max, steering_rate_min, steering_rate_max)
                    for state_index in range(7):
                        plus_state[state_index] = plus_values[state_index]
                        minus_state[state_index] = minus_values[state_index]
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
    rollout_ackermann_batch_nb = None
    rollout_ackermann_single_nb = None
    nominal_controls_to_track_path_nb = None
    nominal_controls_to_goal_nb = None
    temporal_smooth_noise_nb = None
    standard_mppi_costs_batch_nb = None
    boundary_penalty_nb = None
    minimum_rectangle_circle_clearance_nb = None
    path_min_clearance_to_circles_nb = None
    rollout_collision_mask_nb = None
    apply_smooth_safe_control_nb = None
    min_clearance_nb = None
    sensitivity_projected_covariances_nb = None

def obstacle_circles_to_arrays(obstacle_circles: List[Tuple[Array, float]]) -> Tuple[Array, Array]:
    if not obstacle_circles:
        return (np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64))
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    return (centers, radii)

def apply_smooth_safe_control(x_current: Array, u: Array, previous_control: Optional[Array], obstacle_circles: List[Tuple[Array, float]], cfg: MPPIConfig) -> Array:
    state = np.asarray(x_current, dtype=np.float64)
    command = np.asarray(u, dtype=np.float64)
    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    if apply_smooth_safe_control_nb is not None:
        has_previous = previous_control is not None
        previous = np.asarray(previous_control, dtype=np.float64) if has_previous else np.zeros(2, dtype=np.float64)
        return apply_smooth_safe_control_nb(state, command, previous, has_previous, centers, radii, float(cfg.max_delta_accel), float(cfg.max_delta_steering_rate), bool(cfg.enforce_one_step_safety), float(cfg.one_step_safety_clearance), float(cfg.vehicle_length), float(cfg.vehicle_width), *_dynamic_model_arguments(cfg))
    cmd = command.copy()
    if previous_control is not None:
        previous = np.asarray(previous_control, dtype=np.float64)
        cmd[0] = previous[0] + float(np.clip(cmd[0] - previous[0], -cfg.max_delta_accel, cfg.max_delta_accel))
        cmd[1] = previous[1] + float(np.clip(cmd[1] - previous[1], -cfg.max_delta_steering_rate, cfg.max_delta_steering_rate))
    cmd[0] = np.clip(cmd[0], cfg.accel_min, cfg.accel_max)
    cmd[1] = np.clip(cmd[1], cfg.steering_rate_min, cfg.steering_rate_max)
    if cfg.enforce_one_step_safety and len(radii) > 0:
        x_next = ackermann_step(state, cmd, cfg)
        current_clearance = minimum_rectangle_circle_clearance(state, centers, radii, cfg.vehicle_length, cfg.vehicle_width)
        next_clearance = minimum_rectangle_circle_clearance(x_next, centers, radii, cfg.vehicle_length, cfg.vehicle_width)
        if next_clearance < cfg.one_step_safety_clearance and next_clearance < current_clearance - 0.0001:
            cmd[0] = cfg.accel_min if state[3] > 0.0 else min(0.0, cmd[0])
    return cmd

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

def path_min_clearance_to_circles(path: Array, obstacle_circles: List[Tuple[Array, float]], vehicle_length: float, vehicle_width: float, substeps: int=2) -> float:
    p = np.asarray(path, dtype=np.float64)
    if len(p) == 0 or not obstacle_circles:
        return 1e309
    centers, radii = obstacle_circles_to_arrays(obstacle_circles)
    if path_min_clearance_to_circles_nb is not None:
        return float(path_min_clearance_to_circles_nb(p, centers, radii, float(vehicle_length), float(vehicle_width), int(substeps)))
    headings = _path_tangent_headings(p)
    states = [np.column_stack([p, headings])]
    if len(p) > 1:
        count = max(0, int(substeps))
        for q in range(1, count + 1):
            alpha = q / float(count + 1)
            positions = p[:-1] + alpha * (p[1:] - p[:-1])
            delta_heading = np.array([wrap_angle(headings[i + 1] - headings[i]) for i in range(len(headings) - 1)])
            interp_heading = headings[:-1] + alpha * delta_heading
            states.append(np.column_stack([positions, interp_heading]))
    minimum = 1e309
    for state in np.vstack(states):
        minimum = min(minimum, minimum_rectangle_circle_clearance(state, centers, radii, vehicle_length, vehicle_width))
    return minimum

def unblocked_mode_indices(local_modes: Sequence[MPPIHomotopyMode], obstacle_circles: List[Tuple[Array, float]], cfg: MPPIConfig) -> Tuple[List[int], Array]:
    if not cfg.suppress_blocked_modes or len(local_modes) <= 1:
        return (list(range(len(local_modes))), np.full(len(local_modes), np.nan, dtype=np.float64))
    clearances = np.asarray([path_min_clearance_to_circles(mode.mean_path, obstacle_circles, cfg.vehicle_length, cfg.vehicle_width, substeps=cfg.mode_blocking_substeps) for mode in local_modes], dtype=np.float64)
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

def nominal_controls_to_track_path(x0: Array, ref: Array, cfg: MPPIConfig) -> Array:
    if nominal_controls_to_track_path_nb is not None:
        return nominal_controls_to_track_path_nb(np.asarray(x0, dtype=np.float64), np.asarray(ref, dtype=np.float64), int(cfg.horizon), *_dynamic_model_arguments(cfg))
    U = np.zeros((cfg.horizon, 2), dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64).copy()
    for t in range(cfg.horizon):
        target = ref[min(t + 3, len(ref) - 1)]
        delta = target - x[:2]
        distance = float(np.linalg.norm(delta))
        desired_heading = math.atan2(delta[1], delta[0])
        heading_error = wrap_angle(desired_heading - x[2])
        heading_scale = max(0.0, math.cos(heading_error)) ** 2
        desired_speed = np.clip(0.2 + 2.4 * distance * heading_scale, 0.0, cfg.v_max)
        accel = np.clip(3.0 * (desired_speed - x[3]), cfg.accel_min, cfg.accel_max)
        lookahead = max(distance, 0.35)
        desired_curvature = 2.0 * math.sin(heading_error) / lookahead
        desired_steering = np.clip(math.atan(cfg.wheelbase * desired_curvature), cfg.steering_min, cfg.steering_max)
        steering_rate = np.clip(4.0 * (desired_steering - x[6]), cfg.steering_rate_min, cfg.steering_rate_max)
        U[t] = [accel, steering_rate]
        x = ackermann_step(x, U[t], cfg)
    return U

def nominal_controls_to_goal(x0: Array, goal: Array, cfg: MPPIConfig) -> Array:
    if nominal_controls_to_goal_nb is not None:
        return nominal_controls_to_goal_nb(np.asarray(x0, dtype=np.float64), np.asarray(goal, dtype=np.float64), int(cfg.horizon), *_dynamic_model_arguments(cfg))
    U = np.zeros((cfg.horizon, 2), dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64).copy()
    for t in range(cfg.horizon):
        delta = goal - x[:2]
        distance = float(np.linalg.norm(delta))
        desired_heading = math.atan2(delta[1], delta[0])
        heading_error = wrap_angle(desired_heading - x[2])
        heading_scale = max(0.0, math.cos(heading_error)) ** 2
        desired_speed = np.clip(0.2 + 2.2 * distance * heading_scale, 0.0, cfg.v_max)
        accel = np.clip(3.0 * (desired_speed - x[3]), cfg.accel_min, cfg.accel_max)
        lookahead = max(distance, 0.35)
        desired_curvature = 2.0 * math.sin(heading_error) / lookahead
        desired_steering = np.clip(math.atan(cfg.wheelbase * desired_curvature), cfg.steering_min, cfg.steering_max)
        steering_rate = np.clip(4.0 * (desired_steering - x[6]), cfg.steering_rate_min, cfg.steering_rate_max)
        U[t] = [accel, steering_rate]
        x = ackermann_step(x, U[t], cfg)
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

def boundary_penalty(X: Array, cfg: MPPIConfig) -> Array:
    states = np.asarray(X, dtype=np.float64)
    if states.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    if boundary_penalty_nb is not None:
        return boundary_penalty_nb(states, float(cfg.boundary_xmin), float(cfg.boundary_xmax), float(cfg.boundary_ymin), float(cfg.boundary_ymax), float(cfg.vehicle_length), float(cfg.vehicle_width), float(cfg.w_boundary), int(cfg.collision_substeps), float(cfg.hard_collision_clearance), float(cfg.hard_collision_penalty))
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
                px, py = states[n, t, :2] + alpha * (states[n, t + 1, :2] - states[n, t, :2])
                heading = wrap_angle(h0 + alpha * dh)
                c = abs(math.cos(heading))
                s = abs(math.sin(heading))
                extent_x = half_length * c + half_width * s
                extent_y = half_length * s + half_width * c
                clearance = min(px - cfg.boundary_xmin - extent_x, cfg.boundary_xmax - px - extent_x, py - cfg.boundary_ymin - extent_y, cfg.boundary_ymax - py - extent_y)
                if q <= substeps:
                    extras[n] += cfg.w_boundary / denominator * float(softplus(8.0 * (0.0 - clearance)) ** 2)
                if clearance < cfg.hard_collision_clearance:
                    penetration = cfg.hard_collision_clearance - clearance
                    extras[n] += cfg.hard_collision_penalty * (1.0 + penetration ** 2)
    return extras

def standard_mppi_costs_batch(X: Array, U: Array, obstacle_circles: List[Tuple[Array, float]], goal: Array, cfg: MPPIConfig) -> Array:
    if standard_mppi_costs_batch_nb is not None:
        centers, radii = obstacle_circles_to_arrays(obstacle_circles)
        costs = standard_mppi_costs_batch_nb(np.asarray(X, dtype=np.float64), np.asarray(U, dtype=np.float64), centers, radii, np.asarray(goal, dtype=np.float64), int(cfg.horizon), float(cfg.vehicle_length), float(cfg.vehicle_width), float(cfg.w_goal), float(cfg.rollout_goal_tolerance), float(cfg.w_obstacle), float(cfg.w_control), float(cfg.w_control_smooth))
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
            costs[n] += cfg.w_goal / H * goal_distance_sq
            steering_angle = float(state[6])
            yaw_rate = float(state[5])
            for center, radius in obstacle_circles:
                clearance = rectangle_circle_clearance(state, center, radius, cfg.vehicle_length, cfg.vehicle_width)
                costs[n] += cfg.w_obstacle * float(softplus(8.0 * (0.0 - clearance)) ** 2)
            if goal_distance_sq <= goal_radius_sq:
                arrival_index = t + 1
                break
        prefix = U[n, :arrival_index]
        costs[n] += cfg.w_control * np.sum(prefix[:, 0] ** 2 + 0.15 * prefix[:, 1] ** 2)
        if arrival_index > 1:
            dU = np.diff(prefix, axis=0)
            costs[n] += cfg.w_control_smooth * np.sum(dU[:, 0] ** 2 + 0.2 * dU[:, 1] ** 2)
    return costs + boundary_penalty(X, cfg)
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

def enforce_ackermann_control_bounds(U: Array, cfg: MPPIConfig) -> Array:
    U = np.asarray(U, dtype=np.float64)
    if U.size == 0:
        return U
    U[:, :, 0] = np.clip(U[:, :, 0], cfg.accel_min, cfg.accel_max)
    U[:, :, 1] = np.clip(U[:, :, 1], cfg.steering_rate_min, cfg.steering_rate_max)
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
    return enforce_ackermann_control_bounds(U, cfg)

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
    return enforce_ackermann_control_bounds(U, cfg)

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
    return enforce_ackermann_control_bounds(U, cfg)

def sensitivity_projected_control_covariances(x_current: Array, nominal_controls: Array, position_covariances: Array, cfg: MPPIConfig) -> Array:
    """Compute Sigma_u,t = J_t^dagger Sigma_p,t J_t^dagger.T."""
    x0 = np.asarray(x_current, dtype=np.float64)
    nominal = np.asarray(nominal_controls, dtype=np.float64)
    covariances = np.asarray(position_covariances, dtype=np.float64)
    if sensitivity_projected_covariances_nb is not None:
        return sensitivity_projected_covariances_nb(x0, nominal, covariances, int(cfg.spg_lookahead_steps), float(cfg.spg_fd_accel), float(cfg.spg_fd_steering_rate), float(cfg.spg_pseudoinverse_damping), float(cfg.spg_covariance_jitter), *_dynamic_model_arguments(cfg))
    horizon = nominal.shape[0]
    nominal_states = rollout_ackermann(x0, nominal, cfg)
    projected = np.zeros((horizon, 2, 2), dtype=np.float64)
    damping = float(cfg.spg_pseudoinverse_damping)
    for t in range(horizon):
        interval = min(max(1, int(cfg.spg_lookahead_steps)), horizon - t)
        jacobian = np.zeros((2, 2), dtype=np.float64)
        for control_index, delta in enumerate((float(cfg.spg_fd_accel), float(cfg.spg_fd_steering_rate))):
            lower = cfg.accel_min if control_index == 0 else cfg.steering_rate_min
            upper = cfg.accel_max if control_index == 0 else cfg.steering_rate_max
            plus_value = float(np.clip(nominal[t, control_index] + delta, lower, upper))
            minus_value = float(np.clip(nominal[t, control_index] - delta, lower, upper))
            denominator = plus_value - minus_value
            if denominator <= 1e-12:
                continue
            plus_controls = nominal[t:t + interval].copy()
            minus_controls = plus_controls.copy()
            plus_controls[0, control_index] = plus_value
            minus_controls[0, control_index] = minus_value
            plus_position = rollout_ackermann(nominal_states[t], plus_controls, cfg)[-1, :2]
            minus_position = rollout_ackermann(nominal_states[t], minus_controls, cfg)[-1, :2]
            jacobian[:, control_index] = (plus_position - minus_position) / denominator
        regularized = jacobian @ jacobian.T + (damping * damping + 1e-18) * np.eye(2)
        pseudoinverse = jacobian.T @ np.linalg.inv(regularized)
        position_covariance = 0.5 * (covariances[t] + covariances[t].T)
        control_covariance = pseudoinverse @ position_covariance @ pseudoinverse.T
        projected[t] = 0.5 * (control_covariance + control_covariance.T) + float(cfg.spg_covariance_jitter) * np.eye(2)
    return projected

def sample_sensitivity_projected_gaussian_controls(x_current: Array, local_mode: MPPIHomotopyMode, n: int, cfg: MPPIConfig, rng: np.random.Generator) -> Array:
    """Sample the SPG proposal without using the fixed baseline covariance."""
    horizon = int(cfg.horizon)
    if n <= 0:
        return np.zeros((0, horizon, 2), dtype=np.float64)
    mean_path = np.asarray(local_mode.mean_path, dtype=np.float64)
    nominal = nominal_controls_to_track_path(x_current, mean_path, cfg)
    projected_covariances = sensitivity_projected_control_covariances(x_current, nominal, np.asarray(local_mode.cov_blocks, dtype=np.float64), cfg)
    standard_noise = make_temporally_correlated_noise(n, horizon, cfg, rng, noise_accel=1.0, noise_steering_rate=1.0)
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
    return enforce_ackermann_control_bounds(controls, cfg)

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
            clearance = path_min_clearance_to_circles(local_mode.mean_path, obstacle_circles, cfg.vehicle_length, cfg.vehicle_width, substeps=0)
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
        X = rollout_ackermann_batch(x_current, U, cfg)
        costs = stable_representation_costs(X, U, obstacle_circles, goal, cfg)
        costs = reject_colliding_rollouts(costs, X, obstacle_circles, goal, cfg)
        all_costs[ids] = costs
        all_U[ids] = U
        if record_optimal_traj:
            local_best = int(np.argmin(costs))
            if float(costs[local_best]) < best_cost:
                best_cost = float(costs[local_best])
                best_traj = np.asarray(X[local_best], dtype=np.float64).copy()
    planned_sequence = mppi_weighted_control_sequence(all_costs, all_U, cfg)
    info = {'cost_min': float(np.min(all_costs)), 'cost_mean': float(np.mean(all_costs)), 'soft_value': float(softmin_score(all_costs, cfg)), 'rep_type': int(rep_type), 'rollout_budget_total': total_budget, 'rollouts_by_mode': counts, 'active_mode_count': int(len(active_mode_indices)), 'suppressed_mode_count': int(len(global_modes) - len(active_mode_indices)), 'nearby_mode_count': int(len(candidate_global_modes)), 'mode_clearances': mode_clearances.tolist(), 'optimal_traj': best_traj, 'planned_control_sequence': planned_sequence}
    return (planned_sequence[0].copy(), info, new_progress_by_mode)

def make_temporally_correlated_noise(n: int, H: int, cfg: MPPIConfig, rng: np.random.Generator, *, noise_accel: Optional[float]=None, noise_steering_rate: Optional[float]=None) -> Array:
    noise_scale = np.array([cfg.noise_accel if noise_accel is None else float(noise_accel), cfg.noise_steering_rate if noise_steering_rate is None else float(noise_steering_rate)], dtype=np.float64)
    noise = rng.normal(size=(n, H, 2)) * noise_scale[None, None, :]
    alpha = float(cfg.temporal_noise_smoothing)
    if temporal_smooth_noise_nb is not None:
        return temporal_smooth_noise_nb(noise, alpha)
    for t in range(1, H):
        noise[:, t, :] = alpha * noise[:, t - 1, :] + (1.0 - alpha) * noise[:, t, :]
    return noise

def rollout_collision_mask(X: Array, obstacle_circles: Sequence[Tuple[Array, float]], goal: Array, cfg: MPPIConfig) -> Array:
    state_batch = np.asarray(X, dtype=np.float64)
    if not obstacle_circles or state_batch.shape[0] == 0:
        return np.zeros(state_batch.shape[0], dtype=bool)
    centers = np.asarray([c for c, _ in obstacle_circles], dtype=np.float64)
    radii = np.asarray([r for _, r in obstacle_circles], dtype=np.float64)
    goal_array = np.asarray(goal, dtype=np.float64)
    if rollout_collision_mask_nb is not None:
        return rollout_collision_mask_nb(state_batch, centers, radii, goal_array, float(cfg.vehicle_length), float(cfg.vehicle_width), float(cfg.hard_collision_clearance), float(cfg.rollout_goal_tolerance))
    points = np.asarray(state_batch[:, 1:, :2], dtype=np.float64)
    collision_by_step = np.zeros(points.shape[:2], dtype=bool)
    for n in range(points.shape[0]):
        for t in range(points.shape[1]):
            clearance = minimum_rectangle_circle_clearance(state_batch[n, t + 1], centers, radii, cfg.vehicle_length, cfg.vehicle_width)
            collision_by_step[n, t] = clearance < float(cfg.hard_collision_clearance)
    goal_by_step = np.linalg.norm(points - np.asarray(goal, dtype=np.float64)[None, None, :], axis=2) <= float(cfg.rollout_goal_tolerance)
    colliding = np.zeros(state_batch.shape[0], dtype=bool)
    for n in range(X.shape[0]):
        reached = np.flatnonzero(goal_by_step[n])
        stop = int(reached[0]) + 1 if len(reached) else points.shape[1]
        colliding[n] = bool(np.any(collision_by_step[n, :stop]))
    return colliding

def reject_colliding_rollouts(costs: Array, X: Array, obstacle_circles: Sequence[Tuple[Array, float]], goal: Array, cfg: MPPIConfig) -> Array:
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

def mppi_weighted_control_sequence(costs: Array, U: Array, cfg: MPPIConfig) -> Array:
    weights = mppi_weights(costs, cfg)
    sequence = np.tensordot(weights, U, axes=(0, 0))
    sequence[:, 0] = np.clip(sequence[:, 0], cfg.accel_min, cfg.accel_max)
    sequence[:, 1] = np.clip(sequence[:, 1], cfg.steering_rate_min, cfg.steering_rate_max)
    return np.asarray(sequence, dtype=np.float64)

def update_display_trajectory(info: Dict[str, object], x_current: Array, executed_u: Array, goal: Array, cfg: MPPIConfig) -> None:
    sequence = info.get('planned_control_sequence')
    if sequence is None:
        return
    display_u = np.asarray(sequence, dtype=np.float64).copy()
    if display_u.ndim != 2 or display_u.shape[1] != 2 or len(display_u) == 0:
        return
    display_u[0] = np.asarray(executed_u, dtype=np.float64)
    trajectory = rollout_ackermann(x_current, display_u, cfg)
    distances = np.linalg.norm(trajectory[:, :2] - np.asarray(goal, dtype=np.float64)[None, :], axis=1)
    reached = np.flatnonzero(distances <= float(cfg.rollout_goal_tolerance))
    if len(reached):
        trajectory = trajectory[:int(reached[0]) + 1]
    info['optimal_traj'] = trajectory

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
    U = enforce_ackermann_control_bounds(U, cfg)
    X = rollout_ackermann_batch(x_current, U, cfg)
    costs = standard_mppi_costs_batch(X, U, obstacle_circles, goal, cfg)
    costs = reject_colliding_rollouts(costs, X, obstacle_circles, goal, cfg)
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
    return np.array([start[0], start[1], heading, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

def ensure_direct_goal_prior(U: Array, x_current: Array, goal: Array, cfg: MPPIConfig) -> Array:
    proposals = np.asarray(U, dtype=np.float64)
    if proposals.ndim != 3 or proposals.shape[0] == 0:
        return proposals
    proposals[-1] = nominal_controls_to_goal(x_current, goal, cfg)
    return enforce_ackermann_control_bounds(proposals, cfg)

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
    reached_goal = goal_pose_satisfied(x, scene.goal, cfg.rollout_goal_tolerance, cfg)
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
        elif variant in (ControllerVariant.STANDARD_MPPI, ControllerVariant.STANDARD_MPPI_128):
            step_cfg = replace(cfg, num_rollouts=128) if variant == ControllerVariant.STANDARD_MPPI_128 else cfg
            u, info = standard_mppi_step(x, active_obstacles, scene.goal, step_cfg, rng, obstacle_circles=active_circles, record_optimal_traj=record)
        else:
            raise ValueError(f'Unsupported variant: {variant}')
        u = apply_smooth_safe_control(x, u, previous_control, active_circles, cfg)
        if record:
            update_display_trajectory(info, x, u, scene.goal, cfg)
            infos.append(info)
        previous_control = u.copy()
        x = ackermann_step(x, u, cfg)
        states.append(x.copy())
        controls.append(u.copy())
        if goal_pose_satisfied(x, scene.goal, cfg.rollout_goal_tolerance, cfg):
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
    trigger_progress: float = 0.35
    wall_width: float = 0.4
    wall_extension: float = 0.2

def default_dynamic_wall_scenarios() -> List[DynamicWallScenario]:
    return [DynamicWallScenario('wall_0_1', ((0, 1),), trigger_progress=0.25), DynamicWallScenario('wall_1_2', ((1, 2),), trigger_progress=0.25), DynamicWallScenario('walls_0_1__1_2', ((0, 1), (1, 2)), trigger_progress=0.25)]

def build_homotopy_modes(scene: Scene, obstacles: Sequence, swarm_seed: int) -> list[MPPIHomotopyMode]:
    generated = run_swarm_planner(start=scene.start, goal=scene.goal, obstacles=obstacles, scale=scene.scale, bounds_xy=scene.bounds_xy, seed=swarm_seed)
    mixture = fit_topological_trajectory_mixture(generated, K=50, beta=1.0, min_mode_samples=3, covariance_jitter=0.0002, bounds=scene.planner_bounds, goal=scene.goal, snap_to_goal_radius=0.2, snap_straight_tail_points=8)
    return mixture_to_mppi_modes(mixture)

def minimum_clearance(states: Array, obstacles: Sequence, cfg: MPPIConfig) -> float:
    return min_clearance(states, obstacles, cfg.vehicle_length, cfg.vehicle_width)