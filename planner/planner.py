from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import copy

import numpy as np

try:
    from numba import njit
except Exception:
    njit = None


Array = np.ndarray


# -----------------------------------------------------------------------------
# Optional numba kernels
# -----------------------------------------------------------------------------

if njit is not None:
    @njit(cache=True)
    def _path_length_nb(path):
        n = path.shape[0]
        if n < 2:
            return 0.0
        total = 0.0
        for i in range(n - 1):
            dx = path[i + 1, 0] - path[i, 0]
            dy = path[i + 1, 1] - path[i, 1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    @njit(cache=True)
    def _path_smoothness_nb(path):
        n = path.shape[0]
        if n < 3:
            return 0.0
        total = 0.0
        for i in range(n - 2):
            ax = path[i + 2, 0] - 2.0 * path[i + 1, 0] + path[i, 0]
            ay = path[i + 2, 1] - 2.0 * path[i + 1, 1] + path[i, 1]
            total += ax * ax + ay * ay
        return total

    @njit(cache=True)
    def _truncate_at_goal_index_nb(path, goal, goal_radius):
        n = path.shape[0]
        rr = goal_radius * goal_radius
        for i in range(n):
            dx = path[i, 0] - goal[0]
            dy = path[i, 1] - goal[1]
            if dx * dx + dy * dy <= rr:
                return i
        return -1

    @njit(cache=True)
    def _valid_and_duplicate_keep_mask_nb(path, min_step_sq):
        n = path.shape[0]
        keep = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            finite = math.isfinite(path[i, 0]) and math.isfinite(path[i, 1])
            if not finite:
                keep[i] = False
            elif i == 0:
                keep[i] = True
            else:
                dx = path[i, 0] - path[i - 1, 0]
                dy = path[i, 1] - path[i - 1, 1]
                keep[i] = (dx * dx + dy * dy) > min_step_sq
        return keep

    @njit(cache=True)
    def _angle_diff_nb(a, b):
        d = a - b
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return d

    @njit(cache=True)
    def _homotopy_signature_array_nb(path, centers):
        m = centers.shape[0]
        sig = np.zeros(m, dtype=np.int64)
        n = path.shape[0]
        for j in range(m):
            cx = centers[j, 0]
            cy = centers[j, 1]
            total = 0.0
            for i in range(n - 1):
                aa = math.atan2(path[i, 1] - cy, path[i, 0] - cx)
                bb = math.atan2(path[i + 1, 1] - cy, path[i + 1, 0] - cx)
                total += _angle_diff_nb(bb, aa)
            sig[j] = int(round(total / (2.0 * math.pi)))
        return sig

    @njit(cache=True)
    def _costmap_value_nb(costmap, xmin, xmax, ymin, ymax, px, py):
        if px < xmin or px > xmax or py < ymin or py > ymax:
            return 1e3

        h = costmap.shape[0]
        w = costmap.shape[1]
        denom_x = xmax - xmin
        denom_y = ymax - ymin
        if denom_x < 1e-12:
            denom_x = 1e-12
        if denom_y < 1e-12:
            denom_y = 1e-12

        u = (px - xmin) / denom_x * (w - 1)
        v = (py - ymin) / denom_y * (h - 1)

        x0 = int(math.floor(u))
        y0 = int(math.floor(v))
        x1 = x0 + 1
        y1 = y0 + 1
        if x1 >= w:
            x1 = w - 1
        if y1 >= h:
            y1 = h - 1

        fu = u - x0
        fv = v - y0

        return (
            (1.0 - fu) * (1.0 - fv) * costmap[y0, x0]
            + fu * (1.0 - fv) * costmap[y0, x1]
            + (1.0 - fu) * fv * costmap[y1, x0]
            + fu * fv * costmap[y1, x1]
        )

    @njit(cache=True)
    def _trajectory_cost_no_costmap_nb(path, w_len, w_smooth):
        return w_len * _path_length_nb(path) + w_smooth * _path_smoothness_nb(path)

    @njit(cache=True)
    def _trajectory_cost_with_costmap_nb(
        path,
        costmap,
        xmin,
        xmax,
        ymin,
        ymax,
        w_len,
        w_grid,
        w_smooth,
    ):
        n = path.shape[0]
        L = _path_length_nb(path)
        grid_int = 0.0

        if n >= 2:
            for i in range(n - 1):
                ax = path[i, 0]
                ay = path[i, 1]
                bx = path[i + 1, 0]
                by = path[i + 1, 1]

                mx = 0.5 * (ax + bx)
                my = 0.5 * (ay + by)
                dx = bx - ax
                dy = by - ay
                seg_len = math.sqrt(dx * dx + dy * dy)

                grid_int += _costmap_value_nb(costmap, xmin, xmax, ymin, ymax, mx, my) * seg_len

        return w_len * L + w_grid * grid_int + w_smooth * _path_smoothness_nb(path)

else:
    _path_length_nb = None
    _path_smoothness_nb = None
    _truncate_at_goal_index_nb = None
    _valid_and_duplicate_keep_mask_nb = None
    _angle_diff_nb = None
    _homotopy_signature_array_nb = None
    _costmap_value_nb = None
    _trajectory_cost_no_costmap_nb = None
    _trajectory_cost_with_costmap_nb = None


# -----------------------------------------------------------------------------
# Basic geometry / trajectory utilities
# -----------------------------------------------------------------------------

def path_length(path: Array) -> float:
    path = np.asarray(path, dtype=np.float64)
    if _path_length_nb is not None:
        return float(_path_length_nb(path))
    if path.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def path_smoothness(path: Array) -> float:
    """Integrated squared second difference. Lower is smoother."""
    path = np.asarray(path, dtype=np.float64)
    if _path_smoothness_nb is not None:
        return float(_path_smoothness_nb(path))
    if path.shape[0] < 3:
        return 0.0
    acc = path[2:] - 2.0 * path[1:-1] + path[:-2]
    return float(np.sum(np.sum(acc * acc, axis=1)))


def truncate_at_goal(path: Array, goal: Array, goal_radius: float) -> Optional[Array]:
    path = np.asarray(path, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2:
        return None

    if _truncate_at_goal_index_nb is not None:
        idx = int(_truncate_at_goal_index_nb(path, goal, float(goal_radius)))
        if idx < 0:
            return None
        out = path[: idx + 1].copy()
        out[-1] = goal
        return out

    d = np.linalg.norm(path - goal[None, :], axis=1)
    hit = np.where(d <= goal_radius)[0]
    if hit.size == 0:
        return None
    out = path[: int(hit[0]) + 1].copy()
    out[-1] = goal
    return out


def fish_trajectories_from_info(
    info: Dict[str, Any],
    goal: Array,
    *,
    scale: float = 1.0,
    goal_radius: float = 1.25,
    require_success: bool = True,
    min_points: int = 3,
) -> List[Array]:
    """
    Convert FishGoalEnv2D info['trajectory_boid_pos'] into a list of paths.

    The environment stores trajectories as T x N x 2. The returned paths are in
    unscaled workspace coordinates.
    """
    if info.get("trajectory_boid_pos", None) is None:
        raise ValueError("Environment was not created with returnTrajectory=True.")

    trajs = np.asarray(info["trajectory_boid_pos"], dtype=np.float64) / float(scale)
    goal = np.asarray(goal, dtype=np.float64)
    if trajs.ndim != 3 or trajs.shape[-1] != 2:
        raise ValueError(f"Expected T x N x 2 trajectory array, got {trajs.shape}.")

    T, N, _ = trajs.shape
    paths: List[Array] = []
    for j in range(N):
        p = trajs[:, j, :]
        p = p[np.isfinite(p).all(axis=1)]
        if p.shape[0] < min_points:
            continue

        # Remove consecutive duplicates / stationary tails.
        if _valid_and_duplicate_keep_mask_nb is not None:
            keep = _valid_and_duplicate_keep_mask_nb(p, 1e-18)
        else:
            keep = np.ones(p.shape[0], dtype=bool)
            if p.shape[0] > 1:
                keep[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9
        p = p[keep]
        if p.shape[0] < min_points:
            continue

        clipped = truncate_at_goal(p, goal, goal_radius)
        if clipped is not None:
            paths.append(clipped)
        elif not require_success:
            paths.append(p.copy())

    return paths


def angle_diff(a: float, b: float) -> float:
    if _angle_diff_nb is not None:
        return float(_angle_diff_nb(float(a), float(b)))
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def obstacle_centers(obstacles: Sequence[Any]) -> Array:
    centers = []
    for obs in obstacles:
        if hasattr(obs, "vertices"):
            xy = np.asarray(obs.vertices, dtype=np.float64)
        else:
            xy = np.asarray(obs, dtype=np.float64)
        centers.append(np.mean(xy[:, :2], axis=0))
    return np.asarray(centers, dtype=np.float64)


def homotopy_signature(path: Array, centers: Array) -> Tuple[int, ...]:
    """2D winding-number signature around obstacle representatives."""
    path = np.asarray(path, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    if _homotopy_signature_array_nb is not None:
        return tuple(int(v) for v in _homotopy_signature_array_nb(path, centers))

    sig: List[int] = []
    for c in centers:
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            aa = math.atan2(a[1] - c[1], a[0] - c[0])
            bb = math.atan2(b[1] - c[1], b[0] - c[0])
            total += angle_diff(bb, aa)
        sig.append(int(round(total / (2.0 * math.pi))))
    return tuple(sig)


def group_by_homotopy(paths: Sequence[Array], obstacles: Sequence[Any]) -> Dict[Tuple[int, ...], List[Array]]:
    centers = obstacle_centers(obstacles)
    groups: Dict[Tuple[int, ...], List[Array]] = {}
    for p in paths:
        groups.setdefault(homotopy_signature(p, centers), []).append(np.asarray(p, dtype=np.float64))
    return groups


def representative_per_homotopy(paths: Sequence[Array], obstacles: Sequence[Any], keep: str = "shortest") -> List[Array]:
    groups = group_by_homotopy(paths, obstacles)
    reps: List[Array] = []
    for _, ps in groups.items():
        if keep == "first":
            reps.append(ps[0])
        elif keep == "shortest":
            reps.append(min(ps, key=path_length))
        else:
            raise ValueError("keep must be 'first' or 'shortest'.")
    return sorted(reps, key=path_length)


def costmap_value(costmap: Optional[Array], bounds: Tuple[Tuple[float, float], Tuple[float, float]], p: Array) -> float:
    if costmap is None:
        return 0.0
    cm = np.asarray(costmap, dtype=np.float64)
    (xmin, xmax), (ymin, ymax) = bounds
    x, y = float(p[0]), float(p[1])

    if _costmap_value_nb is not None:
        return float(_costmap_value_nb(cm, float(xmin), float(xmax), float(ymin), float(ymax), x, y))

    if x < xmin or x > xmax or y < ymin or y > ymax:
        return 1e3
    u = (x - xmin) / max(xmax - xmin, 1e-12) * (cm.shape[1] - 1)
    v = (y - ymin) / max(ymax - ymin, 1e-12) * (cm.shape[0] - 1)
    x0 = int(np.floor(u)); y0 = int(np.floor(v))
    x1 = min(x0 + 1, cm.shape[1] - 1); y1 = min(y0 + 1, cm.shape[0] - 1)
    fu = u - x0; fv = v - y0
    return float((1 - fu) * (1 - fv) * cm[y0, x0] + fu * (1 - fv) * cm[y0, x1] + (1 - fu) * fv * cm[y1, x0] + fu * fv * cm[y1, x1])


def trajectory_cost(
    path: Array,
    *,
    costmap: Optional[Array] = None,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 10.0), (0.0, 10.0)),
    w_len: float = 1.0,
    w_grid: float = 1.0,
    w_smooth: float = 0.0,
) -> float:
    path = np.asarray(path, dtype=np.float64)

    if costmap is None and _trajectory_cost_no_costmap_nb is not None:
        return float(_trajectory_cost_no_costmap_nb(path, float(w_len), float(w_smooth)))

    if costmap is not None and _trajectory_cost_with_costmap_nb is not None:
        cm = np.asarray(costmap, dtype=np.float64)
        (xmin, xmax), (ymin, ymax) = bounds
        return float(_trajectory_cost_with_costmap_nb(
            path,
            cm,
            float(xmin),
            float(xmax),
            float(ymin),
            float(ymax),
            float(w_len),
            float(w_grid),
            float(w_smooth),
        ))

    L = path_length(path)
    grid_int = 0.0
    if costmap is not None and path.shape[0] >= 2:
        for a, b in zip(path[:-1], path[1:]):
            m = 0.5 * (a + b)
            grid_int += costmap_value(costmap, bounds, m) * np.linalg.norm(b - a)
    return float(w_len * L + w_grid * grid_int + w_smooth * path_smoothness(path))


def dynamic_obstacle_penalty(
    path: Array,
    dynamic_obstacles: Optional[Sequence[Callable[[float], Tuple[Array, float]]]],
    *,
    dt: float,
    safety_margin: float = 0.0,
    collision_penalty: float = 1e6,
) -> float:
    """
    dynamic_obstacles: list of functions f(t)->(center_xy, radius).
    Returns large penalty if the time-indexed path intersects one.
    """
    if not dynamic_obstacles:
        return 0.0
    p = np.asarray(path, dtype=np.float64)
    penalty = 0.0
    for k, x in enumerate(p):
        t = k * float(dt)
        for obs_fn in dynamic_obstacles:
            c, r = obs_fn(t)
            c = np.asarray(c, dtype=np.float64)
            clearance = np.linalg.norm(x - c) - float(r) - float(safety_margin)
            if clearance <= 0.0:
                penalty += collision_penalty + 100.0 * (-clearance)
            else:
                penalty += 1.0 / (clearance + 1e-6)
    return float(penalty)


def rollout_fish(
    env_cls: Any,
    action: Array,
    *,
    start: Array,
    goals: Array,
    goal_W: Optional[Array] = None,
    segs: Optional[Array] = None,
    scale: float = 1.0,
    seed: int = 0,
    boid_count: int = 600,
    bound: float = 40.0,
    max_steps: int = 600,
    dt: float = 1.0,
    goal_radius: float = 5.0,
    start_spread: float = 0.0,
    robins_number: int = 1,
    starts: Optional[Array] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Small adapter around your FishGoalEnv2D/FishGoalEnv2DDiverse interface."""
    env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
    kwargs = dict(
        boid_count=int(boid_count),
        bound=float(bound),
        max_steps=int(max_steps),
        dt=float(dt),
        goals=np.asarray(goals, dtype=np.float32),
        segs=segs,
        doAnimation=False,
        returnTrajectory=True,
        start_spread=float(start_spread),
        goal_radius=float(goal_radius),
        robins_number=int(robins_number),
    )
    if starts is not None:
        kwargs["starts"] = np.asarray(starts, dtype=np.float32)
    else:
        kwargs["start"] = np.asarray(start, dtype=np.float32)
    if goal_W is not None:
        kwargs["goal_W"] = np.asarray(goal_W, dtype=np.float32)
    kwargs.update(env_kwargs)

    env = env_cls(**kwargs)
    env.reset(seed=int(seed))
    _, _, _, _, info = env.step(np.asarray(action, dtype=np.float32))
    return env, info

def numba_enabled() -> bool:
    """Return True when optional numba acceleration is available."""
    return njit is not None


@dataclass
class GenerativePlanningResult:
    samples: List[Array]
    representatives: List[Array]
    homotopy_groups: Dict[Tuple[int, ...], List[Array]]
    probabilities: Dict[Tuple[int, ...], float]
    info: Dict[str, Any]


@dataclass
class HomotopyAwareGenerativePlanner:
    """
    Nonparametric generative planner: tau ~ p(tau | environment, start, goal).

    It samples a trajectory distribution online from stochastic fish rollouts,
    without training data. The generated samples are grouped into homotopy modes.
    """

    env_cls: Any
    action: Array
    obstacles: Sequence[Any]
    segs: Optional[Array]
    scale: float = 4.0
    bound_scaled: float = 40.0
    boid_count: int = 1200
    max_steps: int = 600
    dt: float = 0.5
    goal_radius_scaled: float = 5.0

    def sample(
        self,
        start_unscaled: Array,
        goal_unscaled: Array,
        *,
        graph_goals: Optional[Array] = None,
        graph_W: Optional[Array] = None,
        seed: int = 0,
        require_success: bool = True,
    ) -> GenerativePlanningResult:
        if graph_goals is None:
            goals_scaled = np.asarray(goal_unscaled, dtype=np.float64).reshape(1, 2) * self.scale
        else:
            goals_scaled = np.asarray(graph_goals, dtype=np.float64)

        _, info = rollout_fish(
            self.env_cls,
            self.action,
            start=np.asarray(start_unscaled, dtype=np.float64) * self.scale,
            goals=goals_scaled,
            goal_W=graph_W,
            segs=self.segs,
            scale=self.scale,
            seed=seed,
            boid_count=self.boid_count,
            bound=self.bound_scaled,
            max_steps=self.max_steps,
            dt=self.dt,
            goal_radius=self.goal_radius_scaled,
            start_spread=0.0,
        )
        samples = fish_trajectories_from_info(
            info,
            np.asarray(goal_unscaled, dtype=np.float64),
            scale=self.scale,
            goal_radius=self.goal_radius_scaled / self.scale,
            require_success=require_success,
        )
        groups = group_by_homotopy(samples, self.obstacles)
        total = max(len(samples), 1)
        probs = {sig: len(ps) / total for sig, ps in groups.items()}
        reps = representative_per_homotopy(samples, self.obstacles, keep="shortest")
        return GenerativePlanningResult(samples, reps, groups, probs, info)

    def resample_modes(
        self,
        result: GenerativePlanningResult,
        n: int,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> List[Array]:
        """Draw paths from the empirical homotopy-mode mixture."""
        rng = np.random.default_rng() if rng is None else rng
        sigs = list(result.homotopy_groups.keys())
        if not sigs:
            return []
        p = np.array([result.probabilities[s] for s in sigs], dtype=np.float64)
        if temperature != 1.0:
            p = np.power(p + 1e-12, 1.0 / max(temperature, 1e-6))
        p /= np.sum(p)
        out: List[Array] = []
        for _ in range(int(n)):
            sig = sigs[int(rng.choice(len(sigs), p=p))]
            paths = result.homotopy_groups[sig]
            out.append(paths[int(rng.integers(0, len(paths)))].copy())
        return out