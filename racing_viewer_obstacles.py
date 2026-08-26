from __future__ import annotations
import itertools
import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass, replace
from typing import Optional
import numpy as np
from numba import njit, prange
from system import ackermann, four_wheel, controller as controller_core
from system.ackermann import _dynamic_ackermann_step_nb
from system.four_wheel import _dynamic_four_wheel_step_nb
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:
    raise SystemExit('Tkinter is required to run the racing viewer.') from exc
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Polygon
VARIANTS = [('Planner iLQR', 'planner_ilqr'), ('Standard MPPI', 'standard_mppi'), ('Corridor prior', 'corridor_prior_mppi'), ('SPG prior', 'sensitivity_projected_gaussian_prior_mppi')]
DISPLAY_TO_VARIANT = dict(VARIANTS)
VARIANT_TO_DISPLAY = {value: label for label, value in VARIANTS}
VEHICLE_SYSTEMS = list(controller_core.VEHICLE_SYSTEMS)
DISPLAY_TO_VEHICLE = dict(VEHICLE_SYSTEMS)
VEHICLE_TO_DISPLAY = {value: label for label, value in VEHICLE_SYSTEMS}
PRIOR_MEAN_VARIANTS = {'corridor_prior_mppi', 'gaussian_prior_mppi', 'sensitivity_projected_gaussian_prior_mppi'}
PRIOR_COVARIANCE_VARIANTS = {'gaussian_prior_mppi', 'sensitivity_projected_gaussian_prior_mppi'}
PRIOR_COV_VIS_SPACING = 1.0
PRIOR_COV_VIS_POINTS = 12
PRIOR_COV_ALPHA_MIN = 0.010
PRIOR_COV_ALPHA_SPAN = 0.055
PRIOR_MEAN_ALPHA_MIN = 0.10
PRIOR_MEAN_ALPHA_SPAN = 0.72
PRIOR_MEAN_LINEWIDTH_MIN = 0.85
PRIOR_MEAN_LINEWIDTH_SPAN = 0.95
PRIOR_PURPLE = (0.48, 0.12, 0.68)
WALL_MODES = [('No wall', 'no_wall'), ('Dynamic 1', 'dynamic_1'), ('Dynamic 2', 'dynamic_2')]
DISPLAY_TO_WALL_MODE = dict(WALL_MODES)
WALL_MODE_TO_DISPLAY = {value: label for label, value in WALL_MODES}
TRACK_HEIGHT = 20.0
TRACK_X0 = 0.0
TRACK_Y0 = 0.0
OUTER_RADIUS = 0.5 * TRACK_HEIGHT
CENTERLINE_RADIUS = 5.0
TURN_PRIOR_SIGMA = 1.5
OBSTACLE_LINEAR_SCALE = 0.75
TURN_BARRIER_LANE_WIDTH = 3.0
TURN_BARRIER_THICKNESS = 0.25
TURN_BARRIER_POINTS = 320
TURN_BARRIER_COLLISION_SEGMENT_LENGTH = 0.15
INNER_STRAIGHT_BARRIER_EXTENSION = 0.0
DYNAMIC_WALL_WIDTH = 0.3
DYNAMIC_WALL_LIFETIME_LAPS = {'dynamic_1': 1.0, 'dynamic_2': 2.0}
MAX_STEPS_PER_LAP = 400
CONTROLLER_START = np.asarray([1.0, 1.0], dtype=np.float64)
CONTROLLER_GOAL = np.asarray([9.0, 9.0], dtype=np.float64)
CONTROLLER_START_GOAL_DISTANCE = float(np.linalg.norm(CONTROLLER_GOAL - CONTROLLER_START))
STRAIGHT_SIDE_EXTENSION = 6.0
STRAIGHT_EXTENSION_POINT_SPACING = 0.25
OBSTACLE_STRAIGHT_LEFT_X = TRACK_X0 + OUTER_RADIUS
OBSTACLE_STRAIGHT_RIGHT_X = OBSTACLE_STRAIGHT_LEFT_X + CONTROLLER_START_GOAL_DISTANCE
LEFT_ARC_X = OBSTACLE_STRAIGHT_LEFT_X - STRAIGHT_SIDE_EXTENSION
RIGHT_ARC_X = OBSTACLE_STRAIGHT_RIGHT_X + STRAIGHT_SIDE_EXTENSION
STRAIGHT_LENGTH = RIGHT_ARC_X - LEFT_ARC_X
TRACK_WIDTH = 2.0 * OUTER_RADIUS + STRAIGHT_LENGTH
TRACK_CENTER_X = 0.5 * (LEFT_ARC_X + RIGHT_ARC_X)
TRACK_CENTER_Y = TRACK_Y0 + 0.5 * TRACK_HEIGHT
BOTTOM_TRACK_Y = TRACK_CENTER_Y - CENTERLINE_RADIUS
TOP_TRACK_Y = TRACK_CENTER_Y + CENTERLINE_RADIUS
CHICANE_RADIUS = CENTERLINE_RADIUS / 3.0
CHICANE_HORIZONTAL_LENGTH = 10.0
CHICANE_INNER_X_OFFSET = 4.0
CHICANE_OUTER_X_OFFSET = CHICANE_INNER_X_OFFSET + CHICANE_HORIZONTAL_LENGTH
CHICANE_ENTRY_GUARD_LENGTH = 10.0
CHICANE_HAIRPIN_LENGTH = math.pi * CHICANE_RADIUS
CHICANE_ENTRY_LENGTH = CHICANE_OUTER_X_OFFSET
CHICANE_EXIT_LENGTH = CHICANE_OUTER_X_OFFSET
CHICANE_LENGTH = CHICANE_ENTRY_LENGTH + 2.0 * CHICANE_HORIZONTAL_LENGTH + CHICANE_EXIT_LENGTH + 3.0 * CHICANE_HAIRPIN_LENGTH
RIGHT_CHICANE_OUTER_X = RIGHT_ARC_X + CHICANE_OUTER_X_OFFSET
RIGHT_CHICANE_INNER_X = RIGHT_ARC_X + CHICANE_INNER_X_OFFSET
RIGHT_CHICANE_Y0 = BOTTOM_TRACK_Y
RIGHT_CHICANE_Y1 = RIGHT_CHICANE_Y0 + 2.0 * CHICANE_RADIUS
RIGHT_CHICANE_Y2 = RIGHT_CHICANE_Y1 + 2.0 * CHICANE_RADIUS
RIGHT_CHICANE_Y3 = RIGHT_CHICANE_Y2 + 2.0 * CHICANE_RADIUS
RIGHT_CHICANE_C1_X = RIGHT_CHICANE_OUTER_X
RIGHT_CHICANE_C1_Y = 0.5 * (RIGHT_CHICANE_Y0 + RIGHT_CHICANE_Y1)
RIGHT_CHICANE_C2_X = RIGHT_CHICANE_INNER_X
RIGHT_CHICANE_C2_Y = 0.5 * (RIGHT_CHICANE_Y1 + RIGHT_CHICANE_Y2)
RIGHT_CHICANE_C3_X = RIGHT_CHICANE_OUTER_X
RIGHT_CHICANE_C3_Y = 0.5 * (RIGHT_CHICANE_Y2 + RIGHT_CHICANE_Y3)
LEFT_CHICANE_OUTER_X = LEFT_ARC_X - CHICANE_OUTER_X_OFFSET
LEFT_CHICANE_INNER_X = LEFT_ARC_X - CHICANE_INNER_X_OFFSET
LEFT_CHICANE_Y0 = TOP_TRACK_Y
LEFT_CHICANE_Y1 = LEFT_CHICANE_Y0 - 2.0 * CHICANE_RADIUS
LEFT_CHICANE_Y2 = LEFT_CHICANE_Y1 - 2.0 * CHICANE_RADIUS
LEFT_CHICANE_Y3 = LEFT_CHICANE_Y2 - 2.0 * CHICANE_RADIUS
LEFT_CHICANE_C1_X = LEFT_CHICANE_OUTER_X
LEFT_CHICANE_C1_Y = 0.5 * (LEFT_CHICANE_Y0 + LEFT_CHICANE_Y1)
LEFT_CHICANE_C2_X = LEFT_CHICANE_INNER_X
LEFT_CHICANE_C2_Y = 0.5 * (LEFT_CHICANE_Y1 + LEFT_CHICANE_Y2)
LEFT_CHICANE_C3_X = LEFT_CHICANE_OUTER_X
LEFT_CHICANE_C3_Y = 0.5 * (LEFT_CHICANE_Y2 + LEFT_CHICANE_Y3)
CHICANE_OUTWARD_EXTENT = CHICANE_OUTER_X_OFFSET + CHICANE_RADIUS
TRACK_PLOT_X_MIN = LEFT_ARC_X - CHICANE_OUTWARD_EXTENT - 0.5 * TURN_BARRIER_LANE_WIDTH - 0.8
TRACK_PLOT_X_MAX = RIGHT_ARC_X + CHICANE_OUTWARD_EXTENT + 0.5 * TURN_BARRIER_LANE_WIDTH + 0.8
TRACK_LENGTH = 2.0 * STRAIGHT_LENGTH + 2.0 * CHICANE_LENGTH
HALF_TRACK_LENGTH = STRAIGHT_LENGTH + CHICANE_LENGTH
START_S = 0.0

@dataclass
class RaceResult:
    states: np.ndarray
    controls: np.ndarray
    nominal_predictions: list[np.ndarray]
    mppi_predictions: list[np.ndarray]
    temperatures: np.ndarray
    esses: np.ndarray
    feasible_counts: np.ndarray
    cumulative_progress: np.ndarray
    lap_times: list[float]
    requested_laps: int
    completed_laps: int
    collision: bool
    max_steps_per_lap: int
    runtime_s: float
    variant_value: str
    wall_mode: str
    cfg: object
    model_name: str
    obstacle_vertices: list[np.ndarray]
    dynamic_wall_history: list[list[np.ndarray]]
    prior_bank: PackedRacingPriorBank
    active_prior_indices_history: list[np.ndarray]

@dataclass
class PackedRacingPriorBank:
    modes: list[controller_core.MPPIHomotopyMode]
    mean_paths: np.ndarray
    cov_blocks: np.ndarray
    arc_lengths: np.ndarray
    lengths: np.ndarray
    localization_lengths: np.ndarray
    localization_unique_lengths: np.ndarray
    localization_block_mins: np.ndarray
    localization_block_maxs: np.ndarray
    localization_block_counts: np.ndarray
    probabilities: np.ndarray
    sample_paths: np.ndarray
    sample_arc_lengths: np.ndarray
    sample_lengths: np.ndarray
    sample_mode_offsets: np.ndarray


@dataclass
class CollisionSectorBank:
    mask_centers: tuple[np.ndarray, ...]
    mask_radii: tuple[np.ndarray, ...]
    sector_mins: np.ndarray
    sector_maxs: np.ndarray

@dataclass
class DynamicWallCandidate:
    pair: tuple[int, int]
    wall: object
    vertices: np.ndarray
    length: float
    p0: np.ndarray
    p1: np.ndarray
    circle_centers: np.ndarray
    circle_radii: np.ndarray

@dataclass
class ActiveDynamicWall:
    candidate: DynamicWallCandidate
    activation_progress: float
    expiration_progress: float

@njit(cache=True, inline='always')
def _clamp_nb(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)

@njit(cache=True, inline='always')
def _segment_orientation_nb(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

@njit(cache=True, inline='always')
def _point_on_segment_nb(px: float, py: float, ax: float, ay: float, bx: float, by: float, eps: float) -> bool:
    if abs(_segment_orientation_nb(ax, ay, bx, by, px, py)) > eps:
        return False
    return min(ax, bx) - eps <= px <= max(ax, bx) + eps and min(ay, by) - eps <= py <= max(ay, by) + eps

@njit(cache=True, inline='always')
def _segments_intersect_nb(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> bool:
    eps = 1e-12
    o1 = _segment_orientation_nb(ax, ay, bx, by, cx, cy)
    o2 = _segment_orientation_nb(ax, ay, bx, by, dx, dy)
    o3 = _segment_orientation_nb(cx, cy, dx, dy, ax, ay)
    o4 = _segment_orientation_nb(cx, cy, dx, dy, bx, by)
    if (o1 > eps and o2 < -eps or (o1 < -eps and o2 > eps)) and (o3 > eps and o4 < -eps or (o3 < -eps and o4 > eps)):
        return True
    if abs(o1) <= eps and _point_on_segment_nb(cx, cy, ax, ay, bx, by, eps):
        return True
    if abs(o2) <= eps and _point_on_segment_nb(dx, dy, ax, ay, bx, by, eps):
        return True
    if abs(o3) <= eps and _point_on_segment_nb(ax, ay, cx, cy, dx, dy, eps):
        return True
    if abs(o4) <= eps and _point_on_segment_nb(bx, by, cx, cy, dx, dy, eps):
        return True
    return False

@njit(cache=True, parallel=True)
def _candidate_path_intersection_mask_nb(path_xy: np.ndarray, p0s: np.ndarray, p1s: np.ndarray) -> np.ndarray:
    """Mark center-to-center walls whose center segment crosses the executed path."""
    candidate_count = p0s.shape[0]
    out = np.zeros(candidate_count, dtype=np.bool_)
    if path_xy.shape[0] < 2:
        return out
    for candidate_index in prange(candidate_count):
        ax = p0s[candidate_index, 0]
        ay = p0s[candidate_index, 1]
        bx = p1s[candidate_index, 0]
        by = p1s[candidate_index, 1]
        hit = False
        for path_index in range(path_xy.shape[0] - 1):
            if _segments_intersect_nb(ax, ay, bx, by, path_xy[path_index, 0], path_xy[path_index, 1], path_xy[path_index + 1, 0], path_xy[path_index + 1, 1]):
                hit = True
                break
        out[candidate_index] = hit
    return out

@njit(cache=True, inline='always')
def _project_arc_candidate_nb(px: float, py: float, cx: float, cy: float, radius: float, theta0: float, sweep: float, s_offset: float, best_s: float, best_d2: float) -> tuple[float, float]:
    theta = math.atan2(py - cy, px - cx)
    span = abs(sweep)
    directed = theta - theta0 if sweep > 0.0 else theta0 - theta
    two_pi = 2.0 * math.pi
    while directed < 0.0:
        directed += two_pi
    while directed >= two_pi:
        directed -= two_pi
    if directed <= span:
        theta_q = theta0 + directed if sweep > 0.0 else theta0 - directed
        qx = cx + radius * math.cos(theta_q)
        qy = cy + radius * math.sin(theta_q)
        dx = px - qx
        dy = py - qy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_s = s_offset + radius * directed
    else:
        qx0 = cx + radius * math.cos(theta0)
        qy0 = cy + radius * math.sin(theta0)
        dx0 = px - qx0
        dy0 = py - qy0
        d20 = dx0 * dx0 + dy0 * dy0
        if d20 < best_d2:
            best_d2 = d20
            best_s = s_offset
        theta1 = theta0 + sweep
        qx1 = cx + radius * math.cos(theta1)
        qy1 = cy + radius * math.sin(theta1)
        dx1 = px - qx1
        dy1 = py - qy1
        d21 = dx1 * dx1 + dy1 * dy1
        if d21 < best_d2:
            best_d2 = d21
            best_s = s_offset + radius * span
    return best_s, best_d2

@njit(cache=True, inline='always')
def _project_segment_candidate_nb(px: float, py: float, ax: float, ay: float, bx: float, by: float, s_offset: float, best_s: float, best_d2: float) -> tuple[float, float]:
    vx = bx - ax
    vy = by - ay
    length2 = vx * vx + vy * vy
    if length2 <= 1e-18:
        t = 0.0
        length = 0.0
    else:
        t = ((px - ax) * vx + (py - ay) * vy) / length2
        t = min(max(t, 0.0), 1.0)
        length = math.sqrt(length2)
    qx = ax + t * vx
    qy = ay + t * vy
    dx = px - qx
    dy = py - qy
    d2 = dx * dx + dy * dy
    if d2 < best_d2:
        best_d2 = d2
        best_s = s_offset + t * length
    return best_s, best_d2

@njit(cache=True, inline='always')
def _project_centerline_nb(px: float, py: float) -> tuple[float, float]:
    qx = _clamp_nb(px, LEFT_ARC_X, RIGHT_ARC_X)
    dx = px - qx
    dy = py - BOTTOM_TRACK_Y
    best_d2 = dx * dx + dy * dy
    best_s = qx - LEFT_ARC_X

    right_offset = STRAIGHT_LENGTH
    best_s, best_d2 = _project_segment_candidate_nb(
        px, py,
        RIGHT_ARC_X, RIGHT_CHICANE_Y0,
        RIGHT_CHICANE_OUTER_X, RIGHT_CHICANE_Y0,
        right_offset, best_s, best_d2,
    )
    right_offset += CHICANE_ENTRY_LENGTH
    best_s, best_d2 = _project_arc_candidate_nb(
        px, py, RIGHT_CHICANE_C1_X, RIGHT_CHICANE_C1_Y, CHICANE_RADIUS,
        -0.5 * math.pi, math.pi, right_offset, best_s, best_d2,
    )
    right_offset += CHICANE_HAIRPIN_LENGTH
    best_s, best_d2 = _project_segment_candidate_nb(
        px, py,
        RIGHT_CHICANE_OUTER_X, RIGHT_CHICANE_Y1,
        RIGHT_CHICANE_INNER_X, RIGHT_CHICANE_Y1,
        right_offset, best_s, best_d2,
    )
    right_offset += CHICANE_HORIZONTAL_LENGTH
    best_s, best_d2 = _project_arc_candidate_nb(
        px, py, RIGHT_CHICANE_C2_X, RIGHT_CHICANE_C2_Y, CHICANE_RADIUS,
        -0.5 * math.pi, -math.pi, right_offset, best_s, best_d2,
    )
    right_offset += CHICANE_HAIRPIN_LENGTH
    best_s, best_d2 = _project_segment_candidate_nb(
        px, py,
        RIGHT_CHICANE_INNER_X, RIGHT_CHICANE_Y2,
        RIGHT_CHICANE_OUTER_X, RIGHT_CHICANE_Y2,
        right_offset, best_s, best_d2,
    )
    right_offset += CHICANE_HORIZONTAL_LENGTH
    best_s, best_d2 = _project_arc_candidate_nb(
        px, py, RIGHT_CHICANE_C3_X, RIGHT_CHICANE_C3_Y, CHICANE_RADIUS,
        -0.5 * math.pi, math.pi, right_offset, best_s, best_d2,
    )
    right_offset += CHICANE_HAIRPIN_LENGTH
    best_s, best_d2 = _project_segment_candidate_nb(
        px, py,
        RIGHT_CHICANE_OUTER_X, RIGHT_CHICANE_Y3,
        RIGHT_ARC_X, TOP_TRACK_Y,
        right_offset, best_s, best_d2,
    )

    qx = _clamp_nb(px, LEFT_ARC_X, RIGHT_ARC_X)
    dx = px - qx
    dy = py - TOP_TRACK_Y
    d2 = dx * dx + dy * dy
    if d2 < best_d2:
        best_d2 = d2
        best_s = STRAIGHT_LENGTH + CHICANE_LENGTH + (RIGHT_ARC_X - qx)

    left_offset = 2.0 * STRAIGHT_LENGTH + CHICANE_LENGTH
    best_s, best_d2 = _project_segment_candidate_nb(
        px, py,
        LEFT_ARC_X, LEFT_CHICANE_Y0,
        LEFT_CHICANE_OUTER_X, LEFT_CHICANE_Y0,
        left_offset, best_s, best_d2,
    )
    left_offset += CHICANE_ENTRY_LENGTH
    best_s, best_d2 = _project_arc_candidate_nb(
        px, py, LEFT_CHICANE_C1_X, LEFT_CHICANE_C1_Y, CHICANE_RADIUS,
        0.5 * math.pi, math.pi, left_offset, best_s, best_d2,
    )
    left_offset += CHICANE_HAIRPIN_LENGTH
    best_s, best_d2 = _project_segment_candidate_nb(
        px, py,
        LEFT_CHICANE_OUTER_X, LEFT_CHICANE_Y1,
        LEFT_CHICANE_INNER_X, LEFT_CHICANE_Y1,
        left_offset, best_s, best_d2,
    )
    left_offset += CHICANE_HORIZONTAL_LENGTH
    best_s, best_d2 = _project_arc_candidate_nb(
        px, py, LEFT_CHICANE_C2_X, LEFT_CHICANE_C2_Y, CHICANE_RADIUS,
        0.5 * math.pi, -math.pi, left_offset, best_s, best_d2,
    )
    left_offset += CHICANE_HAIRPIN_LENGTH
    best_s, best_d2 = _project_segment_candidate_nb(
        px, py,
        LEFT_CHICANE_INNER_X, LEFT_CHICANE_Y2,
        LEFT_CHICANE_OUTER_X, LEFT_CHICANE_Y2,
        left_offset, best_s, best_d2,
    )
    left_offset += CHICANE_HORIZONTAL_LENGTH
    best_s, best_d2 = _project_arc_candidate_nb(
        px, py, LEFT_CHICANE_C3_X, LEFT_CHICANE_C3_Y, CHICANE_RADIUS,
        0.5 * math.pi, math.pi, left_offset, best_s, best_d2,
    )
    left_offset += CHICANE_HAIRPIN_LENGTH
    best_s, best_d2 = _project_segment_candidate_nb(
        px, py,
        LEFT_CHICANE_OUTER_X, LEFT_CHICANE_Y3,
        LEFT_ARC_X, BOTTOM_TRACK_Y,
        left_offset, best_s, best_d2,
    )
    if best_s >= TRACK_LENGTH:
        best_s -= TRACK_LENGTH
    return best_s, best_d2

@njit(cache=True, inline='always')
def _signed_progress_delta_nb(new_s: float, old_s: float) -> float:
    ds = new_s - old_s
    half = 0.5 * TRACK_LENGTH
    if ds > half:
        ds -= TRACK_LENGTH
    elif ds < -half:
        ds += TRACK_LENGTH
    return ds

@njit(cache=True, inline='always')
def _accept_near_s_projection_nb(
    candidate_s: float,
    candidate_d2: float,
    reference_s: float,
    backward_limit: float,
    forward_limit: float,
    best_s: float,
    best_d2: float,
    best_abs_ds: float,
) -> tuple[float, float, float]:
    candidate_s %= TRACK_LENGTH
    ds = _signed_progress_delta_nb(candidate_s, reference_s)
    if ds < -backward_limit or ds > forward_limit:
        return best_s, best_d2, best_abs_ds
    abs_ds = abs(ds)
    if candidate_d2 < best_d2 - 1e-12 or (
        abs(candidate_d2 - best_d2) <= 1e-12 and abs_ds < best_abs_ds
    ):
        return candidate_s, candidate_d2, abs_ds
    return best_s, best_d2, best_abs_ds

@njit(cache=True, inline='always')
def _project_centerline_near_s_nb(
    px: float,
    py: float,
    reference_s: float,
    backward_limit: float,
    forward_limit: float,
) -> tuple[float, float]:
    """Project to the geometrically closest track branch near the known progress.

    The chicanes fold several distant arc-length locations into a small XY area.
    A global closest-point projection can therefore jump to another hairpin branch.
    This projection evaluates the same exact track primitives, but rejects candidates
    that are not reachable from ``reference_s`` during one simulation step.
    """
    reference_s %= TRACK_LENGTH
    backward_limit = max(0.0, float(backward_limit))
    forward_limit = max(0.0, float(forward_limit))
    best_s = reference_s
    best_d2 = 1e300
    best_abs_ds = 1e300

    qx = _clamp_nb(px, LEFT_ARC_X, RIGHT_ARC_X)
    dx = px - qx
    dy = py - BOTTOM_TRACK_Y
    candidate_s = qx - LEFT_ARC_X
    candidate_d2 = dx * dx + dy * dy
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )

    right_offset = STRAIGHT_LENGTH
    candidate_s, candidate_d2 = _project_segment_candidate_nb(
        px, py, RIGHT_ARC_X, RIGHT_CHICANE_Y0, RIGHT_CHICANE_OUTER_X,
        RIGHT_CHICANE_Y0, right_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    right_offset += CHICANE_ENTRY_LENGTH
    candidate_s, candidate_d2 = _project_arc_candidate_nb(
        px, py, RIGHT_CHICANE_C1_X, RIGHT_CHICANE_C1_Y, CHICANE_RADIUS,
        -0.5 * math.pi, math.pi, right_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    right_offset += CHICANE_HAIRPIN_LENGTH
    candidate_s, candidate_d2 = _project_segment_candidate_nb(
        px, py, RIGHT_CHICANE_OUTER_X, RIGHT_CHICANE_Y1, RIGHT_CHICANE_INNER_X,
        RIGHT_CHICANE_Y1, right_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    right_offset += CHICANE_HORIZONTAL_LENGTH
    candidate_s, candidate_d2 = _project_arc_candidate_nb(
        px, py, RIGHT_CHICANE_C2_X, RIGHT_CHICANE_C2_Y, CHICANE_RADIUS,
        -0.5 * math.pi, -math.pi, right_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    right_offset += CHICANE_HAIRPIN_LENGTH
    candidate_s, candidate_d2 = _project_segment_candidate_nb(
        px, py, RIGHT_CHICANE_INNER_X, RIGHT_CHICANE_Y2, RIGHT_CHICANE_OUTER_X,
        RIGHT_CHICANE_Y2, right_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    right_offset += CHICANE_HORIZONTAL_LENGTH
    candidate_s, candidate_d2 = _project_arc_candidate_nb(
        px, py, RIGHT_CHICANE_C3_X, RIGHT_CHICANE_C3_Y, CHICANE_RADIUS,
        -0.5 * math.pi, math.pi, right_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    right_offset += CHICANE_HAIRPIN_LENGTH
    candidate_s, candidate_d2 = _project_segment_candidate_nb(
        px, py, RIGHT_CHICANE_OUTER_X, RIGHT_CHICANE_Y3, RIGHT_ARC_X,
        TOP_TRACK_Y, right_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )

    qx = _clamp_nb(px, LEFT_ARC_X, RIGHT_ARC_X)
    dx = px - qx
    dy = py - TOP_TRACK_Y
    candidate_s = STRAIGHT_LENGTH + CHICANE_LENGTH + (RIGHT_ARC_X - qx)
    candidate_d2 = dx * dx + dy * dy
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )

    left_offset = 2.0 * STRAIGHT_LENGTH + CHICANE_LENGTH
    candidate_s, candidate_d2 = _project_segment_candidate_nb(
        px, py, LEFT_ARC_X, LEFT_CHICANE_Y0, LEFT_CHICANE_OUTER_X,
        LEFT_CHICANE_Y0, left_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    left_offset += CHICANE_ENTRY_LENGTH
    candidate_s, candidate_d2 = _project_arc_candidate_nb(
        px, py, LEFT_CHICANE_C1_X, LEFT_CHICANE_C1_Y, CHICANE_RADIUS,
        0.5 * math.pi, math.pi, left_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    left_offset += CHICANE_HAIRPIN_LENGTH
    candidate_s, candidate_d2 = _project_segment_candidate_nb(
        px, py, LEFT_CHICANE_OUTER_X, LEFT_CHICANE_Y1, LEFT_CHICANE_INNER_X,
        LEFT_CHICANE_Y1, left_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    left_offset += CHICANE_HORIZONTAL_LENGTH
    candidate_s, candidate_d2 = _project_arc_candidate_nb(
        px, py, LEFT_CHICANE_C2_X, LEFT_CHICANE_C2_Y, CHICANE_RADIUS,
        0.5 * math.pi, -math.pi, left_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    left_offset += CHICANE_HAIRPIN_LENGTH
    candidate_s, candidate_d2 = _project_segment_candidate_nb(
        px, py, LEFT_CHICANE_INNER_X, LEFT_CHICANE_Y2, LEFT_CHICANE_OUTER_X,
        LEFT_CHICANE_Y2, left_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    left_offset += CHICANE_HORIZONTAL_LENGTH
    candidate_s, candidate_d2 = _project_arc_candidate_nb(
        px, py, LEFT_CHICANE_C3_X, LEFT_CHICANE_C3_Y, CHICANE_RADIUS,
        0.5 * math.pi, math.pi, left_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    left_offset += CHICANE_HAIRPIN_LENGTH
    candidate_s, candidate_d2 = _project_segment_candidate_nb(
        px, py, LEFT_CHICANE_OUTER_X, LEFT_CHICANE_Y3, LEFT_ARC_X,
        BOTTOM_TRACK_Y, left_offset, 0.0, 1e300,
    )
    best_s, best_d2, best_abs_ds = _accept_near_s_projection_nb(
        candidate_s, candidate_d2, reference_s, backward_limit, forward_limit,
        best_s, best_d2, best_abs_ds,
    )
    if best_d2 >= 1e299:
        return reference_s, best_d2
    return best_s, best_d2

@njit(cache=True, inline='always')
def _state_hits_circles_nb(state: np.ndarray, circle_centers: np.ndarray, circle_radii: np.ndarray, vehicle_length: float, vehicle_width: float, hard_collision_clearance: float) -> bool:
    """Conservative Ackermann rectangle-vs-circle collision test.

    Every physical polygon is conservatively covered by circles once, outside the
    rollout loop.  A cheap circumscribed-radius reject eliminates almost all
    circles before the oriented rectangle distance is evaluated.
    """
    count = circle_radii.shape[0]
    if count == 0:
        return False
    px = state[0]
    py = state[1]
    heading = state[2]
    c = math.cos(heading)
    sn = math.sin(heading)
    half_length = 0.5 * vehicle_length
    half_width = 0.5 * vehicle_width
    vehicle_radius = math.sqrt(half_length * half_length + half_width * half_width)
    for j in range(count):
        dx = circle_centers[j, 0] - px
        dy = circle_centers[j, 1] - py
        broad = vehicle_radius + circle_radii[j] + hard_collision_clearance
        if dx * dx + dy * dy > broad * broad:
            continue
        local_x = c * dx + sn * dy
        local_y = -sn * dx + c * dy
        qx = abs(local_x) - half_length
        qy = abs(local_y) - half_width
        outside_x = max(qx, 0.0)
        outside_y = max(qy, 0.0)
        outside = math.sqrt(outside_x * outside_x + outside_y * outside_y)
        inside = min(max(qx, qy), 0.0)
        clearance = outside + inside - circle_radii[j]
        if clearance < hard_collision_clearance:
            return True
    return False

@njit(cache=True, inline='always')
def _transition_hits_circles_nb(state0: np.ndarray, state1: np.ndarray, circle_centers: np.ndarray, circle_radii: np.ndarray, vehicle_length: float, vehicle_width: float, hard_collision_clearance: float, collision_substeps: int) -> bool:
    """Conservative swept transition check with cheap segment broad phase.

    The obstacle bank remains a conservative circular cover. For each circle that
    lies within the vehicle circumscribed-radius tube of the center transition,
    interpolate the Ackermann rectangle pose between the two simulator states.
    This closes the discrete-state tunneling gap without paying the substep cost
    for circles that are nowhere near the transition.
    """
    count = circle_radii.shape[0]
    if count == 0:
        return False
    x0 = state0[0]
    y0 = state0[1]
    x1 = state1[0]
    y1 = state1[1]
    dx_seg = x1 - x0
    dy_seg = y1 - y0
    seg2 = dx_seg * dx_seg + dy_seg * dy_seg
    psi0 = state0[2]
    psi1 = state1[2]
    dpsi = math.atan2(math.sin(psi1 - psi0), math.cos(psi1 - psi0))
    half_length = 0.5 * vehicle_length
    half_width = 0.5 * vehicle_width
    vehicle_radius = math.sqrt(half_length * half_length + half_width * half_width)
    samples = max(1, int(collision_substeps) + 1)
    for j in range(count):
        cx = circle_centers[j, 0]
        cy = circle_centers[j, 1]
        if seg2 > 1e-18:
            t = ((cx - x0) * dx_seg + (cy - y0) * dy_seg) / seg2
            t = min(max(t, 0.0), 1.0)
            qx_seg = x0 + t * dx_seg
            qy_seg = y0 + t * dy_seg
        else:
            qx_seg = x0
            qy_seg = y0
        ex = cx - qx_seg
        ey = cy - qy_seg
        broad = vehicle_radius + circle_radii[j] + hard_collision_clearance
        if ex * ex + ey * ey > broad * broad:
            continue
        for r in range(1, samples + 1):
            a = r / float(samples)
            px = x0 + a * dx_seg
            py = y0 + a * dy_seg
            heading = psi0 + a * dpsi
            c = math.cos(heading)
            sn = math.sin(heading)
            ox = cx - px
            oy = cy - py
            local_x = c * ox + sn * oy
            local_y = -sn * ox + c * oy
            qx = abs(local_x) - half_length
            qy = abs(local_y) - half_width
            outside_x = max(qx, 0.0)
            outside_y = max(qy, 0.0)
            outside = math.sqrt(outside_x * outside_x + outside_y * outside_y)
            inside = min(max(qx, qy), 0.0)
            clearance = outside + inside - circle_radii[j]
            if clearance < hard_collision_clearance:
                return True
    return False

@njit(cache=True, inline='always')
def _point_in_polygon_nb(px: float, py: float, poly: np.ndarray, n: int) -> bool:
    inside = False
    j = n - 1
    for i in range(n):
        xi = poly[i, 0]
        yi = poly[i, 1]
        xj = poly[j, 0]
        yj = poly[j, 1]
        if (yi > py) != (yj > py):
            x_cross = (xj - xi) * (py - yi) / (yj - yi + 1e-18) + xi
            if px < x_cross:
                inside = not inside
        j = i
    return inside

@njit(cache=True, inline='always')
def _vehicle_hits_polygons_nb(state: np.ndarray, vehicle_length: float, vehicle_width: float, polygons: np.ndarray, polygon_lengths: np.ndarray) -> bool:
    """Exact oriented Ackermann rectangle versus physical polygon geometry."""
    if polygon_lengths.shape[0] == 0:
        return False
    px = state[0]
    py = state[1]
    psi = state[2]
    c = math.cos(psi)
    s = math.sin(psi)
    hl = 0.5 * vehicle_length
    hw = 0.5 * vehicle_width
    corners = np.empty((4, 2), dtype=np.float64)
    local_x = (-hl, hl, hl, -hl)
    local_y = (-hw, -hw, hw, hw)
    for i in range(4):
        lx = local_x[i]
        ly = local_y[i]
        corners[i, 0] = px + c * lx - s * ly
        corners[i, 1] = py + s * lx + c * ly
    for m in range(polygon_lengths.shape[0]):
        n = int(polygon_lengths[m])
        if n < 3:
            continue
        poly = polygons[m]
        for i in range(4):
            if _point_in_polygon_nb(corners[i, 0], corners[i, 1], poly, n):
                return True
        for j in range(n):
            dx = poly[j, 0] - px
            dy = poly[j, 1] - py
            bx = c * dx + s * dy
            by = -s * dx + c * dy
            if -hl <= bx <= hl and -hw <= by <= hw:
                return True
        for i in range(4):
            k = (i + 1) % 4
            ax = corners[i, 0]
            ay = corners[i, 1]
            bx = corners[k, 0]
            by = corners[k, 1]
            for j in range(n):
                q = (j + 1) % n
                if _segments_intersect_nb(ax, ay, bx, by, poly[j, 0], poly[j, 1], poly[q, 0], poly[q, 1]):
                    return True
    return False

@njit(cache=True, inline='always')
def _interpolate_state_nb(state0: np.ndarray, state1: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(state0)
    for i in range(state0.shape[0]):
        out[i] = state0[i] + alpha * (state1[i] - state0[i])
    dpsi = math.atan2(math.sin(state1[2] - state0[2]), math.cos(state1[2] - state0[2]))
    out[2] = state0[2] + alpha * dpsi
    return out

@njit(cache=True)
def _first_exact_transition_collision_nb(state0: np.ndarray, state1: np.ndarray, vehicle_length: float, vehicle_width: float, polygons: np.ndarray, polygon_lengths: np.ndarray, collision_substeps: int) -> tuple[bool, np.ndarray]:
    """Return the first visibly colliding pose along one executed transition.

    Rollouts still use the fast conservative circular cover. This exact polygon test
    is executed only once for the accepted control transition, so it has negligible
    impact on MPPI runtime. A coarse adaptive sweep brackets first contact and a
    short bisection places the terminal animation frame on the collision boundary.
    """
    no_hit = state1.copy()
    if polygon_lengths.shape[0] == 0:
        return (False, no_hit)
    dx = state1[0] - state0[0]
    dy = state1[1] - state0[1]
    travel = math.sqrt(dx * dx + dy * dy)
    dpsi = math.atan2(math.sin(state1[2] - state0[2]), math.cos(state1[2] - state0[2]))
    spatial_step = max(0.02, 0.2 * min(vehicle_length, vehicle_width))
    spatial_samples = max(1, int(math.ceil(travel / spatial_step)))
    angular_samples = max(1, int(math.ceil(abs(dpsi) / math.radians(3.0))))
    samples = max(int(collision_substeps) + 1, spatial_samples, angular_samples)
    previous_alpha = 0.0
    previous_state = state0
    if _vehicle_hits_polygons_nb(previous_state, vehicle_length, vehicle_width, polygons, polygon_lengths):
        return (True, previous_state.copy())
    for r in range(1, samples + 1):
        alpha = r / float(samples)
        probe = _interpolate_state_nb(state0, state1, alpha)
        if _vehicle_hits_polygons_nb(probe, vehicle_length, vehicle_width, polygons, polygon_lengths):
            lo = previous_alpha
            hi = alpha
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                middle = _interpolate_state_nb(state0, state1, mid)
                if _vehicle_hits_polygons_nb(middle, vehicle_length, vehicle_width, polygons, polygon_lengths):
                    hi = mid
                else:
                    lo = mid
            return (True, _interpolate_state_nb(state0, state1, hi))
        previous_alpha = alpha
        previous_state = probe
    return (False, no_hit)

@njit(cache=True, parallel=True)
def _racing_rollout_costs_ackermann_nb(
    x0: np.ndarray,
    controls: np.ndarray,
    current_s: float,
    circle_centers: np.ndarray,
    circle_radii: np.ndarray,
    vehicle_length: float,
    vehicle_width: float,
    hard_collision_clearance: float,
    collision_substeps: int,
    dt: float,
    front_axle_distance: float,
    rear_axle_distance: float,
    mass: float,
    yaw_inertia: float,
    cornering_stiffness_front: float,
    cornering_stiffness_rear: float,
    tire_friction_coefficient: float,
    gravity: float,
    aerodynamic_drag_coefficient: float,
    rolling_resistance_force: float,
    minimum_tire_speed: float,
    dynamics_substeps: int,
    v_min: float,
    v_max: float,
    lateral_velocity_limit: float,
    yaw_rate_limit: float,
    accel_min: float,
    accel_max: float,
    steering_min: float,
    steering_max: float,
    steering_rate_min: float,
    steering_rate_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rollouts = controls.shape[0]
    horizon = controls.shape[1]
    costs = np.empty(n_rollouts, dtype=np.float64)
    collisions = np.zeros(n_rollouts, dtype=np.bool_)
    terminal_progress = np.empty(n_rollouts, dtype=np.float64)
    progress_step_limit = max(
        1.0,
        2.0 * max(abs(float(v_min)), abs(float(v_max)), abs(float(lateral_velocity_limit))) * float(dt) + 0.25,
    )
    for n in prange(n_rollouts):
        state = np.empty(7, dtype=np.float64)
        next_state = np.empty(7, dtype=np.float64)
        for j in range(7):
            state[j] = x0[j]
        previous_s = current_s
        cumulative = 0.0
        prefix_sum = 0.0
        hit = False
        for t in range(horizon):
            values = _dynamic_ackermann_step_nb(
                state,
                controls[n, t, 0],
                controls[n, t, 1],
                dt,
                front_axle_distance,
                rear_axle_distance,
                mass,
                yaw_inertia,
                cornering_stiffness_front,
                cornering_stiffness_rear,
                tire_friction_coefficient,
                gravity,
                aerodynamic_drag_coefficient,
                rolling_resistance_force,
                minimum_tire_speed,
                dynamics_substeps,
                v_min,
                v_max,
                lateral_velocity_limit,
                yaw_rate_limit,
                accel_min,
                accel_max,
                steering_min,
                steering_max,
                steering_rate_min,
                steering_rate_max,
            )
            for j in range(7):
                next_state[j] = values[j]
            if _transition_hits_circles_nb(
                state,
                next_state,
                circle_centers,
                circle_radii,
                vehicle_length,
                vehicle_width,
                hard_collision_clearance,
                collision_substeps,
            ):
                hit = True
                break
            s_mod, _ = _project_centerline_near_s_nb(
                next_state[0], next_state[1], previous_s,
                progress_step_limit, progress_step_limit,
            )
            cumulative += _signed_progress_delta_nb(s_mod, previous_s)
            previous_s = s_mod
            prefix_sum += cumulative
            for j in range(7):
                state[j] = next_state[j]
        collisions[n] = hit
        terminal_progress[n] = cumulative
        costs[n] = math.inf if hit else -prefix_sum / max(1, horizon)
    return costs, collisions, terminal_progress



@njit(cache=True, parallel=True)
def _racing_rollout_costs_four_wheel_nb(
    x0: np.ndarray,
    controls: np.ndarray,
    current_s: float,
    circle_centers: np.ndarray,
    circle_radii: np.ndarray,
    vehicle_length: float,
    vehicle_width: float,
    hard_collision_clearance: float,
    collision_substeps: int,
    dt: float,
    front_axle_distance: float,
    rear_axle_distance: float,
    mass: float,
    yaw_inertia: float,
    cornering_stiffness_front: float,
    cornering_stiffness_rear: float,
    tire_friction_coefficient: float,
    gravity: float,
    aerodynamic_drag_coefficient: float,
    rolling_resistance_force: float,
    minimum_tire_speed: float,
    dynamics_substeps: int,
    v_min: float,
    v_max: float,
    lateral_velocity_limit: float,
    yaw_rate_limit: float,
    accel_min: float,
    accel_max: float,
    steering_min: float,
    steering_max: float,
    steering_rate_min: float,
    steering_rate_max: float,
    track_width: float,
    wheel_radius: float,
    wheel_inertia: float,
    longitudinal_tire_stiffness: float,
    roll_inertia: float,
    roll_stiffness: float,
    roll_damping: float,
    cg_height: float,
    roll_center_height: float,
    wheel_damping: float,
    wheel_speed_limit: float,
    drive_bias_front: float,
    minimum_normal_load_fraction: float,
    roll_angle_limit: float,
    roll_rate_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rollouts = controls.shape[0]
    horizon = controls.shape[1]
    costs = np.empty(n_rollouts, dtype=np.float64)
    collisions = np.zeros(n_rollouts, dtype=np.bool_)
    terminal_progress = np.empty(n_rollouts, dtype=np.float64)
    progress_step_limit = max(
        1.0,
        2.0 * max(abs(float(v_min)), abs(float(v_max)), abs(float(lateral_velocity_limit))) * float(dt) + 0.25,
    )
    for n in prange(n_rollouts):
        state = np.empty(13, dtype=np.float64)
        next_state = np.empty(13, dtype=np.float64)
        for j in range(13):
            state[j] = x0[j]
        previous_s = current_s
        cumulative = 0.0
        prefix_sum = 0.0
        hit = False
        for t in range(horizon):
            values = _dynamic_four_wheel_step_nb(
                state,
                controls[n, t, 0],
                controls[n, t, 1],
                dt,
                front_axle_distance,
                rear_axle_distance,
                mass,
                yaw_inertia,
                cornering_stiffness_front,
                cornering_stiffness_rear,
                tire_friction_coefficient,
                gravity,
                aerodynamic_drag_coefficient,
                rolling_resistance_force,
                minimum_tire_speed,
                dynamics_substeps,
                v_min,
                v_max,
                lateral_velocity_limit,
                yaw_rate_limit,
                accel_min,
                accel_max,
                steering_min,
                steering_max,
                steering_rate_min,
                steering_rate_max,
                track_width,
                wheel_radius,
                wheel_inertia,
                longitudinal_tire_stiffness,
                roll_inertia,
                roll_stiffness,
                roll_damping,
                cg_height,
                roll_center_height,
                wheel_damping,
                wheel_speed_limit,
                drive_bias_front,
                minimum_normal_load_fraction,
                roll_angle_limit,
                roll_rate_limit,
            )
            for j in range(13):
                next_state[j] = values[j]
            if _transition_hits_circles_nb(
                state,
                next_state,
                circle_centers,
                circle_radii,
                vehicle_length,
                vehicle_width,
                hard_collision_clearance,
                collision_substeps,
            ):
                hit = True
                break
            s_mod, _ = _project_centerline_near_s_nb(
                next_state[0], next_state[1], previous_s,
                progress_step_limit, progress_step_limit,
            )
            cumulative += _signed_progress_delta_nb(s_mod, previous_s)
            previous_s = s_mod
            prefix_sum += cumulative
            for j in range(13):
                state[j] = next_state[j]
        collisions[n] = hit
        terminal_progress[n] = cumulative
        costs[n] = math.inf if hit else -prefix_sum / max(1, horizon)
    return costs, collisions, terminal_progress


def _evaluate_control_batch(
    state: np.ndarray,
    controls: np.ndarray,
    current_s: float,
    cfg: object,
    model: object,
    circle_centers: np.ndarray,
    circle_radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = _racing_rollout_costs_ackermann_nb if model.MODEL_NAME == 'ackermann' else _racing_rollout_costs_four_wheel_nb
    return kernel(
        np.ascontiguousarray(np.asarray(state, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(controls, dtype=np.float64)),
        float(current_s),
        np.ascontiguousarray(np.asarray(circle_centers, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(circle_radii, dtype=np.float64)),
        float(cfg.vehicle_length),
        float(cfg.vehicle_width),
        float(cfg.hard_collision_clearance),
        int(cfg.collision_substeps),
        *model._dynamic_model_arguments(cfg),
    )


@njit(cache=True, parallel=True)
def _localize_prior_ranges_nb(
    mean_paths: np.ndarray,
    arc_lengths: np.ndarray,
    lengths: np.ndarray,
    localization_lengths: np.ndarray,
    localization_unique_lengths: np.ndarray,
    localization_block_mins: np.ndarray,
    localization_block_maxs: np.ndarray,
    localization_block_counts: np.ndarray,
    active_indices: np.ndarray,
    cursors: np.ndarray,
    px: float,
    py: float,
    current_s: float,
    horizon: int,
    step_distance: float,
    search_back: int,
    search_forward: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Localize each prior by ordered track progress, then by XY distance.

    XY alone is ambiguous in a folded chicane.  The executed track progress is the
    causal coordinate: a point on another hairpin branch can be physically close but
    is many metres away in track arc length.  Searching the first-lap prior by track
    progress makes the selected prior window deterministic even after cursor resets,
    wall changes, or non-sequential visualization frame access.
    """
    count = active_indices.shape[0]
    starts = np.zeros(count, dtype=np.int64)
    ends = np.zeros(count, dtype=np.int64)
    updated = cursors.copy()
    preview_span = max(0, int(horizon) - 1) * max(float(step_distance), 1e-09)
    current_s %= TRACK_LENGTH

    for q in prange(count):
        m = int(active_indices[q])
        n = int(lengths[m])
        nloc = min(int(localization_lengths[m]), n)
        unique_n = min(int(localization_unique_lengths[m]), nloc)
        if n < 2 or unique_n < 1:
            starts[q] = 0
            ends[q] = max(0, n - 1)
            updated[m] = 0
            continue

        previous = int(cursors[m])
        if previous >= 0 and previous < unique_n:
            previous %= unique_n
        else:
            previous = -1

        nearest = 0
        best_progress_error = 1e300
        best_xy = 1e300
        best_cursor_arc = 1e300
        previous_arc = arc_lengths[m, previous] if previous >= 0 else 0.0
        loop_arc = arc_lengths[m, nloc - 1] - arc_lengths[m, 0]

        for i in range(unique_n):
            point_s, _ = _project_centerline_nb(mean_paths[m, i, 0], mean_paths[m, i, 1])
            progress_error = abs(_signed_progress_delta_nb(point_s, current_s))
            dx = mean_paths[m, i, 0] - px
            dy = mean_paths[m, i, 1] - py
            d2 = dx * dx + dy * dy

            cursor_arc = 0.0
            if previous >= 0:
                cursor_arc = arc_lengths[m, i] - previous_arc
                if loop_arc > 1e-12:
                    half_loop_arc = 0.5 * loop_arc
                    if cursor_arc > half_loop_arc:
                        cursor_arc -= loop_arc
                    elif cursor_arc < -half_loop_arc:
                        cursor_arc += loop_arc
                cursor_arc = abs(cursor_arc)

            if (
                progress_error < best_progress_error - 1e-9
                or (
                    abs(progress_error - best_progress_error) <= 1e-9
                    and (
                        d2 < best_xy - 1e-12
                        or (abs(d2 - best_xy) <= 1e-12 and cursor_arc < best_cursor_arc)
                    )
                )
            ):
                nearest = i
                best_progress_error = progress_error
                best_xy = d2
                best_cursor_arc = cursor_arc

        start = min(nearest, n - 2)
        updated[m] = start
        s0 = arc_lengths[m, start]
        preview_end = min(s0 + preview_span, arc_lengths[m, n - 1])
        end = start + 1
        while end < n - 1 and arc_lengths[m, end] < preview_end:
            end += 1
        starts[q] = start
        ends[q] = end
    return starts, ends, updated


@njit(cache=True, parallel=True)
def _pack_local_prior_windows_nb(
    mean_paths: np.ndarray,
    cov_blocks: np.ndarray,
    arc_lengths: np.ndarray,
    active_indices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = active_indices.shape[0]
    max_len = 2
    for q in range(count):
        n = int(ends[q] - starts[q] + 1)
        if n > max_len:
            max_len = n
    refs = np.zeros((count, max_len, 2), dtype=np.float64)
    covs = np.zeros((count, max_len, 2, 2), dtype=np.float64)
    arcs = np.zeros((count, max_len), dtype=np.float64)
    lengths = np.zeros(count, dtype=np.int64)
    for q in prange(count):
        m = int(active_indices[q])
        a = int(starts[q])
        b = int(ends[q])
        n = b - a + 1
        lengths[q] = n
        s0 = arc_lengths[m, a]
        for i in range(n):
            src = a + i
            refs[q, i, 0] = mean_paths[m, src, 0]
            refs[q, i, 1] = mean_paths[m, src, 1]
            covs[q, i, 0, 0] = cov_blocks[m, src, 0, 0]
            covs[q, i, 0, 1] = cov_blocks[m, src, 0, 1]
            covs[q, i, 1, 0] = cov_blocks[m, src, 1, 0]
            covs[q, i, 1, 1] = cov_blocks[m, src, 1, 1]
            arcs[q, i] = arc_lengths[m, src] - s0
    return refs, covs, arcs, lengths


@njit(cache=True, parallel=True)
def _localize_sample_paths_nb(
    sample_paths: np.ndarray,
    sample_arc_lengths: np.ndarray,
    sample_lengths: np.ndarray,
    sample_ids: np.ndarray,
    px: float,
    py: float,
    current_s: float,
    horizon: int,
    step_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = sample_ids.shape[0]
    starts = np.zeros(count, dtype=np.int64)
    ends = np.zeros(count, dtype=np.int64)
    preview_span = max(0, int(horizon) - 1) * max(float(step_distance), 1e-09)
    current_s %= TRACK_LENGTH
    for q in prange(count):
        sample_id = int(sample_ids[q])
        n = int(sample_lengths[sample_id])
        nearest = 0
        best_progress_error = 1e300
        best_xy = 1e300
        for i in range(n):
            point_s, _ = _project_centerline_nb(sample_paths[sample_id, i, 0], sample_paths[sample_id, i, 1])
            progress_error = abs(_signed_progress_delta_nb(point_s, current_s))
            dx = sample_paths[sample_id, i, 0] - px
            dy = sample_paths[sample_id, i, 1] - py
            d2 = dx * dx + dy * dy
            if (
                progress_error < best_progress_error - 1e-9
                or (abs(progress_error - best_progress_error) <= 1e-9 and d2 < best_xy)
            ):
                best_progress_error = progress_error
                best_xy = d2
                nearest = i
        start = min(nearest, max(0, n - 2))
        s0 = sample_arc_lengths[sample_id, start]
        preview_end = min(s0 + preview_span, sample_arc_lengths[sample_id, n - 1])
        end = min(start + 1, n - 1)
        while end < n - 1 and sample_arc_lengths[sample_id, end] < preview_end:
            end += 1
        starts[q] = start
        ends[q] = end
    max_len = 2
    for q in range(count):
        n = int(ends[q] - starts[q] + 1)
        if n > max_len:
            max_len = n
    refs = np.zeros((count, max_len, 2), dtype=np.float64)
    lengths = np.zeros(count, dtype=np.int64)
    for q in prange(count):
        sample_id = int(sample_ids[q])
        a = int(starts[q])
        b = int(ends[q])
        n = b - a + 1
        lengths[q] = n
        for i in range(n):
            refs[q, i, 0] = sample_paths[sample_id, a + i, 0]
            refs[q, i, 1] = sample_paths[sample_id, a + i, 1]
    return refs, lengths


@njit(cache=True, inline='always')
def _path_heading_nb(path: np.ndarray, n: int, i: int) -> float:
    if n <= 1:
        return 0.0
    if i <= 0:
        dx = path[1, 0] - path[0, 0]
        dy = path[1, 1] - path[0, 1]
    elif i >= n - 1:
        dx = path[n - 1, 0] - path[n - 2, 0]
        dy = path[n - 1, 1] - path[n - 2, 1]
    else:
        dx = path[i + 1, 0] - path[i - 1, 0]
        dy = path[i + 1, 1] - path[i - 1, 1]
    return math.atan2(dy, dx)

@njit(cache=True, parallel=True)
def _prior_feasible_mask_nb(mean_paths: np.ndarray, localization_lengths: np.ndarray, circle_centers: np.ndarray, circle_radii: np.ndarray, vehicle_length: float, vehicle_width: float, hard_collision_clearance: float, substeps: int) -> np.ndarray:
    """Check complete first-lap prior means against dynamic-wall circles in parallel."""
    mode_count = localization_lengths.shape[0]
    feasible = np.ones(mode_count, dtype=np.bool_)
    interp = max(0, int(substeps))
    for m in prange(mode_count):
        n = int(localization_lengths[m])
        if n <= 0:
            feasible[m] = False
            continue
        path = mean_paths[m]
        hit = False
        for i in range(n):
            heading = _path_heading_nb(path, n, i)
            state = np.empty(3, dtype=np.float64)
            state[0] = path[i, 0]
            state[1] = path[i, 1]
            state[2] = heading
            if _state_hits_circles_nb(state, circle_centers, circle_radii, vehicle_length, vehicle_width, hard_collision_clearance):
                hit = True
                break
        if not hit and interp > 0 and (n > 1):
            denom = float(interp + 1)
            for i in range(n - 1):
                h0 = _path_heading_nb(path, n, i)
                h1 = _path_heading_nb(path, n, i + 1)
                dh = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))
                for r in range(1, interp + 1):
                    a = r / denom
                    state = np.empty(3, dtype=np.float64)
                    state[0] = path[i, 0] + a * (path[i + 1, 0] - path[i, 0])
                    state[1] = path[i, 1] + a * (path[i + 1, 1] - path[i, 1])
                    state[2] = h0 + a * dh
                    if _state_hits_circles_nb(state, circle_centers, circle_radii, vehicle_length, vehicle_width, hard_collision_clearance):
                        hit = True
                        break
                if hit:
                    break
        feasible[m] = not hit
    return feasible

def _arc_point_tangent(cx: float, cy: float, theta: float, turn_sign: float) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray([cx + CHICANE_RADIUS * math.cos(theta), cy + CHICANE_RADIUS * math.sin(theta)], dtype=np.float64)
    if turn_sign > 0.0:
        tangent = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    else:
        tangent = np.asarray([math.sin(theta), -math.cos(theta)], dtype=np.float64)
    return point, tangent

def _right_chicane_point_tangent(local_s: float) -> tuple[np.ndarray, np.ndarray]:
    s = min(max(float(local_s), 0.0), CHICANE_LENGTH)
    if s < CHICANE_ENTRY_LENGTH:
        return np.asarray([RIGHT_ARC_X + s, RIGHT_CHICANE_Y0], dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)
    s -= CHICANE_ENTRY_LENGTH
    if s < CHICANE_HAIRPIN_LENGTH:
        theta = -0.5 * math.pi + s / CHICANE_RADIUS
        return _arc_point_tangent(RIGHT_CHICANE_C1_X, RIGHT_CHICANE_C1_Y, theta, 1.0)
    s -= CHICANE_HAIRPIN_LENGTH
    if s < CHICANE_HORIZONTAL_LENGTH:
        return np.asarray([RIGHT_CHICANE_OUTER_X - s, RIGHT_CHICANE_Y1], dtype=np.float64), np.asarray([-1.0, 0.0], dtype=np.float64)
    s -= CHICANE_HORIZONTAL_LENGTH
    if s < CHICANE_HAIRPIN_LENGTH:
        theta = -0.5 * math.pi - s / CHICANE_RADIUS
        return _arc_point_tangent(RIGHT_CHICANE_C2_X, RIGHT_CHICANE_C2_Y, theta, -1.0)
    s -= CHICANE_HAIRPIN_LENGTH
    if s < CHICANE_HORIZONTAL_LENGTH:
        return np.asarray([RIGHT_CHICANE_INNER_X + s, RIGHT_CHICANE_Y2], dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)
    s -= CHICANE_HORIZONTAL_LENGTH
    if s < CHICANE_HAIRPIN_LENGTH:
        theta = -0.5 * math.pi + s / CHICANE_RADIUS
        return _arc_point_tangent(RIGHT_CHICANE_C3_X, RIGHT_CHICANE_C3_Y, theta, 1.0)
    s -= CHICANE_HAIRPIN_LENGTH
    return np.asarray([RIGHT_CHICANE_OUTER_X - s, RIGHT_CHICANE_Y3], dtype=np.float64), np.asarray([-1.0, 0.0], dtype=np.float64)

def _left_chicane_point_tangent(local_s: float) -> tuple[np.ndarray, np.ndarray]:
    s = min(max(float(local_s), 0.0), CHICANE_LENGTH)
    if s < CHICANE_ENTRY_LENGTH:
        return np.asarray([LEFT_ARC_X - s, LEFT_CHICANE_Y0], dtype=np.float64), np.asarray([-1.0, 0.0], dtype=np.float64)
    s -= CHICANE_ENTRY_LENGTH
    if s < CHICANE_HAIRPIN_LENGTH:
        theta = 0.5 * math.pi + s / CHICANE_RADIUS
        return _arc_point_tangent(LEFT_CHICANE_C1_X, LEFT_CHICANE_C1_Y, theta, 1.0)
    s -= CHICANE_HAIRPIN_LENGTH
    if s < CHICANE_HORIZONTAL_LENGTH:
        return np.asarray([LEFT_CHICANE_OUTER_X + s, LEFT_CHICANE_Y1], dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)
    s -= CHICANE_HORIZONTAL_LENGTH
    if s < CHICANE_HAIRPIN_LENGTH:
        theta = 0.5 * math.pi - s / CHICANE_RADIUS
        return _arc_point_tangent(LEFT_CHICANE_C2_X, LEFT_CHICANE_C2_Y, theta, -1.0)
    s -= CHICANE_HAIRPIN_LENGTH
    if s < CHICANE_HORIZONTAL_LENGTH:
        return np.asarray([LEFT_CHICANE_INNER_X - s, LEFT_CHICANE_Y2], dtype=np.float64), np.asarray([-1.0, 0.0], dtype=np.float64)
    s -= CHICANE_HORIZONTAL_LENGTH
    if s < CHICANE_HAIRPIN_LENGTH:
        theta = 0.5 * math.pi + s / CHICANE_RADIUS
        return _arc_point_tangent(LEFT_CHICANE_C3_X, LEFT_CHICANE_C3_Y, theta, 1.0)
    s -= CHICANE_HAIRPIN_LENGTH
    return np.asarray([LEFT_CHICANE_OUTER_X + s, LEFT_CHICANE_Y3], dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)

def centerline_point_tangent(s_value: float) -> tuple[np.ndarray, np.ndarray]:
    s = float(s_value % TRACK_LENGTH)
    if s < STRAIGHT_LENGTH:
        return np.asarray([LEFT_ARC_X + s, BOTTOM_TRACK_Y], dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)
    s -= STRAIGHT_LENGTH
    if s < CHICANE_LENGTH:
        return _right_chicane_point_tangent(s)
    s -= CHICANE_LENGTH
    if s < STRAIGHT_LENGTH:
        return np.asarray([RIGHT_ARC_X - s, TOP_TRACK_Y], dtype=np.float64), np.asarray([-1.0, 0.0], dtype=np.float64)
    s -= STRAIGHT_LENGTH
    return _left_chicane_point_tangent(s)

def sample_centerline(start_s: float, distance: float, count: int) -> np.ndarray:
    values = np.linspace(float(start_s), float(start_s) + float(distance), int(count))
    points = np.empty((len(values), 2), dtype=np.float64)
    for i, value in enumerate(values):
        points[i] = centerline_point_tangent(float(value))[0]
    return points

def racing_prior_covariance(reference: np.ndarray) -> np.ndarray:
    """Fixed NASCAR covariance used only on the two deterministic chicanes."""
    variance = TURN_PRIOR_SIGMA * TURN_PRIOR_SIGMA
    cov = np.zeros((len(reference), 2, 2), dtype=np.float64)
    cov[:, 0, 0] = variance
    cov[:, 1, 1] = variance
    return np.ascontiguousarray(cov)
P2 = np.asarray([LEFT_ARC_X, BOTTOM_TRACK_Y], dtype=np.float64)
P3 = np.asarray([RIGHT_ARC_X, BOTTOM_TRACK_Y], dtype=np.float64)
P4 = np.asarray([RIGHT_ARC_X, TOP_TRACK_Y], dtype=np.float64)
P1 = np.asarray([LEFT_ARC_X, TOP_TRACK_Y], dtype=np.float64)

FISH_P2 = np.asarray([OBSTACLE_STRAIGHT_LEFT_X, BOTTOM_TRACK_Y], dtype=np.float64)
FISH_P3 = np.asarray([OBSTACLE_STRAIGHT_RIGHT_X, BOTTOM_TRACK_Y], dtype=np.float64)
FISH_P4 = np.asarray([OBSTACLE_STRAIGHT_RIGHT_X, TOP_TRACK_Y], dtype=np.float64)
FISH_P1 = np.asarray([OBSTACLE_STRAIGHT_LEFT_X, TOP_TRACK_Y], dtype=np.float64)
TURN_POINTS = 80
MAX_EMPIRICAL_PATHS_PER_JOINT_MODE = 24
PRIOR_LOCALIZATION_BLOCK_SIZE = 32
MAX_PRIOR = 20

def _poly_vertices(obstacle: object) -> np.ndarray:
    vertices = getattr(obstacle, 'vertices', obstacle)
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f'Unsupported obstacle geometry with shape {array.shape}.')
    return np.ascontiguousarray(array[:, :2])

def _segment_rigid_transform(source_start: np.ndarray, source_goal: np.ndarray, target_start: np.ndarray, target_goal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (source origin, target origin, R) for a 1:1 start-goal map."""
    s0 = np.asarray(source_start, dtype=np.float64)[:2]
    s1 = np.asarray(source_goal, dtype=np.float64)[:2]
    t0 = np.asarray(target_start, dtype=np.float64)[:2]
    t1 = np.asarray(target_goal, dtype=np.float64)[:2]
    sd = s1 - s0
    td = t1 - t0
    sl = float(np.linalg.norm(sd))
    tl = float(np.linalg.norm(td))
    if sl <= 1e-12 or tl <= 1e-12:
        raise ValueError('Cannot transform a zero-length start-goal segment.')
    if not math.isclose(sl, tl, rel_tol=0.0, abs_tol=1e-09):
        raise ValueError(f'NASCAR straight must match the controller start-goal distance exactly: {tl} != {sl}.')
    st = sd / sl
    sn = np.asarray([-st[1], st[0]], dtype=np.float64)
    tt = td / tl
    tn = np.asarray([-tt[1], tt[0]], dtype=np.float64)
    R = np.outer(tt, st) + np.outer(tn, sn)
    return (s0, t0, np.ascontiguousarray(R, dtype=np.float64))

def _rigid_transform_points(points: np.ndarray, source_start: np.ndarray, source_goal: np.ndarray, target_start: np.ndarray, target_goal: np.ndarray) -> np.ndarray:
    s0, t0, R = _segment_rigid_transform(source_start, source_goal, target_start, target_goal)
    p = np.asarray(points, dtype=np.float64)[:, :2]
    return np.ascontiguousarray(t0[None, :] + (p - s0[None, :]) @ R.T, dtype=np.float64)

def _rigid_transform_covariances(covariances: np.ndarray, source_start: np.ndarray, source_goal: np.ndarray, target_start: np.ndarray, target_goal: np.ndarray) -> np.ndarray:
    _, _, R = _segment_rigid_transform(source_start, source_goal, target_start, target_goal)
    cov = np.asarray(covariances, dtype=np.float64)
    if cov.ndim != 3 or cov.shape[1:] != (2, 2):
        raise ValueError(f'Expected covariance blocks with shape (N,2,2), got {cov.shape}.')
    mapped = np.einsum('ab,nbc,dc->nad', R, cov, R, optimize=True)
    return np.ascontiguousarray(0.5 * (mapped + np.swapaxes(mapped, 1, 2)), dtype=np.float64)

def _transform_planner_mode(mode: controller_core.MPPIHomotopyMode, source_start: np.ndarray, source_goal: np.ndarray, target_start: np.ndarray, target_goal: np.ndarray) -> controller_core.MPPIHomotopyMode:
    mean = _rigid_transform_points(mode.mean_path, source_start, source_goal, target_start, target_goal)
    cov = _rigid_transform_covariances(mode.cov_blocks, source_start, source_goal, target_start, target_goal)
    mean[0] = np.asarray(target_start, dtype=np.float64)[:2]
    mean[-1] = np.asarray(target_goal, dtype=np.float64)[:2]
    samples: list[np.ndarray] = []
    for raw in list(mode.sample_paths or []):
        path = _rigid_transform_points(raw, source_start, source_goal, target_start, target_goal)
        path[0] = np.asarray(target_start, dtype=np.float64)[:2]
        path[-1] = np.asarray(target_goal, dtype=np.float64)[:2]
        samples.append(path)
    return controller_core.prepare_mode_prior_cache(controller_core.MPPIHomotopyMode(signature=tuple(mode.signature), probability=float(mode.probability), mean_path=mean, cov_blocks=cov, sample_paths=samples))

def _scale_polygon_about_centroid(vertices: np.ndarray, scale: float) -> np.ndarray:
    p = np.asarray(vertices, dtype=np.float64)[:, :2]
    factor = float(scale)
    if factor <= 0.0:
        raise ValueError('Obstacle scale must be positive.')
    center = np.mean(p, axis=0)
    return np.ascontiguousarray(center[None, :] + factor * (p - center[None, :]), dtype=np.float64)

def _sample_chicane_geometry(side: str, count: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.linspace(0.0, CHICANE_LENGTH, max(2, int(count)))
    points = np.empty((len(values), 2), dtype=np.float64)
    tangents = np.empty((len(values), 2), dtype=np.float64)
    sampler = _right_chicane_point_tangent if side == 'right' else _left_chicane_point_tangent
    for i, value in enumerate(values):
        point, tangent = sampler(float(value))
        points[i] = point
        tangents[i] = tangent
    return points, tangents

def _chicane_boundary_polygon(side: str, lateral_sign: float) -> np.ndarray:
    half_lane = 0.5 * TURN_BARRIER_LANE_WIDTH
    half_thickness = 0.5 * TURN_BARRIER_THICKNESS
    if half_lane <= half_thickness:
        raise ValueError('Chicane lane width must exceed the turn-barrier thickness.')
    points, tangents = _sample_chicane_geometry(side, int(TURN_BARRIER_POINTS))
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    center_offset = float(lateral_sign) * half_lane
    edge_a = points + (center_offset - half_thickness) * normals
    edge_b = points + (center_offset + half_thickness) * normals
    return np.ascontiguousarray(np.vstack((edge_a, edge_b[::-1])), dtype=np.float64)

def _chicane_entry_guard_walls() -> list[object]:
    half_lane = 0.5 * TURN_BARRIER_LANE_WIDTH
    right_a = np.asarray([RIGHT_CHICANE_INNER_X - CHICANE_INNER_X_OFFSET, RIGHT_CHICANE_Y0 - half_lane], dtype=np.float64)
    right_b = np.asarray([RIGHT_CHICANE_INNER_X - CHICANE_INNER_X_OFFSET, RIGHT_CHICANE_Y0 - half_lane - CHICANE_ENTRY_GUARD_LENGTH], dtype=np.float64)
    left_a = np.asarray([LEFT_CHICANE_INNER_X + CHICANE_INNER_X_OFFSET, LEFT_CHICANE_Y0 + half_lane], dtype=np.float64)
    left_b = np.asarray([LEFT_CHICANE_INNER_X + CHICANE_INNER_X_OFFSET, LEFT_CHICANE_Y0 + half_lane + CHICANE_ENTRY_GUARD_LENGTH], dtype=np.float64)
    return [
        controller_core.make_wall_between_points(right_a, right_b, width=TURN_BARRIER_THICKNESS, extension=0.0),
        controller_core.make_wall_between_points(left_a, left_b, width=TURN_BARRIER_THICKNESS, extension=0.0),
    ]

def _build_turn_barriers() -> list[object]:
    PolyObstacle, *_ = controller_core._planner_symbols()
    barriers = [
        PolyObstacle(_chicane_boundary_polygon('right', 1.0)),
        PolyObstacle(_chicane_boundary_polygon('right', -1.0)),
        PolyObstacle(_chicane_boundary_polygon('left', 1.0)),
        PolyObstacle(_chicane_boundary_polygon('left', -1.0)),
    ]
    barriers.extend(_chicane_entry_guard_walls())
    return barriers

def _build_inner_straight_barriers() -> list[object]:
    """Build the H-shaped infield barrier used for shortcut prevention.

    The vertical walls join the two endpoints of each inner chicane barrier,
    and the horizontal wall joins the midpoints of those chords.
    """
    half_lane = 0.5 * TURN_BARRIER_LANE_WIDTH
    inner_radius = CENTERLINE_RADIUS - half_lane
    bottom_y = TRACK_CENTER_Y - inner_radius
    top_y = TRACK_CENTER_Y + inner_radius
    mid_y = 0.5 * (bottom_y + top_y)
    extension = float(INNER_STRAIGHT_BARRIER_EXTENSION)
    left_bottom = np.asarray([LEFT_ARC_X, bottom_y], dtype=np.float64)
    left_top = np.asarray([LEFT_ARC_X, top_y], dtype=np.float64)
    right_bottom = np.asarray([RIGHT_ARC_X, bottom_y], dtype=np.float64)
    right_top = np.asarray([RIGHT_ARC_X, top_y], dtype=np.float64)
    left_mid = np.asarray([LEFT_ARC_X, mid_y], dtype=np.float64)
    right_mid = np.asarray([RIGHT_ARC_X, mid_y], dtype=np.float64)
    return [controller_core.make_wall_between_points(left_bottom, left_top, width=TURN_BARRIER_THICKNESS, extension=extension), controller_core.make_wall_between_points(right_bottom, right_top, width=TURN_BARRIER_THICKNESS, extension=extension), controller_core.make_wall_between_points(left_mid, right_mid, width=TURN_BARRIER_THICKNESS, extension=extension)]

def _build_fixed_barriers() -> tuple[list[object], list[object]]:
    """Return (all fixed barriers, straight inner barriers used for prior filtering)."""
    turn_barriers = _build_turn_barriers()
    inner_straight_barriers = _build_inner_straight_barriers()
    return (turn_barriers + inner_straight_barriers, inner_straight_barriers)

def _turn_barrier_collision_circles() -> list[tuple[np.ndarray, float]]:
    half_lane = 0.5 * TURN_BARRIER_LANE_WIDTH
    half_thickness = 0.5 * TURN_BARRIER_THICKNESS
    max_seg = max(1e-06, float(TURN_BARRIER_COLLISION_SEGMENT_LENGTH))
    segment_count = max(2, int(math.ceil(CHICANE_LENGTH / max_seg)))
    circles: list[tuple[np.ndarray, float]] = []
    guards = _chicane_entry_guard_walls()
    for side_index, side in enumerate(('right', 'left')):
        sampler = _right_chicane_point_tangent if side == 'right' else _left_chicane_point_tangent
        for lateral_sign in (1.0, -1.0):
            for i in range(segment_count):
                s0 = CHICANE_LENGTH * i / float(segment_count)
                s1 = CHICANE_LENGTH * (i + 1) / float(segment_count)
                sm = 0.5 * (s0 + s1)
                pm, tm = sampler(sm)
                nm = np.asarray([-tm[1], tm[0]], dtype=np.float64)
                circle_center = pm + lateral_sign * half_lane * nm
                max_centerline_distance = 0.0
                for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                    sp = s0 + fraction * (s1 - s0)
                    pp, tp = sampler(sp)
                    np_left = np.asarray([-tp[1], tp[0]], dtype=np.float64)
                    barrier_point = pp + lateral_sign * half_lane * np_left
                    distance = float(np.linalg.norm(barrier_point - circle_center))
                    max_centerline_distance = max(max_centerline_distance, distance)
                circles.append((np.asarray(circle_center, dtype=np.float64), max_centerline_distance + half_thickness + 1e-06))
        circles.extend(controller_core.obstacle_bounding_circles([guards[side_index]]))
    return circles

def _build_collision_sector_bank(lower_obstacles: list[object], upper_obstacles: list[object], inner_straight_barriers: list[object]) -> CollisionSectorBank:
    lower_centers, lower_radii = ackermann.pack_obstacle_circles(
        controller_core.obstacle_bounding_circles(list(lower_obstacles))
    )
    upper_centers, upper_radii = ackermann.pack_obstacle_circles(
        controller_core.obstacle_bounding_circles(list(upper_obstacles))
    )
    shared_centers, shared_radii = ackermann.pack_obstacle_circles(
        controller_core.obstacle_bounding_circles(list(inner_straight_barriers))
    )
    turn_circles = _turn_barrier_collision_circles()
    half = len(turn_circles) // 2
    right_centers, right_radii = ackermann.pack_obstacle_circles(turn_circles[:half])
    left_centers, left_radii = ackermann.pack_obstacle_circles(turn_circles[half:])
    sectors = (
        (np.ascontiguousarray(lower_centers, dtype=np.float64), np.ascontiguousarray(lower_radii, dtype=np.float64)),
        (np.ascontiguousarray(right_centers, dtype=np.float64), np.ascontiguousarray(right_radii, dtype=np.float64)),
        (np.ascontiguousarray(upper_centers, dtype=np.float64), np.ascontiguousarray(upper_radii, dtype=np.float64)),
        (np.ascontiguousarray(left_centers, dtype=np.float64), np.ascontiguousarray(left_radii, dtype=np.float64)),
    )
    shared = (
        np.ascontiguousarray(shared_centers, dtype=np.float64),
        np.ascontiguousarray(shared_radii, dtype=np.float64),
    )
    sector_mins = np.zeros((4, 2), dtype=np.float64)
    sector_maxs = np.zeros((4, 2), dtype=np.float64)
    for section, (centers, radii) in enumerate(sectors):
        if len(radii):
            sector_mins[section] = np.min(centers - radii[:, None], axis=0)
            sector_maxs[section] = np.max(centers + radii[:, None], axis=0)
        else:
            sector_mins[section] = np.asarray([np.inf, np.inf], dtype=np.float64)
            sector_maxs[section] = np.asarray([-np.inf, -np.inf], dtype=np.float64)
    mask_centers: list[np.ndarray] = []
    mask_radii: list[np.ndarray] = []
    for mask in range(16):
        parts = [shared]
        for section in range(4):
            if mask & (1 << section):
                parts.append(sectors[section])
        centers, radii = _concat_circle_arrays(parts)
        mask_centers.append(centers)
        mask_radii.append(radii)
    return CollisionSectorBank(
        tuple(mask_centers),
        tuple(mask_radii),
        np.ascontiguousarray(sector_mins),
        np.ascontiguousarray(sector_maxs),
    )


def _sector_mask_for_prediction(
    current_s: float,
    state: np.ndarray,
    cfg: ackermann.MPPIConfig,
    bank: CollisionSectorBank,
) -> int:
    backward = max(0.0, -float(cfg.v_min)) * float(cfg.dt) * int(cfg.horizon)
    forward = max(0.0, float(cfg.v_max)) * float(cfg.dt) * int(cfg.horizon)
    margin = float(cfg.vehicle_length) + float(cfg.hard_collision_clearance) + 0.5
    lo = float(current_s) - backward - margin
    hi = float(current_s) + forward + margin
    if hi - lo >= TRACK_LENGTH:
        return 15
    boundaries = (
        0.0,
        STRAIGHT_LENGTH,
        STRAIGHT_LENGTH + CHICANE_LENGTH,
        2.0 * STRAIGHT_LENGTH + CHICANE_LENGTH,
        TRACK_LENGTH,
    )
    intervals: list[tuple[float, float]] = []
    lo_mod = lo % TRACK_LENGTH
    hi_mod = hi % TRACK_LENGTH
    if lo_mod <= hi_mod and math.floor(lo / TRACK_LENGTH) == math.floor(hi / TRACK_LENGTH):
        intervals.append((lo_mod, hi_mod))
    else:
        intervals.append((lo_mod, TRACK_LENGTH))
        intervals.append((0.0, hi_mod))
    mask = 0
    for section in range(4):
        a = boundaries[section]
        b = boundaries[section + 1]
        for x0, x1 in intervals:
            if x1 >= a and x0 <= b:
                mask |= 1 << section
                break
    px = float(state[0])
    py = float(state[1])
    speed_bound = math.hypot(
        max(abs(float(cfg.v_min)), abs(float(cfg.v_max))),
        abs(float(cfg.lateral_velocity_limit)),
    )
    travel = speed_bound * float(cfg.dt) * int(cfg.horizon)
    vehicle_radius = 0.5 * math.hypot(float(cfg.vehicle_length), float(cfg.vehicle_width))
    reach = travel + vehicle_radius + float(cfg.hard_collision_clearance)
    reach2 = reach * reach
    for section in range(4):
        minx = float(bank.sector_mins[section, 0])
        miny = float(bank.sector_mins[section, 1])
        maxx = float(bank.sector_maxs[section, 0])
        maxy = float(bank.sector_maxs[section, 1])
        if not math.isfinite(minx):
            continue
        dx = 0.0 if minx <= px <= maxx else min(abs(px - minx), abs(px - maxx))
        dy = 0.0 if miny <= py <= maxy else min(abs(py - miny), abs(py - maxy))
        if dx * dx + dy * dy <= reach2:
            mask |= 1 << section
    return mask if mask else 15



def _build_prediction_collision_cache(
    bank: CollisionSectorBank,
    dynamic_centers: np.ndarray,
    dynamic_radii: np.ndarray,
) -> CollisionSectorBank:
    if dynamic_radii.size == 0:
        return bank
    centers: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    for mask in range(16):
        c, r = _concat_circle_arrays([
            (bank.mask_centers[mask], bank.mask_radii[mask]),
            (dynamic_centers, dynamic_radii),
        ])
        centers.append(c)
        radii.append(r)
    return CollisionSectorBank(tuple(centers), tuple(radii), bank.sector_mins, bank.sector_maxs)

def _inner_barrier_filter_circle_arrays(inner_straight_barriers: list[object]) -> tuple[np.ndarray, np.ndarray]:
    circles = controller_core.obstacle_bounding_circles(list(inner_straight_barriers))
    centers, radii = ackermann.pack_obstacle_circles(circles)
    return (np.ascontiguousarray(np.asarray(centers, dtype=np.float64)), np.ascontiguousarray(np.asarray(radii, dtype=np.float64)))

def _canonical_controller_scene() -> tuple[object, list[object], np.ndarray, np.ndarray]:
    scene = controller_core.build_default_scene()
    start = np.asarray(scene.start, dtype=np.float64)[:2]
    goal = np.asarray(scene.goal, dtype=np.float64)[:2]
    source_obstacles = list(scene.obstacles)
    if not np.allclose(start, CONTROLLER_START) or not np.allclose(goal, CONTROLLER_GOAL):
        raise RuntimeError(f'Expected controller benchmark start/goal {CONTROLLER_START.tolist()}->{CONTROLLER_GOAL.tolist()}, got {start.tolist()}->{goal.tolist()}.')
    if not math.isclose(float(scene.scale), 4.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f'Expected the standard viewer planner scale 4.0, got {scene.scale!r}.')
    low = np.asarray(scene.bounds_xy[0], dtype=np.float64)
    high = np.asarray(scene.bounds_xy[1], dtype=np.float64)
    if not (np.allclose(low, [0.0, 0.0]) and np.allclose(high, [10.0, 10.0])):
        raise RuntimeError(f'Expected canonical planner bounds [0,0]-[10,10], got {low.tolist()}-{high.tolist()}.')
    PolyObstacle, *_ = controller_core._planner_symbols()
    obstacles = [PolyObstacle(_scale_polygon_about_centroid(_poly_vertices(obs), OBSTACLE_LINEAR_SCALE)) for obs in source_obstacles]
    scene = replace(scene, obstacles=tuple(obstacles))
    return (scene, obstacles, start, goal)

def build_racing_obstacles() -> tuple[list[object], list[object], list[object]]:
    source_scene, source_obstacles, source_start, source_goal = _canonical_controller_scene()
    del source_scene
    PolyObstacle, *_ = controller_core._planner_symbols()
    lower = [PolyObstacle(_rigid_transform_points(_poly_vertices(obs), source_start, source_goal, FISH_P2, FISH_P3)) for obs in source_obstacles]
    upper = [PolyObstacle(_rigid_transform_points(_poly_vertices(obs), source_start, source_goal, FISH_P4, FISH_P1)) for obs in source_obstacles]
    fixed_barriers, _ = _build_fixed_barriers()
    return (lower, upper, lower + upper + fixed_barriers)

def _fixed_turn(start_s: float) -> tuple[np.ndarray, np.ndarray]:
    points = sample_centerline(start_s, CHICANE_LENGTH, TURN_POINTS)
    return (points, racing_prior_covariance(points))

def _force_path_endpoints(path: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    out = np.asarray(path, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[0] < 2 or out.shape[1] < 2:
        raise ValueError('Planner path must contain at least two planar points.')
    out = np.ascontiguousarray(out[:, :2])
    out[0] = np.asarray(start, dtype=np.float64)
    out[-1] = np.asarray(goal, dtype=np.float64)
    return out

def _concat_pieces(pieces: list[np.ndarray]) -> np.ndarray:
    merged: list[np.ndarray] = []
    for piece in pieces:
        p = np.asarray(piece, dtype=np.float64)
        if not merged:
            merged.append(p)
        elif np.linalg.norm(merged[-1][-1] - p[0]) <= 1e-08:
            merged.append(p[1:])
        else:
            merged.append(p)
    return np.ascontiguousarray(np.vstack(merged), dtype=np.float64)

def _concat_covariances(paths: list[np.ndarray], covariances: list[np.ndarray]) -> np.ndarray:
    blocks: list[np.ndarray] = []
    first = True
    previous_end: Optional[np.ndarray] = None
    for path, cov in zip(paths, covariances):
        p = np.asarray(path, dtype=np.float64)
        c = np.asarray(cov, dtype=np.float64)
        if c.shape != (len(p), 2, 2):
            raise ValueError('Covariance length must match its path piece.')
        drop = not first and previous_end is not None and (np.linalg.norm(previous_end - p[0]) <= 1e-08)
        blocks.append(c[1:] if drop else c)
        previous_end = p[-1]
        first = False
    return np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype=np.float64)

def _straight_connector(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Sample one deterministic straight connector for an added end section."""
    a = np.asarray(start, dtype=np.float64)[:2]
    b = np.asarray(goal, dtype=np.float64)[:2]
    distance = float(np.linalg.norm(b - a))
    if distance <= 1e-12:
        return np.ascontiguousarray(a[None, :], dtype=np.float64)
    spacing = max(float(STRAIGHT_EXTENSION_POINT_SPACING), 1e-06)
    count = max(2, int(math.ceil(distance / spacing)) + 1)
    return np.ascontiguousarray(np.linspace(a, b, count), dtype=np.float64)

def _extend_mode_with_straight_ends(
    mode: controller_core.MPPIHomotopyMode,
    full_start: np.ndarray,
    fish_start: np.ndarray,
    fish_goal: np.ndarray,
    full_goal: np.ndarray,
) -> controller_core.MPPIHomotopyMode:
    """Add equal straight tails while leaving the middle section untouched."""
    core = _force_path_endpoints(mode.mean_path, fish_start, fish_goal)
    core_cov = np.asarray(mode.cov_blocks, dtype=np.float64)
    if core_cov.shape != (len(core), 2, 2):
        raise ValueError('Covariance does not match the transformed mean path.')

    entry = _straight_connector(full_start, fish_start)
    exit_ = _straight_connector(fish_goal, full_goal)
    entry_cov = np.repeat(core_cov[0:1], len(entry), axis=0)
    exit_cov = np.repeat(core_cov[-1:], len(exit_), axis=0)
    mean = _concat_pieces([entry, core, exit_])
    cov = _concat_covariances([entry, core, exit_], [entry_cov, core_cov, exit_cov])

    samples: list[np.ndarray] = []
    for raw in list(mode.sample_paths or []):
        sample_core = _force_path_endpoints(raw, fish_start, fish_goal)
        samples.append(_concat_pieces([entry, sample_core, exit_]))

    return controller_core.prepare_mode_prior_cache(
        controller_core.MPPIHomotopyMode(
            signature=tuple(mode.signature),
            probability=float(mode.probability),
            mean_path=mean,
            cov_blocks=cov,
            sample_paths=samples,
        )
    )

def _duplicate_loop(path: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    repeated_path = np.vstack((path, path[1:]))
    repeated_cov = np.concatenate((cov, cov[1:]), axis=0)
    return (np.ascontiguousarray(repeated_path), np.ascontiguousarray(repeated_cov))

def _joint_empirical_paths(bottom_mode: controller_core.MPPIHomotopyMode, top_mode: controller_core.MPPIHomotopyMode, right_turn: np.ndarray, left_turn: np.ndarray) -> list[np.ndarray]:
    bottom_samples = list(bottom_mode.sample_paths or [])
    top_samples = list(top_mode.sample_paths or [])
    if not bottom_samples or not top_samples:
        return []
    out: list[np.ndarray] = []
    for i, bottom_raw in enumerate(bottom_samples):
        for j, top_raw in enumerate(top_samples):
            bottom = _force_path_endpoints(bottom_raw, P2, P3)
            top = _force_path_endpoints(top_raw, P4, P1)
            loop = _concat_pieces([bottom, right_turn, top, left_turn])
            repeated = np.vstack((loop, loop[1:]))
            out.append(np.ascontiguousarray(repeated, dtype=np.float64))
            if len(out) >= MAX_EMPIRICAL_PATHS_PER_JOINT_MODE:
                return out
    return out

def build_racing_prior_modes(seed: int) -> tuple[list[controller_core.MPPIHomotopyMode], list[object], list[object], list[object], list[object]]:
    """Run planner exactly as the standard viewer, then map priors to the chicane circuit."""
    source_scene, source_obstacles, source_start, source_goal = _canonical_controller_scene()
    canonical_bottom = controller_core.build_homotopy_modes(source_scene, source_obstacles, int(seed))
    canonical_top = controller_core.build_homotopy_modes(source_scene, source_obstacles, int(seed) + 1)
    if not canonical_bottom:
        raise RuntimeError('Canonical planner returned no modes for the P2->P3 straight.')
    if not canonical_top:
        raise RuntimeError('Canonical planner returned no modes for the P4->P1 straight.')
    bottom_core_modes = [_transform_planner_mode(mode, source_start, source_goal, FISH_P2, FISH_P3) for mode in canonical_bottom]
    top_core_modes = [_transform_planner_mode(mode, source_start, source_goal, FISH_P4, FISH_P1) for mode in canonical_top]
    bottom_modes = [_extend_mode_with_straight_ends(mode, P2, FISH_P2, FISH_P3, P3) for mode in bottom_core_modes]
    top_modes = [_extend_mode_with_straight_ends(mode, P4, FISH_P4, FISH_P1, P1) for mode in top_core_modes]
    PolyObstacle, *_ = controller_core._planner_symbols()
    lower_obstacles = [PolyObstacle(_rigid_transform_points(_poly_vertices(obs), source_start, source_goal, FISH_P2, FISH_P3)) for obs in source_obstacles]
    upper_obstacles = [PolyObstacle(_rigid_transform_points(_poly_vertices(obs), source_start, source_goal, FISH_P4, FISH_P1)) for obs in source_obstacles]
    fixed_barriers, inner_straight_barriers = _build_fixed_barriers()
    all_obstacles = lower_obstacles + upper_obstacles + fixed_barriers
    right_turn, right_cov = _fixed_turn(STRAIGHT_LENGTH)
    p1_s = 2.0 * STRAIGHT_LENGTH + CHICANE_LENGTH
    left_turn, left_cov = _fixed_turn(p1_s)
    joint: list[controller_core.MPPIHomotopyMode] = []
    for bi, bottom_mode in enumerate(bottom_modes):
        bottom = _force_path_endpoints(bottom_mode.mean_path, P2, P3)
        bottom_cov = np.asarray(bottom_mode.cov_blocks, dtype=np.float64)
        if bottom_cov.shape[0] != len(bottom):
            raise ValueError('Bottom covariance does not match its mean path.')
        for ti, top_mode in enumerate(top_modes):
            top = _force_path_endpoints(top_mode.mean_path, P4, P1)
            top_cov = np.asarray(top_mode.cov_blocks, dtype=np.float64)
            if top_cov.shape[0] != len(top):
                raise ValueError('Top covariance does not match its mean path.')
            pieces = [bottom, right_turn, top, left_turn]
            covs = [bottom_cov, right_cov, top_cov, left_cov]
            loop = _concat_pieces(pieces)
            loop_cov = _concat_covariances(pieces, covs)
            loop, loop_cov = _duplicate_loop(loop, loop_cov)
            samples = _joint_empirical_paths(bottom_mode, top_mode, right_turn, left_turn)
            signature = (9001, bi) + tuple(bottom_mode.signature) + (9002, ti) + tuple(top_mode.signature)
            joint.append(controller_core.prepare_mode_prior_cache(controller_core.MPPIHomotopyMode(signature=signature, probability=float(bottom_mode.probability) * float(top_mode.probability), mean_path=loop, cov_blocks=loop_cov, sample_paths=samples)))
    total = sum((max(0.0, float(mode.probability)) for mode in joint))
    if total <= 1e-15:
        total = float(len(joint))
        for mode in joint:
            mode.probability = 1.0 / total
    else:
        for mode in joint:
            mode.probability = float(mode.probability) / total
    joint.sort(key=lambda mode: mode.probability, reverse=True)
    return (joint, lower_obstacles, upper_obstacles, fixed_barriers, inner_straight_barriers)

def _pack_racing_prior_bank(prior_modes: list[controller_core.MPPIHomotopyMode]) -> PackedRacingPriorBank:
    if not prior_modes:
        raise ValueError('Cannot pack an empty racing prior bank.')
    cached = [controller_core.prepare_mode_prior_cache(mode) for mode in prior_modes]
    lengths = np.asarray([len(mode.mean_path) for mode in cached], dtype=np.int64)
    localization_lengths = np.asarray([(int(n) + 1) // 2 for n in lengths], dtype=np.int64)
    localization_unique_lengths = localization_lengths.copy()
    for m, mode in enumerate(cached):
        nloc = int(localization_lengths[m])
        if nloc > 2:
            first = np.asarray(mode.mean_path[0], dtype=np.float64)[:2]
            last = np.asarray(mode.mean_path[nloc - 1], dtype=np.float64)[:2]
            if float(np.dot(first - last, first - last)) <= 1e-14:
                localization_unique_lengths[m] = nloc - 1
    block_size = int(PRIOR_LOCALIZATION_BLOCK_SIZE)
    max_blocks = max(1, int(math.ceil(float(np.max(localization_unique_lengths)) / float(block_size))))
    localization_block_mins = np.full((len(cached), max_blocks, 2), np.inf, dtype=np.float64)
    localization_block_maxs = np.full((len(cached), max_blocks, 2), -np.inf, dtype=np.float64)
    localization_block_counts = np.zeros(len(cached), dtype=np.int64)
    max_len = int(np.max(lengths))
    mode_count = len(cached)
    mean_paths = np.zeros((mode_count, max_len, 2), dtype=np.float64)
    cov_blocks = np.zeros((mode_count, max_len, 2, 2), dtype=np.float64)
    arc_lengths = np.zeros((mode_count, max_len), dtype=np.float64)
    sample_mode_offsets = np.zeros(mode_count + 1, dtype=np.int64)
    sample_list: list[np.ndarray] = []
    for m, mode in enumerate(cached):
        n = int(lengths[m])
        mean_paths[m, :n] = np.asarray(mode.mean_path, dtype=np.float64)[:, :2]
        cov_blocks[m, :n] = np.asarray(mode.cov_blocks, dtype=np.float64)[:n]
        arc_lengths[m, :n] = np.asarray(mode.arc_length, dtype=np.float64)[:n]
        unique_n = int(localization_unique_lengths[m])
        block_count = max(1, int(math.ceil(float(unique_n) / float(block_size))))
        localization_block_counts[m] = block_count
        for block in range(block_count):
            a = block * block_size
            b = min(unique_n, a + block_size)
            segment = mean_paths[m, a:b]
            localization_block_mins[m, block] = np.min(segment, axis=0)
            localization_block_maxs[m, block] = np.max(segment, axis=0)
        for raw in list(mode.sample_paths or []):
            sample = np.ascontiguousarray(np.asarray(raw, dtype=np.float64)[:, :2])
            if len(sample) >= 2:
                sample_list.append(sample)
        sample_mode_offsets[m + 1] = len(sample_list)
    probabilities = np.asarray([max(0.0, float(mode.probability)) for mode in cached], dtype=np.float64)
    mass = float(np.sum(probabilities))
    if not math.isfinite(mass) or mass <= 1e-15:
        probabilities[:] = 1.0 / float(mode_count)
    else:
        probabilities /= mass
    if sample_list:
        sample_lengths = np.asarray([len(path) for path in sample_list], dtype=np.int64)
        max_sample_len = int(np.max(sample_lengths))
        sample_paths = np.zeros((len(sample_list), max_sample_len, 2), dtype=np.float64)
        sample_arc_lengths = np.zeros((len(sample_list), max_sample_len), dtype=np.float64)
        for i, path in enumerate(sample_list):
            n = int(sample_lengths[i])
            sample_paths[i, :n] = path
            if n > 1:
                delta = np.diff(path, axis=0)
                sample_arc_lengths[i, 1:n] = np.cumsum(np.linalg.norm(delta, axis=1))
    else:
        sample_paths = np.zeros((0, 0, 2), dtype=np.float64)
        sample_arc_lengths = np.zeros((0, 0), dtype=np.float64)
        sample_lengths = np.zeros(0, dtype=np.int64)
    return PackedRacingPriorBank(
        modes=cached,
        mean_paths=np.ascontiguousarray(mean_paths),
        cov_blocks=np.ascontiguousarray(cov_blocks),
        arc_lengths=np.ascontiguousarray(arc_lengths),
        lengths=np.ascontiguousarray(lengths),
        localization_lengths=np.ascontiguousarray(localization_lengths),
        localization_unique_lengths=np.ascontiguousarray(localization_unique_lengths),
        localization_block_mins=np.ascontiguousarray(localization_block_mins),
        localization_block_maxs=np.ascontiguousarray(localization_block_maxs),
        localization_block_counts=np.ascontiguousarray(localization_block_counts),
        probabilities=np.ascontiguousarray(probabilities),
        sample_paths=np.ascontiguousarray(sample_paths),
        sample_arc_lengths=np.ascontiguousarray(sample_arc_lengths),
        sample_lengths=np.ascontiguousarray(sample_lengths),
        sample_mode_offsets=np.ascontiguousarray(sample_mode_offsets),
    )

def _obstacle_center_xy(obstacle: object) -> np.ndarray:
    vertices = _poly_vertices(obstacle)
    return np.asarray(np.mean(vertices, axis=0), dtype=np.float64)[:2]

def _concat_circle_arrays(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    valid = [(np.asarray(c, dtype=np.float64), np.asarray(r, dtype=np.float64)) for c, r in parts if len(r)]
    if not valid:
        return (np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64))
    return (np.ascontiguousarray(np.concatenate([c for c, _ in valid], axis=0), dtype=np.float64), np.ascontiguousarray(np.concatenate([r for _, r in valid], axis=0), dtype=np.float64))

def _dynamic_wall_candidates(obstacles: list[object]) -> list[DynamicWallCandidate]:
    """Prebuild all center-to-center walls, shortest first, with collision circles."""
    centers = [_obstacle_center_xy(obstacle) for obstacle in obstacles]
    out: list[DynamicWallCandidate] = []
    for i, j in itertools.combinations(range(len(obstacles)), 2):
        c0 = np.ascontiguousarray(centers[int(i)], dtype=np.float64)
        c1 = np.ascontiguousarray(centers[int(j)], dtype=np.float64)
        wall = controller_core.make_wall_between_points(c0, c1, width=DYNAMIC_WALL_WIDTH, extension=0.0)
        circles = controller_core.obstacle_bounding_circles([wall])
        circle_centers, circle_radii = ackermann.pack_obstacle_circles(circles)
        out.append(DynamicWallCandidate(pair=(int(i), int(j)), wall=wall, vertices=_poly_vertices(wall), length=float(np.linalg.norm(c1 - c0)), p0=c0, p1=c1, circle_centers=np.ascontiguousarray(circle_centers, dtype=np.float64), circle_radii=np.ascontiguousarray(circle_radii, dtype=np.float64)))
    out.sort(key=lambda item: item.length)
    return out

def _dynamic_candidate_segment_arrays(candidates: list[DynamicWallCandidate]) -> tuple[np.ndarray, np.ndarray]:
    if not candidates:
        return (np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64))
    return (np.ascontiguousarray(np.asarray([item.p0 for item in candidates], dtype=np.float64)), np.ascontiguousarray(np.asarray([item.p1 for item in candidates], dtype=np.float64)))

def _dynamic_circle_arrays_from_walls(active: list[ActiveDynamicWall]) -> tuple[np.ndarray, np.ndarray]:
    return _concat_circle_arrays([(item.candidate.circle_centers, item.candidate.circle_radii) for item in active])

def _dynamic_walls_leave_feasible_prior(prior_bank: PackedRacingPriorBank, base_feasible_mask: np.ndarray, circle_centers: np.ndarray, circle_radii: np.ndarray, cfg: ackermann.MPPIConfig) -> bool:
    """Test dynamic walls only, reusing the fixed-barrier feasibility mask."""
    base = np.ascontiguousarray(np.asarray(base_feasible_mask, dtype=np.bool_))
    if not np.any(base):
        return False
    if circle_radii.size == 0:
        return True
    dynamic_mask = _prior_feasible_mask_nb(prior_bank.mean_paths, prior_bank.localization_lengths, np.ascontiguousarray(circle_centers, dtype=np.float64), np.ascontiguousarray(circle_radii, dtype=np.float64), float(cfg.vehicle_length), float(cfg.vehicle_width), float(cfg.hard_collision_clearance), int(cfg.mode_blocking_substeps))
    return bool(np.any(np.logical_and(base, dynamic_mask)))

def _select_shortest_cutting_wall(candidates: list[DynamicWallCandidate], candidate_p0s: np.ndarray, candidate_p1s: np.ndarray, taken_path: np.ndarray, prior_bank: PackedRacingPriorBank, base_feasible_mask: np.ndarray, cfg: ackermann.MPPIConfig, reserved_circle_centers: np.ndarray, reserved_circle_radii: np.ndarray, reserved_candidate_ids: set[int]) -> Optional[DynamicWallCandidate]:
    """Return the shortest wall crossing the path while preserving a feasible prior.

    Candidates are pre-sorted by center-to-center length.  The path-crossing test is
    one parallel Numba pass; only crossing candidates reach the more expensive prior
    feasibility test.  If the geometrically shortest crossing wall would close every
    prior, the next-shortest crossing wall is tried.
    """
    if not candidates or np.asarray(taken_path).shape[0] < 2:
        return None
    path_xy = np.ascontiguousarray(np.asarray(taken_path, dtype=np.float64)[:, :2])
    crossing = _candidate_path_intersection_mask_nb(path_xy, np.ascontiguousarray(candidate_p0s, dtype=np.float64), np.ascontiguousarray(candidate_p1s, dtype=np.float64))
    for candidate_index in np.flatnonzero(crossing):
        candidate = candidates[int(candidate_index)]
        if id(candidate) in reserved_candidate_ids:
            continue
        combined_centers, combined_radii = _concat_circle_arrays([(reserved_circle_centers, reserved_circle_radii), (candidate.circle_centers, candidate.circle_radii)])
        if _dynamic_walls_leave_feasible_prior(prior_bank, base_feasible_mask, combined_centers, combined_radii, cfg):
            return candidate
    return None

def _active_prior_selection(prior_bank: PackedRacingPriorBank, fixed_feasible_mask: np.ndarray, dynamic_circle_centers: np.ndarray, dynamic_circle_radii: np.ndarray, cfg: ackermann.MPPIConfig, max_prior: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    feasible = np.ascontiguousarray(np.asarray(fixed_feasible_mask, dtype=np.bool_)).copy()
    if dynamic_circle_radii.size:
        feasible &= _prior_feasible_mask_nb(prior_bank.mean_paths, prior_bank.localization_lengths, dynamic_circle_centers, dynamic_circle_radii, float(cfg.vehicle_length), float(cfg.vehicle_width), float(cfg.hard_collision_clearance), int(cfg.mode_blocking_substeps))
    active_indices = np.ascontiguousarray(np.flatnonzero(feasible), dtype=np.int64)
    if active_indices.size == 0:
        raise RuntimeError('No complete prior remains feasible with the active barriers.')
    if max_prior is not None:
        limit = int(max_prior)
        if limit < 1:
            raise ValueError('max_prior must be at least 1 or None.')
        if active_indices.size > limit:
            feasible_probabilities = np.asarray(prior_bank.probabilities[active_indices], dtype=np.float64)
            order = np.argsort(-feasible_probabilities, kind='stable')[:limit]
            active_indices = np.ascontiguousarray(active_indices[order], dtype=np.int64)
    probabilities = np.asarray(prior_bank.probabilities[active_indices], dtype=np.float64)
    mass = float(np.sum(probabilities))
    if mass <= 1e-15:
        probabilities = np.full(len(probabilities), 1.0 / len(probabilities), dtype=np.float64)
    else:
        probabilities /= mass
    return (active_indices, np.ascontiguousarray(probabilities))

def _local_racing_ranges(
    state: np.ndarray,
    current_s: float,
    prior_bank: PackedRacingPriorBank,
    cfg: ackermann.MPPIConfig,
    active_indices: np.ndarray,
    localization_cursors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = controller_core.prior_preview_step_distance(cfg)
    return _localize_prior_ranges_nb(
        prior_bank.mean_paths,
        prior_bank.arc_lengths,
        prior_bank.lengths,
        prior_bank.localization_lengths,
        prior_bank.localization_unique_lengths,
        prior_bank.localization_block_mins,
        prior_bank.localization_block_maxs,
        prior_bank.localization_block_counts,
        np.ascontiguousarray(active_indices, dtype=np.int64),
        np.ascontiguousarray(localization_cursors, dtype=np.int64),
        float(state[0]),
        float(state[1]),
        float(current_s),
        int(cfg.horizon),
        float(ds),
        24,
        96,
        int(PRIOR_LOCALIZATION_BLOCK_SIZE),
    )


def _local_racing_batch(
    state: np.ndarray,
    current_s: float,
    prior_bank: PackedRacingPriorBank,
    cfg: ackermann.MPPIConfig,
    active_indices: np.ndarray,
    localization_cursors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts, ends, updated = _local_racing_ranges(
        state, current_s, prior_bank, cfg, active_indices, localization_cursors
    )
    refs, covs, arcs, lengths = _pack_local_prior_windows_nb(
        prior_bank.mean_paths,
        prior_bank.cov_blocks,
        prior_bank.arc_lengths,
        np.ascontiguousarray(active_indices, dtype=np.int64),
        starts,
        ends,
    )
    return refs, covs, arcs, lengths, updated


def _is_fixed_turn_covariance_sample(mean: np.ndarray, covariance: np.ndarray) -> bool:
    turn_variance = TURN_PRIOR_SIGMA * TURN_PRIOR_SIGMA
    if not (
        abs(float(covariance[0, 0]) - turn_variance) <= 1e-10
        and abs(float(covariance[1, 1]) - turn_variance) <= 1e-10
        and abs(float(covariance[0, 1])) <= 1e-10
        and abs(float(covariance[1, 0])) <= 1e-10
    ):
        return False
    s_value, distance_sq = _project_centerline_nb(float(mean[0]), float(mean[1]))
    if distance_sq > 1e-10:
        return False
    right_start = STRAIGHT_LENGTH
    right_end = right_start + CHICANE_LENGTH
    left_start = 2.0 * STRAIGHT_LENGTH + CHICANE_LENGTH
    return right_start - 1e-10 <= s_value <= right_end + 1e-10 or left_start - 1e-10 <= s_value < TRACK_LENGTH


def _sparse_covariance_mode_geometry(
    prior_bank: PackedRacingPriorBank,
    mode_index: int,
    spacing: float = PRIOR_COV_VIS_SPACING,
    points_per_ellipse: int = PRIOR_COV_VIS_POINTS,
) -> tuple[list[np.ndarray], np.ndarray]:
    m = int(mode_index)
    n = min(int(prior_bank.localization_lengths[m]), int(prior_bank.lengths[m]))
    if n <= 0:
        return [], np.zeros(0, dtype=np.int64)
    path = np.asarray(prior_bank.mean_paths[m, :n], dtype=np.float64)
    covs = np.asarray(prior_bank.cov_blocks[m, :n], dtype=np.float64)
    arcs = np.asarray(prior_bank.arc_lengths[m, :n], dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * math.pi, max(10, int(points_per_ellipse)), endpoint=True)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    polygons: list[np.ndarray] = []
    indices: list[int] = []
    last_arc = -1e300
    minimum_spacing = max(float(spacing), 1e-6)
    for i in range(n):
        if i == 0 or i == n - 1:
            continue
        covariance = 0.5 * (covs[i] + covs[i].T)
        if not np.all(np.isfinite(covariance)):
            continue
        if _is_fixed_turn_covariance_sample(path[i], covariance):
            continue
        if float(arcs[i]) - last_arc < minimum_spacing:
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        if float(np.max(eigenvalues)) <= 1e-14:
            continue
        transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        polygons.append(np.asarray(path[i][None, :] + unit @ transform.T, dtype=np.float64))
        indices.append(i)
        last_arc = float(arcs[i])
    return polygons, np.ascontiguousarray(np.asarray(indices, dtype=np.int64))


def _prior_visual_strength(probability: float, peak_probability: float) -> float:
    peak = max(float(peak_probability), 1e-15)
    return math.sqrt(max(0.0, min(1.0, float(probability) / peak)))


def _prior_covariance_alpha(probability: float, peak_probability: float) -> float:
    return PRIOR_COV_ALPHA_MIN + PRIOR_COV_ALPHA_SPAN * _prior_visual_strength(probability, peak_probability)


def _prior_mean_alpha(probability: float, peak_probability: float) -> float:
    return PRIOR_MEAN_ALPHA_MIN + PRIOR_MEAN_ALPHA_SPAN * _prior_visual_strength(probability, peak_probability)


def _prior_mean_linewidth(probability: float, peak_probability: float) -> float:
    return PRIOR_MEAN_LINEWIDTH_MIN + PRIOR_MEAN_LINEWIDTH_SPAN * _prior_visual_strength(probability, peak_probability)


def _covariance_sample_visibility(
    sample_indices: np.ndarray,
    start: int,
    end: int,
    physical_length: int,
) -> np.ndarray:
    ids = np.asarray(sample_indices, dtype=np.int64)
    n = int(physical_length)
    if ids.size == 0 or n <= 0:
        return np.zeros(ids.size, dtype=np.bool_)
    a = max(0, int(start))
    b = max(a, int(end))
    if b < n:
        return np.logical_and(ids >= a, ids <= b)
    wrapped_end = min(n - 1, b - (n - 1))
    return np.logical_or(ids >= a, ids <= wrapped_end)


def _probability_rollout_counts(total: int, probabilities: np.ndarray) -> np.ndarray:
    total = max(1, int(total))
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.maximum(p, 0.0)
    if p.size == 0:
        return np.zeros(0, dtype=np.int64)
    if float(np.sum(p)) <= 1e-15:
        p[:] = 1.0 / len(p)
    else:
        p /= float(np.sum(p))
    exact = p * total
    counts = np.floor(exact).astype(np.int64)
    remainder = total - int(np.sum(counts))
    if remainder > 0:
        order = np.argsort(-(exact - counts), kind='stable')
        counts[order[:remainder]] += 1
    if total >= len(p):
        zero_ids = np.flatnonzero(counts == 0)
        for zero_id in zero_ids:
            donors = np.flatnonzero(counts > 1)
            if donors.size == 0:
                break
            surplus = counts[donors].astype(np.float64) - exact[donors]
            donor = int(donors[int(np.argmax(surplus))])
            counts[donor] -= 1
            counts[int(zero_id)] += 1
    return np.ascontiguousarray(counts, dtype=np.int64)


def _sample_gaussian_raw(
    state: np.ndarray,
    ref: np.ndarray,
    cov: np.ndarray,
    arc: np.ndarray,
    length: int,
    nominal: np.ndarray,
    control_positions: np.ndarray,
    ilqr_positions: np.ndarray,
    count: int,
    cfg: object,
    model: object,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(length)
    _, _, variance = controller_core._prior_second_moment_about_ilqr_nb(
        np.ascontiguousarray(ref[:n]),
        np.ascontiguousarray(cov[:n]),
        np.ascontiguousarray(arc[:n]),
        np.ascontiguousarray(control_positions),
        np.ascontiguousarray(ilqr_positions),
    )
    noise = controller_core.make_temporally_correlated_noise(
        model, count, int(cfg.horizon), cfg, rng
    )
    sigma_ref = max(float(cfg.sigma_ref), 1e-9)
    noise *= (np.sqrt(np.maximum(variance, 0.0)) / sigma_ref)[None, :, None]
    noise += np.asarray(nominal, dtype=np.float64)[None, :, :]
    return model.clip_control_batch(noise, cfg)


def _sample_spg_raw(
    state: np.ndarray,
    ref: np.ndarray,
    cov: np.ndarray,
    arc: np.ndarray,
    length: int,
    nominal: np.ndarray,
    control_positions: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    ilqr_positions: np.ndarray,
    count: int,
    cfg: object,
    model: object,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(length)
    corrected, _, _ = controller_core._prior_second_moment_about_ilqr_nb(
        np.ascontiguousarray(ref[:n]),
        np.ascontiguousarray(cov[:n]),
        np.ascontiguousarray(arc[:n]),
        np.ascontiguousarray(control_positions),
        np.ascontiguousarray(ilqr_positions),
    )
    projected = model.project_control_covariances_from_jacobians(A, B, corrected, cfg)
    standard_noise = controller_core.make_temporally_correlated_noise(
        model,
        count,
        int(cfg.horizon),
        cfg,
        rng,
        scale_override=np.ones(2, dtype=np.float64),
    )
    noise = controller_core._apply_projected_covariance_nb(
        standard_noise, np.ascontiguousarray(projected, dtype=np.float64)
    )
    noise += np.asarray(nominal, dtype=np.float64)[None, :, :]
    return model.clip_control_batch(noise, cfg)


def _active_sample_ids(
    prior_bank: PackedRacingPriorBank,
    active_indices: np.ndarray,
    max_count: int,
) -> np.ndarray:
    selected: list[int] = []
    limit = max(1, int(max_count))
    for global_index in np.asarray(active_indices, dtype=np.int64):
        m = int(global_index)
        start = int(prior_bank.sample_mode_offsets[m])
        end = int(prior_bank.sample_mode_offsets[m + 1])
        for sample_id in range(start, end):
            selected.append(sample_id)
            if len(selected) >= limit:
                return np.ascontiguousarray(np.asarray(selected, dtype=np.int64))
    return np.ascontiguousarray(np.asarray(selected, dtype=np.int64))


def _planner_control_bank(
    state: np.ndarray,
    current_s: float,
    prior_bank: PackedRacingPriorBank,
    active_indices: np.ndarray,
    cfg: object,
    model: object,
    fallback_refs: np.ndarray,
    fallback_lengths: np.ndarray,
) -> np.ndarray:
    sample_ids = _active_sample_ids(prior_bank, active_indices, int(cfg.num_rollouts))
    if sample_ids.size:
        refs, lengths = _localize_sample_paths_nb(
            prior_bank.sample_paths,
            prior_bank.sample_arc_lengths,
            prior_bank.sample_lengths,
            sample_ids,
            float(state[0]),
            float(state[1]),
            float(current_s),
            int(cfg.horizon),
            float(controller_core.prior_preview_step_distance(cfg)),
        )
    else:
        refs = np.ascontiguousarray(fallback_refs)
        lengths = np.ascontiguousarray(fallback_lengths)
    fast_batch = getattr(model, 'batch_nominal_controls', None)
    if fast_batch is not None:
        controls = fast_batch(
            state,
            np.ascontiguousarray(refs),
            np.ascontiguousarray(lengths),
            cfg,
            None,
        )
    else:
        controls, _, _, _, _ = model.batch_nominal_solutions(
            state,
            np.ascontiguousarray(refs),
            np.ascontiguousarray(lengths),
            cfg,
            None,
        )
    return model.clip_control_batch(np.asarray(controls, dtype=np.float64), cfg)

def initial_race_state(model_name: str = 'ackermann') -> np.ndarray:
    point, tangent = centerline_point_tangent(START_S)
    heading = math.atan2(float(tangent[1]), float(tangent[0]))
    if model_name == 'four_wheel':
        return np.asarray([point[0], point[1], heading, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray([point[0], point[1], heading, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

def make_racing_config(num_rollouts: int, lbps_delta: float, *, horizon: int, temporal_noise_smoothing: float, sigma0_scale: float, v_max: float, hard_collision_clearance: float, model_name: str = 'ackermann') -> object:
    model = controller_core.resolve_vehicle_model(model_name)
    cfg = model.MPPIConfig(num_rollouts=int(num_rollouts))
    cfg.adaptive_temperature_lbps = True
    cfg.lbps_delta = float(lbps_delta)
    cfg.horizon = int(horizon)
    cfg.temporal_noise_smoothing = float(temporal_noise_smoothing)
    sigma0_std_scale = math.sqrt(float(sigma0_scale))
    cfg.noise_accel *= sigma0_std_scale
    cfg.noise_steering_rate *= sigma0_std_scale
    cfg.v_max = float(v_max)
    cfg.hard_collision_clearance = float(hard_collision_clearance)
    return cfg

def racing_controller_step(
    variant_value: str,
    state: np.ndarray,
    current_s: float,
    cfg: object,
    model: object,
    rng: np.random.Generator,
    prior_bank: PackedRacingPriorBank,
    active_indices: np.ndarray,
    active_probabilities: np.ndarray,
    circle_centers: np.ndarray,
    circle_radii: np.ndarray,
    localization_cursors: np.ndarray,
    record_predictions: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        variant = controller_core.ControllerVariant(variant_value)
    except ValueError as exc:
        raise ValueError(f'Unsupported racing controller variant: {variant_value}') from exc
    active_indices = np.ascontiguousarray(np.asarray(active_indices, dtype=np.int64))
    probabilities = np.ascontiguousarray(np.asarray(active_probabilities, dtype=np.float64))
    if active_indices.size == 0:
        raise RuntimeError('No racing prior mode is available.')
    updated_cursors = np.ascontiguousarray(np.asarray(localization_cursors, dtype=np.int64)).copy()
    prior_variant = variant in {
        controller_core.ControllerVariant.CORRIDOR_PRIOR_MPPI,
        controller_core.ControllerVariant.GAUSSIAN_PRIOR_MPPI,
        controller_core.ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI,
    }

    if prior_variant:
        counts = _probability_rollout_counts(int(cfg.num_rollouts), probabilities)
        refs, covs, arcs, local_lengths, updated_cursors = _local_racing_batch(
            state, current_s, prior_bank, cfg, active_indices, updated_cursors
        )
        if variant == controller_core.ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI:
            nominal_controls, control_positions, As, Bs, ilqr_positions = model.batch_nominal_solutions(
                state,
                np.ascontiguousarray(refs),
                np.ascontiguousarray(local_lengths),
                cfg,
                np.ascontiguousarray(covs),
            )
        else:
            fast_batch = getattr(model, 'batch_nominal_controls_and_trajectories', None)
            if fast_batch is not None:
                nominal_controls, control_positions, ilqr_positions = fast_batch(
                    state,
                    np.ascontiguousarray(refs),
                    np.ascontiguousarray(local_lengths),
                    cfg,
                    np.ascontiguousarray(covs),
                )
                As = None
                Bs = None
            else:
                nominal_controls, control_positions, As, Bs, ilqr_positions = model.batch_nominal_solutions(
                    state,
                    np.ascontiguousarray(refs),
                    np.ascontiguousarray(local_lengths),
                    cfg,
                    np.ascontiguousarray(covs),
                )
        offsets = np.zeros(len(active_indices) + 1, dtype=np.int64)
        for i, count in enumerate(counts):
            offsets[i + 1] = offsets[i] + int(count)
        controls = np.empty((int(offsets[-1]), int(cfg.horizon), 2), dtype=np.float64)
        for i in range(len(active_indices)):
            count = int(counts[i])
            nominal_i = np.ascontiguousarray(nominal_controls[i])
            if variant == controller_core.ControllerVariant.CORRIDOR_PRIOR_MPPI:
                batch = controller_core.sample_controls_around_nominal(
                    model, nominal_i, count, cfg, rng
                )
            elif variant == controller_core.ControllerVariant.GAUSSIAN_PRIOR_MPPI:
                batch = _sample_gaussian_raw(
                    state,
                    refs[i],
                    covs[i],
                    arcs[i],
                    int(local_lengths[i]),
                    nominal_i,
                    control_positions[i],
                    ilqr_positions[i],
                    count,
                    cfg,
                    model,
                    rng,
                )
            else:
                batch = _sample_spg_raw(
                    state,
                    refs[i],
                    covs[i],
                    arcs[i],
                    int(local_lengths[i]),
                    nominal_i,
                    control_positions[i],
                    As[i],
                    Bs[i],
                    ilqr_positions[i],
                    count,
                    cfg,
                    model,
                    rng,
                )
            if count > 0:
                batch[0] = nominal_i
            controls[offsets[i]:offsets[i + 1]] = batch
        costs, collisions, _ = _evaluate_control_batch(
            state, controls, current_s, cfg, model, circle_centers, circle_radii
        )
        temperature_info = controller_core.resolve_mppi_temperature(costs, cfg)
        temperature = float(temperature_info[0])
        ess = float(temperature_info[2])
        candidate = controller_core.mppi_weighted_control_sequence(
            model, costs, controls, cfg, temperature=temperature
        )
        nominal = np.tensordot(probabilities, nominal_controls, axes=(0, 0))
        nominal = model.clip_control_batch(nominal[None, :, :], cfg)[0]
        finite_count = int(np.count_nonzero(np.isfinite(costs)))
        collision_rollouts = int(np.count_nonzero(collisions))
    else:
        primary_position = int(np.argmax(probabilities))
        primary_index = int(active_indices[primary_position])
        if variant == controller_core.ControllerVariant.CONTROL_BANK_MPPI:
            local_active_refs, local_active_covs, _, local_active_lengths, updated_cursors = _local_racing_batch(
                state, current_s, prior_bank, cfg, active_indices, updated_cursors
            )
            primary_local_position = primary_position
            primary_ref = np.ascontiguousarray(local_active_refs[primary_local_position, :int(local_active_lengths[primary_local_position])])
            primary_cov = np.ascontiguousarray(local_active_covs[primary_local_position, :int(local_active_lengths[primary_local_position])])
            nominal = np.asarray(
                model.nominal_controls_to_track_path(state, primary_ref, cfg, primary_cov),
                dtype=np.float64,
            )
            bank_controls = _planner_control_bank(
                state,
                current_s,
                prior_bank,
                active_indices,
                cfg,
                model,
                local_active_refs,
                local_active_lengths,
            )
            costs, collisions, _ = _evaluate_control_batch(
                state, bank_controls, current_s, cfg, model, circle_centers, circle_radii
            )
            finite_ids = np.flatnonzero(np.isfinite(costs))
            if finite_ids.size:
                best = int(finite_ids[np.argmin(costs[finite_ids])])
                candidate = np.asarray(bank_controls[best], dtype=np.float64).copy()
            else:
                candidate = nominal.copy()
            temperature = float('nan')
            finite_count = int(finite_ids.size)
            ess = 1.0 if finite_count else 0.0
            collision_rollouts = int(np.count_nonzero(collisions))
        else:
            refs, covs, _, local_lengths, updated_cursors = _local_racing_batch(
                state,
                current_s,
                prior_bank,
                cfg,
                np.asarray([primary_index], dtype=np.int64),
                updated_cursors,
            )
            n = int(local_lengths[0])
            primary_ref = np.ascontiguousarray(refs[0, :n])
            primary_cov = np.ascontiguousarray(covs[0, :n])
            if variant == controller_core.ControllerVariant.STANDARD_MPPI:
                nominal = np.asarray(
                    model.nominal_controls_to_track_path(state, primary_ref, cfg),
                    dtype=np.float64,
                )
                controls = controller_core.sample_controls_around_nominal(
                    model, nominal, int(cfg.num_rollouts), cfg, rng
                )
                controls[0] = nominal
                costs, collisions, _ = _evaluate_control_batch(
                    state, controls, current_s, cfg, model, circle_centers, circle_radii
                )
                temperature_info = controller_core.resolve_mppi_temperature(costs, cfg)
                temperature = float(temperature_info[0])
                ess = float(temperature_info[2])
                candidate = controller_core.mppi_weighted_control_sequence(
                    model, costs, controls, cfg, temperature=temperature
                )
                finite_count = int(np.count_nonzero(np.isfinite(costs)))
                collision_rollouts = int(np.count_nonzero(collisions))
            elif variant == controller_core.ControllerVariant.PLANNER_ILQR:
                nominal = np.asarray(
                    model.nominal_controls_to_track_path(state, primary_ref, cfg, primary_cov),
                    dtype=np.float64,
                )
                candidate = nominal.copy()
                costs, collisions, _ = _evaluate_control_batch(
                    state, candidate[None, :, :], current_s, cfg, model, circle_centers, circle_radii
                )
                temperature = float('nan')
                finite_count = int(np.count_nonzero(np.isfinite(costs)))
                ess = 1.0 if finite_count else 0.0
                collision_rollouts = int(np.count_nonzero(collisions))
            else:
                raise ValueError(f'Unsupported racing controller variant: {variant_value}')

    nominal = model.clip_control_batch(np.asarray(nominal, dtype=np.float64)[None, :, :], cfg)[0]
    candidate = np.ascontiguousarray(np.asarray(candidate, dtype=np.float64))
    nominal_states = (
        np.asarray(model.rollout_single(state, nominal, cfg), dtype=np.float64)
        if record_predictions else np.zeros((0, int(model.STATE_DIM)), dtype=np.float64)
    )
    candidate_states = (
        np.asarray(model.rollout_single(state, candidate, cfg), dtype=np.float64)
        if record_predictions else np.zeros((0, int(model.STATE_DIM)), dtype=np.float64)
    )
    return (
        candidate[0].copy(),
        {
            'optimal_traj': candidate_states,
            'nominal_traj': nominal_states,
            'temperature': temperature,
            'ess': ess,
            'finite_rollouts': finite_count,
            'collision_rollouts': collision_rollouts,
            'localization_cursors': updated_cursors,
        },
    )

def _warm_racing_kernels(
    variant_value: str,
    state: np.ndarray,
    current_s: float,
    cfg: object,
    model: object,
    prior_bank: PackedRacingPriorBank,
    active_indices: np.ndarray,
    active_probabilities: np.ndarray,
    circle_centers: np.ndarray,
    circle_radii: np.ndarray,
    localization_cursors: np.ndarray,
    seed: int,
) -> None:
    warm_cfg = replace(cfg)
    warm_cfg.horizon = min(int(cfg.horizon), 8)
    warm_cfg.num_rollouts = min(max(16, int(cfg.num_rollouts)), 32)
    racing_controller_step(
        variant_value,
        state,
        current_s,
        warm_cfg,
        model,
        np.random.default_rng(int(seed) + 999983),
        prior_bank,
        active_indices,
        active_probabilities,
        circle_centers,
        circle_radii,
        np.ascontiguousarray(localization_cursors, dtype=np.int64).copy(),
        record_predictions=False,
    )

def run_race(
    *,
    variant_value: str = 'standard_mppi',
    laps: int,
    num_rollouts: int,
    lbps_delta: float,
    seed: int,
    wall_mode: str = 'no_wall',
    horizon: int = 50,
    temporal_noise_smoothing: float = 0.1,
    sigma0_scale: float = 1.0,
    v_max: float = 4.0,
    hard_collision_clearance: float = 0.01,
    model_name: str = 'ackermann',
    max_prior: Optional[int] = MAX_PRIOR,
    record_predictions: bool = True,
    max_steps_per_lap: int = MAX_STEPS_PER_LAP,
    max_steps: Optional[int] = None,
) -> RaceResult:
    if variant_value not in VARIANT_TO_DISPLAY:
        raise ValueError(f'Unsupported racing variant: {variant_value}')
    model = controller_core.resolve_vehicle_model(model_name)
    wall_mode = str(wall_mode)
    if wall_mode not in WALL_MODE_TO_DISPLAY:
        raise ValueError(f'Unsupported wall mode: {wall_mode}')
    laps = int(laps)
    if laps < 1:
        raise ValueError('Laps must be at least 1.')
    if num_rollouts < 32:
        raise ValueError('Rollouts per step must be at least 32.')
    if not math.isfinite(lbps_delta) or not 0.0 < lbps_delta < 1.0:
        raise ValueError('LBPS delta must be strictly between 0 and 1.')
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError('Horizon H must be at least 1.')
    temporal_noise_smoothing = float(temporal_noise_smoothing)
    if not math.isfinite(temporal_noise_smoothing) or not 0.0 <= temporal_noise_smoothing < 1.0:
        raise ValueError('Temporal noise smoothing must be in [0, 1).')
    sigma0_scale = float(sigma0_scale)
    if not math.isfinite(sigma0_scale) or sigma0_scale <= 0.0:
        raise ValueError('Sigma_0 covariance scale must be positive.')
    v_max = float(v_max)
    if not math.isfinite(v_max) or v_max <= 0.0:
        raise ValueError('v_max must be positive.')
    hard_collision_clearance = float(hard_collision_clearance)
    if not math.isfinite(hard_collision_clearance) or hard_collision_clearance < 0.0:
        raise ValueError('hard_collision_clearance must be nonnegative.')
    if max_prior is not None:
        max_prior = int(max_prior)
        if max_prior < 1:
            raise ValueError('max_prior must be at least 1 or None.')
    max_steps_per_lap = int(max_steps_per_lap)
    if max_steps_per_lap < 1:
        raise ValueError('max_steps_per_lap must be at least 1.')
    active_prior_limit = max_prior if variant_value in PRIOR_MEAN_VARIANTS else None

    cfg = make_racing_config(
        num_rollouts,
        lbps_delta,
        horizon=horizon,
        temporal_noise_smoothing=temporal_noise_smoothing,
        sigma0_scale=sigma0_scale,
        v_max=v_max,
        hard_collision_clearance=hard_collision_clearance,
        model_name=model_name,
    )
    rng = np.random.default_rng(int(seed))
    prior_modes, lower_obstacles, upper_obstacles, fixed_barriers, inner_straight_barriers = build_racing_prior_modes(int(seed))
    prior_bank = _pack_racing_prior_bank(prior_modes)
    base_obstacles = lower_obstacles + upper_obstacles + fixed_barriers
    sector_bank = _build_collision_sector_bank(lower_obstacles, upper_obstacles, inner_straight_barriers)
    active_polygons, active_polygon_lengths = controller_core.obstacle_polygons_to_padded_arrays(base_obstacles)
    active_polygons = np.ascontiguousarray(active_polygons, dtype=np.float64)
    active_polygon_lengths = np.ascontiguousarray(active_polygon_lengths, dtype=np.int64)
    fixed_filter_centers, fixed_filter_radii = _inner_barrier_filter_circle_arrays(inner_straight_barriers)
    fixed_feasible_mask = _prior_feasible_mask_nb(
        prior_bank.mean_paths,
        prior_bank.localization_lengths,
        fixed_filter_centers,
        fixed_filter_radii,
        float(cfg.vehicle_length),
        float(cfg.vehicle_width),
        float(cfg.hard_collision_clearance),
        int(cfg.mode_blocking_substeps),
    )
    if not np.any(fixed_feasible_mask):
        raise RuntimeError('The H-shaped infield barrier blocks every complete prior. Increase the infield clearance or regenerate a wider prior bank.')
    fixed_feasible_mask = np.ascontiguousarray(fixed_feasible_mask, dtype=np.bool_)

    lower_wall_candidates: list[DynamicWallCandidate] = []
    upper_wall_candidates: list[DynamicWallCandidate] = []
    lower_candidate_p0s = np.zeros((0, 2), dtype=np.float64)
    lower_candidate_p1s = np.zeros((0, 2), dtype=np.float64)
    upper_candidate_p0s = np.zeros((0, 2), dtype=np.float64)
    upper_candidate_p1s = np.zeros((0, 2), dtype=np.float64)
    wall_lifetime_laps = float(DYNAMIC_WALL_LIFETIME_LAPS.get(wall_mode, 0.0))
    if wall_lifetime_laps > 0.0:
        lower_wall_candidates = _dynamic_wall_candidates(lower_obstacles)
        upper_wall_candidates = _dynamic_wall_candidates(upper_obstacles)
        lower_candidate_p0s, lower_candidate_p1s = _dynamic_candidate_segment_arrays(lower_wall_candidates)
        upper_candidate_p0s, upper_candidate_p1s = _dynamic_candidate_segment_arrays(upper_wall_candidates)

    half_index = 0
    half_start_state_index = 0
    active_dynamic_walls: list[ActiveDynamicWall] = []
    dynamic_circle_centers = np.zeros((0, 2), dtype=np.float64)
    dynamic_circle_radii = np.zeros(0, dtype=np.float64)
    prediction_collision_cache = _build_prediction_collision_cache(
        sector_bank, dynamic_circle_centers, dynamic_circle_radii
    )
    active_indices, active_probabilities = _active_prior_selection(
        prior_bank, fixed_feasible_mask, dynamic_circle_centers, dynamic_circle_radii, cfg, active_prior_limit
    )
    localization_cursors = np.full(len(prior_bank.modes), -1, dtype=np.int64)

    state = initial_race_state(model_name)
    start_s, _ = _project_centerline_nb(float(state[0]), float(state[1]))
    current_s = float(start_s)
    cumulative = 0.0
    progress_step_limit = max(
        1.0,
        2.0 * max(abs(float(cfg.v_min)), abs(float(cfg.v_max)), abs(float(cfg.lateral_velocity_limit))) * float(cfg.dt) + 0.25,
    )
    target_progress = laps * TRACK_LENGTH
    if max_steps is None:
        max_steps = int(laps * max_steps_per_lap)

    states = [state.copy()]
    controls: list[np.ndarray] = []
    nominal_predictions: list[np.ndarray] = []
    mppi_predictions: list[np.ndarray] = []
    temperatures: list[float] = []
    esses: list[float] = []
    feasible_counts: list[int] = []
    cumulative_history = [0.0]
    lap_times: list[float] = []
    completed_laps = 0
    collision = False
    lap_start_step = 0
    dynamic_wall_history: list[list[np.ndarray]] = [[]]
    active_prior_indices_history: list[np.ndarray] = []

    initial_mask = _sector_mask_for_prediction(current_s, state, cfg, sector_bank)
    _warm_racing_kernels(
        variant_value,
        state,
        current_s,
        cfg,
        model,
        prior_bank,
        active_indices,
        active_probabilities,
        prediction_collision_cache.mask_centers[initial_mask],
        prediction_collision_cache.mask_radii[initial_mask],
        localization_cursors,
        int(seed),
    )

    t0 = time.perf_counter()
    for step in range(int(max_steps)):
        if step - lap_start_step >= max_steps_per_lap:
            break
        sector_mask = _sector_mask_for_prediction(current_s, state, cfg, sector_bank)
        prediction_centers = prediction_collision_cache.mask_centers[sector_mask]
        prediction_radii = prediction_collision_cache.mask_radii[sector_mask]
        if record_predictions and variant_value in PRIOR_MEAN_VARIANTS:
            active_prior_indices_history.append(np.asarray(active_indices, dtype=np.int64).copy())
        control, info = racing_controller_step(
            variant_value,
            state,
            current_s,
            cfg,
            model,
            rng,
            prior_bank,
            active_indices,
            active_probabilities,
            prediction_centers,
            prediction_radii,
            localization_cursors,
            record_predictions=record_predictions,
        )
        localization_cursors = np.ascontiguousarray(info['localization_cursors'], dtype=np.int64)
        control = model.apply_final_output(
            state, control, controls[-1] if controls else None, [], np.zeros(2), cfg
        )
        next_state = np.asarray(model.vehicle_step(state, control, cfg), dtype=np.float64)
        exact_hit, collision_state = _first_exact_transition_collision_nb(
            state,
            next_state,
            float(cfg.vehicle_length),
            float(cfg.vehicle_width),
            active_polygons,
            active_polygon_lengths,
            int(cfg.collision_substeps),
        )
        if exact_hit:
            terminal_state = np.asarray(collision_state, dtype=np.float64)
            new_s, _ = _project_centerline_near_s_nb(
                float(terminal_state[0]), float(terminal_state[1]), float(current_s),
                progress_step_limit, progress_step_limit,
            )
            ds = float(_signed_progress_delta_nb(float(new_s), float(current_s)))
            cumulative += ds
            current_s = float(new_s)
            controls.append(control.copy())
            states.append(terminal_state.copy())
            temperatures.append(float(info['temperature']))
            esses.append(float(info['ess']))
            feasible_counts.append(int(info['finite_rollouts']))
            cumulative_history.append(cumulative)
            if record_predictions:
                nominal_predictions.append(np.asarray(info['nominal_traj'], dtype=np.float64).copy())
                mppi_predictions.append(np.asarray(info['optimal_traj'], dtype=np.float64).copy())
            new_completed = int(math.floor(max(cumulative, 0.0) / TRACK_LENGTH + 1e-12))
            previous_completed_laps = completed_laps
            while completed_laps < min(new_completed, laps):
                completed_laps += 1
                lap_times.append((step + 1) * float(cfg.dt))
            if completed_laps > previous_completed_laps:
                lap_start_step = step + 1
            dynamic_wall_history.append([
                active.candidate.vertices
                for active in active_dynamic_walls
            ])
            collision = True
            state = terminal_state
            break

        new_s, _ = _project_centerline_near_s_nb(
            float(next_state[0]), float(next_state[1]), float(current_s),
            progress_step_limit, progress_step_limit,
        )
        ds = float(_signed_progress_delta_nb(float(new_s), float(current_s)))
        cumulative += ds
        current_s = float(new_s)
        controls.append(control.copy())
        states.append(next_state.copy())
        temperatures.append(float(info['temperature']))
        esses.append(float(info['ess']))
        feasible_counts.append(int(info['finite_rollouts']))
        cumulative_history.append(cumulative)
        if record_predictions:
            nominal_predictions.append(np.asarray(info['nominal_traj'], dtype=np.float64).copy())
            mppi_predictions.append(np.asarray(info['optimal_traj'], dtype=np.float64).copy())
        new_completed = int(math.floor(max(cumulative, 0.0) / TRACK_LENGTH + 1e-12))
        previous_completed_laps = completed_laps
        while completed_laps < min(new_completed, laps):
            completed_laps += 1
            lap_times.append((step + 1) * float(cfg.dt))
        if completed_laps > previous_completed_laps:
            lap_start_step = step + 1

        wall_configuration_changed = False
        if wall_lifetime_laps > 0.0:
            kept_active = [
                active for active in active_dynamic_walls
                if cumulative < float(active.expiration_progress)
            ]
            if len(kept_active) != len(active_dynamic_walls):
                wall_configuration_changed = True
            active_dynamic_walls = kept_active
            new_half_index = int(math.floor(max(cumulative, 0.0) / HALF_TRACK_LENGTH + 1e-12))
            while new_half_index > half_index:
                completed_half_index = half_index
                if completed_half_index % 2 == 0:
                    candidates = lower_wall_candidates
                    candidate_p0s = lower_candidate_p0s
                    candidate_p1s = lower_candidate_p1s
                else:
                    candidates = upper_wall_candidates
                    candidate_p0s = upper_candidate_p0s
                    candidate_p1s = upper_candidate_p1s
                reserved_centers, reserved_radii = _dynamic_circle_arrays_from_walls(active_dynamic_walls)
                reserved_candidate_ids = {id(active.candidate) for active in active_dynamic_walls}
                taken_path = np.asarray(states[half_start_state_index:], dtype=np.float64)[:, :2]
                candidate = _select_shortest_cutting_wall(
                    candidates,
                    candidate_p0s,
                    candidate_p1s,
                    taken_path,
                    prior_bank,
                    fixed_feasible_mask,
                    cfg,
                    reserved_centers,
                    reserved_radii,
                    reserved_candidate_ids,
                )
                activation_progress = float(completed_half_index + 1) * HALF_TRACK_LENGTH
                if candidate is not None:
                    active_dynamic_walls.append(
                        ActiveDynamicWall(
                            candidate=candidate,
                            activation_progress=activation_progress,
                            expiration_progress=float(
                                activation_progress + wall_lifetime_laps * TRACK_LENGTH
                            ),
                        )
                    )
                    wall_configuration_changed = True
                half_index += 1
                half_start_state_index = max(0, len(states) - 1)
            if wall_configuration_changed:
                dynamic_circle_centers, dynamic_circle_radii = _dynamic_circle_arrays_from_walls(active_dynamic_walls)
                prediction_collision_cache = _build_prediction_collision_cache(
                    sector_bank, dynamic_circle_centers, dynamic_circle_radii
                )
                exact_active_obstacles = list(base_obstacles) + [
                    active.candidate.wall for active in active_dynamic_walls
                ]
                active_polygons, active_polygon_lengths = controller_core.obstacle_polygons_to_padded_arrays(
                    exact_active_obstacles
                )
                active_polygons = np.ascontiguousarray(active_polygons, dtype=np.float64)
                active_polygon_lengths = np.ascontiguousarray(active_polygon_lengths, dtype=np.int64)
                active_indices, active_probabilities = _active_prior_selection(
                    prior_bank,
                    fixed_feasible_mask,
                    dynamic_circle_centers,
                    dynamic_circle_radii,
                    cfg,
                    active_prior_limit,
                )
                localization_cursors.fill(-1)

        dynamic_wall_history.append([
            active.candidate.vertices
            for active in active_dynamic_walls
        ])
        state = next_state
        if cumulative >= target_progress:
            completed_laps = laps
            break

    runtime = time.perf_counter() - t0
    obstacle_vertices = [
        np.asarray(getattr(obs, 'vertices', obs), dtype=np.float64)[:, :2].copy()
        for obs in base_obstacles
    ]
    return RaceResult(
        states=np.asarray(states, dtype=np.float64),
        controls=np.asarray(controls, dtype=np.float64),
        nominal_predictions=nominal_predictions,
        mppi_predictions=mppi_predictions,
        temperatures=np.asarray(temperatures, dtype=np.float64),
        esses=np.asarray(esses, dtype=np.float64),
        feasible_counts=np.asarray(feasible_counts, dtype=np.int64),
        cumulative_progress=np.asarray(cumulative_history, dtype=np.float64),
        lap_times=lap_times,
        requested_laps=laps,
        completed_laps=completed_laps,
        collision=collision,
        max_steps_per_lap=max_steps_per_lap,
        runtime_s=runtime,
        variant_value=variant_value,
        wall_mode=wall_mode,
        cfg=cfg,
        model_name=model_name,
        obstacle_vertices=obstacle_vertices,
        dynamic_wall_history=dynamic_wall_history,
        prior_bank=prior_bank,
        active_prior_indices_history=active_prior_indices_history,
    )

class NASCARViewer:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title('MPPI NASCAR racing viewer')
        self.root.minsize(1080, 700)
        try:
            self.root.state('zoomed')
        except tk.TclError:
            self.root.geometry(f'{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0')
        self.worker: Optional[threading.Thread] = None
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.result: Optional[RaceResult] = None
        self.frame_index = 0
        self.playing = False
        self.after_id: Optional[str] = None
        self.poll_after_id: Optional[str] = None
        self.closing = False
        self.updating_slider = False
        self._prior_visual_cursors: Optional[np.ndarray] = None
        self._prior_mean_artists: dict[int, object] = {}
        self._prior_cov_artists: dict[int, PolyCollection] = {}
        self._prior_cov_sample_indices: dict[int, np.ndarray] = {}
        self._prior_mean_ranges: dict[int, tuple[int, int]] = {}
        self._prior_mean_probabilities: dict[int, tuple[float, float]] = {}
        self._prior_cov_states: dict[int, tuple[int, int, float, float]] = {}
        self._prior_cov_visibility: dict[int, bool] = {}
        _, _, default_obstacles = build_racing_obstacles()
        self._default_obstacle_vertices = [_poly_vertices(obstacle).copy() for obstacle in default_obstacles]
        self._build_ui()
        self.poll_after_id = self.root.after(100, self._poll_worker)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0, minsize=330)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        controls = ttk.Frame(self.root, padding=12, width=330)
        controls.grid(row=0, column=0, sticky='ns')
        controls.grid_propagate(False)
        controls.columnconfigure(0, weight=1)
        plot_frame = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        plot_frame.grid(row=0, column=1, sticky='nsew')
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        ttk.Label(controls, text='NASCAR race', font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=0, sticky='w', pady=(0, 8))
        base_cfg = ackermann.MPPIConfig()
        self.vehicle_var = tk.StringVar(value='Four-wheel')
        self.variant_var = tk.StringVar(value='SPG prior')
        self.wall_mode_var = tk.StringVar(value='No wall')
        self.laps_var = tk.StringVar(value='3')
        self.rollouts_var = tk.StringVar(value='256')
        self.lbps_delta_var = tk.StringVar(value='0.9')
        self.seed_var = tk.StringVar(value='1')
        self.max_prior_var = tk.StringVar(value=str(MAX_PRIOR))
        self.speed_var = tk.StringVar(value='4.0')
        self.horizon_var = tk.DoubleVar(value=float(15))
        self.temporal_noise_var = tk.DoubleVar(value=float(0.3))
        self.sigma0_scale_var = tk.DoubleVar(value=1.0)
        self.vmax_var = tk.DoubleVar(value=6.0)
        self.hard_collision_clearance_var = tk.DoubleVar(value=float(0.02))
        self.show_prior_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value='Ready')
        self.frame_label_var = tk.StringVar(value='Frame 0 / 0')
        row = 2
        row = self._add_combo(controls, row, 'Vehicle model', self.vehicle_var, [label for label, _ in VEHICLE_SYSTEMS])
        row = self._add_combo(controls, row, 'Controller variant', self.variant_var, [label for label, _ in VARIANTS])
        row = self._add_combo(controls, row, 'Wall mode', self.wall_mode_var, [label for label, _ in WALL_MODES])
        row = self._add_entry(controls, row, 'Number of laps', self.laps_var)
        row = self._add_entry(controls, row, 'Rollouts per step', self.rollouts_var)
        row = self._add_entry(controls, row, 'LBPS delta', self.lbps_delta_var)
        row = self._add_entry(controls, row, 'Controller seed', self.seed_var)
        row = self._add_entry(controls, row, 'Max active priors', self.max_prior_var)
        ttk.Separator(controls).grid(row=row, column=0, sticky='ew', pady=(10, 6))
        row += 1
        ttk.Label(controls, text='Racing controller config', font=('TkDefaultFont', 11, 'bold')).grid(row=row, column=0, sticky='w')
        row += 1
        row = self._add_slider(controls, row, 'Horizon H', self.horizon_var, 10.0, 30.0, formatter=lambda value: str(int(round(value))))
        row = self._add_slider(controls, row, 'Temporal noise', self.temporal_noise_var, 0.0, 0.95, formatter=lambda value: f'{value:.2f}')
        row = self._add_slider(controls, row, 'Sigma_0 covariance scale', self.sigma0_scale_var, 0.1, 4.0, formatter=lambda value: f'{value:.2f}x')
        row = self._add_slider(controls, row, 'V_max [m/s]', self.vmax_var, 4.0, 10.0, formatter=lambda value: f'{value:.1f}')
        row = self._add_slider(controls, row, 'Collision safety [m]', self.hard_collision_clearance_var, 0.0, 0.1, formatter=lambda value: f'{value:.3f}')
        buttons = ttk.Frame(controls)
        buttons.grid(row=row, column=0, sticky='ew', pady=(14, 6))
        buttons.columnconfigure((0, 1), weight=1)
        self.run_button = ttk.Button(buttons, text='Run race', command=self.run_selected)
        self.run_button.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        self.play_button = ttk.Button(buttons, text='Play', command=self.toggle_play, state='disabled')
        self.play_button.grid(row=1, column=0, sticky='ew', padx=(0, 3))
        self.restart_button = ttk.Button(buttons, text='Restart', command=self.restart_animation, state='disabled')
        self.restart_button.grid(row=1, column=1, sticky='ew', padx=(3, 0))
        self.export_gif_button = ttk.Button(buttons, text='Export GIF', command=self.export_gif, state='disabled')
        self.export_gif_button.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(6, 0))
        row += 1
        ttk.Separator(controls).grid(row=row, column=0, sticky='ew', pady=10)
        row += 1
        ttk.Label(controls, text='Animation', font=('TkDefaultFont', 11, 'bold')).grid(row=row, column=0, sticky='w')
        row += 1
        self.show_prior_check = ttk.Checkbutton(
            controls, text='Show prior', variable=self.show_prior_var, command=self._on_show_prior_changed
        )
        self.show_prior_check.grid(row=row, column=0, sticky='w', pady=(4, 2))
        row += 1
        ttk.Label(controls, textvariable=self.frame_label_var).grid(row=row, column=0, sticky='w', pady=(4, 2))
        row += 1
        self.frame_scale = ttk.Scale(controls, from_=0, to=0, orient='horizontal', command=self._on_frame_slider, state='disabled')
        self.frame_scale.grid(row=row, column=0, sticky='ew')
        row += 1
        ttk.Separator(controls).grid(row=row, column=0, sticky='ew', pady=10)
        row += 1
        ttk.Label(controls, text='Status', font=('TkDefaultFont', 11, 'bold')).grid(row=row, column=0, sticky='w')
        row += 1
        self.status_label = ttk.Label(controls, textvariable=self.status_var, wraplength=295, justify='left', anchor='nw')
        self.status_label.grid(row=row, column=0, sticky='nsew', pady=(4, 0))
        controls.rowconfigure(row, weight=1)
        self.figure = Figure(figsize=(10, 7))
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.93)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky='ew')
        self._init_plot_artists()
        self._draw_frame(0)

    @staticmethod
    def _add_entry(parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=(5, 2))
        ttk.Entry(parent, textvariable=variable, width=32).grid(row=row + 1, column=0, sticky='ew')
        return row + 2

    @staticmethod
    def _add_combo(parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, values: list[str]) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=(5, 2))
        ttk.Combobox(parent, textvariable=variable, values=values, state='readonly', width=30).grid(row=row + 1, column=0, sticky='ew')
        return row + 2

    @staticmethod
    def _add_slider(parent: ttk.Frame, row: int, label: str, variable: tk.DoubleVar, lower: float, upper: float, *, formatter) -> int:
        header = ttk.Frame(parent)
        header.grid(row=row, column=0, sticky='ew', pady=(5, 0))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=label).grid(row=0, column=0, sticky='w')
        value_var = tk.StringVar(value=formatter(float(variable.get())))
        ttk.Label(header, textvariable=value_var, width=8, anchor='e').grid(row=0, column=1, sticky='e')

        def update_value(raw: str) -> None:
            value = float(raw)
            value_var.set(formatter(value))
        slider = ttk.Scale(parent, from_=float(lower), to=float(upper), orient='horizontal', variable=variable, command=update_value)
        slider.grid(row=row + 1, column=0, sticky='ew', pady=(0, 2))
        return row + 2

    def run_selected(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo('Simulation running', 'Wait for the current race to finish.')
            return
        try:
            vehicle_label = self.vehicle_var.get()
            if vehicle_label not in DISPLAY_TO_VEHICLE:
                raise ValueError('Choose a valid vehicle model.')
            model_name = DISPLAY_TO_VEHICLE[vehicle_label]
            variant_label = self.variant_var.get()
            if variant_label not in DISPLAY_TO_VARIANT:
                raise ValueError('Choose a valid controller variant.')
            variant_value = DISPLAY_TO_VARIANT[variant_label]
            wall_mode_label = self.wall_mode_var.get()
            if wall_mode_label not in DISPLAY_TO_WALL_MODE:
                raise ValueError('Choose a valid wall mode.')
            wall_mode = DISPLAY_TO_WALL_MODE[wall_mode_label]
            laps = int(self.laps_var.get())
            rollouts = int(self.rollouts_var.get())
            lbps_delta = float(self.lbps_delta_var.get())
            seed = int(self.seed_var.get())
            max_prior = int(self.max_prior_var.get())
            horizon = int(round(float(self.horizon_var.get())))
            temporal_noise = float(self.temporal_noise_var.get())
            sigma0_scale = float(self.sigma0_scale_var.get())
            v_max = float(self.vmax_var.get())
            hard_collision_clearance = float(self.hard_collision_clearance_var.get())
            if laps < 1:
                raise ValueError('Number of laps must be at least 1.')
            if laps > 100:
                raise ValueError('Number of laps must be 100 or less for the interactive viewer.')
            if rollouts < 32:
                raise ValueError('Rollouts per step must be at least 32.')
            if not 0.0 < lbps_delta < 1.0:
                raise ValueError('LBPS delta must be strictly between 0 and 1.')
            if max_prior < 1:
                raise ValueError('Max active priors must be at least 1.')
            if not 10 <= horizon <= 100:
                raise ValueError('Horizon H must be between 10 and 100.')
            if not 0.0 <= temporal_noise < 1.0:
                raise ValueError('Temporal noise smoothing must be in [0, 1).')
            if sigma0_scale <= 0.0:
                raise ValueError('Sigma_0 covariance scale must be positive.')
            if v_max <= 0.0:
                raise ValueError('v_max must be positive.')
            if hard_collision_clearance < 0.0:
                raise ValueError('hard_collision_clearance must be nonnegative.')
        except ValueError as exc:
            messagebox.showerror('Invalid settings', str(exc))
            return
        self._stop_animation()
        self.result = None
        self.frame_index = 0
        self.run_button.configure(state='disabled')
        self.play_button.configure(state='disabled', text='Play')
        self.restart_button.configure(state='disabled')
        self.export_gif_button.configure(state='disabled')
        self.frame_scale.configure(state='disabled')
        self.status_var.set(f'Running {vehicle_label} / {variant_label} / {wall_mode_label}: H={horizon}, max_prior={max_prior}, temporal={temporal_noise:.2f}, Sigma_0 scale={sigma0_scale:.2f}x, v_max={v_max:.1f} m/s. First run also builds the two straight-line prior banks.')
        settings = dict(variant_value=variant_value, laps=laps, num_rollouts=rollouts, lbps_delta=lbps_delta, seed=seed, wall_mode=wall_mode, horizon=horizon, temporal_noise_smoothing=temporal_noise, sigma0_scale=sigma0_scale, v_max=v_max, hard_collision_clearance=hard_collision_clearance, model_name=model_name, max_prior=max_prior)
        self.worker = threading.Thread(target=self._worker_run, kwargs=settings, daemon=True)
        self.worker.start()

    def _worker_run(self, **settings: object) -> None:
        try:
            result = run_race(**settings)
            self.worker_queue.put(('success', result))
        except Exception:
            self.worker_queue.put(('error', traceback.format_exc()))

    def _poll_worker(self) -> None:
        if self.closing:
            return
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == 'success':
                    self._on_ready(payload)
                else:
                    self._on_error(str(payload))
        except queue.Empty:
            pass
        if not self.closing:
            self.poll_after_id = self.root.after(100, self._poll_worker)

    def _on_ready(self, payload: object) -> None:
        self._reset_prior_visual_artists()
        self.result = payload if isinstance(payload, RaceResult) else None
        self.run_button.configure(state='normal')
        if self.result is None:
            self.status_var.set('Race returned no result.')
            return
        total_frames = max(0, len(self.result.states) - 1)
        self.frame_scale.configure(from_=0, to=total_frames, state='normal')
        self.play_button.configure(state='normal')
        self.restart_button.configure(state='normal')
        self.export_gif_button.configure(state='normal')
        self.frame_index = 0
        self._set_slider(0)
        self._draw_frame(0)
        if self.result.completed_laps >= self.result.requested_laps:
            lap_text = ', '.join((f'{value:.2f}s' for value in self.result.lap_times))
            self.status_var.set(f'Finished {self.result.completed_laps}/{self.result.requested_laps} laps ({WALL_MODE_TO_DISPLAY.get(self.result.wall_mode, self.result.wall_mode)}). Race time {len(self.result.controls) * self.result.cfg.dt:.2f}s. Lap crossing times: {lap_text}. Compute time {self.result.runtime_s:.2f}s.')
        elif self.result.collision:
            self.status_var.set(f'DNF: obstacle collision after {self.result.completed_laps}/{self.result.requested_laps} laps; final frame is first contact. Simulated time {len(self.result.controls) * self.result.cfg.dt:.2f}s.')
        else:
            self.status_var.set(f'DNF: {self.result.max_steps_per_lap}-step per-lap limit reached after {self.result.completed_laps}/{self.result.requested_laps} laps.')
        if len(self.result.states) > 1:
            self.playing = True
            self.play_button.configure(text='Pause')
            self._schedule_next_frame()

    def _on_error(self, text: str) -> None:
        self.run_button.configure(state='normal')
        self.play_button.configure(state='disabled', text='Play')
        self.restart_button.configure(state='disabled')
        self.export_gif_button.configure(state='disabled')
        self.status_var.set('Race failed. See error dialog.')
        messagebox.showerror('Race failed', text)

    def toggle_play(self) -> None:
        if self.result is None or len(self.result.states) <= 1:
            return
        if self.playing:
            self._stop_animation()
            return
        if self.frame_index >= len(self.result.states) - 1:
            self.frame_index = 0
        self.playing = True
        self.play_button.configure(text='Pause')
        self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        if not self.playing or self.result is None:
            return
        try:
            playback = max(float(self.speed_var.get()), 0.1)
        except ValueError:
            playback = 1.0
        delay_ms = max(10, int(1000.0 * self.result.cfg.dt / playback))
        self.after_id = self.root.after(delay_ms, self._advance_frame)

    def _advance_frame(self) -> None:
        self.after_id = None
        if not self.playing or self.result is None:
            return
        if self.frame_index >= len(self.result.states) - 1:
            self._stop_animation()
            return
        self.frame_index += 1
        self._set_slider(self.frame_index)
        self._draw_frame(self.frame_index)
        self._schedule_next_frame()

    def _stop_animation(self) -> None:
        self.playing = False
        if hasattr(self, 'play_button'):
            self.play_button.configure(text='Play')
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def restart_animation(self) -> None:
        self._stop_animation()
        self.frame_index = 0
        self._set_slider(0)
        self._draw_frame(0)

    def export_gif(self) -> None:
        if self.result is None or len(self.result.states) == 0:
            return

        was_playing = self.playing
        original_frame = int(self.frame_index)
        self._stop_animation()
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title='Export race animation as GIF',
            defaultextension='.gif',
            filetypes=[('GIF animation', '*.gif'), ('All files', '*.*')],
            initialfile='nascar_race.gif',
        )
        if not path:
            if was_playing and original_frame < len(self.result.states) - 1:
                self.playing = True
                self.play_button.configure(text='Pause')
                self._schedule_next_frame()
            return

        try:
            import imageio.v2 as imageio
            from PIL import Image
        except ImportError:
            messagebox.showerror(
                'GIF export unavailable',
                'GIF export requires imageio and Pillow. Install them with: pip install imageio pillow',
                parent=self.root,
            )
            if was_playing and original_frame < len(self.result.states) - 1:
                self.playing = True
                self.play_button.configure(text='Pause')
                self._schedule_next_frame()
            return

        previous_status = self.status_var.get()
        axis_was_on = bool(self.ax.axison)
        title_was_visible = bool(self.ax.title.get_visible())
        legend = self.ax.get_legend()
        legend_was_visible = bool(legend.get_visible()) if legend is not None else False
        export_width = 720
        self.export_gif_button.configure(state='disabled')
        try:
            try:
                playback = max(float(self.speed_var.get()), 0.1)
            except ValueError:
                playback = 1.0
            delay_ms = max(10, int(1000.0 * float(self.result.cfg.dt) / playback))
            total_frames = len(self.result.states)
            self._prior_visual_cursors = None
            with imageio.get_writer(path, mode='I', duration=delay_ms, loop=0) as writer:
                for frame in range(total_frames):
                    self.frame_index = frame
                    self._set_slider(frame)
                    self._draw_frame(frame)

                    # Export only the clean plot: no axes, labels, ticks, grid, title, or legend.
                    self.ax.set_axis_off()
                    self.ax.title.set_visible(False)
                    if legend is not None:
                        legend.set_visible(False)
                    self.canvas.draw()

                    rgba = np.asarray(self.canvas.buffer_rgba())
                    renderer = self.canvas.get_renderer()
                    bbox = self.ax.get_window_extent(renderer=renderer)
                    x0 = max(0, int(math.floor(bbox.x0)))
                    x1 = min(rgba.shape[1], int(math.ceil(bbox.x1)))
                    y0 = max(0, int(math.floor(rgba.shape[0] - bbox.y1)))
                    y1 = min(rgba.shape[0], int(math.ceil(rgba.shape[0] - bbox.y0)))
                    rgb = np.ascontiguousarray(rgba[y0:y1, x0:x1, :3])
                    if rgb.size == 0:
                        raise RuntimeError('GIF export produced an empty plot crop.')

                    export_height = max(1, int(round(rgb.shape[0] * export_width / rgb.shape[1])))
                    image = Image.fromarray(rgb).resize(
                        (export_width, export_height),
                        Image.Resampling.LANCZOS,
                    )
                    writer.append_data(np.ascontiguousarray(np.asarray(image)))

                    if frame % 10 == 0 or frame == total_frames - 1:
                        self.status_var.set(
                            f'Exporting GIF: {frame + 1}/{total_frames}  ({export_width}x{export_height})'
                        )
                        self.root.update_idletasks()
            self.status_var.set(f'GIF exported: {path}')
        except Exception as exc:
            self.status_var.set(previous_status)
            messagebox.showerror('GIF export failed', str(exc), parent=self.root)
        finally:
            if axis_was_on:
                self.ax.set_axis_on()
            else:
                self.ax.set_axis_off()
            self.ax.title.set_visible(title_was_visible)
            if legend is not None:
                legend.set_visible(legend_was_visible)
            self._prior_visual_cursors = None
            self.frame_index = original_frame
            self._set_slider(original_frame)
            self._draw_frame(original_frame)
            self.canvas.draw_idle()
            self.export_gif_button.configure(state='normal' if self.result is not None else 'disabled')
            if was_playing and self.result is not None and original_frame < len(self.result.states) - 1:
                self.playing = True
                self.play_button.configure(text='Pause')
                self._schedule_next_frame()

    def _set_slider(self, value: int) -> None:
        self.updating_slider = True
        self.frame_scale.set(value)
        self.updating_slider = False

    def _on_show_prior_changed(self) -> None:
        self._draw_frame(self.frame_index)

    def _on_frame_slider(self, value: str) -> None:
        if self.updating_slider or self.result is None:
            return
        try:
            frame = int(round(float(value)))
        except ValueError:
            return
        frame = int(np.clip(frame, 0, len(self.result.states) - 1))
        self.frame_index = frame
        self._draw_frame(frame)

    @staticmethod
    def _transform_vehicle_points(points: np.ndarray, origin: tuple[float, float], angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        rotation = np.asarray([[c, -s], [s, c]], dtype=float)
        return np.asarray(points, dtype=float) @ rotation.T + np.asarray(origin, dtype=float)

    def _sync_obstacle_artists(self, obstacle_vertices: list[np.ndarray]) -> None:
        while len(self.obstacle_patches) < len(obstacle_vertices):
            patch = Polygon(
                np.zeros((3, 2)),
                closed=True,
                facecolor='0.25',
                edgecolor='0.05',
                linewidth=1.1,
                alpha=0.78,
                zorder=3,
            )
            self.ax.add_patch(patch)
            self.obstacle_patches.append(patch)
        for i, patch in enumerate(self.obstacle_patches):
            if i < len(obstacle_vertices):
                patch.set_xy(np.asarray(obstacle_vertices[i], dtype=float))
                patch.set_visible(True)
            else:
                patch.set_visible(False)

    def _init_plot_artists(self) -> None:
        self.ax.set_xlim(TRACK_PLOT_X_MIN, TRACK_PLOT_X_MAX)
        self.ax.set_ylim(-0.8, TRACK_HEIGHT + 0.8)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.grid(True, alpha=0.15)
        self.ax.set_facecolor('white')
        self.obstacle_patches: list[Polygon] = []
        self._sync_obstacle_artists(self._default_obstacle_vertices)
        if self.obstacle_patches:
            self.obstacle_patches[0].set_label('Obstacles')
        self.dynamic_wall_patches: list[Polygon] = []
        dynamic_legend_patch = Polygon(
            np.zeros((3, 2)),
            closed=True,
            facecolor='0.08',
            edgecolor='0.02',
            linewidth=1.3,
            alpha=0.92,
            zorder=4
        )
        dynamic_legend_patch.set_visible(False)
        self.ax.add_patch(dynamic_legend_patch)
        self.dynamic_wall_patches.append(dynamic_legend_patch)
        start, _ = centerline_point_tangent(START_S)
        self.ax.plot(
            [start[0], start[0]], [start[1] - 1.0, start[1] + 1.0],
            color='0.15', linewidth=2.0, zorder=4,
        )
        self.executed_line, = self.ax.plot(
            [], [], color='#1f77b4', linewidth=2.4, zorder=5, label='Executed trajectory'
        )
        self.prior_cov_legend = PolyCollection(
            [], facecolors=[(*PRIOR_PURPLE, 0.045)],
            edgecolors='none', linewidths=0.0, zorder=4.5, label='Prior covariance (1σ)'
        )
        self.prior_cov_legend.set_visible(False)
        self.ax.add_collection(self.prior_cov_legend)
        self.prior_mean_legend = LineCollection(
            [], colors=[(*PRIOR_PURPLE, 1.0)], linewidths=1.5, linestyles='--',
            alpha=0.65, zorder=5.5, label='Prior μ'
        )
        self.prior_mean_legend.set_visible(False)
        self.ax.add_collection(self.prior_mean_legend)
        self.nominal_line, = self.ax.plot(
            [], [], color='#0066cc', linewidth=2.0, linestyle='--', alpha=0.95,
            zorder=6, label='Nominal'
        )
        self.prediction_line, = self.ax.plot(
            [], [], color='#ff7f0e', linewidth=2.2, linestyle='--', alpha=0.95,
            zorder=7, label='MPPI prediction'
        )
        self.vehicle_body = Polygon(
            np.zeros((4, 2)), closed=True, facecolor='#17becf', edgecolor='black',
            linewidth=1.1, alpha=0.92, zorder=12,
        )
        self.ax.add_patch(self.vehicle_body)
        self.vehicle_wheels = []
        for _ in range(4):
            wheel = Polygon(
                np.zeros((4, 2)), closed=True, facecolor='0.10', edgecolor='black',
                linewidth=0.6, zorder=13,
            )
            self.ax.add_patch(wheel)
            self.vehicle_wheels.append(wheel)
        self.velocity_arrow = FancyArrowPatch(
            (0.0, 0.0), (0.0, 0.0), arrowstyle='-|>', mutation_scale=10.0,
            linewidth=1.2, color='#1f77b4', zorder=14,
        )
        self.ax.add_patch(self.velocity_arrow)
        self.nose_line, = self.ax.plot([], [], color='black', linewidth=1.1, zorder=14)
        self._obstacle_source_id = None
        self.ax.legend(loc='upper center', ncol=6, fontsize=8)

    def _reset_prior_visual_artists(self) -> None:
        for artist in self._prior_mean_artists.values():
            artist.remove()
        for artist in self._prior_cov_artists.values():
            artist.remove()
        self._prior_mean_artists.clear()
        self._prior_cov_artists.clear()
        self._prior_cov_sample_indices.clear()
        self._prior_mean_ranges.clear()
        self._prior_mean_probabilities.clear()
        self._prior_cov_states.clear()
        self._prior_cov_visibility.clear()
        self._prior_visual_cursors = None

    def _ensure_prior_mean_artist(self, mode_index: int):
        m = int(mode_index)
        artist = self._prior_mean_artists.get(m)
        if artist is None:
            artist, = self.ax.plot(
                [], [], color=PRIOR_PURPLE, linewidth=1.35, linestyle='--',
                alpha=0.5, zorder=5.5, label='_nolegend_'
            )
            artist.set_visible(False)
            self._prior_mean_artists[m] = artist
        return artist

    def _ensure_prior_cov_artist(self, mode_index: int) -> tuple[PolyCollection, np.ndarray]:
        m = int(mode_index)
        artist = self._prior_cov_artists.get(m)
        if artist is None:
            if self.result is None:
                raise RuntimeError('Prior visualization requires a race result.')
            polygons, sample_indices = _sparse_covariance_mode_geometry(self.result.prior_bank, m)
            purple = (*PRIOR_PURPLE, 0.0)
            artist = PolyCollection(
                polygons, facecolors=[purple] * len(polygons), edgecolors='none',
                linewidths=0.0, zorder=4.5, label='_nolegend_'
            )
            artist.set_visible(False)
            self.ax.add_collection(artist)
            self._prior_cov_artists[m] = artist
            self._prior_cov_sample_indices[m] = sample_indices
        return artist, self._prior_cov_sample_indices[m]

    def _hide_prior_visuals(self) -> None:
        for artist in self._prior_mean_artists.values():
            artist.set_visible(False)
        for artist in self._prior_cov_artists.values():
            artist.set_visible(False)

    def _update_vehicle_artists(self, state: np.ndarray, cfg: object, model_name: str) -> None:
        if state.size < 7:
            raise ValueError('Vehicle state must contain [x, y, psi, vx, vy, r, delta].')
        x, y, heading, vx, vy, _, steering = map(float, state[:7])
        lf = float(cfg.front_axle_distance)
        lr = float(cfg.rear_axle_distance)
        body_length = float(cfg.vehicle_length)
        body_width = float(cfg.vehicle_width)
        body = np.asarray([
            [-0.5 * body_length, -0.5 * body_width],
            [0.5 * body_length, -0.5 * body_width],
            [0.5 * body_length, 0.5 * body_width],
            [-0.5 * body_length, 0.5 * body_width],
        ])
        self.vehicle_body.set_xy(self._transform_vehicle_points(body, (x, y), heading))
        self.vehicle_body.set_facecolor(four_wheel.BODY_COLOR if model_name == 'four_wheel' else ackermann.BODY_COLOR)
        wheel_shape = np.asarray([
            [-0.11, -0.0275], [0.11, -0.0275], [0.11, 0.0275], [-0.11, 0.0275]
        ])
        half_track = 0.5 * float(getattr(cfg, 'track_width', 0.86 * body_width))
        c = math.cos(heading)
        s = math.sin(heading)

        def body_to_world(longitudinal: float, lateral: float) -> tuple[float, float]:
            return (
                x + longitudinal * c - lateral * s,
                y + longitudinal * s + lateral * c,
            )

        wheel_specs = (
            (lf, half_track, heading + steering),
            (lf, -half_track, heading + steering),
            (-lr, half_track, heading),
            (-lr, -half_track, heading),
        )
        for wheel_index, (artist, (longitudinal, lateral, wheel_heading)) in enumerate(zip(self.vehicle_wheels, wheel_specs)):
            center = body_to_world(longitudinal, lateral)
            artist.set_xy(self._transform_vehicle_points(wheel_shape, center, wheel_heading))
            if model_name == 'four_wheel' and state.size >= 13:
                normalized_speed = min(abs(float(state[9 + wheel_index])) / max(float(cfg.wheel_speed_limit), 1e-9), 1.0)
                artist.set_linewidth(0.55 + 1.15 * normalized_speed)
            else:
                artist.set_linewidth(0.6)
        world_vx = vx * c - vy * s
        world_vy = vx * s + vy * c
        if math.hypot(world_vx, world_vy) > 0.001:
            self.velocity_arrow.set_positions((x, y), (x + 0.22 * world_vx, y + 0.22 * world_vy))
            self.velocity_arrow.set_visible(True)
        else:
            self.velocity_arrow.set_visible(False)
        nose = body_to_world(0.52 * body_length, 0.0)
        self.nose_line.set_data([x, nose[0]], [y, nose[1]])

    def _update_dynamic_wall_artists(self, frame: int) -> None:
        walls: list[np.ndarray] = []
        if self.result is not None and self.result.dynamic_wall_history:
            idx = int(np.clip(frame, 0, len(self.result.dynamic_wall_history) - 1))
            walls = self.result.dynamic_wall_history[idx]
        while len(self.dynamic_wall_patches) < max(1, len(walls)):
            patch = Polygon(
                np.zeros((3, 2)), closed=True, facecolor='0.08', edgecolor='0.02',
                linewidth=1.3, alpha=0.92, zorder=4,
            )
            self.ax.add_patch(patch)
            self.dynamic_wall_patches.append(patch)
        for i, patch in enumerate(self.dynamic_wall_patches):
            if i < len(walls):
                patch.set_xy(np.asarray(walls[i], dtype=float))
                patch.set_visible(True)
            else:
                patch.set_visible(False)

    def _draw_frame(self, frame: int) -> None:
        if self.result is None:
            self.executed_line.set_data([], [])
            self.nominal_line.set_data([], [])
            self.prediction_line.set_data([], [])
            self.nominal_line.set_visible(False)
            self.prediction_line.set_visible(False)
            self._hide_prior_visuals()
            self._update_dynamic_wall_artists(0)
            model_name = DISPLAY_TO_VEHICLE.get(self.vehicle_var.get(), 'ackermann')
            model = controller_core.resolve_vehicle_model(model_name)
            self._update_vehicle_artists(initial_race_state(model_name), model.MPPIConfig(), model_name)
            self.ax.set_title('Choose a variant, laps, and run')
            self.frame_label_var.set('Frame 0 / 0')
            self.canvas.draw_idle()
            return

        if self._obstacle_source_id != id(self.result):
            self._sync_obstacle_artists(self.result.obstacle_vertices)
            self._obstacle_source_id = id(self.result)
        frame = int(np.clip(frame, 0, len(self.result.states) - 1))
        self._update_dynamic_wall_artists(frame)
        state = self.result.states[frame]
        path = self.result.states[:frame + 1, :2]
        recent = path[-100:]
        self.executed_line.set_data(recent[:, 0], recent[:, 1])
        pred_idx = min(frame, max(0, len(self.result.mppi_predictions) - 1))
        if (
            self.show_prior_var.get()
            and self.result.variant_value in PRIOR_MEAN_VARIANTS
            and self.result.active_prior_indices_history
            and pred_idx < len(self.result.active_prior_indices_history)
        ):
            active_prior_indices = np.asarray(
                self.result.active_prior_indices_history[pred_idx], dtype=np.int64
            )
            if self._prior_visual_cursors is None or len(self._prior_visual_cursors) != len(self.result.prior_bank.modes):
                self._prior_visual_cursors = np.full(len(self.result.prior_bank.modes), -1, dtype=np.int64)
            prior_state = self.result.states[pred_idx]
            prior_s = float((START_S + self.result.cumulative_progress[pred_idx]) % TRACK_LENGTH)
            starts, ends, updated_visual_cursors = _local_racing_ranges(
                prior_state,
                prior_s,
                self.result.prior_bank,
                self.result.cfg,
                active_prior_indices,
                self._prior_visual_cursors,
            )
            self._prior_visual_cursors = np.ascontiguousarray(updated_visual_cursors, dtype=np.int64)
            active_probabilities = np.asarray(
                self.result.prior_bank.probabilities[active_prior_indices], dtype=np.float64
            )
            probability_mass = float(np.sum(active_probabilities))
            if probability_mass <= 1e-15:
                active_probabilities = np.full(
                    len(active_prior_indices), 1.0 / float(len(active_prior_indices)), dtype=np.float64
                )
            else:
                active_probabilities /= probability_mass
            peak_probability = float(np.max(active_probabilities)) if len(active_probabilities) else 1.0
            active_set = {int(value) for value in active_prior_indices}
            for mode_index, artist in self._prior_mean_artists.items():
                if mode_index not in active_set:
                    artist.set_visible(False)
            for mode_index, artist in self._prior_cov_artists.items():
                if mode_index not in active_set:
                    artist.set_visible(False)
            for q, global_index in enumerate(active_prior_indices):
                m = int(global_index)
                start_index = int(starts[q])
                end_index = int(ends[q])
                probability = float(active_probabilities[q])
                mean_artist = self._ensure_prior_mean_artist(m)
                range_key = (start_index, end_index)
                if self._prior_mean_ranges.get(m) != range_key:
                    path = np.asarray(
                        self.result.prior_bank.mean_paths[m, start_index:end_index + 1], dtype=np.float64
                    )
                    mean_artist.set_data(path[:, 0], path[:, 1])
                    self._prior_mean_ranges[m] = range_key
                mean_probability_key = (round(probability, 12), round(peak_probability, 12))
                if self._prior_mean_probabilities.get(m) != mean_probability_key:
                    mean_artist.set_alpha(_prior_mean_alpha(probability, peak_probability))
                    mean_artist.set_linewidth(_prior_mean_linewidth(probability, peak_probability))
                    self._prior_mean_probabilities[m] = mean_probability_key
                mean_artist.set_visible(True)
                if self.result.variant_value in PRIOR_COVARIANCE_VARIANTS:
                    cov_artist, sample_indices = self._ensure_prior_cov_artist(m)
                    cov_key = (start_index, end_index, round(probability, 12), round(peak_probability, 12))
                    if self._prior_cov_states.get(m) != cov_key:
                        visible = _covariance_sample_visibility(
                            sample_indices,
                            start_index,
                            end_index,
                            int(self.result.prior_bank.localization_lengths[m]),
                        )
                        colors = np.zeros((len(sample_indices), 4), dtype=np.float64)
                        if len(sample_indices):
                            colors[:, 0] = PRIOR_PURPLE[0]
                            colors[:, 1] = PRIOR_PURPLE[1]
                            colors[:, 2] = PRIOR_PURPLE[2]
                            colors[visible, 3] = _prior_covariance_alpha(probability, peak_probability)
                        cov_artist.set_facecolors(colors)
                        self._prior_cov_states[m] = cov_key
                        self._prior_cov_visibility[m] = bool(np.any(visible))
                    cov_artist.set_visible(self._prior_cov_visibility.get(m, False))
                elif m in self._prior_cov_artists:
                    self._prior_cov_artists[m].set_visible(False)
        else:
            self._hide_prior_visuals()
        if (
            self.result.variant_value != 'planner_ilqr'
            and self.result.mppi_predictions
            and pred_idx < len(self.result.mppi_predictions)
        ):
            pred = self.result.mppi_predictions[pred_idx]
            self.prediction_line.set_data(pred[:, 0], pred[:, 1])
            self.prediction_line.set_visible(True)
        else:
            self.prediction_line.set_visible(False)
        if (
            self.result.variant_value == 'planner_ilqr'
            and self.result.nominal_predictions
            and pred_idx < len(self.result.nominal_predictions)
        ):
            nominal = self.result.nominal_predictions[pred_idx]
            self.nominal_line.set_data(nominal[:, 0], nominal[:, 1])
            self.nominal_line.set_visible(True)
        else:
            self.nominal_line.set_visible(False)
        self._update_vehicle_artists(state, self.result.cfg, self.result.model_name)
        speed = math.hypot(float(state[3]), float(state[4]))
        progress = float(self.result.cumulative_progress[frame])
        laps_progress = max(progress, 0.0) / TRACK_LENGTH
        completed = min(int(math.floor(laps_progress + 1e-12)), self.result.requested_laps)
        sim_time = min(frame, len(self.result.controls)) * float(self.result.cfg.dt)
        variant_label = VARIANT_TO_DISPLAY.get(self.result.variant_value, self.result.variant_value)
        vehicle_label = VEHICLE_TO_DISPLAY.get(self.result.model_name, self.result.model_name)
        title = (
            f'{vehicle_label} / {variant_label}  |  Lap {min(completed + 1, self.result.requested_laps)}/{self.result.requested_laps}  |  '
            f'progress {laps_progress:.2f} laps  |  speed {speed:.2f} m/s  |  t={sim_time:.1f}s'
        )
        if self.result.model_name == 'four_wheel' and state.size >= 13:
            title += f'  |  roll={math.degrees(float(state[7])):.1f} deg'
        if frame > 0 and frame - 1 < len(self.result.temperatures):
            lam = self.result.temperatures[frame - 1]
            ess = self.result.esses[frame - 1]
            feasible = int(self.result.feasible_counts[frame - 1])
            denominator = 1 if self.result.variant_value == 'planner_ilqr' else self.result.cfg.num_rollouts
            if math.isfinite(float(lam)):
                title += f'  |  lambda={lam:.3g}, ESS={ess:.1f}, feasible={feasible}/{denominator}'
            else:
                title += f'  |  feasible={feasible}/{denominator}'
        self.ax.set_title(title)
        self.frame_label_var.set(f'Frame {frame} / {len(self.result.states) - 1}')
        self.canvas.draw_idle()

    def _on_close(self) -> None:
        self.closing = True
        self._stop_animation()
        if self.poll_after_id is not None:
            try:
                self.root.after_cancel(self.poll_after_id)
            except tk.TclError:
                pass
        self.root.destroy()

def main() -> None:
    root = tk.Tk()
    NASCARViewer(root)
    root.mainloop()
if __name__ == '__main__':
    main()