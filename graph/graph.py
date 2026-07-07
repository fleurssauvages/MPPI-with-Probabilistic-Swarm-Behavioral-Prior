"""
Informed RRT* (2D) with:
- Polygon obstacles (convex or concave, non-self-intersecting)
- Fast nearest / near queries via scipy.spatial.cKDTree (rebuilt periodically)
- Numba-accelerated collision + geometry kernels
- Parallelized heavy scans (separator scoring + clearance + pruning-mark)

Dependencies:
  pip install numpy matplotlib scipy numba

Run:
  python RRTstar_reroute_numba_full.py
"""

import numpy as np
import matplotlib.pyplot as plt
import numba
import time
import heapq
from dataclasses import dataclass
from typing import List, Tuple
import math
from scipy.spatial import cKDTree
from numba import njit, prange

@dataclass
class PolyObstacle:
    vertices: np.ndarray  # (m,2)

def chaikin_closed(poly, n_iters=2):
    """
    Chaikin corner-cutting for a closed polygon.
    poly: (M,2)
    returns smoother closed polyline (K,2) (not explicitly closed)
    """
    p = np.asarray(poly, dtype=float)
    for _ in range(n_iters):
        p_next = []
        for i in range(len(p)):
            p0 = p[i]
            p1 = p[(i + 1) % len(p)]
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            p_next.extend([Q, R])
        p = np.asarray(p_next)
    return p

def resample_closed_polyline(polyline, n_points=16):
    """
    Evenly resample a closed polyline to n_points along arc-length.
    polyline: (M,2), closed implicitly
    returns: (n_points,2)
    """
    p = np.asarray(polyline, dtype=float)
    closed = np.vstack([p, p[0]])
    seg = closed[1:] - closed[:-1]
    L = np.linalg.norm(seg, axis=1)
    per = L.sum()
    cum = np.concatenate([[0.0], np.cumsum(L)])
    s = np.linspace(0.0, per, n_points + 1)[:-1]

    out = np.empty((n_points, 2), dtype=float)
    for k, sk in enumerate(s):
        i = np.searchsorted(cum, sk, side="right") - 1
        i = min(i, len(seg) - 1)
        if L[i] < 1e-12:
            out[k] = closed[i]
        else:
            a = (sk - cum[i]) / L[i]
            out[k] = closed[i] + a * seg[i]
    return out

def round_obstacle(poly, n_iters=2, n_points=16):
    smooth = chaikin_closed(poly, n_iters=n_iters)
    rounded = resample_closed_polyline(smooth, n_points=n_points)
    return rounded

# ============================================================
# NUMBA geometry
# ============================================================
@njit(cache=True, fastmath=True)
def orient2d(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


@njit(cache=True, fastmath=True)
def on_segment(a, b, p, eps=1e-12):
    if abs(orient2d(a, b, p)) > eps:
        return False
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


@njit(cache=True, fastmath=True)
def segments_intersect(a, b, c, d, eps=1e-12):
    o1 = orient2d(a, b, c)
    o2 = orient2d(a, b, d)
    o3 = orient2d(c, d, a)
    o4 = orient2d(c, d, b)

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    if abs(o1) <= eps and on_segment(a, b, c, eps):
        return True
    if abs(o2) <= eps and on_segment(a, b, d, eps):
        return True
    if abs(o3) <= eps and on_segment(c, d, a, eps):
        return True
    if abs(o4) <= eps and on_segment(c, d, b, eps):
        return True

    return False


@njit(cache=True, fastmath=True)
def segment_intersects_poly(a, b, poly):
    if point_in_poly_numba(a, poly) or point_in_poly_numba(b, poly):
        return True

    n = poly.shape[0]
    for i in range(n):
        j = (i + 1) % n
        if segments_intersect(a, b, poly[i], poly[j]):
            return True
    return False

@njit(cache=True, fastmath=True, inline="always")
def orient2d_xy(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True, fastmath=True, inline="always")
def on_segment_xy(ax, ay, bx, by, px, py, eps=1e-12):
    if abs(orient2d_xy(ax, ay, bx, by, px, py)) > eps:
        return False
    return (min(ax, bx) - eps <= px <= max(ax, bx) + eps and
            min(ay, by) - eps <= py <= max(ay, by) + eps)


@njit(cache=True, fastmath=True, inline="always")
def segments_intersect_xy(ax, ay, bx, by, cx, cy, dx, dy, eps=1e-12):
    o1 = orient2d_xy(ax, ay, bx, by, cx, cy)
    o2 = orient2d_xy(ax, ay, bx, by, dx, dy)
    o3 = orient2d_xy(cx, cy, dx, dy, ax, ay)
    o4 = orient2d_xy(cx, cy, dx, dy, bx, by)

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    if abs(o1) <= eps and on_segment_xy(ax, ay, bx, by, cx, cy, eps):
        return True
    if abs(o2) <= eps and on_segment_xy(ax, ay, bx, by, dx, dy, eps):
        return True
    if abs(o3) <= eps and on_segment_xy(cx, cy, dx, dy, ax, ay, eps):
        return True
    if abs(o4) <= eps and on_segment_xy(cx, cy, dx, dy, bx, by, eps):
        return True

    return False

@njit(cache=True, fastmath=True, inline="always")
def point_segment_distance_and_projection_nb(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    ab2 = abx * abx + aby * aby

    if ab2 < 1e-18:
        dx = px - ax
        dy = py - ay
        return np.sqrt(dx * dx + dy * dy), ax, ay

    t = ((px - ax) * abx + (py - ay) * aby) / ab2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    qx = ax + t * abx
    qy = ay + t * aby
    dx = px - qx
    dy = py - qy
    return np.sqrt(dx * dx + dy * dy), qx, qy


@njit(cache=True, fastmath=True)
def segment_segment_closest_points_nb(a0, a1, b0, b1):
    a0x, a0y = a0[0], a0[1]
    a1x, a1y = a1[0], a1[1]
    b0x, b0y = b0[0], b0[1]
    b1x, b1y = b1[0], b1[1]

    ux = a1x - a0x
    uy = a1y - a0y
    vx = b1x - b0x
    vy = b1y - b0y
    wx = a0x - b0x
    wy = a0y - b0y

    aa = ux * ux + uy * uy
    bb = ux * vx + uy * vy
    cc = vx * vx + vy * vy
    dd = ux * wx + uy * wy
    ee = vx * wx + vy * wy

    D = aa * cc - bb * bb
    eps = 1e-18

    best_pa = np.empty(2, dtype=np.float64)
    best_pb = np.empty(2, dtype=np.float64)
    best_d = 1e300

    if D > eps:
        s = (bb * ee - cc * dd) / D
        t = (aa * ee - bb * dd) / D

        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0

        pax = a0x + s * ux
        pay = a0y + s * uy
        pbx = b0x + t * vx
        pby = b0y + t * vy

        dx = pax - pbx
        dy = pay - pby
        best_d = np.sqrt(dx * dx + dy * dy)
        best_pa[0], best_pa[1] = pax, pay
        best_pb[0], best_pb[1] = pbx, pby

    # endpoint projections, more robust
    d, qx, qy = point_segment_distance_and_projection_nb(a0x, a0y, b0x, b0y, b1x, b1y)
    if d < best_d:
        best_d = d
        best_pa[0], best_pa[1] = a0x, a0y
        best_pb[0], best_pb[1] = qx, qy

    d, qx, qy = point_segment_distance_and_projection_nb(a1x, a1y, b0x, b0y, b1x, b1y)
    if d < best_d:
        best_d = d
        best_pa[0], best_pa[1] = a1x, a1y
        best_pb[0], best_pb[1] = qx, qy

    d, qx, qy = point_segment_distance_and_projection_nb(b0x, b0y, a0x, a0y, a1x, a1y)
    if d < best_d:
        best_d = d
        best_pa[0], best_pa[1] = qx, qy
        best_pb[0], best_pb[1] = b0x, b0y

    d, qx, qy = point_segment_distance_and_projection_nb(b1x, b1y, a0x, a0y, a1x, a1y)
    if d < best_d:
        best_d = d
        best_pa[0], best_pa[1] = qx, qy
        best_pb[0], best_pb[1] = b1x, b1y

    return best_pa, best_pb, best_d

@njit(cache=True, fastmath=True)
def segment_aabb_intersect_2d(a, b, bb_min, bb_max, eps=1e-12):
    d = b - a
    tmin = 0.0
    tmax = 1.0

    for k in range(2):
        if abs(d[k]) < eps:
            if a[k] < bb_min[k] or a[k] > bb_max[k]:
                return False
        else:
            inv_d = 1.0 / d[k]
            t1 = (bb_min[k] - a[k]) * inv_d
            t2 = (bb_max[k] - a[k]) * inv_d
            ta = t1 if t1 < t2 else t2
            tb = t2 if t1 < t2 else t1
            if ta > tmin:
                tmin = ta
            if tb < tmax:
                tmax = tb
            if tmin > tmax:
                return False
    return True


@njit(cache=True, fastmath=True)
def point_in_poly_numba(p: np.ndarray, poly: np.ndarray) -> bool:
    x = p[0]
    y = p[1]
    inside = False
    n = poly.shape[0]
    eps = 1e-12

    for i in range(n):
        j = (i + 1) % n
        x1 = poly[i, 0]
        y1 = poly[i, 1]
        x2 = poly[j, 0]
        y2 = poly[j, 1]

        vx = x2 - x1
        vy = y2 - y1
        wx = x - x1
        wy = y - y1
        cross = vx * wy - vy * wx

        if abs(cross) < eps:
            if (min(x1, x2) - eps <= x <= max(x1, x2) + eps and
                min(y1, y2) - eps <= y <= max(y1, y2) + eps):
                return True

        if (y1 > y) != (y2 > y):
            x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-18) + x1
            if x < x_int:
                inside = not inside

    return inside

@njit(cache=True, fastmath=True)
def point_valid_numba(p, polys):
    for k in range(len(polys)):
        if point_in_poly_numba(p, polys[k]):
            return False
    return True


@njit(cache=True, fastmath=True)
def segment_valid_numba_aabb(a, b, polys, bb_mins, bb_maxs):
    for k in range(len(polys)):
        if not segment_aabb_intersect_2d(a, b, bb_mins[k], bb_maxs[k]):
            continue
        if segment_intersects_poly(a, b, polys[k]):
            return False
    return True


@njit(cache=True, fastmath=True)
def winding_number_about_point(path, rep):
    total = 0.0
    for i in range(path.shape[0] - 1):
        x1 = path[i, 0] - rep[0]
        y1 = path[i, 1] - rep[1]
        x2 = path[i + 1, 0] - rep[0]
        y2 = path[i + 1, 1] - rep[1]
        total += math.atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)
    return int(np.rint(total / (2.0 * np.pi)))


@njit(cache=True, fastmath=True)
def homotopy_signature_numba(path, reps):
    out = np.empty(reps.shape[0], dtype=np.int64)
    for k in range(reps.shape[0]):
        out[k] = winding_number_about_point(path, reps[k])
    return out

@njit(cache=True, fastmath=True)
def segment_aabb_intersect_2d(a, b, bb_min, bb_max, eps=1e-12):
    d = b - a
    tmin = 0.0
    tmax = 1.0

    for k in range(2):
        if abs(d[k]) < eps:
            if a[k] < bb_min[k] or a[k] > bb_max[k]:
                return False
        else:
            inv_d = 1.0 / d[k]
            t1 = (bb_min[k] - a[k]) * inv_d
            t2 = (bb_max[k] - a[k]) * inv_d
            ta = t1 if t1 < t2 else t2
            tb = t2 if t1 < t2 else t1
            if ta > tmin:
                tmin = ta
            if tb < tmax:
                tmax = tb
            if tmin > tmax:
                return False
    return True
# ============================================================
# Helpers
# ============================================================

def preprocess_obstacles(obstacles):
    polys = [np.asarray(obs.vertices, dtype=np.float64) for obs in obstacles]
    bb_mins = np.asarray([p.min(axis=0) for p in polys], dtype=np.float64)
    bb_maxs = np.asarray([p.max(axis=0) for p in polys], dtype=np.float64)
    return polys, bb_mins, bb_maxs

def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    x = poly[:, 0]
    y = poly[:, 1]
    xp = np.roll(x, -1)
    yp = np.roll(y, -1)
    cross = x * yp - xp * y
    A = 0.5 * np.sum(cross)

    if abs(A) < 1e-12:
        return np.mean(poly, axis=0)

    cx = np.sum((x + xp) * cross) / (6.0 * A)
    cy = np.sum((y + yp) * cross) / (6.0 * A)
    return np.array([cx, cy], dtype=np.float64)


def obstacle_representatives(obstacles: List[PolyObstacle]) -> np.ndarray:
    return np.asarray(
        [polygon_centroid(np.asarray(obs.vertices, dtype=np.float64)) for obs in obstacles],
        dtype=np.float64
    )


def deduplicate_points(points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    if len(points) == 0:
        return points
    q = np.round(points / tol).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    idx = np.sort(idx)
    return points[idx]


def poly_edges(poly: np.ndarray):
    n = poly.shape[0]
    for i in range(n):
        yield poly[i], poly[(i + 1) % n]


@njit(cache=True, fastmath=True, inline="always")
def point_segment_distance_and_projection_nb(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    ab2 = abx * abx + aby * aby

    if ab2 < 1e-18:
        dx = px - ax
        dy = py - ay
        return np.sqrt(dx * dx + dy * dy), ax, ay

    t = ((px - ax) * abx + (py - ay) * aby) / ab2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    qx = ax + t * abx
    qy = ay + t * aby
    dx = px - qx
    dy = py - qy
    return np.sqrt(dx * dx + dy * dy), qx, qy


@njit(cache=True, fastmath=True)
def segment_segment_closest_points_nb(a0, a1, b0, b1):
    a0x, a0y = a0[0], a0[1]
    a1x, a1y = a1[0], a1[1]
    b0x, b0y = b0[0], b0[1]
    b1x, b1y = b1[0], b1[1]

    ux = a1x - a0x
    uy = a1y - a0y
    vx = b1x - b0x
    vy = b1y - b0y
    wx = a0x - b0x
    wy = a0y - b0y

    aa = ux * ux + uy * uy
    bb = ux * vx + uy * vy
    cc = vx * vx + vy * vy
    dd = ux * wx + uy * wy
    ee = vx * wx + vy * wy

    D = aa * cc - bb * bb
    eps = 1e-18

    best_pa = np.empty(2, dtype=np.float64)
    best_pb = np.empty(2, dtype=np.float64)
    best_d = 1e300

    def try_pair(px, py, qx, qy):
        return np.sqrt((px - qx) ** 2 + (py - qy) ** 2)

    if D < eps:
        d, qx, qy = point_segment_distance_and_projection_nb(a0x, a0y, b0x, b0y, b1x, b1y)
        if d < best_d:
            best_d = d
            best_pa[0], best_pa[1] = a0x, a0y
            best_pb[0], best_pb[1] = qx, qy

        d, qx, qy = point_segment_distance_and_projection_nb(a1x, a1y, b0x, b0y, b1x, b1y)
        if d < best_d:
            best_d = d
            best_pa[0], best_pa[1] = a1x, a1y
            best_pb[0], best_pb[1] = qx, qy

        d, qx, qy = point_segment_distance_and_projection_nb(b0x, b0y, a0x, a0y, a1x, a1y)
        if d < best_d:
            best_d = d
            best_pa[0], best_pa[1] = qx, qy
            best_pb[0], best_pb[1] = b0x, b0y

        d, qx, qy = point_segment_distance_and_projection_nb(b1x, b1y, a0x, a0y, a1x, a1y)
        if d < best_d:
            best_d = d
            best_pa[0], best_pa[1] = qx, qy
            best_pb[0], best_pb[1] = b1x, b1y

        return best_pa, best_pb, best_d

    s = (bb * ee - cc * dd) / D
    t = (aa * ee - bb * dd) / D

    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    pax = a0x + s * ux
    pay = a0y + s * uy
    pbx = b0x + t * vx
    pby = b0y + t * vy

    # refine with endpoint projections
    candidates = np.empty((6, 5), dtype=np.float64)
    k = 0

    d, qx, qy = point_segment_distance_and_projection_nb(pax, pay, b0x, b0y, b1x, b1y)
    candidates[k] = np.array([d, pax, pay, qx, qy]); k += 1
    d, qx, qy = point_segment_distance_and_projection_nb(pbx, pby, a0x, a0y, a1x, a1y)
    candidates[k] = np.array([d, qx, qy, pbx, pby]); k += 1
    d, qx, qy = point_segment_distance_and_projection_nb(a0x, a0y, b0x, b0y, b1x, b1y)
    candidates[k] = np.array([d, a0x, a0y, qx, qy]); k += 1
    d, qx, qy = point_segment_distance_and_projection_nb(a1x, a1y, b0x, b0y, b1x, b1y)
    candidates[k] = np.array([d, a1x, a1y, qx, qy]); k += 1
    d, qx, qy = point_segment_distance_and_projection_nb(b0x, b0y, a0x, a0y, a1x, a1y)
    candidates[k] = np.array([d, qx, qy, b0x, b0y]); k += 1
    d, qx, qy = point_segment_distance_and_projection_nb(b1x, b1y, a0x, a0y, a1x, a1y)
    candidates[k] = np.array([d, qx, qy, b1x, b1y])

    best = 0
    for i in range(1, 6):
        if candidates[i, 0] < candidates[best, 0]:
            best = i

    best_pa[0], best_pa[1] = candidates[best, 1], candidates[best, 2]
    best_pb[0], best_pb[1] = candidates[best, 3], candidates[best, 4]
    best_d = candidates[best, 0]
    return best_pa, best_pb, best_d

@njit(cache=True, fastmath=True, inline="always")
def get_bound_segment(bound_id, bmin, bmax, q0, q1):
    if bound_id == 0:   # left
        q0[0], q0[1] = bmin[0], bmin[1]
        q1[0], q1[1] = bmin[0], bmax[1]
    elif bound_id == 1: # right
        q0[0], q0[1] = bmax[0], bmin[1]
        q1[0], q1[1] = bmax[0], bmax[1]
    elif bound_id == 2: # bottom
        q0[0], q0[1] = bmin[0], bmin[1]
        q1[0], q1[1] = bmax[0], bmin[1]
    else:               # top
        q0[0], q0[1] = bmin[0], bmax[1]
        q1[0], q1[1] = bmax[0], bmax[1]

# ============================================================
# Shortest valid corridor segments
# ============================================================
@njit(cache=True, fastmath=True)
def point_in_poly_packed(px, py, poly, nv):
    inside = False
    j = nv - 1
    for i in range(nv):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]

        intersect = ((yi > py) != (yj > py))
        if intersect:
            x_cross = xi + (py - yi) * (xj - xi) / ((yj - yi) + 1e-18)
            if px < x_cross:
                inside = not inside
        j = i
    return inside

@njit(cache=True, inline="always")
def aabb_overlap(min1x, min1y, max1x, max1y, min2x, min2y, max2x, max2y):
    return not (max1x < min2x or max2x < min1x or max1y < min2y or max2y < min1y)

@njit(cache=True, inline="always")
def aabb_dist_sq(min1x, min1y, max1x, max1y, min2x, min2y, max2x, max2y):
    dx = 0.0
    if max1x < min2x:
        dx = min2x - max1x
    elif max2x < min1x:
        dx = min1x - max2x

    dy = 0.0
    if max1y < min2y:
        dy = min2y - max1y
    elif max2y < min1y:
        dy = min1y - max2y

    return dx * dx + dy * dy

@njit(cache=True)
def build_obstacle_bboxes(verts, nverts):
    n_obs = verts.shape[0]
    bb_mins = np.empty((n_obs, 2), dtype=np.float64)
    bb_maxs = np.empty((n_obs, 2), dtype=np.float64)

    for i in range(n_obs):
        nv = nverts[i]
        minx = verts[i, 0, 0]
        miny = verts[i, 0, 1]
        maxx = minx
        maxy = miny
        for k in range(1, nv):
            x = verts[i, k, 0]
            y = verts[i, k, 1]
            if x < minx: minx = x
            if y < miny: miny = y
            if x > maxx: maxx = x
            if y > maxy: maxy = y
        bb_mins[i, 0] = minx
        bb_mins[i, 1] = miny
        bb_maxs[i, 0] = maxx
        bb_maxs[i, 1] = maxy

    return bb_mins, bb_maxs

@njit(cache=True)
def segment_valid_against_other_obstacles(pa, pb, verts, nverts, bb_mins, bb_maxs, skip_a, skip_b):
    ax, ay = pa[0], pa[1]
    bx, by = pb[0], pb[1]

    seg_minx = ax if ax < bx else bx
    seg_miny = ay if ay < by else by
    seg_maxx = ax if ax > bx else bx
    seg_maxy = ay if ay > by else by

    midx = 0.5 * (ax + bx)
    midy = 0.5 * (ay + by)

    for k in range(verts.shape[0]):
        if k == skip_a or k == skip_b:
            continue

        ominx = bb_mins[k, 0]
        ominy = bb_mins[k, 1]
        omaxx = bb_maxs[k, 0]
        omaxy = bb_maxs[k, 1]

        # cheap reject
        if not aabb_overlap(seg_minx, seg_miny, seg_maxx, seg_maxy,
                            ominx, ominy, omaxx, omaxy):
            continue

        poly = verts[k]
        nv = nverts[k]

        # midpoint inside => invalid
        if point_in_poly_packed(midx, midy, poly, nv):
            return False

        # exact edge intersections only for overlapping AABBs
        for i in range(nv):
            j = 0 if i + 1 == nv else i + 1
            cx, cy = poly[i, 0], poly[i, 1]
            dx, dy = poly[j, 0], poly[j, 1]

            emin_x = cx if cx < dx else dx
            emin_y = cy if cy < dy else dy
            emax_x = cx if cx > dx else dx
            emax_y = cy if cy > dy else dy

            if not aabb_overlap(seg_minx, seg_miny, seg_maxx, seg_maxy,
                                emin_x, emin_y, emax_x, emax_y):
                continue

            if segments_intersect_xy(ax, ay, bx, by, cx, cy, dx, dy):
                return False

    return True

@njit(cache=True)
def shortest_valid_segment_between_obstacles_nb(verts, nverts, bb_mins, bb_maxs, ia, ib):
    poly_a = verts[ia]
    poly_b = verts[ib]
    nva = nverts[ia]
    nvb = nverts[ib]

    best_pa = np.empty(2, dtype=np.float64)
    best_pb = np.empty(2, dtype=np.float64)
    best_d2 = 1e300
    found = False

    # obstacle-level lower bound
    if aabb_dist_sq(bb_mins[ia,0], bb_mins[ia,1], bb_maxs[ia,0], bb_maxs[ia,1],
                    bb_mins[ib,0], bb_mins[ib,1], bb_maxs[ib,0], bb_maxs[ib,1]) >= best_d2:
        return found, best_pa, best_pb

    for ea in range(nva):
        ea2 = 0 if ea + 1 == nva else ea + 1
        a0 = poly_a[ea]
        a1 = poly_a[ea2]

        aminx = a0[0] if a0[0] < a1[0] else a1[0]
        aminy = a0[1] if a0[1] < a1[1] else a1[1]
        amaxx = a0[0] if a0[0] > a1[0] else a1[0]
        amaxy = a0[1] if a0[1] > a1[1] else a1[1]

        for eb in range(nvb):
            eb2 = 0 if eb + 1 == nvb else eb + 1
            b0 = poly_b[eb]
            b1 = poly_b[eb2]

            bminx = b0[0] if b0[0] < b1[0] else b1[0]
            bminy = b0[1] if b0[1] < b1[1] else b1[1]
            bmaxx = b0[0] if b0[0] > b1[0] else b1[0]
            bmaxy = b0[1] if b0[1] > b1[1] else b1[1]

            # edge-level lower bound
            if aabb_dist_sq(aminx, aminy, amaxx, amaxy,
                            bminx, bminy, bmaxx, bmaxy) >= best_d2:
                continue

            pa, pb, d = segment_segment_closest_points_nb(a0, a1, b0, b1)
            d2 = d * d
            if d2 >= best_d2:
                continue

            if not segment_valid_against_other_obstacles(pa, pb, verts, nverts, bb_mins, bb_maxs, ia, ib):
                continue

            best_d2 = d2
            best_pa[0], best_pa[1] = pa[0], pa[1]
            best_pb[0], best_pb[1] = pb[0], pb[1]
            found = True

    return found, best_pa, best_pb

@njit(cache=True)
def shortest_valid_segment_obstacle_to_bound_nb(
    verts, nverts, bb_mins, bb_maxs, obs_idx, bmin, bmax, bound_id
):
    """
    Returns:
        found : bool
        best_pa : point on obstacle boundary
        best_pb : point on chosen workspace boundary
    """
    poly = verts[obs_idx]
    nv = nverts[obs_idx]

    q0 = np.empty(2, dtype=np.float64)
    q1 = np.empty(2, dtype=np.float64)
    get_bound_segment(bound_id, bmin, bmax, q0, q1)

    best_pa = np.empty(2, dtype=np.float64)
    best_pb = np.empty(2, dtype=np.float64)
    best_d2 = 1e300
    found = False

    # boundary segment AABB
    qminx = q0[0] if q0[0] < q1[0] else q1[0]
    qminy = q0[1] if q0[1] < q1[1] else q1[1]
    qmaxx = q0[0] if q0[0] > q1[0] else q1[0]
    qmaxy = q0[1] if q0[1] > q1[1] else q1[1]

    for ei in range(nv):
        ej = 0 if ei + 1 == nv else ei + 1

        a0 = poly[ei]
        a1 = poly[ej]

        # edge AABB
        aminx = a0[0] if a0[0] < a1[0] else a1[0]
        aminy = a0[1] if a0[1] < a1[1] else a1[1]
        amaxx = a0[0] if a0[0] > a1[0] else a1[0]
        amaxy = a0[1] if a0[1] > a1[1] else a1[1]

        # cheap lower bound: if already worse than best, skip
        lb2 = aabb_dist_sq(aminx, aminy, amaxx, amaxy, qminx, qminy, qmaxx, qmaxy)
        if lb2 >= best_d2:
            continue

        pa, pb, d = segment_segment_closest_points_nb(a0, a1, q0, q1)
        d2 = d * d
        if d2 >= best_d2:
            continue

        # validate corridor against all other obstacles
        if not segment_valid_against_other_obstacles(
            pa, pb, verts, nverts, bb_mins, bb_maxs, obs_idx, -1
        ):
            continue

        best_d2 = d2
        best_pa[0], best_pa[1] = pa[0], pa[1]
        best_pb[0], best_pb[1] = pb[0], pb[1]
        found = True

    return found, best_pa, best_pb

# ============================================================
# Node generation: midpoint of shortest valid corridor segments
# ============================================================
def pack_obstacles(obstacles):
    polys = [np.asarray(obs.vertices, dtype=np.float64) for obs in obstacles]
    n_obs = len(polys)
    nverts = np.array([p.shape[0] for p in polys], dtype=np.int64)
    max_nv = int(nverts.max())

    verts = np.empty((n_obs, max_nv, 2), dtype=np.float64)
    verts[:] = np.nan
    for i, p in enumerate(polys):
        verts[i, :p.shape[0], :] = p
    return verts, nverts

@njit(cache=True, fastmath=True)
def point_on_polygon_boundary_packed(px, py, poly, nv, eps=1e-10):
    for i in range(nv):
        j = 0 if i + 1 == nv else i + 1
        ax, ay = poly[i, 0], poly[i, 1]
        bx, by = poly[j, 0], poly[j, 1]

        # distance to segment
        d, _, _ = point_segment_distance_and_projection_nb(px, py, ax, ay, bx, by)
        if d <= eps:
            return True
    return False

@njit(cache=True, fastmath=True)
def point_valid_packed(p, verts, nverts):
    """
    True if point is not strictly inside any obstacle.
    Boundary points are accepted.
    """
    px, py = p[0], p[1]

    for obs_idx in range(verts.shape[0]):
        nv = nverts[obs_idx]
        poly = verts[obs_idx]

        if point_on_polygon_boundary_packed(px, py, poly, nv):
            continue

        if point_in_poly_packed(px, py, poly, nv):
            return False

    return True

@njit(cache=True)
def build_shortest_segment_midpoint_nodes(verts, nverts, bmin, bmax, start, goal, keep_only_forward=True):
    n_obs = verts.shape[0]
    bb_mins, bb_maxs = build_obstacle_bboxes(verts, nverts)

    max_nodes = 2 + (n_obs * (n_obs - 1)) // 2 + 2 * n_obs
    nodes = np.empty((max_nodes, 2), dtype=np.float64)
    m = 0

    nodes[m] = start; m += 1
    nodes[m] = goal;  m += 1

    # obstacle-obstacle
    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            found, pa, pb = shortest_valid_segment_between_obstacles_nb(
                verts, nverts, bb_mins, bb_maxs, i, j
            )
            if found:
                nodes[m, 0] = 0.5 * (pa[0] + pb[0])
                nodes[m, 1] = 0.5 * (pa[1] + pb[1])
                m += 1

    # obstacle-bound: ideally only 2 nearest bounds per obstacle
    for i in range(n_obs):
        # left/right/bottom/top distances from bbox
        d = np.array([
            bb_mins[i,0] - bmin[0],
            bmax[0] - bb_maxs[i,0],
            bb_mins[i,1] - bmin[1],
            bmax[1] - bb_maxs[i,1]
        ])
        order = np.argsort(d)[:2]

        for t in range(order.shape[0]):
            bound_id = int(order[t])
            found, pa, pb = shortest_valid_segment_obstacle_to_bound_nb(
                verts, nverts, bb_mins, bb_maxs, i, bmin, bmax, bound_id
            )
            if found:
                nodes[m, 0] = 0.5 * (pa[0] + pb[0])
                nodes[m, 1] = 0.5 * (pa[1] + pb[1])
                m += 1

    nodes = nodes[:m]

    keep = np.empty(m, dtype=np.bool_)
    for i in range(m):
        p = nodes[i]
        keep[i] = (
            p[0] >= bmin[0] and p[0] <= bmax[0] and
            p[1] >= bmin[1] and p[1] <= bmax[1] and
            point_valid_packed(p, verts, nverts)
        )

    count = 0
    for i in range(m):
        if keep[i]:
            nodes[count] = nodes[i]
            count += 1
    nodes = nodes[:count]

    if keep_only_forward:
        gx = goal[0] - start[0]
        gy = goal[1] - start[1]
        gnorm = np.sqrt(gx * gx + gy * gy) + 1e-18
        gdx = gx / gnorm
        gdy = gy / gnorm

        count2 = 0
        for i in range(nodes.shape[0]):
            dx = nodes[i, 0] - start[0]
            dy = nodes[i, 1] - start[1]
            proj = dx * gdx + dy * gdy
            if -1e-12 <= proj <= gnorm + 1e-12:
                nodes[count2] = nodes[i]
                count2 += 1
        nodes = nodes[:count2]

    return nodes

def prune_close_nodes(
    nodes: np.ndarray,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    min_dist: float,
    proj_tol: float | None = None,
    keep_endpoints: bool = True,
) -> np.ndarray:
    """
    Remove nodes that are too close to each other.

    Parameters
    ----------
    nodes : (N,2) array
    start, goal : used to compute forward projection
    min_dist : minimum Euclidean distance between kept nodes
    proj_tol : if not None, only merge nodes that are also close in goal projection
    keep_endpoints : keep start and goal even if close to others

    Assumes start and goal are present in nodes.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    gvec = goal - start
    gnorm = np.linalg.norm(gvec)
    gdir = gvec / (gnorm + 1e-18)
    proj = (nodes - start[None, :]) @ gdir

    keep = np.ones(len(nodes), dtype=bool)

    # assume start and goal are the first two if keep_endpoints=True
    protected = set()
    if keep_endpoints and len(nodes) >= 2:
        protected.add(0)
        protected.add(1)

    # sort by projection so we keep a forward-spread set
    order = np.argsort(proj)

    kept_ids = []
    for idx in order:
        if idx in protected:
            kept_ids.append(idx)
            continue

        p = nodes[idx]
        ok = True

        for j in kept_ids:
            q = nodes[j]

            if np.linalg.norm(p - q) < min_dist:
                if proj_tol is None or abs(proj[idx] - proj[j]) < proj_tol:
                    ok = False
                    break

        if ok:
            kept_ids.append(idx)

    kept_ids = np.array(sorted(kept_ids), dtype=np.int64)
    return nodes[kept_ids]

# ============================================================
# Graph build, same logic as before
# ============================================================

def build_graph_from_nodes(
    nodes: np.ndarray,
    obstacles: List[PolyObstacle],
    start: Tuple[float, float],
    goal: Tuple[float, float],
    forward_tolerance: float = 0.0,
    max_neighbors: int | None = None,
    max_dist: float | None = None,
    query_factor: int = 4,
):
    polys, bb_mins, bb_maxs = preprocess_obstacles(obstacles)
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    # rebuild with start and goal first
    pts = [start, goal]
    for p in nodes:
        if np.linalg.norm(p - start) < 1e-12:
            continue
        if np.linalg.norm(p - goal) < 1e-12:
            continue
        pts.append(p)
    points = np.asarray(pts, dtype=np.float64)

    start_id = 0
    goal_id = 1

    gvec = goal - start
    gnorm = np.linalg.norm(gvec)
    gdir = gvec / (gnorm + 1e-18)
    proj = (points - start[None, :]) @ gdir

    n = len(points)
    adj = [[] for _ in range(n)]
    edge_cost_map = {}

    tree = None
    candidate_lists = None

    if max_neighbors is not None and max_neighbors < n - 1:
        tree = cKDTree(points)
        candidate_lists = []

        # query more than max_neighbors because some will be discarded
        k_query = min(query_factor * max_neighbors + 1, n)

        for i in range(n):
            dists, idx = tree.query(points[i], k=k_query)
            idx = np.atleast_1d(idx)
            dists = np.atleast_1d(dists)

            keep = []
            for j, d in zip(idx, dists):
                if j == i:
                    continue
                if proj[j] < proj[i] - forward_tolerance:
                    continue
                if max_dist is not None and d > max_dist:
                    continue
                keep.append(int(j))
                if len(keep) >= max_neighbors:
                    break

            candidate_lists.append(np.asarray(keep, dtype=np.int64))

    for i in range(n):
        a = points[i]

        if candidate_lists is None:
            js = range(n)
        else:
            js = candidate_lists[i]

        for j in js:
            if i == j:
                continue

            # needed only in the all-pairs branch
            if candidate_lists is None:
                if proj[j] < proj[i] - forward_tolerance:
                    continue

            b = points[j]

            dist = np.linalg.norm(b - a)
            if max_dist is not None and dist > max_dist:
                continue

            if not segment_valid_numba_aabb(a, b, polys, bb_mins, bb_maxs):
                continue

            adj[i].append((j, float(dist)))
            edge_cost_map[(i, j)] = float(dist)

    return {
        "points": points,
        "adj": adj,
        "edge_cost_map": edge_cost_map,
        "start_id": start_id,
        "goal_id": goal_id,
    }

# ============================================================
# Plot Graph
# ============================================================
def plot_graph(graph, obstacles, start, goal, bounds,
               show_nodes=True,
               show_edges=True,
               edge_alpha=0.3,
               node_size=20, ax=None):
    
    if graph is not None:
        points = graph["points"]
        adj = graph["adj"]
    else:
        points = []
        adj = []

    # fig, ax = plt.subplots(figsize=(8,8))

    # -----------------------------
    # obstacles
    # -----------------------------
    for obs in obstacles:
        poly = obs.vertices
        poly = np.vstack([poly, poly[0]])
        ax.plot(poly[:,0], poly[:,1], color="black", linewidth=2)
        ax.fill(poly[:,0], poly[:,1], color="gray", alpha=0.3)

    if graph is not None:
        # -----------------------------
        # edges
        # -----------------------------
        if show_edges:
            for i, edges in enumerate(adj):
                p = points[i]
                for j, _ in edges:
                    q = points[j]
                    ax.plot([p[0], q[0]], [p[1], q[1]],
                            color="blue",
                            alpha=edge_alpha,
                            linewidth=0.8)

        # -----------------------------
        # nodes
        # -----------------------------
        if show_nodes:
            ax.scatter(points[:,0], points[:,1],
                    s=node_size,
                    color="red",
                    zorder=5)

    # -----------------------------
    # start / goal
    # -----------------------------
    ax.scatter(start[0], start[1], s=120, marker="o")
    ax.scatter(goal[0], goal[1], s=150, marker="*")

    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])

    ax.set_aspect("equal")

def plot_routes(graph, routes, ax=None,
                colors=None,
                linewidth=3.0,
                alpha=0.95,
                show_route_nodes=False,
                route_node_size=40,
                show_labels=True,
                cmap_name="tab10"):
    """
    Plot routes on top of an existing graph plot.

    Parameters
    ----------
    graph : dict
        Graph returned by build_graph_from_nodes / build_graph
    routes : list
        Can be one of:
          - list of node-id paths: [[0, 5, 9, 1], [0, 3, 7, 1], ...]
          - list of dicts with key "path_nodes":
                [{"path_nodes": [...]}, ...]
    ax : matplotlib axis or None
        If None, uses current axis (plt.gca()).
    colors : list or None
        Optional list of colors, one per route.
    linewidth : float
    alpha : float
    show_route_nodes : bool
    route_node_size : float
    show_labels : bool
        Label routes as r0, r1, ...
    cmap_name : str
        Colormap used if colors is None.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    points = np.asarray(graph["points"], dtype=np.float64)

    if ax is None:
        ax = plt.gca()

    if routes is None or len(routes) == 0:
        return ax

    # normalize input
    path_list = []
    for r in routes:
        if isinstance(r, dict):
            if "path_nodes" not in r:
                raise ValueError("Route dict must contain key 'path_nodes'")
            path = r["path_nodes"]
        else:
            path = r
        path_list.append(np.asarray(path, dtype=np.int64))

    # colors
    if colors is None:
        cmap = plt.get_cmap(cmap_name, max(len(path_list), 1))
        colors = [cmap(i) for i in range(len(path_list))]

    for i, path in enumerate(path_list):
        if path.size == 0:
            continue

        xy = points[path]

        ax.plot(
            xy[:, 0], xy[:, 1],
            color=colors[i % len(colors)],
            linewidth=linewidth,
            alpha=alpha,
            zorder=20,
            solid_capstyle="round",
        )

        if show_route_nodes:
            ax.scatter(
                xy[:, 0], xy[:, 1],
                s=route_node_size,
                color=[colors[i % len(colors)]],
                zorder=21,
            )

        # if show_labels and xy.shape[0] > 0:
        #     mid = xy.shape[0] // 2
        #     ax.text(
        #         xy[mid, 0], xy[mid, 1],
        #         f"r{i}",
        #         color=colors[i % len(colors)],
        #         fontsize=20,
        #         ha="center", va="center",
        #         zorder=22,
        #         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        #     )

    return ax

def build_graph(obstacles, start, goal, bounds):
    verts, nverts = pack_obstacles(obstacles)
    bmin = np.array([bounds[0][0], bounds[0][1]], dtype=np.float64)
    bmax = np.array([bounds[1][0], bounds[1][1]], dtype=np.float64)
  
    nodes = build_shortest_segment_midpoint_nodes(
        verts, nverts, bmin, bmax, start, goal, keep_only_forward=True
    )
    
    nodes = prune_close_nodes(
        nodes=nodes,
        start=start,
        goal=goal,
        min_dist=2.0,
        proj_tol=None,
        keep_endpoints=True,
    )

    graph = build_graph_from_nodes(
        nodes=nodes,
        obstacles=obstacles,
        start=start,
        goal=goal,
        max_neighbors=8,
    )

    return graph

# ============================================================
# Find shortest paths
# ============================================================
def graph_to_csr(graph):
    adj = graph["adj"]
    n = len(adj)

    row_ptr = np.zeros(n + 1, dtype=np.int32)
    nnz = sum(len(x) for x in adj)

    col = np.empty(nnz, dtype=np.int32)
    w = np.empty(nnz, dtype=np.float64)

    k = 0
    for i in range(n):
        row_ptr[i] = k
        for j, c in adj[i]:
            col[k] = int(j)
            w[k] = float(c)
            k += 1
    row_ptr[n] = k
    return row_ptr, col, w


def shortest_path_csr(row_ptr, col, w, start, goal):
    n = len(row_ptr) - 1
    dist = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int32)
    parent_edge = np.full(n, -1, dtype=np.int32)

    dist[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == goal:
            break

        for e in range(row_ptr[u], row_ptr[u + 1]):
            v = col[e]
            nd = d + w[e]
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                parent_edge[v] = e
                heapq.heappush(pq, (nd, v))

    if not np.isfinite(dist[goal]):
        return None, None, np.inf

    path = []
    edges = []
    u = goal
    while u != -1:
        path.append(int(u))
        e = parent_edge[u]
        if e != -1:
            edges.append(int(e))
        u = parent[u]

    path.reverse()
    edges.reverse()
    return path, edges, float(dist[goal])


def extract_k_routes_fast(graph, K=5, edge_penalty=1000.0, node_penalty=200.0,
                          forbid_reuse=False, protect_start_goal=True):
    row_ptr, col, base_w = graph_to_csr(graph)
    w = base_w.copy()

    start = int(graph["start_id"])
    goal = int(graph["goal_id"])

    routes = []

    for _ in range(K):
        path, path_edges, cost = shortest_path_csr(row_ptr, col, w, start, goal)
        if path is None:
            break

        routes.append({
            "path_nodes": path,
            "cost": cost,
        })

        # discourage reusing the same route
        if forbid_reuse:
            for e in path_edges:
                w[e] = np.inf
        else:
            for e in path_edges:
                w[e] += edge_penalty

            for node in path[1:-1]:
                if protect_start_goal and (node == start or node == goal):
                    continue

                # penalize all outgoing edges of reused intermediate nodes
                for e in range(row_ptr[node], row_ptr[node + 1]):
                    w[e] += node_penalty

    return routes

def routes_to_xy(graph, routes):
    """
    Convert routes returned by extract_k_routes_fast to XY polylines.
    """
    pts = np.asarray(graph["points"], dtype=np.float64)

    xy_routes = []
    for r in routes:
        if isinstance(r, dict):
            nodes = r["path_nodes"]
        else:
            nodes = r

        xy_routes.append(pts[np.asarray(nodes, dtype=np.int32)])

    return xy_routes

@njit(cache=True, fastmath=True)
def _costmap_lookup_bilinear_nb(costmap, xmin, xmax, ymin, ymax, x, y, outside_cost):
    H, W = costmap.shape  # H=ny, W=nx

    if x < xmin or x > xmax or y < ymin or y > ymax:
        return outside_cost

    # map world -> index space
    u = (x - xmin) / (xmax - xmin) * (W - 1.0)
    v = (y - ymin) / (ymax - ymin) * (H - 1.0)

    x0 = int(np.floor(u))
    y0 = int(np.floor(v))
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    x1 = x0 + 1
    y1 = y0 + 1
    if x1 >= W:
        x1 = W - 1
    if y1 >= H:
        y1 = H - 1

    fu = u - x0
    fv = v - y0

    c00 = costmap[y0, x0]
    c10 = costmap[y0, x1]
    c01 = costmap[y1, x0]
    c11 = costmap[y1, x1]

    return (1.0 - fu) * (1.0 - fv) * c00 + fu * (1.0 - fv) * c10 + (1.0 - fu) * fv * c01 + fu * fv * c11


@njit(cache=True, fastmath=True)
def _segment_integral_midpoint(costmap, xmin, xmax, ymin, ymax, ax, ay, bx, by, outside_cost):
    dx = bx - ax
    dy = by - ay
    ds = (dx * dx + dy * dy) ** 0.5
    if ds <= 1e-12:
        return 0.0, 0.0

    mx = 0.5 * (ax + bx)
    my = 0.5 * (ay + by)
    c = _costmap_lookup_bilinear_nb(costmap, xmin, xmax, ymin, ymax, mx, my, outside_cost)
    return c * ds, ds


@njit(parallel=True, cache=True, fastmath=True)
def _fish_costs_numba(trajs, goal, goal_radius, costmap,
                      xmin, xmax, ymin, ymax,
                      w_len, w_grid, keep_last_outside, outside_cost):
    """
    trajs: (T,N,2) or (T,N,D) but below assumes D==2 for speed.
    Returns Js (N,)
    """
    T, N, D = trajs.shape
    Js = np.empty(N, dtype=np.float64)
    r2 = goal_radius * goal_radius
    gx = goal[0]
    gy = goal[1]

    for j in prange(N):
        # find first index inside goal
        cut_idx = -1
        for i in range(T):
            dx = trajs[i, j, 0] - gx
            dy = trajs[i, j, 1] - gy
            if dx * dx + dy * dy <= r2:
                cut_idx = i
                break

        # Determine last kept point index in original trajectory after snapping logic
        # Original snap:
        # - if any inside: keep p[:first_in]
        # - else: keep all
        if cut_idx >= 0:
            last_kept = cut_idx - 1
        else:
            last_kept = T - 1

        if last_kept < 0:
            # snapped would be [goal] only => no segments
            Js[j] = 0.0
            continue

        if not keep_last_outside:
            last_kept -= 1
            if last_kept < 0:
                Js[j] = 0.0
                continue

        total_int = 0.0
        total_len = 0.0

        # integrate along kept polyline: 0..last_kept
        for i in range(last_kept):
            ax = trajs[i, j, 0]
            ay = trajs[i, j, 1]
            bx = trajs[i + 1, j, 0]
            by = trajs[i + 1, j, 1]
            seg_int, seg_len = _segment_integral_midpoint(
                costmap, xmin, xmax, ymin, ymax, ax, ay, bx, by, outside_cost
            )
            total_int += seg_int
            total_len += seg_len

        # final segment from last kept point to goal (because snap appends goal)
        ax = trajs[last_kept, j, 0]
        ay = trajs[last_kept, j, 1]
        seg_int, seg_len = _segment_integral_midpoint(
            costmap, xmin, xmax, ymin, ymax, ax, ay, gx, gy, outside_cost
        )
        total_int += seg_int
        total_len += seg_len

        Js[j] = w_len * total_len + w_grid * total_int

    return Js

def _snap_to_goal(p, goal, goal_radius, keep_last_outside=True):
    """Same trimming logic as your original function, then append goal."""
    d = np.linalg.norm(p - goal, axis=1)
    inside = d <= goal_radius
    if np.any(inside):
        first_in = int(np.argmax(inside))  # first True
        p_keep = p[:first_in, :]
    else:
        p_keep = p.copy()

    if p_keep.shape[0] == 0:
        return goal[None, :].copy()

    if not keep_last_outside:
        p_keep = p_keep[:-1, :]
        if p_keep.shape[0] == 0:
            return goal[None, :].copy()

    return np.vstack([p_keep, goal])


def best_fish_trajectory_same_cost_as_ompl(
    trajs: np.ndarray,
    goal: np.ndarray,
    goal_radius: float,
    costmap: np.ndarray,
    bounds=((0.0, 10.0), (0.0, 10.0)),
    w_len: float = 1.0,
    w_grid: float = 2.0,
    keep_last_outside: bool = True,
    outside_cost: float = 1000.0,
):
    """
    Fast: computes Js in parallel with numba.
    Returns best_idx, best_J, best_traj_snapped, all_J
    """
    trajs = np.asarray(trajs, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
    costmap = np.asarray(costmap, dtype=np.float64)

    if trajs.ndim != 3:
        raise ValueError(f"Expected trajs shape (T,N,D), got {trajs.shape}")
    T, N, D = trajs.shape
    if D != 2:
        raise ValueError("This numba path is written for D==2 (x,y). If you need D>2, say so.")
    if goal.shape[0] != 2:
        raise ValueError("goal must be shape (2,) for this numba version.")
    if costmap.ndim != 2:
        raise ValueError(f"Expected costmap shape (ny,nx), got {costmap.shape}")

    (xmin, xmax), (ymin, ymax) = bounds

    Js = _fish_costs_numba(
        trajs, goal, float(goal_radius), costmap,
        float(xmin), float(xmax), float(ymin), float(ymax),
        float(w_len), float(w_grid),
        bool(keep_last_outside), float(outside_cost)
    )

    best_idx = int(np.argmin(Js))

    # Only build snapped trajectory for the winner (cheap)
    best_traj = trajs[:, best_idx, :]
    snapped = _snap_to_goal(best_traj, goal, goal_radius, keep_last_outside=keep_last_outside)

    return best_idx, float(Js[best_idx]), snapped, Js


def stack_routes_with_shared_start(routes_xy):
    """
    routes_xy: list of arrays, each (Ni, 2)

    Returns
    -------
    points_all    : (M, 2)
        Stacked XY points with one shared start at index 0.
    route_starts  : (R,)
        Start index of each route in points_all.
    route_lengths : (R,)
        Number of points of each route after removing the shared first point.
    """
    routes_xy = [np.asarray(r, dtype=np.float64) for r in routes_xy]

    if len(routes_xy) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    shared_start = routes_xy[0][0].copy()

    # remove the shared first point from each route
    trimmed = [r[1:] for r in routes_xy]

    route_lengths = np.asarray([len(r) for r in trimmed], dtype=np.int32)
    route_starts = np.zeros(len(trimmed), dtype=np.int32)

    if len(trimmed) > 0:
        route_starts[0] = 1
    if len(trimmed) > 1:
        route_starts[1:] = 1 + np.cumsum(route_lengths[:-1])

    total = 1 + int(route_lengths.sum())
    points_all = np.empty((total, 2), dtype=np.float64)

    points_all[0] = shared_start

    k = 1
    for r in trimmed:
        n = r.shape[0]
        if n > 0:
            points_all[k:k+n] = r
            k += n

    return points_all, route_starts, route_lengths


def build_route_adjacency_matrix(points_all, route_starts, route_lengths):
    """
    Build adjacency / transition matrix for route chains.

    Behavior:
    - row 0 distributes equally to the first point of every route
    - each route then follows its chain deterministically
    - last point of each route is absorbing
    """
    n = points_all.shape[0]
    W = np.zeros((n, n), dtype=np.float64)

    n_routes = route_starts.shape[0]

    # shared start branches equally to all routes
    valid_routes = 0
    for r in range(n_routes):
        if route_lengths[r] > 0:
            valid_routes += 1

    if valid_routes > 0:
        p0 = 1.0 / valid_routes
        for r in range(n_routes):
            s = route_starts[r]
            L = route_lengths[r]
            if L > 0:
                W[0, s] = p0
    else:
        W[0, 0] = 1.0

    # deterministic chain transitions
    for r in range(n_routes):
        s = route_starts[r]
        L = route_lengths[r]

        if L <= 0:
            continue

        for k in range(L - 1):
            u = s + k
            v = s + k + 1
            W[u, v] = 1.0

        # last node absorbing
        W[s + L - 1, s + L - 1] = 1.0

    return W

def interpolate_route_xy(route_xy, n_points=200):
    route_xy = np.asarray(route_xy, dtype=np.float64)

    if route_xy.shape[0] == 1:
        return np.repeat(route_xy, n_points, axis=0)

    seg = np.diff(route_xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)

    s = np.zeros(route_xy.shape[0], dtype=np.float64)
    s[1:] = np.cumsum(seg_len)

    total_len = s[-1]
    if total_len <= 1e-12:
        return np.repeat(route_xy[:1], n_points, axis=0)

    s_new = np.linspace(0.0, total_len, n_points)
    x_new = np.interp(s_new, s, route_xy[:, 0])
    y_new = np.interp(s_new, s, route_xy[:, 1])

    return np.column_stack((x_new, y_new))

@njit(cache=True)
def renormalize_rows_numba(W, eps=1e-12):
    n = W.shape[0]
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += W[i, j]

        if s > eps:
            inv = 1.0 / s
            for j in range(n):
                W[i, j] *= inv
        else:
            W[i, i] = 1.0


@njit(cache=True)
def add_local_forward_transitions_kernel(
    points_all,
    W,
    max_dist,
    shortcut_weight,
    gdx,
    gdy,
    backtrack_tol,
    polys,
    bb_mins,
    bb_maxs,
    allow_self=False,
):
    n = points_all.shape[0]
    max_dist2 = max_dist * max_dist
    use_obstacles = len(polys) > 0

    for i in range(n-10):
        xi = points_all[i, 0]
        yi = points_all[i, 1]
        a = points_all[i]

        for j in range(n):
            if (not allow_self) and (i == j):
                continue

            xj = points_all[j, 0]
            yj = points_all[j, 1]
            b = points_all[j]

            dx = xj - xi
            dy = yj - yi

            # distance filter
            d2 = dx * dx + dy * dy
            if d2 > max_dist2:
                continue

            # forward filter
            proj = dx * gdx + dy * gdy
            if proj < -backtrack_tol:
                continue

            # obstacle check only if obstacles exist
            if use_obstacles:
                if not segment_valid_numba_aabb(a, b, polys, bb_mins, bb_maxs):
                    continue

            W[i, j] += shortcut_weight


def add_local_forward_transitions(
    points_all,
    W,
    start_xy,
    goal_xy,
    poly_obstacles,
    max_dist=1.0,
    shortcut_weight=0.05,
    backtrack_tol=0.0,
    renormalize=True,
):
    """
    Add transitions between every pair of nodes within max_dist,
    keeping only approximately forward transitions in the XY plane,
    and rejecting transitions whose segment crosses an obstacle.

    Parameters
    ----------
    points_all : (N,2)
    W : (N,N)
        Existing adjacency / transition matrix, modified on a copy.
    start_xy, goal_xy : (2,)
        Define the global forward direction.
    poly_obstacles : list[PolyObstacle]
    max_dist : float
    shortcut_weight : float
        Raw weight added for each valid shortcut.
    backtrack_tol : float
        Allows small backward exploration along goal direction.
        0.0 = strict forward only
    renormalize : bool
    """
    points_all = np.asarray(points_all, dtype=np.float64)
    W2 = np.asarray(W, dtype=np.float64).copy()

    start_xy = np.asarray(start_xy, dtype=np.float64)
    goal_xy = np.asarray(goal_xy, dtype=np.float64)

    g = goal_xy - start_xy
    gn = np.linalg.norm(g)
    gdir = g / (gn + 1e-18)

    polys, bb_mins, bb_maxs = preprocess_obstacles(poly_obstacles)

    add_local_forward_transitions_kernel(
        points_all,
        W2,
        float(max_dist),
        float(shortcut_weight),
        float(gdir[0]),
        float(gdir[1]),
        float(backtrack_tol),
        polys,
        bb_mins,
        bb_maxs,
        False,
    )

    if renormalize:
        renormalize_rows_numba(W2)

    return W2

@njit(cache=True, fastmath=True)
def _extract_unique_edges_per_boid_flat(goal_hist, boid_scores, G, max_edges_flat):
    """
    goal_hist: (T,N) int
    boid_scores: (N,) float
    G: number of goals (for encoding)
    max_edges_flat: prealloc length, typically N*(T-1)
    Returns:
      eids_flat[:M], scores_flat[:M], M
    Notes:
      - trims trailing zeros (assumes 0 is padding, not a real goal!)
      - keeps each edge at most once per boid (unique_per_boid=True)
    """
    T, N = goal_hist.shape
    eids_flat = np.empty(max_edges_flat, dtype=np.int64)
    scores_flat = np.empty(max_edges_flat, dtype=np.float64)
    M = 0

    for j in range(N):
        # find last nonzero index (trim trailing zeros)
        last = -1
        for t in range(T - 1, -1, -1):
            if goal_hist[t, j] != 0:
                last = t
                break
        if last <= 0:
            continue  # no valid transitions

        # per-boid seen edges (max transitions <= last)
        seen = np.empty(last, dtype=np.int64)
        seen_n = 0

        prev = int(goal_hist[0, j])
        for t in range(1, last + 1):
            cur = int(goal_hist[t, j])
            if cur != prev:
                u = prev
                v = cur

                # skip invalid ids early (optional safety)
                if 0 <= u < G and 0 <= v < G:
                    eid = u * G + v

                    # unique per boid: skip if already seen
                    already = False
                    for k in range(seen_n):
                        if seen[k] == eid:
                            already = True
                            break
                    if not already:
                        seen[seen_n] = eid
                        seen_n += 1

                        eids_flat[M] = eid
                        scores_flat[M] = boid_scores[j]
                        M += 1

                prev = cur
            else:
                prev = cur

    return eids_flat, scores_flat, M


@njit(cache=True, fastmath=True)
def _aggregate_by_edge(eids_flat, scores_flat, M):
    """
    Aggregate MIN score per eid using sort+reduce.
    Returns:
      uniq_eids, min_scores, counts, K
    """
    if M == 0:
        return (np.empty(0, np.int64),
                np.empty(0, np.float64),
                np.empty(0, np.int64),
                0)

    e = eids_flat[:M].copy()
    s = scores_flat[:M].copy()

    order = np.argsort(e)
    e = e[order]
    s = s[order]

    uniq_eids = np.empty(M, dtype=np.int64)
    min_scores = np.empty(M, dtype=np.float64)
    counts = np.empty(M, dtype=np.int64)

    K = 0
    cur_e = e[0]
    cur_min = s[0]
    cnt = 1

    for i in range(1, M):
        if e[i] == cur_e:
            if s[i] < cur_min:
                cur_min = s[i]
            cnt += 1
        else:
            uniq_eids[K] = cur_e
            min_scores[K] = cur_min
            counts[K] = cnt
            K += 1

            cur_e = e[i]
            cur_min = s[i]
            cnt = 1

    # flush last
    uniq_eids[K] = cur_e
    min_scores[K] = cur_min
    counts[K] = cnt
    K += 1

    return uniq_eids, min_scores, counts, K


def edge_scores(goal_hist, boid_scores, G):
    """
    Convenience wrapper:
      returns arrays:
        u, v, edge_mean, edge_count
    """
    goal_hist = np.asarray(goal_hist, dtype=np.int64)
    boid_scores = np.asarray(boid_scores, dtype=np.float64)

    T, N = goal_hist.shape
    max_edges_flat = N * max(0, (T - 1))

    eids_flat, scores_flat, M = _extract_unique_edges_per_boid_flat(
        goal_hist, boid_scores, int(G), int(max_edges_flat)
    )

    uniq_eids, mean_scores, counts, K = _aggregate_by_edge(eids_flat, scores_flat, M)

    uniq_eids = uniq_eids[:K]
    mean_scores = mean_scores[:K]
    counts = counts[:K]

    u = (uniq_eids // G).astype(np.int64)
    v = (uniq_eids %  G).astype(np.int64)
    return u, v, mean_scores, counts

def remove_worst_transitions(goal_W, u, v, edge_cost, frac_remove=0.2, set_value=0.0):
    """
    Minimization: higher cost is worse => remove largest edge_cost.
    """
    W = np.array(goal_W, copy=True)
    K = edge_cost.shape[0]
    if K == 0:
        return W, []

    k = int(np.floor(frac_remove * K))
    k = max(0, min(k, K))
    if k == 0:
        return W, []

    # indices of worst edges (largest costs)
    worst_idx = np.argsort(edge_cost)[::-1][:k]

    removed = []
    for idx in worst_idx:
        uu = int(u[idx]); vv = int(v[idx]); cc = float(edge_cost[idx])
        if 0 <= uu < W.shape[0] and 0 <= vv < W.shape[1]:
            W[uu, vv] = set_value
            removed.append((uu, vv, cc))

    return W, removed

def build_full_graph(obstacles, start, goal, scale, bounds):
    graph = build_graph(obstacles, start, goal, bounds)
    routes = extract_k_routes_fast(
        graph,
        K=5,
        edge_penalty=1.0,
        node_penalty=1.0,
        forbid_reuse=False,
    )
    routes = routes_to_xy(graph, routes)
    routes = [interpolate_route_xy(r, n_points=100) for r in routes]

    points_all, route_starts, route_lengths = stack_routes_with_shared_start(routes)
    W = build_route_adjacency_matrix(points_all, route_starts, route_lengths)

    W = add_local_forward_transitions(
        points_all,
        W,
        start_xy=start,
        goal_xy=goal,
        poly_obstacles=obstacles,
        max_dist=3.0,
        shortcut_weight=0.1,
        backtrack_tol=1.0,
        renormalize=True,
    )

    return points_all*scale, W

# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    # Threading: use all available threads by default
    try:
        numba.set_num_threads(numba.get_num_threads())
    except Exception:
        pass

    start = np.array([1.0, 1.0])
    goal  = np.array([9.0, 9.0])
    bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))

    obstacles = [
        PolyObstacle(round_obstacle(np.array([[3.0, 1.5], [5.2, 2.2], [4.7, 4.0], [2.8, 3.4]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[6.2, 6.0], [8.5, 6.3], [8.1, 8.4], [6.8, 8.9], [5.9, 7.4]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[1.8, 4.2], [2.7, 4.0], [3.0, 4.8], [2.3, 5.3], [1.7, 4.9]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[4.6, 5.1], [5.4, 5.0], [5.8, 5.7], [5.0, 6.2], [4.4, 5.7]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[7.9, 3.0], [9.0, 3.2], [8.8, 4.2], [7.7, 4.0]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[5.7, 1.0], [6.6, 1.2], [6.4, 2.3], [5.6, 2.1]]), n_iters=4, n_points=32)),
    ]

    graph = build_graph(obstacles, start, goal, bounds)
    routes = extract_k_routes_fast(
        graph,
        K=6,
        edge_penalty=500.0,
        node_penalty=100.0,
        forbid_reuse=False,
    )
    xy_routes = routes_to_xy(graph, routes)

    t0 = time.time()
    graph = build_graph(obstacles, start, goal, bounds)
    routes = extract_k_routes_fast(
        graph,
        K=5,
        edge_penalty=1.0,
        node_penalty=1.0,
        forbid_reuse=False,
    )
    xy_routes = routes_to_xy(graph, routes)
    t1 = time.time()
    print(f"Found graph in {t1-t0:.4f} seconds")

    plot_graph(graph, obstacles, start, goal, bounds)
    plot_routes(graph, routes)
    fig = plt.gcf()
    fig.savefig("save/figures/"+"graph"+".svg", dpi=1200, bbox_inches="tight")
    plt.show()