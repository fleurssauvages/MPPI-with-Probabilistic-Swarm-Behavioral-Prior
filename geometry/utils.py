from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

@dataclass
class PolyObstacle:
    vertices: np.ndarray  # shape (N,2)

Bounds2D = Tuple[Tuple[float, float], Tuple[float, float]]  # ((xmin,xmax),(ymin,ymax))


# -------------------------
# Geometry helpers (2D)
# -------------------------
def polygon_edges(verts: np.ndarray):
    """Yield edges (a,b) for a closed polygon vertex loop."""
    n = len(verts)
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        yield a, b

def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0]*b[1] - a[1]*b[0])

def segments_intersect(a, b, c, d, eps=1e-12) -> bool:
    """Proper/colinear intersection test for segments AB and CD."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = np.asarray(c, float); d = np.asarray(d, float)

    ab = b - a
    cd = d - c

    def orient(p, q, r):
        return cross2(q - p, r - p)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    # General case
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True

    # Colinear / touching cases
    def on_segment(p, q, r):
        # q on pr (colinear assumed)
        return (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and
                min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps)

    if abs(o1) <= eps and on_segment(a, c, b): return True
    if abs(o2) <= eps and on_segment(a, d, b): return True
    if abs(o3) <= eps and on_segment(c, a, d): return True
    if abs(o4) <= eps and on_segment(c, b, d): return True

    return False

def point_in_poly(p: np.ndarray, poly: np.ndarray) -> bool:
    """Ray casting. Returns True if p is inside poly (boundary counts as inside)."""
    p = np.asarray(p, float)
    x, y = p
    inside = False
    n = len(poly)

    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        # Check if point is exactly on edge
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x - x1, y - y1])
        if abs(cross2(v1, v2)) < 1e-12:
            dot = (x - x1)*(x - x2) + (y - y1)*(y - y2)
            if dot <= 1e-12:
                return True

        # Ray cast
        cond = ((y1 > y) != (y2 > y))
        if cond:
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_int >= x:
                inside = not inside

    return inside

def point_to_segment_closest(p, a, b):
    """Closest point q on segment AB to point P, plus distance."""
    p = np.asarray(p, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-18:
        q = a.copy()
        return q, float(np.linalg.norm(p - q))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return q, float(np.linalg.norm(p - q))

def segment_to_segment_closest(a, b, c, d):
    """
    Closest points between segments AB and CD, plus distance.
    If they intersect, returns distance 0 and one intersection-ish point pair.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = np.asarray(c, float); d = np.asarray(d, float)

    if segments_intersect(a, b, c, d):
        # Return any consistent point pair; we choose closest endpoint projection
        qc, dc = point_to_segment_closest(a, c, d)
        qb, db = point_to_segment_closest(b, c, d)
        if dc <= db:
            return a, qc, 0.0
        else:
            return b, qb, 0.0

    # Candidates: endpoints to opposite segments
    q1, d1 = point_to_segment_closest(a, c, d)
    q2, d2 = point_to_segment_closest(b, c, d)
    q3, d3 = point_to_segment_closest(c, a, b)
    q4, d4 = point_to_segment_closest(d, a, b)

    ds = [d1, d2, d3, d4]
    k = int(np.argmin(ds))

    if k == 0: return a, q1, d1
    if k == 1: return b, q2, d2
    if k == 2: return q3, c, d3  # (closest on AB, C)
    return q4, d, d4              # (closest on AB, D)


# -------------------------
# Polygon distances
# -------------------------
def polygons_overlap(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    # Edge intersection
    for a, b in polygon_edges(poly1):
        for c, d in polygon_edges(poly2):
            if segments_intersect(a, b, c, d):
                return True
    # Containment
    if point_in_poly(poly1[0], poly2): return True
    if point_in_poly(poly2[0], poly1): return True
    return False

def shortest_segment_between_polygons(poly1: np.ndarray, poly2: np.ndarray):
    """
    Returns (p_closest_on_poly1, q_closest_on_poly2, distance).
    If polygons overlap, distance=0 and p=q (some point pair).
    """
    if polygons_overlap(poly1, poly2):
        # Return a point known to be inside/overlapping
        p = poly1[0].copy()
        return p, p.copy(), 0.0

    best_d = np.inf
    best_p = None
    best_q = None

    for a, b in polygon_edges(poly1):
        for c, d in polygon_edges(poly2):
            p, q, dist = segment_to_segment_closest(a, b, c, d)
            if dist < best_d:
                best_d = dist
                best_p = p
                best_q = q

    return best_p, best_q, float(best_d)


# -------------------------
# Bounds handling
# -------------------------
def bounds_as_polygon(bounds: Bounds2D) -> np.ndarray:
    (xmin, xmax), (ymin, ymax) = bounds
    return np.array([[xmin, ymin],
                     [xmax, ymin],
                     [xmax, ymax],
                     [xmin, ymax]], dtype=float)

def shortest_segment_polygon_to_bounds(poly: np.ndarray, bounds: Bounds2D):
    """
    Shortest segment from polygon to rectangle boundary.
    Returns (p_on_poly, q_on_boundary, distance).
    If polygon intersects/inside touches boundary => distance 0.
    """
    rect = bounds_as_polygon(bounds)

    # If any polygon edge intersects any boundary edge -> distance 0
    for a, b in polygon_edges(poly):
        for c, d in polygon_edges(rect):
            if segments_intersect(a, b, c, d):
                # similar choice as above: just return 0 with some consistent points
                qc, _ = point_to_segment_closest(a, c, d)
                return a.copy(), qc, 0.0

    # If polygon is outside but rectangle contains a polygon vertex, then it’s inside;
    # distance to boundary is min distance from polygon to boundary edges (positive).
    # (If polygon is fully inside rectangle, this gives distance to boundary.)
    best_d = np.inf
    best_p = None
    best_q = None

    for a, b in polygon_edges(poly):
        for c, d in polygon_edges(rect):
            p, q, dist = segment_to_segment_closest(a, b, c, d)
            if dist < best_d:
                best_d = dist
                best_p = p
                best_q = q

    return best_p, best_q, float(best_d)


# -------------------------
# Main API
# -------------------------
def compute_obstacle_segments(
    obstacles: List[PolyObstacle],
    bounds: Bounds2D,
    close_to_bounds_thresh: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Returns:
      - pairwise: list of dicts {i,j, p, q, dist}
      - to_bounds: list of dicts {i, p, q, dist} (optionally filtered by thresh)
    """
    pairwise = []
    n = len(obstacles)

    for i in range(n):
        for j in range(i + 1, n):
            p, q, d = shortest_segment_between_polygons(obstacles[i].vertices,
                                                       obstacles[j].vertices)
            pairwise.append(dict(i=i, j=j, p=np.asarray(p), q=np.asarray(q), dist=float(d)))

    to_bounds = []
    for i in range(n):
        p, q, d = shortest_segment_polygon_to_bounds(obstacles[i].vertices, bounds)
        if (close_to_bounds_thresh is None) or (d <= close_to_bounds_thresh):
            to_bounds.append(dict(i=i, p=np.asarray(p), q=np.asarray(q), dist=float(d)))

    return dict(pairwise=pairwise, to_bounds=to_bounds)


def segment_intersects_polygon(p: np.ndarray, q: np.ndarray, poly: np.ndarray, eps: float = 1e-12) -> bool:
    """
    Returns True if segment PQ intersects polygon boundary OR lies inside the polygon.
    (Touching counts as intersection.)
    """
    p = np.asarray(p, float)
    q = np.asarray(q, float)

    # boundary intersection
    for a, b in polygon_edges(poly):
        if segments_intersect(p, q, a, b, eps=eps):
            return True

    # if either endpoint is inside/on polygon, we consider it intersecting
    if point_in_poly(p, poly) or point_in_poly(q, poly):
        return True

    return False


def filter_segments_and_centers(
    segments: List[Dict[str, Any]],
    obstacles: List[PolyObstacle],
    max_len: float,
    ignore_ends_from: bool = True,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Keep only segments:
      - with dist <= max_len
      - that do NOT intersect ANY obstacle polygon (optionally ignoring the two polygons they came from)

    Parameters
    ----------
    segments : list of dict
        Each dict must contain at least {'p': (2,), 'q': (2,), 'dist': float}
        Can also contain indices like {'i','j'} for pairwise, or {'i'} for to_bounds.
    obstacles : list[PolyObstacle]
    max_len : float
        threshold on segment length
    ignore_ends_from : bool
        If True, for pairwise segments it will ignore intersection checks with the two source polygons (i,j),
        because p and q lie on their boundaries by construction and would "intersect" trivially.

    Returns
    -------
    kept_segments : list of dict
    centers : (K,2) ndarray
        Centers of kept segments (midpoints)
    """
    kept = []
    centers = []

    for s in segments:
        d = float(s["dist"])
        if d > max_len:
            continue

        p = np.asarray(s["p"], float)
        q = np.asarray(s["q"], float)

        # Determine which polygons to skip (optional)
        skip = set()
        if ignore_ends_from:
            if "i" in s: skip.add(int(s["i"]))
            if "j" in s: skip.add(int(s["j"]))

        ok = True
        for k, obs in enumerate(obstacles):
            if k in skip:
                continue
            if segment_intersects_polygon(p, q, obs.vertices):
                ok = False
                break

        if ok:
            kept.append(s)
            centers.append(0.5 * (p + q))

    centers = np.asarray(centers, dtype=float) if centers else np.zeros((0, 2), dtype=float)
    return kept, centers

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

@dataclass
class PolyObstacle:
    vertices: np.ndarray  # (m,2)

def obstacles_to_segs(obstacles, scale = 1.0):
    """
    obstacles: list of PolyObstacle objects
    returns: segs array (M,4) float32
    """
    segs_all = []

    for obs in obstacles:
        # change this if your attribute name is different
        poly = np.asarray(obs.vertices, dtype=np.float32)

        if poly.shape[0] < 3:
            continue

        # edges i -> i+1 and closing edge last -> first
        p1 = poly
        p2 = np.roll(poly, -1, axis=0)

        segs = np.column_stack([
            p1[:, 0], p1[:, 1],
            p2[:, 0], p2[:, 1]
        ])  # (m,4)

        segs_all.append(segs * scale)

    if not segs_all:
        return np.zeros((0,4), dtype=np.float32)

    return np.vstack(segs_all).astype(np.float32)