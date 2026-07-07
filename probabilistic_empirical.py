"""
Obstacle-aware probabilistic homotopy-aware generative planning pipeline.

This script turns fish rollout samples into a topological probabilistic
trajectory representation with two layers:

    p(h | E, xs, xg) = pi_h

    p(tau | h, E, xs, xg) = sum_i alpha_i delta(tau_i)

where each tau_i is an actual collision-free fish trajectory assigned to
homotopy mode h. A Gaussian moment model is still fitted per mode, but only
for compact Bayesian homotopy inference p(h | y). The posterior predictive
trajectories are sampled from the empirical support, not from symmetric
Gaussians, so they remain on the obstacle-aware trajectory manifold.

It demonstrates:
  1. training-free fish trajectory generation
  2. quality-weighted topological empirical trajectory mixture
  3. Bayesian homotopy inference from a partial observed prefix
  4. empirical posterior predictive sampling over valid trajectories
  5. visualization with non-symmetric sample-supported hulls, not ellipses
"""

import math
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import Delaunay, ConvexHull
from shapely.geometry import MultiPoint, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, polygonize

from geometry.utils import round_obstacle, PolyObstacle, obstacles_to_segs
from RL.env2Ddiverse import FishGoalEnv2D
from graph.graph import build_full_graph
from planner import (
    HomotopyAwareGenerativePlanner,
    trajectory_cost,
)


# -----------------------------------------------------------------------------
# Small trajectory utilities
# -----------------------------------------------------------------------------

def resample_path(path: np.ndarray, K: int) -> np.ndarray:
    """Arc-length resample a 2D path to K points."""
    p = np.asarray(path, dtype=np.float64)
    if p.shape[0] == 1:
        return np.repeat(p, K, axis=0)

    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.zeros(p.shape[0], dtype=np.float64)
    s[1:] = np.cumsum(d)
    total = s[-1]
    if total <= 1e-12:
        return np.repeat(p[:1], K, axis=0)

    s_new = np.linspace(0.0, total, K)
    x = np.interp(s_new, s, p[:, 0])
    y = np.interp(s_new, s, p[:, 1])
    return np.column_stack([x, y])


def flatten_path(path_K: np.ndarray) -> np.ndarray:
    return np.asarray(path_K, dtype=np.float64).reshape(-1)


def unflatten_path(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    return v.reshape((-1, 2))


def stable_softmax_from_cost(costs: np.ndarray, beta: float) -> np.ndarray:
    """Convert costs to normalized exp(-beta cost) weights."""
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


def gaussian_logpdf(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Numerically stable log N(y; mean, cov)."""
    y = np.asarray(y, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    n = y.shape[0]
    cov = cov + 1e-7 * np.eye(n)
    try:
        L = np.linalg.cholesky(cov)
        r = y - mean
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        return float(-0.5 * (r @ alpha + logdet + n * np.log(2.0 * np.pi)))
    except np.linalg.LinAlgError:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return -np.inf
        r = y - mean
        alpha = np.linalg.pinv(cov) @ r
        return float(-0.5 * (r @ alpha + logdet + n * np.log(2.0 * np.pi)))




# -----------------------------------------------------------------------------
# Empirical hull utilities
# -----------------------------------------------------------------------------

def obstacle_union_geometry(obstacles):
    polys = []
    for obs in obstacles:
        if hasattr(obs, "vertices"):
            xy = np.asarray(obs.vertices, dtype=np.float64)
        else:
            xy = np.asarray(obs, dtype=np.float64)
        if xy.ndim == 2 and xy.shape[0] >= 3:
            poly = Polygon(xy)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                polys.append(poly)
    if len(polys) == 0:
        return GeometryCollection()
    return unary_union(polys).buffer(0)


def _unique_points(points: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return pts.reshape(0, 2)
    q = np.round(pts / float(tol)).astype(np.int64)
    _, ids = np.unique(q, axis=0, return_index=True)
    return pts[np.sort(ids)]


def convex_hull_polygon(points: np.ndarray):
    pts = _unique_points(points)
    if pts.shape[0] < 3:
        if pts.shape[0] == 0:
            return GeometryCollection()
        return MultiPoint(pts).buffer(0.05)
    try:
        hull = ConvexHull(pts)
        return Polygon(pts[hull.vertices]).buffer(0)
    except Exception:
        return MultiPoint(pts).convex_hull.buffer(0)


def alpha_shape_polygon(points: np.ndarray, alpha: float = 2.0):
    """
    Concave hull / alpha shape for a 2D point cloud.

    Larger alpha gives tighter, more detailed hulls. Smaller alpha approaches
    a convex hull. Falls back to a convex hull when the triangulation is
    degenerate or too sparse.
    """
    pts = _unique_points(points)
    if pts.shape[0] < 4:
        return convex_hull_polygon(pts)

    try:
        tri = Delaunay(pts)
    except Exception:
        return convex_hull_polygon(pts)

    edges = set()
    edge_segments = []
    radius_thresh = 1.0 / max(float(alpha), 1e-9)

    for simplex in tri.simplices:
        pa, pb, pc = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)
        s = 0.5 * (a + b + c)
        area2 = max(s * (s - a) * (s - b) * (s - c), 0.0)
        area = math.sqrt(area2)
        if area <= 1e-12:
            continue
        circum_r = (a * b * c) / (4.0 * area)
        if circum_r > radius_thresh:
            continue

        for ii, jj in ((simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])):
            e = tuple(sorted((int(ii), int(jj))))
            if e in edges:
                # Interior edge; remove once seen twice.
                try:
                    edge_segments.remove((pts[e[0]], pts[e[1]]))
                except ValueError:
                    pass
            else:
                edges.add(e)
                edge_segments.append((pts[e[0]], pts[e[1]]))

    if len(edge_segments) == 0:
        return convex_hull_polygon(pts)

    # Build polygon from boundary edges.
    from shapely.geometry import LineString
    lines = [LineString([a, b]) for a, b in edge_segments]
    polys = list(polygonize(lines))
    if len(polys) == 0:
        return convex_hull_polygon(pts)
    return unary_union(polys).buffer(0)


def trajectory_tube_hull(paths_K: np.ndarray, alpha: float = 2.2, temporal_window: int = 7, temporal_stride: int = 4,
                         expand: float = 0.04, obstacles=None):
    """
    Build a non-symmetric empirical tube from real trajectory samples.

    Instead of one global convex hull, this computes alpha shapes over temporal
    windows and unions them. This preserves route geometry better and avoids
    filling the whole workspace between early and late trajectory points.
    Obstacles are subtracted from the hull before plotting.
    """
    P = np.asarray(paths_K, dtype=np.float64)
    if P.ndim != 3 or P.shape[-1] != 2:
        raise ValueError(f"Expected paths_K with shape M x K x 2, got {P.shape}")
    M, K, _ = P.shape
    if M == 0:
        return GeometryCollection()

    geoms = []
    half = max(1, int(temporal_window) // 2)
    stride = max(1, int(temporal_stride))
    for t0 in range(0, K, stride):
        lo = max(0, t0 - half)
        hi = min(K, t0 + half + 1)
        pts = P[:, lo:hi, :].reshape(-1, 2)
        pts = _unique_points(pts)
        if pts.shape[0] < 3:
            continue
        g = alpha_shape_polygon(pts, alpha=alpha)
        if expand > 0.0:
            g = g.buffer(float(expand), join_style=1).buffer(0)
        if not g.is_empty:
            geoms.append(g)

    if len(geoms) == 0:
        geom = convex_hull_polygon(P.reshape(-1, 2))
    else:
        geom = unary_union(geoms).buffer(0)

    if obstacles is not None:
        obs = obstacle_union_geometry(obstacles)
        if not obs.is_empty:
            geom = geom.difference(obs).buffer(0)
    return geom


def plot_shapely_geometry(ax, geom, *, alpha=0.22, linewidth=1.2, label=None, facecolor=None, edgecolor=None):
    """Plot Polygon/MultiPolygon/GeometryCollection on an axis."""
    if geom is None or geom.is_empty:
        return
    geoms = []
    if isinstance(geom, Polygon):
        geoms = [geom]
    elif isinstance(geom, MultiPolygon):
        geoms = list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        geoms = [g for g in geom.geoms if isinstance(g, Polygon)]
    else:
        try:
            geoms = [g for g in geom.geoms if isinstance(g, Polygon)]
        except Exception:
            geoms = []

    first = True
    for poly in geoms:
        if poly.area <= 1e-8:
            continue
        xy = np.asarray(poly.exterior.coords, dtype=np.float64)
        patch = MplPolygon(
            xy,
            closed=True,
            alpha=alpha,
            linewidth=linewidth,
            fill=True,
            label=(label if first else None),
            facecolor=facecolor,
            edgecolor=edgecolor if edgecolor is not None else facecolor,
        )
        ax.add_patch(patch)
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            linewidth=linewidth,
            alpha=0.85,
            color=edgecolor if edgecolor is not None else facecolor,
        )
        first = False


# -----------------------------------------------------------------------------
# Probabilistic topological trajectory mixture
# -----------------------------------------------------------------------------

@dataclass
class GaussianTrajectoryMode:
    signature: Tuple[int, ...]
    probability: float
    mean: np.ndarray          # D = 2K, moment summary for p(h | y)
    cov: np.ndarray           # D x D, moment summary for p(h | y)
    samples: np.ndarray       # M x D, empirical support for p(tau | h)
    weights: np.ndarray       # M, empirical probabilities alpha_i
    mean_cost: float
    count: int

    @property
    def mean_path(self) -> np.ndarray:
        return unflatten_path(self.mean)


@dataclass
class TopologicalTrajectoryMixture:
    modes: Dict[Tuple[int, ...], GaussianTrajectoryMode]
    K: int
    beta: float

    @property
    def signatures(self) -> List[Tuple[int, ...]]:
        return list(self.modes.keys())

    def prior(self) -> Dict[Tuple[int, ...], float]:
        return {h: m.probability for h, m in self.modes.items()}

    def infer_homotopy_posterior(self, observed_prefix: np.ndarray, obs_steps: int) -> Dict[Tuple[int, ...], float]:
        """
        Bayesian homotopy inference from a partial observed path.

        observed_prefix is a short 2D prefix. It is resampled to obs_steps points
        and compared against the marginal Gaussian of each homotopy mode.
        """
        obs = resample_path(observed_prefix, obs_steps)
        y = flatten_path(obs)
        d = y.shape[0]

        logps = []
        sigs = []
        for h, mode in self.modes.items():
            mu_a = mode.mean[:d]
            cov_aa = mode.cov[:d, :d]
            log_prior = np.log(max(mode.probability, 1e-300))
            log_like = gaussian_logpdf(y, mu_a, cov_aa)
            sigs.append(h)
            logps.append(log_prior + log_like)

        logps = np.asarray(logps, dtype=np.float64)
        logps -= np.max(logps)
        p = np.exp(logps)
        p /= np.sum(p) + 1e-300
        return {h: float(p[i]) for i, h in enumerate(sigs)}

    def sample_empirical_prior(self, n: int, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
        """
        Sample trajectories from the prior empirical mixture.

        This samples stored fish trajectories, not Gaussian perturbations:

            h ~ Categorical(pi)
            tau ~ sum_i alpha_i delta(tau_i | h)

        Therefore samples stay on the valid sample support produced by the
        fish planner.
        """
        rng = np.random.default_rng() if rng is None else rng
        sigs = self.signatures
        probs = np.array([self.modes[h].probability for h in sigs], dtype=np.float64)
        probs /= np.sum(probs) + 1e-300
        out = []
        for _ in range(n):
            h = sigs[int(rng.choice(len(sigs), p=probs))]
            mode = self.modes[h]
            local_probs = np.asarray(mode.weights, dtype=np.float64)
            local_probs /= np.sum(local_probs) + 1e-300
            idx = int(rng.choice(mode.samples.shape[0], p=local_probs))
            out.append(unflatten_path(mode.samples[idx]))
        return out

    # Backward-compatible name, but intentionally empirical rather than Gaussian.
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
        return self.sample_empirical_prior(n=n, rng=rng)

    def sample_posterior_predictive(
        self,
        observed_prefix: np.ndarray,
        obs_steps: int,
        n: int,
        rng: Optional[np.random.Generator] = None,
        prefix_distance_gamma: float = 25.0,
    ) -> Tuple[List[np.ndarray], Dict[Tuple[int, ...], float]]:
        """
        Sample trajectories from an obstacle-aware empirical posterior:

            p(tau_i | y) proportional to
                p(h_i | y) * alpha_i * exp(-gamma D(tau_i_prefix, y))

        where tau_i is an actual fish trajectory. This avoids symmetric
        Gaussian posterior samples that can cut through obstacles.
        """
        rng = np.random.default_rng() if rng is None else rng
        posterior = self.infer_homotopy_posterior(observed_prefix, obs_steps)

        obs = resample_path(observed_prefix, obs_steps)
        y = flatten_path(obs)
        d_obs = y.shape[0]

        candidate_paths = []
        candidate_weights = []

        for h, mode in self.modes.items():
            ph = float(posterior.get(h, 0.0))
            if ph <= 0.0:
                continue

            X = np.asarray(mode.samples, dtype=np.float64)
            local_w = np.asarray(mode.weights, dtype=np.float64)
            local_w = local_w / (np.sum(local_w) + 1e-300)

            prefixes = X[:, :d_obs]
            diff = prefixes - y[None, :]
            mse = np.mean(diff * diff, axis=1)
            prefix_w = np.exp(-float(prefix_distance_gamma) * mse)

            w = ph * local_w * prefix_w
            for i in range(X.shape[0]):
                if w[i] > 0.0 and np.isfinite(w[i]):
                    candidate_paths.append(X[i])
                    candidate_weights.append(float(w[i]))

        if len(candidate_paths) == 0:
            # Fallback: sample from posterior modes, ignoring prefix distance.
            for h, mode in self.modes.items():
                ph = float(posterior.get(h, 0.0))
                if ph <= 0.0:
                    continue
                local_w = np.asarray(mode.weights, dtype=np.float64)
                local_w = local_w / (np.sum(local_w) + 1e-300)
                for i in range(mode.samples.shape[0]):
                    candidate_paths.append(mode.samples[i])
                    candidate_weights.append(float(ph * local_w[i]))

        weights = np.asarray(candidate_weights, dtype=np.float64)
        weights = weights / (np.sum(weights) + 1e-300)
        idxs = rng.choice(len(candidate_paths), size=int(n), replace=True, p=weights)
        samples = [unflatten_path(candidate_paths[int(i)]) for i in idxs]
        return samples, posterior


def fit_topological_trajectory_mixture(
    gen_out,
    obstacles,
    *,
    K: int = 50,
    beta: float = 1.0,
    min_mode_samples: int = 3,
    covariance_jitter: float = 1e-4,
    costmap=None,
    bounds=((0.0, 10.0), (0.0, 10.0)),
) -> TopologicalTrajectoryMixture:
    """
    Fit p(tau,h) = pi_h N(tau; mu_h, Sigma_h) from fish samples.

    The mixture weights are quality-weighted:
        w_i proportional to exp(-beta C(tau_i))
        pi_h = sum_{i in h} w_i / sum_i w_i
    """
    # Compute one global quality weight for every generated sample.
    all_paths = list(gen_out.samples)
    all_costs = np.array([
        trajectory_cost(p, costmap=costmap, bounds=bounds, w_len=1.0, w_smooth=0.05)
        for p in all_paths
    ], dtype=np.float64)
    all_weights = stable_softmax_from_cost(all_costs, beta=beta)

    # Map object id(path) -> global weight/cost.
    weight_by_id = {id(p): float(w) for p, w in zip(all_paths, all_weights)}
    cost_by_id = {id(p): float(c) for p, c in zip(all_paths, all_costs)}

    mode_raw = {}
    total_mode_weight = 0.0

    for sig, paths in gen_out.homotopy_groups.items():
        if len(paths) < min_mode_samples:
            continue

        X = np.stack([flatten_path(resample_path(p, K)) for p in paths], axis=0)
        w = np.array([weight_by_id.get(id(p), 1.0) for p in paths], dtype=np.float64)
        c = np.array([cost_by_id.get(id(p), np.nan) for p in paths], dtype=np.float64)
        if np.sum(w) <= 1e-12:
            w = np.ones(len(paths), dtype=np.float64) / len(paths)
        else:
            w = w / np.sum(w)

        mu = np.sum(X * w[:, None], axis=0)
        Xc = X - mu[None, :]
        cov = (Xc * w[:, None]).T @ Xc
        cov = cov + covariance_jitter * np.eye(cov.shape[0])

        mode_weight = float(np.sum([weight_by_id.get(id(p), 0.0) for p in paths]))
        total_mode_weight += mode_weight
        mode_raw[sig] = dict(X=X, w=w, mu=mu, cov=cov, mode_weight=mode_weight, mean_cost=float(np.nanmean(c)))

    modes = {}
    if total_mode_weight <= 1e-12:
        total_mode_weight = float(len(mode_raw))
        for sig in mode_raw:
            mode_raw[sig]["mode_weight"] = 1.0

    for sig, d in mode_raw.items():
        pi = float(d["mode_weight"] / total_mode_weight)
        modes[sig] = GaussianTrajectoryMode(
            signature=sig,
            probability=pi,
            mean=d["mu"],
            cov=d["cov"],
            samples=d["X"],
            weights=d["w"],
            mean_cost=d["mean_cost"],
            count=int(d["X"].shape[0]),
        )

    return TopologicalTrajectoryMixture(modes=modes, K=K, beta=beta)




def normalize_plot_bounds(bounds):
    """
    Accept either:
      - bounds = (lower_xy, upper_xy), e.g. (array([0,0]), array([10,10]))
      - bounds = ((xmin, xmax), (ymin, ymax))

    Return xmin, xmax, ymin, ymax.
    """
    b0 = np.asarray(bounds[0], dtype=np.float64)
    b1 = np.asarray(bounds[1], dtype=np.float64)

    # Case 1: lower_xy, upper_xy. This is what graph.build_full_graph expects.
    if b0.shape == (2,) and b1.shape == (2,):
        # Ambiguity: ((xmin, xmax), (ymin, ymax)) also has this shape.
        # Detect the common plotting-range convention by checking whether
        # bounds[0] is increasing and bounds[1] is increasing while lower_xy/upper_xy
        # would normally satisfy b0[0] <= b1[0] and b0[1] <= b1[1].
        # For your standard case, (array([0,0]), array([10,10])), this returns 0,10,0,10.
        if b0[0] <= b1[0] and b0[1] <= b1[1]:
            xmin, ymin = b0
            xmax, ymax = b1
            return float(xmin), float(xmax), float(ymin), float(ymax)

    # Case 2: explicit axis ranges.
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    return float(xmin), float(xmax), float(ymin), float(ymax)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def setup_workspace(ax, obstacles, start, goal, bounds=((0.0, 10.0), (0.0, 10.0)), title=None):
    xmin, xmax, ymin, ymax = normalize_plot_bounds(bounds)

    for obs in obstacles:
        p = np.asarray(obs.vertices, dtype=np.float64)
        ax.fill(p[:, 0], p[:, 1], alpha=0.25)
        ax.plot(np.r_[p[:, 0], p[0, 0]], np.r_[p[:, 1], p[0, 1]], linewidth=1.0)

    ax.scatter([start[0]], [start[1]], s=80, marker="o", label="start")
    ax.scatter([goal[0]], [goal[1]], s=140, marker="*", label="goal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)



def visualize_probabilistic_pipeline(
    gen_out,
    mixture: TopologicalTrajectoryMixture,
    posterior_samples: List[np.ndarray],
    posterior: Dict[Tuple[int, ...], float],
    observed_prefix: np.ndarray,
    obstacles,
    start,
    goal,
    *,
    max_raw_samples: int = 200,
    bounds=((0.0, 10.0), (0.0, 10.0)),
    save_path: Optional[str] = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax0, ax1, ax2, ax3 = axes.ravel()

    # Panel 1: raw generated samples and per-mode mean paths.
    setup_workspace(ax0, obstacles, start, goal, bounds, "Raw fish samples and mixture means")
    raw = list(gen_out.samples)
    if len(raw) > max_raw_samples:
        ids = np.linspace(0, len(raw) - 1, max_raw_samples).astype(int)
        raw = [raw[i] for i in ids]
    for p in raw:
        ax0.plot(p[:, 0], p[:, 1], linewidth=0.5, alpha=0.14)
    mode_items = sorted(mixture.modes.items(), key=lambda kv: kv[1].probability, reverse=True)
    mode_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(mode_colors) == 0:
        mode_colors = [f"C{i}" for i in range(max(1, len(mode_items)))]

    for k, (sig, mode) in enumerate(mode_items):
        color = mode_colors[k % len(mode_colors)]
        mp = mode.mean_path
        ax0.plot(mp[:, 0], mp[:, 1], linewidth=2.5, color=color, label=f"h{k}: pi={mode.probability:.2f}")
    ax0.legend(fontsize=8, loc="lower right")

    # Panel 2: sample-supported, non-symmetric hull/tube.
    # This visualizes empirical trajectory support using alpha-shape hulls over
    # temporal windows, then subtracts obstacle polygons. This avoids symmetric
    # Gaussian ellipses and avoids filling obstacle interiors.
    setup_workspace(ax1, obstacles, start, goal, bounds, "Obstacle-aware empirical trajectory hulls")
    for k, (sig, mode) in enumerate(mode_items):
        color = mode_colors[k % len(mode_colors)]
        X_paths = np.asarray([unflatten_path(v) for v in mode.samples], dtype=np.float64)
        hull = trajectory_tube_hull(
            X_paths,
            alpha=2.4,
            temporal_window=7,
            temporal_stride=4,
            expand=0.035,
            obstacles=obstacles,
        )
        plot_shapely_geometry(
            ax1,
            hull,
            alpha=0.18,
            linewidth=1.0,
            label=f"h{k} support",
            facecolor=color,
            edgecolor=color,
        )
        mp = mode.mean_path
        ax1.plot(mp[:, 0], mp[:, 1], linewidth=2.2, color=color, label=f"h{k} mean")
    ax1.legend(fontsize=8, loc="lower right")

    # Panel 3: prior and posterior over homotopy modes.
    ordered = sorted(mixture.modes.keys(), key=lambda h: mixture.modes[h].probability, reverse=True)
    labels = [f"h{i}" for i in range(len(ordered))]
    prior = np.array([mixture.modes[h].probability for h in ordered])
    post = np.array([posterior.get(h, 0.0) for h in ordered])
    x = np.arange(len(ordered))
    width = 0.38
    ax2.bar(x - width / 2, prior, width, label="prior p(h)")
    ax2.bar(x + width / 2, post, width, label="posterior p(h | prefix)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0)
    ax2.set_ylabel("probability")
    ax2.set_title("Bayesian homotopy belief")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.25)

    # Panel 4: posterior predictive samples.
    setup_workspace(ax3, obstacles, start, goal, bounds, "Empirical posterior predictive trajectories")
    for p in posterior_samples[:80]:
        ax3.plot(p[:, 0], p[:, 1], linewidth=0.7, alpha=0.22)
    ax3.plot(observed_prefix[:, 0], observed_prefix[:, 1], linewidth=4.0, label="observed prefix")
    ax3.legend(loc="lower right")

    fig.suptitle(
        "Topological probabilistic trajectory mixture: "
        f"{len(gen_out.samples)} samples, {len(mixture.modes)} probabilistic modes",
        fontsize=14,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


# -----------------------------------------------------------------------------
# Main experiment: your original script plus probabilistic fitting/conditioning
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    scale = 4.0

    # Use separate bounds for graph construction and scalar-range utilities.
    # build_full_graph expects (lower_xy, upper_xy). Plotting accepts both.
    bounds_xy = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    bounds_ranges = ((0.0, 10.0), (0.0, 10.0))

    start = np.array([1.0, 1.0])
    goal = np.array([9.0, 9.0])

    obstacles = [
        PolyObstacle(round_obstacle(np.array([[3.0, 1.5], [5.2, 2.2], [4.7, 4.0], [2.8, 3.4]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[6.2, 6.0], [8.5, 6.3], [8.1, 8.4], [6.8, 8.9], [5.9, 7.4]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]]), n_iters=4, n_points=32)),
    ]

    segs = obstacles_to_segs(obstacles, scale=scale)

    base_action = pickle.load(open("save/best_policy.pkl", "rb"))["best_theta"]

    graph_goals, graph_W = build_full_graph(
        obstacles=obstacles,
        start=start,
        goal=goal,
        scale=scale,
        bounds=bounds_xy,
    )

    planner = HomotopyAwareGenerativePlanner(
        env_cls=FishGoalEnv2D,
        action=base_action,
        obstacles=obstacles,
        segs=segs,
        scale=scale,
        boid_count=1200,
        max_steps=700,
        dt=0.5,
    )

    # 1. Sample trajectories from the training-free implicit generator.
    gen_out = planner.sample(
        start_unscaled=start,
        goal_unscaled=goal,
        graph_goals=graph_goals,
        graph_W=graph_W,
        seed=3,
    )

    print("Generated trajectories:", len(gen_out.samples))
    print("Empirical homotopy modes:", len(gen_out.homotopy_groups))
    print("Raw empirical probabilities:")
    for sig, prob in sorted(gen_out.probabilities.items(), key=lambda kv: kv[1], reverse=True):
        print(" ", sig, prob)

    # 2. Fit the probabilistic object:
    #       p(h) = pi_h
    #       p(tau | h) = sum_i alpha_i delta(tau_i)
    #    with quality-weighted pi_h and alpha_i. A Gaussian moment model is
    #    also fitted only to infer p(h | observed prefix).
    mixture = fit_topological_trajectory_mixture(
        gen_out,
        obstacles,
        K=50,
        beta=1.0,
        min_mode_samples=3,
        covariance_jitter=2e-4,
        costmap=None,
        bounds=bounds_ranges,
    )

    print("\nQuality-weighted probabilistic mixture:")
    for i, (sig, mode) in enumerate(sorted(mixture.modes.items(), key=lambda kv: kv[1].probability, reverse=True)):
        print(
            f"  h{i}: sig={sig}, pi={mode.probability:.3f}, "
            f"count={mode.count}, mean_cost={mode.mean_cost:.3f}"
        )

    # 3. Use the distribution for Bayesian homotopy inference.
    #    Here we simulate a partial observation by taking the first 30% of one
    #    generated trajectory from the largest-probability mode.
    top_sig = max(mixture.modes.keys(), key=lambda h: mixture.modes[h].probability)
    observed_source = gen_out.homotopy_groups[top_sig][0]
    prefix_len = max(5, int(0.30 * len(observed_source)))
    observed_prefix = observed_source[:prefix_len]

    posterior_samples, posterior = mixture.sample_posterior_predictive(
        observed_prefix,
        obs_steps=15,
        n=120,
        rng=np.random.default_rng(7),
        prefix_distance_gamma=25.0,
    )

    print("\nPosterior p(h | observed prefix):")
    for sig, prob in sorted(posterior.items(), key=lambda kv: kv[1], reverse=True):
        print(" ", sig, prob)

    # 4. Visualize samples, empirical hull support, posterior homotopy belief,
    #    and empirical posterior predictive trajectories.
    visualize_probabilistic_pipeline(
        gen_out,
        mixture,
        posterior_samples,
        posterior,
        observed_prefix,
        obstacles,
        start,
        goal,
        bounds=bounds_xy,
        save_path="probabilistic_generative_pipeline.png",
    )

    plt.show()