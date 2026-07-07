"""
Probabilistic homotopy-aware generative planning pipeline.

This script turns fish rollout samples into a topological probabilistic
trajectory mixture:

    p(tau | E, xs, xg) = sum_h pi_h N(tau; mu_h, Sigma_h)

Then it demonstrates Bayesian conditioning from a partial observed trajectory:

    p(h | y) proportional to p(y | h) pi_h

and visualizes:
  1. raw generated samples grouped by homotopy
  2. probabilistic trajectory tubes
  3. prior vs posterior homotopy probabilities
  4. posterior predictive trajectory samples
"""

import math
import pickle
import colorsys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse

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
# Collision checking for Gaussian posterior predictive samples
# -----------------------------------------------------------------------------

def _poly_vertices(obs) -> np.ndarray:
    """Return obstacle vertices as an (N,2) float array."""
    if hasattr(obs, "vertices"):
        return np.asarray(obs.vertices, dtype=np.float64)[:, :2]
    return np.asarray(obs, dtype=np.float64)[:, :2]


def _orient(a, b, c) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a, b, p, eps=1e-10) -> bool:
    if abs(_orient(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
        min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(a, b, c, d, eps=1e-10) -> bool:
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    if abs(o1) <= eps and _on_segment(a, b, c, eps):
        return True
    if abs(o2) <= eps and _on_segment(a, b, d, eps):
        return True
    if abs(o3) <= eps and _on_segment(c, d, a, eps):
        return True
    if abs(o4) <= eps and _on_segment(c, d, b, eps):
        return True

    return False


def _point_in_poly(p, poly, eps=1e-10) -> bool:
    """Return True if p is inside or on the boundary of a polygon."""
    p = np.asarray(p, dtype=np.float64)
    poly = np.asarray(poly, dtype=np.float64)

    inside = False
    n = poly.shape[0]
    x, y = p[0], p[1]

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        if _on_segment(a, b, p, eps):
            return True

        yi = a[1]
        yj = b[1]
        xi = a[0]
        xj = b[0]

        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / ((yj - yi) + 1e-18)
            if x < x_cross:
                inside = not inside

    return inside


def segment_collides_with_obstacles(a, b, obstacles, eps=1e-10) -> bool:
    """Return True if segment a-b intersects or lies inside any obstacle."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    for obs in obstacles:
        poly = _poly_vertices(obs)

        if _point_in_poly(a, poly, eps) or _point_in_poly(b, poly, eps):
            return True

        n = poly.shape[0]
        for i in range(n):
            c = poly[i]
            d = poly[(i + 1) % n]
            if _segments_intersect(a, b, c, d, eps):
                return True

    return False


def path_is_collision_free(path, obstacles, bounds=None, eps=1e-10) -> bool:
    """
    Validate a 2D path against polygon obstacles and optional workspace bounds.

    This is used only for filtering Gaussian posterior predictive trajectories
    before plotting. It does not change the probabilistic inference step.
    """
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 2:
        return False
    if not np.isfinite(p).all():
        return False

    if bounds is not None:
        xmin, xmax, ymin, ymax = normalize_plot_bounds(bounds)
        if np.any(p[:, 0] < xmin - eps) or np.any(p[:, 0] > xmax + eps):
            return False
        if np.any(p[:, 1] < ymin - eps) or np.any(p[:, 1] > ymax + eps):
            return False

    for obs in obstacles:
        poly = _poly_vertices(obs)
        for q in p:
            if _point_in_poly(q, poly, eps):
                return False

    for a, b in zip(p[:-1], p[1:]):
        if segment_collides_with_obstacles(a, b, obstacles, eps):
            return False

    return True


def filter_collision_free_paths(paths, obstacles, bounds=None, max_keep=None) -> List[np.ndarray]:
    """Keep only collision-free paths, preserving order."""
    valid = []
    for p in paths:
        if path_is_collision_free(p, obstacles, bounds=bounds):
            valid.append(p)
            if max_keep is not None and len(valid) >= int(max_keep):
                break
    return valid

# -----------------------------------------------------------------------------
# Probabilistic topological trajectory mixture
# -----------------------------------------------------------------------------

@dataclass
class GaussianTrajectoryMode:
    signature: Tuple[int, ...]
    probability: float
    mean: np.ndarray          # D = 2K
    cov: np.ndarray           # D x D, regularized
    samples: np.ndarray       # M x D
    weights: np.ndarray       # M
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

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
        """Sample full trajectories from the prior mixture."""
        rng = np.random.default_rng() if rng is None else rng
        sigs = self.signatures
        probs = np.array([self.modes[h].probability for h in sigs], dtype=np.float64)
        probs /= np.sum(probs)
        out = []
        for _ in range(n):
            h = sigs[int(rng.choice(len(sigs), p=probs))]
            mode = self.modes[h]
            vec = rng.multivariate_normal(mode.mean, mode.cov, method="svd")
            out.append(unflatten_path(vec))
        return out

    def sample_posterior_predictive(
        self,
        observed_prefix: np.ndarray,
        obs_steps: int,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[List[np.ndarray], Dict[Tuple[int, ...], float]]:
        """
        Sample trajectories from p(tau | observed_prefix), using Gaussian
        conditioning inside each homotopy mode.
        """
        rng = np.random.default_rng() if rng is None else rng
        posterior = self.infer_homotopy_posterior(observed_prefix, obs_steps)
        sigs = list(posterior.keys())
        probs = np.array([posterior[h] for h in sigs], dtype=np.float64)
        probs /= np.sum(probs)

        obs = resample_path(observed_prefix, obs_steps)
        y = flatten_path(obs)
        d_obs = y.shape[0]
        D = 2 * self.K

        samples = []
        for _ in range(n):
            h = sigs[int(rng.choice(len(sigs), p=probs))]
            mode = self.modes[h]

            mu = mode.mean
            cov = mode.cov + 1e-7 * np.eye(D)

            mu_a = mu[:d_obs]
            mu_b = mu[d_obs:]
            cov_aa = cov[:d_obs, :d_obs]
            cov_ab = cov[:d_obs, d_obs:]
            cov_ba = cov[d_obs:, :d_obs]
            cov_bb = cov[d_obs:, d_obs:]

            inv_term = np.linalg.solve(cov_aa + 1e-7 * np.eye(d_obs), (y - mu_a))
            cond_mean_b = mu_b + cov_ba @ inv_term
            cond_cov_b = cov_bb - cov_ba @ np.linalg.solve(cov_aa + 1e-7 * np.eye(d_obs), cov_ab)
            cond_cov_b = 0.5 * (cond_cov_b + cond_cov_b.T) + 1e-7 * np.eye(D - d_obs)

            tail = rng.multivariate_normal(cond_mean_b, cond_cov_b, method="svd")
            full = np.concatenate([y, tail])
            samples.append(unflatten_path(full))

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



def shift_hls_color(color, hue_shift=0.0, lightness_scale=1.0, saturation_scale=1.0):
    """
    Return a slightly adjusted version of a Matplotlib color.

    Used so prior/posterior bars keep the same base color as the corresponding
    mean trajectory while remaining visually distinguishable.
    """
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + hue_shift) % 1.0
    l = float(np.clip(l * lightness_scale, 0.0, 1.0))
    s = float(np.clip(s * saturation_scale, 0.0, 1.0))
    return colorsys.hls_to_rgb(h, l, s)


def add_covariance_ellipse(ax, mean_xy, cov_xy, nsig=2.0, **kwargs):
    cov_xy = 0.5 * (cov_xy + cov_xy.T)
    vals, vecs = np.linalg.eigh(cov_xy)
    vals = np.maximum(vals, 1e-12)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    width = 2.0 * nsig * math.sqrt(vals[0])
    height = 2.0 * nsig * math.sqrt(vals[1])
    ell = Ellipse(xy=mean_xy, width=width, height=height, angle=angle, fill=False, **kwargs)
    ax.add_patch(ell)
    return ell


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

    ordered_modes = sorted(mixture.modes.items(), key=lambda kv: kv[1].probability, reverse=True)
    ordered = [sig for sig, _ in ordered_modes]
    labels = [f"h{i}" for i in range(len(ordered))]

    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mode_colors = {
        sig: base_colors[k % len(base_colors)]
        for k, (sig, _) in enumerate(ordered_modes)
    }

    # Panel 1: raw generated samples and per-mode mean paths.
    setup_workspace(ax0, obstacles, start, goal, bounds, "Raw fish samples and mixture means")
    raw = list(gen_out.samples)
    if len(raw) > max_raw_samples:
        ids = np.linspace(0, len(raw) - 1, max_raw_samples).astype(int)
        raw = [raw[i] for i in ids]
    for p in raw:
        ax0.plot(p[:, 0], p[:, 1], linewidth=0.5, alpha=0.14)
    for k, (sig, mode) in enumerate(ordered_modes):
        mp = mode.mean_path
        ax0.plot(
            mp[:, 0],
            mp[:, 1],
            color=mode_colors[sig],
            linewidth=2.5,
            label=f"h{k}: pi={mode.probability:.2f}",
        )
    # ax0.legend(fontsize=8, loc="lower right")

    # Panel 2: probabilistic trajectory tubes.
    setup_workspace(ax1, obstacles, start, goal, bounds, "Topological Gaussian trajectory tubes")
    for k, (sig, mode) in enumerate(ordered_modes):
        mp = mode.mean_path
        mean_color = mode_colors[sig]
        ax1.plot(mp[:, 0], mp[:, 1], color=mean_color, linewidth=2.0)
        for t in np.linspace(3, mixture.K - 4, 7).astype(int):
            cov_xy = mode.cov[2*t:2*t+2, 2*t:2*t+2]
            add_covariance_ellipse(
                ax1,
                mp[t],
                cov_xy,
                nsig=2.0,
                edgecolor=mean_color,
                linewidth=0.8,
                alpha=0.45,
            )

    # Panel 3: prior and posterior over homotopy modes.
    prior = np.array([mixture.modes[h].probability for h in ordered])
    post = np.array([posterior.get(h, 0.0) for h in ordered])
    x = np.arange(len(ordered))
    width = 0.38

    prior_colors = [
        shift_hls_color(mode_colors[h], hue_shift=-0.015, lightness_scale=1.18, saturation_scale=0.82)
        for h in ordered
    ]
    posterior_colors = [
        shift_hls_color(mode_colors[h], hue_shift=0.015, lightness_scale=0.82, saturation_scale=1.10)
        for h in ordered
    ]

    ax2.bar(
        x - width / 2,
        prior,
        width,
        color=prior_colors,
        edgecolor=[mode_colors[h] for h in ordered],
        linewidth=0.8,
        label="prior p(h)",
    )
    ax2.bar(
        x + width / 2,
        post,
        width,
        color=posterior_colors,
        edgecolor=[mode_colors[h] for h in ordered],
        linewidth=0.8,
        label="posterior p(h | prefix)",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0)
    ax2.set_ylabel("probability")
    ax2.set_title("Bayesian homotopy belief")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.25)

    # Panel 4: posterior predictive samples.
    # Gaussian conditioning can produce trajectories with support inside obstacles.
    # For this visualization, keep the previous Gaussian posterior predictive model
    # but plot only samples whose polyline is collision-free.
    free_posterior_samples = filter_collision_free_paths(
        posterior_samples,
        obstacles=obstacles,
        bounds=bounds,
        max_keep=80,
    )

    setup_workspace(
        ax3,
        obstacles,
        start,
        goal,
        bounds,
        f"Posterior predictive trajectories, collision-free only ({len(free_posterior_samples)}/{len(posterior_samples)})",
    )
    for p in free_posterior_samples:
        ax3.plot(p[:, 0], p[:, 1], linewidth=0.7, alpha=0.24)

    if len(free_posterior_samples) == 0:
        ax3.text(
            0.5,
            0.5,
            "No collision-free Gaussian posterior samples\nTry increasing n or reducing covariance_jitter",
            transform=ax3.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )

    ax3.plot(observed_prefix[:, 0], observed_prefix[:, 1], linewidth=4.0, label="observed prefix")
    # ax3.legend(loc="lower right")

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
        PolyObstacle(round_obstacle(np.array([[1.8, 4.2], [2.7, 4.0], [3.0, 4.8], [2.3, 5.3], [1.7, 4.9]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[4.6, 5.1], [5.4, 5.0], [5.8, 5.7], [5.0, 6.2], [4.4, 5.7]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[7.9, 3.0], [9.0, 3.2], [8.8, 4.2], [7.7, 4.0]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(np.array([[5.7, 1.0], [6.6, 1.2], [6.4, 2.3], [5.6, 2.1]]), n_iters=4, n_points=32)),
    ]

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
    #       p(tau,h) = pi_h N(tau; mu_h, Sigma_h)
    #    with quality-weighted pi_h and Gaussian trajectory tubes per mode.
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
    )

    print("\nPosterior p(h | observed prefix):")
    for sig, prob in sorted(posterior.items(), key=lambda kv: kv[1], reverse=True):
        print(" ", sig, prob)

    n_free = sum(
        path_is_collision_free(p, obstacles, bounds=bounds_xy)
        for p in posterior_samples
    )
    print(f"\nCollision-free posterior predictive samples: {n_free}/{len(posterior_samples)}")

    # 4. Visualize samples, fitted probability tubes, posterior homotopy belief,
    #    and posterior predictive trajectories.
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