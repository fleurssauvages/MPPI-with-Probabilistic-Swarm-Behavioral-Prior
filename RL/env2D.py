import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numba import njit, prange
import pickle

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

def get_terminal_goals(goals, goal_W, eps=1e-6):
    terminal_indices = []
    for gi in range(goal_W.shape[0]):
        row = goal_W[gi]
        self_w = row[gi]
        s_other = np.sum(row) - self_w

        if self_w > eps and s_other <= eps:
            terminal_indices.append(gi)

    return goals[terminal_indices], terminal_indices

@njit(cache=True, fastmath=True)
def _lcg_rand01(seed_arr):
    m = 4294967296.0
    a = 1664525.0
    c = 1.0
    seed = seed_arr[0]
    seed = (a * seed + c) % m
    seed_arr[0] = seed
    return seed / m


@njit(cache=True, fastmath=True)
def _cubic_interpolate(v0, v1, v2, v3, x):
    x2 = x * x
    a0 = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3
    a1 = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3
    a2 = -0.5 * v0 + 0.5 * v2
    a3 = v1
    return a0 * x * x2 + a1 * x2 + a2 * x + a3


@njit(cache=True, fastmath=True)
def _noise(time, cum_wavlen, rv0, rv1, rv2, rv3, seed_arr):
    wavelen = 0.3
    if time >= cum_wavlen:
        cum_wavlen = cum_wavlen + wavelen
        rv0, rv1, rv2 = rv1, rv2, rv3
        rv3 = _lcg_rand01(seed_arr)

    frac = (time % wavelen) / wavelen
    value = _cubic_interpolate(rv0, rv1, rv2, rv3, frac)
    return (value * 2.0 - 1.0), cum_wavlen, rv0, rv1, rv2, rv3


# -----------------------------
# Small vector helpers
# -----------------------------
@njit(cache=True, fastmath=True)
def _norm2(v0, v1):
    return math.sqrt(v0 * v0 + v1 * v1)


@njit(cache=True, fastmath=True)
def _clamp_len2(v0, v1, max_len):
    n = _norm2(v0, v1)
    if n <= 1e-12:
        return 0.0, 0.0
    if n > max_len:
        s = max_len / n
        return v0 * s, v1 * s
    return v0, v1


@njit(cache=True, fastmath=True)
def _bounds_steer2(px, py, bound_size):
    # steer back inside [0,bound]
    min_b = 0.0
    max_b = bound_size
    sx = 0.0
    sy = 0.0

    if px < min_b:
        sx = min_b - px
    elif px > max_b:
        sx = max_b - px

    if py < min_b:
        sy = min_b - py
    elif py > max_b:
        sy = max_b - py

    return sx, sy


# -----------------------------
# Reynolds boids (2D)
# -----------------------------
@njit(cache=True, fastmath=True)
def _reynolds2(i, pos, vel, count, sep_r, ali_r, coh_r):
    px, py = pos[i, 0], pos[i, 1]

    sep0 = sep1 = 0.0
    ali0 = ali1 = 0.0
    coh0 = coh1 = 0.0
    max_d2 = max(sep_r * sep_r, ali_r * ali_r, coh_r * coh_r)

    for j in range(count):
        if j == i:
            continue
        dx = px - pos[j, 0]
        dy = py - pos[j, 1]
        d2 = dx * dx + dy * dy
        if d2 <= 1e-24 or d2 > max_d2:
            continue

        d = math.sqrt(d2)

        if d2 < sep_r * sep_r:
            mag = 1.0 - d / sep_r
            sep0 += (dx / d) * mag
            sep1 += (dy / d) * mag

        if d2 < ali_r * ali_r:
            mag = 1.0 - d / ali_r
            vx, vy = vel[j, 0], vel[j, 1]
            vn = _norm2(vx, vy)
            if vn > 1e-12:
                ali0 += (vx / vn) * mag
                ali1 += (vy / vn) * mag

        if d2 < coh_r * coh_r:
            mag = 1.0 - d / coh_r
            coh0 += (-dx / d) * mag
            coh1 += (-dy / d) * mag

    sep0, sep1 = _clamp_len2(sep0, sep1, 1.0)
    ali0, ali1 = _clamp_len2(ali0, ali1, 1.0)
    coh0, coh1 = _clamp_len2(coh0, coh1, 1.0)
    return sep0, sep1, ali0, ali1, coh0, coh1


# -----------------------------
# Segment DF (2D) + AF from DF + sampling
# -----------------------------
@njit(cache=True, fastmath=True)
def _point_segment_dist2(px, py, x1, y1, x2, y2):
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        dx = px - x1
        dy = py - y1
        return dx * dx + dy * dy

    t = (wx * vx + wy * vy) / vv
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    cx = x1 + t * vx
    cy = y1 + t * vy
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy


@njit(cache=True, fastmath=True, parallel=True)
def build_distance_field_2d(segs, origin, spacing, R, avoid_r):
    """
    DF stores normalized distance d/avoid_r, clamped to 1 beyond avoid_r.
    df=1 => far from obstacles, df~0 => on obstacle.
    """
    df = np.empty((R, R), dtype=np.float32)
    avoid2 = avoid_r * avoid_r

    for idx in prange(R * R):
        i = idx // R
        j = idx % R

        px = origin[0] + spacing * i
        py = origin[1] + spacing * j

        best2 = avoid2
        for m in range(segs.shape[0]):
            x1 = segs[m, 0]
            y1 = segs[m, 1]
            x2 = segs[m, 2]
            y2 = segs[m, 3]
            d2 = _point_segment_dist2(px, py, x1, y1, x2, y2)
            if d2 < best2:
                best2 = d2
                if best2 <= 1e-12:
                    break

        d = math.sqrt(best2)
        if d >= avoid_r:
            df[i, j] = 1.0
        else:
            df[i, j] = d / avoid_r

    return df


@njit(cache=True, fastmath=True)
def build_avoid_field_from_df_2d(df, power=1.0, alpha=4.0):
    """
    Build AF by taking a local stencil gradient of DF and mapping proximity -> magnitude.
    af points "away from obstacles" (towards increasing df) with exponential barrier.
    """
    R = df.shape[0]
    af = np.zeros((R, R, 2), dtype=np.float32)

    for i in range(1, R - 1):
        for j in range(1, R - 1):
            s0 = 0.0
            s1 = 0.0

            # 8-neighborhood
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    w = df[i + di, j + dj]
                    s0 += di * w
                    s1 += dj * w

            n2 = s0 * s0 + s1 * s1
            if n2 <= 1e-18:
                continue
            invn = 1.0 / math.sqrt(n2)
            nx = s0 * invn
            ny = s1 * invn

            x = 1.0 - df[i, j]  # proximity in [0,1]
            if x <= 0.0:
                continue
            if power != 1.0:
                x = x ** power

            mag = math.exp(alpha * x) - 1.0

            af[i, j, 0] = nx * mag
            af[i, j, 1] = ny * mag

    return af


@njit(cache=True, fastmath=True)
def sample_field_2d_bilinear(field, px, py, origin, spacing):
    """
    Bilinear sample of field (R,R,C) at world point (px,py).
    """
    R = field.shape[0]

    fx = (px - origin[0]) / spacing
    fy = (py - origin[1]) / spacing

    ix = int(math.floor(fx))
    iy = int(math.floor(fy))
    tx = fx - ix
    ty = fy - iy

    if ix < 0:
        ix = 0
        tx = 0.0
    if iy < 0:
        iy = 0
        ty = 0.0
    if ix > R - 2:
        ix = R - 2
        tx = 1.0
    if iy > R - 2:
        iy = R - 2
        ty = 1.0

    v00 = field[ix, iy]
    v10 = field[ix + 1, iy]
    v01 = field[ix, iy + 1]
    v11 = field[ix + 1, iy + 1]

    out0 = (v00[0] * (1 - tx) + v10[0] * tx) * (1 - ty) + (v01[0] * (1 - tx) + v11[0] * tx) * ty
    out1 = (v00[1] * (1 - tx) + v10[1] * tx) * (1 - ty) + (v01[1] * (1 - tx) + v11[1] * tx) * ty
    return out0, out1


@njit(cache=True, fastmath=True)
def sample_df_2d(df, px, py, origin, spacing):
    """
    Bilinear sample of scalar DF (R,R).
    """
    R = df.shape[0]

    fx = (px - origin[0]) / spacing
    fy = (py - origin[1]) / spacing

    ix = int(math.floor(fx))
    iy = int(math.floor(fy))
    tx = fx - ix
    ty = fy - iy

    if ix < 0:
        ix = 0
        tx = 0.0
    if iy < 0:
        iy = 0
        ty = 0.0
    if ix > R - 2:
        ix = R - 2
        tx = 1.0
    if iy > R - 2:
        iy = R - 2
        ty = 1.0

    v00 = df[ix, iy]
    v10 = df[ix + 1, iy]
    v01 = df[ix, iy + 1]
    v11 = df[ix + 1, iy + 1]

    v0 = v00 * (1 - tx) + v10 * tx
    v1 = v01 * (1 - tx) + v11 * tx
    return v0 * (1 - ty) + v1 * ty


def precompute_segments_avoidance_2d(
    segs: np.ndarray,
    origin: np.ndarray,
    field_length: float,
    R: int,
    avoid_r: float,
    power: float = 1.0,
    alpha: float = 4.0,
):
    origin = np.asarray(origin, dtype=np.float32)
    segs = np.asarray(segs, dtype=np.float32)
    spacing = np.float32(field_length / (R - 1))

    df = build_distance_field_2d(segs, origin, spacing, int(R), np.float32(avoid_r))
    af = build_avoid_field_from_df_2d(df, power=np.float32(power), alpha=np.float32(alpha))
    return af, df, origin, spacing


# -----------------------------
# Goal switching
# -----------------------------
@njit(cache=True, fastmath=True)
def _weighted_next_goal(row_w, u01):
    s = 0.0
    for k in range(row_w.shape[0]):
        s += row_w[k]
    if s <= 1e-12:
        return int(u01 * row_w.shape[0])

    thresh = u01 * s
    c = 0.0
    for k in range(row_w.shape[0]):
        c += row_w[k]
        if c >= thresh:
            return k
    return row_w.shape[0] - 1



@njit(cache=True, fastmath=True)
def _is_terminal_goal(goal_W_row, gi, eps=1e-6):
    """
    Terminal if the transition row is (approximately) a pure self-loop:
      W[gi,gi] > 0 and sum(other) ~ 0
    """
    self_w = goal_W_row[gi]
    if self_w <= eps:
        return False
    s_other = 0.0
    for k in range(goal_W_row.shape[0]):
        if k == gi:
            continue
        s_other += goal_W_row[k]
    return s_other <= eps


@njit(cache=True, fastmath=True)
def update_goal_events_2d(
    boid_pos,
    alive,
    ever_hit,
    first_hit_t,
    goal_idx,
    goals,
    goal_W,
    goal_radius,
    step_idx,
    dt,
    seed_arr,
):
    gr2 = goal_radius * goal_radius
    t_now = (step_idx + 1) * dt
    n_active_after = 0

    for i in range(boid_pos.shape[0]):
        if not alive[i]:
            continue

        gi = goal_idx[i]
        gx = goals[gi, 0]
        gy = goals[gi, 1]

        dx = boid_pos[i, 0] - gx
        dy = boid_pos[i, 1] - gy
        d2g = dx * dx + dy * dy

        if d2g <= gr2:
            # Check terminal status of the CURRENT goal
            terminal = _is_terminal_goal(goal_W[gi], gi)

            if terminal:
                # Arrived for real: mark and deactivate
                if not ever_hit[i]:
                    ever_hit[i] = True
                    first_hit_t[i] = t_now
                continue
            else:
                # Not terminal: switch goal, but keep this boid active
                u = _lcg_rand01(seed_arr)
                nxt = _weighted_next_goal(goal_W[gi], u)
                goal_idx[i] = nxt
                n_active_after += 1
        else:
            # Still en route
            n_active_after += 1

    return n_active_after

# -----------------------------
# Heading diversity (entropy)
# -----------------------------
@njit(cache=True, fastmath=True)
def heading_entropy2d(vel, n_ang=16):
    counts = np.zeros(n_ang, dtype=np.int32)
    total = 0

    two_pi = 2.0 * math.pi
    inv_two_pi = 1.0 / two_pi

    for i in range(vel.shape[0]):
        vx = vel[i, 0]
        vy = vel[i, 1]
        sp2 = vx * vx + vy * vy
        if sp2 <= 1e-24:
            continue

        ang = math.atan2(vy, vx)  # [-pi, pi]
        b = int(math.floor((ang + math.pi) * inv_two_pi * n_ang))
        if b < 0:
            b = 0
        if b >= n_ang:
            b = n_ang - 1
        counts[b] += 1
        total += 1

    if total == 0:
        return 0.0

    H = 0.0
    inv_total = 1.0 / total
    for b in range(n_ang):
        c = counts[b]
        if c > 0:
            p = c * inv_total
            H -= p * math.log(p)

    Hmax = math.log(n_ang)
    return H / (Hmax + 1e-12)


# -----------------------------
# Simulation step (Numba)
# -----------------------------
@njit(cache=True, fastmath=True, parallel=True)
def step_sim2d_df(
    boid_pos,
    boid_vel,
    boid_time,
    boid_noise_cum,
    boid_noise_vals,
    seed_arr,
    dt,
    *,
    bound_size,
    boid_count,
    rule_scalar,
    max_speed,
    sep_r,
    ali_r,
    coh_r,
    sep_s,
    ali_s,
    coh_s,
    bnd_s,
    rand_s,
    obs_avoid_s,
    rand_wavelen_scalar,
    goal_gain,
    goals,
    goal_idx,
    alive,
    mesh_af2d,
    mesh_origin2d,
    mesh_spacing2d,
    # optional "hard collision" using DF
    mesh_df2d,
    df_kill_thresh,
):
    if (max_speed == 0.0) or boid_count <= 0:
        return

    for i in prange(boid_count):
        if not alive[i]:
            continue

        boid_time[i] += dt
        ax = ay = 0.0

        # Optional: "kill" if too close to obstacle based on DF threshold
        if mesh_df2d is not None and df_kill_thresh > 0.0:
            dfn = sample_df_2d(mesh_df2d, boid_pos[i, 0], boid_pos[i, 1], mesh_origin2d, mesh_spacing2d)
            if dfn <= df_kill_thresh:
                alive[i] = False
                continue

        sep0, sep1, ali0, ali1, coh0, coh1 = _reynolds2(i, boid_pos, boid_vel, boid_count, sep_r, ali_r, coh_r)

        ax += sep0 * sep_s
        ay += sep1 * sep_s

        ax += ali0 * ali_s
        ay += ali1 * ali_s

        ax += coh0 * coh_s
        ay += coh1 * coh_s

        # bounds
        sx, sy = _bounds_steer2(boid_pos[i, 0], boid_pos[i, 1], bound_size)
        ax += sx * bnd_s
        ay += sy * bnd_s

        # random (smooth) noise
        if rand_s != 0.0:
            t = boid_time[i] * rand_wavelen_scalar * math.sqrt(dt)

            # x channel
            rv0 = boid_noise_vals[i, 0, 0]
            rv1 = boid_noise_vals[i, 0, 1]
            rv2 = boid_noise_vals[i, 0, 2]
            rv3 = boid_noise_vals[i, 0, 3]
            nx, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.0, boid_noise_cum[i, 0], rv0, rv1, rv2, rv3, seed_arr)
            boid_noise_cum[i, 0] = cwl
            boid_noise_vals[i, 0, 0] = rv0
            boid_noise_vals[i, 0, 1] = rv1
            boid_noise_vals[i, 0, 2] = rv2
            boid_noise_vals[i, 0, 3] = rv3

            # y channel
            rv0 = boid_noise_vals[i, 1, 0]
            rv1 = boid_noise_vals[i, 1, 1]
            rv2 = boid_noise_vals[i, 1, 2]
            rv3 = boid_noise_vals[i, 1, 3]
            ny, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.1, boid_noise_cum[i, 1], rv0, rv1, rv2, rv3, seed_arr)
            boid_noise_cum[i, 1] = cwl
            boid_noise_vals[i, 1, 0] = rv0
            boid_noise_vals[i, 1, 1] = rv1
            boid_noise_vals[i, 1, 2] = rv2
            boid_noise_vals[i, 1, 3] = rv3

            ax += nx * rand_s
            ay += ny * rand_s

        # obstacle avoid: sample precomputed AF
        if mesh_af2d is not None and obs_avoid_s != 0.0:
            ox, oy = sample_field_2d_bilinear(mesh_af2d, boid_pos[i, 0], boid_pos[i, 1], mesh_origin2d, mesh_spacing2d)
            ax += obs_avoid_s * ox
            ay += obs_avoid_s * oy

        # goal attraction
        if goal_gain != 0.0:
            gi = goal_idx[i]
            gx = goals[gi, 0]
            gy = goals[gi, 1]

            dx = gx - boid_pos[i, 0]
            dy = gy - boid_pos[i, 1]
            d = math.sqrt(dx * dx + dy * dy) + 1e-12

            pwr = 3.0
            mag = goal_gain * (d ** pwr)
            mag = min(mag, goal_gain * 8.0)

            ax += (dx / d) * mag
            ay += (dy / d) * mag

        # integrate
        ax *= rule_scalar
        ay *= rule_scalar

        boid_vel[i, 0] += ax * dt
        boid_vel[i, 1] += ay * dt
        boid_vel[i, 0], boid_vel[i, 1] = _clamp_len2(boid_vel[i, 0], boid_vel[i, 1], max_speed)

        boid_pos[i, 0] += boid_vel[i, 0] * dt
        boid_pos[i, 1] += boid_vel[i, 1] * dt


# -----------------------------
# Init agents
# -----------------------------
def _init_agents2d(total_boids, start_xy, start_spread, seed=0.1):
    sx, sy = float(start_xy[0]), float(start_xy[1])

    boid_pos = np.empty((total_boids, 2), dtype=np.float32)
    boid_vel = np.zeros((total_boids, 2), dtype=np.float32)
    boid_time = np.zeros((total_boids,), dtype=np.float32)

    rng = np.random.default_rng(int(seed * 1e6) % (2**32 - 1))
    offsets = rng.normal(0.0, 1.0, size=(total_boids, 2))
    norms = np.linalg.norm(offsets, axis=1) + 1e-12
    offsets = offsets / norms[:, None]
    radii = rng.random(total_boids) ** 0.5
    offsets = offsets * (radii[:, None] * float(start_spread))

    boid_pos[:, 0] = sx + offsets[:, 0]
    boid_pos[:, 1] = sy + offsets[:, 1]

    boid_noise_cum = np.zeros((total_boids, 2), dtype=np.float32)
    boid_noise_vals = np.empty((total_boids, 2, 4), dtype=np.float32)

    seed_arr = np.array([math.floor(seed * 4294967296.0)], dtype=np.float32)

    for i in range(total_boids):
        for a in range(2):
            boid_noise_vals[i, a, 0] = _lcg_rand01(seed_arr)
            boid_noise_vals[i, a, 1] = _lcg_rand01(seed_arr)
            boid_noise_vals[i, a, 2] = _lcg_rand01(seed_arr)
            boid_noise_vals[i, a, 3] = _lcg_rand01(seed_arr)

    seed_arr[0] = math.floor(seed * 4294967296.0)
    return boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals, seed_arr


# -----------------------------
# Metrics
# -----------------------------
@dataclass
class EpisodeMetrics:
    frac_goal: float
    avg_time_to_goal: float
    diversity_entropy: float


# -----------------------------
# Gym environment
# -----------------------------
class FishGoalEnv2D(gym.Env):
    """
    2D environment that keeps your original "mesh df/af" philosophy:
    - user provides a "mesh" (here: segments)
    - precompute df + af once
    - runtime: sample af at boid positions (fast)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        boid_count: int = 200,
        bound: float = 40.0,
        max_steps: int = 2000,
        dt: float = 0.5,
        start_spread: float = 3.0,
        goal_radius: float = 2.0,
        seed: int = 0,
        # reward weights
        w_goal: float = 1.0,
        w_time: float = 0.2,
        w_div: float = 0.1,
        # fixed sim params
        rule_scalar: float = 1.0,
        max_speed: float = 0.18,
        sep_r: float = 1.6,
        ali_r: float = 4.0,
        coh_r: float = 5.5,
        rand_wavelen_scalar: float = 0.01,
        # goal graph
        start: np.ndarray = np.array([6.0, 20.0], dtype=np.float32),
        goals: Optional[np.ndarray] = None,     # (G,2) or (G,3) (will take [:,:2])
        goal_W: Optional[np.ndarray] = None,    # (G,G)
        start_goal_idx: Optional[int] = 0,
        # "mesh" as segments + df/af settings
        segs: Optional[np.ndarray] = None,      # (M,4)
        df_origin: np.ndarray = np.array([0.0, 0.0], dtype=np.float32),
        df_length: float = 40.0,
        df_R: int = 256,
        avoid_r: float = 2.5,
        avoid_power: float = 1.0,
        avoid_alpha: float = 4.0,
        df_kill_thresh: float = 0.0,            # if >0: kill boid when df <= thresh (e.g. 0.02)
        # debug/visual
        doAnimation: bool = False,
        returnTrajectory: bool = False,
        returnGoalsIdx: bool = False,
    ):
        super().__init__()

        self.start = np.asarray(start, dtype=np.float32)

        if goals is None:
            self.goals = np.asarray([[34.0, 20.0]], dtype=np.float32)
        else:
            g = np.asarray(goals, dtype=np.float32)
            if g.ndim != 2:
                raise ValueError("goals must be (G,2) or (G,3)")
            if g.shape[1] == 3:
                g = g[:, :2]
            if g.shape[1] != 2:
                raise ValueError("goals must be (G,2) or (G,3)")
            self.goals = g

        G = self.goals.shape[0]
        if goal_W is None:
            self.goal_W = np.ones((G, G), dtype=np.float32)
            np.fill_diagonal(self.goal_W, 0.0)
        else:
            self.goal_W = np.asarray(goal_W, dtype=np.float32)
            if self.goal_W.shape != (G, G):
                raise ValueError(f"goal_W must be {(G,G)}")
        self.start_goal_idx = start_goal_idx

        self.boid_count = int(boid_count)
        self.bound = float(bound)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.start_spread = float(start_spread)
        self.goal_radius = float(goal_radius)

        self.rule_scalar = float(rule_scalar)
        self.max_speed = float(max_speed)
        self.sep_r = float(sep_r)
        self.ali_r = float(ali_r)
        self.coh_r = float(coh_r)
        self.rand_wavelen_scalar = float(rand_wavelen_scalar)

        self.w_goal = float(w_goal)
        self.w_time = float(w_time)
        self.w_div = float(w_div)

        # IMPORTANT: (7,) consistent everywhere
        self.action_space = spaces.Box(low=0.0, high=10.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self._rng = np.random.default_rng(int(seed))
        self._episode_seed: Optional[int] = None

        self._alive = np.empty((self.boid_count,), dtype=np.bool_)

        # optional trajectories
        self.returnTrajectory = bool(returnTrajectory)
        if self.returnTrajectory:
            self.trajectory_boid_pos = np.empty((self.max_steps, self.boid_count, 2), dtype=np.float32)
            self.trajectory_boid_vel = np.empty((self.max_steps, self.boid_count, 2), dtype=np.float32)
        self.returnGoalsIdx = bool(returnGoalsIdx)
        if self.returnGoalsIdx:
            self.trajectory_goal_idx = np.empty((self.max_steps,self.boid_count), dtype=np.int32)

        # --- DF/AF mesh precompute (segments) ---
        self.segs = None if segs is None else np.asarray(segs, dtype=np.float32)
        self.mesh_af2d = None
        self.mesh_df2d = None
        self.mesh_origin2d = np.asarray(df_origin, dtype=np.float32)
        self.mesh_spacing2d = np.float32(df_length / (df_R - 1))
        self.df_kill_thresh = float(df_kill_thresh)

        if self.segs is not None and self.segs.size > 0:
            af, df, origin, spacing = precompute_segments_avoidance_2d(
                self.segs,
                origin=self.mesh_origin2d,
                field_length=float(df_length),
                R=int(df_R),
                avoid_r=float(avoid_r),
                power=float(avoid_power),
                alpha=float(avoid_alpha),
            )
            self.mesh_af2d = af
            self.mesh_df2d = df
            self.mesh_origin2d = origin
            self.mesh_spacing2d = spacing

        # animation
        self.doAnimation = bool(doAnimation)
        self._plt = None
        self._ax = None
        self._sc = None

        if self.doAnimation:
            self._plt = plt
            self._fig, self._ax = plt.subplots(figsize=(8, 8))
            self._ax.set_xlim(0, self.bound)
            self._ax.set_ylim(0, self.bound)
            self._ax.set_xlabel("")
            self._ax.set_ylabel("")
            self._ax.set_xticks([])
            self._ax.set_yticks([])
            terminal_goals, idx = get_terminal_goals(self.goals, self.goal_W)
            self._ax.scatter(terminal_goals[:, 0], terminal_goals[:, 1], s=140, marker="*", c="r")

            if self.segs is not None and self.segs.size > 0:
                for m in range(self.segs.shape[0]):
                    x1, y1, x2, y2 = self.segs[m]
                    self._ax.plot([x1, x2], [y1, y2], 'k', linewidth=2)

            self._sc = self._ax.scatter([], [], s=10, c="b")
            plt.ion()
            plt.show()

        # compile numba once
        self._warmup()

    def _warmup(self):
        # compile the kernels (tiny dummy run)
        (boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals, seed_arr) = _init_agents2d(
            8, self.start, 1.0, seed=0.123
        )
        alive = np.ones((8,), dtype=np.bool_)
        gi = np.zeros((8,), dtype=np.int32)

        step_sim2d_df(
            boid_pos,
            boid_vel,
            boid_time,
            boid_noise_cum,
            boid_noise_vals,
            seed_arr,
            np.float32(self.dt),
            bound_size=np.float32(self.bound),
            boid_count=8,
            rule_scalar=np.float32(self.rule_scalar),
            max_speed=np.float32(self.max_speed),
            sep_r=np.float32(self.sep_r),
            ali_r=np.float32(self.ali_r),
            coh_r=np.float32(self.coh_r),
            sep_s=np.float32(1.0),
            ali_s=np.float32(1.0),
            coh_s=np.float32(1.0),
            bnd_s=np.float32(1.0),
            rand_s=np.float32(0.1),
            obs_avoid_s=np.float32(1.0),
            rand_wavelen_scalar=np.float32(self.rand_wavelen_scalar),
            goal_gain=np.float32(0.0),
            goals=self.goals,
            goal_idx=gi,
            alive=alive,
            mesh_af2d=self.mesh_af2d,
            mesh_origin2d=self.mesh_origin2d,
            mesh_spacing2d=self.mesh_spacing2d,
            mesh_df2d=self.mesh_df2d,
            df_kill_thresh=np.float32(self.df_kill_thresh),
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._episode_seed = int(self._rng.integers(0, 2**31 - 1))
        obs = np.zeros((6,), dtype=np.float32)
        return obs, {}

    def step(self, action, goalW=None):
        if goalW is not None:
            self.goal_W = goalW
        if self._episode_seed is None:
            raise RuntimeError("Call reset() before step().")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"action must have shape (7,), got {action.shape}")

        metrics, info = self._rollout_episode(action)

        time_pen = 0.0
        if not math.isnan(metrics.avg_time_to_goal):
            time_pen = metrics.avg_time_to_goal / (self.max_steps * self.dt + 1e-12)

        reward = (
            self.w_goal * metrics.frac_goal
            - self.w_time * time_pen
            + self.w_div * metrics.diversity_entropy
        )

        info.update(
            {
                "frac_goal": metrics.frac_goal,
                "avg_time_to_goal": metrics.avg_time_to_goal,
                "diversity_entropy": metrics.diversity_entropy,
                "reward": float(reward),
            }
        )

        obs = np.zeros((6,), dtype=np.float32)
        self._episode_seed = None
        return obs, float(reward), True, False, info

    def _rollout_episode(self, action: np.ndarray) -> Tuple[EpisodeMetrics, Dict]:
        seed = int(self._episode_seed)
        rng = np.random.default_rng(seed)

        (boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals, seed_arr) = _init_agents2d(
            total_boids=self.boid_count,
            start_xy=self.start,
            start_spread=self.start_spread,
            seed=float((seed % 1000000) / 1000000.0 + 0.123),
        )

        goal_idx = np.empty((self.boid_count,), dtype=np.int32)
        if self.start_goal_idx is None:
            goal_idx[:] = rng.integers(0, self.goals.shape[0], size=self.boid_count, dtype=np.int32)
        else:
            goal_idx[:] = int(self.start_goal_idx)

        ever_hit = np.zeros((self.boid_count,), dtype=np.bool_)
        first_hit_t = np.full((self.boid_count,), np.nan, dtype=np.float32)

        sep_s = float(action[0])
        ali_s = float(action[1])
        coh_s = float(action[2])
        bnd_s = float(action[3])
        rand_s = float(action[4])
        obs_avoid_s = float(action[5])
        goal_gain = float(action[6])

        self._alive.fill(True)

        n_active = self.boid_count
        for step in range(self.max_steps):
            step_sim2d_df(
                boid_pos,
                boid_vel,
                boid_time,
                boid_noise_cum,
                boid_noise_vals,
                seed_arr,
                np.float32(self.dt),
                bound_size=np.float32(self.bound),
                boid_count=self.boid_count,
                rule_scalar=np.float32(self.rule_scalar),
                max_speed=np.float32(self.max_speed),
                sep_r=np.float32(self.sep_r),
                ali_r=np.float32(self.ali_r),
                coh_r=np.float32(self.coh_r),
                sep_s=np.float32(sep_s),
                ali_s=np.float32(ali_s),
                coh_s=np.float32(coh_s),
                bnd_s=np.float32(bnd_s),
                rand_s=np.float32(rand_s),
                obs_avoid_s=np.float32(obs_avoid_s),
                rand_wavelen_scalar=np.float32(self.rand_wavelen_scalar),
                goal_gain=np.float32(goal_gain),
                goals=self.goals,
                goal_idx=goal_idx,
                alive=self._alive,
                mesh_af2d=self.mesh_af2d,
                mesh_origin2d=self.mesh_origin2d,
                mesh_spacing2d=self.mesh_spacing2d,
                mesh_df2d=self.mesh_df2d,
                df_kill_thresh=np.float32(self.df_kill_thresh),
            )

            if self.returnGoalsIdx:
                self.trajectory_goal_idx[step, :] = goal_idx

            # goal switching
            n_active = update_goal_events_2d(
                boid_pos,
                self._alive,
                ever_hit,
                first_hit_t,
                goal_idx,
                self.goals,
                self.goal_W,
                np.float32(self.goal_radius),
                step,
                np.float32(self.dt),
                seed_arr,
            )

            if self.returnTrajectory:
                self.trajectory_boid_pos[step, :, :] = boid_pos
                self.trajectory_boid_vel[step, :, :] = boid_vel

            if self.doAnimation and self._plt is not None and self._ax is not None:
                self._sc.set_offsets(boid_pos)
                self._plt.pause(0.001)
                plt.savefig(f"frames/frame_{step:04d}.png")
                # if step == 150:
                #     fig = plt.gcf()
                #     fig.savefig("save/figures/"+"boids"+".svg", dpi=1200, bbox_inches="tight")


            if n_active == 0:
                break

        frac_goal = 1.0
        avg_time_to_goal = float(np.nanmean(first_hit_t)) if np.any(ever_hit) else float("nan")
        diversity = float(heading_entropy2d(boid_vel))

        metrics = EpisodeMetrics(frac_goal=frac_goal, avg_time_to_goal=avg_time_to_goal, diversity_entropy=diversity)

        info = {
            "start": self.start.copy(),
            "reached_count": int(np.sum(ever_hit)),
            "steps_executed": int(step + 1) if self.max_steps > 0 else 0,
        }
        if self.returnTrajectory:
            info["trajectory_boid_pos"] = self.trajectory_boid_pos
            info["trajectory_boid_vel"] = self.trajectory_boid_vel
        if self.returnGoalsIdx:
            info["trajectory_goal_idx"] = self.trajectory_goal_idx

        return metrics, info

# -----------------------------
# Example: "short maze" with segments
# -----------------------------
if __name__ == "__main__":
    start = np.array([20.0, 20.0])
    goals = np.array([[20.0, 20.0],
                      [4.0, 20.0],
                      [36.0, 20.0],
                      [4.0, 4.0],
                      [20.0, 4.0],
                      [36.0, 36.0],
                      [20.0, 36.0],])
    
    W = np.array([
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    segs = np.array([[0, 25, 32, 25],
                     [32, 25, 32, 31],
                     [32, 31, 0, 31],
                     [40, 15, 8, 15],
                     [8, 15, 8, 9],
                     [8, 9, 40, 9],
    ], dtype=np.int32)

    load_theta = True
    if load_theta:
        theta_path = "save/best_policy.pkl"
        action = pickle.load(open(theta_path, "rb"))['best_theta']
    else:
        action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    env = FishGoalEnv2D(
        boid_count=250,
        bound=40.0,
        max_steps=600,
        dt=0.5,
        start=np.array(start, dtype=np.float32),
        start_spread=2.0,
        goals=goals,
        goal_W=W,
        start_goal_idx=0,
        segs=segs,
        df_origin=np.array([0.0, 0.0], dtype=np.float32),
        df_length=40.0,
        df_R=256,
        avoid_r=2.5,
        avoid_power=1.0,
        avoid_alpha=4.0,
        df_kill_thresh=0.0,  # set e.g. 0.02 to "kill" when too near walls
        doAnimation=True,
        returnTrajectory=False,
    )

    for _ in range(10):
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(action)
