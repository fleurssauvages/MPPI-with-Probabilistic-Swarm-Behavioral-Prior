import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
import pickle

import numpy as np
from numba import njit, prange

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from geometry.utils import round_obstacle, PolyObstacle, obstacles_to_segs

# -----------------------------
# Small public helpers
# -----------------------------
def _as_xy_array(arr, *, name: str) -> np.ndarray:
    """Accept (2,), (3,), (N,2), or (N,3); return float32 XY array."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        if a.shape[0] == 3:
            return a[:2].astype(np.float32)
        if a.shape[0] == 2:
            return a.astype(np.float32)
        raise ValueError(f"{name} must have length 2 or 3, got {a.shape}")
    if a.ndim == 2:
        if a.shape[1] == 3:
            return a[:, :2].astype(np.float32)
        if a.shape[1] == 2:
            return a.astype(np.float32)
        raise ValueError(f"{name} must have shape (N,2) or (N,3), got {a.shape}")
    raise ValueError(f"{name} must be 1D or 2D, got {a.shape}")


def get_terminal_goals(goals, goal_W, eps=1e-6):
    terminal_indices = []
    for gi in range(goal_W.shape[0]):
        row = goal_W[gi]
        self_w = row[gi]
        s_other = np.sum(row) - self_w
        if self_w > eps and s_other <= eps:
            terminal_indices.append(gi)
    return goals[terminal_indices], terminal_indices


# -----------------------------
# Random/noise helpers
# -----------------------------
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
def _smoothstep01(x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)

@njit(cache=True, fastmath=True)
def _noise(time, cum_wavlen, rv0, rv1, rv2, rv3, seed_arr):
    wavelen = 10.0
    if time >= cum_wavlen:
        cum_wavlen = cum_wavlen + wavelen
        rv0, rv1, rv2 = rv1, rv2, rv3
        rv3 = _lcg_rand01(seed_arr)

    frac = (time % wavelen) / wavelen
    value = _cubic_interpolate(rv0, rv1, rv2, rv3, frac)
    return (value * 2.0 - 1.0), cum_wavlen, rv0, rv1, rv2, rv3


# -----------------------------
# Vector helpers
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
    min_b = 0.0
    max_b = bound_size
    sx = 0.0
    sy = 0.0

    if px < min_b:
        sx = min_b - px
    elif px > max_b:
        sx = max_b - px

    if py < min_b:
        sy = (min_b - py) * 2.0
    elif py > max_b:
        sy = max_b - py

    sy = sy * 2.0
    return sx, sy


# -----------------------------
# Metrics
# -----------------------------
@njit(cache=True, fastmath=True)
def _axis_std2(pos, alive):
    n_alive = 0
    cx = cy = 0.0

    for i in range(pos.shape[0]):
        if alive[i]:
            cx += pos[i, 0]
            cy += pos[i, 1]
            n_alive += 1

    if n_alive == 0:
        return 0.0, 0.0, 0

    cx /= n_alive
    cy /= n_alive

    vx = vy = 0.0
    for i in range(pos.shape[0]):
        if alive[i]:
            dx = pos[i, 0] - cx
            dy = pos[i, 1] - cy
            vx += dx * dx
            vy += dy * dy

    vx /= n_alive
    vy /= n_alive

    return math.sqrt(vx), math.sqrt(vy), n_alive


@njit(cache=True, fastmath=True)
def compute_cohesion2d(pos, alive):
    sx, sy, n_alive = _axis_std2(pos, alive)
    if n_alive == 0:
        return 0.0

    eps = 1e-12
    mean_spread = (sx + sy) / 2.0
    isotropy = 2.0 * min(sx, sy) / (sx + sy + eps)
    compactness = math.exp(-mean_spread / 40.0)
    return compactness * isotropy


@njit(cache=True, fastmath=True)
def compute_dispersion2d(pos, alive):
    sx, sy, n_alive = _axis_std2(pos, alive)
    if n_alive == 0:
        return 0.0
    return math.sqrt(sx * sy)


@njit(cache=True, fastmath=True)
def compute_alignment2d(vel, alive):
    n_alive = 0
    mx = my = 0.0

    for i in range(vel.shape[0]):
        if alive[i]:
            vx = vel[i, 0]
            vy = vel[i, 1]
            vn = math.sqrt(vx * vx + vy * vy)
            if vn > 1e-12:
                mx += vx / vn
                my += vy / vn
                n_alive += 1

    if n_alive == 0:
        return 0.0

    mx /= n_alive
    my /= n_alive
    align_mag = math.sqrt(mx * mx + my * my)

    ax = abs(mx)
    ay = abs(my)
    eps = 1e-12
    isotropy = 2.0 * min(ax, ay) / (ax + ay + eps)
    return align_mag * isotropy


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

        ang = math.atan2(vy, vx)
        b = int(math.floor((ang + math.pi) * inv_two_pi * n_ang))
        if b < 0:
            b = 0
        elif b >= n_ang:
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
# Reynolds boids with start/robin predator behavior
# -----------------------------
@njit(cache=True, fastmath=True)
def _reynolds2(i, pos, vel, count, sep_r, ali_r, coh_r, pred_r, start_id, robin_id):
    px, py = pos[i, 0], pos[i, 1]
    si = start_id[i]
    ri = robin_id[i]

    sep0 = sep1 = 0.0
    ali0 = ali1 = 0.0
    coh0 = coh1 = 0.0
    pred0 = pred1 = 0.0
    max_d2 = max(sep_r * sep_r, ali_r * ali_r, coh_r * coh_r, pred_r * pred_r)

    for j in range(count):
        if j == i:
            continue

        dx = px - pos[j, 0]
        dy = py - pos[j, 1]
        d2 = dx * dx + dy * dy
        if d2 <= 1e-24 or d2 > max_d2:
            continue

        d = math.sqrt(d2)
        same_start = start_id[j] == si
        same_robin = robin_id[j] == ri

        if not same_robin:
            # Agents from other start-based robin groups are perceived as predators.
            if d2 < pred_r * pred_r:
                mag = 1.0 - d / pred_r
                pred0 += (dx / d) * mag
                pred1 += (dy / d) * mag
            continue

        if not same_start or not same_robin:
            continue

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
    pred0, pred1 = _clamp_len2(pred0, pred1, 1.0)
    return sep0, sep1, ali0, ali1, coh0, coh1, pred0, pred1


# -----------------------------
# Segment DF/AF obstacle model
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


@njit(cache=True, fastmath=True)
def _point_circle_boundary_dist2(px, py, cx, cy, r):
    dx = px - cx
    dy = py - cy
    d = math.sqrt(dx * dx + dy * dy)
    signed = d - r
    return signed * signed


@njit(cache=True, fastmath=True, parallel=True)
def build_distance_field_2d(segs, circles, origin, spacing, R, avoid_r):
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

        if best2 > 1e-12:
            for m in range(circles.shape[0]):
                cx = circles[m, 0]
                cy = circles[m, 1]
                cr = circles[m, 2]
                d2 = _point_circle_boundary_dist2(px, py, cx, cy, cr)
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
    R = df.shape[0]
    af = np.zeros((R, R, 2), dtype=np.float32)

    for i in range(1, R - 1):
        for j in range(1, R - 1):
            s0 = 0.0
            s1 = 0.0

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

            x = 1.0 - df[i, j]
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


def precompute_obstacle_avoidance_2d(
    segs: Optional[np.ndarray],
    circles: Optional[np.ndarray],
    origin: np.ndarray,
    field_length: float,
    R: int,
    avoid_r: float,
    power: float = 1.0,
    alpha: float = 4.0,
):
    origin = np.asarray(origin, dtype=np.float32)
    if segs is None:
        segs_arr = np.empty((0, 4), dtype=np.float32)
    else:
        segs_arr = np.asarray(segs, dtype=np.float32).reshape(-1, 4)

    if circles is None:
        circles_arr = np.empty((0, 3), dtype=np.float32)
    else:
        circles_arr = np.asarray(circles, dtype=np.float32).reshape(-1, 3)

    spacing = np.float32(field_length / (R - 1))
    df = build_distance_field_2d(segs_arr, circles_arr, origin, spacing, int(R), np.float32(avoid_r))
    af = build_avoid_field_from_df_2d(df, power=np.float32(power), alpha=np.float32(alpha))
    return af, df, origin, spacing


# Backward-compatible name.
def precompute_segments_avoidance_2d(
    segs: np.ndarray,
    origin: np.ndarray,
    field_length: float,
    R: int,
    avoid_r: float,
    power: float = 1.0,
    alpha: float = 4.0,
):
    return precompute_obstacle_avoidance_2d(
        segs=segs,
        circles=None,
        origin=origin,
        field_length=field_length,
        R=R,
        avoid_r=avoid_r,
        power=power,
        alpha=alpha,
    )


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
            if not ever_hit[i]:
                ever_hit[i] = True
                first_hit_t[i] = t_now

            terminal = _is_terminal_goal(goal_W[gi], gi)
            if terminal:
                alive[i] = False
                continue

            u = _lcg_rand01(seed_arr)
            nxt = _weighted_next_goal(goal_W[gi], u)
            goal_idx[i] = nxt
            n_active_after += 1
        else:
            n_active_after += 1

    return n_active_after


@njit(cache=True, fastmath=True)
def mean_time_to_goal2d(t_reach, reached):
    s = 0.0
    c = 0
    for i in range(t_reach.shape[0]):
        if reached[i]:
            v = t_reach[i]
            if not math.isnan(v):
                s += v
                c += 1
    if c == 0:
        return np.nan
    return s / c


# -----------------------------
# Simulation step
# -----------------------------
@njit(cache=True, fastmath=True, parallel=True)
def step_sim2d_df(
    boid_pos,
    boid_vel,
    boid_acc,
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
    pred_r,
    sep_s,
    ali_s,
    coh_s,
    pred_s,
    pred_goal_fade_r,
    bnd_s,
    rand_s,
    obs_avoid_s,
    rand_wavelen_scalar,
    goal_gain,
    goals,
    goal_idx,
    start_id,
    robin_id,
    alive,
    mesh_af2d,
    mesh_origin2d,
    mesh_spacing2d,
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

        if mesh_df2d is not None and df_kill_thresh > 0.0:
            dfn = sample_df_2d(mesh_df2d, boid_pos[i, 0], boid_pos[i, 1], mesh_origin2d, mesh_spacing2d)
            if dfn <= df_kill_thresh:
                alive[i] = False
                boid_vel[i, 0] = 0.0
                boid_vel[i, 1] = 0.0
                continue

        sep0, sep1, ali0, ali1, coh0, coh1, pred0, pred1 = _reynolds2(
            i, boid_pos, boid_vel, boid_count, sep_r, ali_r, coh_r, pred_r, start_id, robin_id
        )

        ax += sep0 * sep_s
        ay += sep1 * sep_s

        ax += ali0 * ali_s
        ay += ali1 * ali_s

        ax += coh0 * coh_s
        ay += coh1 * coh_s

        gi = goal_idx[i]
        gx = goals[gi, 0]
        gy = goals[gi, 1]

        gdx = gx - boid_pos[i, 0]
        gdy = gy - boid_pos[i, 1]
        goal_d = math.sqrt(gdx * gdx + gdy * gdy) + 1e-12

        pred_fade = 1.0
        if pred_goal_fade_r > 0.0:
            pred_fade = _smoothstep01(goal_d / pred_goal_fade_r)

        ax += pred0 * pred_s * pred_fade
        ay += pred1 * pred_s * pred_fade

        sx, sy = _bounds_steer2(boid_pos[i, 0], boid_pos[i, 1], bound_size)
        ax += sx * bnd_s
        ay += sy * bnd_s

        if rand_s != 0.0:
            t = boid_time[i] * rand_wavelen_scalar * math.sqrt(dt)

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

        if mesh_af2d is not None and obs_avoid_s != 0.0:
            ox, oy = sample_field_2d_bilinear(mesh_af2d, boid_pos[i, 0], boid_pos[i, 1], mesh_origin2d, mesh_spacing2d)
            ax += obs_avoid_s * ox
            ay += obs_avoid_s * oy

        if goal_gain != 0.0:
            pwr = 3.0
            mag = goal_gain * (goal_d ** pwr)
            mag = min(mag, goal_gain * 8.0)

            ax += (gdx / goal_d) * mag
            ay += (gdy / goal_d) * mag

        ax *= rule_scalar
        ay *= rule_scalar

        boid_vel[i, 0] += ax * dt
        boid_vel[i, 1] += ay * dt
        boid_vel[i, 0], boid_vel[i, 1] = _clamp_len2(boid_vel[i, 0], boid_vel[i, 1], max_speed)

        boid_pos[i, 0] += boid_vel[i, 0] * dt
        boid_pos[i, 1] += boid_vel[i, 1] * dt

        boid_acc[i, 0] = ax
        boid_acc[i, 1] = ay


# -----------------------------
# Init/action normalization
# -----------------------------
def _validate_initial_velocity(initial_velocity, starts):
    if initial_velocity is None:
        return np.zeros_like(starts, dtype=np.float32)

    initial_velocity = _as_xy_array(initial_velocity, name="initial_velocity")
    if initial_velocity.shape != starts.shape:
        raise ValueError(
            f"initial_velocity must have the same shape as starts "
            f"({starts.shape}), got {initial_velocity.shape}"
        )
    return initial_velocity.astype(np.float32)


def _normalize_action(action, sep_r=1.6, ali_r=4.0, coh_r=5.5, pred_r=8.0, pred_s=2.0):
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] < 7:
        raise ValueError(f"action must contain at least 7 base parameters; got shape {a.shape}")

    defaults = np.array([sep_r, ali_r, coh_r, pred_r, pred_s], dtype=np.float32)
    out = np.empty((12,), dtype=np.float32)

    n_copy = min(a.shape[0], 12)
    out[:n_copy] = a[:n_copy]

    if n_copy < 12:
        missing_start = max(0, n_copy - 7)
        out[n_copy:] = defaults[missing_start:]

    return out


def _init_agents2d(total_boids, starts, start_spread, initial_velocity=None, robins_number=1, seed=0.1):
    starts = _as_xy_array(starts, name="starts")
    if starts.ndim == 1:
        starts = starts.reshape(1, 2)

    S = starts.shape[0]
    initial_velocity = _validate_initial_velocity(initial_velocity, starts)
    robins_number = max(1, int(robins_number))

    boid_pos = np.empty((total_boids, 2), dtype=np.float32)
    boid_vel = np.zeros((total_boids, 2), dtype=np.float32)
    boid_acc = np.zeros((total_boids, 2), dtype=np.float32)
    boid_time = np.zeros((total_boids,), dtype=np.float32)
    start_id = np.empty((total_boids,), dtype=np.int32)
    robin_id = np.empty((total_boids,), dtype=np.int32)

    rng = np.random.default_rng(int(seed * 1e6) % (2**32 - 1))

    counts = np.full(S, total_boids // S, dtype=np.int32)
    counts[: total_boids % S] += 1

    idx = 0
    for s in range(S):
        n = int(counts[s])
        sx, sy = starts[s]

        offsets = rng.normal(0.0, 1.0, size=(n, 2))
        norms = np.linalg.norm(offsets, axis=1) + 1e-12
        offsets = offsets / norms[:, None]

        radii = rng.random(n) ** 0.5
        offsets = offsets * (radii[:, None] * float(start_spread))

        boid_pos[idx:idx+n, 0] = sx + offsets[:, 0]
        boid_pos[idx:idx+n, 1] = sy + offsets[:, 1]

        boid_vel[idx:idx+n, 0] = initial_velocity[s, 0]
        boid_vel[idx:idx+n, 1] = initial_velocity[s, 1]

        start_id[idx:idx+n] = s
        robin_id[idx:idx+n] = s % robins_number
        idx += n

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
    return boid_pos, boid_vel, boid_acc,boid_time, boid_noise_cum, boid_noise_vals, seed_arr, start_id, robin_id


def _goal_idx_from_start_goal_idx(start_goal_idx, start_id, boid_count, n_goals, rng):
    goal_idx = np.empty((boid_count,), dtype=np.int32)

    if start_goal_idx is None:
        goal_idx[:] = rng.integers(0, n_goals, size=boid_count, dtype=np.int32)
        return goal_idx

    arr = np.asarray(start_goal_idx)
    if arr.ndim == 0:
        goal_idx[:] = np.int32(int(arr))
        return goal_idx

    arr = arr.astype(np.int32).reshape(-1)
    if arr.shape[0] == boid_count:
        goal_idx[:] = arr
        return goal_idx

    max_start = int(np.max(start_id)) if start_id.shape[0] > 0 else -1
    if arr.shape[0] >= max_start + 1:
        for i in range(boid_count):
            goal_idx[i] = arr[start_id[i]]
        return goal_idx

    raise ValueError(
        "start_goal_idx must be None, a scalar, an array with one entry per start, "
        "or an array with one entry per boid"
    )


@dataclass
class EpisodeMetrics:
    frac_goal: float
    avg_time_to_goal: float
    diversity_entropy: float


class FishGoalEnv2D(gym.Env):
    """
    2D environment with the 3D environment's recent functionality:
    - multiple starts
    - one initial velocity per start
    - start-based robin groups
    - predator avoidance between different robin groups
    - 12-parameter action with padding for shorter legacy actions
    - goal graph transitions
    - segment/circle-based DF/AF obstacle avoidance
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
        w_coh: float = 0.1,
        w_dis: float = 0.1,
        w_ali: float = 0.1,
        # fixed sim params
        rule_scalar: float = 1.0,
        max_speed: float = 0.18,
        sep_r: float = 1.6,
        ali_r: float = 4.0,
        coh_r: float = 5.5,
        pred_r: float = 4.0,
        pred_s: float = 1.0,
        rand_wavelen_scalar: float = 0.01,
        # starts and goals
        start: np.ndarray = np.array([6.0, 20.0], dtype=np.float32),
        starts: Optional[np.ndarray] = None,
        initial_velocity: Optional[np.ndarray] = None,
        robins_number: int = 1,
        goals: Optional[np.ndarray] = None,
        goal_W: Optional[np.ndarray] = None,
        start_goal_idx: Optional[int] = 0,
        # segment/circle and DF/AF obstacle settings
        segs: Optional[np.ndarray] = None,
        circles: Optional[np.ndarray] = None,
        df_origin: np.ndarray = np.array([0.0, 0.0], dtype=np.float32),
        df_length: float = 40.0,
        df_R: int = 256,
        avoid_r: float = 2.5,
        avoid_power: float = 1.0,
        avoid_alpha: float = 4.0,
        df_kill_thresh: float = 0.0,
        # debug/visual
        doAnimation: bool = False,
        returnTrajectory: bool = False,
        saveAnimation: bool = False,
    ):
        super().__init__()

        if starts is None:
            self.starts = _as_xy_array(start, name="start").reshape(1, 2)
        else:
            self.starts = _as_xy_array(starts, name="starts")
            if self.starts.ndim == 1:
                self.starts = self.starts.reshape(1, 2)

        self.start = self.starts[0].copy()
        self.initial_velocity = _validate_initial_velocity(initial_velocity, self.starts)
        self.robins_number = max(1, int(robins_number))

        if goals is None:
            self.goals = np.asarray([[34.0, 20.0]], dtype=np.float32)
        else:
            g = _as_xy_array(goals, name="goals")
            if g.ndim == 1:
                g = g.reshape(1, 2)
            self.goals = g

        G = self.goals.shape[0]
        if goal_W is None:
            self.goal_W = np.ones((G, G), dtype=np.float32)
            np.fill_diagonal(self.goal_W, 0.0)
        else:
            self.goal_W = np.asarray(goal_W, dtype=np.float32)
            if self.goal_W.shape != (G, G):
                raise ValueError(f"goal_W must be {(G, G)}, got {self.goal_W.shape}")
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
        self.pred_r = float(pred_r)
        self.pred_s = float(pred_s)
        self.rand_wavelen_scalar = float(rand_wavelen_scalar)

        self.w_goal = float(w_goal)
        self.w_time = float(w_time)
        self.w_div = float(w_div)
        self.w_coh = float(w_coh)
        self.w_dis = float(w_dis)
        self.w_ali = float(w_ali)

        self.action_space = spaces.Box(low=0.0, high=10.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self._rng = np.random.default_rng(int(seed))
        self._episode_seed: Optional[int] = None
        self._last_obs: Optional[np.ndarray] = None

        self._alive = np.empty((self.boid_count,), dtype=np.bool_)
        self._reached = np.empty((self.boid_count,), dtype=np.bool_)
        self._t_reach = np.empty((self.boid_count,), dtype=np.float32)

        self.returnTrajectory = bool(returnTrajectory)
        if self.returnTrajectory:
            self.trajectory_boid_pos = np.empty((self.max_steps, self.boid_count, 2), dtype=np.float32)
            self.trajectory_boid_vel = np.empty((self.max_steps, self.boid_count, 2), dtype=np.float32)
            self.trajectory_boid_acc = np.empty((self.max_steps, self.boid_count, 2), dtype=np.float32)

        self.segs = None if segs is None else np.asarray(segs, dtype=np.float32).reshape(-1, 4)
        self.circles = None if circles is None else np.asarray(circles, dtype=np.float32).reshape(-1, 3)
        self.mesh_af2d = None
        self.mesh_df2d = None
        self.mesh_origin2d = np.asarray(df_origin, dtype=np.float32)
        self.mesh_spacing2d = np.float32(df_length / (df_R - 1))
        self.df_kill_thresh = float(df_kill_thresh)

        has_segs = self.segs is not None and self.segs.size > 0
        has_circles = self.circles is not None and self.circles.size > 0
        if has_segs or has_circles:
            af, df, origin, spacing = precompute_obstacle_avoidance_2d(
                segs=self.segs,
                circles=self.circles,
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

        self.doAnimation = bool(doAnimation)
        self.saveAnimation = bool(saveAnimation)
        self._plt = None
        self._fig = None
        self._ax = None
        self.boid_scatters = []

        if self.doAnimation:
            self._init_plot()

        self._warmup()

    def _init_plot(self):
        self._plt = plt
        self._fig, self._ax = plt.subplots(figsize=(8, 6))
        self._ax.set_xlim(0, self.bound)
        self._ax.set_ylim(0, self.bound)
        self._ax.set_aspect("equal", adjustable="box")

        terminal_goals, _ = get_terminal_goals(self.goals, self.goal_W)
        if terminal_goals.shape[0] > 0:
            self._ax.scatter(terminal_goals[:, 0], terminal_goals[:, 1], s=140, marker="*", label="terminal goals")
        else:
            self._ax.scatter(self.goals[:, 0], self.goals[:, 1], s=100, marker="*", label="goals")

        self._ax.scatter(self.starts[:, 0], self.starts[:, 1], s=80, marker="o", label="starts")

        if self.segs is not None and self.segs.size > 0:
            for m in range(self.segs.shape[0]):
                x1, y1, x2, y2 = self.segs[m]
                self._ax.plot([x1, x2], [y1, y2], "k")

        if self.circles is not None and self.circles.size > 0:
            for m in range(self.circles.shape[0]):
                cx, cy, cr = self.circles[m]
                patch = plt.Circle((float(cx), float(cy)), float(cr), fill=False, color="k", linewidth=1.5)
                self._ax.add_patch(patch)

        self.boid_scatters = []
        for r in range(self.robins_number):
            sc = self._ax.scatter([], [], s=10, label=f"Robin subgroup {r}")
            self.boid_scatters.append(sc)

        plt.ion()
        plt.show()

    def _warmup(self):
        n = min(8, self.boid_count)
        warmup_alive = np.ones((n,), dtype=np.bool_)
        (boid_pos, boid_vel, boid_acc, boid_time, boid_noise_cum, boid_noise_vals,
         seed_arr, start_id, robin_id) = _init_agents2d(
            n,
            self.starts,
            1.0,
            initial_velocity=self.initial_velocity,
            robins_number=self.robins_number,
            seed=0.123,
        )
        goal_idx = np.zeros((n,), dtype=np.int32)

        step_sim2d_df(
            boid_pos,
            boid_vel,
            boid_acc,
            boid_time,
            boid_noise_cum,
            boid_noise_vals,
            seed_arr,
            np.float32(self.dt),
            bound_size=np.float32(self.bound),
            boid_count=n,
            rule_scalar=np.float32(self.rule_scalar),
            max_speed=np.float32(self.max_speed),
            sep_r=np.float32(self.sep_r),
            ali_r=np.float32(self.ali_r),
            coh_r=np.float32(self.coh_r),
            pred_r=np.float32(self.pred_r),
            sep_s=np.float32(1.0),
            ali_s=np.float32(1.0),
            coh_s=np.float32(1.0),
            pred_s=np.float32(self.pred_s),
            pred_goal_fade_r=np.float32(self.goal_radius * 20.0),
            bnd_s=np.float32(1.0),
            rand_s=np.float32(0.1),
            obs_avoid_s=np.float32(1.0),
            rand_wavelen_scalar=np.float32(self.rand_wavelen_scalar),
            goal_gain=np.float32(0.0),
            goals=self.goals,
            goal_idx=goal_idx,
            start_id=start_id,
            robin_id=robin_id,
            alive=warmup_alive,
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
        self._last_obs = obs
        return obs, {}

    def step(self, action):
        if self._episode_seed is None:
            raise RuntimeError("Call reset() before step().")

        metrics, info = self._rollout_episode(np.asarray(action, dtype=np.float32).reshape(-1))
        cohesion = compute_cohesion2d(self._boid_pos, self._alive)
        dispersion = compute_dispersion2d(self._boid_pos, self._alive)
        alignment = compute_alignment2d(self._boid_vel, self._alive)

        time_pen = 0.0
        if not math.isnan(metrics.avg_time_to_goal):
            time_pen = metrics.avg_time_to_goal / (self.max_steps * self.dt + 1e-12)

        reward = (
            self.w_goal * metrics.frac_goal
            - self.w_time * time_pen
            + self.w_div * metrics.diversity_entropy
            + self.w_coh * cohesion
            + self.w_dis * dispersion
            + self.w_ali * alignment
        )

        info.update({
            "frac_goal": metrics.frac_goal,
            "avg_time_to_goal": metrics.avg_time_to_goal,
            "diversity_entropy": metrics.diversity_entropy,
            "cohesion": float(cohesion),
            "dispersion": float(dispersion),
            "alignment": float(alignment),
            "reward": float(reward),
        })

        obs = np.zeros((6,), dtype=np.float32)
        self._last_obs = obs
        self._episode_seed = None
        return obs, float(reward), True, False, info

    def _rollout_episode(self, action: np.ndarray) -> Tuple[EpisodeMetrics, Dict]:
        action = _normalize_action(
            action,
            sep_r=self.sep_r,
            ali_r=self.ali_r,
            coh_r=self.coh_r,
            pred_r=self.pred_r,
            pred_s=self.pred_s,
        )

        seed = int(self._episode_seed)
        rng = np.random.default_rng(seed)

        (boid_pos, boid_vel, boid_acc, boid_time, boid_noise_cum, boid_noise_vals,
         seed_arr, start_id, robin_id) = _init_agents2d(
            total_boids=self.boid_count,
            starts=self.starts,
            start_spread=self.start_spread,
            initial_velocity=self.initial_velocity,
            robins_number=self.robins_number,
            seed=float((seed % 1000000) / 1000000.0 + 0.123),
        )

        goal_idx = _goal_idx_from_start_goal_idx(
            self.start_goal_idx, start_id, self.boid_count, self.goals.shape[0], rng
        )

        ever_hit = np.zeros((self.boid_count,), dtype=np.bool_)
        first_hit_t = np.full((self.boid_count,), np.nan, dtype=np.float32)

        sep_s = float(action[0])
        ali_s = float(action[1])
        coh_s = float(action[2])
        bnd_s = float(action[3])
        rand_s = float(action[4])
        obs_avoid_s = float(action[5])
        goal_gain = float(action[6])
        sep_r = float(action[7])
        ali_r = float(action[8])
        coh_r = float(action[9])
        pred_r = float(action[10])
        pred_s = float(action[11])

        self._alive.fill(True)
        self._reached.fill(False)
        self._t_reach.fill(np.nan)

        n_active = self.boid_count
        step = -1
        for step in range(self.max_steps):
            step_sim2d_df(
                boid_pos,
                boid_vel,
                boid_acc,
                boid_time,
                boid_noise_cum,
                boid_noise_vals,
                seed_arr,
                np.float32(self.dt),
                bound_size=np.float32(self.bound),
                boid_count=self.boid_count,
                rule_scalar=np.float32(self.rule_scalar),
                max_speed=np.float32(self.max_speed),
                sep_r=np.float32(sep_r),
                ali_r=np.float32(ali_r),
                coh_r=np.float32(coh_r),
                pred_r=np.float32(pred_r),
                sep_s=np.float32(sep_s),
                ali_s=np.float32(ali_s),
                coh_s=np.float32(coh_s),
                pred_s=np.float32(pred_s),
                pred_goal_fade_r=np.float32(self.goal_radius * 20.0),
                bnd_s=np.float32(bnd_s),
                rand_s=np.float32(rand_s),
                obs_avoid_s=np.float32(obs_avoid_s),
                rand_wavelen_scalar=np.float32(self.rand_wavelen_scalar),
                goal_gain=np.float32(goal_gain),
                goals=self.goals,
                goal_idx=goal_idx,
                start_id=start_id,
                robin_id=robin_id,
                alive=self._alive,
                mesh_af2d=self.mesh_af2d,
                mesh_origin2d=self.mesh_origin2d,
                mesh_spacing2d=self.mesh_spacing2d,
                mesh_df2d=self.mesh_df2d,
                df_kill_thresh=np.float32(self.df_kill_thresh),
            )

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
                self.trajectory_boid_acc[step, :, :] = boid_acc

            if self.doAnimation and self._plt is not None and self._ax is not None:
                self._update_plot(boid_pos, robin_id, step)

            if n_active == 0:
                break

        frac_goal = float(np.sum(ever_hit)) / float(self.boid_count)
        avg_time_to_goal = float(mean_time_to_goal2d(first_hit_t, ever_hit))
        diversity = float(heading_entropy2d(boid_vel))

        metrics = EpisodeMetrics(frac_goal=frac_goal, avg_time_to_goal=avg_time_to_goal, diversity_entropy=diversity)

        info = {
            "goals": self.goals,
            "starts": self.starts,
            "initial_velocity": self.initial_velocity,
            "robins_number": self.robins_number,
            "sep_r": sep_r,
            "ali_r": ali_r,
            "coh_r": coh_r,
            "pred_r": pred_r,
            "pred_s": pred_s,
            "start_id": start_id,
            "robin_id": robin_id,
            "reached_count": int(np.sum(ever_hit)),
            "active_count": int(n_active),
            "steps_executed": int(step + 1) if self.max_steps > 0 else 0,
        }

        self._boid_pos = boid_pos
        self._boid_vel = boid_vel
        self.goal_idx = goal_idx
        self.ever_hit = ever_hit
        self.first_hit_t = first_hit_t
        self.start_id = start_id
        self.robin_id = robin_id

        if self.returnTrajectory:
            info["trajectory_boid_pos"] = self.trajectory_boid_pos
            info["trajectory_boid_vel"] = self.trajectory_boid_vel
            info["trajectory_boid_acc"] = self.trajectory_boid_acc
        else:
            info["trajectory_boid_pos"] = None
            info["trajectory_boid_vel"] = None
            info["trajectory_boid_acc"] = None

        return metrics, info

    def _update_plot(self, boid_pos, robin_id, step):
        for r, scatter in enumerate(self.boid_scatters):
            mask = robin_id == r
            scatter.set_offsets(boid_pos[mask])
        self._plt.pause(0.001)
        if self.saveAnimation:
            self._fig.savefig(f"frames/frame_{step:04d}.png")

    def init_rollout(self, action: np.ndarray, goal_idx_init: Optional[np.ndarray] = None) -> None:
        if self._episode_seed is None:
            self._episode_seed = int(self._rng.integers(0, 2**31 - 1))

        action = _normalize_action(
            action,
            sep_r=self.sep_r,
            ali_r=self.ali_r,
            coh_r=self.coh_r,
            pred_r=self.pred_r,
            pred_s=self.pred_s,
        )

        self._episode_step = 0
        self._episode_done = False
        self._current_action = np.array(action, dtype=np.float32, copy=True)

        seed = int(self._episode_seed)
        self._rng = np.random.default_rng(seed)

        (
            self.boid_pos,
            self.boid_vel,
            self.boid_time,
            self.boid_noise_cum,
            self.boid_noise_vals,
            self.seed_arr,
            self.start_id,
            self.robin_id,
        ) = _init_agents2d(
            total_boids=self.boid_count,
            starts=self.starts,
            start_spread=self.start_spread,
            initial_velocity=self.initial_velocity,
            robins_number=self.robins_number,
            seed=float((seed % 1000000) / 1000000.0 + 0.123),
        )

        if goal_idx_init is not None:
            self.goal_idx = np.asarray(goal_idx_init, dtype=np.int32)
            if self.goal_idx.shape != (self.boid_count,):
                raise ValueError(f"goal_idx_init must have shape ({self.boid_count},), got {self.goal_idx.shape}")
        else:
            self.goal_idx = _goal_idx_from_start_goal_idx(
                self.start_goal_idx, self.start_id, self.boid_count, self.goals.shape[0], self._rng
            )

        self.ever_hit = np.zeros((self.boid_count,), dtype=np.bool_)
        self.first_hit_t = np.full((self.boid_count,), np.nan, dtype=np.float32)

        self._alive.fill(True)
        self._reached.fill(False)
        self._t_reach.fill(np.nan)

        if self.returnTrajectory:
            self.trajectory_boid_pos.fill(np.nan)
            self.trajectory_boid_vel.fill(np.nan)
            self.trajectory_boid_acc.fill(np.nan)
        self._update_action_cache()

    def _update_action_cache(self) -> None:
        a = self._current_action
        self.sep_s = float(a[0])
        self.ali_s = float(a[1])
        self.coh_s = float(a[2])
        self.bnd_s = float(a[3])
        self.rand_s = float(a[4])
        self.obs_avoid_s = float(a[5])
        self.goal_gain = float(a[6])
        self.sep_r = float(a[7])
        self.ali_r = float(a[8])
        self.coh_r = float(a[9])
        self.pred_r = float(a[10])
        self.pred_s = float(a[11])
        self.extra_action = a[7:].copy()

    def update_action(self, action: np.ndarray) -> None:
        action = _normalize_action(
            action,
            sep_r=self.sep_r,
            ali_r=self.ali_r,
            coh_r=self.coh_r,
            pred_r=self.pred_r,
            pred_s=self.pred_s,
        )
        self._current_action = np.array(action, dtype=np.float32, copy=True)
        self._update_action_cache()

    def update_goal(
        self,
        goals: Optional[np.ndarray] = None,
        goal_idx: Optional[np.ndarray] = None,
        goal_gain: Optional[float] = None,
    ) -> None:
        if goals is not None:
            self.goals = _as_xy_array(goals, name="goals")
            if self.goals.ndim == 1:
                self.goals = self.goals.reshape(1, 2)

        if goal_idx is not None:
            goal_idx = np.asarray(goal_idx, dtype=np.int32)
            if goal_idx.shape != (self.boid_count,):
                raise ValueError(f"goal_idx must have shape ({self.boid_count},), got {goal_idx.shape}")
            self.goal_idx[:] = goal_idx

        if goal_gain is not None:
            self.goal_gain = float(goal_gain)

    def step_rollout(self) -> Tuple[bool, Dict]:
        if getattr(self, "_episode_done", True):
            return True, {
                "n_active": int(np.sum(self._alive)),
                "step": getattr(self, "_episode_step", 0),
                "reason": "episode_not_initialized_or_already_done",
            }

        step = self._episode_step

        step_sim2d_df(
            self.boid_pos,
            self.boid_vel,
            self.boid_time,
            self.boid_noise_cum,
            self.boid_noise_vals,
            self.seed_arr,
            np.float32(self.dt),
            bound_size=np.float32(self.bound),
            boid_count=self.boid_count,
            rule_scalar=np.float32(self.rule_scalar),
            max_speed=np.float32(self.max_speed),
            sep_r=np.float32(self.sep_r),
            ali_r=np.float32(self.ali_r),
            coh_r=np.float32(self.coh_r),
            pred_r=np.float32(self.pred_r),
            sep_s=np.float32(self.sep_s),
            ali_s=np.float32(self.ali_s),
            coh_s=np.float32(self.coh_s),
            pred_s=np.float32(self.pred_s),
            pred_goal_fade_r=np.float32(self.goal_radius * 20.0),
            bnd_s=np.float32(self.bnd_s),
            rand_s=np.float32(self.rand_s),
            obs_avoid_s=np.float32(self.obs_avoid_s),
            rand_wavelen_scalar=np.float32(self.rand_wavelen_scalar),
            goal_gain=np.float32(self.goal_gain),
            goals=self.goals,
            goal_idx=self.goal_idx,
            start_id=self.start_id,
            robin_id=self.robin_id,
            alive=self._alive,
            mesh_af2d=self.mesh_af2d,
            mesh_origin2d=self.mesh_origin2d,
            mesh_spacing2d=self.mesh_spacing2d,
            mesh_df2d=self.mesh_df2d,
            df_kill_thresh=np.float32(self.df_kill_thresh),
        )

        n_active = update_goal_events_2d(
            self.boid_pos,
            self._alive,
            self.ever_hit,
            self.first_hit_t,
            self.goal_idx,
            self.goals,
            self.goal_W,
            np.float32(self.goal_radius),
            step,
            np.float32(self.dt),
            self.seed_arr,
        )

        if self.returnTrajectory and step < self.max_steps:
            self.trajectory_boid_pos[step, :, :] = self.boid_pos
            self.trajectory_boid_vel[step, :, :] = self.boid_vel
            self.trajectory_boid_acc[step, :, :] = self.boid_acc
        if self.doAnimation and self._plt is not None and self._ax is not None:
            self._update_plot(self.boid_pos, self.robin_id, step)

        self._episode_step += 1
        self._episode_done = (n_active == 0) or (self._episode_step >= self.max_steps)

        return self._episode_done, {
            "n_active": int(n_active),
            "step": int(self._episode_step),
        }

    def finalize_rollout(self) -> Tuple[EpisodeMetrics, Dict]:
        frac_goal = float(np.sum(self.ever_hit)) / float(self.boid_count)
        avg_time_to_goal = float(mean_time_to_goal2d(self.first_hit_t, self.ever_hit))
        diversity = float(heading_entropy2d(self.boid_vel))

        metrics = EpisodeMetrics(
            frac_goal=frac_goal,
            avg_time_to_goal=avg_time_to_goal,
            diversity_entropy=diversity,
        )

        info = {
            "goals": self.goals,
            "starts": self.starts,
            "initial_velocity": self.initial_velocity,
            "robins_number": self.robins_number,
            "start_id": self.start_id,
            "robin_id": self.robin_id,
            "reached_count": int(np.sum(self.ever_hit)),
            "steps_executed": int(self._episode_step),
        }

        if self.returnTrajectory:
            info["trajectory_boid_pos"] = self.trajectory_boid_pos
            info["trajectory_boid_vel"] = self.trajectory_boid_vel
            info["trajectory_boid_acc"] = self.trajectory_boid_acc
        else:
            info["trajectory_boid_pos"] = None
            info["trajectory_boid_vel"] = None
            info["trajectory_boid_acc"] = None

        return metrics, info


# -----------------------------
# Example with two starts and opposite goals
# -----------------------------
if __name__ == "__main__":
    starts = 4*np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    goals  = 4*np.array([9.0, 9.0])
    bounds = (np.array([0.0, 0.0]), 4*np.array([10.0, 10.0]))

    obstacles = [
        PolyObstacle(round_obstacle(4*np.array([[3.0, 1.5], [5.2, 2.2], [4.7, 4.0], [2.8, 3.4]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(4*np.array([[6.2, 6.0], [8.5, 6.3], [8.1, 8.4], [6.8, 8.9], [5.9, 7.4]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(4*np.array([[2.0, 6.8], [3.3, 6.1], [4.2, 6.9], [3.2, 7.3], [3.7, 8.1], [2.6, 8.3]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(4*np.array([[1.8, 4.2], [2.7, 4.0], [3.0, 4.8], [2.3, 5.3], [1.7, 4.9]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(4*np.array([[4.6, 5.1], [5.4, 5.0], [5.8, 5.7], [5.0, 6.2], [4.4, 5.7]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(4*np.array([[7.9, 3.0], [9.0, 3.2], [8.8, 4.2], [7.7, 4.0]]), n_iters=4, n_points=32)),
        PolyObstacle(round_obstacle(4*np.array([[5.7, 1.0], [6.6, 1.2], [6.4, 2.3], [5.6, 2.1]]), n_iters=4, n_points=32)),
    ]

    # Additional wall/segment obstacles can still be provided here as (x1, y1, x2, y2).
    segs = obstacles_to_segs(obstacles, scale=1.0)

    action = np.array([
        1.0,   # sep weight
        1.0,   # ali weight
        1.0,   # coh weight
        1.0,   # boundary
        1.0,   # random
        1.0,   # obstacle
        0.3,   # goal
        1.5,   # sep radius
        4.0,   # ali radius
        5.5,   # coh radius
        10.0,  # predator detection radius
        3.0,   # predator avoidance scalar
    ], dtype=np.float32)

    load_theta = True
    if load_theta:
        theta_path = "save/goal.pkl"
        action = pickle.load(open(theta_path, "rb"))["best_theta"]
        # Add predator avoidance scalar
        action = np.append(action, 5.0)
        action = np.append(action, 3.0) 

    env = FishGoalEnv2D(
        boid_count=1200,
        bound=40.0,
        max_steps=1000,
        dt=1.0,
        starts=starts,
        goals=goals,
        robins_number=4,
        segs=segs,
        circles=None,
        df_origin=np.array([0.0, 0.0], dtype=np.float32),
        df_length=40.0,
        df_R=256,
        avoid_r=2.5,
        doAnimation=True,
        returnTrajectory=True,
        start_spread=1.0,
    )

    t0 = time.time()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
    t1 = time.time()
    print("Reward:", reward)
    print("Info:", {k: v for k, v in info.items() if k not in ("start_id", "robin_id")})
    print("Runtime:", t1 - t0)