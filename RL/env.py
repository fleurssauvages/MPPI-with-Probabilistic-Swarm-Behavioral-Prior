import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import time

import numpy as np
from numba import njit, prange

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pickle

# -----------------------------
# Helpers
# -----------------------------
@njit(cache=True, fastmath=True)
def _dot(ax, ay, az, bx, by, bz):
    return ax*bx + ay*by + az*bz

@njit(cache=True, fastmath=True)
def _closest_point_on_triangle(px, py, pz,
                              ax, ay, az,
                              bx, by, bz,
                              cx, cy, cz):
    # Ericson-style closest point (branchy but robust)
    abx, aby, abz = bx-ax, by-ay, bz-az
    acx, acy, acz = cx-ax, cy-ay, cz-az
    apx, apy, apz = px-ax, py-ay, pz-az

    d1 = _dot(abx,aby,abz, apx,apy,apz)
    d2 = _dot(acx,acy,acz, apx,apy,apz)
    if d1 <= 0.0 and d2 <= 0.0:
        return ax, ay, az

    bpx, bpy, bpz = px-bx, py-by, pz-bz
    d3 = _dot(abx,aby,abz, bpx,bpy,bpz)
    d4 = _dot(acx,acy,acz, bpx,bpy,bpz)
    if d3 >= 0.0 and d4 <= d3:
        return bx, by, bz

    vc = d1*d4 - d3*d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return ax + v*abx, ay + v*aby, az + v*abz

    cpx, cpy, cpz = px-cx, py-cy, pz-cz
    d5 = _dot(abx,aby,abz, cpx,cpy,cpz)
    d6 = _dot(acx,acy,acz, cpx,cpy,cpz)
    if d6 >= 0.0 and d5 <= d6:
        return cx, cy, cz

    vb = d5*d2 - d1*d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return ax + w*acx, ay + w*acy, az + w*acz

    va = d3*d6 - d5*d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        # edge BC
        bcx, bcy, bcz = cx-bx, cy-by, cz-bz
        return bx + w*bcx, by + w*bcy, bz + w*bcz

    # inside face region
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return ax + abx*v + acx*w, ay + aby*v + acy*w, az + abz*v + acz*w

@njit(cache=True, fastmath=True)
def _point_triangle_dist2(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    qx, qy, qz = _closest_point_on_triangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz)
    dx, dy, dz = px-qx, py-qy, pz-qz
    return dx*dx + dy*dy + dz*dz

@njit(cache=True, fastmath=True)
def _norm3(v0, v1, v2):
    return math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)

@njit(cache=True, fastmath=True, parallel=True)
def build_distance_field(verts, faces, origin, spacing, R, avoid_r):
    df = np.empty((R, R, R), dtype=np.float32)
    avoid2 = avoid_r * avoid_r

    # flatten loop for parallelism
    for idx in prange(R*R*R):
        i = idx // (R*R)
        j = (idx // R) % R
        k = idx % R

        px = origin[0] + spacing * i
        py = origin[1] + spacing * j
        pz = origin[2] + spacing * k

        best2 = avoid2  # clamp early: we don't care beyond avoid_r
        for f in range(faces.shape[0]):
            i0 = faces[f, 0]
            i1 = faces[f, 1]
            i2 = faces[f, 2]

            ax, ay, az = verts[i0,0], verts[i0,1], verts[i0,2]
            bx, by, bz = verts[i1,0], verts[i1,1], verts[i1,2]
            cx, cy, cz = verts[i2,0], verts[i2,1], verts[i2,2]

            d2 = _point_triangle_dist2(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz)
            if d2 < best2:
                best2 = d2
                if best2 <= 1e-12:
                    break

        d = math.sqrt(best2)
        if d >= avoid_r:
            df[i,j,k] = 1.0
        else:
            df[i,j,k] = d / avoid_r  # in [0,1)

    return df

@njit(cache=True, fastmath=True)
def build_avoid_field_from_df(df, power=1.0, alpha=4.0):
    R = df.shape[0]
    af = np.zeros((R, R, R, 3), dtype=np.float32)

    # 26-neighborhood stencil
    # sum( offset * neighbor_value )
    for i in range(1, R-1):
        for j in range(1, R-1):
            for k in range(1, R-1):
                s0 = 0.0
                s1 = 0.0
                s2 = 0.0

                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            w = df[i+di, j+dj, k+dk]
                            s0 += di * w
                            s1 += dj * w
                            s2 += dk * w

                # normalize direction
                n2 = s0*s0 + s1*s1 + s2*s2
                if n2 <= 1e-18:
                    continue
                invn = 1.0 / math.sqrt(n2)
                nx, ny, nz = s0*invn, s1*invn, s2*invn

                # magnitude shaping from normalized distance
                # df in [0,1], proximity x = 1 - df
                x = 1.0 - df[i,j,k]
                if x <= 0.0:
                    continue

                # optional power shaping
                if power != 1.0:
                    x = x ** power

                mag = math.exp(alpha * x) - 1.0

                af[i,j,k,0] = nx * mag
                af[i,j,k,1] = ny * mag
                af[i,j,k,2] = nz * mag

    return af

@njit(cache=True, fastmath=True)
def sample_avoid_field(af, px, py, pz, origin, spacing):
    R = af.shape[0]

    fx = (px - origin[0]) / spacing
    fy = (py - origin[1]) / spacing
    fz = (pz - origin[2]) / spacing

    ix = int(math.floor(fx)); tx = fx - ix
    iy = int(math.floor(fy)); ty = fy - iy
    iz = int(math.floor(fz)); tz = fz - iz

    # clamp to [0, R-2]
    if ix < 0: ix = 0; tx = 0.0
    if iy < 0: iy = 0; ty = 0.0
    if iz < 0: iz = 0; tz = 0.0
    if ix > R-2: ix = R-2; tx = 1.0
    if iy > R-2: iy = R-2; ty = 1.0
    if iz > R-2: iz = R-2; tz = 1.0

    # eight corners
    v000 = af[ix,   iy,   iz  ]
    v100 = af[ix+1, iy,   iz  ]
    v010 = af[ix,   iy+1, iz  ]
    v110 = af[ix+1, iy+1, iz  ]
    v001 = af[ix,   iy,   iz+1]
    v101 = af[ix+1, iy,   iz+1]
    v011 = af[ix,   iy+1, iz+1]
    v111 = af[ix+1, iy+1, iz+1]

    # interpolate
    outx = 0.0; outy = 0.0; outz = 0.0
    for c in range(3):
        a00 = v000[c]*(1-tx) + v100[c]*tx
        a10 = v010[c]*(1-tx) + v110[c]*tx
        a01 = v001[c]*(1-tx) + v101[c]*tx
        a11 = v011[c]*(1-tx) + v111[c]*tx
        b0 = a00*(1-ty) + a10*ty
        b1 = a01*(1-ty) + a11*ty
        val = b0*(1-tz) + b1*tz
        if c == 0: outx = val
        elif c == 1: outy = val
        else: outz = val

    return outx, outy, outz

@njit(cache=True, fastmath=True)
def _clamp_len(v0, v1, v2, max_len):
    n = _norm3(v0, v1, v2)
    if n <= 1e-12:
        return 0.0, 0.0, 0.0
    if n > max_len:
        s = max_len / n
        return v0 * s, v1 * s, v2 * s
    return v0, v1, v2


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
    # Paul Breeuwsma coefficients
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
        # Wavelen segment
        cum_wavlen = cum_wavlen + wavelen
        rv0, rv1, rv2 = rv1, rv2, rv3
        rv3 = _lcg_rand01(seed_arr)

    frac = (time % wavelen) / wavelen
    value = _cubic_interpolate(rv0, rv1, rv2, rv3, frac)
    return (value * 2.0 - 1.0), cum_wavlen, rv0, rv1, rv2, rv3


# -----------------------------
# Core rules
# -----------------------------
@njit(cache=True, fastmath=True)
def _bounds_steer(px, py, pz, bound_size):
    min_b = 0.0
    max_b = bound_size
    sx = 0.0
    sy = 0.0
    sz = 0.0

    if px < min_b:
        sx = min_b - px
    elif px > max_b:
        sx = max_b - px

    if py < min_b:
        sy = (min_b - py) * 2.0
    elif py > max_b:
        sy = max_b - py

    if pz < min_b:
        sz = min_b - pz
    elif pz > max_b:
        sz = max_b - pz

    sy = sy * 2.0 
    return sx, sy, sz

@njit(cache=True, fastmath=True)
def _reynolds(i, pos, vel, count, sep_r, ali_r, coh_r):
    px, py, pz = pos[i, 0], pos[i, 1], pos[i, 2]

    sep0 = sep1 = sep2 = 0.0
    ali0 = ali1 = ali2 = 0.0
    coh0 = coh1 = coh2 = 0.0
    max_d2 = max(sep_r * sep_r, ali_r * ali_r, coh_r * coh_r)

    for j in range(count):
        if j == i:
            continue
        dx = px - pos[j, 0]
        dy = py - pos[j, 1]
        dz = pz - pos[j, 2]
        d2 = dx * dx + dy * dy + dz * dz
        if d2 <= 1e-24 or d2 > max_d2:
            continue

        d = math.sqrt(d2)
        if d2 < sep_r * sep_r:
            mag = 1.0 - d / sep_r
            sep0 += (dx / d) * mag
            sep1 += (dy / d) * mag
            sep2 += (dz / d) * mag

        if d2 < ali_r * ali_r:
            mag = 1.0 - d / ali_r
            vx, vy, vz = vel[j, 0], vel[j, 1], vel[j, 2]
            vn = _norm3(vx, vy, vz)
            if vn > 1e-12:
                ali0 += (vx / vn) * mag
                ali1 += (vy / vn) * mag
                ali2 += (vz / vn) * mag

        if d2 < coh_r * coh_r:
            mag = 1.0 - d / coh_r
            coh0 += (-dx / d) * mag
            coh1 += (-dy / d) * mag
            coh2 += (-dz / d) * mag

    sep0, sep1, sep2 = _clamp_len(sep0, sep1, sep2, 1.0)
    ali0, ali1, ali2 = _clamp_len(ali0, ali1, ali2, 1.0)
    coh0, coh1, coh2 = _clamp_len(coh0, coh1, coh2, 1.0)
    return sep0, sep1, sep2, ali0, ali1, ali2, coh0, coh1, coh2

@njit(cache=True, fastmath=True)
def _obstacle_avoid(px, py, pz, cx, cy, cz, sphere_r, avoid_r):
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    d2 = dx*dx + dy*dy + dz*dz

    influence = sphere_r + avoid_r
    if influence <= 0.0:
        return 0.0, 0.0, 0.0
    infl2 = influence * influence

    # reject if outside influence or almost at center
    if d2 >= infl2 or d2 <= 1e-18:
        return 0.0, 0.0, 0.0

    d = math.sqrt(d2)

    # normalized proximity in [0,1]
    x = 1.0 - d / influence

    # exponential shaping (steeper near obstacle)
    alpha = 4.0
    mag = (math.exp(alpha * x) - 1.0)

    invd = 1.0 / d
    return (dx * invd) * mag, (dy * invd) * mag, (dz * invd) * mag

@njit(cache=True, fastmath=True, parallel=True)
def step_sim(
    boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
    seed_arr, dt,
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
    mesh_af,
    mesh_origin,
    mesh_spacing,
    alive
):

    if (max_speed == 0.0) or boid_count <= 0:
        return

    # --- Boids ---
    for i in prange(boid_count):
        if not alive[i]:
            continue
        boid_time[i] += dt

        ax = ay = az = 0.0

        sep0, sep1, sep2, ali0, ali1, ali2, coh0, coh1, coh2 = _reynolds(
            i, boid_pos, boid_vel, boid_count, sep_r, ali_r, coh_r
        )

        ax += sep0 * sep_s
        ay += sep1 * sep_s
        az += sep2 * sep_s

        ax += ali0 * ali_s
        ay += ali1 * ali_s
        az += ali2 * ali_s

        ax += coh0 * coh_s
        ay += coh1 * coh_s
        az += coh2 * coh_s

        # bounds
        sx, sy, sz = _bounds_steer(boid_pos[i, 0], boid_pos[i, 1], boid_pos[i, 2], bound_size)
        ax += sx * bnd_s
        ay += sy * bnd_s
        az += sz * bnd_s

        # random motion (smooth noise)
        if rand_s != 0.0:
            t = boid_time[i] * rand_wavelen_scalar * math.sqrt(dt)

            rv0, rv1, rv2, rv3 = boid_noise_vals[i, 0, 0], boid_noise_vals[i, 0, 1], boid_noise_vals[i, 0, 2], boid_noise_vals[i, 0, 3]
            nx, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.0, boid_noise_cum[i, 0], rv0, rv1, rv2, rv3, seed_arr)
            boid_noise_cum[i, 0] = cwl
            boid_noise_vals[i, 0, 0], boid_noise_vals[i, 0, 1], boid_noise_vals[i, 0, 2], boid_noise_vals[i, 0, 3] = rv0, rv1, rv2, rv3

            rv0, rv1, rv2, rv3 = boid_noise_vals[i, 1, 0], boid_noise_vals[i, 1, 1], boid_noise_vals[i, 1, 2], boid_noise_vals[i, 1, 3]
            ny, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.1, boid_noise_cum[i, 1], rv0, rv1, rv2, rv3, seed_arr)
            boid_noise_cum[i, 1] = cwl
            boid_noise_vals[i, 1, 0], boid_noise_vals[i, 1, 1], boid_noise_vals[i, 1, 2], boid_noise_vals[i, 1, 3] = rv0, rv1, rv2, rv3

            rv0, rv1, rv2, rv3 = boid_noise_vals[i, 2, 0], boid_noise_vals[i, 2, 1], boid_noise_vals[i, 2, 2], boid_noise_vals[i, 2, 3]
            nz, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.2, boid_noise_cum[i, 2], rv0, rv1, rv2, rv3, seed_arr)
            boid_noise_cum[i, 2] = cwl
            boid_noise_vals[i, 2, 0], boid_noise_vals[i, 2, 1], boid_noise_vals[i, 2, 2], boid_noise_vals[i, 2, 3] = rv0, rv1, rv2, rv3

            ax += nx * rand_s
            ay += (ny * 0.2) * rand_s
            az += nz * rand_s

        # obstacle avoid
        if mesh_af is not None and obs_avoid_s != 0.0:
            ox, oy, oz = sample_avoid_field(mesh_af,
                                            boid_pos[i,0], boid_pos[i,1], boid_pos[i,2],
                                            mesh_origin, mesh_spacing)
            ax += obs_avoid_s * ox
            ay += obs_avoid_s * oy
            az += obs_avoid_s * oz

        # goal attraction
        if goal_gain != 0.0:
            gi = goal_idx[i]
            gx = goals[gi, 0]
            gy = goals[gi, 1]
            gz = goals[gi, 2]

            dx = gx - boid_pos[i, 0]
            dy = gy - boid_pos[i, 1]
            dz = gz - boid_pos[i, 2]
            d = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12

            pwr = 3.0
            mag = goal_gain * (d ** pwr)
            mag = min(mag, goal_gain * 8.0)

            ax += (dx / d) * mag
            ay += (dy / d) * mag
            az += (dz / d) * mag

        # integrate
        ax *= rule_scalar
        ay *= rule_scalar
        az *= rule_scalar

        boid_vel[i, 0] += ax * dt
        boid_vel[i, 1] += ay * dt
        boid_vel[i, 2] += az * dt
        boid_vel[i, 0], boid_vel[i, 1], boid_vel[i, 2] = _clamp_len(
            boid_vel[i, 0], boid_vel[i, 1], boid_vel[i, 2], max_speed
        )

        boid_pos[i, 0] += boid_vel[i, 0] * dt
        boid_pos[i, 1] += boid_vel[i, 1] * dt
        boid_pos[i, 2] += boid_vel[i, 2] * dt

def _init_agents(total_boids, start_xyz, start_spread, seed=0.1):
    sx, sy, sz = float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2])

    boid_pos = np.empty((total_boids, 3), dtype=np.float32)
    boid_vel = np.zeros((total_boids, 3), dtype=np.float32)
    boid_time = np.zeros((total_boids,), dtype=np.float32)

    rng = np.random.default_rng(int(seed * 1e6) % (2**32 - 1))
    offsets = rng.normal(0.0, 1.0, size=(total_boids, 3))
    norms = np.linalg.norm(offsets, axis=1) + 1e-12
    offsets = offsets / norms[:, None]
    radii = rng.random(total_boids) ** (1.0/3.0)
    offsets = offsets * (radii[:, None] * float(start_spread))
    boid_pos[:, 0] = sx + offsets[:, 0]
    boid_pos[:, 1] = sy + offsets[:, 1]
    boid_pos[:, 2] = sz + offsets[:, 2]

    boid_noise_cum = np.zeros((total_boids, 3), dtype=np.float32)
    boid_noise_vals = np.empty((total_boids, 3, 4), dtype=np.float32)

    seed_arr = np.array([math.floor(seed * 4294967296.0)], dtype=np.float32)

    for i in prange(total_boids):
        for a in range(3):
            boid_noise_vals[i, a, 0] = _lcg_rand01(seed_arr)
            boid_noise_vals[i, a, 1] = _lcg_rand01(seed_arr)
            boid_noise_vals[i, a, 2] = _lcg_rand01(seed_arr)
            boid_noise_vals[i, a, 3] = _lcg_rand01(seed_arr)

    seed_arr[0] = math.floor(seed * 4294967296.0)

    return (boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals, seed_arr)

@njit(cache=True, fastmath=True)
def update_events_numba(
    boid_pos, boid_vel,
    alive,
    ever_hit, first_hit_t,
    goal_idx, goals, goal_W,
    goal_radius,
    step_idx, dt,
    seed_arr,
    mesh_df, mesh_origin, mesh_spacing,
):
    gr2 = goal_radius * goal_radius
    t_now = (step_idx + 1) * dt

    n_active_after = 0
    n_new_hits = 0

    for i in range(boid_pos.shape[0]):
        if not alive[i]:
            continue
        if mesh_df is not None:
            dnorm = sample_df(mesh_df,
                            boid_pos[i,0], boid_pos[i,1], boid_pos[i,2],
                            mesh_origin, mesh_spacing)
            if math.isfinite(dnorm) and dnorm <= 0.01:
                alive[i] = False
                boid_vel[i,0] = 0.0; boid_vel[i,1] = 0.0; boid_vel[i,2] = 0.0
                continue
        # --- goal check (fish-specific) ---
        gi = goal_idx[i]
        gx = goals[gi, 0]; gy = goals[gi, 1]; gz = goals[gi, 2]

        dx = boid_pos[i, 0] - gx
        dy = boid_pos[i, 1] - gy
        dz = boid_pos[i, 2] - gz
        d2g = dx*dx + dy*dy + dz*dz

        if d2g <= gr2:
            if not ever_hit[i]:
                ever_hit[i] = True
                first_hit_t[i] = t_now

            # sample next goal from weighted outgoing edges
            u = _lcg_rand01(seed_arr)
            nxt = _weighted_next_goal(goal_W[gi], u)
            goal_idx[i] = nxt
        else:
            if alive[i]:
                n_active_after += 1

    return n_active_after

@njit(cache=True, fastmath=True)
def mean_time_to_goal(t_reach, reached):
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

@njit(cache=True, fastmath=True)
def count_reached(reached):
    n_goal = 0
    for i in range(reached.shape[0]):
        if reached[i]:
            n_goal += 1
    return n_goal


@dataclass
class EpisodeMetrics:
    frac_goal: float
    avg_time_to_goal: float
    diversity_entropy: float

@njit(cache=True, fastmath=True)
def heading_entropy(vel, n_az=12, n_el=6):
    # count bins
    n_bins = n_az * n_el
    counts = np.zeros(n_bins, dtype=np.int32)

    two_pi = 2.0 * math.pi
    inv_two_pi = 1.0 / two_pi
    inv_pi = 1.0 / math.pi

    total = 0

    for i in range(vel.shape[0]):
        vx = vel[i, 0]
        vy = vel[i, 1]
        vz = vel[i, 2]
        sp2 = vx*vx + vy*vy + vz*vz
        if sp2 <= 1e-24:
            continue

        sp = math.sqrt(sp2)
        vx /= sp
        vy /= sp
        vz /= sp

        az = math.atan2(vy, vx)
        if vz > 1.0:
            vz = 1.0
        elif vz < -1.0:
            vz = -1.0
        el = math.asin(vz)

        az_bin = int(math.floor((az + math.pi) * inv_two_pi * n_az))
        el_bin = int(math.floor((el + (math.pi / 2.0)) * inv_pi * n_el))

        if az_bin < 0:
            az_bin = 0
        elif az_bin >= n_az:
            az_bin = n_az - 1

        if el_bin < 0:
            el_bin = 0
        elif el_bin >= n_el:
            el_bin = n_el - 1

        idx = el_bin * n_az + az_bin
        counts[idx] += 1
        total += 1

    if total == 0:
        return 0.0

    H = 0.0
    inv_total = 1.0 / total
    for b in range(n_bins):
        c = counts[b]
        if c > 0:
            p = c * inv_total
            H -= p * math.log(p)

    Hmax = math.log(n_bins)
    return H / (Hmax + 1e-12)

@njit(cache=True, fastmath=True)
def sample_df(df, px, py, pz, origin, spacing):
    R = df.shape[0]
    fx = (px - origin[0]) / spacing
    fy = (py - origin[1]) / spacing
    fz = (pz - origin[2]) / spacing

    ix = int(math.floor(fx))
    iy = int(math.floor(fy))
    iz = int(math.floor(fz))

    # clamp to grid
    if ix < 0: ix = 0
    if iy < 0: iy = 0
    if iz < 0: iz = 0
    if ix > R-1: ix = R-1
    if iy > R-1: iy = R-1
    if iz > R-1: iz = R-1

    return df[ix, iy, iz]

@njit(cache=True, fastmath=True)
def _weighted_next_goal(row_w, u01):
    """
    row_w: (G,) nonnegative weights
    u01: uniform in [0,1)
    returns: int index
    """
    s = 0.0
    for k in range(row_w.shape[0]):
        s += row_w[k]

    # if no outgoing weights, fall back to uniform
    if s <= 1e-12:
        return int(u01 * row_w.shape[0])

    thresh = u01 * s
    c = 0.0
    for k in range(row_w.shape[0]):
        c += row_w[k]
        if c >= thresh:
            return k
    return row_w.shape[0] - 1

class FishGoalEnv(gym.Env):
    """Parameter-optimization RL environment.

    One `step(action)` runs a full rollout using the action as behavior scalars.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        boid_count: int = 200,
        bound: float = 40.0,
        max_steps: int = 2000,
        dt: float = 0.01,
        start_spread: float = 3.0,
        eat_radius: float = 2.0,
        goal_radius: float = 2.0,
        avoid_radius: float = 2.0,
        seed: int = 0,
        # reward weights
        w_goal: float = 1.0,
        w_time: float = 0.2,
        w_div: float = 0.1,
        # fixed sim params (you can override)
        rule_scalar: float = 1.0,
        rule_scalar_p: float = 1.0,
        max_speed: float = 0.18,
        max_speed_p: float = 0.28,
        sep_r: float = 1.6,
        ali_r: float = 4.0,
        coh_r: float = 5.5,
        rand_wavelen_scalar: float = 0.01,
        attack_s: float = 1.0,
        doAnimation: bool = False,
        returnTrajectory: bool = False,
        start =  np.array([6.0, 20.0, 20.0], dtype=np.float32),
        goal =  np.array([34.0, 20.0, 20.0], dtype=np.float32),
        verts = None,
        faces = None,
        goals = None, 
        goal_W = None, 
        start_goal_idx = 0,
    ):
        super().__init__()

        self.start = start
        self.goal = goal

        if goals is None:
            self.goals = np.asarray([goal], dtype=np.float32)   # fallback: 1 goal
        else:
            self.goals = np.asarray(goals, dtype=np.float32)

        G = self.goals.shape[0]

        if goal_W is None:
            self.goal_W = np.ones((G, G), dtype=np.float32)     # fully connected uniform
            np.fill_diagonal(self.goal_W, 0.0)                  # no self-loop
            self.goal_terminal = (self.goal_W.sum(axis=1) < 1e-12)
        else:
            self.goal_W = np.asarray(goal_W, dtype=np.float32)

        self.start_goal_idx = start_goal_idx

        self.boid_count = int(boid_count)

        self.bound = float(bound)
        self.max_steps = int(max_steps)
        self.dt = float(dt)

        self.start_spread = float(start_spread)
        self.avoid_radius = float(avoid_radius)
        self.eat_radius = float(eat_radius)
        self.goal_radius = float(goal_radius)

        self.rule_scalar = float(rule_scalar)
        self.rule_scalar_p = float(rule_scalar_p)
        self.max_speed = float(max_speed)
        self.max_speed_p = float(max_speed_p)

        self.sep_r = float(sep_r)
        self.ali_r = float(ali_r)
        self.coh_r = float(coh_r)
        self.rand_wavelen_scalar = float(rand_wavelen_scalar)
        self.attack_s = float(attack_s)

        self._rng = np.random.default_rng(int(seed))

        self.w_goal = float(w_goal)
        self.w_time = float(w_time)
        self.w_div = float(w_div)

        # Action: 9 scalars (sep, ali, coh, bnd, rand, obs_avoid, goal_gain, obs_gain)
        self.action_space = spaces.Box(low=0.0, high=10.0, shape=(9,), dtype=np.float32)

        self._last_obs: Optional[np.ndarray] = None
        self._episode_seed: Optional[int] = None

        self._alive = np.empty((self.boid_count,), dtype=np.bool_)
        self._reached = np.empty((self.boid_count,), dtype=np.bool_)
        self._t_reach = np.empty((self.boid_count,), dtype=np.float32)

        if verts is not None and faces is not None:
            # build once (e.g., in env.reset() or after loading mesh)
            self.mesh_verts = verts
            self.mesh_faces = faces
            af, df, field_origin, field_spacing = precompute_mesh_avoidance(
                verts, faces,
                origin=np.array([-self.bound, -self.bound, -self.bound], np.float32),  # grid corner
                field_length=2.0*self.bound,   # cover whole cube [-bound, bound]^3
                R=128,
                avoid_r=self.avoid_radius,
                power=1.0,
                alpha=4.0
            )
            self.mesh_af = af                    # (R,R,R,3) float32
            self.mesh_df = df
            self.mesh_origin = field_origin      # (3,) float32
            self.mesh_spacing = np.float32(field_spacing)
        else:
            self.mesh_verts = None
            self.mesh_faces = None
            self.mesh_af = None
            self.mesh_df = None
            self.mesh_origin = None
            self.mesh_spacing = None

        self.doAnimation = doAnimation
        self.returnTrajectory = returnTrajectory
        if self.returnTrajectory:
            self.trajectory_boid_pos = np.empty((self.max_steps, self.boid_count, 3), dtype=np.float32)
            self.trajectory_boid_vel = np.empty((self.max_steps, self.boid_count, 3), dtype=np.float32)
        self._warmup()

    def _warmup(self):
        (boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals, seed_arr) = _init_agents(8, self.start, 1.0, seed=0.123)

        step_sim(
            boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
            seed_arr, self.dt,
            bound_size=self.bound,
            boid_count=8,
            rule_scalar=self.rule_scalar,
            max_speed=self.max_speed,
            sep_r=self.sep_r,
            ali_r=self.ali_r,
            coh_r=self.coh_r,
            sep_s=1.0,
            ali_s=1.0,
            coh_s=1.0,
            bnd_s=1.0,
            rand_s=0.1,
            obs_avoid_s=1.0,
            rand_wavelen_scalar=self.rand_wavelen_scalar,
            goal_gain=0.0,
            mesh_af=self.mesh_af,
            mesh_origin=self.mesh_origin,
            mesh_spacing=self.mesh_spacing,
            goals = self.goals,
            goal_idx=np.zeros((8,), dtype=np.int32),
            alive = self._alive
        )

        if self.doAnimation:
            self.fig = plt.figure(figsize=(11, 8))
            self.ax = self.fig.add_subplot(111, projection="3d")
            plt.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.98)

            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")
            self.ax.set_xlim3d(0, self.bound)
            self.ax.set_ylim3d(0, self.bound)
            self.ax.set_zlim3d(0, self.bound)

            self.boid_scatter = self.ax.scatter(boid_pos[:, 0],
                              boid_pos[:, 1],
                              boid_pos[:, 2],
                              s=6, depthshade=False)

            for goal in self.goals:
                self.goal_scatter = self.ax.scatter([goal[0]], [goal[1]], [goal[2]],
                                        s=80, marker="*", depthshade=False)
            
            if self.mesh_verts is not None and self.mesh_faces is not None:
                verts = self.mesh_verts
                faces = self.mesh_faces

                # build triangle vertex lists for Poly3DCollection
                tris = [verts[faces[f]].tolist() for f in range(faces.shape[0])]

                self.mesh_poly = Poly3DCollection(
                    tris,
                    alpha=0.25,
                    linewidths=0.5,
                    edgecolor="k",
                    facecolor=(0.6, 0.6, 0.6, 0.25),
                )
                self.ax.add_collection3d(self.mesh_poly)
            self.ax.view_init(elev=20, azim=25)
            plt.ion()
            plt.show()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        episode_seed = int(self._rng.integers(0, 2**31 - 1))
        self._episode_seed = episode_seed

    def step(self, action):
        if self._episode_seed is None:
            raise RuntimeError("Call reset() before step().")

        metrics, info = self._rollout_episode(np.asarray(action, dtype=np.float32).reshape(-1))

        time_pen = 0.0
        if not math.isnan(metrics.avg_time_to_goal):
            time_pen = metrics.avg_time_to_goal / (self.max_steps * self.dt + 1e-12)

        reward = (
            self.w_goal * metrics.frac_goal
            - self.w_time * time_pen
            + self.w_div * metrics.diversity_entropy
        )

        info.update({
            "frac_goal": metrics.frac_goal,
            "avg_time_to_goal": metrics.avg_time_to_goal,
            "diversity_entropy": metrics.diversity_entropy,
            "reward": float(reward),
        })

        obs = self._last_obs if self._last_obs is not None else np.zeros((6,), dtype=np.float32)

        self._episode_seed = None

        return obs, float(reward), True, False, info

    def _rollout_episode(self, action: np.ndarray) -> Tuple[EpisodeMetrics, Dict]:
        if action.shape[0] != 7:
            raise ValueError(f"action must have shape (7,), got {action.shape}")

        seed = int(self._episode_seed)
        rng = np.random.default_rng(seed)

        start = self.start
        goal = self.goal

        (boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
         seed_arr) = _init_agents(
            total_boids=self.boid_count,
            start_xyz=start,
            start_spread=self.start_spread,
            seed=float((seed % 1000000) / 1000000.0 + 0.123),
        )

        # Per-fish current goal index
        goal_idx = np.empty((self.boid_count,), dtype=np.int32)

        if self.start_goal_idx is None:
            # Random initial goal per fish
            goal_idx[:] = rng.integers(
                0,
                self.goals.shape[0],
                size=self.boid_count,
                dtype=np.int32
            )
        else:
            # All fish start at same node
            goal_idx[:] = np.int32(self.start_goal_idx)

        goal_idx = np.empty((self.boid_count,), dtype=np.int32)
        ever_hit = np.zeros((self.boid_count,), dtype=np.bool_)
        first_hit_t = np.full((self.boid_count,), np.nan, dtype=np.float32)

        if self.start_goal_idx is None:
            goal_idx[:] = rng.integers(0, self.goals.shape[0], size=self.boid_count, dtype=np.int32)
        else:
            goal_idx[:] = int(self.start_goal_idx)

        sep_s = float(action[0])
        ali_s = float(action[1])
        coh_s = float(action[2])
        bnd_s = float(action[3])
        rand_s = float(action[4])
        obs_avoid_s = float(action[5])
        goal_gain = float(action[6])

        self._alive.fill(True)
        self._reached.fill(False)
        self._t_reach.fill(np.nan)

        for step in range(self.max_steps):
            step_sim(
                boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
                seed_arr, self.dt,
                bound_size=self.bound,
                boid_count=self.boid_count,
                rule_scalar=self.rule_scalar,
                max_speed=self.max_speed,
                sep_r=self.sep_r,
                ali_r=self.ali_r,
                coh_r=self.coh_r,
                sep_s=sep_s,
                ali_s=ali_s,
                coh_s=coh_s,
                bnd_s=bnd_s,
                rand_s=rand_s,
                obs_avoid_s=obs_avoid_s,
                rand_wavelen_scalar=self.rand_wavelen_scalar,
                goal_gain=goal_gain,
                mesh_af=self.mesh_af,
                mesh_origin=self.mesh_origin,
                mesh_spacing=self.mesh_spacing,
                goals = self.goals,
                goal_idx=goal_idx,
                alive = self._alive
            )
            
            n_active = update_events_numba(
                boid_pos, boid_vel,
                self._alive,
                ever_hit, first_hit_t,
                goal_idx, self.goals, self.goal_W,
                self.goal_radius,
                step, self.dt,
                seed_arr,
                self.mesh_df, self.mesh_origin, self.mesh_spacing,
            )
            
            if n_active == 0:
                break
            
            if self.doAnimation and plt.fignum_exists(self.fig.number):
                self.boid_scatter._offsets3d = (boid_pos[:, 0], boid_pos[:, 1], boid_pos[:, 2])
                plt.draw()
                plt.pause(0.001)

            if self.returnTrajectory:
                self.trajectory_boid_pos[step, :, :] = boid_pos
                self.trajectory_boid_vel[step, :, :] = boid_vel

        frac_goal = float(np.sum(ever_hit)) / float(self.boid_count)
        avg_time_to_goal = float(mean_time_to_goal(first_hit_t, ever_hit))
        diversity = float(heading_entropy(boid_vel))

        metrics = EpisodeMetrics(
            frac_goal=frac_goal,
            avg_time_to_goal=avg_time_to_goal,
            diversity_entropy=diversity,
        )

        info = {
            "goal": self.goal,
            "start": self.start,
            "reached_count": self.boid_count - n_active,
            "steps_executed": step + 1 if self.max_steps > 0 else 0,
        }

        if self.returnTrajectory:
            info["trajectory_boid_pos"] = self.trajectory_boid_pos
            info["trajectory_boid_vel"] = self.trajectory_boid_vel
        else:
            info["trajectory_boid_pos"] = None
        return metrics, info

def precompute_mesh_avoidance(verts, faces,
                              origin, field_length, R,
                              avoid_r,
                              power=3.0, alpha=4.0):
    origin = np.asarray(origin, dtype=np.float32)
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    spacing = np.float32(field_length / (R - 1))

    df = build_distance_field(verts, faces, origin, spacing, R, np.float32(avoid_r))
    af = build_avoid_field_from_df(df, power=np.float32(power), alpha=np.float32(alpha))
    return af, df, origin, spacing


def make_torus_mesh(
        R=10.0,
        r=3.0,
        segR=48,
        segr=24,
        center=(25.0, 25.0, 25.0),
        yaw=0.0  # rotation around Z (radians)
    ):
    """
    Torus whose main ring lies in the YZ plane
    (wraps around X axis), with optional rotation
    around global Z axis.
    """

    cx, cy, cz = center
    cyaw = np.cos(yaw)
    syaw = np.sin(yaw)

    verts = []
    faces = []

    # ---- vertices ----
    for i in range(segR):
        theta = 2.0 * np.pi * i / segR
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for j in range(segr):
            phi = 2.0 * np.pi * j / segr
            cos_p = np.cos(phi)
            sin_p = np.sin(phi)

            # Torus around X-axis (local coords)
            x = r * sin_p
            y = (R + r * cos_p) * cos_t
            z = (R + r * cos_p) * sin_t

            # ---- rotate around Z ----
            x_rot = x * cyaw - y * syaw
            y_rot = x * syaw + y * cyaw
            z_rot = z

            verts.append([
                cx + x_rot,
                cy + y_rot,
                cz + z_rot
            ])

    verts = np.array(verts, dtype=np.float32)

    # ---- faces ----
    for i in range(segR):
        for j in range(segr):
            a = i * segr + j
            b = ((i + 1) % segR) * segr + j
            c = ((i + 1) % segR) * segr + (j + 1) % segr
            d = i * segr + (j + 1) % segr

            faces.append([a, b, d])
            faces.append([b, c, d])

    faces = np.array(faces, dtype=np.int32)

    return verts, faces

def make_sphere_mesh(
    R=2.0,
    seg_theta=16,     # longitude divisions
    seg_phi=16,       # latitude divisions
    center=(0.0, 0.0, 0.0)
):
    cx, cy, cz = center

    verts = []
    faces = []

    # Create vertices
    for i in range(seg_phi + 1):
        phi = np.pi * i / seg_phi
        for j in range(seg_theta):
            theta = 2.0 * np.pi * j / seg_theta

            x = R * np.sin(phi) * np.cos(theta)
            y = R * np.sin(phi) * np.sin(theta)
            z = R * np.cos(phi)

            verts.append([x + cx, y + cy, z + cz])

    verts = np.array(verts, dtype=np.float32)

    # Create faces
    for i in range(seg_phi):
        for j in range(seg_theta):
            next_j = (j + 1) % seg_theta

            a = i * seg_theta + j
            b = i * seg_theta + next_j
            c = (i + 1) * seg_theta + j
            d = (i + 1) * seg_theta + next_j

            if i != 0:
                faces.append([a, c, b])
            if i != seg_phi - 1:
                faces.append([b, c, d])

    faces = np.array(faces, dtype=np.int32)

    return verts, faces

def merge_meshes(meshes):
    """
    meshes: list of (verts, faces)
      verts: (Nv,3) float32/float64
      faces: (Nf,3) int32/int64 indices into verts
    returns: (V,F) merged
    """
    V_all = []
    F_all = []
    v_off = 0

    for V, F in meshes:
        V = np.asarray(V, dtype=np.float32)
        F = np.asarray(F, dtype=np.int32)

        V_all.append(V)
        F_all.append(F + v_off)

        v_off += V.shape[0]

    Vm = np.vstack(V_all).astype(np.float32)
    Fm = np.vstack(F_all).astype(np.int32)
    return Vm, Fm

if __name__ == "__main__":
    verts, faces = make_torus_mesh(
                        R=3.0,
                        r=1.0,
                        segR=12,
                        segr=12,
                        center=(20.0, 20.0, 20.0)
                    )
    goals = np.array([
        [34.0, 20.0, 20.0],  # 0 - initial
        [40.0, 20.0, 20.0],  # 1
        [40.0, 30.0, 20.0],  # 2
        [40.0, 10.0, 20.0],  # 3
    ], dtype=np.float32)
    goal_W = np.array([
        [0.0, 2.0, 1.0, 1.0],  # from 0 → {1,2,3}
        [0.0, 1.0, 0.0, 0.0],  # from 1 → 0
        [0.0, 0.0, 1.0, 0.0],  # from 2 → 0
        [0.0, 0.0, 0.0, 1.0],  # from 3 → 0
    ], dtype=np.float32)

    t0 = time.time()
    env = FishGoalEnv(boid_count=600, max_steps=500, dt=1, doAnimation = True, returnTrajectory = False, verts=verts, faces=faces, goals=goals, goal_W=goal_W)
    t1 = time.time()
    print("Time to make env:", t1 - t0)
    
    env.reset(seed=0)
    """
    Parameters order:
    0: separation scalar
    1: alignment scalar
    2: cohesion scalar
    3: boundary scalar
    4: randomness scalar
    5: obstacle avoidance scalar
    6: goal attraction gain
    """
    action = np.array([1.0, #0: separation scalar
                       1.0, #1: alignment scalar
                       1.0, #2: cohesion scalar
                       1.0, #3: boundary scalar
                       1.0, #4: randomness scalar
                       1.0, #5: obstacle avoidance scalar
                       0.3, #6: goal attraction gain
                       ], dtype=np.float32)
    
    load_theta = False
    if load_theta:
        theta_path = "save/best_policy.pkl"
        action = pickle.load(open(theta_path, "rb"))['best_theta']
    
    t = []
    for _ in range(10):
        t0 = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        env.reset(seed=0)
        t1 = time.time()
        t.append(t1 - t0)
    print(f"Median step time: {np.median(t)*1000.0:.2f} ms")