from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Polygon

import racing_viewer as regular
import racing_viewer_obstacles as obstacles
from system import ackermann, four_wheel


SPG_VARIANT = "sensitivity_projected_gaussian_prior_mppi"
VEHICLES = ("ackermann", "four_wheel")
OBSTACLE_WALL_MODES = ("no_wall", "dynamic_1", "dynamic_2")
REGULAR_DEFAULTS = {
    "laps": 10,
    "num_rollouts": 4096,
    "lbps_delta": 0.9,
    "seed": 1,
    "horizon": 10,
    "temporal_noise_smoothing": float(ackermann.MPPIConfig().temporal_noise_smoothing),
    "sigma0_scale": 1.0,
    "v_max": 8.0,
    "hard_collision_clearance": float(ackermann.MPPIConfig().hard_collision_clearance),
}
OBSTACLE_DEFAULTS = {
    "laps": 10,
    "num_rollouts": 4096,
    "lbps_delta": 0.9,
    "seed": 1,
    "horizon": 25,
    "temporal_noise_smoothing": float(ackermann.MPPIConfig().temporal_noise_smoothing),
    "sigma0_scale": 1.0,
    "v_max": 6.0,
    "hard_collision_clearance": 0.02,
}
REGULAR_PLAYBACK = 1.0
OBSTACLE_PLAYBACK = 4.0


def _transform(points: np.ndarray, origin: tuple[float, float], angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ rotation.T + np.asarray(origin, dtype=np.float64)


class VehicleArtists:
    def __init__(self, ax) -> None:
        self.ax = ax
        self.body = Polygon(
            np.zeros((4, 2)),
            closed=True,
            facecolor=ackermann.BODY_COLOR,
            edgecolor="black",
            linewidth=1.1,
            alpha=0.92,
            zorder=12,
        )
        ax.add_patch(self.body)
        self.wheels: list[Polygon] = []
        for _ in range(4):
            wheel = Polygon(
                np.zeros((4, 2)),
                closed=True,
                facecolor="0.10",
                edgecolor="black",
                linewidth=0.6,
                zorder=13,
            )
            ax.add_patch(wheel)
            self.wheels.append(wheel)
        self.velocity_arrow = FancyArrowPatch(
            (0.0, 0.0),
            (0.0, 0.0),
            arrowstyle="-|>",
            mutation_scale=10.0,
            linewidth=1.2,
            color="#1f77b4",
            zorder=14,
        )
        ax.add_patch(self.velocity_arrow)
        self.nose_line, = ax.plot([], [], color="black", linewidth=1.1, zorder=14)

    def update(self, state: np.ndarray, cfg: object, model_name: str) -> None:
        x, y, heading, vx, vy, _, steering = map(float, state[:7])
        lf = float(cfg.front_axle_distance)
        lr = float(cfg.rear_axle_distance)
        body_length = float(cfg.vehicle_length)
        body_width = float(cfg.vehicle_width)
        body_shape = np.asarray(
            [
                [-0.5 * body_length, -0.5 * body_width],
                [0.5 * body_length, -0.5 * body_width],
                [0.5 * body_length, 0.5 * body_width],
                [-0.5 * body_length, 0.5 * body_width],
            ],
            dtype=np.float64,
        )
        self.body.set_facecolor(four_wheel.BODY_COLOR if model_name == "four_wheel" else ackermann.BODY_COLOR)
        self.body.set_xy(_transform(body_shape, (x, y), heading))
        wheel_shape = np.asarray(
            [[-0.11, -0.0275], [0.11, -0.0275], [0.11, 0.0275], [-0.11, 0.0275]],
            dtype=np.float64,
        )
        half_track = 0.5 * float(getattr(cfg, "track_width", 0.86 * body_width))
        c = math.cos(heading)
        s = math.sin(heading)

        def body_to_world(longitudinal: float, lateral: float) -> tuple[float, float]:
            return x + longitudinal * c - lateral * s, y + longitudinal * s + lateral * c

        wheel_specs = (
            (lf, half_track, heading + steering),
            (lf, -half_track, heading + steering),
            (-lr, half_track, heading),
            (-lr, -half_track, heading),
        )
        for wheel_index, (artist, spec) in enumerate(zip(self.wheels, wheel_specs)):
            longitudinal, lateral, wheel_heading = spec
            center = body_to_world(longitudinal, lateral)
            artist.set_xy(_transform(wheel_shape, center, wheel_heading))
            if model_name == "four_wheel" and state.size >= 13:
                normalized_speed = min(
                    abs(float(state[9 + wheel_index])) / max(float(cfg.wheel_speed_limit), 1e-9),
                    1.0,
                )
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


class RegularRenderer:
    def __init__(self, result, width_inches: float) -> None:
        aspect = (regular.TRACK_WIDTH + 1.6) / (regular.TRACK_HEIGHT + 1.6)
        self.fig = Figure(figsize=(width_inches, width_inches / aspect), facecolor="white")
        FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.ax.set_position([0.0, 0.0, 1.0, 1.0])
        self.ax.set_xlim(-0.8, regular.TRACK_WIDTH + 0.8)
        self.ax.set_ylim(-0.8, regular.TRACK_HEIGHT + 0.8)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_axis_off()
        outside_color = "0.92"
        self.ax.set_facecolor(outside_color)
        self.fig.patch.set_facecolor(outside_color)
        self.ax.add_patch(
            Polygon(
                regular._stadium_boundary(regular.OUTER_RADIUS),
                closed=True,
                facecolor="white",
                edgecolor="0.25",
                linewidth=2.0,
                zorder=1,
            )
        )
        self.ax.add_patch(
            Polygon(
                regular._stadium_boundary(regular.OUTER_RADIUS - regular.ROAD_WIDTH),
                closed=True,
                facecolor=outside_color,
                edgecolor="0.25",
                linewidth=2.0,
                zorder=2,
            )
        )
        start, _ = regular.centerline_point_tangent(regular.START_S)
        self.ax.plot(
            [start[0], start[0]],
            [start[1] - 1.0, start[1] + 1.0],
            color="0.15",
            linewidth=2.0,
            zorder=4,
        )
        self.executed_line, = self.ax.plot([], [], color="#1f77b4", linewidth=2.4, zorder=5)
        self.prediction_line, = self.ax.plot(
            [], [], color="#ff7f0e", linewidth=2.2, linestyle="--", alpha=0.95, zorder=7
        )
        self.vehicle = VehicleArtists(self.ax)
        self.result = result

    def update(self, frame: int) -> None:
        frame = int(np.clip(frame, 0, len(self.result.states) - 1))
        state = self.result.states[frame]
        path = self.result.states[: frame + 1, :2]
        recent = path[-100:]
        self.executed_line.set_data(recent[:, 0], recent[:, 1])
        if self.result.mppi_predictions:
            pred_idx = min(frame, len(self.result.mppi_predictions) - 1)
            pred = self.result.mppi_predictions[pred_idx]
            self.prediction_line.set_data(pred[:, 0], pred[:, 1])
            self.prediction_line.set_visible(True)
        else:
            self.prediction_line.set_visible(False)
        self.vehicle.update(state, self.result.cfg, self.result.model_name)


class ObstacleRenderer:
    def __init__(self, result, width_inches: float) -> None:
        aspect = (obstacles.TRACK_WIDTH + 1.6) / (obstacles.TRACK_HEIGHT + 1.6)
        self.fig = Figure(figsize=(width_inches, width_inches / aspect), facecolor="white")
        FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.ax.set_position([0.0, 0.0, 1.0, 1.0])
        self.ax.set_xlim(-0.8, obstacles.TRACK_WIDTH + 0.8)
        self.ax.set_ylim(-0.8, obstacles.TRACK_HEIGHT + 0.8)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_axis_off()
        self.ax.set_facecolor("white")
        self.fig.patch.set_facecolor("white")
        for vertices in result.obstacle_vertices:
            self.ax.add_patch(
                Polygon(
                    np.asarray(vertices, dtype=np.float64),
                    closed=True,
                    facecolor="0.25",
                    edgecolor="0.05",
                    linewidth=1.1,
                    alpha=0.78,
                    zorder=3,
                )
            )
        start, _ = obstacles.centerline_point_tangent(obstacles.START_S)
        self.ax.plot(
            [start[0], start[0]],
            [start[1] - 1.0, start[1] + 1.0],
            color="0.15",
            linewidth=2.0,
            zorder=4,
        )
        self.executed_line, = self.ax.plot([], [], color="#1f77b4", linewidth=2.4, zorder=5)
        self.prediction_line, = self.ax.plot(
            [], [], color="#ff7f0e", linewidth=2.2, linestyle="--", alpha=0.95, zorder=7
        )
        self.vehicle = VehicleArtists(self.ax)
        self.result = result
        self.dynamic_wall_patches: list[Polygon] = []
        self.prior_mean_artists: dict[int, object] = {}
        self.prior_cov_artists: dict[int, PolyCollection] = {}
        self.prior_cov_sample_indices: dict[int, np.ndarray] = {}
        self.prior_visual_cursors = np.full(len(result.prior_bank.modes), -1, dtype=np.int64)

    def _sync_dynamic_walls(self, frame: int) -> None:
        walls: list[np.ndarray] = []
        if self.result.dynamic_wall_history:
            idx = int(np.clip(frame, 0, len(self.result.dynamic_wall_history) - 1))
            walls = self.result.dynamic_wall_history[idx]
        while len(self.dynamic_wall_patches) < len(walls):
            patch = Polygon(
                np.zeros((3, 2)),
                closed=True,
                facecolor="0.08",
                edgecolor="0.02",
                linewidth=1.3,
                alpha=0.92,
                zorder=4,
            )
            self.ax.add_patch(patch)
            self.dynamic_wall_patches.append(patch)
        for i, patch in enumerate(self.dynamic_wall_patches):
            if i < len(walls):
                patch.set_xy(np.asarray(walls[i], dtype=np.float64))
                patch.set_visible(True)
            else:
                patch.set_visible(False)

    def _mean_artist(self, mode_index: int):
        m = int(mode_index)
        artist = self.prior_mean_artists.get(m)
        if artist is None:
            artist, = self.ax.plot(
                [], [], color=obstacles.PRIOR_PURPLE, linewidth=1.35, linestyle="--", alpha=0.5, zorder=5.5
            )
            self.prior_mean_artists[m] = artist
        return artist

    def _cov_artist(self, mode_index: int) -> tuple[PolyCollection, np.ndarray]:
        m = int(mode_index)
        artist = self.prior_cov_artists.get(m)
        if artist is None:
            polygons, sample_indices = obstacles._sparse_covariance_mode_geometry(self.result.prior_bank, m)
            artist = PolyCollection(
                polygons,
                facecolors=[(*obstacles.PRIOR_PURPLE, 0.0)] * len(polygons),
                edgecolors="none",
                linewidths=0.0,
                zorder=4.5,
            )
            self.ax.add_collection(artist)
            self.prior_cov_artists[m] = artist
            self.prior_cov_sample_indices[m] = sample_indices
        return artist, self.prior_cov_sample_indices[m]

    def _hide_inactive_priors(self, active_set: set[int]) -> None:
        for mode_index, artist in self.prior_mean_artists.items():
            if mode_index not in active_set:
                artist.set_visible(False)
        for mode_index, artist in self.prior_cov_artists.items():
            if mode_index not in active_set:
                artist.set_visible(False)

    def _update_priors(self, pred_idx: int) -> None:
        if not self.result.active_prior_indices_history or pred_idx >= len(self.result.active_prior_indices_history):
            self._hide_inactive_priors(set())
            return
        active_indices = np.asarray(self.result.active_prior_indices_history[pred_idx], dtype=np.int64)
        if active_indices.size == 0:
            self._hide_inactive_priors(set())
            return
        prior_state = self.result.states[pred_idx]
        starts, ends, updated_cursors = obstacles._local_racing_ranges(
            prior_state,
            self.result.prior_bank,
            self.result.cfg,
            active_indices,
            self.prior_visual_cursors,
        )
        self.prior_visual_cursors = np.ascontiguousarray(updated_cursors, dtype=np.int64)
        probabilities = np.asarray(self.result.prior_bank.probabilities[active_indices], dtype=np.float64)
        mass = float(np.sum(probabilities))
        if mass <= 1e-15:
            probabilities = np.full(len(active_indices), 1.0 / float(len(active_indices)), dtype=np.float64)
        else:
            probabilities /= mass
        peak_probability = float(np.max(probabilities))
        active_set = {int(value) for value in active_indices}
        self._hide_inactive_priors(active_set)
        for q, global_index in enumerate(active_indices):
            m = int(global_index)
            start_index = int(starts[q])
            end_index = int(ends[q])
            probability = float(probabilities[q])
            mean = np.asarray(
                self.result.prior_bank.mean_paths[m, start_index : end_index + 1],
                dtype=np.float64,
            )
            mean_artist = self._mean_artist(m)
            mean_artist.set_data(mean[:, 0], mean[:, 1])
            mean_artist.set_alpha(obstacles._prior_mean_alpha(probability, peak_probability))
            mean_artist.set_linewidth(obstacles._prior_mean_linewidth(probability, peak_probability))
            mean_artist.set_visible(True)
            cov_artist, sample_indices = self._cov_artist(m)
            visible = obstacles._covariance_sample_visibility(
                sample_indices,
                start_index,
                end_index,
                int(self.result.prior_bank.localization_lengths[m]),
            )
            colors = np.zeros((len(sample_indices), 4), dtype=np.float64)
            if len(sample_indices):
                colors[:, 0] = obstacles.PRIOR_PURPLE[0]
                colors[:, 1] = obstacles.PRIOR_PURPLE[1]
                colors[:, 2] = obstacles.PRIOR_PURPLE[2]
                colors[visible, 3] = obstacles._prior_covariance_alpha(probability, peak_probability)
            cov_artist.set_facecolors(colors)
            cov_artist.set_visible(bool(np.any(visible)))

    def update(self, frame: int) -> None:
        frame = int(np.clip(frame, 0, len(self.result.states) - 1))
        state = self.result.states[frame]
        path = self.result.states[: frame + 1, :2]
        recent = path[-100:]
        self.executed_line.set_data(recent[:, 0], recent[:, 1])
        self._sync_dynamic_walls(frame)
        if self.result.mppi_predictions:
            pred_idx = min(frame, len(self.result.mppi_predictions) - 1)
            pred = self.result.mppi_predictions[pred_idx]
            self.prediction_line.set_data(pred[:, 0], pred[:, 1])
            self.prediction_line.set_visible(True)
            self._update_priors(pred_idx)
        else:
            self.prediction_line.set_visible(False)
            self._hide_inactive_priors(set())
        self.vehicle.update(state, self.result.cfg, self.result.model_name)


def _frame_plan(result, playback: float, max_fps: float) -> tuple[list[int], float]:
    raw_fps = float(playback) / max(float(result.cfg.dt), 1e-9)
    stride = max(1, int(math.ceil(raw_fps / max(float(max_fps), 1.0))))
    frames = list(range(0, len(result.states), stride))
    if frames[-1] != len(result.states) - 1:
        frames.append(len(result.states) - 1)
    fps = raw_fps / float(stride)
    return frames, fps


def _save_gif(renderer, result, path: Path, playback: float, max_fps: float, dpi: int) -> None:
    frames, fps = _frame_plan(result, playback, max_fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    print(f"Writing {path.name}: {len(frames)} frames at {fps:.2f} fps")
    with writer.saving(renderer.fig, str(path), dpi=int(dpi)):
        for position, frame in enumerate(frames, start=1):
            renderer.update(frame)
            renderer.fig.canvas.draw()
            writer.grab_frame(facecolor=renderer.fig.get_facecolor())
            if position == 1 or position == len(frames) or position % 100 == 0:
                print(f"  frame {position}/{len(frames)}")


def _run_regular(model_name: str, output_dir: Path, width_inches: float, dpi: int, max_fps: float) -> Path:
    print(f"Simulating regular racing / {model_name} / SPG")
    result = regular.run_race(
        variant_value=SPG_VARIANT,
        model_name=model_name,
        record_predictions=True,
        **REGULAR_DEFAULTS,
    )
    path = output_dir / f"racing_spg_{model_name}.gif"
    renderer = RegularRenderer(result, width_inches)
    _save_gif(renderer, result, path, REGULAR_PLAYBACK, max_fps, dpi)
    return path


def _run_obstacles(
    model_name: str,
    wall_mode: str,
    output_dir: Path,
    width_inches: float,
    dpi: int,
    max_fps: float,
) -> Path:
    print(f"Simulating obstacle racing / {model_name} / {wall_mode} / SPG")
    result = obstacles.run_race(
        variant_value=SPG_VARIANT,
        model_name=model_name,
        wall_mode=wall_mode,
        record_predictions=True,
        **OBSTACLE_DEFAULTS,
    )
    suffix = "" if wall_mode == "no_wall" else f"_{wall_mode}"
    path = output_dir / f"racing_obstacles{suffix}_spg_{model_name}.gif"
    renderer = ObstacleRenderer(result, width_inches)
    _save_gif(renderer, result, path, OBSTACLE_PLAYBACK, max_fps, dpi)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the default SPG races as clean GIFs.")
    parser.add_argument("--output-dir", type=Path, default=Path("gifs"))
    parser.add_argument("--viewer", choices=("all", "regular", "obstacles"), default="obstacles")
    parser.add_argument("--vehicle", choices=("all", "ackermann", "four_wheel"), default="four_wheel")
    parser.add_argument("--width-inches", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--max-fps", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    vehicles = VEHICLES if args.vehicle == "all" else (args.vehicle,)
    output_dir = args.output_dir.expanduser().resolve()
    outputs: list[Path] = []
    if args.viewer in ("all", "regular"):
        for model_name in vehicles:
            outputs.append(_run_regular(model_name, output_dir, args.width_inches, args.dpi, args.max_fps))
    if args.viewer in ("all", "obstacles"):
        for model_name in vehicles:
            for wall_mode in OBSTACLE_WALL_MODES:
                outputs.append(
                    _run_obstacles(
                        model_name, wall_mode, output_dir, args.width_inches, args.dpi, args.max_fps
                    )
                )
    print("Finished")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()