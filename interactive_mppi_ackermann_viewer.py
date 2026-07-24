#!/usr/bin/env python3
"""Interactive viewer for the dynamic Ackermann MPPI navigation experiment.

Run this file from the project root so ``runs_ackerman_dynamic.py`` can import
geometry/, RL/, graph/, planner.py, and save/best_policy.pkl. The geometric
planner and homotopy-prior construction are unchanged; only the predictive
vehicle dynamics and path-to-control conversion use the dynamic single-track
Ackermann model.

The viewer loads the controller module without executing its batch entry point.
It supports controller, environment, wall scenario, seed, rollout-count, and
playback-speed selection. Each selected trial is simulated in a worker thread
and then animated in a Tk window with the Ackermann body, wheel steering,
executed path, current MPPI output, active obstacles, and trajectory prior.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import os
import queue
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:  # pragma: no cover - platform dependent
    raise SystemExit("Tkinter is required to run the interactive viewer.") from exc

import matplotlib

# TkAgg must be selected before importing pyplot/Tk canvas helpers.
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Polygon


VARIANTS = [
    ("Standard MPPI", "standard_mppi"),
    ("Control bank", "control_bank_mppi"),
    ("Gaussian prior", "gaussian_prior_mppi"),
    ("Corridor prior", "corridor_prior_mppi"),
    ("Frenet prior", "frenet_corridor_mppi"),
]

CONDITIONS = [
    ("No wall", "no_wall"),
    ("Static wall", "static_wall"),
    ("Dynamic wall", "dynamic_wall"),
]

SCENARIOS = [
    ("Wall 0-1", "wall_0_1"),
    ("Wall 0-2", "wall_0_2"),
    ("Wall 1-2", "wall_1_2"),
    ("Walls 0-1 and 1-2", "walls_0_1__1_2"),
]

DISPLAY_TO_VARIANT = dict(VARIANTS)
VARIANT_TO_DISPLAY = {value: label for label, value in VARIANTS}
DISPLAY_TO_CONDITION = dict(CONDITIONS)
DISPLAY_TO_SCENARIO = dict(SCENARIOS)


@dataclass
class TrialBundle:
    module: Any
    condition: str
    scenario_id: str
    variant_value: str
    cfg: Any
    result: dict[str, Any]
    modes: list[Any]
    start: np.ndarray
    goal: np.ndarray
    bounds_xy: Any
    base_obstacles: list[Any]
    blocker: list[Any]
    controller_seed: int
    swarm_seed: int


def load_python_module(path: Path, module_name: str) -> Any:
    """Load a Python file as a module without invoking its __main__ block."""
    if not path.exists():
        raise FileNotFoundError(f"Controller module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def first_existing_path(explicit: Optional[str], candidates: list[str]) -> Path:
    """Resolve an explicit module path or the first existing candidate."""
    if explicit:
        return Path(explicit).expanduser().resolve()

    roots = [Path.cwd(), Path(__file__).resolve().parent]
    checked: list[Path] = []
    for root in roots:
        for candidate in candidates:
            path = (root / candidate).resolve()
            checked.append(path)
            if path.exists():
                return path

    formatted = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "No suitable controller module was found. Checked:\n" + formatted
    )


def polygon_vertices(obstacle: Any) -> np.ndarray:
    vertices = getattr(obstacle, "vertices", obstacle)
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"Unsupported obstacle geometry with shape {array.shape}")
    return array[:, :2]


def normalized_bounds(bounds_xy: Any) -> tuple[float, float, float, float]:
    """Accept either ((xmin,ymin),(xmax,ymax)) or ((xmin,xmax),(ymin,ymax))."""
    a = np.asarray(bounds_xy[0], dtype=float)
    b = np.asarray(bounds_xy[1], dtype=float)
    if a.shape == (2,) and b.shape == (2,) and a[0] <= b[0] and a[1] <= b[1]:
        return float(a[0]), float(b[0]), float(a[1]), float(b[1])
    return float(bounds_xy[0][0]), float(bounds_xy[0][1]), float(bounds_xy[1][0]), float(bounds_xy[1][1])


def make_protocol_config(module: Any, num_rollouts: int) -> Any:
    """Create the dynamic Ackermann configuration used by the batch script."""
    cfg = module.MPPIConfig()
    settings = {
        "horizon": 50,
        "num_rollouts": int(num_rollouts),
        "dt": 0.12,
        "dynamics_substeps": 4,
        "max_empirical_nominals_per_mode": 16,
        "swarm_init_probability": 0.60,
        "sigma_floor": 0.25,
        "max_precision": 10.0,
        "w_control_smooth": 0.40,
        "smooth_accel_weight": 0.5,
        "smooth_steering_rate_weight": 2.0,
        "apply_control_lowpass": False,
        "control_lowpass_alpha": 0.0,
        "base_safety_margin": 0.0,
        "collision_substeps": 5,
        "hard_collision_clearance": 0.01,
        "hard_collision_penalty": 800_000.0,
        "suppress_blocked_modes": True,
        "mode_blocking_clearance": 0.02,
        "mode_blocking_substeps": 2,
        "v_min": -2.8,
        "v_max": 2.8,
        "accel_min": -3.5,
        "accel_max": 5.0,
        "steering_min": -1.2,
        "steering_max": 1.2,
        "steering_rate_min": -50.0,
        "steering_rate_max": 50.0,
        "w_goal": 110.0,
        "w_control": 0.004,
        "w_steering_angle": 0.0,
        "w_yaw_rate": 0.0,
        "lambda_temperature": 2.2,
        "max_delta_accel": 1.20,
        "max_delta_steering_rate": 5.20,
        # Control-space exploration is applied to acceleration and steering rate.
        "noise_accel": 0.80,
        "noise_steering_rate": 1.00,
    }
    for name, value in settings.items():
        if hasattr(cfg, name):
            setattr(cfg, name, value)
    # Re-run validation because dataclass fields were modified after construction.
    post_init = getattr(cfg, "__post_init__", None)
    if callable(post_init):
        post_init()
    return cfg


class InteractiveMPPIViewer:
    def __init__(self, root: tk.Tk, module_path: Path) -> None:
        self.root = root
        self.root.title("Interactive MPPI viewer - dynamic Ackermann")
        self.root.geometry("1450x900")
        self.root.minsize(1080, 700)

        self.module_path = module_path
        self.module: Optional[Any] = None
        self.mode_cache: dict[tuple[Any, ...], list[Any]] = {}
        self.numba_warm_cache: set[str] = set()
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.bundle: Optional[TrialBundle] = None
        self.frame_index = 0
        self.playing = False
        self.after_id: Optional[str] = None
        self.poll_after_id: Optional[str] = None
        self.updating_slider = False
        self.closing = False

        self._build_ui()
        self.poll_after_id = self.root.after(100, self._poll_worker)

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="nsw")
        controls.columnconfigure(0, weight=1)

        plot_frame = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        # ttk.Label(controls, text="Experiment", font=("TkDefaultFont", 12, "bold")).grid(
        #     row=0, column=0, sticky="w", pady=(0, 10)
        # )

        self.variant_var = tk.StringVar(value="Gaussian prior")
        self.condition_var = tk.StringVar(value="Dynamic wall")
        self.scenario_var = tk.StringVar(value="Walls 0-1 and 1-2")
        self.seed_var = tk.StringVar(value="1")
        self.swarm_seed_var = tk.StringVar(value="5")
        self.rollouts_var = tk.StringVar(value="32")
        self.speed_var = tk.StringVar(value="4.0")
        self.show_collision_var = tk.BooleanVar(value=True)
        self.show_all_modes_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.frame_label_var = tk.StringVar(value="Frame 0 / 0")

        row = 1
        row = self._add_combo(controls, row, "Controller variant", self.variant_var, [label for label, _ in VARIANTS])
        row = self._add_combo(controls, row, "Condition", self.condition_var, [label for label, _ in CONDITIONS])
        row = self._add_combo(controls, row, "Wall scenario", self.scenario_var, [label for label, _ in SCENARIOS])
        row = self._add_entry(controls, row, "Controller seed", self.seed_var)
        row = self._add_entry(controls, row, "Swarm seed", self.swarm_seed_var)
        row = self._add_entry(controls, row, "Rollouts per step", self.rollouts_var)
        row = self._add_combo(controls, row, "Playback speed", self.speed_var, ["0.5", "1.0", "2.0", "4.0", "8.0"])

        ttk.Checkbutton(
            controls,
            text="Show collision representation",
            variable=self.show_collision_var,
            command=self._redraw_current,
        ).grid(row=row, column=0, sticky="w", pady=(8, 2))
        row += 1
        ttk.Checkbutton(
            controls,
            text="Show all mode means",
            variable=self.show_all_modes_var,
            command=self._redraw_current,
        ).grid(row=row, column=0, sticky="w", pady=2)
        row += 1

        buttons = ttk.Frame(controls)
        buttons.grid(row=row, column=0, sticky="ew", pady=(14, 6))
        buttons.columnconfigure((0, 1), weight=1)
        self.run_button = ttk.Button(buttons, text="Run selected case", command=self.run_selected)
        self.run_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.play_button = ttk.Button(buttons, text="Play", command=self.toggle_play, state="disabled")
        self.play_button.grid(row=1, column=0, sticky="ew", padx=(0, 3))
        self.restart_button = ttk.Button(buttons, text="Restart", command=self.restart_animation, state="disabled")
        self.restart_button.grid(row=1, column=1, sticky="ew", padx=(3, 0))
        row += 1

        ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=10)
        row += 1
        ttk.Label(controls, text="Animation", font=("TkDefaultFont", 11, "bold")).grid(
            row=row, column=0, sticky="w"
        )
        row += 1
        ttk.Label(controls, textvariable=self.frame_label_var).grid(row=row, column=0, sticky="w", pady=(4, 2))
        row += 1
        self.frame_scale = ttk.Scale(
            controls,
            from_=0,
            to=0,
            orient="horizontal",
            command=self._on_frame_slider,
            state="disabled",
        )
        self.frame_scale.grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=10)
        row += 1
        ttk.Label(controls, text="Status", font=("TkDefaultFont", 11, "bold")).grid(
            row=row, column=0, sticky="w"
        )
        row += 1
        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=280,
            justify="left",
        ).grid(row=row, column=0, sticky="nw", pady=(4, 0))
        controls.rowconfigure(row, weight=1)

        self.condition_var.trace_add("write", lambda *_: self._update_condition_controls())
        self._update_condition_controls()

        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.figure.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.93)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")
        self._draw_empty()

    @staticmethod
    def _add_combo(parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, values: list[str]) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(5, 2))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=31)
        combo.grid(row=row + 1, column=0, sticky="ew")
        variable._combo_widget = combo  # type: ignore[attr-defined]
        return row + 2

    @staticmethod
    def _add_entry(parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(5, 2))
        entry = ttk.Entry(parent, textvariable=variable, width=33)
        entry.grid(row=row + 1, column=0, sticky="ew")
        return row + 2

    def _update_condition_controls(self) -> None:
        combo = getattr(self.scenario_var, "_combo_widget", None)
        if combo is not None:
            combo.configure(state="disabled" if self.condition_var.get() == "No wall" else "readonly")

    def _draw_empty(self) -> None:
        self.ax.clear()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.2)
        self.ax.set_title("Select a controller and run a case")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.canvas.draw_idle()

    def _get_module(self) -> Any:
        if self.module is None:
            self.module = load_python_module(
                self.module_path, "interactive_ackermann_mppi_engine"
            )
            available = {member.value for member in self.module.ControllerVariant}
            unsupported = [value for _, value in VARIANTS if value not in available]
            if unsupported:
                raise RuntimeError(
                    "Viewer/controller variant mismatch. Unsupported values: "
                    + ", ".join(unsupported)
                )

            cfg = self.module.MPPIConfig()
            required_fields = {
                "mass",
                "yaw_inertia",
                "front_axle_distance",
                "rear_axle_distance",
                "steering_rate_max",
                "cornering_stiffness_front",
            }
            missing = sorted(name for name in required_fields if not hasattr(cfg, name))
            if missing:
                raise RuntimeError(
                    "The selected module is not the dynamic Ackermann controller. "
                    "Missing MPPIConfig fields: " + ", ".join(missing)
                )
        return self.module

    def _warm_numba_for_case(
        self,
        *,
        module: Any,
        variant: Any,
        variant_value: str,
        modes: list[Any],
        base_obstacles: list[Any],
        runtime_blocker: list[Any],
        start: np.ndarray,
        goal: np.ndarray,
        condition: str,
        trigger_progress: Optional[float],
        blocker_from_start: bool,
        cfg: Any,
        seed: int,
    ) -> None:
        """Compile the selected controller's Numba kernels before timed execution.

        A one-step dry run is used because it follows the same controller branch as
        the real trial. The dry-run runtime is discarded. Numba specializes on
        array dtype and dimensionality rather than the particular rollout count or
        horizon length, so a small configuration is sufficient for compilation.
        """
        key = variant_value
        if key in self.numba_warm_cache:
            return

        # Nothing to preload when the imported controller module has no Numba.
        if getattr(module, "njit", None) is None:
            self.numba_warm_cache.add(key)
            return

        warm_cfg = copy.deepcopy(cfg)
        if hasattr(warm_cfg, "horizon"):
            warm_cfg.horizon = min(int(getattr(warm_cfg, "horizon", 28)), 6)
        if hasattr(warm_cfg, "num_rollouts"):
            warm_cfg.num_rollouts = 32
        if hasattr(warm_cfg, "max_empirical_nominals_per_mode"):
            warm_cfg.max_empirical_nominals_per_mode = min(
                int(getattr(warm_cfg, "max_empirical_nominals_per_mode", 16)), 2
            )

        module.run_dynamic_blockage_controller(
            variant=variant,
            modes=modes,
            base_obstacles=base_obstacles,
            blocker=runtime_blocker,
            start=start,
            goal=goal,
            seed=seed,
            trigger_progress=trigger_progress,
            activation_preview_clearance=0.75,
            blocker_active_from_start=blocker_from_start,
            condition=condition,
            max_steps=1,
            goal_tolerance=0.30,
            mppi_cfg=warm_cfg,
            record_infos=False,
            record_obstacle_history=False,
        )
        self.numba_warm_cache.add(key)

    def run_selected(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Simulation running", "Wait for the current simulation to finish.")
            return

        try:
            seed = int(self.seed_var.get())
            swarm_seed = int(self.swarm_seed_var.get())
            rollouts = int(self.rollouts_var.get())
            if rollouts < 32:
                raise ValueError("Rollouts per step must be at least 32.")
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        condition = DISPLAY_TO_CONDITION[self.condition_var.get()]
        scenario_id = DISPLAY_TO_SCENARIO[self.scenario_var.get()]
        variant_value = DISPLAY_TO_VARIANT[self.variant_var.get()]

        self._stop_animation()
        self.bundle = None
        self.run_button.configure(state="disabled")
        self.play_button.configure(state="disabled")
        self.restart_button.configure(state="disabled")
        self.frame_scale.configure(state="disabled")
        self.status_var.set(
            "Loading controller and generating the trajectory prior. "
            "The first prior construction can take several minutes."
        )

        settings = {
            "condition": condition,
            "scenario_id": scenario_id,
            "variant_value": variant_value,
            "seed": seed,
            "swarm_seed": swarm_seed,
            "rollouts": rollouts,
        }
        self.worker = threading.Thread(target=self._simulation_worker, args=(settings,), daemon=True)
        self.worker.start()

    def _simulation_worker(self, settings: dict[str, Any]) -> None:
        try:
            bundle = self._prepare_and_run_trial(**settings)
            self.worker_queue.put(("success", bundle))
        except Exception:
            self.worker_queue.put(("error", traceback.format_exc()))

    def _prepare_and_run_trial(
        self,
        *,
        condition: str,
        scenario_id: str,
        variant_value: str,
        seed: int,
        swarm_seed: int,
        rollouts: int,
    ) -> TrialBundle:
        module = self._get_module()
        cfg = make_protocol_config(module, rollouts)

        scale, bounds_xy, bounds_ranges, start, goal, original_obstacles = module.build_default_scene()
        # The environment is fixed: always use the default obstacle placement.
        default_obstacles = list(original_obstacles)

        scenarios = {scenario.scenario_id: scenario for scenario in module.default_dynamic_wall_scenarios()}
        if scenario_id not in scenarios:
            available = ", ".join(sorted(scenarios))
            raise KeyError(f"Scenario {scenario_id!r} is unavailable. Available: {available}")
        scenario = scenarios[scenario_id]

        fixed_centers = tuple(module.obstacle_center(obs).copy() for obs in original_obstacles)
        blocker = module.make_wall_blockers_between_centers(
            centers=fixed_centers,
            pairs=scenario.wall_pairs,
            width=scenario.wall_width,
            extension=scenario.wall_extension,
        )

        if condition == "static_wall":
            prior_obstacles = list(default_obstacles) + list(blocker)
            base_obstacles = prior_obstacles
            runtime_blocker: list[Any] = []
            blocker_from_start = True
            trigger_progress = None
            prior_key = ("static", scenario_id, swarm_seed)
        else:
            prior_obstacles = list(default_obstacles)
            base_obstacles = list(default_obstacles)
            runtime_blocker = [] if condition == "no_wall" else list(blocker)
            blocker_from_start = False
            trigger_progress = None if condition == "no_wall" else float(scenario.trigger_progress)
            prior_key = ("base", swarm_seed)

        if prior_key not in self.mode_cache:
            self.mode_cache[prior_key] = module.build_homotopy_modes_for_obstacles(
                start,
                goal,
                prior_obstacles,
                scale,
                bounds_xy,
                bounds_ranges,
                swarm_seed,
            )
        modes = self.mode_cache[prior_key]

        try:
            variant = module.ControllerVariant(variant_value)
        except ValueError as exc:
            raise ValueError(f"The selected module does not implement {variant_value!r}.") from exc

        # Precompile the selected Numba controller and collision kernels. The
        # one-step dry run is completed before the real controller timer starts.
        self._warm_numba_for_case(
            module=module,
            variant=variant,
            variant_value=variant_value,
            modes=modes,
            base_obstacles=base_obstacles,
            runtime_blocker=runtime_blocker,
            start=start,
            goal=goal,
            condition=condition,
            trigger_progress=trigger_progress,
            blocker_from_start=blocker_from_start,
            cfg=cfg,
            seed=seed,
        )

        result = module.run_dynamic_blockage_controller(
            variant=variant,
            modes=modes,
            base_obstacles=base_obstacles,
            blocker=runtime_blocker,
            start=start,
            goal=goal,
            seed=seed,
            trigger_progress=trigger_progress,
            activation_preview_clearance=0.75,
            blocker_active_from_start=blocker_from_start,
            condition=condition,
            max_steps=130,
            goal_tolerance=0.30,
            mppi_cfg=cfg,
            record_infos=True,
            record_obstacle_history=True,
        )

        return TrialBundle(
            module=module,
            condition=condition,
            scenario_id=scenario_id,
            variant_value=variant_value,
            cfg=cfg,
            result=result,
            modes=modes,
            start=np.asarray(start, dtype=float),
            goal=np.asarray(goal, dtype=float),
            bounds_xy=bounds_xy,
            base_obstacles=list(base_obstacles),
            blocker=list(runtime_blocker),
            controller_seed=seed,
            swarm_seed=swarm_seed,
        )

    def _poll_worker(self) -> None:
        if self.closing:
            return
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "success":
                    self._on_simulation_ready(payload)
                else:
                    self._on_simulation_error(payload)
        except queue.Empty:
            pass
        if not self.closing:
            try:
                self.poll_after_id = self.root.after(100, self._poll_worker)
            except tk.TclError:
                self.poll_after_id = None

    def _on_simulation_ready(self, bundle: TrialBundle) -> None:
        if self.closing:
            return
        self.bundle = bundle
        self.frame_index = 0
        max_frame = max(0, len(bundle.result["states"]) - 1)
        self.frame_scale.configure(from_=0, to=max_frame, state="normal")
        self.frame_scale.set(0)
        self.run_button.configure(state="normal")
        self.play_button.configure(state="normal")
        self.restart_button.configure(state="normal")

        summary = self._trial_summary(bundle)
        self.status_var.set(summary)
        self._draw_frame(0)
        self.playing = True
        self.play_button.configure(text="Pause")
        self._schedule_tick()

    def _on_simulation_error(self, details: str) -> None:
        if self.closing:
            return
        self.run_button.configure(state="normal")
        self.status_var.set("Simulation failed. See the error dialog.")
        messagebox.showerror(
            "Simulation failed",
            details
            + "\n\nRun the viewer from the project root and verify that both controller "
              "module and save/best_policy.pkl are available.",
        )

    def _trial_summary(self, bundle: TrialBundle) -> str:
        states = np.asarray(bundle.result["states"])
        goal_distance = float(np.linalg.norm(states[-1, :2] - bundle.goal))
        reached = bool(bundle.result.get("reached_goal", goal_distance <= 0.30))
        collision = False
        try:
            obstacles = bundle.result.get("obstacle_history", [])
            for index, state in enumerate(states):
                active = obstacles[min(index, len(obstacles) - 1)] if obstacles else bundle.base_obstacles
                if bundle.module.min_clearance(state[None, :], active, bundle.cfg.robot_radius) < 0.0:
                    collision = True
                    break
        except Exception:
            collision = False
        outcome = "success" if reached and not collision else ("collision" if collision else "not reaching")
        runtime = float(bundle.result.get("runtime", float("nan")))
        per_step_ms = 1000.0 * runtime / max(1, len(states) - 1)
        return (
            f"Completed: {outcome}\n"
            f"Steps: {len(states) - 1}\n"
            f"Final goal distance: {goal_distance:.3f}\n"
            f"Average computation: {per_step_ms:.1f} ms/step\n"
            "Numba warm-up excluded from timing"
        )

    def toggle_play(self) -> None:
        if self.bundle is None:
            return
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._schedule_tick()
        else:
            self._cancel_tick()

    def restart_animation(self) -> None:
        if self.bundle is None:
            return
        self.frame_index = 0
        self._set_slider(0)
        self._draw_frame(0)
        self.playing = True
        self.play_button.configure(text="Pause")
        self._schedule_tick()

    def _stop_animation(self) -> None:
        self.playing = False
        self._cancel_tick()

    def close(self) -> None:
        """Stop Tk callbacks and close the window without callback-after-destroy errors."""
        if self.closing:
            return
        self.closing = True
        self.playing = False
        self._cancel_tick()

        if self.poll_after_id is not None:
            try:
                self.root.after_cancel(self.poll_after_id)
            except tk.TclError:
                pass
            self.poll_after_id = None

        try:
            plt.close(self.figure)
        except Exception:
            pass

        try:
            self.root.quit()
        except tk.TclError:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _cancel_tick(self) -> None:
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def _schedule_tick(self) -> None:
        self._cancel_tick()
        if self.closing or self.bundle is None or not self.playing:
            return
        try:
            speed = max(0.05, float(self.speed_var.get()))
        except ValueError:
            speed = 1.0
        delay_ms = max(20, int(120 / speed))
        self.after_id = self.root.after(delay_ms, self._tick)

    def _tick(self) -> None:
        self.after_id = None
        if self.closing or self.bundle is None or not self.playing:
            return
        max_frame = len(self.bundle.result["states"]) - 1
        if self.frame_index >= max_frame:
            self.playing = False
            self.play_button.configure(text="Play")
            return
        self.frame_index += 1
        self._set_slider(self.frame_index)
        self._draw_frame(self.frame_index)
        self._schedule_tick()

    def _set_slider(self, frame: int) -> None:
        self.updating_slider = True
        self.frame_scale.set(frame)
        self.updating_slider = False

    def _on_frame_slider(self, value: str) -> None:
        if self.updating_slider or self.bundle is None:
            return
        self.frame_index = int(round(float(value)))
        self._draw_frame(self.frame_index)

    def _redraw_current(self) -> None:
        if self.bundle is not None:
            self._draw_frame(self.frame_index)

    def _active_obstacles(self, frame: int) -> list[Any]:
        assert self.bundle is not None
        history = self.bundle.result.get("obstacle_history", [])
        if history:
            return list(history[min(frame, len(history) - 1)])
        if self.bundle.condition == "dynamic_wall":
            activation = self.bundle.result.get("activation_step")
            if activation is not None and frame >= int(activation):
                return self.bundle.base_obstacles + self.bundle.blocker
        return self.bundle.base_obstacles

    def _draw_frame(self, frame: int) -> None:
        if self.bundle is None:
            self._draw_empty()
            return

        bundle = self.bundle
        result = bundle.result
        states = np.asarray(result["states"], dtype=float)
        controls = np.asarray(result.get("controls", []), dtype=float)
        frame = int(np.clip(frame, 0, len(states) - 1))
        self.frame_index = frame
        state = states[frame]
        active_obstacles = self._active_obstacles(frame)

        self.ax.clear()
        xmin, xmax, ymin, ymax = normalized_bounds(bundle.bounds_xy)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.18)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self._draw_obstacles(active_obstacles)
        if self.show_collision_var.get():
            self._draw_collision_representation(active_obstacles)

        if self.show_all_modes_var.get() and bundle.variant_value != "standard_mppi":
            for mode in bundle.modes:
                mean = np.asarray(mode.mean_path)
                self.ax.plot(mean[:, 0], mean[:, 1], color="0.65", linewidth=0.9, alpha=0.35, zorder=1)

        selected_mode_index = self._selected_mode_index(frame, state)
        self._draw_prior(frame, state, selected_mode_index)

        # Executed trajectory and current MPPI output.
        self.ax.plot(
            states[: frame + 1, 0],
            states[: frame + 1, 1],
            color="#1f77b4",
            linewidth=2.4,
            label="executed path",
            zorder=7,
        )
        output = self._optimal_output(frame)
        if output is not None:
            self.ax.plot(
                output[:, 0],
                output[:, 1],
                color="#ff7f0e",
                linewidth=2.2,
                linestyle="--",
                label="MPPI output",
                zorder=8,
            )

        self.ax.scatter([bundle.start[0]], [bundle.start[1]], s=65, marker="o", color="#2ca02c", zorder=9)
        self.ax.scatter([bundle.goal[0]], [bundle.goal[1]], s=140, marker="*", color="#d62728", zorder=9)
        self._draw_robot(state)

        activation_step = result.get("activation_step")
        wall_state = "inactive"
        if bundle.condition == "static_wall":
            wall_state = "active from start"
        elif bundle.condition == "dynamic_wall" and activation_step is not None:
            wall_state = "active" if frame >= int(activation_step) else "inactive"

        info = self._frame_info(frame)
        control_text = "terminal frame"
        if frame < len(controls):
            control_text = (
                f"a={controls[frame, 0]:.2f}, "
                f"delta_dot={controls[frame, 1]:.2f}"
            )
        mode_text = "none"
        if selected_mode_index is not None and 0 <= selected_mode_index < len(bundle.modes):
            mode = bundle.modes[selected_mode_index]
            mode_text = f"{selected_mode_index}: {mode.signature}"

        extra = []
        if info:
            if info.get("num_feasible_rollouts") is not None:
                extra.append(f"feasible rollouts={info['num_feasible_rollouts']}")
            if info.get("cost_min") is not None:
                extra.append(f"min cost={float(info['cost_min']):.2f}")
        extra_text = " | ".join(extra)

        title = (
            f"{VARIANT_TO_DISPLAY.get(bundle.variant_value, bundle.variant_value)} | "
            f"{bundle.condition.replace('_', ' ')} | step {frame}/{len(states)-1}"
        )
        # self.ax.set_title(title)
        vehicle_text = "state unavailable"
        if state.size >= 7:
            vehicle_text = (
                f"vx={state[3]:.2f}, vy={state[4]:.2f}, "
                f"yaw rate={state[5]:.2f}, delta={state[6]:.2f}"
            )
        status = (
            f"control: {control_text}\n"
            f"vehicle: {vehicle_text}\n"
            f"displayed mode: {mode_text}\n"
            f"wall: {wall_state}"
        )
        if extra_text:
            status += "\n" + extra_text
        # self.ax.text(
        #     0.01,
        #     0.99,
        #     status,
        #     transform=self.ax.transAxes,
        #     va="top",
        #     ha="left",
        #     fontsize=8.5,
        #     bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.82, edgecolor="0.7"),
        #     zorder=20,
        # )

        legend_handles = [
            Line2D([0], [0], color="#1f77b4", linewidth=2.4, label="Executed path"),
            Line2D([0], [0], color="#ff7f0e", linewidth=2.2, linestyle="--", label="MPPI output"),
            Line2D([0], [0], color="#9467bd", linewidth=2.0, label="Prior"),
        ]
        self.ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.88)

        self.frame_label_var.set(f"Frame {frame} / {len(states) - 1}")
        self.canvas.draw_idle()

    def _draw_obstacles(self, obstacles: list[Any]) -> None:
        for obstacle in obstacles:
            vertices = polygon_vertices(obstacle)
            self.ax.add_patch(
                Polygon(
                    vertices,
                    closed=True,
                    facecolor="0.25",
                    edgecolor="0.08",
                    linewidth=1.2,
                    alpha=0.72,
                    zorder=3,
                )
            )

    def _draw_collision_representation(self, obstacles: list[Any]) -> None:
        assert self.bundle is not None
        try:
            circles = self.bundle.module.obstacle_bounding_circles(obstacles)
        except Exception:
            circles = []
        for center, radius in circles:
            self.ax.add_patch(
                Circle(
                    np.asarray(center)[:2],
                    float(radius) + float(self.bundle.cfg.robot_radius),
                    fill=False,
                    edgecolor="#e377c2",
                    linewidth=0.75,
                    linestyle=":",
                    alpha=0.45,
                    zorder=4,
                )
            )

    def _selected_mode_index(self, frame: int, state: np.ndarray) -> Optional[int]:
        assert self.bundle is not None
        if self.bundle.variant_value == "standard_mppi" or not self.bundle.modes:
            return None
        info = self._frame_info(frame)
        if info and info.get("selected_mode_index") is not None:
            index = int(info["selected_mode_index"])
            if 0 <= index < len(self.bundle.modes):
                return index
        distances = [
            float(np.min(np.linalg.norm(np.asarray(mode.mean_path) - state[:2], axis=1)))
            for mode in self.bundle.modes
        ]
        return int(np.argmin(distances))

    def _frame_info(self, frame: int) -> Optional[dict[str, Any]]:
        assert self.bundle is not None
        infos = self.bundle.result.get("infos", [])
        if not infos:
            return None
        return infos[min(frame, len(infos) - 1)]

    def _optimal_output(self, frame: int) -> Optional[np.ndarray]:
        info = self._frame_info(frame)
        if not info:
            return None
        output = info.get("optimal_traj")
        if output is None:
            return None
        output = np.asarray(output, dtype=float)
        if output.ndim != 2 or output.shape[0] < 2 or output.shape[1] < 2:
            return None
        return output

    def _draw_prior(self, frame: int, state: np.ndarray, mode_index: Optional[int]) -> None:
        assert self.bundle is not None
        bundle = self.bundle
        variant = bundle.variant_value

        if variant == "standard_mppi" or mode_index is None:
            self.ax.text(
                0.99,
                0.99,
                "No trajectory prior",
                transform=self.ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#9467bd",
            )
            return

        global_mode = bundle.modes[mode_index]
        local_mode = bundle.module.localize_mode_for_state(global_mode, state, bundle.cfg.horizon)
        mean = np.asarray(local_mode.mean_path, dtype=float)
        self.ax.plot(mean[:, 0], mean[:, 1], color="#9467bd", linewidth=2.2, zorder=6)

        gaussian_variants = {"gaussian_prior_mppi"}
        empirical_variants = {"control_bank_mppi"}
        corridor_variants = {"corridor_prior_mppi"}

        if variant in gaussian_variants:
            self._draw_covariance_ellipses(local_mode)
        elif variant == "frenet_corridor_mppi":
            self._draw_frenet_covariance(local_mode)
        elif variant in empirical_variants:
            self._draw_empirical_samples(global_mode, state)
        elif variant in corridor_variants:
            # The corridor proposal uses fixed control-space noise around
            # the mean nominal. No covariance-derived tube is drawn.
            pass

    def _draw_covariance_ellipses(self, local_mode: Any) -> None:
        """Draw one-standard-deviation Cartesian Gaussian proposal ellipses."""
        assert self.bundle is not None
        mean = np.asarray(local_mode.mean_path, dtype=float)
        covariances = np.asarray(local_mode.cov_blocks, dtype=float)
        variance_floor = float(getattr(self.bundle.cfg, "sigma_floor", 0.0)) ** 2
        covariance_scale = float(getattr(self.bundle.cfg, "gaussian_covariance_scale", 1.0))

        for index in range(0, len(mean), 4):
            covariance = 0.5 * (covariances[index] + covariances[index].T)
            values, vectors = np.linalg.eigh(covariance)
            values = np.maximum(values, variance_floor)
            order = np.argsort(values)[::-1]
            values = values[order]
            vectors = vectors[:, order]
            angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
            width, height = 2.0 * covariance_scale * np.sqrt(values)
            self.ax.add_patch(
                Ellipse(
                    mean[index],
                    width=float(width),
                    height=float(height),
                    angle=angle,
                    facecolor="#9467bd",
                    edgecolor="#9467bd",
                    alpha=0.10,
                    linewidth=0.8,
                    zorder=2,
                )
            )

    def _draw_frenet_covariance(self, local_mode: Any) -> None:
        """Draw the tangent/normal covariance scales used by Frenet sampling."""
        assert self.bundle is not None
        mean = np.asarray(local_mode.mean_path, dtype=float)
        covariances = np.asarray(local_mode.cov_blocks, dtype=float)
        tangent = np.gradient(mean, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        variance_floor = float(getattr(self.bundle.cfg, "sigma_floor", 0.0)) ** 2
        lateral_scale = float(getattr(self.bundle.cfg, "frenet_lateral_noise_scale", 1.0))
        longitudinal_scale = float(getattr(self.bundle.cfg, "frenet_longitudinal_noise_scale", 1.0))

        for index in range(0, len(mean), 4):
            covariance = 0.5 * (covariances[index] + covariances[index].T)
            lateral_variance = max(
                float(normal[index] @ covariance @ normal[index]), variance_floor
            )
            longitudinal_variance = max(
                float(tangent[index] @ covariance @ tangent[index]), variance_floor
            )
            angle = math.degrees(math.atan2(tangent[index, 1], tangent[index, 0]))
            self.ax.add_patch(
                Ellipse(
                    mean[index],
                    width=2.0 * longitudinal_scale * math.sqrt(longitudinal_variance),
                    height=2.0 * lateral_scale * math.sqrt(lateral_variance),
                    angle=angle,
                    facecolor="#9467bd",
                    edgecolor="#9467bd",
                    alpha=0.10,
                    linewidth=0.8,
                    zorder=2,
                )
            )

    def _draw_empirical_samples(self, global_mode: Any, state: np.ndarray) -> None:
        assert self.bundle is not None
        paths = list(global_mode.sample_paths or [])
        if not paths:
            return
        # Deterministic subset keeps the animation stable across frames.
        indices = np.linspace(0, len(paths) - 1, min(10, len(paths)), dtype=int)
        for index in indices:
            path = self.bundle.module.localize_path_for_state(
                paths[int(index)], state, self.bundle.cfg.horizon
            )
            path = np.asarray(path, dtype=float)
            self.ax.plot(
                path[:, 0],
                path[:, 1],
                color="#9467bd",
                linewidth=0.8,
                alpha=0.18,
                zorder=2,
            )

    @staticmethod
    def _transform_vehicle_points(
        points: np.ndarray,
        origin: tuple[float, float],
        angle: float,
    ) -> np.ndarray:
        """Rotate body-frame points and translate them into world coordinates."""
        c = math.cos(angle)
        s = math.sin(angle)
        rotation = np.array([[c, -s], [s, c]], dtype=float)
        return np.asarray(points, dtype=float) @ rotation.T + np.asarray(origin)

    def _draw_robot(self, state: np.ndarray) -> None:
        """Draw the dynamic Ackermann body, wheels, steering, and velocity."""
        assert self.bundle is not None
        if state.size < 7:
            raise ValueError(
                "Dynamic Ackermann viewer expects state [x,y,psi,vx,vy,r,delta]."
            )

        x, y, heading, vx, vy, _, steering = map(float, state[:7])
        cfg = self.bundle.cfg
        lf = float(getattr(cfg, "front_axle_distance", 0.275))
        lr = float(getattr(cfg, "rear_axle_distance", 0.275))
        wheelbase = max(lf + lr, 0.40)
        body_length = max(wheelbase + 0.26, 0.72)
        body_width = max(2.0 * float(cfg.robot_radius), 0.36)

        body = np.array([
            [-0.5 * body_length, -0.5 * body_width],
            [ 0.5 * body_length, -0.5 * body_width],
            [ 0.5 * body_length,  0.5 * body_width],
            [-0.5 * body_length,  0.5 * body_width],
        ])
        body_world = self._transform_vehicle_points(body, (x, y), heading)
        self.ax.add_patch(
            Polygon(
                body_world,
                closed=True,
                facecolor="#17becf",
                edgecolor="black",
                linewidth=1.1,
                alpha=0.92,
                zorder=12,
            )
        )

        wheel_length = 0.22
        wheel_width = 0.055
        half_track = 0.43 * body_width
        wheel_shape = np.array([
            [-0.5 * wheel_length, -0.5 * wheel_width],
            [ 0.5 * wheel_length, -0.5 * wheel_width],
            [ 0.5 * wheel_length,  0.5 * wheel_width],
            [-0.5 * wheel_length,  0.5 * wheel_width],
        ])

        def body_to_world(longitudinal: float, lateral: float) -> tuple[float, float]:
            c = math.cos(heading)
            s = math.sin(heading)
            return (
                x + longitudinal * c - lateral * s,
                y + longitudinal * s + lateral * c,
            )

        wheels = [
            (lf, half_track, heading + steering),
            (lf, -half_track, heading + steering),
            (-lr, half_track, heading),
            (-lr, -half_track, heading),
        ]
        for longitudinal, lateral, wheel_heading in wheels:
            center = body_to_world(longitudinal, lateral)
            vertices = self._transform_vehicle_points(
                wheel_shape, center, wheel_heading
            )
            self.ax.add_patch(
                Polygon(
                    vertices,
                    closed=True,
                    facecolor="0.10",
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=13,
                )
            )

        # Body-frame velocity transformed to the world frame.
        world_vx = vx * math.cos(heading) - vy * math.sin(heading)
        world_vy = vx * math.sin(heading) + vy * math.cos(heading)
        speed = math.hypot(world_vx, world_vy)
        if speed > 1e-3:
            velocity_scale = 0.22
            self.ax.arrow(
                x,
                y,
                velocity_scale * world_vx,
                velocity_scale * world_vy,
                width=0.012,
                head_width=0.10,
                head_length=0.11,
                length_includes_head=True,
                color="#1f77b4",
                zorder=14,
            )

        # Short heading marker keeps orientation visible at low speed.
        nose = body_to_world(0.52 * body_length, 0.0)
        self.ax.plot(
            [x, nose[0]], [y, nose[1]], color="black", linewidth=1.1, zorder=14
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive dynamic Ackermann MPPI simulation viewer."
    )
    parser.add_argument(
        "--module",
        default=os.environ.get("MPPI_MODULE"),
        help="Path to the controller module.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module_path = first_existing_path(
        args.module,
        [
            "runs_ackerman.py",
            "runs_ackerman_dynamic.py",
        ],
    )

    root = tk.Tk()
    app = InteractiveMPPIViewer(root, module_path=module_path)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()