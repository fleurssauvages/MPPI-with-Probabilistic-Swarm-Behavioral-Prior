from __future__ import annotations
import copy
import math
import queue
import threading
import traceback
from dataclasses import dataclass, replace
from typing import Any, Optional
import numpy as np
from system import ackermann, controller as controller_core, planar_quadrotor, planar_quadrotor_payload, unicycle
try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:
    raise SystemExit('Tkinter is required to run the interactive viewer.') from exc
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Ellipse, Polygon
UNICYCLE_VARIANTS = [('Planner iLQR', 'planner_ilqr'), ('Standard MPPI', 'standard_mppi'), ('Control bank', 'control_bank_mppi'), ('Corridor prior', 'corridor_prior_mppi'), ('Gaussian prior', 'gaussian_prior_mppi'), ('SPG prior', 'sensitivity_projected_gaussian_prior_mppi'), ('Mode-selecting Gaussian', 'mode_selecting_gaussian_mppi'), ('Mode-selecting corridor', 'mode_selecting_corridor_mppi')]
ACKERMAN_VARIANTS = [('Planner iLQR', 'planner_ilqr'), ('Standard MPPI', 'standard_mppi'), ('Control bank', 'control_bank_mppi'), ('Corridor prior', 'corridor_prior_mppi'), ('Gaussian prior', 'gaussian_prior_mppi'), ('SPG prior', 'sensitivity_projected_gaussian_prior_mppi')]
PLANAR_QUADROTOR_VARIANTS = [('Planner iLQR', 'planner_ilqr'), ('Standard MPPI', 'standard_mppi'), ('Control bank', 'control_bank_mppi'), ('Corridor prior', 'corridor_prior_mppi'), ('Gaussian prior', 'gaussian_prior_mppi'), ('SPG prior', 'sensitivity_projected_gaussian_prior_mppi')]
PLANAR_QUADROTOR_PAYLOAD_VARIANTS = [('Planner iLQR', 'planner_ilqr'), ('Standard MPPI', 'standard_mppi'), ('Control bank', 'control_bank_mppi'), ('Corridor prior', 'corridor_prior_mppi'), ('Gaussian prior', 'gaussian_prior_mppi'), ('SPG prior', 'sensitivity_projected_gaussian_prior_mppi')]
MODEL_OPTIONS = {
    'Ackermann': ('ackerman', ackermann, ACKERMAN_VARIANTS),
    'Unicycle': ('unicycle', unicycle, UNICYCLE_VARIANTS),
    'Planar quadrotor': ('planar_quadrotor', planar_quadrotor, PLANAR_QUADROTOR_VARIANTS),
    'Planar quadrotor + hanging package': ('planar_quadrotor_payload', planar_quadrotor_payload, PLANAR_QUADROTOR_PAYLOAD_VARIANTS),
}
VARIANTS = list(ACKERMAN_VARIANTS)
CONDITIONS = [('No wall', 'no_wall'), ('Static wall', 'static_wall'), ('Dynamic wall', 'dynamic_wall')]
SCENARIOS = [('Wall 0-1', 'wall_0_1'), ('Wall 1-2', 'wall_1_2'), ('Walls 0-1 and 1-2', 'walls_0_1__1_2')]
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
    result: Any
    modes: list[Any]
    start: np.ndarray
    goal: np.ndarray
    bounds_xy: Any
    base_obstacles: list[Any]
    blocker: list[Any]
    controller_seed: int
    swarm_seed: int

def polygon_vertices(obstacle: Any) -> np.ndarray:
    vertices = getattr(obstacle, 'vertices', obstacle)
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f'Unsupported obstacle geometry with shape {array.shape}')
    return array[:, :2]

def normalized_bounds(bounds_xy: Any) -> tuple[float, float, float, float]:
    a = np.asarray(bounds_xy[0], dtype=float)
    b = np.asarray(bounds_xy[1], dtype=float)
    if a.shape == (2,) and b.shape == (2,) and (a[0] <= b[0]) and (a[1] <= b[1]):
        return (float(a[0]), float(b[0]), float(a[1]), float(b[1]))
    return (float(bounds_xy[0][0]), float(bounds_xy[0][1]), float(bounds_xy[1][0]), float(bounds_xy[1][1]))

def make_protocol_config(module: Any, num_rollouts: int) -> Any:
    return module.MPPIConfig(num_rollouts=int(num_rollouts))

def obstacle_from_vertices(template: Any, vertices: np.ndarray) -> Any:
    """Rebuild an obstacle of the same type with edited polygon vertices."""
    edited = np.asarray(vertices, dtype=np.float64).copy()
    try:
        return type(template)(edited)
    except Exception:
        candidate = copy.deepcopy(template)
        if hasattr(candidate, 'vertices'):
            try:
                candidate.vertices = edited
                return candidate
            except Exception:
                pass
    raise TypeError(f'Cannot rebuild obstacle type {type(template).__name__} from polygon vertices.')

def obstacle_layout_key(vertices_list: list[np.ndarray]) -> tuple[tuple[float, ...], ...]:
    """Stable cache key for an interactively edited obstacle layout."""
    return tuple(tuple(np.round(np.asarray(vertices, dtype=float).ravel(), 6)) for vertices in vertices_list)


def state_layout_key(value: np.ndarray) -> tuple[float, ...]:
    """Stable cache key for an interactively edited start or goal state."""
    return tuple(np.round(np.asarray(value, dtype=float).ravel(), 6))

class InteractiveMPPIViewer:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title('Interactive MPPI viewer - Ackermann')
        self.root.minsize(1080, 700)
        self._maximize_window()
        self.modules = {key: module for key, module, _ in MODEL_OPTIONS.values()}
        self.model_key = 'ackerman'
        self.module = self.modules[self.model_key]
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
        self._states = np.empty((0, 0), dtype=float)
        self._controls = np.empty((0, 0), dtype=float)
        self._obstacle_cache: dict[int, list[Any]] = {}
        self._circle_cache: dict[tuple[int, ...], list[tuple[np.ndarray, float]]] = {}
        self._mode_mean_cache: list[np.ndarray] = []
        self._saved_trajectories: list[dict[str, Any]] = []
        self._edit_scene: Any = None
        self._editable_obstacle_vertices: list[np.ndarray] = []
        self._editable_start = np.empty(0, dtype=float)
        self._editable_goal = np.empty(0, dtype=float)
        self._drag_obstacle_index: Optional[int] = None
        self._drag_point_kind: Optional[str] = None
        self._drag_anchor: Optional[np.ndarray] = None
        self._drag_original_vertices: Optional[np.ndarray] = None
        self._drag_original_point: Optional[np.ndarray] = None
        self._reset_editable_obstacles(redraw=False)
        self._build_ui()
        self.poll_after_id = self.root.after(100, self._poll_worker)

    def _maximize_window(self) -> None:
        """Start maximized while retaining normal window decorations."""
        try:
            self.root.state('zoomed')
            return
        except tk.TclError:
            pass
        try:
            self.root.attributes('-zoomed', True)
            return
        except tk.TclError:
            pass
        width = max(1080, int(self.root.winfo_screenwidth()))
        height = max(700, int(self.root.winfo_screenheight()))
        self.root.geometry(f'{width}x{height}+0+0')

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
        ttk.Label(controls, text='Experiment', font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=0, sticky='w', pady=(0, 10))
        self.model_var = tk.StringVar(value='Ackermann')
        self.variant_var = tk.StringVar(value='SPG prior')
        self.condition_var = tk.StringVar(value='No wall')
        self.scenario_var = tk.StringVar(value='Walls 0-1 and 1-2')
        self.seed_var = tk.StringVar(value='1')
        self.swarm_seed_var = tk.StringVar(value='5')
        self.rollouts_var = tk.StringVar(value='512')
        self.speed_var = tk.StringVar(value='4.0')
        self.show_collision_var = tk.BooleanVar(value=False)
        self.show_all_modes_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value='Ready')
        self.frame_label_var = tk.StringVar(value='Frame 0 / 0')
        row = 1
        row = self._add_combo(controls, row, 'Vehicle model', self.model_var, list(MODEL_OPTIONS.keys()))
        self.model_var._combo_widget.bind('<<ComboboxSelected>>', self._on_model_changed)
        row = self._add_combo(controls, row, 'Controller variant', self.variant_var, [label for label, _ in VARIANTS])
        row = self._add_combo(controls, row, 'Condition', self.condition_var, [label for label, _ in CONDITIONS])
        row = self._add_combo(controls, row, 'Wall scenario', self.scenario_var, [label for label, _ in SCENARIOS])
        row = self._add_entry(controls, row, 'Controller seed', self.seed_var)
        row = self._add_entry(controls, row, 'Swarm seed', self.swarm_seed_var)
        row = self._add_entry(controls, row, 'Rollouts per step', self.rollouts_var)
        row = self._add_combo(controls, row, 'Playback speed', self.speed_var, ['0.5', '1.0', '2.0', '4.0', '8.0'])
        ttk.Checkbutton(controls, text='Show collision representation', variable=self.show_collision_var, command=self._redraw_current).grid(row=row, column=0, sticky='w', pady=(8, 2))
        row += 1
        ttk.Checkbutton(controls, text='Show all prior means', variable=self.show_all_modes_var, command=self._redraw_current).grid(row=row, column=0, sticky='w', pady=(2, 2))
        row += 1
        buttons = ttk.Frame(controls)
        buttons.grid(row=row, column=0, sticky='ew', pady=(14, 6))
        buttons.columnconfigure((0, 1), weight=1)
        self.run_button = ttk.Button(buttons, text='Run selected case', command=self.run_selected)
        self.run_button.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        self.play_button = ttk.Button(buttons, text='Play', command=self.toggle_play, state='disabled')
        self.play_button.grid(row=1, column=0, sticky='ew', padx=(0, 3))
        self.restart_button = ttk.Button(buttons, text='Restart', command=self.restart_animation, state='disabled')
        self.restart_button.grid(row=1, column=1, sticky='ew', padx=(3, 0))
        self.save_trajectory_button = ttk.Button(buttons, text='Save trajectory', command=self.save_trajectory, state='disabled')
        self.save_trajectory_button.grid(row=2, column=0, sticky='ew', padx=(0, 3), pady=(6, 0))
        self.reset_trajectories_button = ttk.Button(buttons, text='Reset trajectories', command=self.reset_trajectories)
        self.reset_trajectories_button.grid(row=2, column=1, sticky='ew', padx=(3, 0), pady=(6, 0))
        self.edit_obstacles_button = ttk.Button(buttons, text='Edit scene', command=self.edit_obstacles)
        self.edit_obstacles_button.grid(row=3, column=0, sticky='ew', padx=(0, 3), pady=(6, 0))
        self.reset_obstacles_button = ttk.Button(buttons, text='Reset scene', command=self.reset_obstacles)
        self.reset_obstacles_button.grid(row=3, column=1, sticky='ew', padx=(3, 0), pady=(6, 0))
        row += 1
        ttk.Separator(controls).grid(row=row, column=0, sticky='ew', pady=10)
        row += 1
        ttk.Label(controls, text='Animation', font=('TkDefaultFont', 11, 'bold')).grid(row=row, column=0, sticky='w')
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
        self.status_label = ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=292,
            justify='left',
            anchor='nw',
        )
        self.status_label.grid(row=row, column=0, sticky='nsew', pady=(4, 0))
        controls.rowconfigure(row, weight=1)
        self.condition_var.trace_add('write', lambda *_: self._update_condition_controls())
        self._update_condition_controls()
        self.figure = Figure(figsize=(10, 8))
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.93)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self.canvas.mpl_connect('button_press_event', self._on_plot_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_plot_motion)
        self.canvas.mpl_connect('button_release_event', self._on_plot_release)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky='ew')
        self._draw_empty()

    @staticmethod
    def _add_combo(parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, values: list[str]) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=(5, 2))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state='readonly', width=31)
        combo.grid(row=row + 1, column=0, sticky='ew')
        variable._combo_widget = combo
        return row + 2

    @staticmethod
    def _add_entry(parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=(5, 2))
        entry = ttk.Entry(parent, textvariable=variable, width=33)
        entry.grid(row=row + 1, column=0, sticky='ew')
        return row + 2

    def _on_model_changed(self, _event: Any=None) -> None:
        selected_label = self.model_var.get()
        if selected_label not in MODEL_OPTIONS:
            return
        requested, requested_module, requested_variants = MODEL_OPTIONS[selected_label]
        if requested == self.model_key:
            return
        if self.worker is not None and self.worker.is_alive():
            current_label = next(label for label, (key, _, _) in MODEL_OPTIONS.items() if key == self.model_key)
            self.model_var.set(current_label)
            messagebox.showinfo('Simulation running', 'Wait for the current simulation before changing the model.')
            return
        self._stop_animation()
        self.model_key = requested
        self.module = requested_module
        self._reset_editable_obstacles(redraw=False)
        self.bundle = None
        self.frame_index = 0
        self.mode_cache.clear()
        self.numba_warm_cache.clear()
        self._states = np.empty((0, 0), dtype=float)
        self._controls = np.empty((0, 0), dtype=float)
        self._obstacle_cache.clear()
        self._circle_cache.clear()
        self._mode_mean_cache.clear()
        global VARIANTS, DISPLAY_TO_VARIANT, VARIANT_TO_DISPLAY
        VARIANTS = list(requested_variants)
        DISPLAY_TO_VARIANT = dict(VARIANTS)
        VARIANT_TO_DISPLAY = {value: label for label, value in VARIANTS}
        labels = [label for label, _ in VARIANTS]
        combo = getattr(self.variant_var, '_combo_widget', None)
        if combo is not None:
            combo.configure(values=labels)
        if self.variant_var.get() not in labels:
            self.variant_var.set('Gaussian prior')
        self.play_button.configure(state='disabled', text='Play')
        self.restart_button.configure(state='disabled')
        self.save_trajectory_button.configure(state='disabled')
        self.frame_scale.configure(from_=0, to=0, state='disabled')
        self.frame_label_var.set('Frame 0 / 0')
        self.status_var.set(f'Loaded {self.model_var.get()} model')
        self.root.title(f'Interactive MPPI viewer - {self.model_var.get()}')
        self._draw_empty()

    def _update_condition_controls(self) -> None:
        combo = getattr(self.scenario_var, '_combo_widget', None)
        if combo is not None:
            combo.configure(state='disabled' if self.condition_var.get() == 'No wall' else 'readonly')

    def _draw_empty(self) -> None:
        self.ax.clear()
        if self._edit_scene is None:
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, 10)
            self.ax.set_title('Select a controller and run a case')
        else:
            xmin, xmax, ymin, ymax = normalized_bounds(self._edit_scene.bounds_xy)
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self._draw_obstacles(self._editable_obstacle_vertices)
            start = self._editable_start
            goal = self._editable_goal
            self.ax.scatter([start[0]], [start[1]], s=65, marker='o', color='#2ca02c', zorder=9)
            self.ax.scatter([goal[0]], [goal[1]], s=140, marker='*', color='#d62728', zorder=9)
            self.ax.set_title('Drag obstacles, start, or goal, then run the selected case')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, alpha=0.2)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.canvas.draw_idle()

    def _reset_editable_obstacles(self, *, redraw: bool=True) -> None:
        self._edit_scene = self.module.build_default_scene()
        self._editable_obstacle_vertices = [polygon_vertices(obstacle).copy() for obstacle in self._edit_scene.obstacles]
        self._editable_start = np.asarray(self._edit_scene.start, dtype=float).copy()
        self._editable_goal = np.asarray(self._edit_scene.goal, dtype=float).copy()
        self._drag_obstacle_index = None
        self._drag_point_kind = None
        self._drag_anchor = None
        self._drag_original_vertices = None
        self._drag_original_point = None
        self.mode_cache.clear()
        if redraw and hasattr(self, 'ax'):
            self._draw_empty()

    def edit_obstacles(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo('Simulation running', 'Wait for the current simulation before editing the scene.')
            return
        self._stop_animation()
        self.bundle = None
        self.frame_index = 0
        self._states = np.empty((0, 0), dtype=float)
        self._controls = np.empty((0, 0), dtype=float)
        self._obstacle_cache.clear()
        self._circle_cache.clear()
        self._mode_mean_cache.clear()
        self.play_button.configure(state='disabled', text='Play')
        self.restart_button.configure(state='disabled')
        self.save_trajectory_button.configure(state='disabled')
        self.frame_scale.configure(from_=0, to=0, state='disabled')
        self.frame_label_var.set('Frame 0 / 0')
        self.status_var.set('Scene editing enabled. Drag an obstacle, the start circle, or the goal star, then press Run selected case.')
        self._draw_empty()

    def reset_obstacles(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo('Simulation running', 'Wait for the current simulation before resetting the scene.')
            return
        self.edit_obstacles()
        self._reset_editable_obstacles(redraw=True)
        self.status_var.set('Scene reset. Drag an obstacle, start, or goal, or run the selected case.')

    def _obstacle_editing_enabled(self) -> bool:
        return self.bundle is None and not (self.worker is not None and self.worker.is_alive())

    def _plot_point_hit(self, event: Any, point: np.ndarray, radius_px: float) -> bool:
        """Return True when a mouse event lands near a data point in screen space."""
        if event.x is None or event.y is None or point.size < 2:
            return False
        display = self.ax.transData.transform(np.asarray(point[:2], dtype=float))
        return math.hypot(float(event.x) - float(display[0]), float(event.y) - float(display[1])) <= radius_px

    def _on_plot_press(self, event: Any) -> None:
        if not self._obstacle_editing_enabled() or event.inaxes is not self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        point = np.asarray([float(event.xdata), float(event.ydata)], dtype=float)

        # Give start/goal markers priority over polygons if they overlap.
        if self._plot_point_hit(event, self._editable_start, 14.0):
            self._drag_point_kind = 'start'
            self._drag_anchor = point
            self._drag_original_point = self._editable_start.copy()
            self.status_var.set('Dragging start. Release to place it.')
            return
        if self._plot_point_hit(event, self._editable_goal, 18.0):
            self._drag_point_kind = 'goal'
            self._drag_anchor = point
            self._drag_original_point = self._editable_goal.copy()
            self.status_var.set('Dragging goal. Release to place it.')
            return

        point_tuple = (float(point[0]), float(point[1]))
        for index in range(len(self._editable_obstacle_vertices) - 1, -1, -1):
            vertices = self._editable_obstacle_vertices[index]
            if MplPath(vertices, closed=True).contains_point(point_tuple, radius=0.03):
                self._drag_obstacle_index = index
                self._drag_anchor = point
                self._drag_original_vertices = vertices.copy()
                self.status_var.set(f'Dragging obstacle {index}. Release to place it.')
                return

    def _on_plot_motion(self, event: Any) -> None:
        if self._drag_anchor is None:
            return
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return

        current = np.asarray([float(event.xdata), float(event.ydata)], dtype=float)
        if self._edit_scene is not None:
            xmin, xmax, ymin, ymax = normalized_bounds(self._edit_scene.bounds_xy)
        else:
            xmin, xmax, ymin, ymax = (-np.inf, np.inf, -np.inf, np.inf)

        if self._drag_point_kind is not None and self._drag_original_point is not None:
            edited = self._drag_original_point.copy()
            edited[0] = float(np.clip(current[0], xmin, xmax))
            edited[1] = float(np.clip(current[1], ymin, ymax))
            if self._drag_point_kind == 'start':
                self._editable_start = edited
            else:
                self._editable_goal = edited
            self._draw_empty()
            return

        if self._drag_obstacle_index is None or self._drag_original_vertices is None:
            return
        delta = current - self._drag_anchor
        original = self._drag_original_vertices
        delta[0] = float(np.clip(delta[0], xmin - np.min(original[:, 0]), xmax - np.max(original[:, 0])))
        delta[1] = float(np.clip(delta[1], ymin - np.min(original[:, 1]), ymax - np.max(original[:, 1])))
        self._editable_obstacle_vertices[self._drag_obstacle_index] = original + delta
        self._draw_empty()

    def _on_plot_release(self, event: Any) -> None:
        moved_point = self._drag_point_kind
        moved_obstacle = self._drag_obstacle_index
        if moved_point is None and moved_obstacle is None:
            return

        self._drag_obstacle_index = None
        self._drag_point_kind = None
        self._drag_anchor = None
        self._drag_original_vertices = None
        self._drag_original_point = None
        self.mode_cache.clear()
        self._obstacle_cache.clear()
        self._circle_cache.clear()
        if moved_point is not None:
            self.status_var.set(f'{moved_point.capitalize()} moved. Press Run selected case to recompute the prior and controller trajectory.')
        else:
            self.status_var.set(f'Obstacle {moved_obstacle} moved. Press Run selected case to recompute the prior and controller trajectory.')
        self._draw_empty()

    def _get_module(self) -> Any:
        module = self.module
        available = {member.value for member in module.ControllerVariant}
        unsupported = [value for _, value in VARIANTS if value not in available]
        if unsupported:
            raise RuntimeError('Viewer/controller variant mismatch. Unsupported values: ' + ', '.join(unsupported))
        return module

    def _warm_numba_for_case(self, *, module: Any, variant: Any, variant_value: str, modes: list[Any], base_obstacles: list[Any], runtime_blocker: list[Any], scene: Any, trigger_progress: Optional[float], blocker_from_start: bool, cfg: Any, seed: int) -> None:
        if variant_value in self.numba_warm_cache:
            return
        if not module.NUMBA_AVAILABLE:
            self.numba_warm_cache.add(variant_value)
            return
        warm_cfg = copy.deepcopy(cfg)
        warm_cfg.horizon = min(warm_cfg.horizon, 6)
        warm_cfg.num_rollouts = 64
        if hasattr(warm_cfg, 'mode_select_rollouts_per_mode'):
            warm_cfg.mode_select_rollouts_per_mode = 8
        module.run_controller(variant, modes, base_obstacles, runtime_blocker, scene, seed=seed, trigger_progress=trigger_progress, blocker_active_from_start=blocker_from_start, max_steps=1, cfg=warm_cfg, record=False)
        self.numba_warm_cache.add(variant_value)

    def run_selected(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo('Simulation running', 'Wait for the current simulation to finish.')
            return
        try:
            seed = int(self.seed_var.get())
            swarm_seed = int(self.swarm_seed_var.get())
            rollouts = int(self.rollouts_var.get())
            if rollouts < 32:
                raise ValueError('Rollouts per step must be at least 32.')
        except ValueError as exc:
            messagebox.showerror('Invalid settings', str(exc))
            return
        condition = DISPLAY_TO_CONDITION[self.condition_var.get()]
        scenario_id = DISPLAY_TO_SCENARIO[self.scenario_var.get()]
        variant_value = DISPLAY_TO_VARIANT[self.variant_var.get()]
        self._stop_animation()
        self.bundle = None
        self.run_button.configure(state='disabled')
        self.play_button.configure(state='disabled')
        self.restart_button.configure(state='disabled')
        self.save_trajectory_button.configure(state='disabled')
        self.edit_obstacles_button.configure(state='disabled')
        self.reset_obstacles_button.configure(state='disabled')
        self.frame_scale.configure(state='disabled')
        self.status_var.set('Loading controller and generating the trajectory prior. The first construction can take several minutes due to the Numba routines.')
        settings = {
            'condition': condition,
            'scenario_id': scenario_id,
            'variant_value': variant_value,
            'seed': seed,
            'swarm_seed': swarm_seed,
            'rollouts': rollouts,
            'obstacle_vertices': [vertices.copy() for vertices in self._editable_obstacle_vertices],
            'start': self._editable_start.copy(),
            'goal': self._editable_goal.copy(),
        }
        self.worker = threading.Thread(target=self._simulation_worker, args=(settings,), daemon=True)
        self.worker.start()

    def _simulation_worker(self, settings: dict[str, Any]) -> None:
        try:
            bundle = self._prepare_and_run_trial(**settings)
            self.worker_queue.put(('success', bundle))
        except Exception:
            self.worker_queue.put(('error', traceback.format_exc()))

    def _prepare_and_run_trial(self, *, condition: str, scenario_id: str, variant_value: str, seed: int, swarm_seed: int, rollouts: int, obstacle_vertices: list[np.ndarray], start: np.ndarray, goal: np.ndarray) -> TrialBundle:
        module = self._get_module()
        cfg = make_protocol_config(module, rollouts)
        scene = module.build_default_scene()
        templates = list(scene.obstacles)
        if len(obstacle_vertices) != len(templates):
            raise ValueError('Edited obstacle layout does not match the current scene.')
        default_obstacles = [obstacle_from_vertices(template, vertices) for template, vertices in zip(templates, obstacle_vertices)]
        edited_start = np.asarray(start, dtype=float).copy()
        edited_goal = np.asarray(goal, dtype=float).copy()
        if edited_start.shape != np.asarray(scene.start).shape:
            raise ValueError('Edited start state shape does not match the current scene.')
        if edited_goal.shape != np.asarray(scene.goal).shape:
            raise ValueError('Edited goal state shape does not match the current scene.')
        scene = replace(scene, obstacles=tuple(default_obstacles), start=edited_start, goal=edited_goal)
        layout_key = obstacle_layout_key(obstacle_vertices)
        scenarios = {scenario.scenario_id: scenario for scenario in module.default_dynamic_wall_scenarios()}
        if scenario_id not in scenarios:
            available = ', '.join(sorted(scenarios))
            raise KeyError(f'Scenario {scenario_id!r} is unavailable. Available: {available}')
        scenario = scenarios[scenario_id]
        fixed_centers = tuple((module.obstacle_center(obs).copy() for obs in scene.obstacles))
        blocker = module.make_wall_blockers_between_centers(centers=fixed_centers, pairs=scenario.wall_pairs, width=scenario.wall_width, extension=scenario.wall_extension)
        base_prior_key = ('base', swarm_seed, layout_key, state_layout_key(scene.start), state_layout_key(scene.goal))

        if condition == 'static_wall':
            # A static wall is known before planning, so include it in both the graph
            # and swarm planner instead of generating a base prior and filtering it later.
            planner_obstacles = default_obstacles + blocker
            prior_key = (
                'static_wall', scenario_id, swarm_seed, layout_key,
                state_layout_key(scene.start), state_layout_key(scene.goal),
            )
            if prior_key not in self.mode_cache:
                self.mode_cache[prior_key] = module.build_homotopy_modes(
                    scene, planner_obstacles, swarm_seed
                )
            modes = list(self.mode_cache[prior_key])
            base_obstacles = planner_obstacles
            runtime_blocker: list[Any] = []
            blocker_from_start = True
            trigger_progress = None
        else:
            # Dynamic walls are intentionally absent from the original prior so that
            # the controller must react after activation.  No-wall reuses the same prior.
            if base_prior_key not in self.mode_cache:
                self.mode_cache[base_prior_key] = module.build_homotopy_modes(
                    scene, default_obstacles, swarm_seed
                )
            modes = list(self.mode_cache[base_prior_key])
            base_obstacles = default_obstacles
            runtime_blocker = [] if condition == 'no_wall' else blocker
            blocker_from_start = False
            trigger_progress = None if condition == 'no_wall' else scenario.trigger_progress
        try:
            variant = module.ControllerVariant(variant_value)
        except ValueError as exc:
            raise ValueError(f'The selected module does not implement {variant_value!r}.') from exc
        self._warm_numba_for_case(module=module, variant=variant, variant_value=variant_value, modes=modes, base_obstacles=base_obstacles, runtime_blocker=runtime_blocker, scene=scene, trigger_progress=trigger_progress, blocker_from_start=blocker_from_start, cfg=cfg, seed=seed)
        result = module.run_controller(variant, modes, base_obstacles, runtime_blocker, scene, seed=seed, trigger_progress=trigger_progress, blocker_active_from_start=blocker_from_start, max_steps=200, cfg=cfg, record=True)
        return TrialBundle(module=module, condition=condition, scenario_id=scenario_id, variant_value=variant_value, cfg=cfg, result=result, modes=modes, start=scene.start.copy(), goal=scene.goal.copy(), bounds_xy=scene.bounds_xy, base_obstacles=list(base_obstacles), blocker=list(runtime_blocker), controller_seed=seed, swarm_seed=swarm_seed)

    def _poll_worker(self) -> None:
        if self.closing:
            return
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == 'success':
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
        self._states = np.asarray(bundle.result.states, dtype=float)
        self._controls = np.asarray(bundle.result.controls, dtype=float)
        self._obstacle_cache.clear()
        self._circle_cache.clear()
        self._mode_mean_cache = [np.asarray(mode.mean_path, dtype=float) for mode in bundle.modes]
        self.frame_index = 0
        max_frame = max(0, len(bundle.result.states) - 1)
        self.frame_scale.configure(from_=0, to=max_frame, state='normal')
        self.frame_scale.set(0)
        self.run_button.configure(state='normal')
        self.play_button.configure(state='normal')
        self.restart_button.configure(state='normal')
        self.save_trajectory_button.configure(state='normal')
        self.edit_obstacles_button.configure(state='normal')
        self.reset_obstacles_button.configure(state='normal')
        summary = self._trial_summary(bundle)
        self.status_var.set(summary)
        self._draw_frame(0)
        self.playing = True
        self.play_button.configure(text='Pause')
        self._schedule_tick()

    def _on_simulation_error(self, details: str) -> None:
        if self.closing:
            return
        self.run_button.configure(state='normal')
        self.save_trajectory_button.configure(state='normal' if self.bundle is not None else 'disabled')
        self.edit_obstacles_button.configure(state='normal')
        self.reset_obstacles_button.configure(state='normal')
        self.status_var.set('Simulation failed. See the error dialog.')
        messagebox.showerror('Simulation failed', details + '\n\nRun the viewer from the project root and verify that both controller module and save/best_policy.pkl are available.')

    def _trial_summary(self, bundle: TrialBundle) -> str:
        states = np.asarray(bundle.result.states)
        goal_distance = float(np.linalg.norm(states[-1, :2] - bundle.goal))
        reached = bool(bundle.result.reached_goal)
        collision = False
        try:
            obstacles = bundle.result.obstacle_history
            for index, state in enumerate(states):
                active = obstacles[min(index, len(obstacles) - 1)] if obstacles else bundle.base_obstacles
                if bundle.module.minimum_clearance(state[None, :], active, bundle.cfg) < 0.0:
                    collision = True
                    break
        except Exception:
            collision = False
        outcome = 'success' if reached and (not collision) else 'collision' if collision else 'not reaching'
        runtime = float(bundle.result.runtime)
        per_step_ms = 1000.0 * runtime / max(1, len(states) - 1)
        return f'Completed: {outcome}\nSteps: {len(states) - 1}\n'

    def toggle_play(self) -> None:
        if self.bundle is None:
            return
        self.playing = not self.playing
        self.play_button.configure(text='Pause' if self.playing else 'Play')
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
        self.play_button.configure(text='Pause')
        self._schedule_tick()

    def save_trajectory(self) -> None:
        """Keep the current executed trajectory as a persistent comparison overlay."""
        if self.bundle is None or self._states.ndim != 2 or len(self._states) < 2:
            messagebox.showinfo('No trajectory', 'Run a case before saving its trajectory.')
            return
        trajectory = np.asarray(self._states[:, :2], dtype=float).copy()
        label = VARIANT_TO_DISPLAY.get(self.bundle.variant_value, self.bundle.variant_value)
        scenario = self.bundle.condition.replace('_', ' ')
        self._saved_trajectories.append({
            'xy': trajectory,
            'label': f'{label} | {scenario}',
            'model_key': self.model_key,
        })
        self.status_var.set(f'Saved trajectory #{len(self._saved_trajectories)} for comparison.')
        self._redraw_current()

    def reset_trajectories(self) -> None:
        """Erase every saved comparison trajectory."""
        count = len(self._saved_trajectories)
        self._saved_trajectories.clear()
        self.status_var.set(f'Cleared {count} saved trajector' + ('y.' if count == 1 else 'ies.'))
        if self.bundle is not None:
            self._draw_frame(self.frame_index)
        else:
            self._draw_empty()

    def _draw_saved_trajectories(self) -> None:
        """Draw saved trajectories below the active executed path."""
        if not self._saved_trajectories:
            return
        fallback_colors = ('#d62728', '#2ca02c', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
        for index, item in enumerate(self._saved_trajectories):
            if item.get('model_key') != self.model_key:
                continue
            xy = np.asarray(item.get('xy'), dtype=float)
            if xy.ndim != 2 or xy.shape[0] < 2 or xy.shape[1] < 2:
                continue
            color = fallback_colors[index % len(fallback_colors)]
            self.ax.plot(
                xy[:, 0], xy[:, 1],
                color=color, linewidth=2.0, linestyle='-', alpha=0.78,
                label=f"saved {index + 1}: {item.get('label', 'trajectory')}", zorder=5,
            )

    def _stop_animation(self) -> None:
        self.playing = False
        self._cancel_tick()

    def close(self) -> None:
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
            self.figure.clear()
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
        if self.closing or self.bundle is None or (not self.playing):
            return
        try:
            speed = max(0.05, float(self.speed_var.get()))
        except ValueError:
            speed = 1.0
        delay_ms = max(20, int(120 / speed))
        self.after_id = self.root.after(delay_ms, self._tick)

    def _tick(self) -> None:
        self.after_id = None
        if self.closing or self.bundle is None or (not self.playing):
            return
        max_frame = len(self._states) - 1
        if self.frame_index >= max_frame:
            self.playing = False
            self.play_button.configure(text='Play')
            return
        try:
            speed = max(0.05, float(self.speed_var.get()))
        except ValueError:
            speed = 1.0
        self.frame_index = min(max_frame, self.frame_index + max(1, int(speed / 4.0)))
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
        cached = self._obstacle_cache.get(frame)
        if cached is not None:
            return cached
        history = self.bundle.result.obstacle_history
        if history:
            obstacles = list(history[min(frame, len(history) - 1)])
        elif self.bundle.condition == 'dynamic_wall' and ((activation := self.bundle.result.activation_step) is not None and frame >= int(activation)):
            obstacles = self.bundle.base_obstacles + self.bundle.blocker
        else:
            obstacles = self.bundle.base_obstacles
        self._obstacle_cache[frame] = obstacles
        return obstacles

    def _draw_frame(self, frame: int) -> None:
        if self.bundle is None:
            self._draw_empty()
            return
        bundle = self.bundle
        result = bundle.result
        states = self._states
        controls = self._controls
        frame = int(np.clip(frame, 0, len(states) - 1))
        self.frame_index = frame
        state = states[frame]
        active_obstacles = self._active_obstacles(frame)
        self.ax.clear()
        xmin, xmax, ymin, ymax = normalized_bounds(bundle.bounds_xy)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, alpha=0.18)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self._draw_obstacles(active_obstacles)
        if self.show_collision_var.get():
            self._draw_collision_representation(active_obstacles)
        selected_mode_index = self._selected_rollout_mode_index(frame)
        if self.show_all_modes_var.get():
            self._draw_all_prior_means(state)
        self._draw_prior(state, selected_mode_index)
        self._draw_saved_trajectories()
        self.ax.plot(states[:frame + 1, 0], states[:frame + 1, 1], color='#1f77b4', linewidth=2.4, label='Executed path', zorder=7)
        nominal_ilqr = self._nominal_ilqr_output(frame)
        if nominal_ilqr is not None:
            self.ax.plot(
                nominal_ilqr[:, 0],
                nominal_ilqr[:, 1],
                color='#0066cc',
                linewidth=2.0,
                linestyle='-.',
                alpha=0.95,
                label='iLQR nominal',
                zorder=7,
            )
        output = self._optimal_output(frame)
        if output is not None and bundle.variant_value != 'planner_ilqr':
            self.ax.plot(output[:, 0], output[:, 1], color='#ff7f0e', linewidth=2.2, linestyle='--', label='MPPI output', zorder=8)
        self.ax.scatter([bundle.start[0]], [bundle.start[1]], s=65, marker='o', color='#2ca02c', zorder=9)
        self.ax.scatter([bundle.goal[0]], [bundle.goal[1]], s=140, marker='*', color='#d62728', zorder=9)
        self._draw_robot(state)
        activation_step = result.activation_step
        wall_state = 'inactive'
        if bundle.condition == 'static_wall':
            wall_state = 'active from start'
        elif bundle.condition == 'dynamic_wall' and activation_step is not None:
            wall_state = 'active' if frame >= int(activation_step) else 'inactive'
        info = self._frame_info(frame)
        control_text = 'terminal frame'
        if frame < len(controls):
            if self.model_key == 'ackerman':
                control_text = f'a={controls[frame, 0]:.2f}, delta_dot={controls[frame, 1]:.2f}'
            elif self.model_key in {'planar_quadrotor', 'planar_quadrotor_payload'}:
                control_text = f'omega_r_dot={controls[frame, 0]:.2f}, omega_l_dot={controls[frame, 1]:.2f}'
            else:
                control_text = f'v={controls[frame, 0]:.2f}, omega={controls[frame, 1]:.2f}'
        mode_text = 'none'
        if selected_mode_index is not None and 0 <= selected_mode_index < len(bundle.modes):
            mode = bundle.modes[selected_mode_index]
            mode_text = f'{selected_mode_index}: {mode.signature}'
        extra = []
        if info:
            if info.get('num_feasible_rollouts') is not None:
                extra.append(f"feasible rollouts={info['num_feasible_rollouts']}")
            if info.get('cost_min') is not None:
                extra.append(f"min cost={float(info['cost_min']):.2f}")
            if info.get('retained_mode_clearances') is not None:
                values = [float(value) for value in info.get('retained_mode_clearances', [])]
                extra.append('retained clearances=[' + ', '.join(f'{value:.3f}' for value in values) + ']')
        extra_text = ' | '.join(extra)
        title = f"{VARIANT_TO_DISPLAY.get(bundle.variant_value, bundle.variant_value)} | {bundle.condition.replace('_', ' ')} | step {frame}/{len(states) - 1}"
        status = f'control: {control_text}\nselected rollout mode: {mode_text}\nwall: {wall_state}'
        if extra_text:
            status += '\n' + extra_text
        legend_handles = [Line2D([0], [0], color='#1f77b4', linewidth=2.4, label='Executed path'), Line2D([0], [0], color='#0066cc', linewidth=2.0, linestyle='-.', label='iLQR nominal'), Line2D([0], [0], color='#ff7f0e', linewidth=2.2, linestyle='--', label='MPPI output'), Line2D([0], [0], color='#9467bd', linewidth=2.0, label='Selected rollout prior')]
        saved_handles = []
        fallback_colors = ('#d62728', '#2ca02c', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
        for index, item in enumerate(self._saved_trajectories):
            if item.get('model_key') == self.model_key:
                saved_handles.append(Line2D([0], [0], color=fallback_colors[index % len(fallback_colors)], linewidth=2.0, label=f"saved {index + 1}: {item.get('label', 'trajectory')}"))
        self.ax.legend(handles=legend_handles + saved_handles, loc='best', fontsize=8)
        self.frame_label_var.set(f'Frame {frame} / {len(states) - 1}')
        self.canvas.draw_idle()

    def _draw_obstacles(self, obstacles: list[Any]) -> None:
        for obstacle in obstacles:
            vertices = polygon_vertices(obstacle)
            self.ax.add_patch(Polygon(vertices, closed=True, facecolor='0.25', edgecolor='0.08', linewidth=1.2, alpha=0.72, zorder=3))

    def _draw_collision_representation(self, obstacles: list[Any]) -> None:
        assert self.bundle is not None
        key = tuple((id(obstacle) for obstacle in obstacles))
        circles = self._circle_cache.get(key)
        if circles is None:
            try:
                circles = self.bundle.module.obstacle_bounding_circles(obstacles)
            except Exception:
                circles = []
            self._circle_cache[key] = circles
        cfg = self.bundle.cfg
        collision_radius = float(getattr(cfg, 'total_drone_radius', cfg.robot_radius))
        payload_radius = (
            float(cfg.payload_radius)
            if self.model_key == 'planar_quadrotor_payload'
            else None
        )
        for center, radius in circles:
            center_xy = np.asarray(center)[:2]
            self.ax.add_patch(Circle(center_xy, float(radius) + collision_radius, fill=False, edgecolor='#e377c2', linewidth=0.75, linestyle=':', alpha=0.45, zorder=4))
            if payload_radius is not None:
                self.ax.add_patch(Circle(center_xy, float(radius) + payload_radius, fill=False, edgecolor='#ff7f0e', linewidth=0.75, linestyle=':', alpha=0.35, zorder=4))

    def _selected_rollout_mode_index(self, frame: int) -> Optional[int]:
        """Return the global mode that produced the displayed best rollout."""
        assert self.bundle is not None
        if self.bundle.variant_value.startswith('standard_mppi') or not self.bundle.modes:
            return None
        info = self._frame_info(frame)
        if not info:
            return None
        value = info.get('selected_rollout_mode_index')
        if value is None:
            # Compatibility with mode-selecting controller outputs.
            value = info.get('selected_mode_index')
        if value is None:
            return None
        index = int(value)
        return index if 0 <= index < len(self.bundle.modes) else None

    def _frame_info(self, frame: int) -> Optional[dict[str, Any]]:
        assert self.bundle is not None
        infos = self.bundle.result.infos
        if not infos:
            return None
        return infos[min(frame, len(infos) - 1)]

    def _trim_output_at_goal_tolerance(self, output: np.ndarray) -> np.ndarray:
        """Stop a displayed prediction at its first point inside goal_tolerance."""
        assert self.bundle is not None
        trajectory = np.asarray(output, dtype=float)
        distances = np.linalg.norm(trajectory[:, :2] - self.bundle.goal[None, :2], axis=1)
        inside = np.flatnonzero(distances <= float(self.bundle.cfg.goal_tolerance))
        if inside.size:
            return trajectory[: int(inside[0]) + 1]
        return trajectory

    def _optimal_output(self, frame: int) -> Optional[np.ndarray]:
        info = self._frame_info(frame)
        if not info:
            return None
        output = info.get('optimal_traj')
        if output is None:
            return None
        output = np.asarray(output, dtype=float)
        if output.ndim != 2 or output.shape[0] < 2 or output.shape[1] < 2:
            return None
        return self._trim_output_at_goal_tolerance(output)

    def _nominal_ilqr_output(self, frame: int) -> Optional[np.ndarray]:
        info = self._frame_info(frame)
        if not info:
            return None
        output = info.get('nominal_ilqr_traj')
        if output is None:
            return None
        output = np.asarray(output, dtype=float)
        if output.ndim != 2 or output.shape[0] < 2 or output.shape[1] < 2:
            return None
        return self._trim_output_at_goal_tolerance(output)

    def _draw_all_prior_means(self, state: np.ndarray) -> None:
        """Draw every localized homotopy mean in gray for prior-selection debugging."""
        assert self.bundle is not None
        bundle = self.bundle
        info = self._frame_info(self.frame_index) or {}
        retained = {int(index) for index in info.get('retained_mode_indices', [])}

        for mode_index, global_mode in enumerate(bundle.modes):
            try:
                local_mode = bundle.module.localize_mode_for_state(
                    global_mode,
                    state,
                    bundle.cfg.horizon,
                    step_distance=controller_core.prior_preview_step_distance(bundle.cfg),
                )
                mean = np.asarray(local_mode.mean_path, dtype=float)
            except Exception:
                if mode_index >= len(self._mode_mean_cache):
                    continue
                mean = np.asarray(self._mode_mean_cache[mode_index], dtype=float)

            if mean.ndim != 2 or mean.shape[0] < 2 or mean.shape[1] < 2:
                continue
            is_retained = mode_index in retained
            self.ax.plot(
                mean[:, 0],
                mean[:, 1],
                color='0.45' if is_retained else '0.70',
                linewidth=1.25 if is_retained else 0.9,
                alpha=0.65 if is_retained else 0.45,
                linestyle='-' if is_retained else '--',
                zorder=4,
            )

    def _draw_prior(self, state: np.ndarray, mode_index: Optional[int]) -> None:
        assert self.bundle is not None
        bundle = self.bundle
        variant = bundle.variant_value
        if variant.startswith('standard_mppi') or mode_index is None:
            self.ax.text(0.99, 0.99, 'No selected rollout prior', transform=self.ax.transAxes, ha='right', va='top', fontsize=8.5, color='#9467bd')
            return
        if not (0 <= mode_index < len(bundle.modes)):
            return
        global_mode = bundle.modes[mode_index]
        local_mode = bundle.module.localize_mode_for_state(
            global_mode,
            state,
            bundle.cfg.horizon,
            step_distance=controller_core.prior_preview_step_distance(bundle.cfg),
        )
        mean = np.asarray(local_mode.mean_path, dtype=float)
        self.ax.plot(mean[:, 0], mean[:, 1], color='#9467bd', linewidth=2.2, alpha=0.9, zorder=6)
        gaussian_variants = {'gaussian_prior_mppi', 'sensitivity_projected_gaussian_prior_mppi', 'mode_selecting_gaussian_mppi'}
        empirical_variants = {'control_bank_mppi'}
        corridor_variants = {'corridor_prior_mppi', 'mode_selecting_corridor_mppi'}
        if variant in gaussian_variants:
            self._draw_covariance_ellipses(local_mode)
        elif variant in empirical_variants:
            self._draw_empirical_samples(global_mode, state)
        elif variant in corridor_variants:
            pass

    def _draw_covariance_ellipses(self, local_mode: Any) -> None:
        assert self.bundle is not None
        mean = np.asarray(local_mode.mean_path, dtype=float)
        covariances = np.asarray(local_mode.cov_blocks, dtype=float)
        covariance_scale = float(getattr(self.bundle.cfg, 'gaussian_covariance_scale', 1.0))
        for index in range(0, len(mean), 4):
            covariance = np.asarray(covariances[index], dtype=float)[:2, :2]
            covariance = 0.5 * (covariance + covariance.T)
            values, vectors = np.linalg.eigh(covariance)
            values = np.maximum(values, 0.0)
            order = np.argsort(values)[::-1]
            values = values[order]
            vectors = vectors[:, order]
            angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
            width, height = 2.0 * covariance_scale * np.sqrt(values)
            self.ax.add_patch(Ellipse(mean[index], width=float(width), height=float(height), angle=angle, facecolor='#9467bd', edgecolor='#9467bd', alpha=0.2, linewidth=0.8, zorder=2))

    def _draw_empirical_samples(self, global_mode: Any, state: np.ndarray) -> None:
        assert self.bundle is not None
        paths = list(global_mode.sample_paths or [])
        if not paths:
            return
        indices = np.linspace(0, len(paths) - 1, min(10, len(paths)), dtype=int)
        for index in indices:
            path = self.bundle.module.localize_path_for_state(
                paths[int(index)],
                state,
                self.bundle.cfg.horizon,
                step_distance=controller_core.prior_preview_step_distance(self.bundle.cfg),
            )
            path = np.asarray(path, dtype=float)
            self.ax.plot(path[:, 0], path[:, 1], color='#9467bd', linewidth=0.8, alpha=0.18, zorder=2)

    @staticmethod
    def _transform_vehicle_points(points: np.ndarray, origin: tuple[float, float], angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        rotation = np.array([[c, -s], [s, c]], dtype=float)
        return np.asarray(points, dtype=float) @ rotation.T + np.asarray(origin)

    def _draw_robot(self, state: np.ndarray) -> None:
        assert self.bundle is not None
        if self.model_key == 'unicycle':
            x, y, heading = map(float, state[:3])
            radius = float(self.bundle.cfg.robot_radius)
            self.ax.add_patch(Circle((x, y), radius=radius, facecolor='#17becf', edgecolor='black', linewidth=1.0, zorder=12))
            arrow_length = max(0.38, 1.8 * radius)
            self.ax.arrow(x, y, arrow_length * math.cos(heading), arrow_length * math.sin(heading), width=0.025, head_width=0.13, head_length=0.14, length_includes_head=True, color='black', zorder=13)
            return
        if self.model_key in {'planar_quadrotor', 'planar_quadrotor_payload'}:
            minimum_state = 10 if self.model_key == 'planar_quadrotor_payload' else 6
            if state.size < minimum_state:
                raise ValueError(
                    'Planar quadrotor payload state must contain 10 values.'
                    if self.model_key == 'planar_quadrotor_payload'
                    else 'Planar quadrotor state must contain [x, y, vx, vy, theta, omega].'
                )
            x, y, _, _, theta, _ = map(float, state[:6])
            cfg = self.bundle.cfg
            total_radius = float(cfg.total_drone_radius)
            rotor_radius = float(cfg.rotor_radius)
            body_radius = float(cfg.body_radius)
            arm = max(0.0, total_radius - rotor_radius)
            endpoints = self._transform_vehicle_points(np.asarray([[-arm, 0.0], [arm, 0.0]]), (x, y), theta)

            if self.model_key == 'planar_quadrotor_payload':
                payload = np.asarray(self.bundle.module.payload_position(state, cfg), dtype=float)
                self.ax.plot([x, payload[0]], [y, payload[1]], color='0.20', linewidth=1.5, zorder=11)
                self.ax.add_patch(Circle((float(payload[0]), float(payload[1])), radius=float(cfg.payload_radius), facecolor='#ffbb78', edgecolor='black', linewidth=1.1, alpha=0.95, zorder=12))

            self.ax.plot(endpoints[:, 0], endpoints[:, 1], color='black', linewidth=2.0, zorder=13)
            self.ax.add_patch(Circle((x, y), radius=body_radius, facecolor='#17becf', edgecolor='black', linewidth=1.0, zorder=14))
            for point in endpoints:
                self.ax.add_patch(Circle((float(point[0]), float(point[1])), radius=rotor_radius, fill=False, edgecolor='black', linewidth=1.2, zorder=14))
            thrust_dir = self._transform_vehicle_points(np.asarray([[0.0, 0.0], [0.0, 0.34]]), (x, y), theta)
            self.ax.plot(thrust_dir[:, 0], thrust_dir[:, 1], color='#17becf', linewidth=1.4, zorder=13)
            return
        if state.size < 7:
            raise ValueError('Ackermann state must contain [x, y, psi, vx, vy, r, delta].')
        x, y, heading, vx, vy, _, steering = map(float, state[:7])
        cfg = self.bundle.cfg
        lf = float(getattr(cfg, 'front_axle_distance', 0.275))
        lr = float(getattr(cfg, 'rear_axle_distance', 0.275))
        wheelbase = max(lf + lr, 0.4)
        body_length = max(wheelbase + 0.26, 0.72)
        body_width = max(2.0 * float(cfg.robot_radius), 0.36)
        body = np.array([[-0.5 * body_length, -0.5 * body_width], [0.5 * body_length, -0.5 * body_width], [0.5 * body_length, 0.5 * body_width], [-0.5 * body_length, 0.5 * body_width]])
        self.ax.add_patch(Polygon(self._transform_vehicle_points(body, (x, y), heading), closed=True, facecolor='#17becf', edgecolor='black', linewidth=1.1, alpha=0.92, zorder=12))
        wheel_shape = np.array([[-0.11, -0.0275], [0.11, -0.0275], [0.11, 0.0275], [-0.11, 0.0275]])
        half_track = 0.43 * body_width
        c = math.cos(heading)
        s = math.sin(heading)

        def body_to_world(longitudinal: float, lateral: float) -> tuple[float, float]:
            return (x + longitudinal * c - lateral * s, y + longitudinal * s + lateral * c)
        for longitudinal, lateral, wheel_heading in ((lf, half_track, heading + steering), (lf, -half_track, heading + steering), (-lr, half_track, heading), (-lr, -half_track, heading)):
            center = body_to_world(longitudinal, lateral)
            vertices = self._transform_vehicle_points(wheel_shape, center, wheel_heading)
            self.ax.add_patch(Polygon(vertices, closed=True, facecolor='0.10', edgecolor='black', linewidth=0.6, zorder=13))
        world_vx = vx * c - vy * s
        world_vy = vx * s + vy * c
        if math.hypot(world_vx, world_vy) > 0.001:
            self.ax.arrow(x, y, 0.22 * world_vx, 0.22 * world_vy, width=0.012, head_width=0.1, head_length=0.11, length_includes_head=True, color='#1f77b4', zorder=14)
        nose = body_to_world(0.52 * body_length, 0.0)
        self.ax.plot([x, nose[0]], [y, nose[1]], color='black', linewidth=1.1, zorder=14)

def main() -> None:
    root = tk.Tk()
    app = InteractiveMPPIViewer(root)
    root.protocol('WM_DELETE_WINDOW', app.close)
    root.mainloop()
if __name__ == '__main__':
    main()