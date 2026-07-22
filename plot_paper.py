#!/usr/bin/env python3
"""Plot empirical and Gaussian trajectory priors in one Matplotlib figure.

The left subplot shows every retained empirical prior trajectory, colored by
homotopy mode. The right subplot shows the corresponding Gaussian mean paths
and covariance ellipses, using the same colors.

Run from the project root, where the project packages and save/best_policy.pkl
are available:

    python plot_paper_fixed.py --module "runs_soft_optimized.py" --show

The figure is also saved to ``trajectory_prior_comparison.png`` by default.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a two-panel plot of empirical homotopy priors and their "
            "Gaussian mean/covariance representation."
        )
    )
    parser.add_argument(
        "--module",
        default=os.environ.get("MPPI_SOFT_MODULE"),
        help=(
            "Path to the experiment module containing the scene and swarm planner. "
            "When omitted, the same soft-module candidates as the interactive viewer "
            "are checked, with runs_soft_optimized.py preferred."
        ),
    )
    parser.add_argument("--seed", type=int, default=5, help="Swarm-planner seed.")
    parser.add_argument(
        "--output",
        default="trajectory_prior_comparison.png",
        help="Output image path. The extension selects the Matplotlib format.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the Matplotlib window after saving the figure.",
    )
    parser.add_argument(
        "--ellipse-sigma",
        type=float,
        default=1.0,
        help="Covariance ellipse radius in standard deviations.",
    )
    parser.add_argument(
        "--ellipse-stride",
        type=int,
        default=4,
        help="Draw one covariance ellipse every this many mean-path points.",
    )
    parser.add_argument(
        "--max-paths-per-mode",
        type=int,
        default=0,
        help="Maximum empirical paths per mode; 0 plots every retained path.",
    )
    parser.add_argument(
        "--min-mode-samples",
        type=int,
        default=3,
        help="Minimum number of trajectories required to retain a homotopy mode.",
    )
    parser.add_argument(
        "--one-based-obstacles",
        action="store_true",
        help="Label obstacles from 1 instead of 0.",
    )
    return parser.parse_args()


def first_existing_module(explicit: str | None) -> Path:
    """Resolve the soft module with the same behavior as the interactive viewer."""
    if explicit:
        return Path(explicit).expanduser().resolve()

    candidates = [
        "runs_soft_optimized.py",
        "runs_soft_no_intermediate.py",
        "runs(7).py",
        "runs(6).py",
        "runs.py",
        "dynamic_block_soft.py",
        "dynamic_block_robustness.py",
    ]
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
        "No suitable soft controller module was found. Checked:\n" + formatted
    )


def load_module(path: Path) -> ModuleType:
    """Load a controller module exactly as the interactive viewer does.

    Registering the module in ``sys.modules`` before ``exec_module`` is required
    by dataclasses and postponed annotations used by the experiment modules.
    The module directory is also added to ``sys.path`` so project-local imports
    resolve when the plot script is launched outside that directory.
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment module not found: {path}")

    module_parent = str(path.parent)
    if module_parent not in sys.path:
        sys.path.insert(0, module_parent)

    module_name = f"mppi_prior_source_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def polygon_vertices(module: ModuleType, obstacle: Any) -> np.ndarray:
    if hasattr(module, "_poly_vertices"):
        vertices = module._poly_vertices(obstacle)
    elif hasattr(obstacle, "vertices"):
        vertices = obstacle.vertices
    else:
        vertices = obstacle

    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] < 2:
        raise ValueError(f"Obstacle vertices have invalid shape: {vertices.shape}")
    return vertices[:, :2]


def obstacle_center(module: ModuleType, obstacle: Any) -> np.ndarray:
    if hasattr(module, "obstacle_center"):
        return np.asarray(module.obstacle_center(obstacle), dtype=float)[:2]
    return polygon_vertices(module, obstacle).mean(axis=0)


def draw_scene(
    ax: plt.Axes,
    module: ModuleType,
    obstacles: Sequence[Any],
    start: np.ndarray,
    goal: np.ndarray,
    bounds_xy: tuple[np.ndarray, np.ndarray],
    *,
    one_based_obstacles: bool,
) -> None:
    for index, obstacle in enumerate(obstacles):
        vertices = polygon_vertices(module, obstacle)
        ax.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor="0.82",
                edgecolor="0.20",
                linewidth=1.1,
                zorder=1,
            )
        )
        center = obstacle_center(module, obstacle)
        label_index = index + 1 if one_based_obstacles else index
        ax.text(
            center[0],
            center[1],
            str(label_index),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox={"boxstyle": "circle,pad=0.18", "fc": "white", "ec": "0.25", "alpha": 0.9},
            zorder=6,
        )

    ax.scatter(
        [start[0]],
        [start[1]],
        marker="o",
        s=75,
        facecolor="white",
        edgecolor="black",
        linewidth=1.3,
        label="Start",
        zorder=8,
    )
    ax.scatter(
        [goal[0]],
        [goal[1]],
        marker="*",
        s=180,
        facecolor="gold",
        edgecolor="black",
        linewidth=1.0,
        label="Goal",
        zorder=8,
    )

    lower = np.asarray(bounds_xy[0], dtype=float)
    upper = np.asarray(bounds_xy[1], dtype=float)
    ax.set_xlim(float(lower[0]), float(upper[0]))
    ax.set_ylim(float(lower[1]), float(upper[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid(True, linewidth=0.5, alpha=0.25)


def retained_paths(mode: Any, maximum: int) -> list[np.ndarray]:
    paths = [np.asarray(path, dtype=float) for path in (mode.sample_paths or [])]
    if maximum <= 0 or len(paths) <= maximum:
        return paths

    # Deterministic, evenly spaced subsampling preserves the range of the bank.
    indices = np.linspace(0, len(paths) - 1, maximum, dtype=int)
    return [paths[int(index)] for index in indices]


def draw_covariance_ellipse(
    ax: plt.Axes,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    color: Any,
    sigma: float,
) -> None:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    major = eigenvectors[:, 0]
    angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    width, height = 2.0 * float(sigma) * np.sqrt(eigenvalues)

    ax.add_patch(
        Ellipse(
            xy=np.asarray(mean, dtype=float),
            width=float(width),
            height=float(height),
            angle=angle,
            facecolor=color,
            edgecolor=color,
            linewidth=0.9,
            alpha=0.14,
            zorder=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.ellipse_sigma <= 0.0:
        raise ValueError("--ellipse-sigma must be positive")
    if args.ellipse_stride <= 0:
        raise ValueError("--ellipse-stride must be positive")
    if args.max_paths_per_mode < 0:
        raise ValueError("--max-paths-per-mode must be nonnegative")

    module_path = first_existing_module(args.module)
    print(f"Loading experiment module: {module_path}")
    module = load_module(module_path)

    scale, bounds_xy, bounds_ranges, start, goal, obstacles = module.build_default_scene()
    generated = module.run_swarm_planner(
        start=start,
        goal=goal,
        obstacles=obstacles,
        scale=scale,
        bounds_xy=bounds_xy,
        seed=args.seed,
    )
    mixture = module.fit_topological_trajectory_mixture(
        generated,
        obstacles,
        K=50,
        beta=1.0,
        min_mode_samples=args.min_mode_samples,
        covariance_jitter=2e-4,
        bounds=bounds_ranges,
        goal=goal,
        snap_to_goal_radius=0.2,
        snap_straight_tail_points=8,
    )
    modes = module.mixture_to_mppi_modes(mixture)
    if not modes:
        raise RuntimeError("No homotopy modes were retained.")

    color_map = plt.get_cmap("tab20")
    colors = [color_map(index % color_map.N) for index in range(len(modes))]

    figure, (left_ax, right_ax) = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax in (left_ax, right_ax):
        draw_scene(
            ax,
            module,
            obstacles,
            np.asarray(start, dtype=float),
            np.asarray(goal, dtype=float),
            bounds_xy,
            one_based_obstacles=args.one_based_obstacles,
        )

    legend_handles: list[Line2D] = []

    for mode_index, (mode, color) in enumerate(zip(modes, colors)):
        paths = retained_paths(mode, args.max_paths_per_mode)
        path_alpha = max(0.07, min(0.28, 3.0 / max(1, len(paths))))
        for path in paths:
            if path.ndim != 2 or path.shape[1] < 2:
                continue
            left_ax.plot(
                path[:, 0],
                path[:, 1],
                color=color,
                linewidth=0.85,
                alpha=path_alpha,
                zorder=3,
            )

        mean = np.asarray(mode.mean_path, dtype=float)
        covariances = np.asarray(mode.cov_blocks, dtype=float)
        right_ax.plot(
            mean[:, 0],
            mean[:, 1],
            color=color,
            linewidth=2.4,
            zorder=5,
        )
        for point_index in range(0, len(mean), args.ellipse_stride):
            draw_covariance_ellipse(
                right_ax,
                mean[point_index],
                covariances[point_index],
                color=color,
                sigma=args.ellipse_sigma,
            )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.2,
                label=(
                    f"$h_{{{mode_index}}}$: "
                    f"$\\pi={float(mode.probability):.2f}$, "
                    f"$N={len(mode.sample_paths or [])}$"
                ),
            )
        )

    left_ax.set_title("Empirical trajectory prior by homotopy")
    right_ax.set_title(
        rf"Gaussian means and {args.ellipse_sigma:g}$\sigma$ covariance ellipses"
    )

    common_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            markerfacecolor="white", markeredgecolor="black", label="Start"
        ),
        Line2D(
            [0], [0], marker="*", linestyle="none", markersize=11,
            markerfacecolor="gold", markeredgecolor="black", label="Goal"
        ),
    ]

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    print(f"Saved figure: {output_path.resolve()}")
    print(f"Retained homotopy modes: {len(modes)}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()