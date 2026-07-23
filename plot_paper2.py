#!/usr/bin/env python3
"""Plot all MPPI controller variants for no-wall, static-wall, and dynamic-wall conditions.

The script uses the default scene without obstacle-center permutations, runs every
ControllerVariant with 32 MPPI rollouts, and creates one figure with three
side-by-side subplots.

Run from the project root, for example:
    python plot_three_conditions.py --module 'runs(14).py' --show
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np


PRETTY_VARIANT_NAMES = {
    "full_swarm_prior_mppi": "Full Swarm Prior MPPI",
    "gaussian_prior_mppi": "Gaussian Prior MPPI",
    "corridor_prior_mppi": "Corridor Prior MPPI",
    "frenet_corridor_mppi": "Frenet Corridor MPPI",
    "heatmap_prior_mppi": "Heatmap Prior MPPI",
    "control_bank_mppi": "Control Bank MPPI",
    "mode_selecting_homotopy_mppi": "Mode-Selecting Homotopy MPPI",
    "mode_selecting_corridor_mppi": "Mode-Selecting Corridor MPPI",
    "standard_mppi": "Standard MPPI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        default=os.environ.get("MPPI_SOFT_MODULE", "runs.py"),
        help="Path to the experiment/controller module.",
    )
    parser.add_argument("--output", default="all_variants_three_conditions.png")
    parser.add_argument("--swarm-seed", type=int, default=5)
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument("--scenario", default="walls_0_1__1_2")
    parser.add_argument("--max-steps", type=int, default=130)
    parser.add_argument("--goal-tolerance", type=float, default=0.30)
    parser.add_argument("--activation-preview-clearance", type=float, default=0.75)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_module(path: Path) -> ModuleType:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment module not found: {path}")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))

    name = f"mppi_three_conditions_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def vertices(module: ModuleType, obstacle: Any) -> np.ndarray:
    if hasattr(module, "_poly_vertices"):
        value = module._poly_vertices(obstacle)
    elif hasattr(obstacle, "vertices"):
        value = obstacle.vertices
    else:
        value = obstacle

    array = np.asarray(value, dtype=float)
    return array[:, :2]


def draw_obstacles(
    ax: plt.Axes,
    module: ModuleType,
    obstacles: Sequence[Any],
    *,
    wall_count: int = 0,
) -> None:
    base_count = len(obstacles) - wall_count

    for index, obstacle in enumerate(obstacles):
        is_wall = index >= base_count
        ax.add_patch(
            Polygon(
                vertices(module, obstacle),
                closed=True,
                facecolor="0.45" if is_wall else "0.82",
                edgecolor="0.12" if is_wall else "0.20",
                linewidth=1.3 if is_wall else 1.0,
                hatch="///" if is_wall else None,
                alpha=0.85 if is_wall else 1.0,
                zorder=2 if is_wall else 1,
            )
        )


def draw_scene(
    ax: plt.Axes,
    module: ModuleType,
    obstacles: Sequence[Any],
    start: np.ndarray,
    goal: np.ndarray,
    bounds_xy: tuple[np.ndarray, np.ndarray],
    *,
    wall_count: int = 0,
) -> None:
    draw_obstacles(ax, module, obstacles, wall_count=wall_count)

    ax.scatter(
        [start[0]],
        [start[1]],
        marker="o",
        s=75,
        facecolor="white",
        edgecolor="black",
        linewidth=1.3,
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
        zorder=8,
    )

    lower, upper = (np.asarray(value, dtype=float) for value in bounds_xy)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x$")
    ax.grid(True, linewidth=0.5, alpha=0.25)


def build_modes(
    module: ModuleType,
    start,
    goal,
    obstacles,
    scale,
    bounds_xy,
    bounds_ranges,
    swarm_seed: int,
):
    return module.build_homotopy_modes_for_obstacles(
        start,
        goal,
        obstacles,
        scale,
        bounds_xy,
        bounds_ranges,
        swarm_seed,
    )


def run_condition(
    module: ModuleType,
    condition: str,
    variants: Sequence[Any],
    modes,
    base_obstacles,
    blocker,
    start,
    goal,
    cfg,
    args: argparse.Namespace,
    scenario,
):
    results = []

    for variant in variants:
        print(f"Running {condition:12s} | {variant.value}")
        result = module.run_dynamic_blockage_controller(
            variant=variant,
            modes=modes,
            base_obstacles=base_obstacles,
            blocker=blocker,
            start=start,
            goal=goal,
            seed=args.controller_seed,
            trigger_progress=(
                None if condition != "dynamic_wall" else scenario.trigger_progress
            ),
            activation_preview_clearance=(
                None
                if condition != "dynamic_wall"
                else args.activation_preview_clearance
            ),
            blocker_active_from_start=(condition == "static_wall"),
            condition=condition,
            max_steps=args.max_steps,
            goal_tolerance=args.goal_tolerance,
            mppi_cfg=cfg,
            record_infos=False,
            record_obstacle_history=False,
        )
        results.append(result)

    return results


def main() -> None:
    args = parse_args()
    module = load_module(Path(args.module))

    (
        scale,
        bounds_xy,
        bounds_ranges,
        start,
        goal,
        base_obstacles,
    ) = module.build_default_scene()

    scenarios = {
        item.scenario_id: item
        for item in module.default_dynamic_wall_scenarios()
    }
    if args.scenario not in scenarios:
        raise ValueError(
            f"Unknown scenario {args.scenario!r}; choose from {sorted(scenarios)}"
        )

    scenario = scenarios[args.scenario]
    module.validate_dynamic_wall_scenario(scenario, len(base_obstacles))

    blocker = module.make_wall_blockers_between_obstacles(
        base_obstacles,
        scenario.wall_pairs,
        width=scenario.wall_width,
        extension=scenario.wall_extension,
    )
    static_obstacles = list(base_obstacles) + list(blocker)

    print("Building default-scene no-wall/dynamic prior...")
    base_modes = build_modes(
        module,
        start,
        goal,
        base_obstacles,
        scale,
        bounds_xy,
        bounds_ranges,
        args.swarm_seed,
    )

    print("Building default-scene static-wall oracle prior...")
    static_modes = build_modes(
        module,
        start,
        goal,
        static_obstacles,
        scale,
        bounds_xy,
        bounds_ranges,
        args.swarm_seed,
    )

    cfg = module.MPPIConfig(num_rollouts=32)
    variants = list(module.ControllerVariant)

    no_wall = run_condition(
        module,
        "no_wall",
        variants,
        base_modes,
        base_obstacles,
        [],
        start,
        goal,
        cfg,
        args,
        scenario,
    )
    static_wall = run_condition(
        module,
        "static_wall",
        variants,
        static_modes,
        static_obstacles,
        [],
        start,
        goal,
        cfg,
        args,
        scenario,
    )
    dynamic_wall = run_condition(
        module,
        "dynamic_wall",
        variants,
        base_modes,
        base_obstacles,
        blocker,
        start,
        goal,
        cfg,
        args,
        scenario,
    )

    color_map = plt.get_cmap("tab10")
    colors = {
        variant.value: color_map(index % color_map.N)
        for index, variant in enumerate(variants)
    }

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 6.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    panels = [
        (axes[0], "No Wall", no_wall, base_obstacles, 0),
        (
            axes[1],
            "Static Wall",
            static_wall,
            static_obstacles,
            len(blocker),
        ),
        (
            axes[2],
            "Dynamic Two-Wall Blockage",
            dynamic_wall,
            static_obstacles,
            len(blocker),
        ),
    ]

    for ax, title, results, displayed_obstacles, wall_count in panels:
        draw_scene(
            ax,
            module,
            displayed_obstacles,
            np.asarray(start),
            np.asarray(goal),
            bounds_xy,
            wall_count=wall_count,
        )

        for result in results:
            states = np.asarray(result["states"], dtype=float)
            color = colors[result["variant"]]

            ax.plot(
                states[:, 0],
                states[:, 1],
                color=color,
                linewidth=1.8,
                alpha=0.92,
                zorder=4,
            )

            if title.startswith("Dynamic") and result.get("activation_step") is not None:
                step = int(result["activation_step"])
                if 0 <= step < len(states):
                    ax.scatter(
                        states[step, 0],
                        states[step, 1],
                        marker="x",
                        s=34,
                        color=color,
                        linewidth=1.2,
                        zorder=7,
                    )

        ax.set_title(title)

    axes[0].set_ylabel("$y$")

    handles = [
        Line2D(
            [0],
            [0],
            color=colors[variant.value],
            linewidth=2.2,
            label=PRETTY_VARIANT_NAMES.get(
                variant.value,
                variant.value.replace("_", " ").title(),
            ),
        )
        for variant in variants
    ]

    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=7,
                markerfacecolor="white",
                markeredgecolor="black",
                label="Start",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="none",
                markersize=11,
                markerfacecolor="gold",
                markeredgecolor="black",
                label="Goal",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="none",
                markersize=7,
                color="black",
                label="Dynamic-Wall Activation",
            ),
        ]
    )

    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=5,
        fontsize=10,
        frameon=True,
    )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    print(f"Saved figure: {output.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()