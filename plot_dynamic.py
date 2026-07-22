#!/usr/bin/env python3
"""
Summarize dynamic MPPI robustness trials by case and variant.

Only the requested results are included:
    - success rate
    - failure-reason distribution
    - control roughness before the wall, when an explicit pre-wall metric exists
    - control roughness over the complete trajectory
    - control roughness after the wall
    - control effort
    - number of steps
    - average runtime per step

Control roughness is the mean squared control change per valid transition, so
intervals of different lengths can be compared fairly; lower values are
smoother. The complete-trajectory metric is never mislabeled as a pre-wall
metric. Continuous metrics are reported as mean +/- standard deviation.
Success is reported as a count and percentage, without a binary standard
deviation. Failure reasons are reported as counts and percentages among failed
trials. Control roughness, control effort, and steps are calculated only over
successful trials. Average runtime per step is calculated over all trials with
valid values.

Default input:
    dynamic_block_robustness_trials.csv

Default outputs:
    dynamic_block_robustness_selected_summary_long.csv
    dynamic_block_robustness_selected_summary_wide.csv
    dynamic_block_robustness_selected_failure_reasons.csv
    dynamic_block_robustness_selected_metrics_table.pdf

Examples:
    python plot_robustness_metrics_selected.py

    python plot_robustness_metrics_selected.py \
        --input dynamic_block_robustness_trials.csv \
        --group-by condition variant

    python plot_robustness_metrics_selected.py \
        --group-by scenario_id variant \
        --output-prefix robustness_by_scenario
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


# The first existing column in each candidate list is used.
# Smoothness sums are handled separately below because they must be normalized
# by the number of control transitions before intervals can be compared.
CONTINUOUS_METRIC_SPECS = [
    (
        "control_effort",
        ["control_effort"],
        "Control effort",
    ),
    (
        "steps",
        ["steps", "number_of_steps", "num_steps"],
        "Number of steps",
    ),
    (
        "runtime_per_step_sec",
        [
            "runtime_per_step_sec",
            "average_time_per_step",
            "avg_time_per_step",
            "time_per_step_sec",
        ],
        "Average time per step (s)",
    ),
]

SMOOTHNESS_SUM_CANDIDATES = {
    "control_smoothness_before": [
        "control_smoothness_before_block",
        "control_smoothness_before",
    ],
    "control_smoothness_overall": [
        "control_smoothness",
        "control_smoothness_overall",
    ],
    "control_smoothness_after": [
        "control_smoothness_after_block",
        "control_smoothness_after",
    ],
}

STEPS_AFTER_CANDIDATES = [
    "steps_after_block",
    "steps_after_wall",
    "number_of_steps_after_block",
]
ACTIVATION_STEP_CANDIDATES = ["activation_step", "block_step", "wall_step"]

METRIC_DISPLAY_NAMES = {
    "control_smoothness_before": "Control roughness before wall (mean/transition)",
    "control_smoothness_overall": "Control roughness overall (mean/transition)",
    "control_smoothness_after": "Control roughness after wall (mean/transition)",
    **{output_name: label for output_name, _, label in CONTINUOUS_METRIC_SPECS},
}

METRIC_ORDER = [
    "control_smoothness_before",
    "control_smoothness_overall",
    "control_smoothness_after",
    "control_effort",
    "steps",
    "runtime_per_step_sec",
]

SUCCESS_CANDIDATES = ["success", "reached_goal"]
FAILURE_REASON_CANDIDATES = ["failure_reason", "reason_of_failure", "failure"]

# These controller-quality metrics are meaningful only for completed,
# successful trajectories. Their mean, SD, and n therefore use successful
# trials as the sample rather than all trials in the group.
SUCCESS_ONLY_METRICS = {
    "control_smoothness_before",
    "control_smoothness_overall",
    "control_smoothness_after",
    "control_effort",
    "steps",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize success, failure reasons, control smoothness, "
            "control effort, steps, and average time per step by case "
            "and variant."
        )
    )
    parser.add_argument(
        "--input",
        default="dynamic_block_robustness_trials.csv",
        help="Detailed trial CSV produced by dynamic_block_robustness.py.",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["condition", "variant"],
        help=(
            "Grouping columns representing case and variant. "
            "Default: condition variant"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="dynamic_block_robustness_selected",
        help="Prefix used for generated CSV and PDF files.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal places for continuous metrics.",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "Standard-deviation convention for continuous metrics: "
            "1=sample standard deviation, 0=population standard deviation."
        ),
    )
    parser.add_argument(
        "--exclude-controller-errors",
        action="store_true",
        help="Remove controller-error trials before calculating all results.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the generated table interactively in addition to saving it.",
    )
    return parser.parse_args()


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def coerce_bool_like(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean encodings to floating-point 0/1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").astype(float)
        return values.where(values.isin([0.0, 1.0]))

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": 1.0,
        "false": 0.0,
        "1": 1.0,
        "0": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "success": 1.0,
        "failure": 0.0,
        "passed": 1.0,
        "failed": 0.0,
    }
    return normalized.map(mapping).astype(float)


def normalize_failure_reason(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    missing = normalized.isna() | normalized.eq("") | normalized.str.lower().isin(
        {"nan", "none", "null", "n/a", "na"}
    )
    return normalized.mask(missing, "unspecified")


def normalized_transition_metric(
    numerator: pd.Series,
    control_count: pd.Series,
) -> pd.Series:
    """Convert a cumulative control-change cost to a mean per transition."""
    sums = pd.to_numeric(numerator, errors="coerce")
    counts = pd.to_numeric(control_count, errors="coerce")
    transitions = counts - 1.0
    values = sums / transitions.where(transitions > 0.0)
    return values.replace([np.inf, -np.inf], np.nan)


def derive_normalized_smoothness_metrics(
    df: pd.DataFrame,
    resolved_metrics: dict[str, str],
) -> None:
    """Add comparable smoothness metrics without relabeling total as pre-wall.

    The trial CSV stores cumulative control-change costs. Dividing by the number
    of within-interval control transitions removes the interval-length bias.
    A true pre-wall value is produced only when the CSV contains an explicit
    pre-wall cumulative metric; the complete-trajectory value is otherwise
    reported as overall.
    """
    steps_column = resolved_metrics.get("steps")
    if steps_column is None:
        return

    overall_source = first_existing_column(
        df, SMOOTHNESS_SUM_CANDIDATES["control_smoothness_overall"]
    )
    if overall_source is not None:
        df[overall_source] = pd.to_numeric(df[overall_source], errors="coerce")
        derived = "_derived_control_smoothness_overall_per_transition"
        df[derived] = normalized_transition_metric(df[overall_source], df[steps_column])
        resolved_metrics["control_smoothness_overall"] = derived

    after_source = first_existing_column(
        df, SMOOTHNESS_SUM_CANDIDATES["control_smoothness_after"]
    )
    if after_source is not None:
        df[after_source] = pd.to_numeric(df[after_source], errors="coerce")
        after_steps_column = first_existing_column(df, STEPS_AFTER_CANDIDATES)
        if after_steps_column is not None:
            df[after_steps_column] = pd.to_numeric(df[after_steps_column], errors="coerce")
            after_control_count = df[after_steps_column]
        else:
            activation_column = first_existing_column(df, ACTIVATION_STEP_CANDIDATES)
            if activation_column is None:
                after_control_count = None
            else:
                df[activation_column] = pd.to_numeric(df[activation_column], errors="coerce")
                after_control_count = df[steps_column] - df[activation_column]

        if after_control_count is not None:
            derived = "_derived_control_smoothness_after_per_transition"
            df[derived] = normalized_transition_metric(df[after_source], after_control_count)
            resolved_metrics["control_smoothness_after"] = derived

    before_source = first_existing_column(
        df, SMOOTHNESS_SUM_CANDIDATES["control_smoothness_before"]
    )
    if before_source is not None:
        df[before_source] = pd.to_numeric(df[before_source], errors="coerce")
        activation_column = first_existing_column(df, ACTIVATION_STEP_CANDIDATES)
        if activation_column is not None:
            df[activation_column] = pd.to_numeric(df[activation_column], errors="coerce")
            before_control_count = df[activation_column]
            derived = "_derived_control_smoothness_before_per_transition"
            df[derived] = normalized_transition_metric(df[before_source], before_control_count)
            resolved_metrics["control_smoothness_before"] = derived


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str | None, dict[str, str]]:
    df = df.copy()

    success_column = first_existing_column(df, SUCCESS_CANDIDATES)
    if success_column is None:
        raise KeyError(
            "No success column found. Expected one of: "
            + ", ".join(SUCCESS_CANDIDATES)
        )
    df[success_column] = coerce_bool_like(df[success_column])

    failure_reason_column = first_existing_column(df, FAILURE_REASON_CANDIDATES)
    if failure_reason_column is not None:
        df[failure_reason_column] = normalize_failure_reason(df[failure_reason_column])

    resolved_metrics: dict[str, str] = {}
    for output_name, candidates, _ in CONTINUOUS_METRIC_SPECS:
        source_column = first_existing_column(df, candidates)
        if source_column is not None:
            df[source_column] = pd.to_numeric(df[source_column], errors="coerce")
            resolved_metrics[output_name] = source_column

    derive_normalized_smoothness_metrics(df, resolved_metrics)

    # Derive time per step when the CSV contains total runtime and steps but no
    # precomputed per-step runtime column. Invalid or zero step counts become NaN.
    if "runtime_per_step_sec" not in resolved_metrics:
        runtime_column = first_existing_column(
            df, ["runtime_sec", "total_runtime_sec", "elapsed_time_sec"]
        )
        steps_column = resolved_metrics.get("steps")
        if runtime_column is not None and steps_column is not None:
            df[runtime_column] = pd.to_numeric(df[runtime_column], errors="coerce")
            derived_column = "_derived_runtime_per_step_sec"
            valid_steps = df[steps_column].where(df[steps_column] > 0)
            df[derived_column] = df[runtime_column] / valid_steps
            df[derived_column] = df[derived_column].replace(
                [np.inf, -np.inf], np.nan
            )
            resolved_metrics["runtime_per_step_sec"] = derived_column

    if not resolved_metrics:
        expected = sorted(
            {
                candidate
                for _, candidates, _ in CONTINUOUS_METRIC_SPECS
                for candidate in candidates
            }
            | {
                candidate
                for candidates in SMOOTHNESS_SUM_CANDIDATES.values()
                for candidate in candidates
            }
        )
        raise KeyError(
            "None of the requested continuous metric columns were found. "
            "Expected one or more of: " + ", ".join(expected)
        )

    return df, success_column, failure_reason_column, resolved_metrics


def calculate_spread(values: pd.Series, ddof: int) -> float:
    """Calculate standard deviation explicitly over finite trial values."""
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= ddof:
        return np.nan
    return float(np.std(clean.to_numpy(dtype=float), ddof=ddof))


def aggregate_continuous_metrics(
    df: pd.DataFrame,
    group_columns: list[str],
    success_column: str,
    resolved_metrics: dict[str, str],
    ddof: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns

    for group_values, group in df.groupby(group_key, dropna=False, sort=True):
        if len(group_columns) == 1:
            group_values = (group_values,)
        group_dict = dict(zip(group_columns, group_values))

        for output_name, source_column in resolved_metrics.items():
            if output_name in SUCCESS_ONLY_METRICS:
                metric_group = group.loc[
                    pd.to_numeric(group[success_column], errors="coerce").eq(1.0)
                ]
                sample_scope = "successful trials only"
            else:
                metric_group = group
                sample_scope = "all trials with valid values"

            values = (
                pd.to_numeric(metric_group[source_column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            rows.append(
                {
                    **group_dict,
                    "metric": output_name,
                    "source_column": source_column,
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "std": calculate_spread(values, ddof=ddof),
                    "n": int(len(values)),
                    "sample_scope": sample_scope,
                }
            )

    return pd.DataFrame(rows)


def aggregate_outcomes(
    df: pd.DataFrame,
    group_columns: list[str],
    success_column: str,
    failure_reason_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    reason_rows: list[dict[str, object]] = []
    group_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns

    for group_values, group in df.groupby(group_key, dropna=False, sort=True):
        if len(group_columns) == 1:
            group_values = (group_values,)
        group_dict = dict(zip(group_columns, group_values))

        success_values = pd.to_numeric(group[success_column], errors="coerce").dropna()
        total_with_outcome = int(len(success_values))
        success_count = int((success_values == 1.0).sum())
        failure_count = int((success_values == 0.0).sum())
        success_rate = (
            float(success_count / total_with_outcome) if total_with_outcome else np.nan
        )

        if failure_reason_column is None:
            reason_counts = pd.Series(dtype="int64")
        else:
            # Failure-reason percentages use failed trials as their denominator.
            failure_mask = pd.to_numeric(group[success_column], errors="coerce").eq(0.0)
            reasons = normalize_failure_reason(group.loc[failure_mask, failure_reason_column])
            reason_counts = reasons.value_counts(dropna=False, sort=False).sort_index()

        reason_text_parts: list[str] = []
        for reason, count in reason_counts.items():
            count_int = int(count)
            percent = 100.0 * count_int / failure_count if failure_count else np.nan
            reason_rows.append(
                {
                    **group_dict,
                    "failure_reason": str(reason),
                    "count": count_int,
                    "percent_of_failures": percent,
                    "failure_trials": failure_count,
                    "total_trials_with_outcome": total_with_outcome,
                }
            )
            reason_text_parts.append(f"{reason}: {count_int} ({percent:.1f}%)")

        if failure_count == 0:
            failure_reasons_text = "none"
        elif reason_text_parts:
            failure_reasons_text = "; ".join(reason_text_parts)
        else:
            failure_reasons_text = f"unspecified: {failure_count} (100.0%)"
            reason_rows.append(
                {
                    **group_dict,
                    "failure_reason": "unspecified",
                    "count": failure_count,
                    "percent_of_failures": 100.0,
                    "failure_trials": failure_count,
                    "total_trials_with_outcome": total_with_outcome,
                }
            )

        summary_rows.append(
            {
                **group_dict,
                "trials": int(len(group)),
                "trials_with_outcome": total_with_outcome,
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": success_rate,
                "failure_reasons": failure_reasons_text,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(reason_rows)


def format_mean_std(mean: float, std: float, n: int, precision: int) -> str:
    if pd.isna(mean):
        return "-"
    std_text = "-" if pd.isna(std) else f"{std:.{precision}f}"
    return f"{mean:.{precision}f} +/- {std_text} (n={n})"


def metric_display_name(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())


def build_wide_table(
    outcome_df: pd.DataFrame,
    continuous_df: pd.DataFrame,
    group_columns: list[str],
    precision: int,
) -> pd.DataFrame:
    result = outcome_df.copy()
    result["Success"] = result.apply(
        lambda row: (
            "-"
            if pd.isna(row["success_rate"])
            else (
                f"{int(row['success_count'])}/{int(row['trials_with_outcome'])} "
                f"({100.0 * float(row['success_rate']):.1f}%)"
            )
        ),
        axis=1,
    )
    result["Failure reasons"] = result["failure_reasons"]

    formatted = continuous_df.copy()
    formatted["display_name"] = formatted["metric"].map(metric_display_name)
    formatted["formatted_value"] = formatted.apply(
        lambda row: format_mean_std(
            mean=float(row["mean"]) if not pd.isna(row["mean"]) else np.nan,
            std=float(row["std"]) if not pd.isna(row["std"]) else np.nan,
            n=int(row["n"]),
            precision=precision,
        ),
        axis=1,
    )

    metric_wide = formatted.pivot(
        index=group_columns,
        columns="display_name",
        values="formatted_value",
    ).reset_index()

    result = result.merge(metric_wide, on=group_columns, how="left")

    present_metrics = set(continuous_df["metric"])
    ordered_metric_columns = [
        metric_display_name(metric)
        for metric in METRIC_ORDER
        if metric in present_metrics
    ]
    columns = group_columns + ["trials", "Success", "Failure reasons"] + ordered_metric_columns
    return result.reindex(columns=columns)


def make_group_label(row: pd.Series, group_columns: list[str]) -> str:
    return " | ".join(f"{column}={row[column]}" for column in group_columns)


def render_pdf_table(
    wide_df: pd.DataFrame,
    group_columns: list[str],
    output_path: Path,
    ddof: int,
    show: bool,
) -> None:
    if wide_df.empty:
        raise ValueError("No aggregated values are available to plot.")

    page = wide_df.copy()
    page.insert(0, "Case / variant", page.apply(lambda row: make_group_label(row, group_columns), axis=1))
    page = page.drop(columns=group_columns)

    # Put failure reasons on their own page because categorical text can be wide.
    compact_columns = [column for column in page.columns if column != "Failure reasons"]
    failure_columns = ["Case / variant", "Failure reasons"]

    std_name = "sample SD" if ddof == 1 else "population SD"

    with PdfPages(output_path) as pdf:
        for title, columns in [
            (
                "Selected robustness metrics\n"
                f"Continuous values: mean +/- {std_name}; "
                "control roughness is mean per transition (lower is smoother); "
                "roughness/effort/steps: successful trials only",
                compact_columns,
            ),
            ("Failure reasons\nCounts and percentages among failed trials", failure_columns),
        ]:
            current = page[columns].copy()
            width = max(12.0, 2.2 + 2.15 * len(current.columns))
            height = max(3.5, 2.0 + 0.58 * len(current.index))

            fig, ax = plt.subplots(figsize=(width, height))
            ax.axis("off")
            table = ax.table(
                cellText=current.values,
                colLabels=current.columns,
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.45)
            ax.set_title(title, pad=18)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    missing_groups = [column for column in args.group_by if column not in df.columns]
    if missing_groups:
        raise KeyError(
            "Grouping column(s) missing from input CSV: " + ", ".join(missing_groups)
        )

    df, success_column, failure_reason_column, resolved_metrics = prepare_dataframe(df)

    if args.exclude_controller_errors and failure_reason_column is not None:
        controller_error_mask = (
            df[failure_reason_column].astype("string").str.strip().str.lower().eq("controller_error")
        )
        df = df.loc[~controller_error_mask].copy()

    outcome_df, failure_reason_df = aggregate_outcomes(
        df=df,
        group_columns=args.group_by,
        success_column=success_column,
        failure_reason_column=failure_reason_column,
    )
    continuous_df = aggregate_continuous_metrics(
        df=df,
        group_columns=args.group_by,
        success_column=success_column,
        resolved_metrics=resolved_metrics,
        ddof=args.ddof,
    )
    wide_df = build_wide_table(
        outcome_df=outcome_df,
        continuous_df=continuous_df,
        group_columns=args.group_by,
        precision=args.precision,
    )

    prefix = Path(args.output_prefix)
    long_csv = prefix.with_name(prefix.name + "_summary_long.csv")
    wide_csv = prefix.with_name(prefix.name + "_summary_wide.csv")
    failure_csv = prefix.with_name(prefix.name + "_failure_reasons.csv")
    table_pdf = prefix.with_name(prefix.name + "_metrics_table.pdf")

    continuous_df.to_csv(long_csv, index=False)
    wide_df.to_csv(wide_csv, index=False)
    failure_reason_df.to_csv(failure_csv, index=False)
    render_pdf_table(
        wide_df=wide_df,
        group_columns=args.group_by,
        output_path=table_pdf,
        ddof=args.ddof,
        show=args.show,
    )

    print(f"Read {len(df)} trials from: {input_path}")
    print(f"Grouped by: {', '.join(args.group_by)}")
    print(f"Success column: {success_column}")
    print(f"Failure-reason column: {failure_reason_column or 'not available'}")
    print("Continuous metric sources:")
    for output_name, source_column in resolved_metrics.items():
        print(f"  {output_name}: {source_column}")
    print(
        "Standard deviation: "
        + ("sample SD (ddof=1)" if args.ddof == 1 else "population SD (ddof=0)")
    )
    print(
        "Control roughness: mean squared control change per transition; "
        "lower is smoother"
    )
    print(
        "A pre-wall roughness metric is shown only when the input contains an "
        "explicit pre-wall column; overall roughness is not relabeled as before-wall"
    )
    print("Control roughness, control effort, and steps: successful trials only")
    print("Average time per step: all trials with valid values")
    print(f"Saved continuous mean/SD data: {long_csv}")
    print(f"Saved formatted summary: {wide_csv}")
    print(f"Saved failure-reason data: {failure_csv}")
    print(f"Saved plotted table: {table_pdf}")


if __name__ == "__main__":
    main()