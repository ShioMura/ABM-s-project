from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Shortened labels for cleaner, single-line plot titles
METRIC_LABELS = {
    "avg_income": "Average Income",
    "rider_density_cv": "Herding (Density CV)",
    "congestion_frequency": "Congestion Frequency",
    "rider_concentration_hhi": "Concentration (HHI)",
}

SCENARIO_LABELS = {
    "baseline": "Baseline",
    "demand_only": "Demand-only",
    "supply_adjusted": "Supply-adjusted",
}

COLORS = {
    "baseline": "#4C78A8",
    "demand_only": "#F58518",
    "supply_adjusted": "#54A24B",
}

RIDER_METRICS = [
    "avg_income",
    "rider_density_cv",
    "congestion_frequency",
]

BETA_METRICS = [
    "avg_income",
    "rider_density_cv",
    "congestion_frequency",
]


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def format_number(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def line_chart_with_error_bars(
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    x_label: str,
    output_path: Path,
    series_order: list[str],
    fig_width: float = 7.0,
    fig_height: float = 5.0,
) -> None:
    mean_key = f"{metric}_mean"
    ci_key = f"{metric}_ci95"

    filtered = [
        row for row in rows
        if mean_key in row and row.get(mean_key, "") != ""
    ]

    if not filtered:
        print(f"No data available for {metric}; skipped {output_path}")
        return

    x_values = sorted({float(row["parameter_value"]) for row in filtered})

    y_lows = []
    y_highs = []
    for row in filtered:
        y = float(row[mean_key])
        ci = float(row.get(ci_key, 0.0))
        y_lows.append(y - ci)
        y_highs.append(y + ci)

    y_min = min(y_lows)
    y_max = max(y_highs)

    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    margin = 0.12 * (y_max - y_min)
    y_min -= margin
    y_max += margin

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 14,  # Slightly reduced for cleaner headers
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    fig.suptitle(
        title,
        fontsize=14,
        x=0.5,
        y=0.94,
        ha="center",
    )

    ax.grid(True, axis="y", color="#eeeeee", linewidth=1.0)
    ax.set_axisbelow(True)

    visible_series = [
        scenario for scenario in series_order
        if any(row["scenario"] == scenario for row in filtered)
    ]

    for scenario in series_order:
        scenario_rows = [row for row in filtered if row["scenario"] == scenario]
        if not scenario_rows:
            continue

        scenario_rows.sort(key=lambda row: float(row["parameter_value"]))
        color = COLORS.get(scenario, "#333333")

        xs = [float(row["parameter_value"]) for row in scenario_rows]
        ys = [float(row[mean_key]) for row in scenario_rows]
        cis = [float(row.get(ci_key, 0.0)) for row in scenario_rows]

        ax.errorbar(
            xs,
            ys,
            yerr=cis,
            fmt="o-",
            color=color,
            linewidth=1.5,
            markersize=5.2,
            capsize=4,
            elinewidth=1.2,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=SCENARIO_LABELS.get(scenario, scenario),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(METRIC_LABELS[metric])

    ax.set_xticks(x_values)
    ax.set_xticklabels([format_number(x) for x in x_values])

    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([format_number(y) for y in y_ticks])

    if len(visible_series) > 1:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=len(visible_series),
            frameon=False,
        )

    fig.subplots_adjust(
        left=0.15,
        right=0.94,
        bottom=0.15,
        top=0.82,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {output_path}")


def scatter_plot_global(
    rows: list[dict[str, str]], 
    metric: str, 
    param: str, 
    output_path: Path,
    fig_width: float = 6.0,
    fig_height: float = 4.0,
):
    mean_key = f"{metric}_mean"
    if not rows or mean_key not in rows[0]: 
        return

    xs = [float(r[param]) for r in rows]
    ys = [float(r[mean_key]) for r in rows]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Use shorter titles for Global SA plots
    clean_param = param.replace("_", " ").capitalize()
    fig.suptitle(f"Global SA: {clean_param} vs. {METRIC_LABELS[metric]}", fontsize=12, y=0.95)
    
    ax.grid(True, color="#eeeeee", linewidth=1.0)
    ax.set_axisbelow(True) 
    ax.scatter(xs, ys, alpha=0.7, color="#4C78A8", edgecolors="white", zorder=3)
    ax.set_xlabel(clean_param)
    ax.set_ylabel(METRIC_LABELS[metric])

    fig.subplots_adjust(left=0.15, right=0.94, bottom=0.15, top=0.85)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved Global SA scatter: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create PDF plots for sensitivity-analysis summaries."
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sensitivity"))
    parser.add_argument("--summary-steps", type=int, default=20)
    parser.add_argument("--metrics", nargs="+", default=None, choices=list(METRIC_LABELS))
    parser.add_argument("--rider-metrics", nargs="+", default=None, choices=list(METRIC_LABELS))
    parser.add_argument("--beta-metrics", nargs="+", default=None, choices=list(METRIC_LABELS))
    parser.add_argument("--fig-width", type=float, default=6.0)
    parser.add_argument("--fig-height", type=float, default=4.5)

    args = parser.parse_args()

    rider_metrics = args.rider_metrics if args.rider_metrics is not None else RIDER_METRICS
    beta_metrics = args.beta_metrics if args.beta_metrics is not None else BETA_METRICS

    if args.metrics is not None:
        rider_metrics = [metric for metric in rider_metrics if metric in args.metrics]
        beta_metrics = [metric for metric in beta_metrics if metric in args.metrics]

    suffix = f"last{args.summary_steps}"
    figure_dir = args.output_root / "figures"

    rider_summary_path = args.output_root / "riders" / f"riders_group_summary_{suffix}.csv"
    beta_summary_path = args.output_root / "heatmap_beta" / f"beta_group_summary_{suffix}.csv"
    global_summary_path = args.output_root / "global" / f"global_group_summary_{suffix}.csv"

    if rider_summary_path.exists():
        rider_rows = read_summary(rider_summary_path)
        for metric in rider_metrics:
            line_chart_with_error_bars(
                rows=rider_rows, metric=metric,
                title=f"Rider Supply vs. {METRIC_LABELS[metric]}",
                x_label="Number of Riders",
                output_path=figure_dir / f"sa_riders_{metric}.pdf",
                series_order=["baseline", "demand_only", "supply_adjusted"],
                fig_width=args.fig_width, fig_height=args.fig_height,
            )

    if beta_summary_path.exists():
        beta_rows = read_summary(beta_summary_path)
        for metric in beta_metrics:
            line_chart_with_error_bars(
                rows=beta_rows, metric=metric,
                title=f"Density Penalty vs. {METRIC_LABELS[metric]}",
                x_label=r"Heatmap Density Penalty ($\beta$)",
                output_path=figure_dir / f"sa_heatmap_beta_{metric}.pdf",
                series_order=["supply_adjusted"],
                fig_width=args.fig_width, fig_height=args.fig_height,
            )

    if global_summary_path.exists():
        global_rows = read_summary(global_summary_path)
        global_params = ["riders", "heatmap_alpha", "heatmap_beta", "heatmap_gamma"]
        for metric in rider_metrics:
            for param in global_params:
                scatter_plot_global(
                    rows=global_rows, metric=metric, param=param,
                    output_path=figure_dir / f"sa_global_{param}_vs_{metric}.pdf",
                    fig_width=args.fig_width * 0.85, fig_height=args.fig_height * 0.8,
                )

    print("Sensitivity plots complete.")

if __name__ == "__main__":
    main()