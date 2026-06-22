from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_LABELS = {
    "avg_income": "Average rider income",
    "idle_rate": "Idle rate",
    "matching_rate": "Matching rate",
    "movement_rate": "Movement rate",
    "congestion_frequency": "Congestion frequency",
    "rider_concentration_hhi": "Rider concentration (HHI)",
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


def esc(text: str) -> str:
    """Escape text for SVG."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def format_number(value: float) -> str:
    """Format tick labels nicely."""
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
    width: int = 920,
    height: int = 580,
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
    x_index = {value: idx for idx, value in enumerate(x_values)}

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

    # Important: larger top padding prevents legend/title overlap.
    pad_left = 88
    pad_right = 36
    pad_top = 132
    pad_bottom = 78

    plot_x = pad_left
    plot_y = pad_top
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def sx(x_value: float) -> float:
        if len(x_values) == 1:
            return plot_x + plot_w / 2
        return plot_x + (x_index[x_value] / (len(x_values) - 1)) * plot_w

    def sy(y_value: float) -> float:
        return plot_y + plot_h - ((y_value - y_min) / (y_max - y_min)) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial, sans-serif; fill:#222}",
        ".main{font-size:23px; font-weight:700}",
        ".subtitle{font-size:12px; fill:#666}",
        ".label{font-size:13px; fill:#333}",
        ".tick{font-size:12px; fill:#666}",
        ".legend{font-size:12px; fill:#333}",
        "</style>",
        f'<text x="{pad_left}" y="34" class="main">{esc(title)}</text>',
        f'<text x="{pad_left}" y="58" class="subtitle">Mean over final simulation steps; error bars show 95% CI across seeds.</text>',
    ]

    # Decide whether to show legend. For beta SA there is only supply_adjusted,
    # so legend is hidden to avoid clutter and overlap.
    visible_series = [
        scenario for scenario in series_order
        if any(row["scenario"] == scenario for row in filtered)
    ]
    show_legend = len(visible_series) > 1

    if show_legend:
        legend_x = pad_left
        legend_y = 92
        legend_idx = 0
        for scenario in visible_series:
            color = COLORS.get(scenario, "#333333")
            lx = legend_x + legend_idx * 175
            elements.append(
                f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 28}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="4"/>'
            )
            elements.append(
                f'<text x="{lx + 36}" y="{legend_y + 4}" class="legend">'
                f'{esc(SCENARIO_LABELS.get(scenario, scenario))}</text>'
            )
            legend_idx += 1

    # Plot border
    elements.append(
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#cccccc"/>'
    )

    # Y-axis grid and labels
    for tick in range(5):
        frac = tick / 4
        y_value = y_min + frac * (y_max - y_min)
        ty = sy(y_value)
        elements.append(
            f'<line x1="{plot_x}" y1="{ty:.1f}" x2="{plot_x + plot_w}" y2="{ty:.1f}" '
            f'stroke="#eeeeee"/>'
        )
        elements.append(
            f'<text x="{plot_x - 10}" y="{ty + 4:.1f}" text-anchor="end" class="tick">'
            f'{format_number(y_value)}</text>'
        )

    # X-axis ticks and labels
    for x_value in x_values:
        tx = sx(x_value)
        elements.append(
            f'<line x1="{tx:.1f}" y1="{plot_y + plot_h}" x2="{tx:.1f}" '
            f'y2="{plot_y + plot_h + 5}" stroke="#999999"/>'
        )
        elements.append(
            f'<text x="{tx:.1f}" y="{plot_y + plot_h + 25}" text-anchor="middle" class="tick">'
            f'{format_number(x_value)}</text>'
        )

    # Axis labels
    elements.append(
        f'<text x="{plot_x + plot_w / 2}" y="{height - 22}" text-anchor="middle" class="label">'
        f'{esc(x_label)}</text>'
    )
    elements.append(
        f'<text x="24" y="{plot_y + plot_h / 2}" text-anchor="middle" '
        f'transform="rotate(-90 24 {plot_y + plot_h / 2})" class="label">'
        f'{esc(METRIC_LABELS[metric])}</text>'
    )

    # Draw each scenario line
    for scenario in series_order:
        scenario_rows = [row for row in filtered if row["scenario"] == scenario]
        if not scenario_rows:
            continue

        scenario_rows.sort(key=lambda row: float(row["parameter_value"]))
        color = COLORS.get(scenario, "#333333")

        points = []
        point_data = []

        for row in scenario_rows:
            x_value = float(row["parameter_value"])
            y_value = float(row[mean_key])
            ci_value = float(row.get(ci_key, 0.0))

            x_pos = sx(x_value)
            y_pos = sy(y_value)
            y_low = sy(y_value - ci_value)
            y_high = sy(y_value + ci_value)

            points.append(f"{x_pos:.1f},{y_pos:.1f}")
            point_data.append((x_pos, y_pos, y_low, y_high))

        # Line first
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )

        # Error bars and markers on top
        for x_pos, y_pos, y_low, y_high in point_data:
            elements.append(
                f'<line x1="{x_pos:.1f}" y1="{y_low:.1f}" x2="{x_pos:.1f}" '
                f'y2="{y_high:.1f}" stroke="{color}" stroke-width="1.5"/>'
            )
            elements.append(
                f'<line x1="{x_pos - 5:.1f}" y1="{y_low:.1f}" x2="{x_pos + 5:.1f}" '
                f'y2="{y_low:.1f}" stroke="{color}" stroke-width="1.5"/>'
            )
            elements.append(
                f'<line x1="{x_pos - 5:.1f}" y1="{y_high:.1f}" x2="{x_pos + 5:.1f}" '
                f'y2="{y_high:.1f}" stroke="{color}" stroke-width="1.5"/>'
            )
            elements.append(
                f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="4.8" '
                f'fill="{color}" stroke="white" stroke-width="1.2"/>'
            )

    elements.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create SVG plots for sensitivity-analysis summaries."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/sensitivity"),
        help="Root folder containing sensitivity summaries.",
    )
    parser.add_argument(
        "--summary-steps",
        type=int,
        default=20,
        help="Number of final steps used in the summary filename.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "avg_income",
            "congestion_frequency",
            "rider_concentration_hhi",
        ],
        choices=list(METRIC_LABELS),
        help="Metrics to plot.",
    )

    args = parser.parse_args()

    suffix = f"last{args.summary_steps}"
    figure_dir = args.output_root / "figures"

    rider_summary_path = (
        args.output_root
        / "riders"
        / f"riders_group_summary_{suffix}.csv"
    )
    beta_summary_path = (
        args.output_root
        / "heatmap_beta"
        / f"beta_group_summary_{suffix}.csv"
    )

    print(f"Looking for rider summary: {rider_summary_path}")
    print(f"Looking for beta summary: {beta_summary_path}")
    print(f"Figures will be saved to: {figure_dir}")

    if rider_summary_path.exists():
        rider_rows = read_summary(rider_summary_path)
        for metric in args.metrics:
            line_chart_with_error_bars(
                rows=rider_rows,
                metric=metric,
                title=f"Rider supply sensitivity: {METRIC_LABELS[metric]}",
                x_label="Number of riders",
                output_path=figure_dir / f"sa_riders_{metric}.svg",
                series_order=[
                    "baseline",
                    "demand_only",
                    "supply_adjusted",
                ],
            )
    else:
        print(f"Missing rider summary file: {rider_summary_path}")

    if beta_summary_path.exists():
        beta_rows = read_summary(beta_summary_path)
        for metric in args.metrics:
            line_chart_with_error_bars(
                rows=beta_rows,
                metric=metric,
                title=f"Heatmap beta sensitivity: {METRIC_LABELS[metric]}",
                x_label="Heatmap density penalty beta",
                output_path=figure_dir / f"sa_heatmap_beta_{metric}.svg",
                series_order=[
                    "supply_adjusted",
                ],
            )
    else:
        print(f"Missing beta summary file: {beta_summary_path}")

    print("Sensitivity plots complete.")


if __name__ == "__main__":
    main()