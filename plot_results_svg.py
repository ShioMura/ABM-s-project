from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCENARIO_FILES = {
    "Baseline": "outputs/baseline.csv",
    "Demand-only": "outputs/demand_only.csv",
    "Supply-adjusted": "outputs/supply_adjusted.csv",
}

METRICS = {
    "avg_income": "Average rider income",
    "idle_rate": "Idle rate",
    "matching_rate": "Matching rate",
    "movement_rate": "Movement rate",
    "congestion_frequency": "Congestion frequency",
    "rider_concentration_hhi": "Rider concentration (HHI)",
}

COLORS = {
    "Baseline": "#4C78A8",
    "Demand-only": "#F58518",
    "Supply-adjusted": "#54A24B",
}


def read_csv(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            parsed = {}
            for key, value in row.items():
                if key == "scenario":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        smoothed.append(sum(values[start : idx + 1]) / (idx - start + 1))
    return smoothed


def last_n_average(rows: list[dict[str, float]], metric: str, n_steps: int) -> float:
    tail = rows[-n_steps:]
    return sum(row[metric] for row in tail) / len(tail)


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def line_chart(
    data: dict[str, list[dict[str, float]]],
    metric: str,
    x: int,
    y: int,
    width: int,
    height: int,
    window: int,
) -> list[str]:
    pad_left, pad_right, pad_top, pad_bottom = 48, 12, 30, 35
    plot_x = x + pad_left
    plot_y = y + pad_top
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    series_values = {
        scenario: moving_average([row[metric] for row in rows], window)
        for scenario, rows in data.items()
    }
    all_values = [value for values in series_values.values() for value in values]
    min_v, max_v = min(all_values), max(all_values)
    if abs(max_v - min_v) < 1e-12:
        max_v = min_v + 1.0
    margin = 0.08 * (max_v - min_v)
    min_v -= margin
    max_v += margin

    max_step = max(len(rows) for rows in data.values()) - 1

    def sx(step: int) -> float:
        return plot_x + (step / max_step) * plot_w if max_step > 0 else plot_x

    def sy(value: float) -> float:
        return plot_y + plot_h - ((value - min_v) / (max_v - min_v)) * plot_h

    elements = [
        f'<text x="{x}" y="{y + 18}" class="title">{esc(METRICS[metric])}</text>',
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#cccccc"/>',
    ]

    for tick in range(4):
        frac = tick / 3
        ty = plot_y + plot_h - frac * plot_h
        value = min_v + frac * (max_v - min_v)
        elements.append(f'<line x1="{plot_x}" y1="{ty:.1f}" x2="{plot_x + plot_w}" y2="{ty:.1f}" stroke="#eeeeee"/>')
        elements.append(f'<text x="{plot_x - 8}" y="{ty + 4:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')

    elements.append(f'<text x="{plot_x}" y="{plot_y + plot_h + 24}" class="tick">0</text>')
    elements.append(f'<text x="{plot_x + plot_w}" y="{plot_y + plot_h + 24}" text-anchor="end" class="tick">{max_step}</text>')

    for scenario, rows in data.items():
        values = series_values[scenario]
        points = " ".join(f"{sx(idx):.1f},{sy(value):.1f}" for idx, value in enumerate(values))
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS[scenario]}" stroke-width="2.2"/>'
        )

    return elements


def create_time_series_svg(data: dict[str, list[dict[str, float]]], output_path: Path, window: int) -> None:
    width, height = 1400, 820
    chart_w, chart_h = 430, 330
    start_x, start_y = 45, 90
    gap_x, gap_y = 35, 45

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial, sans-serif; fill:#222}",
        ".title{font-size:16px; font-weight:700}",
        ".tick{font-size:11px; fill:#666}",
        ".main{font-size:24px; font-weight:700}",
        "</style>",
        f'<text x="45" y="42" class="main">Scenario comparison over time ({window}-step moving average)</text>',
    ]

    legend_x = 920
    for idx, scenario in enumerate(data):
        lx = legend_x + idx * 145
        elements.append(f'<line x1="{lx}" y1="38" x2="{lx + 30}" y2="38" stroke="{COLORS[scenario]}" stroke-width="4"/>')
        elements.append(f'<text x="{lx + 38}" y="42" class="tick">{esc(scenario)}</text>')

    for idx, metric in enumerate(METRICS):
        row, col = divmod(idx, 3)
        x = start_x + col * (chart_w + gap_x)
        y = start_y + row * (chart_h + gap_y)
        elements.extend(line_chart(data, metric, x, y, chart_w, chart_h, window))

    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def create_summary_svg(data: dict[str, list[dict[str, float]]], output_path: Path, n_steps: int) -> None:
    width, height = 1400, 820
    chart_w, chart_h = 430, 330
    start_x, start_y = 45, 90
    gap_x, gap_y = 35, 45
    scenarios = list(data.keys())

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial, sans-serif; fill:#222}",
        ".title{font-size:16px; font-weight:700}",
        ".tick{font-size:11px; fill:#666}",
        ".main{font-size:24px; font-weight:700}",
        "</style>",
        f'<text x="45" y="42" class="main">Scenario comparison using last {n_steps} time steps</text>',
    ]

    for idx, (metric, title) in enumerate(METRICS.items()):
        row, col = divmod(idx, 3)
        x = start_x + col * (chart_w + gap_x)
        y = start_y + row * (chart_h + gap_y)
        pad_left, pad_right, pad_top, pad_bottom = 48, 12, 30, 55
        plot_x = x + pad_left
        plot_y = y + pad_top
        plot_w = chart_w - pad_left - pad_right
        plot_h = chart_h - pad_top - pad_bottom
        values = [last_n_average(data[scenario], metric, n_steps) for scenario in scenarios]
        max_v = max(values) * 1.18 if max(values) > 0 else 1.0

        elements.append(f'<text x="{x}" y="{y + 18}" class="title">{esc(title)}</text>')
        elements.append(f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#cccccc"/>')
        for tick in range(4):
            frac = tick / 3
            ty = plot_y + plot_h - frac * plot_h
            value = frac * max_v
            elements.append(f'<line x1="{plot_x}" y1="{ty:.1f}" x2="{plot_x + plot_w}" y2="{ty:.1f}" stroke="#eeeeee"/>')
            elements.append(f'<text x="{plot_x - 8}" y="{ty + 4:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')

        bar_gap = 22
        bar_w = (plot_w - bar_gap * (len(scenarios) + 1)) / len(scenarios)
        for s_idx, (scenario, value) in enumerate(zip(scenarios, values)):
            bx = plot_x + bar_gap + s_idx * (bar_w + bar_gap)
            bh = (value / max_v) * plot_h
            by = plot_y + plot_h - bh
            elements.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{COLORS[scenario]}"/>')
            elements.append(f'<text x="{bx + bar_w/2:.1f}" y="{by - 5:.1f}" text-anchor="middle" class="tick">{value:.3f}</text>')
            elements.append(
                f'<text x="{bx + bar_w/2:.1f}" y="{plot_y + plot_h + 18}" text-anchor="middle" class="tick">{esc(scenario)}</text>'
            )

    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dependency-free SVG plots for simulation results.")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--summary-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    data = {scenario: read_csv(Path(path)) for scenario, path in SCENARIO_FILES.items()}
    time_series_path = args.output_dir / "scenario_time_series.svg"
    summary_path = args.output_dir / "scenario_summary_bars.svg"

    create_time_series_svg(data, time_series_path, args.window)
    create_summary_svg(data, summary_path, args.summary_steps)

    print(f"Saved time-series plot to: {time_series_path}")
    print(f"Saved summary bar plot to: {summary_path}")


if __name__ == "__main__":
    main()
