from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


SCENARIOS = ["baseline", "demand_only", "supply_adjusted"]
METRICS = [
    "avg_income",
    "idle_rate",
    "matching_rate",
    "movement_rate",
    "congestion_frequency",
    "rider_concentration_hhi",
    "rider_density_cv",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def last_n_average(rows: list[dict[str, str]], metric: str, n_steps: int) -> float:
    tail = rows[-n_steps:]
    return mean(float(row[metric]) for row in tail)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize multi-seed simulation results.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/seeds"))
    parser.add_argument("--output", type=Path, default=Path("outputs/seeds/multi_seed_summary.csv"))
    parser.add_argument("--summary-steps", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 31)))
    args = parser.parse_args()

    summary_rows = []

    for scenario in SCENARIOS:
        metric_values = {metric: [] for metric in METRICS}

        for seed in args.seeds:
            path = args.input_dir / f"{scenario}_seed{seed}.csv"
            rows = read_rows(path)
            for metric in METRICS:
                metric_values[metric].append(last_n_average(rows, metric, args.summary_steps))

        row = {"scenario": scenario}
        for metric, values in metric_values.items():
            row[f"{metric}_mean"] = mean(values)
            metric_std = stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_std"] = metric_std
            row[f"{metric}_ci95"] = 1.96 * metric_std / (len(values) ** 0.5)
        summary_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as csv_file:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved multi-seed summary to: {args.output}")
    for row in summary_rows:
        print(row["scenario"])
        for metric in METRICS:
            print(
                f"  {metric}: "
                f"{row[f'{metric}_mean']:.4f} "
                f"± {row[f'{metric}_ci95']:.4f} 95% CI "
                f"(std={row[f'{metric}_std']:.4f})"
            )


if __name__ == "__main__":
    main()
