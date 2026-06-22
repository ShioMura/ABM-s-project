from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev


METRICS = [
    "avg_income",
    "idle_rate",
    "matching_rate",
    "movement_rate",
    "congestion_frequency",
    "rider_concentration_hhi",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def last_n_mean(rows: list[dict[str, str]], metric: str, n_steps: int) -> float:
    tail = rows[-n_steps:]
    if not tail:
        raise ValueError("Cannot summarize an empty CSV file.")
    return mean(float(row[metric]) for row in tail)


def parse_scenario_and_seed(path: Path) -> tuple[str, int]:
    """Parse names like baseline_seed1.csv or supply_adjusted_seed10.csv."""
    stem = path.stem
    if "_seed" not in stem:
        raise ValueError(f"Could not parse scenario/seed from filename: {path.name}")
    scenario, seed_text = stem.rsplit("_seed", 1)
    return scenario, int(seed_text)


def parse_beta_label(folder_name: str) -> float:
    """Parse beta_0_5 into 0.5."""
    if not folder_name.startswith("beta_"):
        raise ValueError(f"Unexpected beta folder name: {folder_name}")
    value_text = folder_name.removeprefix("beta_").replace("minus_", "-").replace("_", ".")
    return float(value_text)


def summarize_single_run(
    path: Path,
    experiment: str,
    parameter_name: str,
    parameter_value: float,
    n_steps: int,
) -> dict[str, str | float | int]:
    rows = read_rows(path)
    scenario, seed = parse_scenario_and_seed(path)
    result: dict[str, str | float | int] = {
        "experiment": experiment,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "scenario": scenario,
        "seed": seed,
        "source_file": str(path),
    }
    for metric in METRICS:
        result[metric] = last_n_mean(rows, metric, n_steps)
    return result


def collect_rider_runs(output_root: Path, n_steps: int) -> list[dict[str, str | float | int]]:
    runs = []
    raw_root = output_root / "riders" / "raw"
    for folder in sorted(raw_root.glob("riders_*")):
        if not folder.is_dir():
            continue
        n_riders = int(folder.name.removeprefix("riders_"))
        for csv_path in sorted(folder.glob("*_seed*.csv")):
            runs.append(
                summarize_single_run(
                    path=csv_path,
                    experiment="riders",
                    parameter_name="n_riders",
                    parameter_value=n_riders,
                    n_steps=n_steps,
                )
            )
    return runs


def collect_beta_runs(output_root: Path, n_steps: int) -> list[dict[str, str | float | int]]:
    runs = []
    raw_root = output_root / "heatmap_beta" / "raw"
    for folder in sorted(raw_root.glob("beta_*")):
        if not folder.is_dir():
            continue
        beta = parse_beta_label(folder.name)
        for csv_path in sorted(folder.glob("*_seed*.csv")):
            runs.append(
                summarize_single_run(
                    path=csv_path,
                    experiment="heatmap_beta",
                    parameter_name="heatmap_beta",
                    parameter_value=beta,
                    n_steps=n_steps,
                )
            )
    return runs


def aggregate_runs(runs: list[dict[str, str | float | int]]) -> list[dict[str, str | float | int]]:
    grouped: dict[tuple[str, str, float, str], list[dict[str, str | float | int]]] = {}
    for row in runs:
        key = (
            str(row["experiment"]),
            str(row["parameter_name"]),
            float(row["parameter_value"]),
            str(row["scenario"]),
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (experiment, parameter_name, parameter_value, scenario), rows in sorted(grouped.items()):
        summary: dict[str, str | float | int] = {
            "experiment": experiment,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "scenario": scenario,
            "n_runs": len(rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            metric_mean = mean(values)
            metric_sd = stdev(values) if len(values) > 1 else 0.0
            metric_se = metric_sd / math.sqrt(len(values)) if len(values) > 0 else 0.0
            metric_ci95 = 1.96 * metric_se
            summary[f"{metric}_mean"] = metric_mean
            summary[f"{metric}_sd"] = metric_sd
            summary[f"{metric}_se"] = metric_se
            summary[f"{metric}_ci95"] = metric_ci95
        summaries.append(summary)
    return summaries


def write_csv(rows: list[dict[str, str | float | int]], path: Path) -> None:
    if not rows:
        print(f"No rows to write for {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize sensitivity-analysis CSV files using the final N simulation steps."
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sensitivity"))
    parser.add_argument("--summary-steps", type=int, default=20)
    args = parser.parse_args()

    rider_runs = collect_rider_runs(args.output_root, args.summary_steps)
    beta_runs = collect_beta_runs(args.output_root, args.summary_steps)
    all_runs = rider_runs + beta_runs

    rider_summary = aggregate_runs(rider_runs)
    beta_summary = aggregate_runs(beta_runs)
    all_summary = aggregate_runs(all_runs)

    suffix = f"last{args.summary_steps}"
    write_csv(rider_runs, args.output_root / "riders" / f"riders_run_summary_{suffix}.csv")
    write_csv(rider_summary, args.output_root / "riders" / f"riders_group_summary_{suffix}.csv")
    write_csv(beta_runs, args.output_root / "heatmap_beta" / f"beta_run_summary_{suffix}.csv")
    write_csv(beta_summary, args.output_root / "heatmap_beta" / f"beta_group_summary_{suffix}.csv")
    write_csv(all_runs, args.output_root / f"all_run_summary_{suffix}.csv")
    write_csv(all_summary, args.output_root / f"all_group_summary_{suffix}.csv")

    print("Sensitivity summaries complete.")


if __name__ == "__main__":
    main()
