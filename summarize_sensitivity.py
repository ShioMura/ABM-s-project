from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev

# 将拥堵和集中度指标加回列表
METRICS = [
    "avg_income",
    "idle_rate",
    "matching_rate",
    "movement_rate",
    "rider_density_cv",
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
    stem = path.stem
    scenario, seed_text = stem.rsplit("_seed", 1)
    return scenario, int(seed_text)

def parse_beta_label(folder_name: str) -> float:
    value_text = folder_name.removeprefix("beta_").replace("minus_", "-").replace("_", ".")
    return float(value_text)

def summarize_single_run(
    path: Path, experiment: str, parameter_name: str, parameter_value: float, n_steps: int
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
        if not folder.is_dir(): continue
        n_riders = int(folder.name.removeprefix("riders_"))
        for csv_path in sorted(folder.glob("*_seed*.csv")):
            runs.append(summarize_single_run(csv_path, "riders", "n_riders", n_riders, n_steps))
    return runs

def collect_beta_runs(output_root: Path, n_steps: int) -> list[dict[str, str | float | int]]:
    runs = []
    raw_root = output_root / "heatmap_beta" / "raw"
    for folder in sorted(raw_root.glob("beta_*")):
        if not folder.is_dir(): continue
        beta = parse_beta_label(folder.name)
        for csv_path in sorted(folder.glob("*_seed*.csv")):
            runs.append(summarize_single_run(csv_path, "heatmap_beta", "heatmap_beta", beta, n_steps))
    return runs

def collect_global_runs(output_root: Path, n_steps: int) -> list[dict[str, str | float | int]]:
    runs = []
    global_dir = output_root / "global"
    params_log = global_dir / "global_params_log.csv"
    if not params_log.exists():
        return runs
        
    param_map = {}
    with open(params_log, "r") as f:
        for row in csv.DictReader(f):
            param_map[row["sample_id"]] = row

    raw_root = global_dir / "raw"
    for folder in sorted(raw_root.glob("sample_*")):
        if not folder.is_dir(): continue
        sample_id = folder.name
        if sample_id not in param_map: continue
        
        for csv_path in sorted(folder.glob("*_seed*.csv")):
            scenario, seed = parse_scenario_and_seed(csv_path)
            row_data = read_rows(csv_path)
            res = {
                "experiment": "global",
                "sample_id": sample_id,
                "scenario": scenario,
                "seed": seed,
            }
            for k, v in param_map[sample_id].items():
                if k != "sample_id":
                    res[k] = float(v)
            for metric in METRICS:
                res[metric] = last_n_mean(row_data, metric, n_steps)
            runs.append(res)
    return runs

def aggregate_runs(runs: list[dict[str, str | float | int]]) -> list[dict[str, str | float | int]]:
    grouped: dict[tuple, list[dict]] = {}
    for row in runs:
        key = (str(row["experiment"]), str(row.get("parameter_name", "N/A")), float(row.get("parameter_value", 0.0)), str(row["scenario"]))
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (experiment, parameter_name, parameter_value, scenario), rows_group in sorted(grouped.items()):
        summary: dict[str, str | float | int] = {
            "experiment": experiment,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "scenario": scenario,
            "n_runs": len(rows_group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows_group]
            metric_mean = mean(values)
            metric_sd = stdev(values) if len(values) > 1 else 0.0
            metric_se = metric_sd / math.sqrt(len(values)) if len(values) > 0 else 0.0
            summary[f"{metric}_mean"] = metric_mean
            summary[f"{metric}_ci95"] = 1.96 * metric_se
        summaries.append(summary)
    return summaries

def aggregate_global_runs(runs: list[dict]) -> list[dict]:
    grouped = {}
    for r in runs:
        grouped.setdefault(r["sample_id"], []).append(r)
    
    summaries = []
    for sid, r_list in grouped.items():
        summary = {"sample_id": sid, "scenario": r_list[0]["scenario"]}
        for k in r_list[0].keys():
            if k not in ["experiment", "sample_id", "scenario", "seed", "source_file"] and k not in METRICS:
                summary[k] = r_list[0][k]
        for metric in METRICS:
            vals = [float(x[metric]) for x in r_list]
            summary[f"{metric}_mean"] = mean(vals)
        summaries.append(summary)
    return summaries

def write_csv(rows: list[dict], path: Path) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sensitivity"))
    parser.add_argument("--summary-steps", type=int, default=20)
    args = parser.parse_args()

    rider_runs = collect_rider_runs(args.output_root, args.summary_steps)
    beta_runs = collect_beta_runs(args.output_root, args.summary_steps)
    global_runs = collect_global_runs(args.output_root, args.summary_steps)

    suffix = f"last{args.summary_steps}"
    
    if rider_runs:
        write_csv(aggregate_runs(rider_runs), args.output_root / "riders" / f"riders_group_summary_{suffix}.csv")
    if beta_runs:
        write_csv(aggregate_runs(beta_runs), args.output_root / "heatmap_beta" / f"beta_group_summary_{suffix}.csv")
    if global_runs:
        write_csv(aggregate_global_runs(global_runs), args.output_root / "global" / f"global_group_summary_{suffix}.csv")

    print("Sensitivity summaries complete.")

if __name__ == "__main__":
    main()
