from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SCENARIOS = ["baseline", "demand_only", "supply_adjusted"]
DEFAULT_RIDER_VALUES = [150, 175, 200, 225, 250]
DEFAULT_BETA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]

def beta_label(beta: float) -> str:
    return f"beta_{beta:g}".replace(".", "_").replace("-", "minus_")

def run_command(cmd: list[str], dry_run: bool) -> None:
    print("Running:", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)

def seed_range(n_seeds: int, seed_start: int) -> list[int]:
    return list(range(seed_start, seed_start + n_seeds))

def run_rider_supply_sensitivity(
    python_executable: str, simulation_script: Path, output_root: Path,
    rider_values: Iterable[int], seeds: Iterable[int], skip_existing: bool, dry_run: bool
) -> None:
    for n_riders in rider_values:
        for scenario in SCENARIOS:
            for seed in seeds:
                output_path = output_root / "riders" / "raw" / f"riders_{n_riders}" / f"{scenario}_seed{seed}.csv"
                if skip_existing and output_path.exists():
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = [python_executable, str(simulation_script), "--scenario", scenario,
                       "--riders", str(n_riders), "--seed", str(seed), "--output", str(output_path)]
                run_command(cmd, dry_run=dry_run)

def run_heatmap_beta_sensitivity(
    python_executable: str, simulation_script: Path, output_root: Path,
    beta_values: Iterable[float], seeds: Iterable[int], skip_existing: bool, dry_run: bool
) -> None:
    scenario = "supply_adjusted"
    for beta in beta_values:
        for seed in seeds:
            output_path = output_root / "heatmap_beta" / "raw" / beta_label(beta) / f"{scenario}_seed{seed}.csv"
            if skip_existing and output_path.exists():
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [python_executable, str(simulation_script), "--scenario", scenario,
                   "--heatmap-beta", str(beta), "--seed", str(seed), "--output", str(output_path)]
            run_command(cmd, dry_run=dry_run)

def run_global_sensitivity(
    python_executable: str, simulation_script: Path, output_root: Path,
    n_samples: int, seeds: Iterable[int], sample_seed: int,
    skip_existing: bool, dry_run: bool
) -> None:
    """Global sensitivity analysis with Monte Carlo random parameter sampling."""
    scenario = "supply_adjusted"
    global_dir = output_root / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    
    params_log_path = global_dir / "global_params_log.csv"
    samples = []
    
    if skip_existing and params_log_path.exists():
        with open(params_log_path, "r") as f:
            reader = csv.DictReader(f)
            samples = [row for row in reader]
    else:
        rng = random.Random(sample_seed)
        for i in range(n_samples):
            samples.append({
                "sample_id": f"sample_{i:03d}",
                "riders": rng.randint(100, 300),
                "heatmap_alpha": round(rng.uniform(0.0, 2.0), 2),
                "heatmap_beta": round(rng.uniform(0.0, 2.0), 2),
                "heatmap_gamma": round(rng.uniform(0.0, 2.0), 2),
            })
        if not dry_run:
            with open(params_log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=samples[0].keys())
                writer.writeheader()
                writer.writerows(samples)

    for sample in samples:
        sample_id = sample["sample_id"]
        for seed in seeds:
            output_path = global_dir / "raw" / sample_id / f"{scenario}_seed{seed}.csv"
            if skip_existing and output_path.exists():
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                python_executable, str(simulation_script),
                "--scenario", scenario,
                "--riders", str(sample["riders"]),
                "--heatmap-alpha", str(sample["heatmap_alpha"]),
                "--heatmap-beta", str(sample["heatmap_beta"]),
                "--heatmap-gamma", str(sample["heatmap_gamma"]),
                "--seed", str(seed),
                "--output", str(output_path)
            ]
            run_command(cmd, dry_run=dry_run)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run local sensitivity analysis for the delivery-rider ABM.")
    parser.add_argument("--experiments", nargs="+", choices=["riders", "heatmap_beta", "global", "all"], default=["all"])
    parser.add_argument("--n-seeds", type=int, default=30, help="Number of random seeds. Changed to 30 per feedback.")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--rider-values", type=int, nargs="+", default=DEFAULT_RIDER_VALUES)
    parser.add_argument("--beta-values", type=float, nargs="+", default=DEFAULT_BETA_VALUES)
    parser.add_argument("--global-samples", type=int, default=20, help="Number of random parameter combinations for Global SA.")
    parser.add_argument("--sample-seed", type=int, default=12345, help="Seed used to generate global sensitivity parameter samples.")
    parser.add_argument("--simulation-script", type=Path, default=Path("baseline_simulation.py"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sensitivity"))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiments = set(args.experiments)
    if "all" in experiments:
        experiments = {"riders", "heatmap_beta", "global"}

    seeds = seed_range(args.n_seeds, args.seed_start)
    print(f"Seeds: {seeds}")

    if "riders" in experiments:
        run_rider_supply_sensitivity(args.python_executable, args.simulation_script, args.output_root, args.rider_values, seeds, args.skip_existing, args.dry_run)
    if "heatmap_beta" in experiments:
        run_heatmap_beta_sensitivity(args.python_executable, args.simulation_script, args.output_root, args.beta_values, seeds, args.skip_existing, args.dry_run)
    if "global" in experiments:
        run_global_sensitivity(args.python_executable, args.simulation_script, args.output_root, args.global_samples, seeds, args.sample_seed, args.skip_existing, args.dry_run)

    print("Sensitivity runs complete.")

if __name__ == "__main__":
    main()
