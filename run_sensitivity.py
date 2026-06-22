from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


SCENARIOS = ["baseline", "demand_only", "supply_adjusted"]
DEFAULT_RIDER_VALUES = [150, 175, 200, 225, 250]
DEFAULT_BETA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]


def beta_label(beta: float) -> str:
    """Turn 0.5 into beta_0_5, avoiding dots in folder names."""
    return f"beta_{beta:g}".replace(".", "_").replace("-", "minus_")


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("Running:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def seed_range(n_seeds: int, seed_start: int) -> list[int]:
    return list(range(seed_start, seed_start + n_seeds))


def run_rider_supply_sensitivity(
    python_executable: str,
    simulation_script: Path,
    output_root: Path,
    rider_values: Iterable[int],
    seeds: Iterable[int],
    skip_existing: bool,
    dry_run: bool,
) -> None:
    """Vary n_riders and compare all three information scenarios."""
    for n_riders in rider_values:
        for scenario in SCENARIOS:
            for seed in seeds:
                output_path = (
                    output_root
                    / "riders"
                    / "raw"
                    / f"riders_{n_riders}"
                    / f"{scenario}_seed{seed}.csv"
                )
                if skip_existing and output_path.exists():
                    print(f"Skipping existing file: {output_path}")
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    python_executable,
                    str(simulation_script),
                    "--scenario",
                    scenario,
                    "--riders",
                    str(n_riders),
                    "--seed",
                    str(seed),
                    "--output",
                    str(output_path),
                ]
                run_command(cmd, dry_run=dry_run)


def run_heatmap_beta_sensitivity(
    python_executable: str,
    simulation_script: Path,
    output_root: Path,
    beta_values: Iterable[float],
    seeds: Iterable[int],
    skip_existing: bool,
    dry_run: bool,
) -> None:
    """Vary heatmap_beta in the supply-adjusted scenario only."""
    scenario = "supply_adjusted"
    for beta in beta_values:
        for seed in seeds:
            output_path = (
                output_root
                / "heatmap_beta"
                / "raw"
                / beta_label(beta)
                / f"{scenario}_seed{seed}.csv"
            )
            if skip_existing and output_path.exists():
                print(f"Skipping existing file: {output_path}")
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                python_executable,
                str(simulation_script),
                "--scenario",
                scenario,
                "--heatmap-beta",
                str(beta),
                "--seed",
                str(seed),
                "--output",
                str(output_path),
            ]
            run_command(cmd, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local sensitivity analysis for the delivery-rider ABM."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["riders", "heatmap_beta", "all"],
        default=["all"],
        help="Which sensitivity experiment(s) to run.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of random seeds. Default is 10; use 5 if runtime is tight.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="First seed value. With --n-seeds 10 and --seed-start 1, seeds are 1..10.",
    )
    parser.add_argument(
        "--rider-values",
        type=int,
        nargs="+",
        default=DEFAULT_RIDER_VALUES,
        help="Values for rider-supply sensitivity.",
    )
    parser.add_argument(
        "--beta-values",
        type=float,
        nargs="+",
        default=DEFAULT_BETA_VALUES,
        help="Values for heatmap_beta sensitivity.",
    )
    parser.add_argument(
        "--simulation-script",
        type=Path,
        default=Path("baseline_simulation.py"),
        help="Path to the main simulation script.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/sensitivity"),
        help="Root folder for sensitivity outputs.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch baseline_simulation.py.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose output CSV already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    experiments = set(args.experiments)
    if "all" in experiments:
        experiments = {"riders", "heatmap_beta"}

    seeds = seed_range(args.n_seeds, args.seed_start)
    print(f"Seeds: {seeds}")
    print(f"Output root: {args.output_root}")

    if "riders" in experiments:
        run_rider_supply_sensitivity(
            python_executable=args.python_executable,
            simulation_script=args.simulation_script,
            output_root=args.output_root,
            rider_values=args.rider_values,
            seeds=seeds,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )

    if "heatmap_beta" in experiments:
        run_heatmap_beta_sensitivity(
            python_executable=args.python_executable,
            simulation_script=args.simulation_script,
            output_root=args.output_root,
            beta_values=args.beta_values,
            seeds=seeds,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )

    print("Sensitivity runs complete.")


if __name__ == "__main__":
    main()
