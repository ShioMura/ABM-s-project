from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCENARIOS = ["baseline", "demand_only", "supply_adjusted"]


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the main simulation outputs, figures, and multi-seed summary."
    )
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="Use 3 seeds for a fast smoke test.")
    parser.add_argument("--include-sensitivity", action="store_true")
    parser.add_argument("--include-sobol-smoke-test", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    n_seeds = 3 if args.quick else args.seeds

    for scenario in SCENARIOS:
        output = Path("outputs") / f"{scenario}.csv"
        if args.skip_existing and output.exists():
            print(f"Skipping existing file: {output}")
            continue
        run([
            python,
            "baseline_simulation.py",
            "--scenario",
            scenario,
            "--output",
            str(output),
        ])

    run([python, "plot_results_svg.py"])

    run([
        python,
        "run_multi_seed.py",
        "--seeds",
        str(n_seeds),
        "--python-executable",
        python,
        *(["--skip-existing"] if args.skip_existing else []),
    ])

    seed_values = [str(seed) for seed in range(1, n_seeds + 1)]
    run([python, "summarize_multi_seed.py", "--seeds", *seed_values])

    if args.include_sensitivity:
        sensitivity_seeds = "3" if args.quick else "10"
        global_samples = "5" if args.quick else "20"
        run([
            python,
            "run_sensitivity.py",
            "--n-seeds",
            sensitivity_seeds,
            "--global-samples",
            global_samples,
            "--python-executable",
            python,
            *(["--skip-existing"] if args.skip_existing else []),
        ])
        run([python, "summarize_sensitivity.py"])
        run([python, "plot_sensitivity_pdf.py"])

    if args.include_sobol_smoke_test:
        run([
            python,
            "run_sobol_sensitivity.py",
            "--base-samples",
            "4",
            "--n-reps",
            "1",
            "--steps",
            "20",
            "--workers",
            "1",
            "--output-root",
            "outputs/sobol_test",
        ])

    print("Reproduction workflow complete.")


if __name__ == "__main__":
    main()
