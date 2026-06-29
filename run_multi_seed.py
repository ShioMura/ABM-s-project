from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCENARIOS = ["baseline", "demand_only", "supply_adjusted"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run each scenario over multiple random seeds.")
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds to run, starting from 1.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/seeds"))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(1, args.seeds + 1):
        for scenario in SCENARIOS:
            output = args.output_dir / f"{scenario}_seed{seed}.csv"
            if args.skip_existing and output.exists():
                print(f"Skipping existing file: {output}")
                continue
            cmd = [
                args.python_executable,
                "baseline_simulation.py",
                "--scenario",
                scenario,
                "--seed",
                str(seed),
                "--output",
                str(output),
            ]
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
