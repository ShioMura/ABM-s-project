import subprocess
from pathlib import Path

scenarios = ["baseline", "demand_only", "supply_adjusted"]
seeds = [1, 2, 3, 4, 5]

Path("outputs/seeds").mkdir(parents=True, exist_ok=True)

for seed in seeds:
    for scenario in scenarios:
        output = f"outputs/seeds/{scenario}_seed{seed}.csv"
        cmd = [
            "py",
            "baseline_simulation.py",
            "--scenario", scenario,
            "--seed", str(seed),
            "--output", output,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)