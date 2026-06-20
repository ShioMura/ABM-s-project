import csv
from statistics import mean

files = {
    "baseline": "outputs/baseline.csv",
    "demand_only": "outputs/demand_only.csv",
    "supply_adjusted": "outputs/supply_adjusted.csv",
}

keys = [
    "avg_income",
    "idle_rate",
    "matching_rate",
    "movement_rate",
    "congestion_frequency",
    "rider_concentration_hhi",
]

for name, path in files.items():
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    last = rows[-20:]
    print(name)
    for key in keys:
        print(f"  {key}: {mean(float(r[key]) for r in last):.4f}")