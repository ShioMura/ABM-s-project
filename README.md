# Heatmaps and Herding in Food Delivery Platforms

This repository contains an agent-based model of food-delivery couriers in a
10 by 10 grid. The model is El Farol-inspired: riders repeatedly decide whether
to stay or move under limited information, and their payoff depends on how many
other riders choose the same zones.

The project compares three scenarios:

- `baseline`: no public platform heatmap.
- `demand_only`: the platform publishes a heatmap based on incoming demand.
- `supply_adjusted`: the heatmap includes demand, queued orders, and a penalty
  for zones that already contain many riders.

## Requirements

- Python 3.10 or newer
- `numpy`
- `pandas`
- `matplotlib` for sensitivity-analysis PDF plots
- `SALib` for Sobol sensitivity analysis

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

On Windows, `py` can be used instead of `python` if that is how Python is
installed:

```powershell
py -m pip install -r requirements.txt
```

## Reproduce the Main Results

From the repository root, run:

```bash
python run_reproduction.py
```

This runs:

1. one trajectory for each scenario,
2. the main SVG figures,
3. 30 random seeds for each scenario,
4. the multi-seed summary with means and 95 percent confidence intervals.

For a fast smoke test, use:

```bash
python run_reproduction.py --quick
```

To also smoke-test the Sobol workflow:

```bash
python run_reproduction.py --quick --include-sobol-smoke-test
```

The main outputs are written to:

- `outputs/baseline.csv`
- `outputs/demand_only.csv`
- `outputs/supply_adjusted.csv`
- `outputs/figures/scenario_time_series.svg`
- `outputs/figures/scenario_summary_bars.svg`
- `outputs/seeds/multi_seed_summary.csv`

## Run Individual Experiments

Run one scenario:

```bash
python baseline_simulation.py --scenario baseline --output outputs/baseline.csv
python baseline_simulation.py --scenario demand_only --output outputs/demand_only.csv
python baseline_simulation.py --scenario supply_adjusted --output outputs/supply_adjusted.csv
```

Create main figures:

```bash
python plot_results_svg.py
```

Run 30 seeds per scenario:

```bash
python run_multi_seed.py --seeds 30
python summarize_multi_seed.py
```

## Sensitivity Analysis

The sensitivity analysis is optional because it requires many simulation runs.
To run the full configured sensitivity workflow:

```bash
python run_reproduction.py --include-sensitivity
```

Or run it manually:

```bash
python run_sensitivity.py --experiments all --n-seeds 10 --global-samples 20
python summarize_sensitivity.py
python plot_sensitivity_pdf.py
```

Sensitivity outputs are written to `outputs/sensitivity/`.

## Sobol Sensitivity Analysis

The Sobol analysis tests the global sensitivity of the `supply_adjusted`
scenario to three heatmap parameters:

- `heatmap_alpha`: weight on incoming demand
- `heatmap_beta`: penalty for current rider density
- `heatmap_gamma`: weight on queued unmet demand

The Sobol script requires `SALib`, `pandas`, `numpy`, and `matplotlib`, all of
which are listed in `requirements.txt`.

Fast smoke test:

```bash
python run_sobol_sensitivity.py --base-samples 4 --n-reps 1 --steps 20 --workers 1 --output-root outputs/sobol_test
```

This command checks that the Sobol workflow runs, but it is not intended for
reporting.

Full Sobol run used for final analysis:

```bash
python run_sobol_sensitivity.py --base-samples 512 --n-reps 10 --workers 1 --output-root outputs/sobol_3param
```

For faster execution on a multi-core machine, increase `--workers`, for example:

```bash
python run_sobol_sensitivity.py --base-samples 512 --n-reps 10 --workers 4 --output-root outputs/sobol_3param
```

The default full run evaluates `512 * (3 + 2) * 10 = 25,600` ABM replications,
so it can take a long time. If interrupted, re-running the same command resumes
from the existing `outputs/sobol_3param/sobol_raw_runs.csv` file.

Sobol outputs are written to:

- `outputs/sobol_3param/sobol_parameter_values.csv`
- `outputs/sobol_3param/sobol_raw_runs.csv`
- `outputs/sobol_3param/sobol_mean_outputs.csv`
- `outputs/sobol_3param/sobol_indices.csv`
- `outputs/sobol_3param/plots/`

## Main Output Metrics

- `avg_income`: mean rider payoff per time step.
- `idle_rate`: share of riders not matched to an order.
- `matching_rate`: share of available orders that are matched.
- `movement_rate`: share of riders that move.
- `congestion_frequency`: share of grid cells with more riders than available
  orders.
- `rider_density_cv`: coefficient of variation of rider density across grid
  cells. Higher values indicate stronger spatial concentration and herding.
- `rider_concentration_hhi`: Herfindahl-Hirschman Index of rider concentration,
  retained as a secondary concentration metric.

## Model Defaults

- Grid size: 10 by 10
- Riders: 200
- Time steps: 144
- Time step interpretation: 10 minutes
- Order arrival process: Poisson with default lambda 1.5 per grid cell
- Maximum order waiting time: 3 time steps
- Movement cost: 1.0
- Fare per matched order: 10.0
- Strategy memory length: 3
- Memory decay: 0.7

## Repository Structure

- `baseline_simulation.py`: core ABM simulation.
- `run_reproduction.py`: recommended one-command reproduction workflow.
- `run_multi_seed.py`: runs all scenarios across multiple random seeds.
- `summarize_multi_seed.py`: aggregates multi-seed outputs.
- `plot_results_svg.py`: creates dependency-free SVG figures for main results.
- `run_sensitivity.py`: local and global sensitivity-analysis runs.
- `summarize_sensitivity.py`: aggregates sensitivity-analysis outputs.
- `plot_sensitivity_pdf.py`: creates sensitivity-analysis PDF figures.
- `run_sobol_sensitivity.py`: Sobol global sensitivity analysis.
