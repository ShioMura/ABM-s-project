from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SALib.analyze import sobol as sobol_analyze

try:
    # Newer SALib API
    from SALib.sample import sobol as sobol_sampler
    USE_NEW_SAMPLER = True
except Exception:
    # Older SALib API, useful if the course environment has old SALib
    from SALib.sample import saltelli
    USE_NEW_SAMPLER = False

from baseline_simulation import run_simulation


# Three-parameter Sobol analysis:
# alpha is fixed at 1.0 as the reference demand weight.

PROBLEM = {
    "num_vars": 3,
    "names": ["heatmap_alpha", "heatmap_beta", "heatmap_gamma"],
    "bounds": [
        [0.0, 2.0],
        [0.0, 2.0],
        [0.0, 2.0],
    ],
}


OUTCOMES = [
    "avg_income",
    "congestion_frequency",
    "rider_density_cv",
]


def generate_sobol_samples(base_samples: int, sampling_seed: int | None = 123) -> np.ndarray:
    """
    Generate N * (D + 2) samples when calc_second_order=False.
    For D=3 and N=512, this gives 512 * 5 = 2560 parameter combinations.
    """
    if USE_NEW_SAMPLER:
        return sobol_sampler.sample(
            PROBLEM,
            base_samples,
            calc_second_order=False,
            scramble=True,
            seed=sampling_seed,
        )

    # Older SALib fallback.
    # This is deprecated in newer SALib, but works in older course environments.
    return saltelli.sample(
        PROBLEM,
        base_samples,
        calc_second_order=False,
    )


def summarize_final_steps(metrics: list[dict[str, float]], final_steps: int) -> dict[str, float]:
    """Average outcomes over the final time steps to reduce transient effects."""
    tail = metrics[-final_steps:]

    return {
        outcome: float(np.mean([row[outcome] for row in tail]))
        for outcome in OUTCOMES
    }


def run_one(job: tuple[int, int, int, float, float, float, int, int]) -> dict[str, float]:
    """
    Run one ABM replication for one Sobol parameter combination.

    job fields:
    sample_id, replicate, seed, alpha, beta, gamma, final_steps, steps
    """
    sample_id, replicate, seed, alpha, beta, gamma, final_steps, steps = job

    metrics = run_simulation(
        scenario="supply_adjusted",
        n_riders=200,                 # fixed rider population
        steps=steps,
        heatmap_alpha=float(alpha),
        heatmap_beta=float(beta),
        heatmap_gamma=float(gamma),
        seed=int(seed),
    )

    summary = summarize_final_steps(metrics, final_steps=final_steps)

    return {
        "sample_id": int(sample_id),
        "replicate": int(replicate),
        "seed": int(seed),
        "n_riders": 200,
        "heatmap_alpha": float(alpha),
        "heatmap_beta": float(beta),
        "heatmap_gamma": float(gamma),
        **summary,
    }


def append_rows(path: Path, rows: list[dict[str, float]]) -> None:
    """Append rows to CSV, creating the file if needed."""
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def plot_sobol_indices(sobol_df: pd.DataFrame, output_root: Path) -> None:
    """Create one simple bar plot per outcome, comparing S1 and ST."""
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for outcome in OUTCOMES:
        sub = sobol_df[sobol_df["outcome"] == outcome].copy()
        x = np.arange(len(sub))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x - width / 2, sub["S1"], width, label="S1")
        ax.bar(x + width / 2, sub["ST"], width, label="ST")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["parameter"], rotation=20, ha="right")
        ax.set_ylabel("Sobol index")
        ax.set_title(f"Sobol sensitivity indices: {outcome}")
        ax.legend()
        fig.tight_layout()

        fig.savefig(plot_dir / f"sobol_indices_{outcome}.pdf")
        fig.savefig(plot_dir / f"sobol_indices_{outcome}.png", dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-parameter Sobol sensitivity analysis for the delivery-rider ABM.")
    parser.add_argument("--base-samples", type=int, default=512)
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--final-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=144)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sampling-seed", type=int, default=123)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sobol_3param"))
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Three-parameter Sobol sensitivity analysis")
    print("=" * 70)
    print(f"Parameters: {PROBLEM['names']}")
    print("Fixed: n_riders = 200")
    print(f"Base samples N: {args.base_samples}")
    print(f"Replications per parameter combination: {args.n_reps}")
    print(f"Workers: {args.workers}")
    print(f"Sampler: {'SALib.sample.sobol.sample' if USE_NEW_SAMPLER else 'SALib.sample.saltelli.sample'}")

    param_values = generate_sobol_samples(args.base_samples, args.sampling_seed)
    n_param_combinations = len(param_values)
    expected_param_combinations = args.base_samples * (PROBLEM["num_vars"] + 2)
    expected_raw_runs = n_param_combinations * args.n_reps

    print(f"Generated parameter combinations: {n_param_combinations}")
    print(f"Expected N * (D + 2): {expected_param_combinations}")
    print(f"Total ABM runs: {expected_raw_runs}")

    if n_param_combinations != expected_param_combinations:
        raise RuntimeError(
            f"Unexpected sample size: got {n_param_combinations}, expected {expected_param_combinations}."
        )

    params_df = pd.DataFrame(param_values, columns=PROBLEM["names"])
    params_df.insert(0, "sample_id", np.arange(n_param_combinations))
    params_df.to_csv(args.output_root / "sobol_parameter_values.csv", index=False)

    raw_path = args.output_root / "sobol_raw_runs.csv"

    completed = set()
    if raw_path.exists():
        existing_df = pd.read_csv(raw_path)
        completed = set(zip(existing_df["sample_id"].astype(int), existing_df["replicate"].astype(int)))
        print(f"Existing completed runs found: {len(completed)}")
    else:
        print("No existing raw run file found. Starting from scratch.")

    jobs = []
    for sample_id, row in params_df.iterrows():
        for replicate in range(1, args.n_reps + 1):
            if (int(sample_id), int(replicate)) in completed:
                continue

            seed = args.seed_start + replicate - 1

            jobs.append((
                int(sample_id),
                int(replicate),
                int(seed),
                float(row["heatmap_alpha"]),
                float(row["heatmap_beta"]),
                float(row["heatmap_gamma"]),
                int(args.final_steps),
                int(args.steps),
            ))

    print(f"Runs still missing: {len(jobs)}")

    start = time.time()

    if jobs:
        if args.workers == 1:
            buffer = []
            for idx, job in enumerate(jobs, start=1):
                result = run_one(job)
                buffer.append(result)

                if len(buffer) >= 20:
                    append_rows(raw_path, buffer)
                    buffer = []

                if idx % 50 == 0:
                    print(f"Completed {idx}/{len(jobs)} missing runs.")

            append_rows(raw_path, buffer)

        else:
            buffer = []
            with Pool(processes=args.workers) as pool:
                for idx, result in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1), start=1):
                    buffer.append(result)

                    if len(buffer) >= 20:
                        append_rows(raw_path, buffer)
                        buffer = []

                    if idx % 100 == 0:
                        elapsed = time.time() - start
                        rate = idx / elapsed if elapsed > 0 else 0
                        remaining = (len(jobs) - idx) / rate if rate > 0 else float("nan")
                        print(
                            f"Completed {idx}/{len(jobs)} missing runs "
                            f"({elapsed/60:.1f} min elapsed, approx {remaining/60:.1f} min remaining)."
                        )

            append_rows(raw_path, buffer)

    elapsed = time.time() - start
    print(f"Simulation stage finished in {elapsed / 3600:.2f} hours.")

    raw_df = pd.read_csv(raw_path)

    # Check completeness.
    n_completed = len(raw_df)
    print(f"Raw runs in file: {n_completed}")
    if n_completed < expected_raw_runs:
        print("WARNING: Not all runs are complete yet. Re-run the same command to resume.")
        print(f"Expected {expected_raw_runs}, found {n_completed}.")
        return

    # Average the 10 replications for each Sobol parameter combination.
    mean_df = (
        raw_df
        .groupby("sample_id", as_index=False)[OUTCOMES]
        .mean()
    )

    mean_df = params_df.merge(mean_df, on="sample_id", how="left")
    mean_df.to_csv(args.output_root / "sobol_mean_outputs.csv", index=False)

    # SALib requires Y to be in the same order as the generated samples.
    mean_df = mean_df.sort_values("sample_id")

    sobol_rows = []

    for outcome in OUTCOMES:
        y = mean_df[outcome].to_numpy(dtype=float)

        si = sobol_analyze.analyze(
            PROBLEM,
            y,
            calc_second_order=False,
            print_to_console=False,
        )

        for i, parameter in enumerate(PROBLEM["names"]):
            s1 = float(si["S1"][i])
            st = float(si["ST"][i])

            sobol_rows.append({
                "outcome": outcome,
                "parameter": parameter,
                "S1": s1,
                "S1_conf": float(si["S1_conf"][i]),
                "ST": st,
                "ST_conf": float(si["ST_conf"][i]),
                "ST_minus_S1": st - s1,
            })

    sobol_df = pd.DataFrame(sobol_rows)
    sobol_df.to_csv(args.output_root / "sobol_indices.csv", index=False)

    plot_sobol_indices(sobol_df, args.output_root)

    print("\nSobol indices saved to:")
    print(args.output_root / "sobol_indices.csv")
    print("\nPlots saved to:")
    print(args.output_root / "plots")
    print("\nResult preview:")
    print(sobol_df)


if __name__ == "__main__":
    main()