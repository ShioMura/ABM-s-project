from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Strategy:
    name: str
    bias: float
    theta_heatmap: float
    theta_move_ratio: float
    theta_density: float
    theta_risk: float
    theta_payoff_gap: float


STRATEGIES = [
    Strategy("conservative", bias=-1.1, theta_heatmap=0.3, theta_move_ratio=0.2, theta_density=1.2, theta_risk=0.4, theta_payoff_gap=0.7),
    Strategy("imitator", bias=-0.4, theta_heatmap=0.6, theta_move_ratio=2.0, theta_density=0.4, theta_risk=0.6, theta_payoff_gap=0.4),
    Strategy("crowd_avoider", bias=-0.3, theta_heatmap=0.6, theta_move_ratio=0.1, theta_density=2.0, theta_risk=0.5, theta_payoff_gap=0.7),
    Strategy("explorer", bias=0.0, theta_heatmap=1.2, theta_move_ratio=0.5, theta_density=0.3, theta_risk=1.0, theta_payoff_gap=0.3),
]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(values: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    scaled = kappa * (values - np.max(values))
    exp_values = np.exp(scaled)
    return exp_values / exp_values.sum()


def neighbourhood(x: int, y: int, grid_size: int, include_self: bool = True) -> list[tuple[int, int]]:
    cells = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if not include_self and dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                cells.append((nx, ny))
    return cells


def count_riders(positions: np.ndarray, grid_size: int) -> np.ndarray:
    counts = np.zeros((grid_size, grid_size), dtype=int)
    for x, y in positions:
        counts[x, y] += 1
    return counts


def local_sum(matrix: np.ndarray, x: int, y: int) -> float:
    xs = slice(max(0, x - 1), min(matrix.shape[0], x + 2))
    ys = slice(max(0, y - 1), min(matrix.shape[1], y + 2))
    return float(matrix[xs, ys].sum())


def local_mean(matrix: np.ndarray, x: int, y: int) -> float:
    xs = slice(max(0, x - 1), min(matrix.shape[0], x + 2))
    ys = slice(max(0, y - 1), min(matrix.shape[1], y + 2))
    return float(matrix[xs, ys].mean())


def normalize_heatmap(raw_heatmap: np.ndarray) -> np.ndarray:
    min_value = float(raw_heatmap.min())
    max_value = float(raw_heatmap.max())
    if np.isclose(max_value, min_value):
        return np.zeros_like(raw_heatmap, dtype=float)
    return (raw_heatmap - min_value) / (max_value - min_value)


def compute_heatmap(
    scenario: str,
    new_orders: np.ndarray,
    queued_orders: np.ndarray,
    rider_counts: np.ndarray,
    alpha: float,
    gamma: float,
    beta: float,
) -> np.ndarray:
    if scenario == "baseline":
        return np.zeros_like(new_orders, dtype=float)
    if scenario == "demand_only":
        raw_heatmap = alpha * new_orders
    elif scenario == "supply_adjusted":
        raw_heatmap = alpha * new_orders + gamma * queued_orders - beta * rider_counts
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return normalize_heatmap(raw_heatmap.astype(float))


def memory_weights(memory_length: int, memory_decay: float) -> np.ndarray:
    raw = np.array([memory_decay**age for age in range(memory_length)], dtype=float)
    return raw / raw.sum()


def weighted_local_move_ratio(
    movement_history: list[tuple[np.ndarray, np.ndarray]],
    x: int,
    y: int,
    memory_length: int,
    memory_decay: float,
) -> float:
    recent_history = movement_history[-memory_length:][::-1]
    weights = memory_weights(len(recent_history), memory_decay)
    ratio = 0.0

    for weight, (move_counts, stay_counts) in zip(weights, recent_history):
        local_moves = local_sum(move_counts, x, y)
        local_stays = local_sum(stay_counts, x, y)
        local_total = local_moves + local_stays
        if local_total > 0:
            ratio += weight * (local_moves / local_total)

    return ratio


def weighted_local_density(
    density_history: list[np.ndarray],
    x: int,
    y: int,
    n_riders: int,
    memory_length: int,
    memory_decay: float,
) -> float:
    recent_history = density_history[-memory_length:][::-1]
    weights = memory_weights(len(recent_history), memory_decay)
    density = 0.0

    for weight, rider_counts in zip(weights, recent_history):
        density += weight * (local_sum(rider_counts, x, y) / max(1, n_riders))

    return density


def initialise_even_positions(n_riders: int, grid_size: int, rng: np.random.Generator) -> np.ndarray:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    positions = []
    full_rounds, remainder = divmod(n_riders, len(cells))

    for _ in range(full_rounds):
        positions.extend(cells)
    if remainder:
        positions.extend(rng.choice(len(cells), size=remainder, replace=False).tolist())
        positions[-remainder:] = [cells[idx] for idx in positions[-remainder:]]

    positions = np.array(positions, dtype=int)
    rng.shuffle(positions)
    return positions


def choose_strategy(scores: np.ndarray, rng: np.random.Generator, kappa: float) -> int:
    probabilities = softmax(scores, kappa)
    return int(rng.choice(len(scores), p=probabilities))


def choose_destination(
    x: int,
    y: int,
    rider_counts: np.ndarray,
    heatmap: np.ndarray,
    strategy: Strategy,
    rng: np.random.Generator,
    kappa: float,
) -> tuple[int, int]:
    candidates = neighbourhood(x, y, rider_counts.shape[0], include_self=False)
    max_density = max(1, int(rider_counts.max()))
    attractions = []

    for nx, ny in candidates:
        density = rider_counts[nx, ny] / max_density
        noise = rng.normal(0.0, 0.05)
        attractions.append(strategy.theta_heatmap * heatmap[nx, ny] - strategy.theta_density * density + noise)

    probabilities = softmax(np.array(attractions), kappa)
    return candidates[int(rng.choice(len(candidates), p=probabilities))]


def run_simulation(
    scenario: str = "baseline",
    grid_size: int = 10,
    n_riders: int = 200,
    steps: int = 144,
    order_lambda: float = 1.5,
    max_wait: int = 3,
    memory_length: int = 3,
    memory_decay: float = 0.7,
    payoff_memory_rate: float = 0.2,
    aspiration_income: float = 5.5,
    heatmap_alpha: float = 1.0,
    heatmap_gamma: float = 0.5,
    heatmap_beta: float = 1.0,
    fare: float = 10.0,
    move_cost: float = 1.0,
    learning_rate: float = 0.2,
    strategy_kappa: float = 0.4,
    destination_kappa: float = 2.0,
    seed: int = 42,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    positions = initialise_even_positions(n_riders, grid_size, rng)
    risks = rng.uniform(0.0, 1.0, size=n_riders)
    strategy_scores = np.zeros((n_riders, len(STRATEGIES)), dtype=float)
    recent_payoffs = np.full(n_riders, aspiration_income, dtype=float)

    queue_by_age = np.zeros((grid_size, grid_size, max_wait), dtype=int)
    prev_move = np.zeros((grid_size, grid_size), dtype=int)
    prev_stay = count_riders(positions, grid_size)
    movement_history = [(prev_move.copy(), prev_stay.copy())]
    density_history = [prev_stay.copy()]
    metrics = []

    for t in range(steps):
        expired_orders = queue_by_age[:, :, -1].copy()
        queue_by_age[:, :, 1:] = queue_by_age[:, :, :-1].copy()
        queue_by_age[:, :, 0] = 0

        new_orders = rng.poisson(order_lambda, size=(grid_size, grid_size))
        rider_counts_before = count_riders(positions, grid_size)
        queued_orders = queue_by_age.sum(axis=2)
        heatmap = compute_heatmap(
            scenario,
            new_orders,
            queued_orders,
            rider_counts_before,
            heatmap_alpha,
            heatmap_gamma,
            heatmap_beta,
        )

        chosen_strategies = np.zeros(n_riders, dtype=int)
        moved = np.zeros(n_riders, dtype=bool)
        new_positions = positions.copy()

        for i, (x, y) in enumerate(positions):
            strategy_idx = choose_strategy(strategy_scores[i], rng, strategy_kappa)
            strategy = STRATEGIES[strategy_idx]
            chosen_strategies[i] = strategy_idx

            local_move_ratio = weighted_local_move_ratio(
                movement_history,
                x,
                y,
                memory_length,
                memory_decay,
            )
            local_density = weighted_local_density(
                density_history,
                x,
                y,
                n_riders,
                memory_length,
                memory_decay,
            )
            local_heatmap = local_mean(heatmap, x, y)

            move_logit = (
                strategy.bias
                + strategy.theta_heatmap * local_heatmap
                + strategy.theta_move_ratio * local_move_ratio
                - strategy.theta_density * local_density
                + strategy.theta_risk * risks[i]
                + strategy.theta_payoff_gap * ((aspiration_income - recent_payoffs[i]) / fare)
            )
            move_probability = sigmoid(move_logit)

            if rng.random() < move_probability:
                nx, ny = choose_destination(
                    x,
                    y,
                    rider_counts_before,
                    heatmap,
                    strategy,
                    rng,
                    destination_kappa,
                )
                new_positions[i] = (nx, ny)
                moved[i] = True

        positions = new_positions
        rider_counts_after = count_riders(positions, grid_size)
        available_orders = new_orders + queue_by_age.sum(axis=2)
        matched_by_zone = np.minimum(available_orders, rider_counts_after)

        payoffs = np.where(moved, -move_cost, 0.0)
        matched = np.zeros(n_riders, dtype=bool)

        for x in range(grid_size):
            for y in range(grid_size):
                rider_indices = np.flatnonzero((positions[:, 0] == x) & (positions[:, 1] == y))
                n_matched = int(matched_by_zone[x, y])
                if n_matched > 0 and len(rider_indices) > 0:
                    winners = rng.choice(rider_indices, size=n_matched, replace=False)
                    matched[winners] = True

        payoffs[matched] += fare
        recent_payoffs = (1 - payoff_memory_rate) * recent_payoffs + payoff_memory_rate * payoffs

        for i, strategy_idx in enumerate(chosen_strategies):
            old_score = strategy_scores[i, strategy_idx]
            strategy_scores[i, strategy_idx] = (1 - learning_rate) * old_score + learning_rate * payoffs[i]

        order_buckets = np.dstack([new_orders, queue_by_age])
        remaining_matches = matched_by_zone.copy()
        for age in range(max_wait, -1, -1):
            matched_from_bucket = np.minimum(order_buckets[:, :, age], remaining_matches)
            order_buckets[:, :, age] -= matched_from_bucket
            remaining_matches -= matched_from_bucket
        expired_orders += order_buckets[:, :, max_wait]
        queue_by_age = order_buckets[:, :, :max_wait]

        prev_move = np.zeros((grid_size, grid_size), dtype=int)
        prev_stay = np.zeros((grid_size, grid_size), dtype=int)
        for i, (x, y) in enumerate(positions):
            if moved[i]:
                prev_move[x, y] += 1
            else:
                prev_stay[x, y] += 1
        movement_history.append((prev_move.copy(), prev_stay.copy()))
        density_history.append(rider_counts_after.copy())
        movement_history = movement_history[-memory_length:]
        density_history = density_history[-memory_length:]

        total_available_orders = int(available_orders.sum())
        total_matched = int(matched.sum())
        congested_zones = rider_counts_after > available_orders
        rider_share = rider_counts_after.flatten() / n_riders
        strategy_counts = np.bincount(chosen_strategies, minlength=len(STRATEGIES)) / n_riders

        row = {
            "step": t,
            "scenario": scenario,
            "avg_income": float(payoffs.mean()),
            "idle_rate": float(1.0 - matched.mean()),
            "movement_rate": float(moved.mean()),
            "matching_rate": float(total_matched / total_available_orders) if total_available_orders else 1.0,
            "congestion_frequency": float(congested_zones.mean()),
            "avg_heatmap": float(heatmap.mean()),
            "recent_payoff_mean": float(recent_payoffs.mean()),
            "avg_queue": float(queue_by_age.sum(axis=2).mean()),
            "expired_orders": float(expired_orders.sum()),
            "rider_concentration_hhi": float(np.sum(rider_share**2)),
        }
        for idx, strategy in enumerate(STRATEGIES):
            row[f"strategy_{strategy.name}"] = float(strategy_counts[idx])
        metrics.append(row)

    return metrics


def write_metrics(metrics: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the food-delivery rider relocation simulation.")
    parser.add_argument("--scenario", choices=["baseline", "demand_only", "supply_adjusted"], default="baseline")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--riders", type=int, default=200)
    parser.add_argument("--steps", type=int, default=144)
    parser.add_argument("--order-lambda", type=float, default=1.5)
    parser.add_argument("--max-wait", type=int, default=3)
    parser.add_argument("--memory-length", type=int, default=3)
    parser.add_argument("--memory-decay", type=float, default=0.7)
    parser.add_argument("--payoff-memory-rate", type=float, default=0.2)
    parser.add_argument("--aspiration-income", type=float, default=5.5)
    parser.add_argument("--heatmap-alpha", type=float, default=1.0)
    parser.add_argument("--heatmap-gamma", type=float, default=0.5)
    parser.add_argument("--heatmap-beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_metrics.csv"))
    args = parser.parse_args()

    metrics = run_simulation(
        scenario=args.scenario,
        grid_size=args.grid_size,
        n_riders=args.riders,
        steps=args.steps,
        order_lambda=args.order_lambda,
        max_wait=args.max_wait,
        memory_length=args.memory_length,
        memory_decay=args.memory_decay,
        payoff_memory_rate=args.payoff_memory_rate,
        aspiration_income=args.aspiration_income,
        heatmap_alpha=args.heatmap_alpha,
        heatmap_gamma=args.heatmap_gamma,
        heatmap_beta=args.heatmap_beta,
        seed=args.seed,
    )
    write_metrics(metrics, args.output)

    final = metrics[-1]
    print(f"{args.scenario} simulation complete.")
    print(f"Metrics written to: {args.output}")
    print(f"Final average income: {final['avg_income']:.3f}")
    print(f"Final idle rate: {final['idle_rate']:.3f}")
    print(f"Final matching rate: {final['matching_rate']:.3f}")
    print(f"Final movement rate: {final['movement_rate']:.3f}")
    print(f"Final congestion frequency: {final['congestion_frequency']:.3f}")


if __name__ == "__main__":
    main()
