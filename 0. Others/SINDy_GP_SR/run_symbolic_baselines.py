"""Train standard SINDy and genetic-programming symbolic-regression baselines.

Both baselines use the same lexicographically sorted every-fifth-point outer
test split as SE, RF, FNN, SSI, and LSSI. Hyperparameters are shared by FR78,
FR79, and FR95. Model selection uses only an inner validation subset of the
outer training data; the held-out test points are evaluated exactly once.

SINDy fits log10(P) from a total-degree-five polynomial library of standardized
log10(f) and log10(B) using sequentially thresholded least squares (STLSQ).
GP-SR evolves an explicit expression for log10(P) with protected operators and
affine output scaling. No external symbolic-regression package is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
MATERIALS = ("FR78", "FR79", "FR95")
BASE_SEED = 42

SINDY_DEGREE = 5
SINDY_THRESHOLDS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.03,
                    0.05, 0.08, 0.12, 0.20, 0.30, 0.50)
SINDY_COMPLEXITY_PENALTY = 0.005

GP_POPULATION = 600
GP_GENERATIONS = 140
GP_TOURNAMENT = 5
GP_ELITES = 12
GP_CROSSOVER_RATE = 0.72
GP_MUTATION_RATE = 0.25
GP_MAX_DEPTH = 8
GP_MAX_NODES = 45
GP_PARSIMONY = 2.0e-4
GP_VALIDATION_COMPLEXITY = 0.006
GP_FUNCTIONS = ("add", "sub", "mul", "div", "log", "sqrt", "tanh")
BINARY_FUNCTIONS = {"add", "sub", "mul", "div"}
UNARY_FUNCTIONS = {"log", "sqrt", "tanh"}


Tree = tuple[Any, ...]


def load_material(material: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = ROOT / "2. LSSI" / material / f"CoreLossData_{material}.mat"
    data = scipy.io.loadmat(path)
    frequency = np.asarray(data["f"], dtype=float).ravel()
    flux_density = np.asarray(data["B"], dtype=float).ravel()
    loss = np.asarray(data["P"], dtype=float).ravel()
    if material == "FR95":
        frequency /= 1000.0
        flux_density /= 10.0
    order = np.lexsort((flux_density, frequency))
    return frequency[order], flux_density[order], loss[order]


def split_masks(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test = np.zeros(n_samples, dtype=bool)
    test[::5] = True
    outer_train = ~test
    training_indices = np.flatnonzero(outer_train)
    validation = np.zeros(n_samples, dtype=bool)
    validation[training_indices[::5]] = True
    evolution = outer_train & ~validation
    return evolution, validation, test


def regression_metrics(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - measured
    denominator = np.sum((measured - measured.mean()) ** 2)
    return {
        "mape_percent": float(np.mean(np.abs(residual / measured)) * 100.0),
        "r2": float(1.0 - np.sum(residual**2) / denominator),
        "rmse_mw_cm3": float(np.sqrt(np.mean(residual**2))),
        "mean_residual_percent": float(np.mean(residual / measured) * 100.0),
        "std_residual_percent": float(np.std(residual / measured) * 100.0),
    }


def polynomial_library(x: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    powers: list[tuple[int, int]] = []
    columns = []
    for total_degree in range(SINDY_DEGREE + 1):
        for frequency_power in range(total_degree + 1):
            flux_power = total_degree - frequency_power
            powers.append((frequency_power, flux_power))
            columns.append(x[:, 0] ** frequency_power * x[:, 1] ** flux_power)
    return np.column_stack(columns), powers


def stlsq(library: np.ndarray, target: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    scale = np.linalg.norm(library, axis=0)
    scale[scale == 0.0] = 1.0
    normalized = library / scale
    normalized_coefficients = np.linalg.lstsq(normalized, target, rcond=1.0e-10)[0]
    active = np.ones(normalized.shape[1], dtype=bool)
    for _ in range(25):
        updated = np.abs(normalized_coefficients) >= threshold
        updated[0] = True
        if np.array_equal(updated, active):
            break
        active = updated
        normalized_coefficients[:] = 0.0
        normalized_coefficients[active] = np.linalg.lstsq(
            normalized[:, active], target, rcond=1.0e-10
        )[0]
    return normalized_coefficients / scale, active


def fit_sindy(
    x: np.ndarray,
    log_loss: np.ndarray,
    loss: np.ndarray,
    evolution: np.ndarray,
    validation: np.ndarray,
    outer_train: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    library, powers = polynomial_library(x)
    candidates = []
    for threshold in SINDY_THRESHOLDS:
        coefficients, active = stlsq(library[evolution], log_loss[evolution], threshold)
        validation_prediction = 10.0 ** np.clip(library[validation] @ coefficients, -12.0, 12.0)
        validation_mape = float(
            np.mean(np.abs((validation_prediction - loss[validation]) / loss[validation])) * 100.0
        )
        score = validation_mape + SINDY_COMPLEXITY_PENALTY * int(active.sum())
        candidates.append((score, threshold, validation_mape, int(active.sum())))
    _, threshold, validation_mape, _ = min(candidates)
    coefficients, active = stlsq(library[outer_train], log_loss[outer_train], threshold)
    prediction = 10.0 ** np.clip(library @ coefficients, -12.0, 12.0)
    terms = [
        {
            "frequency_power": powers[index][0],
            "flux_power": powers[index][1],
            "coefficient": float(coefficients[index]),
        }
        for index in np.flatnonzero(active)
    ]
    result = {
        "description": "degree-5 polynomial SINDy with STLSQ in standardized log coordinates",
        "selected_threshold": threshold,
        "inner_validation_mape_percent": validation_mape,
        "active_term_count": int(active.sum()),
        "candidate_term_count": len(powers),
        "terms": terms,
    }
    return result, prediction


def terminal(rng: random.Random) -> Tree:
    draw = rng.random()
    if draw < 0.40:
        return ("x", 0)
    if draw < 0.80:
        return ("x", 1)
    return ("const", rng.uniform(-2.0, 2.0))


def random_tree(rng: random.Random, depth: int, grow: bool = True) -> Tree:
    if depth <= 0 or (grow and rng.random() < 0.25):
        return terminal(rng)
    operation = rng.choice(GP_FUNCTIONS)
    if operation in BINARY_FUNCTIONS:
        return (operation, random_tree(rng, depth - 1, grow), random_tree(rng, depth - 1, grow))
    return (operation, random_tree(rng, depth - 1, grow))


def evaluate_tree(tree: Tree, x: np.ndarray) -> np.ndarray:
    operation = tree[0]
    if operation == "x":
        return x[:, int(tree[1])]
    if operation == "const":
        return np.full(x.shape[0], float(tree[1]))
    left = evaluate_tree(tree[1], x)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        if operation == "log":
            value = np.log(np.abs(left) + 1.0e-6)
        elif operation == "sqrt":
            value = np.sqrt(np.abs(left) + 1.0e-9)
        elif operation == "tanh":
            value = np.tanh(left)
        else:
            right = evaluate_tree(tree[2], x)
            if operation == "add":
                value = left + right
            elif operation == "sub":
                value = left - right
            elif operation == "mul":
                value = left * right
            elif operation == "div":
                safe_denominator = np.where(np.abs(right) < 1.0e-6, 1.0, right)
                value = left / safe_denominator
            else:
                raise KeyError(operation)
    return np.nan_to_num(value, nan=0.0, posinf=1.0e6, neginf=-1.0e6).clip(-1.0e6, 1.0e6)


def node_count(tree: Tree) -> int:
    if tree[0] in {"x", "const"}:
        return 1
    if tree[0] in UNARY_FUNCTIONS:
        return 1 + node_count(tree[1])
    return 1 + node_count(tree[1]) + node_count(tree[2])


def tree_depth(tree: Tree) -> int:
    if tree[0] in {"x", "const"}:
        return 0
    if tree[0] in UNARY_FUNCTIONS:
        return 1 + tree_depth(tree[1])
    return 1 + max(tree_depth(tree[1]), tree_depth(tree[2]))


def subtree_paths(tree: Tree, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    paths = [prefix]
    if tree[0] in UNARY_FUNCTIONS:
        paths.extend(subtree_paths(tree[1], prefix + (1,)))
    elif tree[0] in BINARY_FUNCTIONS:
        paths.extend(subtree_paths(tree[1], prefix + (1,)))
        paths.extend(subtree_paths(tree[2], prefix + (2,)))
    return paths


def get_subtree(tree: Tree, path: tuple[int, ...]) -> Tree:
    current = tree
    for branch in path:
        current = current[branch]
    return current


def replace_subtree(tree: Tree, path: tuple[int, ...], replacement: Tree) -> Tree:
    if not path:
        return replacement
    branch = path[0]
    children = list(tree)
    children[branch] = replace_subtree(children[branch], path[1:], replacement)
    return tuple(children)


def crossover(first: Tree, second: Tree, rng: random.Random) -> Tree:
    first_path = rng.choice(subtree_paths(first))
    second_path = rng.choice(subtree_paths(second))
    return replace_subtree(first, first_path, get_subtree(second, second_path))


def mutate(tree: Tree, rng: random.Random) -> Tree:
    paths = subtree_paths(tree)
    path = rng.choice(paths)
    selected = get_subtree(tree, path)
    if selected[0] == "const" and rng.random() < 0.55:
        replacement = ("const", float(selected[1]) + rng.gauss(0.0, 0.35))
    else:
        replacement = random_tree(rng, rng.randint(1, 3), grow=True)
    return replace_subtree(tree, path, replacement)


def affine_fit(values: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    variance = float(np.var(values))
    if not np.isfinite(variance) or variance < 1.0e-12:
        return float(target.mean()), 0.0
    slope = float(np.cov(values, target, ddof=0)[0, 1] / variance)
    intercept = float(target.mean() - slope * values.mean())
    return intercept, slope


@dataclass(frozen=True)
class GPRecord:
    tree: Tree
    fitness: float
    validation_score: float
    validation_mape: float
    intercept: float
    slope: float


def fit_gp(
    x: np.ndarray,
    log_loss: np.ndarray,
    loss: np.ndarray,
    evolution: np.ndarray,
    validation: np.ndarray,
    outer_train: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    rng = random.Random(seed)
    x_evolution = x[evolution]
    y_evolution = log_loss[evolution]
    cache: dict[Tree, tuple[float, float, float]] = {}

    def fitness(tree: Tree) -> tuple[float, float, float]:
        cached = cache.get(tree)
        if cached is not None:
            return cached
        if node_count(tree) > GP_MAX_NODES or tree_depth(tree) > GP_MAX_DEPTH:
            value = (1.0e9, 0.0, 0.0)
        else:
            output = evaluate_tree(tree, x_evolution)
            intercept, slope = affine_fit(output, y_evolution)
            residual = intercept + slope * output - y_evolution
            mse = float(np.mean(residual**2))
            value = (mse + GP_PARSIMONY * node_count(tree), intercept, slope)
        cache[tree] = value
        return value

    def validation_record(tree: Tree) -> GPRecord:
        train_output = evaluate_tree(tree, x_evolution)
        intercept, slope = affine_fit(train_output, y_evolution)
        validation_log = intercept + slope * evaluate_tree(tree, x[validation])
        validation_prediction = 10.0 ** np.clip(validation_log, -12.0, 12.0)
        validation_mape = float(
            np.mean(np.abs((validation_prediction - loss[validation]) / loss[validation])) * 100.0
        )
        score = validation_mape + GP_VALIDATION_COMPLEXITY * node_count(tree)
        return GPRecord(tree, fitness(tree)[0], score, validation_mape, intercept, slope)

    population = []
    for index in range(GP_POPULATION):
        depth = 2 + index % 4
        population.append(random_tree(rng, depth, grow=(index % 2 == 0)))

    archive: dict[Tree, GPRecord] = {}
    progress = []
    for generation in range(GP_GENERATIONS + 1):
        ranked = sorted(population, key=lambda item: fitness(item)[0])
        for tree in ranked[: max(30, GP_ELITES)]:
            record = validation_record(tree)
            previous = archive.get(tree)
            if previous is None or record.validation_score < previous.validation_score:
                archive[tree] = record
        best_validation = min(archive.values(), key=lambda item: item.validation_score)
        progress.append({
            "generation": generation,
            "training_objective": float(fitness(ranked[0])[0]),
            "best_validation_mape_percent": best_validation.validation_mape,
            "best_node_count": node_count(best_validation.tree),
        })
        if generation == GP_GENERATIONS:
            break

        def tournament() -> Tree:
            competitors = rng.sample(population, GP_TOURNAMENT)
            return min(competitors, key=lambda item: fitness(item)[0])

        next_population = ranked[:GP_ELITES]
        while len(next_population) < GP_POPULATION:
            parent = tournament()
            child = parent
            if rng.random() < GP_CROSSOVER_RATE:
                child = crossover(parent, tournament(), rng)
            if rng.random() < GP_MUTATION_RATE:
                child = mutate(child, rng)
            if node_count(child) > GP_MAX_NODES or tree_depth(child) > GP_MAX_DEPTH:
                child = mutate(parent, rng)
            next_population.append(child)
        population = next_population

    best = min(archive.values(), key=lambda item: item.validation_score)
    full_output = evaluate_tree(best.tree, x[outer_train])
    intercept, slope = affine_fit(full_output, log_loss[outer_train])
    prediction = 10.0 ** np.clip(intercept + slope * evaluate_tree(best.tree, x), -12.0, 12.0)
    result = {
        "description": "tree-based GP symbolic regression in standardized log coordinates",
        "inner_validation_mape_percent": best.validation_mape,
        "node_count": node_count(best.tree) + 2,
        "tree_node_count": node_count(best.tree),
        "depth": tree_depth(best.tree),
        "affine_intercept": intercept,
        "affine_slope": slope,
        "expression": expression_string(best.tree, intercept, slope),
        "primitive_set": list(GP_FUNCTIONS),
        "progress": progress,
    }
    return result, prediction


def expression_string(tree: Tree, intercept: float, slope: float) -> str:
    def render(node: Tree) -> str:
        operation = node[0]
        if operation == "x":
            return "u_f" if int(node[1]) == 0 else "u_B"
        if operation == "const":
            return f"{float(node[1]):.6g}"
        left = render(node[1])
        if operation == "log":
            return f"log(abs({left}) + 1e-6)"
        if operation == "sqrt":
            return f"sqrt(abs({left}) + 1e-9)"
        if operation == "tanh":
            return f"tanh({left})"
        right = render(node[2])
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[operation]
        if operation == "div":
            return f"protected_div({left}, {right})"
        return f"({left} {symbol} {right})"

    return f"log10(P) = {intercept:.8g} + ({slope:.8g}) * ({render(tree)})"


def train_material(material: str, seed: int, output_dir: Path) -> dict[str, Any]:
    frequency, flux_density, loss = load_material(material)
    evolution, validation, test = split_masks(len(loss))
    outer_train = ~test
    raw_x = np.column_stack((np.log10(frequency), np.log10(flux_density)))
    mean = raw_x[outer_train].mean(axis=0)
    standard_deviation = raw_x[outer_train].std(axis=0)
    x = (raw_x - mean) / standard_deviation
    log_loss = np.log10(loss)

    sindy, sindy_prediction = fit_sindy(
        x, log_loss, loss, evolution, validation, outer_train
    )
    gp, gp_prediction = fit_gp(
        x, log_loss, loss, evolution, validation, outer_train, seed
    )
    sindy["test"] = regression_metrics(loss[test], sindy_prediction[test])
    gp["test"] = regression_metrics(loss[test], gp_prediction[test])

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{material}_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "frequency_kHz", "flux_density_mT", "measured_mW_cm3", "split",
            "sindy_predicted_mW_cm3", "gp_sr_predicted_mW_cm3",
        ])
        for index in range(len(loss)):
            split = "test" if test[index] else ("validation" if validation[index] else "train")
            writer.writerow([
                frequency[index], flux_density[index], loss[index], split,
                sindy_prediction[index], gp_prediction[index],
            ])

    result = {
        "material": material,
        "seed": seed,
        "split": "lexicographically sorted every fifth point; inner every fifth training point for validation",
        "coordinate_definition": {
            "u_f": "(log10(f) - mean_log10_f) / std_log10_f",
            "u_B": "(log10(B) - mean_log10_B) / std_log10_B",
            "mean_log10_f": float(mean[0]),
            "std_log10_f": float(standard_deviation[0]),
            "mean_log10_B": float(mean[1]),
            "std_log10_B": float(standard_deviation[1]),
        },
        "SINDy": sindy,
        "GP-SR": gp,
        "prediction_path": str(prediction_path),
    }
    (output_dir / f"{material}_symbolic_baselines.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def settings_record() -> dict[str, Any]:
    return {
        "SINDy": {
            "degree": SINDY_DEGREE,
            "threshold_candidates": list(SINDY_THRESHOLDS),
            "validation_complexity_penalty": SINDY_COMPLEXITY_PENALTY,
        },
        "GP-SR": {
            "population": GP_POPULATION,
            "generations": GP_GENERATIONS,
            "tournament_size": GP_TOURNAMENT,
            "elites": GP_ELITES,
            "crossover_rate": GP_CROSSOVER_RATE,
            "mutation_rate": GP_MUTATION_RATE,
            "maximum_depth": GP_MAX_DEPTH,
            "maximum_nodes": GP_MAX_NODES,
            "training_parsimony": GP_PARSIMONY,
            "validation_complexity_penalty": GP_VALIDATION_COMPLEXITY,
            "primitive_set": list(GP_FUNCTIONS),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", choices=(*MATERIALS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    selected = MATERIALS if args.material == "all" else (args.material,)
    results = []
    for offset, material in enumerate(selected):
        print(f"Training SINDy and GP-SR for {material} ...", flush=True)
        result = train_material(material, args.seed + MATERIALS.index(material), args.output)
        results.append(result)
        print(json.dumps({
            "material": material,
            "SINDy_MAPE": result["SINDy"]["test"]["mape_percent"],
            "SINDy_terms": result["SINDy"]["active_term_count"],
            "GP-SR_MAPE": result["GP-SR"]["test"]["mape_percent"],
            "GP-SR_nodes": result["GP-SR"]["node_count"],
        }, indent=2), flush=True)
    summary = {
        "settings": settings_record(),
        "results": {
            item["material"]: {
                "SINDy": item["SINDy"],
                "GP-SR": item["GP-SR"],
                "coordinate_definition": item["coordinate_definition"],
                "prediction_path": item["prediction_path"],
            }
            for item in results
        },
    }
    (args.output / "symbolic_baseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
