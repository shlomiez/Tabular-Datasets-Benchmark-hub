#!/usr/bin/env python3
"""Optimize a hard subset from a synthetic spiked-covariance dataset.

This script compares two meta-heuristics over a binary selection vector w:
- Simulated Annealing (SA)
- Genetic Algorithm (GA)

The objective minimizes subset size while enforcing hard-dataset constraints
through large penalties.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tqdm import tqdm

from src.model_training import compute_auc
from src.data_preprocessing import load_dataset_xy
from src.utils import ensure_dir, set_global_seed


@dataclass
class EvalResult:
    cost: float
    auc_all: float
    auc_gt: float
    hardness_penalty: float
    size_penalty: float
    balance_penalty: float
    gt_penalty: float
    selected_size: int
    positive_ratio: float
    minority_ratio: float

    @property
    def total_penalty(self) -> float:
        return self.hardness_penalty + self.size_penalty + self.balance_penalty + self.gt_penalty

    @property
    def is_feasible(self) -> bool:
        return self.total_penalty == 0.0


@dataclass
class OptimizerResult:
    method: str
    best_w: np.ndarray
    best_eval: EvalResult
    best_feasible_w: np.ndarray | None
    best_feasible_eval: EvalResult | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find a hard subset with SA and GA on spiked-covariance data."
    )
    parser.add_argument(
        "--data-path",
        nargs="+",
        default=["data/spiked_covariance_dataset(200,5000,50).npz"],
        help="Path to dataset file (.npz or .mat).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/optimization_peeling_experiment",
        help="Directory for run summary artifacts.",
    )

    parser.add_argument("--sa-iters", type=int, default=1000, help="SA iterations.")
    parser.add_argument("--sa-temp0", type=float, default=1.0, help="SA initial temperature.")
    parser.add_argument("--sa-temp-min", type=float, default=1e-3, help="SA minimum temperature.")

    parser.add_argument("--ga-pop-size", type=int, default=50, help="GA population size.")
    parser.add_argument("--ga-generations", type=int, default=30, help="GA generations.")
    parser.add_argument(
        "--ga-cxpb",
        type=float,
        default=0.8,
        help="GA uniform crossover probability per mating event.",
    )
    parser.add_argument(
        "--ga-mutpb",
        type=float,
        default=None,
        help="GA per-bit mutation probability (default: 1 / n_pool).",
    )

    return parser.parse_args()


def resolve_data_path_arg(data_path_tokens: list[str]) -> Path:
    # Support unquoted paths with spaces by joining all tokens after --data-path.
    joined = " ".join(data_path_tokens)
    return Path(joined).expanduser().resolve()


def _infer_gt_indices(path: Path) -> np.ndarray | None:
    candidate_keys = ("gt_indices", "gt_idx", "support", "relevant_features", "feature_indices")

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            for key in candidate_keys:
                if key in data:
                    return np.asarray(data[key], dtype=np.int64).reshape(-1)

    if path.suffix.lower() == ".mat":
        mat = loadmat(path)
        for key in candidate_keys:
            if key in mat:
                return np.asarray(mat[key], dtype=np.int64).reshape(-1)

    # No known GT support metadata found.
    return None


def load_data(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix not in {".npz", ".mat"}:
        raise ValueError(f"Unsupported dataset format: {dataset_path.name}. Use .npz or .mat")

    X, y = load_dataset_xy(dataset_path)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    gt_indices = _infer_gt_indices(dataset_path)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows ({X.shape[0]}) and y length ({y.shape[0]}) must match")
    if gt_indices is not None:
        if gt_indices.ndim != 1:
            raise ValueError(f"gt_indices must be 1D, got shape {gt_indices.shape}")
        if gt_indices.size == 0:
            raise ValueError("gt_indices is empty")
        if np.any(gt_indices < 0) or np.any(gt_indices >= X.shape[1]):
            raise ValueError("gt_indices contains out-of-range feature indices")

    classes, y_encoded = np.unique(y, return_inverse=True)
    if classes.size != 2:
        raise ValueError(
            f"This script expects binary labels; found {classes.size} classes: {classes.tolist()}"
        )

    return X, y_encoded.astype(np.int64), gt_indices


def recover_gt_indices_from_fs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    max_k: int = 60,
) -> np.ndarray:
    """Recover proxy GT indices by choosing FS subset with best Dtest AUC."""
    n_features = X_train.shape[1]
    if n_features == 0:
        raise ValueError("Cannot recover gt_indices from an empty feature matrix")

    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return np.arange(n_features, dtype=np.int64)

    k = max(1, min(max_k, n_features))
    best_auc = -1.0
    best_indices = np.arange(n_features, dtype=np.int64)

    selectors: list[tuple[str, np.ndarray]] = []

    skb_f = SelectKBest(score_func=f_classif, k=k)
    skb_f.fit(X_train, y_train)
    selectors.append(("f_classif", skb_f.get_support(indices=True)))

    skb_mi = SelectKBest(score_func=mutual_info_classif, k=k)
    skb_mi.fit(X_train, y_train)
    selectors.append(("mutual_info", skb_mi.get_support(indices=True)))

    et = ExtraTreesClassifier(
        n_estimators=200,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    et.fit(X_train, y_train)
    et_idx = np.argsort(et.feature_importances_)[-k:]
    selectors.append(("etree_importance", np.sort(et_idx.astype(np.int64))))

    svc = LinearSVC(penalty="l1", dual=False, random_state=seed, max_iter=50000)
    svc.fit(X_train, y_train)
    svc_scores = np.abs(np.ravel(svc.coef_))
    svc_idx = np.argsort(svc_scores)[-k:]
    selectors.append(("linear_svc_l1", np.sort(svc_idx.astype(np.int64))))

    for _, idx in selectors:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            continue
        auc = fit_etree_auc(X_train[:, idx], y_train, X_test[:, idx], y_test)
        if auc > best_auc:
            best_auc = auc
            best_indices = idx

    return np.unique(best_indices.astype(np.int64))


def split_pool_test(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return train_test_split(
            X,
            y,
            train_size=0.5,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        # Fallback for edge cases where stratification constraints are violated.
        return train_test_split(X, y, train_size=0.5, random_state=seed, stratify=None)


def fit_etree_auc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    if X_train.shape[0] < 2 or np.unique(y_train).size < 2:
        return 0.5
    if y_eval.size < 2 or np.unique(y_eval).size < 2:
        return 0.5

    clf = ExtraTreesClassifier(
        max_depth=3,
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_eval)
    return float(compute_auc(y_eval, y_proba))


def build_evaluator(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gt_indices: np.ndarray,
):
    cache: dict[bytes, EvalResult] = {}
    penalty_mult = 1000.0

    def evaluate_subset(w: np.ndarray) -> EvalResult:
        w = np.asarray(w, dtype=np.uint8)
        key = w.tobytes()
        if key in cache:
            return cache[key]

        selected = np.flatnonzero(w)
        selected_size = int(selected.size)

        if selected_size == 0:
            result = EvalResult(
                cost=5.0 * penalty_mult,
                auc_all=0.5,
                auc_gt=0.5,
                hardness_penalty=0.0,
                size_penalty=penalty_mult * 50.0,
                balance_penalty=penalty_mult * 0.40,
                gt_penalty=penalty_mult * 0.40,
                selected_size=0,
                positive_ratio=0.0,
                minority_ratio=0.0,
            )
            cache[key] = result
            return result

        X_sub = X_pool[selected]
        y_sub = y_pool[selected]

        positive_ratio = float(np.mean(y_sub == 1))
        minority_ratio = min(positive_ratio, 1.0 - positive_ratio)

        auc_all = fit_etree_auc(X_sub, y_sub, X_test, y_test)
        auc_gt = fit_etree_auc(X_sub[:, gt_indices], y_sub, X_test[:, gt_indices], y_test)

        hardness_penalty = penalty_mult * max(0.0, auc_all - 0.70)
        gt_penalty = penalty_mult * max(0.0, 0.90 - auc_gt)
        size_penalty = penalty_mult * max(0.0, 50.0 - float(selected_size))
        balance_penalty = penalty_mult * max(0.0, 0.40 - minority_ratio)

        cost = float(
            float(selected_size)
            + hardness_penalty
            + gt_penalty
            + size_penalty
            + balance_penalty
        )

        result = EvalResult(
            cost=cost,
            auc_all=float(auc_all),
            auc_gt=float(auc_gt),
            hardness_penalty=float(hardness_penalty),
            size_penalty=size_penalty,
            balance_penalty=balance_penalty,
            gt_penalty=float(gt_penalty),
            selected_size=selected_size,
            positive_ratio=positive_ratio,
            minority_ratio=minority_ratio,
        )
        cache[key] = result
        return result

    return evaluate_subset


def optimize_simulated_annealing(
    evaluate_subset,
    n_pool: int,
    rng: np.random.Generator,
    iters: int = 1000,
    temp0: float = 1.0,
    temp_min: float = 1e-3,
) -> OptimizerResult:
    current_w = np.ones(n_pool, dtype=np.uint8)
    current_eval = evaluate_subset(current_w)

    best_w = current_w.copy()
    best_eval = current_eval

    best_feasible_w = current_w.copy() if current_eval.is_feasible else None
    best_feasible_eval = current_eval if current_eval.is_feasible else None

    for idx in tqdm(range(iters), desc="Simulated Annealing", unit="iter"):
        frac = idx / max(1, iters - 1)
        temp = temp0 * ((temp_min / temp0) ** frac)
        temp = max(temp, 1e-12)

        candidate_w = current_w.copy()
        n_flip = int(rng.integers(1, 4))
        flip_idx = rng.choice(n_pool, size=n_flip, replace=False)
        candidate_w[flip_idx] = 1 - candidate_w[flip_idx]

        candidate_eval = evaluate_subset(candidate_w)
        delta = candidate_eval.cost - current_eval.cost

        if delta <= 0.0:
            accept = True
        else:
            accept_prob = math.exp(-delta / temp)
            accept = rng.random() < accept_prob

        if accept:
            current_w = candidate_w
            current_eval = candidate_eval

        if candidate_eval.cost < best_eval.cost:
            best_w = candidate_w.copy()
            best_eval = candidate_eval

        if candidate_eval.is_feasible:
            if best_feasible_eval is None or candidate_eval.cost < best_feasible_eval.cost:
                best_feasible_w = candidate_w.copy()
                best_feasible_eval = candidate_eval

    return OptimizerResult(
        method="Simulated Annealing",
        best_w=best_w,
        best_eval=best_eval,
        best_feasible_w=best_feasible_w,
        best_feasible_eval=best_feasible_eval,
    )


def tournament_select(population: np.ndarray, evals: list[EvalResult], rng: np.random.Generator) -> np.ndarray:
    i1 = int(rng.integers(0, population.shape[0]))
    i2 = int(rng.integers(0, population.shape[0]))
    if evals[i1].cost <= evals[i2].cost:
        return population[i1]
    return population[i2]


def uniform_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mask = rng.random(p1.size) < 0.5
    c1 = np.where(mask, p1, p2).astype(np.uint8)
    c2 = np.where(mask, p2, p1).astype(np.uint8)
    return c1, c2


def mutate_bits(individual: np.ndarray, mutpb: float, rng: np.random.Generator) -> np.ndarray:
    flips = rng.random(individual.size) < mutpb
    if np.any(flips):
        individual = individual.copy()
        individual[flips] = 1 - individual[flips]
    return individual


def optimize_genetic_algorithm(
    evaluate_subset,
    n_pool: int,
    rng: np.random.Generator,
    pop_size: int = 50,
    generations: int = 30,
    cxpb: float = 0.8,
    mutpb: float | None = None,
) -> OptimizerResult:
    if mutpb is None:
        mutpb = 1.0 / float(n_pool)

    pop = rng.integers(0, 2, size=(pop_size, n_pool), dtype=np.uint8)
    pop[0, :] = 1

    evals = [evaluate_subset(ind) for ind in pop]

    best_idx = int(np.argmin([e.cost for e in evals]))
    best_w = pop[best_idx].copy()
    best_eval = evals[best_idx]

    best_feasible_w = None
    best_feasible_eval = None
    for ind, ind_eval in zip(pop, evals):
        if ind_eval.is_feasible:
            if best_feasible_eval is None or ind_eval.cost < best_feasible_eval.cost:
                best_feasible_w = ind.copy()
                best_feasible_eval = ind_eval

    for _ in tqdm(range(generations), desc="Genetic Algorithm", unit="gen"):
        elite_idx = int(np.argmin([e.cost for e in evals]))
        elite = pop[elite_idx].copy()

        new_pop = [elite]
        while len(new_pop) < pop_size:
            parent1 = tournament_select(pop, evals, rng)
            parent2 = tournament_select(pop, evals, rng)

            if rng.random() < cxpb:
                child1, child2 = uniform_crossover(parent1, parent2, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = mutate_bits(child1, mutpb, rng)
            child2 = mutate_bits(child2, mutpb, rng)

            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)

        pop = np.asarray(new_pop, dtype=np.uint8)
        evals = [evaluate_subset(ind) for ind in pop]

        gen_best_idx = int(np.argmin([e.cost for e in evals]))
        gen_best_w = pop[gen_best_idx]
        gen_best_eval = evals[gen_best_idx]

        if gen_best_eval.cost < best_eval.cost:
            best_w = gen_best_w.copy()
            best_eval = gen_best_eval

        for ind, ind_eval in zip(pop, evals):
            if ind_eval.is_feasible:
                if best_feasible_eval is None or ind_eval.cost < best_feasible_eval.cost:
                    best_feasible_w = ind.copy()
                    best_feasible_eval = ind_eval

    return OptimizerResult(
        method="Genetic Algorithm",
        best_w=best_w,
        best_eval=best_eval,
        best_feasible_w=best_feasible_w,
        best_feasible_eval=best_feasible_eval,
    )


def summarize_result(result: OptimizerResult) -> dict[str, float | int | str | bool]:
    best = result.best_eval
    return {
        "method": result.method,
        "auc_all": best.auc_all,
        "auc_gt": best.auc_gt,
        "cost": best.cost,
        "hardness_penalty": best.hardness_penalty,
        "size_penalty": best.size_penalty,
        "balance_penalty": best.balance_penalty,
        "gt_penalty": best.gt_penalty,
        "total_penalty": best.total_penalty,
        "feasible": bool(best.is_feasible),
        "selected_size": best.selected_size,
        "positive_ratio": best.positive_ratio,
        "minority_ratio": best.minority_ratio,
    }


def print_method_report(result: OptimizerResult) -> None:
    best = result.best_eval
    print(f"\n[{result.method}] Best subset report")
    print(f"  cost           : {best.cost:.4f}")
    print(f"  objective(size): {best.selected_size}")
    print(f"  auc_all        : {best.auc_all:.4f}")
    print(f"  auc_gt         : {best.auc_gt:.4f}")
    print(f"  selected_size  : {best.selected_size}")
    print(f"  positive_ratio : {best.positive_ratio:.4f}")
    print(
        "  penalties      : "
        f"hardness={best.hardness_penalty:.4f}, "
        f"size={best.size_penalty:.4f}, "
        f"balance={best.balance_penalty:.4f}, "
        f"gt={best.gt_penalty:.4f}"
    )
    print(f"  feasible       : {best.is_feasible}")


def compare_methods(sa_result: OptimizerResult, ga_result: OptimizerResult) -> tuple[str, str]:
    sa_best = sa_result.best_eval
    ga_best = ga_result.best_eval

    sa_feasible = sa_best.is_feasible
    ga_feasible = ga_best.is_feasible

    if sa_feasible and ga_feasible:
        if sa_best.cost < ga_best.cost:
            return sa_result.method, "Both best solutions are feasible; SA has lower size-first cost."
        if ga_best.cost < sa_best.cost:
            return ga_result.method, "Both best solutions are feasible; GA has lower size-first cost."
        if sa_best.auc_all < ga_best.auc_all:
            return sa_result.method, "Both best solutions are feasible with equal cost; SA has lower auc_all."
        if ga_best.auc_all < sa_best.auc_all:
            return ga_result.method, "Both best solutions are feasible with equal cost; GA has lower auc_all."
        return "Tie", "Both best solutions are feasible with equal cost and auc_all."

    if sa_feasible and not ga_feasible:
        return sa_result.method, "Only SA best solution is feasible (GA best has penalties)."

    if ga_feasible and not sa_feasible:
        return ga_result.method, "Only GA best solution is feasible (SA best has penalties)."

    if sa_result.best_feasible_eval is not None and ga_result.best_feasible_eval is not None:
        if sa_result.best_feasible_eval.cost < ga_result.best_feasible_eval.cost:
            return sa_result.method, "Neither global best is feasible; SA has better feasible size-first solution encountered."
        if ga_result.best_feasible_eval.cost < sa_result.best_feasible_eval.cost:
            return ga_result.method, "Neither global best is feasible; GA has better feasible size-first solution encountered."
        return "Tie", "Neither global best is feasible; both methods found equal feasible cost."

    if sa_result.best_feasible_eval is not None and ga_result.best_feasible_eval is None:
        return sa_result.method, "Neither global best is feasible; only SA found at least one feasible solution."

    if ga_result.best_feasible_eval is not None and sa_result.best_feasible_eval is None:
        return ga_result.method, "Neither global best is feasible; only GA found at least one feasible solution."

    return "No feasible winner", "No feasible solution found by either optimizer under current budget."


def save_artifacts(
    output_dir: Path,
    args: argparse.Namespace,
    sa_result: OptimizerResult,
    ga_result: OptimizerResult,
    winner: str,
    winner_reason: str,
) -> tuple[Path, Path]:
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    csv_path = output_dir / f"optimization_peeling_summary_{timestamp}.csv"
    json_path = output_dir / f"optimization_peeling_summary_{timestamp}.json"

    rows = [summarize_result(sa_result), summarize_result(ga_result)]
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "timestamp": timestamp,
        "args": vars(args),
        "winner": winner,
        "winner_reason": winner_reason,
        "simulated_annealing": {
            "best": asdict(sa_result.best_eval),
            "best_feasible": asdict(sa_result.best_feasible_eval) if sa_result.best_feasible_eval else None,
        },
        "genetic_algorithm": {
            "best": asdict(ga_result.best_eval),
            "best_feasible": asdict(ga_result.best_feasible_eval) if ga_result.best_feasible_eval else None,
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return csv_path, json_path


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    data_path = resolve_data_path_arg(args.data_path)
    output_dir = Path(args.output_dir).expanduser().resolve()

    X, y, gt_indices = load_data(data_path)
    X_pool, X_test, y_pool, y_test = split_pool_test(X, y, args.seed)

    gt_source = "metadata"
    if gt_indices is None:
        print("No gt_indices metadata found. Running Ground Truth Recovery via feature selection...")
        gt_indices = recover_gt_indices_from_fs(
            X_train=X_pool,
            y_train=y_pool,
            X_test=X_test,
            y_test=y_test,
            seed=args.seed,
        )
        gt_source = "recovered_fs"

    print("Loaded dataset and split into optimization pool/test")
    print(f"  data_path   : {data_path}")
    print(f"  X shape     : {X.shape}")
    print(f"  y classes   : {np.unique(y).tolist()}")
    print(f"  gt_count    : {gt_indices.size}")
    print(f"  gt_source   : {gt_source}")
    print(f"  pool size   : {X_pool.shape[0]}")
    print(f"  test size   : {X_test.shape[0]}")

    evaluate_subset = build_evaluator(X_pool, y_pool, X_test, y_test, gt_indices)

    rng_sa = np.random.default_rng(args.seed)
    rng_ga = np.random.default_rng(args.seed + 1)

    print("\nRunning Simulated Annealing...")
    sa_result = optimize_simulated_annealing(
        evaluate_subset=evaluate_subset,
        n_pool=X_pool.shape[0],
        rng=rng_sa,
        iters=args.sa_iters,
        temp0=args.sa_temp0,
        temp_min=args.sa_temp_min,
    )

    print("Running Genetic Algorithm...")
    ga_result = optimize_genetic_algorithm(
        evaluate_subset=evaluate_subset,
        n_pool=X_pool.shape[0],
        rng=rng_ga,
        pop_size=args.ga_pop_size,
        generations=args.ga_generations,
        cxpb=args.ga_cxpb,
        mutpb=args.ga_mutpb,
    )

    print_method_report(sa_result)
    print_method_report(ga_result)

    winner, winner_reason = compare_methods(sa_result, ga_result)
    print("\nComparison")
    print(f"  winner : {winner}")
    print(f"  reason : {winner_reason}")

    csv_path, json_path = save_artifacts(
        output_dir=output_dir,
        args=args,
        sa_result=sa_result,
        ga_result=ga_result,
        winner=winner,
        winner_reason=winner_reason,
    )

    print("\nSaved artifacts")
    print(f"  csv  : {csv_path}")
    print(f"  json : {json_path}")


if __name__ == "__main__":
    main()
