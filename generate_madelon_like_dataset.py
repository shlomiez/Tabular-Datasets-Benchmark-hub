#!/usr/bin/env python3
"""Generate Madelon-like synthetic datasets using sklearn.datasets.make_classification.

This script exposes a reusable function:
    generate_madelon_like_dataset(n_samples, n_features, n_informative, n_redundant, n_repeated, ...)

and a CLI for writing an .npz artifact that is immediately usable with scikit-learn.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.datasets import make_classification


def _validate_inputs(
    n_samples: int,
    n_features: int,
    n_informative: int,
    n_redundant: int,
    n_repeated: int,
    n_classes: int,
    n_clusters_per_class: int,
    class_sep: float,
    flip_y: float,
) -> None:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if n_features <= 0:
        raise ValueError("n_features must be > 0")
    if n_informative <= 0:
        raise ValueError("n_informative must be > 0")
    if n_redundant < 0:
        raise ValueError("n_redundant must be >= 0")
    if n_repeated < 0:
        raise ValueError("n_repeated must be >= 0")
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError(
            "n_informative + n_redundant + n_repeated must be <= n_features"
        )
    if n_classes <= 0:
        raise ValueError("n_classes must be > 0")
    if n_clusters_per_class <= 0:
        raise ValueError("n_clusters_per_class must be > 0")
    if n_classes * n_clusters_per_class > 2 ** n_informative:
        raise ValueError(
            "n_classes * n_clusters_per_class must be <= 2 ** n_informative"
        )
    if class_sep <= 0:
        raise ValueError("class_sep must be > 0")
    if not (0.0 <= flip_y < 1.0):
        raise ValueError("flip_y must be in [0, 1)")


def generate_madelon_like_dataset(
    n_samples: int,
    n_features: int,
    n_informative: int,
    n_redundant: int,
    n_repeated: int,
    *,
    n_classes: int = 2,
    n_clusters_per_class: int = 2,
    class_sep: float = 1.0,
    flip_y: float = 0.01,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a Madelon-like dataset for feature selection.

    Args:
        n_samples: Number of observations.
        n_features: Total number of observed features.
        n_informative: Number of informative (useful) features.
        n_redundant: Number of redundant features built as linear combinations
            of the informative features.
        n_repeated: Number of duplicated features drawn from the informative
            and redundant features.
        n_classes: Number of target classes.
        n_clusters_per_class: Number of clusters per class.
        class_sep: Factor multiplying the hypercube size, controls class
            separability.
        flip_y: Fraction of labels randomly flipped to add label noise.
        random_state: Seed for reproducibility.

    Returns:
        X: Feature matrix of shape (n_samples, n_features), dtype float64.
        y: Integer target array of shape (n_samples,), dtype int64.
        gt_indices: Sorted ground-truth indices (informative + redundant +
            repeated features), shape (n_informative + n_redundant + n_repeated,),
            dtype int64.
    """
    _validate_inputs(
        n_samples,
        n_features,
        n_informative,
        n_redundant,
        n_repeated,
        n_classes,
        n_clusters_per_class,
        class_sep,
        flip_y,
    )

    # shuffle=False keeps features ordered as [informative | redundant | repeated | noise],
    # so the ground-truth support is simply the first block of columns.
    x_matrix, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=flip_y,
        shuffle=False,
        random_state=random_state,
    )

    gt_indices = np.arange(n_informative + n_redundant + n_repeated, dtype=np.int64)

    return np.asarray(x_matrix, dtype=np.float64), np.asarray(y, dtype=np.int64), gt_indices


def _default_output_name(n_samples: int, n_features: int, n_informative: int) -> str:
    return f"madelon_like_dataset({n_samples},{n_features},{n_informative}).npz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Madelon-like synthetic data via sklearn.datasets.make_classification and save as .npz"
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=5000)
    parser.add_argument("--n-informative", type=int, default=5)
    parser.add_argument("--n-redundant", type=int, default=20)
    parser.add_argument("--n-repeated", type=int, default=25)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--n-clusters-per-class", type=int, default=2)
    parser.add_argument(
        "--class-sep",
        type=float,
        default=1.0,
        help="Class separability factor, must be > 0",
    )
    parser.add_argument(
        "--flip-y",
        type=float,
        default=0.01,
        help="Fraction of labels randomly flipped, must be in [0, 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .npz path. If omitted, uses "
            "data/madelon_like_dataset(n_samples,n_features,n_informative).npz"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    x_matrix, y, gt_indices = generate_madelon_like_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        n_redundant=args.n_redundant,
        n_repeated=args.n_repeated,
        n_classes=args.n_classes,
        n_clusters_per_class=args.n_clusters_per_class,
        class_sep=args.class_sep,
        flip_y=args.flip_y,
        random_state=args.seed,
    )

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_name = _default_output_name(args.n_samples, args.n_features, args.n_informative)
        out_path = (Path("data") / out_name).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=x_matrix,
        y=y,
        gt_indices=gt_indices,
        n_samples=np.int64(args.n_samples),
        n_features=np.int64(args.n_features),
        n_informative=np.int64(args.n_informative),
        n_redundant=np.int64(args.n_redundant),
        n_repeated=np.int64(args.n_repeated),
        n_classes=np.int64(args.n_classes),
        n_clusters_per_class=np.int64(args.n_clusters_per_class),
        class_sep=np.float64(args.class_sep),
        flip_y=np.float64(args.flip_y),
        seed=np.int64(args.seed),
    )

    class_counts = np.bincount(y)
    class_distribution = ", ".join(
        f"{cls}: {count / y.size:.4f}" for cls, count in enumerate(class_counts)
    )

    print("Saved synthetic dataset:")
    print(f"- path: {out_path}")
    print(f"- X shape: {x_matrix.shape}, dtype={x_matrix.dtype}")
    print(f"- y shape: {y.shape}, dtype={y.dtype}, class_distribution={{{class_distribution}}}")
    print(f"- gt_indices count: {gt_indices.size}")


if __name__ == "__main__":
    main()
