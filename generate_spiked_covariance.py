#!/usr/bin/env python3
"""Generate synthetic datasets from a sparse spiked covariance model.

This script exposes a reusable function:
    generate_spiked_covariance_dataset(n_samples, p_features, n_spikes, n_active_features, snr)

and a CLI for writing an .npz artifact that is immediately usable with scikit-learn.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x_clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _validate_inputs(
    n_samples: int,
    p_features: int,
    n_spikes: int,
    n_active_features: int,
    snr: float,
    positive_rate: float,
) -> None:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if p_features <= 0:
        raise ValueError("p_features must be > 0")
    if n_spikes <= 0:
        raise ValueError("n_spikes must be > 0")
    if n_active_features <= 0:
        raise ValueError("n_active_features must be > 0")
    if n_active_features > p_features:
        raise ValueError("n_active_features must be <= p_features")
    if n_spikes > n_active_features:
        raise ValueError(
            "n_spikes must be <= n_active_features to keep row-orthogonality in the sparse spike matrix"
        )
    if snr <= 0:
        raise ValueError("snr must be > 0")
    if not (0.0 < positive_rate < 1.0):
        raise ValueError("positive_rate must be in (0, 1)")


def _build_sparse_orthogonal_rho(
    p_features: int,
    n_spikes: int,
    n_active_features: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sparse row-orthonormal spikes and return (Rho, gt_indices)."""
    gt_indices = np.sort(rng.choice(p_features, size=n_active_features, replace=False))

    # Build an orthonormal basis on active coordinates and embed into full feature space.
    active_block = rng.normal(loc=0.0, scale=1.0, size=(n_active_features, n_spikes))
    q_matrix, _ = np.linalg.qr(active_block)
    rho_active = q_matrix[:, :n_spikes].T  # Shape: (n_spikes, n_active_features)

    rho = np.zeros((n_spikes, p_features), dtype=float)
    rho[:, gt_indices] = rho_active
    return rho, gt_indices


def _compute_sigma_for_snr(signal: np.ndarray, noise: np.ndarray, snr: float) -> float:
    """Set sigma so ||signal||_F^2 / ||sigma * noise||_F^2 = snr."""
    signal_power = float(np.sum(signal ** 2))
    noise_power = float(np.sum(noise ** 2))
    if signal_power <= 0.0:
        raise ValueError("Degenerate signal power (0). Increase dimensions or check spike generation.")
    if noise_power <= 0.0:
        raise ValueError("Degenerate noise power (0).")
    return np.sqrt(signal_power / (snr * noise_power))


def generate_spiked_covariance_dataset(
    n_samples: int,
    p_features: int,
    n_spikes: int,
    n_active_features: int,
    snr: float,
    *,
    positive_rate: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a sparse spiked-covariance dataset for feature selection.

    Args:
        n_samples: Number of observations.
        p_features: Total number of observed features.
        n_spikes: Number of latent spikes/components.
        n_active_features: Number of active observed features (ground truth support).
        snr: Desired global signal-to-noise ratio in linear scale.
        positive_rate: Target fraction of positive labels in y.
        random_state: Seed for reproducibility.

    Returns:
        X: Feature matrix of shape (n_samples, p_features), dtype float64.
        y: Binary target array of shape (n_samples,), dtype int64.
        gt_indices: Sorted active feature indices, shape (n_active_features,), dtype int64.
    """
    _validate_inputs(n_samples, p_features, n_spikes, n_active_features, snr, positive_rate)
    rng = np.random.default_rng(random_state)

    # 1) Ground-truth sparse spike matrix Rho with row-orthonormal spikes.
    rho, gt_indices = _build_sparse_orthogonal_rho(p_features, n_spikes, n_active_features, rng)

    # 2) Latent factors V.
    v_matrix = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_spikes))

    # 3) Gaussian white noise Z.
    z_matrix = rng.normal(loc=0.0, scale=1.0, size=(n_samples, p_features))

    # 4) Construct X with dynamic sigma from requested SNR.
    signal = v_matrix @ rho
    sigma = _compute_sigma_for_snr(signal=signal, noise=z_matrix, snr=snr)
    x_matrix = signal + sigma * z_matrix

    # 5) Target y strictly from latent factors V.
    beta = rng.normal(loc=0.0, scale=1.0, size=(n_spikes,))
    logits = v_matrix @ beta
    logits = logits - np.mean(logits)
    probs = _sigmoid(logits)

    # Quantile thresholding gives a controllable finite-sample positive class rate.
    threshold = float(np.quantile(probs, 1.0 - positive_rate))
    y = (probs >= threshold).astype(np.int64)

    return np.asarray(x_matrix, dtype=np.float64), np.asarray(y, dtype=np.int64), np.asarray(gt_indices, dtype=np.int64)


def _empirical_snr(signal: np.ndarray, sigma: float, noise: np.ndarray) -> float:
    signal_power = float(np.sum(signal ** 2))
    noise_power = float(np.sum((sigma * noise) ** 2))
    if noise_power == 0.0:
        return float("inf")
    return signal_power / noise_power


def _default_output_name(n_samples: int, p_features: int, n_active_features: int, snr: float) -> str:
    return f"spiked_covariance_dataset({n_samples},{p_features},{n_active_features},snr{snr:g}).npz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sparse spiked-covariance synthetic data and save as .npz"
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--p-features", type=int, default=5000)
    parser.add_argument("--n-spikes", type=int, default=5)
    parser.add_argument("--n-active-features", type=int, default=50)
    parser.add_argument("--snr", type=float, default=2.0, help="Linear-scale SNR, must be > 0")
    parser.add_argument(
        "--positive-rate",
        type=float,
        default=0.5,
        help="Target positive class fraction in y, must be in (0,1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .npz path. If omitted, uses "
            "data/spiked_covariance_dataset(n_samples,p_features,n_active_features).npz"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    x_matrix, y, gt_indices = generate_spiked_covariance_dataset(
        n_samples=args.n_samples,
        p_features=args.p_features,
        n_spikes=args.n_spikes,
        n_active_features=args.n_active_features,
        snr=args.snr,
        positive_rate=args.positive_rate,
        random_state=args.seed,
    )

    # Reconstruct components for user-facing diagnostics.
    rng = np.random.default_rng(args.seed)
    rho, _ = _build_sparse_orthogonal_rho(
        p_features=args.p_features,
        n_spikes=args.n_spikes,
        n_active_features=args.n_active_features,
        rng=rng,
    )
    v_matrix = rng.normal(loc=0.0, scale=1.0, size=(args.n_samples, args.n_spikes))
    z_matrix = rng.normal(loc=0.0, scale=1.0, size=(args.n_samples, args.p_features))
    signal = v_matrix @ rho
    sigma = _compute_sigma_for_snr(signal=signal, noise=z_matrix, snr=args.snr)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_name = _default_output_name(args.n_samples, args.p_features, args.n_active_features, args.snr)
        out_path = (Path("data") / out_name).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=x_matrix,
        y=y,
        gt_indices=gt_indices,
        sigma=np.float64(sigma),
        snr=np.float64(args.snr),
        n_samples=np.int64(args.n_samples),
        p_features=np.int64(args.p_features),
        n_spikes=np.int64(args.n_spikes),
        n_active_features=np.int64(args.n_active_features),
        positive_rate=np.float64(args.positive_rate),
        seed=np.int64(args.seed),
    )

    empirical_snr = _empirical_snr(signal=signal, sigma=sigma, noise=z_matrix)
    active_rate = float(np.mean(y))

    print("Saved synthetic dataset:")
    print(f"- path: {out_path}")
    print(f"- X shape: {x_matrix.shape}, dtype={x_matrix.dtype}")
    print(f"- y shape: {y.shape}, dtype={y.dtype}, positive_rate_empirical={active_rate:.4f}")
    print(f"- gt_indices count: {gt_indices.size}")
    print(f"- requested snr: {args.snr:.6f}, empirical snr: {empirical_snr:.6f}")


if __name__ == "__main__":
    main()
