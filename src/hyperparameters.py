"""Dataset-specific hyperparameter policies."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.compat import Path


def get_hyperparameters(dataset_name: str, n_train_samples: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return STG, LSPIN, and ExtraTrees hyperparameters by dataset group."""
    _ = n_train_samples
    name = str(dataset_name).strip().lower()

    lss_datasets = {"breast", "colon", "leukemia", "smk-can-187"}
    med_large_datasets = {"madelon", "relathe"}

    if name in lss_datasets:
        stg_params = {
            "epochs": 300,
            "batch_size": 16,
            "learning_rate": 0.5,
            "lam": 1.0,
            "activation": "tanh",
            "optimizer": "SGD",
            "hidden_dims": (128, 64),
            "sigma": 0.5,
        }

        lspin_params = {
            "epochs": 300,
            "batch_size": 16,
            "learning_rate": 0.05,
            "lam": 1.0,
            "activation_gating": "tanh",
            "activation_pred": "tanh",
            "optimizer": "SGD",
            "hidden_layers": (100, 50, 30),
            "gating_hidden_layers": (100, 10),
            "sigma": 0.5,
        }

    elif name in med_large_datasets:
        if name == "relathe":
            stg_epochs = 200
            stg_batch_size = 40
            lspin_batch_size = 40
            lam_stg = 10.0
            lam_lspin = 100.0
        else:
            stg_epochs = 1000
            stg_batch_size = 200
            lspin_batch_size = 200
            lam_stg = 1.0
            lam_lspin = 10.0

        stg_params = {
            "epochs": stg_epochs,
            "batch_size": stg_batch_size,
            "learning_rate": 0.05,
            "lam": lam_stg,
            "activation": "tanh",
            "optimizer": "SGD",
            "hidden_dims": (128, 64),
            "sigma": 0.5,
        }

        lspin_params = {
            "epochs": 500,
            "batch_size": lspin_batch_size,
            "learning_rate": 0.1,
            "lam": lam_lspin,
            "activation_gating": "tanh",
            "activation_pred": "tanh",
            "optimizer": "SGD",
            "hidden_layers": (100, 50, 30),
            "gating_hidden_layers": (500,),
            "sigma": 0.5,
        }

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            "Expected one of: Breast, colon, leukemia, SMK-CAN-187, madelon, RELATHE."
        )

    etree_params = {"n_estimators": 100, "max_depth": 2}
    return stg_params, lspin_params, etree_params

def output_hyperparameters_to_yaml(path: Path, stg_params: dict[str, Any], lspin_params: dict[str, Any], etree_params: dict[str, Any]) -> None:
    """Output hyperparameters to a YAML file for record-keeping."""
    import yaml

    all_params = {
        "STG": stg_params,
        "LSPIN": lspin_params,
        "ExtraTrees": etree_params,
    }

    with path.open("w") as f:
        yaml.dump(all_params, f, default_flow_style=False)

def default_lambda_ranges_by_dataset() -> dict[str, dict[str, list[float]]]:
    """Return paired STG/LSPIN lambda ranges used for tuning."""
    return {
        "madelon": {
            "stg": np.geomspace(0.9, 3, num=7).tolist(),
            "lspin": np.geomspace(0.01, 10, num=7).tolist(),
        },
        "RELATHE": {
            "stg": np.geomspace(10, 17, num=7).tolist(),
            "lspin": np.geomspace(0.03, 200, num=7).tolist(),
        },
    }
