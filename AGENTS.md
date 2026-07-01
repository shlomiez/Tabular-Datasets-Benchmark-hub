# AGENTS.md

## Project

This repository benchmarks tabular feature-selection pipelines. The main entry point is `main.py`, which reads `config.yml` and runs the pipeline in `src/pipeline.py`.

## Working Rules

- Prefer small, local changes that preserve the existing pipeline behavior.
- Do not edit generated artifacts in `output/` unless the task explicitly asks for it.
- Keep changes ASCII-only unless the file already contains Unicode.
- If you touch the experiment flow, validate the narrowest affected path before expanding scope.

## Common Commands

- Activate the local environment: `source .venv/bin/activate`
- Run the main pipeline: `python main.py`
- Plot results from an output folder: `python plot_results.py --concrete` or the script's normal CLI usage for a specific output directory.

## Important Paths

- `config.yml` controls dataset selection, selector choices, and run settings.
- `data/` contains the benchmark datasets.
- `output/` stores per-run artifacts, CSV summaries, and plots.
- `tabicl_cache/` is a runtime cache directory for tabICL/tabICLv2 runs.

## Output Conventions

- The main pipeline writes `iterative_feature_curve_summary.csv`, `iterative_feature_curve_fold_results.csv`, and `iterative_feature_curve_loss_history.csv` into the run output directory.
- Concrete Autoencoder runs also write per-dataset `*_concrete_metrics.csv` files, which `plot_results.py --concrete` aggregates into `concrete_metrics_summary.csv` and `plots_concrete_summary/`.

## Repo Notes

- `src/feature_selection.py` can resolve `project-featselectlib` from either the repo root or the parent directory, so the library may live outside this repo.
- If you need to inspect run history, prefer the latest timestamped folder under `output/`.