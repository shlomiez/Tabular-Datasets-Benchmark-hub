# Notebook Report: main.ipynb

Date: 2026-04-14
Notebook: main.ipynb

## 1) Executive Summary

This notebook runs a feature-reduction benchmark over six datasets using:
- Baseline ExtraTrees (full feature set)
- STG feature selection + ExtraTrees evaluation
- LSPIN feature selection + ExtraTrees evaluation

The experiment uses 5-fold stratified CV and evaluates feature-retention ratios from 100% down to 5%.

At a global level (66 dataset-ratio summary rows):
- Mean STG AUC delta vs baseline: -0.0081
- Mean LSPIN AUC delta vs baseline: -0.0160
- Mean STG accuracy delta vs baseline: -0.0133
- Mean LSPIN accuracy delta vs baseline: -0.0175

Interpretation: on average, both selectors underperform the baseline across all tested ratios, but there are ratio-specific gains on selected datasets.

## 2) Notebook Structure

- Total cells: 22
- Markdown cells: 10
- Code cells: 12
- Executed code cells: 12 (execution counts 2 through 13)

High-level workflow in notebook:
1. Install/import dependencies and configure runtime paths.
2. Resolve and load .mat datasets.
3. Define preprocessing, model fitting, feature selection, and CV evaluation functions.
4. Run dataset-level experiments over feature ratios.
5. Generate and save train/test/loss plots.
6. Export CSV artifacts.
7. Run repeatability file-existence checks.

## 3) Data and Experimental Setup

Datasets used:
- Breast
- madelon
- SMK-CAN-187
- colon
- leukemia
- RELATHE

Evaluation design:
- 5-fold StratifiedKFold
- Ratios: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
- Metrics: AUC and accuracy (train and test)
- Comparison metric in summary: delta of selector metric vs baseline metric

## 4) Artifact Outputs Verified

The notebook writes:
- output/iterative_feature_curve_summary.csv
- output/iterative_feature_curve_fold_results.csv
- output/iterative_feature_curve_loss_history.csv
- output/plots/*.png (dataset-specific train/test/loss curves)
- output/checkpoint_*.pkl (resume checkpoints)

Repeatability checks in notebook verify the three CSV files exist.

## 5) Per-Dataset AUC Delta Findings

From output/iterative_feature_curve_summary.csv:

| Dataset | Avg STG AUC delta | Avg LSPIN AUC delta | Best STG delta (feature %) | Best LSPIN delta (feature %) |
|---|---:|---:|---:|---:|
| Breast | -0.0165 | -0.0220 | +0.0050 (50%) | +0.0112 (40%) |
| colon | -0.0172 | -0.0153 | +0.0050 (90%) | +0.0088 (60%) |
| leukemia | +0.0069 | +0.0003 | +0.0160 (10%) | +0.0120 (20%) |
| madelon | -0.0174 | -0.0294 | +0.0031 (80%) | +0.0046 (100%) |
| RELATHE | -0.0031 | -0.0326 | +0.0024 (70%) | +0.0035 (80%) |
| SMK-CAN-187 | -0.0011 | +0.0030 | +0.0089 (20%) | +0.0132 (60%) |

Notes:
- STG average AUC delta is positive for leukemia only.
- LSPIN average AUC delta is slightly positive for SMK-CAN-187 and nearly neutral for leukemia.
- The strongest observed positive STG point is leukemia at 10% features (+0.0160).
- The strongest observed positive LSPIN point is SMK-CAN-187 at 60% features (+0.0132).

## 6) Technical Observations

- Baseline train metrics are often at or near 1.0 on several datasets, indicating high model fit capacity and potential overfitting pressure in high-dimensional low-sample settings.
- The notebook includes early-stopping logic in STG and LSPIN training loops.
- Checkpointing is implemented per dataset to support resume behavior and avoid recomputing completed folds.

## 7) Risks and Limitations

- Many long loss-history arrays in fold-level outputs make analysis heavy and CSV files large.
- Several datasets show selector degradation relative to baseline at most ratios.
- Hyperparameters are manually grouped by dataset type and may not be globally optimal.

## 8) Recommended Next Steps

1. Add confidence intervals or standard deviations across folds for deltas in the summary export.
2. Run repeated CV seeds (not only one seed) to estimate stability of ratio-level improvements.
3. Add statistical significance checks (for example, paired fold-wise tests on AUC deltas).
4. Use automated hyperparameter search per dataset for STG/LSPIN before final comparison.
5. Add a compact final ranking table per dataset (best method, best ratio, and margin).
