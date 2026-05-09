# Comparative Evaluation of STG and LSPIN for Iterative Feature Reduction

## Abstract
This report analyzes the experimental outputs produced by main.ipynb, whose purpose is to study how different feature-selection algorithms and retained-feature ratios affect performance on datasets of varying difficulty levels (hard, medium, and easy). The notebook benchmarks two embedded feature-selection approaches, STG and LSPIN, against a full-feature ExtraTrees baseline across six high-dimensional classification datasets. The protocol uses 5-fold stratified cross-validation and evaluates retained-feature ratios from 100% to 5%. Across 66 dataset-ratio summary conditions, mean performance deltas relative to baseline were negative for both methods (STG AUC delta: -0.0081; LSPIN AUC delta: -0.0160). Positive AUC deltas occurred in 24.2% of conditions for STG and 33.3% for LSPIN, indicating localized benefits rather than consistent global gains. The strongest observed improvements were STG on leukemia at 5% retained features (+0.0160 AUC) and LSPIN on SMK-CAN-187 at 60% retained features (+0.0132 AUC). Results suggest that selector efficacy is dataset- and ratio-dependent, with no universal improvement over a strong tree-based baseline.

## 1. Introduction
Feature selection is central in high-dimensional, low-sample learning, where reducing irrelevant features may improve generalization, interpretability, and computational efficiency. The purpose of this experiment is to examine how feature-selection algorithm choice and retained-feature ratio influence model performance across datasets with different difficulty levels. In particular, the study compares two feature selection mechanisms, STG and LSPIN, under a unified downstream classifier (ExtraTrees), and contrasts them with a no-selection baseline trained on all available features.

The objective is not only to identify whether selectors can outperform baseline, but also to characterize how their behavior changes across hard, medium, and easy datasets and at which retained-feature ratios gains are most likely.

## 2. Materials and Methods

### 2.1 Datasets
Six datasets were evaluated:
- Breast
- madelon
- SMK-CAN-187
- colon
- leukemia
- RELATHE

### 2.2 Experimental Design
- Cross-validation: StratifiedKFold (5 folds)
- Feature-retention ratios: 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05
- Methods compared:
  - Baseline ExtraTrees on full features
  - STG-selected features + ExtraTrees
  - LSPIN-selected features + ExtraTrees
- Reported metrics: AUC and accuracy (train and test)
- Primary comparative quantity: mean fold-level delta against baseline at each dataset-ratio condition

### 2.3 Artifacts Analyzed
This report is based on:
- output/iterative_feature_curve_summary.csv
- output/iterative_feature_curve_fold_results.csv
- output/iterative_feature_curve_loss_history.csv

and notebook-generated plots under output/plots.

## 3. Results

### 3.1 Global Aggregate Outcomes
Across 66 summary rows (6 datasets x 11 ratios):
- Mean STG AUC delta vs baseline: -0.0081
- Mean LSPIN AUC delta vs baseline: -0.0160
- Mean STG accuracy delta vs baseline: -0.0133
- Mean LSPIN accuracy delta vs baseline: -0.0175

Interpretation: averaged over all dataset-ratio conditions, both selectors underperform the baseline.

### 3.2 Frequency of Positive AUC Improvement
- STG positive AUC delta in 16/66 conditions (24.2%)
- LSPIN positive AUC delta in 22/66 conditions (33.3%)

This indicates non-uniform behavior, where improvements appear intermittently rather than systematically.

### 3.3 Per-Dataset Mean AUC Delta
| Dataset | Avg STG AUC delta | Avg LSPIN AUC delta |
|---|---:|---:|
| Breast | -0.0165 | -0.0220 |
| colon | -0.0172 | -0.0153 |
| leukemia | +0.0069 | +0.0003 |
| madelon | -0.0174 | -0.0294 |
| RELATHE | -0.0031 | -0.0326 |
| SMK-CAN-187 | -0.0011 | +0.0030 |

Per-dataset positive-counts across 11 ratios:
- Breast: STG 2, LSPIN 2
- colon: STG 2, LSPIN 2
- leukemia: STG 8, LSPIN 6
- madelon: STG 0, LSPIN 0
- RELATHE: STG 0, LSPIN 3
- SMK-CAN-187: STG 2, LSPIN 8

### 3.4 Best Observed Improvements
- Best STG point: leukemia at 5% retained features, AUC delta = +0.0160
- Best LSPIN point: SMK-CAN-187 at 60% retained features, AUC delta = +0.0132

### 3.5 Train-Test Separation (AUC)
Average train-test AUC gaps (train minus test):
- Baseline: 0.1365
- STG pipeline: 0.1374
- LSPIN pipeline: 0.1465

These gaps indicate notable overfitting pressure across settings, with the largest average gap under the LSPIN pipeline.

## 4. Discussion
The evidence supports three main conclusions:

1. No global dominance over baseline.
Both STG and LSPIN show negative mean deltas across all conditions when aggregated, despite occasional improvements.

2. Strong heterogeneity by dataset and ratio.
Leukemia and SMK-CAN-187 are the most favorable environments for selector gains, whereas madelon exhibits no positive AUC delta for either selector under tested settings.

3. Ratio sensitivity is substantial.
Best-performing points occur at different retained-feature levels (for example, 5% for STG on leukemia and 60% for LSPIN on SMK-CAN-187), suggesting that a single fixed ratio is unlikely to be optimal across datasets.

## 5. Limitations
- Single-seed evaluation limits robustness claims.
- No confidence intervals or formal hypothesis tests are included in current outputs.
- Hyperparameters are manually grouped by dataset families, not optimized with a uniform automated search protocol.
- Fold-level losses are extensive, and compact diagnostics (for example, variance summaries) are not yet exported.

## 6. Conclusion
Under the current protocol, STG and LSPIN provide selective, context-dependent gains but do not consistently surpass a full-feature ExtraTrees baseline in aggregate. The most promising future direction is to treat retained-feature ratio and selector hyperparameters as dataset-specific tuning variables, accompanied by repeated-seed and inferential statistical analysis.

## 7. Reproducibility and Reporting Notes
- Notebook execution completed with output artifact generation and repeatability checks.
- Core outputs are stored in CSV form and can be re-analyzed without rerunning training.
- Recommended additions for publication readiness:
  1. Multi-seed repeated CV.
  2. Fold-level paired significance testing (delta distributions).
  3. Confidence intervals around reported means.
  4. Explicit model-selection protocol to avoid post-hoc ratio selection bias.
