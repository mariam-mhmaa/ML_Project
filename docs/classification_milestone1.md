# Classification Milestone 1: data preparation baseline

This note documents the **booking-cancellation classification** data preparation used before any model training.

## Prediction scenario and snapshot limitation

The intended prediction time is **booking creation** and the target is `IsCanceled` (`1` = canceled, positive class).  
This baseline keeps all rows from `data/preprocessed/train.csv` and does not remove no-shows from the positive class.

Important limitation:

- The source dataset stores values extracted relative to the day before arrival, with change logs where available.
- It does **not** provide verified booking-creation snapshots.
- This implementation is a booking-time-oriented baseline that uses conservative feature choices and explicit assumptions.
- It is **not** proof of deployment-ready booking-time performance.

For mutable retained fields, this baseline assumes their recorded values reasonably approximate what could have been available at booking creation. Historical booking outcomes are also assumed to have been known at that time. Excluding selected columns reduces risk but does not eliminate every timing-related leakage risk.

## Input columns and classification-specific exclusions

Prepared predictors (23):

- `LeadTime`
- `ArrivalDateYear`
- `ArrivalDateMonth`
- `ArrivalDateWeekNumber`
- `ArrivalDateDayOfMonth`
- `StaysInWeekendNights`
- `StaysInWeekNights`
- `Adults`
- `Children`
- `Babies`
- `Meal`
- `Country`
- `MarketSegment`
- `DistributionChannel`
- `IsRepeatedGuest`
- `PreviousCancellations`
- `PreviousBookingsNotCanceled`
- `ReservedRoomType`
- `Agent`
- `Company`
- `CustomerType`
- `RequiredCarParkingSpaces`
- `TotalOfSpecialRequests`

Classification predictors exclude:

- `ReservationStatus` (reveals eventual outcome)
- `ReservationStatusDate` (outcome-related timing information)
- `AssignedRoomType` (assignment may occur after booking creation)
- `BookingChanges` (cumulative changes after creation)
- `DaysInWaitingList` (completed waiting duration may be unavailable at creation)
- `DepositType` (transaction-derived category with uncertain booking-time availability)
- `ADR` (transaction-derived rate with uncertain booking-time availability)

## Deterministic cleanup and missing-value rules

Applied before any fold-based learning:

- Strip leading/trailing whitespace from text values (keep internal spaces).
- `Children`: marker-like missing values are replaced with `0` (assumption: zero is common, missingness is rare).
- `Country`: marker-like missing values are replaced with `Unknown`.
- `Agent`, `Company`: `NULL` is replaced with `NotApplicable`.
- Non-missing `Agent`/`Company` values are treated as category IDs.
- Numeric predictors and target are parsed as numeric with explicit failure checks.
- `IsRepeatedGuest` is kept as binary 0/1.
- Unexpected missing markers in other columns are reported rather than silently filled.
- No extra row exclusions, outlier clipping, country remapping, or duplicate removal are applied here.

## Encoding and reusable preprocessing

- One-hot encoding is configured for:
  - `ArrivalDateMonth`, `Meal`, `Country`, `MarketSegment`, `DistributionChannel`,
    `ReservedRoomType`, `Agent`, `Company`, `CustomerType`
- `handle_unknown="ignore"` is used so unseen categories do not fail transform.
- No numeric scaling is applied for the planned Random Forest baseline.
- Preprocessing is provided as reusable scikit-learn `ColumnTransformer`/`Pipeline` code.
- Learned transformations are intended to be fitted inside each training fold during later cross-validation.

The preparation scripts do not create a globally fitted encoded dataset.
Milestone 1 now includes a completed training-only grouped cross-validation
experiment and a separately saved full-training pipeline artifact.

## Scripts and run commands (from repository root)

```powershell
python src/preprocessing/classification/prepare_classification_data.py
python src/classification/audit_training_data.py
```

## Generated outputs

- Prepared dataset: `data/preprocessed/classification/train_classification.csv`
- Audit and logs:
  - `outputs/classification/audit_summary.csv`
  - `outputs/classification/target_distribution.csv`
  - `outputs/classification/whitespace_cleanup_counts.csv`
  - `outputs/classification/missing_value_replacements.csv`
  - `outputs/classification/column_inventory.csv`
  - `outputs/classification/categorical_cardinality.csv`
  - `outputs/classification/repeated_selected_input_groups.csv`
  - `outputs/classification/conflicting_label_groups.csv`
  - `outputs/classification/focused_cancellation_summary.csv`
  - `outputs/classification/numeric_dtype_check.csv`

## Remaining data-quality questions

- How accurately do day-before-arrival values represent booking-time information for mutable fields?
- Should additional time-sensitive columns be excluded after model-error analysis?
- Do conflicting-label groups reflect real repeated bookings or unresolved data quality issues?
- Are rare `Agent` and `Company` categories stable enough for production-style generalization?

## Milestone 1 untuned experiment implementation

The baseline experiment is implemented in:

```powershell
python src/classification/run_milestone1_baseline.py
```

### Model-selection justification

- Main baseline model: `RandomForestClassifier(random_state=42, n_jobs=-1)` with all other defaults from scikit-learn 1.9.1.
- Reference model: `DummyClassifier(strategy="prior")`.

The dummy model ignores booking characteristics and predicts using only class prevalence. It provides a minimum reference to check whether the random forest adds useful signal.

### Validation and grouping strategy

- Data used: `data/preprocessed/classification/train_classification.csv` only.
- Five-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
- Group rule: rows with identical cleaned selected predictors (target excluded) stay in one fold, including conflicting-label groups.
- Fold assignments are generated once and reused for both models.
- For each fold, preprocessing and model fitting are performed from scratch on that fold's training split only.
- Per-fold training metrics are recorded as overfitting diagnostics.

### Metric definitions and primary summary

- Primary metric: cancellation-class F1 (`f1_canceled`).
- Supporting metrics: accuracy, cancellation-class precision, cancellation-class recall, confusion-matrix counts, ROC-AUC, average precision.
- Ranking metrics (ROC-AUC, average precision) are computed from predicted cancellation probabilities.
- For undefined precision/F1 edge cases, `zero_division=0` is used. This occurs for the dummy-prior model because it predicts only class 0 in all folds.

### Cross-validation results (training data only)

Mean ± standard deviation across folds:

- Random forest:
  - F1 (canceled): **0.8048 ± 0.0073**
  - Accuracy: 0.8470 ± 0.0046
  - Precision (canceled): 0.8621 ± 0.0048
  - Recall (canceled): 0.7547 ± 0.0130
  - ROC-AUC: 0.9278 ± 0.0028
  - Average precision: 0.9119 ± 0.0035
- Dummy prior:
  - F1 (canceled): **0.0000 ± 0.0000**
  - Accuracy: 0.5822 ± 0.0000
  - Precision/Recall (canceled): 0.0000 / 0.0000
  - ROC-AUC: 0.5000
  - Average precision: 0.4178

Pooled out-of-fold confusion matrix (random forest):

- TN: 33,672
- FP: 3,195
- FN: 6,492
- TP: 19,970

Interpretation (descriptive, not causal):

- The random forest strongly outperforms the dummy baseline on F1, recall, ROC-AUC, and average precision, indicating non-trivial predictive value from the retained features.
- Fold variability is moderate; no single fold dominates the result.
- Precision is higher than recall for the cancellation class, showing a conservative positive prediction tendency at the default threshold.

### Training-versus-validation differences

Random-forest training scores are substantially higher than validation scores (for example training F1 ≈ 0.99 vs validation F1 ≈ 0.80), indicating notable overfitting under default settings. This is expected for an untuned baseline and motivates structured tuning only in later milestones.

### Reusable fitted baseline artifact

After cross-validation, a fresh unchanged pipeline is fitted on all training rows and saved to:

- `artifacts/classification/random_forest_baseline_pipeline.joblib`

Artifact reload check confirms identical predictions and probabilities on a held sample after reload (`outputs/classification/artifact_reload_check_sample.csv`).

### Reproducibility records

- Input hash, source commit, seed, fold setup, feature list, model parameters, and package versions are saved in:
  - `outputs/classification/experiment_metadata.json`
- Historical-run limitation: the existing metadata file records one Git HEAD
  (`source_commit`) but does not include historical run timestamps, source-file
  hashes, or worktree dirty/untracked state. Therefore, that HEAD alone does
  not uniquely identify all source code used in the already-generated run.
- Fold assignments:
  - `outputs/classification/cv_fold_assignments.csv`
- Per-fold and pooled metrics:
  - `outputs/classification/cv_per_fold_validation_metrics.csv`
  - `outputs/classification/cv_per_fold_training_metrics.csv`
  - `outputs/classification/cv_validation_summary_mean_std.csv`
  - `outputs/classification/cv_oof_pooled_metrics.csv`
  - `outputs/classification/rf_vs_dummy_comparison.csv`
- OOF diagnostics:
  - `outputs/classification/oof_confusion_matrix_random_forest.png`
  - `outputs/classification/oof_confusion_matrix_dummy_prior.png`
  - `outputs/classification/oof_roc_curve_comparison.png`
  - `outputs/classification/oof_precision_recall_curve_comparison.png`

### Remaining Milestone 2 work

Milestone 1 uses grouped cross-validation and training data only. The reserved
`data/preprocessed/test.csv` file remains untouched for Milestone 2.

Correct final evaluation order:

1. Compare models and tune hyperparameters using training-only grouped validation.
2. If needed, select the operating threshold using training-only validation, with a justified precision/recall or business-cost objective.
3. Freeze the complete preprocessing, model, and threshold.
4. Evaluate the frozen workflow on the reserved final test set.
5. Report the results without test-driven adjustments.
