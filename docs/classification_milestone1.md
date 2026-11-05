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

The current preparation scripts do not create a globally fitted encoded dataset
or fitted preprocessing/model artifact. The initial classification experiment
remains required for Milestone 1 and has not yet been run.

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

## Planned validation approach (required for Milestone 1, not executed yet)

- Model: untuned `RandomForestClassifier` with defaults, plus `random_state=42` and `n_jobs=-1`.
- Evaluation design: grouped, stratified 5-fold cross-validation on training data.
- Group definition: identical cleaned selected predictors (target excluded) stay in one fold.
- Preprocessing fit: separately inside each fold.
- Primary metric: cancellation-class F1.
- Supporting metrics: accuracy, precision, recall, confusion matrix, ROC-AUC, average precision.
- Baseline comparator: simple dummy classifier on same folds.
- Final `test.csv`: reserved for Milestone 2 only.
