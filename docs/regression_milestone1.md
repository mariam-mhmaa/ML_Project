# Regression Milestone 1: ADR preparation and untuned baseline

## Scope and prediction point

Predict the recorded **Average Daily Rate (`ADR`, EUR per staying night)** from
booking attributes assumed available **when the booking is created**. This is a
booking-creation-oriented research baseline, with the snapshot limitations below.
It estimates an observed rate; it does not optimize prices or predict total revenue.

Milestone 1 includes a training-data audit, reusable preprocessing, an untuned
Linear Regression experiment, grouped cross-validation, and this progress note.
Three-model comparison, hyperparameter tuning, PCA, final test evaluation, and
the application are Milestone 2 work. A dummy comparator is included for context;
it does not count as one of the three substantive regression model families.

## Milestone 1 coverage

| Regression requirement | Completed evidence |
| --- | --- |
| Data understanding, quality checks, distributions, skewness, and outliers | Shared analysis plus regression training audit tables and figures in `outputs/regression/` |
| Prediction point, predictor choices, and leakage risks | Input contract and preliminary leakage audit below; all 31 source columns accounted for in `column_inventory.csv` |
| Reproducible validation strategy | Existing outer split preserved; five grouped folds on training data, seed 42, with row/group assignments saved in `oof_predictions.csv` |
| Reusable missing-value handling, categorical encoding, and suitable scaling | `src/preprocessing/regression/preprocessing.py`; learned transformations fitted independently in each training fold |
| At least one initial untuned regression experiment | Default Linear Regression, compared with a training-fold mean dummy on identical folds |
| Preliminary metrics and interpretation; final test reserved | MAE, MSE, RMSE, R2, fold variation, and residual diagnostics below; regression scripts do not open `test.csv` |
| Regression portion of the progress report and remaining plan | This note records completed work, results, limitations, contribution scope, and the Milestone 2 plan |

This note is the regression section to incorporate into the team's combined
progress report. Member/account identification and the final submitted source
commit are recorded by the team when assembling that submission.

## Inputs, population, and row lineage

- Source: `data/preprocessed/train.csv`, 63,329 rows. The regression scripts never
  open `test.csv`, `H2_preprocessed.csv`, or the raw CSV.
- Shared preprocessing and the existing train/test membership are preserved.
- Exclude 204 **zero-booked-night** training records in the regression view: the
  denominator in the article's ADR definition is zero. This is a task-specific
  population choice, not a claim that those bookings are erroneous.
- Retain **63,125 positive-night bookings**, including **627 with ADR = 0**.
  Of those zero-rate bookings, 377 have `MarketSegment = Complementary`; the
  other 250 remain unexplained. Neither group is silently relabeled or removed.
- Retain canceled bookings and no-shows: the target is the recorded rate associated
  with a booking, not revenue earned from completed stays. Outcomes are excluded
  from predictors and are not used to select the regression population.
- Retain duplicate rows and all remaining rate extremes (ADR range 0 to 510).
  There is no target imputation, winsorization, clipping, log transform, or filtering
  based on an ADR threshold.
- The deterministic saved view is
  `data/preprocessed/regression/train_regression.csv`. Four missing `Children`
  values and 16 missing `Country` values remain missing in it. Numeric float
  representation is intentional to support missing values.
- `outputs/regression/preparation_exclusions.csv` identifies excluded records by
  their **1-based data-record number in shared train.csv**, excluding its header.
  Out-of-fold predictions use the same numbering, so filtering does not lose lineage.
- At eventual test evaluation, apply the same positive-night eligibility rule and
  report eligible/excluded counts. Do not construct another random split. Inference
  rejects zero-night inputs rather than returning an unsupported per-night estimate.

## Selected predictors and transformations

There are **23 predictors**. Source labels are unchanged.

| Predictor group | Columns | Preparation |
| --- | --- | --- |
| Planned arrival and lead time | `LeadTime`, `ArrivalDateYear`, `ArrivalDateWeekNumber`, `ArrivalDateDayOfMonth` | Numeric; fold-local median imputation and standard scaling |
| Booked stay | `StaysInWeekendNights`, `StaysInWeekNights` | Numeric; same pipeline; positive total nights required |
| Party size | `Adults`, `Children`, `Babies` | Numeric; same pipeline; missing children imputed from fold-training median |
| Guest history | `IsRepeatedGuest`, `PreviousCancellations`, `PreviousBookingsNotCanceled` | Numeric; same pipeline; repeated guest validated as binary |
| Requests | `RequiredCarParkingSpaces`, `TotalOfSpecialRequests` | Numeric; same pipeline |
| Nominal categories | `ArrivalDateMonth`, `Meal`, `Country`, `MarketSegment`, `DistributionChannel`, `ReservedRoomType`, `Agent`, `Company`, `CustomerType` | Constant `Unknown` imputation, then sparse one-hot encoding |

Detailed **per-column availability assumptions** are saved in
`outputs/regression/column_inventory.csv` (all 31 source columns accounted for).

Deterministic cleanup strips surrounding whitespace and parses numeric values.
Missing markers are normalized only for the relevant fields. `NULL` in `Agent`
or `Company` becomes **`NotApplicable`**, following the source paper, and those IDs
remain nominal strings. Missing `Country` is not treated as an agency-style NULL.
The literal `NA` is not automatically discarded as a country identifier.
Missing values outside `Children` and `Country`, non-finite or malformed numbers,
negative counts, and fractional count values raise errors instead of being hidden.

`SimpleImputer(strategy="median", add_indicator=True)` learns numeric medians
inside each fold; indicators are created only for fields observed missing in that
fold's training rows. Numeric scaling uses training-fold means and standard
deviations. Scaling is not essential to ordinary least squares' intended
predictions, but provides a consistent numerical representation. Categorical
vocabulary is learned inside each fold; unseen categories produce all-zero
one-hot entries without failing. An unseen category is therefore not a learned
new level and can have less reliable predictions.

No data-driven feature selection or derived predictor is used in this baseline.
Adding total nights/guests alongside their components would be redundant for an
additive linear model. Total nights is computed only for eligibility and diagnostics;
no shared `src/features/` module is needed yet. Full one-hot encoding and overlapping
calendar fields introduce redundant columns; predictions are the baseline focus,
and individual coefficients should not be interpreted as unique causal effects.

## Preliminary leakage audit

| Excluded column | Reason at booking creation |
| --- | --- |
| `ADR` | Target; never an input to the feature transformer |
| `IsCanceled` | Eventual cancellation outcome |
| `ReservationStatus` | Final reservation outcome |
| `ReservationStatusDate` | Date of a later outcome |
| `AssignedRoomType` | Ultimate room assignment can occur after creation |
| `BookingChanges` | Cumulative changes through check-in or cancellation |
| `DaysInWaitingList` | Completed waiting duration is not yet known |
| `DepositType` | Derived from payments and stay cost; both timing and rate-proxy risks |

**Snapshot limitation:** the [original dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/)
describes extraction relative to the day before arrival, using change logs where
available. It does not establish booking-creation snapshots for this task. It also
warns that party size, duration, room preferences, and nationality can differ later
in a booking's lifecycle. Retained mutable fields are assumed to approximate the
original booking request, country is assumed supplied at booking, and historical
outcomes are assumed already known. These assumptions cannot be verified from this
CSV. Recorded ADR is transaction-derived and may differ from the original quote.
Therefore the results are exploratory evidence, not verified deployment performance
at booking creation. Removing obvious later fields reduces risk but does not resolve
this limitation. A stricter feature set or a properly timestamped source should be
considered in Milestone 2.

**Validation boundaries:** deterministic cleanup uses no fitted statistics.
`RegressionInputCleaner -> ColumnTransformer -> estimator` keeps learned imputation,
encoding, and scaling within each fold. An optional full-training fit happens only
after the out-of-fold results are computed and does not produce evaluation scores.
No PCA, tuning, or feature selection sees the final test set.

**Duplicates:** group rows by identical cleaned selected predictors, **excluding
ADR**, with missing values retained as group values. The eligible training data has
40,962 groups, including 4,998 repeated groups (27,161 rows). There are 436 groups
with multiple ADR values. These are kept and placed together; differing targets
are not used to create separate groups. Grouping does not establish booking identity.
The existing outer split groups full records, so it cannot guarantee no selected-input
matches across train/test after exclusions. It is preserved as agreed; disclose this
limitation during final evaluation. Without customer IDs, customer-level separation
also cannot be established.

**Existing shared decisions:** shared EDA and the removal of ADR 5,400 happened
before the split. This regression run neither revisits those rules using holdout
performance nor claims the complete project used a fully blind holdout design.
Grouped random CV also estimates within-period generalization, not forecasting
future years. Calendar drift should be considered before deployment.

## Initial experiment and results

- `GroupKFold(n_splits=5, shuffle=True, random_state=42)` on eligible training rows.
- Identical folds for untuned `LinearRegression()` and `DummyRegressor(strategy="mean")`.
- Every row receives exactly one out-of-fold prediction; each fold has zero shared
  predictor groups between its fit and validation partitions.
- Dummy means are learned separately on each fold's training targets.
- Metrics use unchanged ADR and raw predictions. No prediction clipping or tuning.
- Preliminary model-selection metric for the remaining work: **CV RMSE**, supported
  by MAE and R2. Final test data remains reserved for the chosen model in Milestone 2.

Pooled out-of-fold results (all 63,125 eligible rows):

| Model | MAE (EUR) | MSE (EUR squared) | RMSE (EUR) | R2 |
| --- | ---: | ---: | ---: | ---: |
| Training-fold mean dummy | 29.260 | 1494.368 | 38.657 | -0.00005 |
| Untuned Linear Regression | 14.563 | 445.410 | 21.105 | 0.70193 |

Linear Regression's mean fold RMSE is **21.098**, with a fold sample standard
deviation of **0.860**; mean fold MAE is **14.566**, standard deviation **0.263**.
Fold dispersion is descriptive, not a confidence interval. Fold means and pooled
metrics differ because the validation folds have different row counts.

The linear baseline improves MAE by about **50.2%** over the mean dummy. It captures
useful associations but still has sizeable errors, including underprediction of
some high rates. MAE is **49.702** on the 627 zero-rate bookings and **14.211** on
positive-rate bookings. There are **230 negative out-of-fold predictions**; these
are retained in the reported metrics. Nonnegative postprocessing or another model
can be studied later using training-only validation, rather than silently improving
this baseline after seeing its errors. Metrics must not be compared with other
models evaluated on a different row population or split.

## Reproduction

From the repository root, after installing `requirements.txt`:

```powershell
python src/preprocessing/regression/prepare_regression_data.py
python src/regression/audit_training_data.py
python src/regression/train_baseline.py --save-artifact
python -m unittest discover -s tests -v
```

Follow the environment setup in the repository's main README. Replace `python`
with `.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux
if the environment is not activated. Python **3.13.13** was used for the recorded
experiment. Package versions and estimator parameters are recorded in
`outputs/regression/experiment_metadata.json`. `joblib` is an explicit dependency
because it is used to save the pipeline. Runtime caches and the local Python
installation are ignored by Git.

Preparation and audit scripts do not fit or save learned transformers. Training
rebuilds its input view directly from shared `train.csv`, so editing a generated
regression CSV cannot accidentally change the experiment. All scripts can be rerun
to regenerate their own files. Runtime measurements can vary across machines.

For loading the optional baseline artifact from the repository root:

```python
import joblib
from src.preprocessing.regression.preprocessing import PREDICTOR_COLUMNS, load_training_frame

pipeline = joblib.load("artifacts/regression/linear_regression_baseline.joblib")
booking = load_training_frame().iloc[[0]][PREDICTOR_COLUMNS]
prediction = pipeline.predict(booking)  # supply a positive-night booking
```

This loading example demonstrates artifact use, not an evaluation result. For new
inputs, supply a DataFrame with the same 23 columns and valid booked nights.

## Generated deliverables

- `data/preprocessed/regression/train_regression.csv`: reproducible deterministic view.
- `outputs/regression/`: exclusions, audit summary, all-column inventory, numeric
  summary/outlier screening, category cardinality, whitespace counts, repeated input
  groups, target summary, associations, and zero-rate profiles.
- `outputs/regression/cv_fold_metrics.csv`: each model/fold's metrics, row/group
  counts, fit/prediction times, and zero shared-group checks.
- `outputs/regression/cv_summary.csv`: pooled metrics and fold means/deviations.
- `outputs/regression/oof_predictions.csv`: row lineage, groups, folds, targets,
  and both models' out-of-fold predictions.
- `outputs/regression/oof_subgroup_metrics.csv`: diagnostic zero/positive-rate errors.
- `outputs/regression/figures/`: target distribution, numeric associations, monthly
  ADR boxes, and out-of-fold prediction/residual diagnostics. Every figure uses
  training data only. Hiding boxplot fliers in the monthly figure does not remove rows.
- `outputs/regression/experiment_metadata.json`: protocol, features, exclusions,
  parameters, dependency versions, and training-input SHA256.
- `artifacts/regression/linear_regression_baseline.joblib`: optional local pipeline,
  including deterministic cleanup and every fitted transformation. Ignored by Git;
  exclude it from the submission ZIP and regenerate it locally when needed.
- `tests/test_regression.py`: behavior checks for population rules, fold grouping,
  training-only statistics, unknown categories, outcome exclusion, and invalid inputs.

## Remaining work and contribution handoff

This regression contribution implements the training audit, target/population and
feature decisions, pipeline, untuned experiment, diagnostics, checks, and this note.
The team should attribute it to the responsible member's actual name/account and
meaningful commits in its combined progress report; names are not invented here.

For Milestone 2, compare at least three substantive regression families (for example,
the linear baseline, Random Forest, and an approved gradient-boosted regressor),
including baseline and tuned results for each. Use justified tunable estimators or
parameters for the linear family, such as Ridge, rather than presenting fixed OLS
as a meaningful optimization search. Include an ensemble and document all search
spaces. Refit preprocessing inside every search/CV fold. Coordinate the required
PCA comparisons with the team; PCA representations must be fitted inside folds,
with explained variance and component counts justified. Finish final model
selection, one-time holdout evaluation, saved inference, and the application.

Further decisions include rare-ID stability, correlated calendar predictors,
zero-rate interpretation, extreme-value errors, negative predictions, snapshot
availability, and whether temporal validation changes the conclusions. These are
documented limitations, not reasons to alter shared data without coordination.
