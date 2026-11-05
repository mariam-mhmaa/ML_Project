# City Hotel Booking ML Project

This project uses the City Hotel booking dataset (`data/raw/H2.csv`) for the CSE382 Introduction to Machine Learning project. The dataset has 79,330 bookings and 31 columns.

Repository: https://github.com/mariam-mhmaa/ML_Project

## Setup

Use **64-bit Python 3.13** (tested with 3.13.13). Install Python separately;
`requirements.txt` installs the project's Python packages. Each teammate creates
their own environment; no downloaded Python installation or environment is shared
through Git.

On Windows, from PowerShell:

```powershell
git clone https://github.com/mariam-mhmaa/ML_Project.git
cd ML_Project
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

If the repository is already cloned, skip the first two commands and open its root
(the folder containing this README, `src/`, and `data/`). If the Python launcher
is unavailable, use `python -m venv .venv` with Python 3.13 on PATH.
If `.venv` already exists, use its interpreter directly and skip environment
creation; the install and dependency-check commands still apply.

On macOS/Linux, the equivalent setup is:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

The Windows setup is verified; macOS/Linux commands follow the same environment
layout but have not been tested here. In VS Code, use **Python: Select Interpreter**
and choose `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on macOS/Linux.

Run all project commands from the repository root. In the commands below,
`python` means the environment's Python: use `.\.venv\Scripts\python.exe` on
Windows or `.venv/bin/python` on macOS/Linux if you have not activated it.

The requirements cover the current shared scripts, classification preparation,
regression, and tests. They include NumPy, pandas, Matplotlib, scikit-learn, and
joblib; pip installs their dependencies automatically. No notebook, CUDA, Codex,
or uv installation is required to run this repository.

The tracked CSVs and country reference are included, so a fresh clone already has
the inputs. Regression can start directly from `data/preprocessed/train.csv`;
rerunning the shared cleaning/split scripts is optional and is not part of a normal
regression run.

## Work completed so far

The scripts below follow the order of our data analysis and preparation. Run them from the project folder.

1. **Inspect the raw data.** `src/data/explore_data.py` shows the dataset size, column names, data types, missing values, duplicate rows, and examples of the original text values. It does not change the data.

   ```powershell
   python src/data/explore_data.py
   ```

2. **Check data quality.** `src/data/check_data_quality.py` checks the expected columns, missing markers, number and date rules, country codes, and unusual values such as zero guests or very high room rates. It flags records for review without changing `H2.csv`. Reports are saved in `outputs/data_quality/`.

   ```powershell
   python src/data/check_data_quality.py
   ```

3. **Cross-check ADR.** `src/data/adr_crosschecks.py` compares zero ADR with total nights, total guests, market segment, and reservation status. It writes `outputs/data_quality/adr_zero_crosscheck.csv` and `adr_zero_case_profiles.csv`. It does not change `H2.csv`.

   ```powershell
   python src/data/adr_crosschecks.py
   ```

   The current H2 report finds 1,208 zero-ADR rows: 331 also have zero nights, 137 also have zero guests, and 67 have both. These counts overlap. The article defines ADR per staying night, so zero nights accounts for why these 331 rows cannot have a normal per-night rate. Of all 542 `Complementary` bookings, 508 have zero ADR; the category suggests a zero-price booking but the article does not confirm each case. Another 327 zero-ADR rows have positive nights and guests in other market segments.

4. **Summarize and plot the data.** `src/data/statistics_plots.py` makes summary tables and charts for numeric and categorical values, skewness, and outliers. For this analysis, it removes surrounding spaces from text and, in memory, treats `NULL` in `Country` as missing. `NULL` in `Agent` and `Company` means "not applicable", so the tables report it separately from missing values. Tables are saved in `outputs/tables/` and figures in `outputs/figures/`. This analysis uses the full raw dataset, including duplicate rows.

   ```powershell
   python src/data/statistics_plots.py
   ```

5. **Preprocess the full dataset.** `src/preprocessing/preprocessing_pipeline.py` removes the 167 bookings with zero recorded guests and the one booking with ADR 5,400. It standardizes `Country` by changing three `TMP` values to `TLS` and 569 `CN` values to `CHN`. It saves all 79,162 remaining bookings in `data/preprocessed/H2_preprocessed.csv`. The raw file is unchanged, and `outputs/data_quality/preprocessing_exclusions.csv` lists the removed raw row numbers and reasons.

   ```powershell
   python src/preprocessing/preprocessing_pipeline.py
   ```

6. **Split the preprocessed data.** `src/preprocessing/split_data.py` assigns each exact-duplicate group wholly to train or test, using one of five stratified group folds and a fixed random seed. It saves 63,329 rows in `data/preprocessed/train.csv` and 15,833 in `data/preprocessed/test.csv` (about 80% / 20%).

   ```powershell
   python src/preprocessing/split_data.py
   ```

7. **Audit the split.** `src/preprocessing/audit_split.py` checks that all preprocessed rows appear in exactly one split and that no exact row appears in both. It compares numeric values and category percentages. Reports are in `outputs/tables/13_split_integrity.csv`, `14_split_numeric_comparison.csv`, and `15_split_category_comparison.csv`.

   ```powershell
   python src/preprocessing/audit_split.py
   ```

8. **Prepare classification training data (Milestone 1 preparation).** `src/preprocessing/classification/prepare_classification_data.py` builds a cleaned training dataset for cancellation classification from `data/preprocessed/train.csv`, with booking-time-oriented feature exclusions and fixed missing-value rules. `src/classification/audit_training_data.py` writes focused audit tables to `outputs/classification/`.

   ```powershell
   python src/preprocessing/classification/prepare_classification_data.py
   python src/classification/audit_training_data.py
   ```

   Details and assumptions are documented in `docs/classification_milestone1.md`.

9. **Run the untuned classification baseline (Milestone 1).** `src/classification/run_milestone1_baseline.py` runs grouped stratified 5-fold cross-validation on `train_classification.csv` using a fold-local preprocessing + `RandomForestClassifier(random_state=42, n_jobs=-1)` pipeline and a `DummyClassifier(strategy="prior")` reference. It saves fold assignments, out-of-fold predictions, metrics, plots, reproducibility metadata, and a full-training fitted pipeline artifact.

   ```powershell
   python src/classification/run_milestone1_baseline.py
   ```

   Results, interpretation, limitations, and remaining Milestone 2 work are documented in `docs/classification_milestone1.md`.

10. **Prepare, audit, and run the regression Milestone 1 baseline.** Regression
   starts from shared `train.csv`, excludes zero-night bookings only in its own
   view, and retains positive-night zero-rate bookings. The untuned Linear
   Regression and mean-dummy comparison uses grouped five-fold cross-validation
   with imputation, encoding, and scaling fitted separately inside each fold.

   ```powershell
   python src/preprocessing/regression/prepare_regression_data.py
   python src/regression/audit_training_data.py
   python src/regression/train_baseline.py --save-artifact
   python -m unittest discover -s tests -v
   ```

   On 63,125 eligible training rows, pooled out-of-fold MAE / RMSE / R2 are
   **14.563 / 21.105 / 0.702** for Linear Regression versus
   **29.260 / 38.657 / approximately 0** for the dummy. These are preliminary
   validation results, not final test results. Tables and figures are in
   `outputs/regression/`; the optional local fitted pipeline is in
   `artifacts/regression/`. The regression scripts never open `test.csv`.

   See [the regression note](docs/regression_milestone1.md) for the feature list,
   prediction-time assumptions, quality decisions, leakage audit, results, and
   remaining plan.

## Task folder convention

| Content | Location |
| --- | --- |
| Shared cleaning and split | `src/preprocessing/`, `data/preprocessed/` |
| Regression-specific preparation | `src/preprocessing/regression/` |
| Optional reproducible regression data | `data/preprocessed/regression/` |
| Regression audits and experiments | `src/regression/` |
| Regression tables and figures | `outputs/regression/` |
| Fitted regression pipelines | `artifacts/regression/` |
| Derived features reused across tasks, if needed | `src/features/` |

Classification and clustering follow the same task-folder pattern. Do not edit
shared CSVs by hand or refit learned preprocessing on test data.

## Documentation

- **This README** is the team entry point: setup, shared workflow, task commands,
  folder conventions, and current status.
- [Regression Milestone 1](docs/regression_milestone1.md) contains the regression
  feature decisions, leakage audit, experiment protocol, results, and remaining plan.
- [Classification Milestone 1](docs/classification_milestone1.md) documents that
  task's current preparation and assumptions.
- [Regression artifacts](artifacts/regression/README.md) explains the generated
  pipeline and how to recreate it locally.

Local `.venv/`, `.python-runtime/`, and `.codex-task-cache/` folders are development
tools, not project deliverables. They are ignored by Git and are not required in a
fresh clone. Do not force-add them. Keep source, requirements, task notes, result
tables, figures, and metadata in the repository. Also exclude local environments,
caches, `.git/`, and fitted pipelines when creating a submission ZIP manually.

## Preprocessing decisions

Here, **remove** means leave a row out of the preprocessed dataset. **Flag** means keep it for review without changing its value. `outputs/data_quality/flagged_values.csv` lists the original raw row numbers with quality flags. Its counts include rows later removed by the pipeline.

| Decision | Condition | Original rows | Rows after preprocessing | Reason |
| --- | --- | ---: | ---: | --- |
| Remove | Total guests = 0 | 167 | 0 | The actual guest count cannot be inferred; 144 are marked `Check-Out`. |
| Remove | ADR = 5,400 | 1 | 0 | Suspected source outlier; the next highest ADR is 510. |
| Flag | Total nights = 0 | 331 | 264 | All have ADR = 0; 308 are marked `Check-Out` on their arrival date. Same-day use is possible but unconfirmed. |
| Flag | ADR = 0, positive nights and guests, `Complementary` | 480 | 480 | A free booking is plausible, but not confirmed for each row. |
| Flag | ADR = 0, positive nights and guests, other segment | 327 | 327 | The zero rate remains unexplained. |
| Flag | Missing `Children` | 4 | 4 | The guest count is incomplete, not proven to be zero. |
| Flag | Missing `Country` | 24 | 16 | Eight also had zero guests and were removed; handle the remaining missing values later. |
| Map | Historical country code `TMP` | 3 | 0 | Replace with `TLS` for Timor-Leste; all three rows stay. |
| Map | Two-letter country code `CN` | 569 | 0 | Replace with `CHN` so all 1,434 China bookings share one code. |

The two removal rules exclude **168 distinct bookings**. The 67 bookings with both zero guests and zero nights are removed under the zero-guest rule. The remaining zero-ADR bookings are 264 zero-night rows, 480 `Complementary` rows, and 327 rows in other market segments. Exact duplicate rows are kept and grouped together when splitting.

## Split assumptions and limitations

The [dataset article](https://repositorio.iscte-iul.pt/bitstream/10071/16929/1/Hotelbookingdemanddatasets.pdf) says each row represents a booking and that identifying information was removed. It does not say identical rows are errors. We therefore keep them because separate bookings may have the same recorded values. Keeping each exact-duplicate group on one side of the split prevents an identical full row from appearing in both training and test files. The `TMP` to `TLS` mapping follows the [ISO country-code change notice](https://www.iso.org/files/live/sites/isoorg/files/archive/pdf/en/newsletter_v-5_east_timor.pdf). The [UN country-code table](https://unstats.un.org/unsd/methodology/m49/overview/) lists `CN` and `CHN` as the two-letter and three-letter codes for China.

The split audit confirms that all **79,162 preprocessed rows** are retained, no exact row crosses the split, and both sets have a 41.8% cancellation rate. ADR medians are 99.25 (train) and 100 (test); both 95th percentiles are 175.5. The largest main market-segment difference is 1.6 percentage points for `Offline TA/TO`. We keep this fixed split for modeling.

Grouping by exact full rows prevents exact matches across the split; it does not establish whether two matching rows are actually the same booking.

The shared pipeline only applies the agreed removal and country-code rules. Each
task handles its own remaining missing values, quality decisions, encoding, and
scaling. Regression now implements these steps and an untuned baseline with grouped
five-fold cross-validation on `train.csv`. Its booking-creation scenario relies on
unverified snapshot assumptions documented in the regression note. The existing
outer split groups complete rows, so selected model inputs may still match across
the holdout boundary after columns are removed. `test.csv` remains reserved for
final model evaluation; the earlier shared split audit inspected its integrity and
distributions but did not train or select a regression model.

## Milestone 1 submission handoff

The specification requires initial regression, classification, and clustering
experiments now. The classification data-preparation note currently defers its
experiment; its owner should add that baseline, and the clustering owner should
complete its portion. The team must combine the progress report and identify
members, GitHub accounts, and contributions.

After committing the final submission version, record its actual SHA in the
submission copy of the README (obtain it with `git rev-parse HEAD`). Exclude local
Python/environment/cache folders and serialized fitted pipelines from the submission ZIP. Include source,
regeneration commands, task notes, result tables, figures, and metadata. Fitted
artifacts remain available locally for later development and the final discussion.
