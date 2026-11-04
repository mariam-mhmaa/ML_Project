# City Hotel Booking ML Project

This project uses the City Hotel booking dataset (`data/raw/H2.csv`) for the CSE382 Introduction to Machine Learning project. The dataset has 79,330 bookings and 31 columns.

## Setup

From the project folder, install the Python packages:

```powershell
python -m pip install -r requirements.txt
```

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

4. **Summarize and plot the data.** `src/data/statistics_plots.py` makes summary tables and charts for numeric and categorical values, skewness, and outliers. For this analysis, it removes surrounding spaces from text and treats `NULL` as missing in memory. Tables are saved in `outputs/tables/` and figures in `outputs/figures/`. This analysis uses the full raw dataset, including duplicate rows.

   ```powershell
   python src/data/statistics_plots.py
   ```

5. **Create training and test files.** `src/preprocessing/split_data.py` keeps all 79,330 rows. It gives each exact-duplicate group a group ID, then holds out one of five stratified group folds for testing. This keeps each group wholly in training or test while preserving the cancellation rate. The result is 63,464 rows in `data/processed/train.csv` and 15,866 in `data/processed/test.csv` (80% / 20%). A fixed random seed makes the split reproducible. The saved values are still raw text; this step does not fill missing values or encode categories.

   ```powershell
   python src/preprocessing/split_data.py
   ```

6. **Audit the split.** `src/preprocessing/audit_split.py` checks that every raw row was retained and no exact row appears in both files. It compares numeric summaries and category percentages between training and test. The results are saved in `outputs/tables/13_split_integrity.csv`, `14_split_numeric_comparison.csv`, and `15_split_category_comparison.csv`.

   ```powershell
   python src/preprocessing/audit_split.py
   ```

7. **Prepare modeling copies.** `src/preprocessing/prepare_modeling_data.py` excludes bookings with zero recorded guests and the single ADR 5,400 booking from the already split files. It writes `data/processed/train_modeling.csv` (63,328 rows), `test_modeling.csv` (15,834 rows), and `outputs/data_quality/modeling_exclusions.csv`, which lists every excluded row and its reason. The raw file and original train/test files stay intact.

   ```powershell
   python src/preprocessing/prepare_modeling_data.py
   ```

## Removal and review decisions

Here, **remove** means leave a row out of the modeling copies. **Flag** means keep it and list it for review; it does not mean changing a value. `outputs/data_quality/flagged_values.csv` contains row-level quality flags from the original data. The counts below show how many remain after the agreed removals.

| Decision | Condition | Original rows | Rows in modeling copies | Reason |
| --- | --- | ---: | ---: | --- |
| Remove | Total guests = 0 | 167 | 0 | The actual guest count cannot be inferred; 144 are marked `Check-Out`. |
| Remove | ADR = 5,400 | 1 | 0 | Suspected source outlier; the next highest ADR is 510. |
| Flag | Total nights = 0 | 331 | 264 | All have ADR = 0; 308 are marked `Check-Out` on their arrival date. Same-day use is possible but unconfirmed. |
| Flag | ADR = 0, positive nights and guests, `Complementary` | 480 | 480 | A free booking is plausible, but not confirmed for each row. |
| Flag | ADR = 0, positive nights and guests, other segment | 327 | 327 | The zero rate remains unexplained. |
| Flag | Missing `Children` | 4 | 4 | The guest count is incomplete, not proven to be zero. |
| Flag | Missing `Country` | 24 | 16 | Keep for later missing-value handling. |
| Flag | Historical country code `TMP` | 3 | 3 | Review its historical meaning; do not remap automatically. |

The two removal rules exclude **168 distinct bookings**, leaving **79,162**. The 67 bookings with both zero guests and zero nights are removed under the zero-guest rule. The remaining zero-ADR bookings are 264 zero-night rows, 480 `Complementary` rows, and 327 rows in other market segments. These review categories are not deletion rules. Exact duplicate rows remain in their original split groups.

## Split assumptions and limitations

The [dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/) says each row represents a booking and that identifying information was removed. It does not say identical rows are errors. We therefore keep them because separate bookings may have the same recorded values. Keeping each exact-duplicate group on one side of the split prevents an identical full row from appearing in both training and test files.

The split audit confirms that all rows are retained, no exact row crosses the split, and both sets have a 41.7% cancellation rate. Typical ADR values are close: the medians are 99.9 (train) and 99.5 (test), and both 95th percentiles are 175.5. Lead time, guest counts, and stay lengths are also close. The largest gap among the main market segments is `Groups`: 18.1% in train and 15.5% in test. We keep this fixed split for modeling.

The extreme ADR value of 5,400 was in the original test set; it is excluded only from the modeling copy under the documented decision above. Seven rare country codes also appear only in test. Grouping by exact full rows prevents exact matches across the split; it does not establish whether two matching rows are actually the same booking.

No feature preprocessing pipeline or machine learning model has been implemented yet. For model selection, we plan five-fold cross-validation on the modeling training file, keeping rows with the same task-specific model inputs in one fold. The modeling test file should be used only after model selection, for final evaluation.
