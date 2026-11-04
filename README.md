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

5. **Preprocess the full dataset.** `src/preprocessing/preprocessing_pipeline.py` removes the 167 bookings with zero recorded guests and the one booking with ADR 5,400. It also changes three `Country` values from the historical code `TMP` to the current code `TLS`. It saves all 79,162 remaining bookings in `data/preprocessed/H2_preprocessed.csv`. The raw file is unchanged, and `outputs/data_quality/preprocessing_exclusions.csv` lists the removed raw row numbers and reasons.

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
| Flag | Missing `Country` | 24 | 16 | Keep for later missing-value handling. |
| Map | Historical country code `TMP` | 3 | 0 | Replace with `TLS` for Timor-Leste; all three rows stay. |

The two removal rules exclude **168 distinct bookings**. The 67 bookings with both zero guests and zero nights are removed under the zero-guest rule. The remaining zero-ADR bookings are 264 zero-night rows, 480 `Complementary` rows, and 327 rows in other market segments. Exact duplicate rows are kept and grouped together when splitting.

## Split assumptions and limitations

The [dataset article](https://repositorio.iscte-iul.pt/bitstream/10071/16929/1/Hotelbookingdemanddatasets.pdf) says each row represents a booking and that identifying information was removed. It does not say identical rows are errors. We therefore keep them because separate bookings may have the same recorded values. Keeping each exact-duplicate group on one side of the split prevents an identical full row from appearing in both training and test files. The `TMP` to `TLS` mapping follows the [ISO country-code change notice](https://www.iso.org/files/live/sites/isoorg/files/archive/pdf/en/newsletter_v-5_east_timor.pdf).

The split audit confirms that all **79,162 preprocessed rows** are retained, no exact row crosses the split, and both sets have a 41.8% cancellation rate. ADR medians are 99.25 (train) and 100 (test); both 95th percentiles are 175.5. The largest main market-segment difference is 1.6 percentage points for `Offline TA/TO`. We keep this fixed split for modeling.

Grouping by exact full rows prevents exact matches across the split; it does not establish whether two matching rows are actually the same booking.

The current pipeline only applies the three agreed cleaning rules. It does not yet handle missing `Children` and `Country`, resolve the remaining zero-ADR and zero-night cases, or encode and scale model features. Those decisions should be made for each modeling task, with any learned transformations fitted on training data only. No model has been implemented yet. For model selection, we plan grouped five-fold cross-validation on `train.csv`; `test.csv` is reserved for final evaluation.
