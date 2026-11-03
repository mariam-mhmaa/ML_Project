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

3. **Summarize and plot the data.** `src/data/statistics_plots.py` makes summary tables and charts for numeric and categorical values, skewness, and outliers. For this analysis, it removes surrounding spaces from text and treats `NULL` as missing in memory. Tables are saved in `outputs/tables/` and figures in `outputs/figures/`. This analysis uses the full raw dataset, including duplicate rows.

   ```powershell
   python src/data/statistics_plots.py
   ```

4. **Create training and test files.** `src/preprocessing/split_data.py` keeps all 79,330 rows. It gives each exact-duplicate group a group ID, then holds out one of five stratified group folds for testing. This keeps each group wholly in training or test while preserving the cancellation rate. The result is 63,464 rows in `data/processed/train.csv` and 15,866 in `data/processed/test.csv` (80% / 20%). A fixed random seed makes the split reproducible. The saved values are still raw text; this step does not fill missing values or encode categories.

   ```powershell
   python src/preprocessing/split_data.py
   ```

5. **Audit the split.** `src/preprocessing/audit_split.py` checks that every raw row was retained and no exact row appears in both files. It compares numeric summaries and category percentages between training and test. The results are saved in `outputs/tables/13_split_integrity.csv`, `14_split_numeric_comparison.csv`, and `15_split_category_comparison.csv`.

   ```powershell
   python src/preprocessing/audit_split.py
   ```

## Split assumptions and limitations

The [dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/) says each row represents a booking and that identifying information was removed. It does not say identical rows are errors. We therefore keep them because separate bookings may have the same recorded values. Keeping each exact-duplicate group on one side of the split prevents an identical full row from appearing in both training and test files.

The split audit confirms that all rows are retained, no exact row crosses the split, and both sets have a 41.7% cancellation rate. Typical ADR values are close: the medians are 99.9 (train) and 99.5 (test), and both 95th percentiles are 175.5. Lead time, guest counts, and stay lengths are also close. The largest gap among the main market segments is `Groups`: 18.1% in train and 15.5% in test. We keep this fixed split for modeling.

One extreme ADR value of 5,400 is in test, while the largest training ADR is 510. We will report this when interpreting regression errors rather than changing the split to move that row. Seven rare country codes also appear only in test. Grouping by exact full rows prevents exact matches across the split; it does not establish whether two matching rows are actually the same booking.

No preprocessing pipeline or machine learning model has been implemented yet. For model selection, we plan five-fold cross-validation on the training file, keeping rows with the same task-specific model inputs in one fold. The final test file should be used only after model selection, for final evaluation.
