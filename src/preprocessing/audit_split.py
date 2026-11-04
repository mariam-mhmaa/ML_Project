"""Check the saved H2 train/test split and write comparison tables.

Run from the project folder:
    python src/preprocessing/audit_split.py
"""

from pathlib import Path

import pandas as pd


project_folder = Path(__file__).resolve().parents[2]
data_folder = project_folder / "data"
output_folder = project_folder / "outputs" / "tables"

NUMERIC_COLS = [
    "ADR", "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "PreviousCancellations",
    "PreviousBookingsNotCanceled", "BookingChanges", "DaysInWaitingList",
    "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
]
CATEGORICAL_COLS = [
    "IsCanceled", "IsRepeatedGuest", "ArrivalDateMonth", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "ReservedRoomType",
    "AssignedRoomType", "DepositType", "CustomerType", "ReservationStatus",
]


def numeric_summary(df, column):
    values = pd.to_numeric(df[column], errors="coerce")
    return {
        "missing": int(values.isna().sum()),
        "mean": round(values.mean(), 3),
        "median": round(values.median(), 3),
        "p95": round(values.quantile(0.95), 3),
        "p99": round(values.quantile(0.99), 3),
        "max": round(values.max(), 3),
    }


def main():
    full = pd.read_csv(data_folder / "preprocessed" / "H2_preprocessed.csv", dtype=str,
                       keep_default_na=False)
    train = pd.read_csv(data_folder / "preprocessed" / "train.csv", dtype=str,
                        keep_default_na=False)
    test = pd.read_csv(data_folder / "preprocessed" / "test.csv", dtype=str,
                       keep_default_na=False)

    if not full.columns.equals(train.columns) or not full.columns.equals(test.columns):
        raise ValueError("Train/test columns do not match H2_preprocessed.csv")

    full_counts = full.value_counts(dropna=False).sort_index()
    combined_counts = pd.concat([train, test]).value_counts(dropna=False).sort_index()
    if not full_counts.equals(combined_counts):
        raise ValueError("Train and test do not retain every preprocessed row")

    shared_rows = len(pd.MultiIndex.from_frame(train).intersection(
        pd.MultiIndex.from_frame(test)
    ))
    if shared_rows:
        raise ValueError(f"{shared_rows} exact rows appear in both splits")

    summary = {
        "preprocessed_rows": len(full),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_repeated_group_rows": int(train.duplicated(keep=False).sum()),
        "test_repeated_group_rows": int(test.duplicated(keep=False).sum()),
        "shared_exact_rows": shared_rows,
        "preprocessed_canceled_pct": round(100 * (full["IsCanceled"] == "1").mean(), 3),
        "train_canceled_pct": round(100 * (train["IsCanceled"] == "1").mean(), 3),
        "test_canceled_pct": round(100 * (test["IsCanceled"] == "1").mean(), 3),
    }

    numeric_rows = []
    for column in NUMERIC_COLS:
        row = {"variable": column}
        for name, part in (("train", train), ("test", test)):
            row.update({f"{name}_{key}": value
                        for key, value in numeric_summary(part, column).items()})
        numeric_rows.append(row)

    category_rows = []
    for column in CATEGORICAL_COLS:
        train_pct = train[column].str.strip().value_counts(normalize=True)
        test_pct = test[column].str.strip().value_counts(normalize=True)
        for category in train_pct.index.union(test_pct.index):
            p = 100 * train_pct.get(category, 0)
            q = 100 * test_pct.get(category, 0)
            category_rows.append({
                "variable": column, "category": category,
                "train_pct": round(p, 3), "test_pct": round(q, 3),
                "difference_pp": round(q - p, 3),
            })

    output_folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_folder / "13_split_integrity.csv", index=False)
    pd.DataFrame(numeric_rows).to_csv(output_folder / "14_split_numeric_comparison.csv",
                                      index=False)
    pd.DataFrame(category_rows).to_csv(output_folder / "15_split_category_comparison.csv",
                                       index=False)

    print(f"All {len(full):,} preprocessed rows retained; no exact rows shared across splits.")
    print(f"Train: {len(train):,} rows, {summary['train_canceled_pct']:.2f}% canceled")
    print(f"Test:  {len(test):,} rows, {summary['test_canceled_pct']:.2f}% canceled")
    print(f"Comparison tables saved in {output_folder}")


if __name__ == "__main__":
    main()
