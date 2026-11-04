"""Exclude the agreed records from modeling copies of the train/test split.

Run from the project folder:
    python src/preprocessing/prepare_modeling_data.py

The original H2.csv and the existing train.csv/test.csv are left unchanged.
"""

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT / "data" / "processed"
REPORT = PROJECT / "outputs" / "data_quality" / "modeling_exclusions.csv"


def prepare_split(name):
    data = pd.read_csv(PROCESSED / f"{name}.csv", dtype=str, keep_default_na=False)
    guests = data[["Adults", "Children", "Babies"]].apply(pd.to_numeric, errors="coerce")
    zero_guests = guests.notna().all(axis=1) & guests.sum(axis=1).eq(0)
    adr_5400 = pd.to_numeric(data["ADR"], errors="coerce").eq(5400)
    remove = zero_guests | adr_5400

    excluded = data.loc[remove, [
        "ADR", "Adults", "Children", "Babies", "StaysInWeekendNights",
        "StaysInWeekNights", "MarketSegment", "ReservationStatus",
    ]].copy()
    excluded.insert(0, "split", name)
    excluded.insert(1, "split_row_number", excluded.index + 1)
    excluded.insert(2, "reason", [
        "; ".join(reason for condition, reason in [
            (zero_guests.loc[index], "zero_guests"),
            (adr_5400.loc[index], "ADR_5400"),
        ] if condition)
        for index in excluded.index
    ])

    data.loc[~remove].to_csv(PROCESSED / f"{name}_modeling.csv", index=False)
    print(f"{name}: removed {int(zero_guests.sum())} zero-guest rows and "
          f"{int(adr_5400.sum())} ADR-5400 rows; kept {int((~remove).sum()):,}")
    return excluded


def main():
    train_excluded = prepare_split("train")
    test_excluded = prepare_split("test")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([train_excluded, test_excluded], ignore_index=True).to_csv(REPORT, index=False)
    print(f"Excluded rows: {REPORT}")


if __name__ == "__main__":
    main()
