"""Create one cleaned H2 dataset before the train/test split.

Run from the project folder:
    python src/preprocessing/preprocessing_pipeline.py

The raw H2.csv is never changed.
"""

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw" / "H2.csv"
OUTPUT = PROJECT / "data" / "preprocessed" / "H2_preprocessed.csv"
EXCLUSIONS = PROJECT / "outputs" / "data_quality" / "preprocessing_exclusions.csv"


def main():
    data = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    guests = data[["Adults", "Children", "Babies"]].apply(pd.to_numeric, errors="coerce")
    zero_guests = guests.notna().all(axis=1) & guests.sum(axis=1).eq(0)
    adr_5400 = pd.to_numeric(data["ADR"], errors="coerce").eq(5400)
    remove = zero_guests | adr_5400

    excluded = data.loc[remove, [
        "ADR", "Adults", "Children", "Babies", "StaysInWeekendNights",
        "StaysInWeekNights", "MarketSegment", "ReservationStatus",
    ]].copy()
    excluded.insert(0, "raw_record_number", excluded.index + 1)
    excluded.insert(1, "reason", [
        "; ".join(reason for condition, reason in [
            (zero_guests.loc[index], "zero_guests"),
            (adr_5400.loc[index], "ADR_5400"),
        ] if condition)
        for index in excluded.index
    ])

    clean = data.loc[~remove].copy()
    old_code = clean["Country"].str.strip().eq("TMP")
    clean.loc[old_code, "Country"] = "TLS"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    EXCLUSIONS.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT, index=False)
    excluded.to_csv(EXCLUSIONS, index=False)

    print(f"Removed {int(zero_guests.sum())} zero-guest rows and "
          f"{int(adr_5400.sum())} ADR-5400 row; kept {len(clean):,} rows.")
    print(f"Changed {int(old_code.sum())} Country values from TMP to TLS.")
    print(f"Preprocessed data: {OUTPUT}")
    print(f"Excluded rows: {EXCLUSIONS}")


if __name__ == "__main__":
    main()
