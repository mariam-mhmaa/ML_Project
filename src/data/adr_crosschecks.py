"""Cross-check zero ADR against nights, guests, and market segment.

Run from the project folder:
    python src/data/adr_crosschecks.py

Writes two CSV reports without changing the source data.
"""

import argparse
from pathlib import Path

import pandas as pd


GUEST_COLUMNS = ["Adults", "Children", "Babies"]
NIGHT_COLUMNS = ["StaysInWeekendNights", "StaysInWeekNights"]


def analyze(csv_path):
    data = pd.read_csv(csv_path, dtype="string", keep_default_na=False)
    columns = ["ADR", "MarketSegment", "ReservationStatus"] + GUEST_COLUMNS + NIGHT_COLUMNS
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data[columns].apply(lambda column: column.str.strip())

    numbers = {column: pd.to_numeric(data[column], errors="coerce")
               for column in ["ADR"] + GUEST_COLUMNS + NIGHT_COLUMNS}
    valid = {column: number.notna() & ~number.isin([float("inf"), float("-inf")])
             for column, number in numbers.items()}
    guest_total = sum(numbers[column] for column in GUEST_COLUMNS)
    night_total = sum(numbers[column] for column in NIGHT_COLUMNS)
    valid_guests = pd.concat([valid[column] for column in GUEST_COLUMNS], axis=1).all(axis=1)
    valid_nights = pd.concat([valid[column] for column in NIGHT_COLUMNS], axis=1).all(axis=1)

    zero_adr = valid["ADR"] & numbers["ADR"].eq(0)
    positive_adr = valid["ADR"] & numbers["ADR"].gt(0)
    zero_guests = valid_guests & guest_total.eq(0)
    positive_guests = valid_guests & guest_total.gt(0)
    zero_nights = valid_nights & night_total.eq(0)
    positive_nights = valid_nights & night_total.gt(0)
    complementary = data["MarketSegment"].eq("Complementary")

    crosscheck = {
        "zero_adr": int(zero_adr.sum()),
        "zero_adr_zero_nights": int((zero_adr & zero_nights).sum()),
        "zero_adr_zero_guests": int((zero_adr & zero_guests).sum()),
        "zero_adr_zero_nights_guests": int((zero_adr & zero_nights & zero_guests).sum()),
        "zero_adr_neither_zero_total": int((zero_adr & ~zero_nights & ~zero_guests).sum()),
        "zero_adr_neither_complementary": int((
            zero_adr & ~zero_nights & ~zero_guests & complementary
        ).sum()),
        "zero_guests": int(zero_guests.sum()),
        "zero_guests_positive_adr": int((zero_guests & positive_adr).sum()),
        "complementary": int(complementary.sum()),
        "complementary_zero_adr": int((complementary & zero_adr).sum()),
    }

    # These cases do not overlap. They cover zero ADR and positive-ADR zero guests.
    cases = {
        "zero_nights_and_zero_guests": zero_nights & zero_guests,
        "zero_nights_and_positive_guests": zero_nights & positive_guests,
        "positive_nights_zero_guests_zero_adr": positive_nights & zero_guests & zero_adr,
        "positive_nights_zero_guests_positive_adr": positive_nights & zero_guests & positive_adr,
        "positive_nights_positive_guests_zero_adr_complementary": (
            positive_nights & positive_guests & zero_adr & complementary
        ),
        "positive_nights_positive_guests_zero_adr_other_segment": (
            positive_nights & positive_guests & zero_adr & ~complementary
        ),
    }
    profiles = []
    for name, mask in cases.items():
        profiles.append({
            "case": name,
            "bookings": int(mask.sum()),
            "checked_out": int((mask & data["ReservationStatus"].eq("Check-Out")).sum()),
            "canceled": int((mask & data["ReservationStatus"].eq("Canceled")).sum()),
            "no_show": int((mask & data["ReservationStatus"].eq("No-Show")).sum()),
        })
    return pd.DataFrame([crosscheck]), pd.DataFrame(profiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/H2.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_quality"))
    args = parser.parse_args()

    crosscheck, profiles = analyze(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    crosscheck.to_csv(args.output / "adr_zero_crosscheck.csv", index=False)
    profiles.to_csv(args.output / "adr_zero_case_profiles.csv", index=False)
    print(crosscheck.to_string(index=False))
    print(f"\nWrote ADR reports to {args.output.resolve()}")


if __name__ == "__main__":
    main()
