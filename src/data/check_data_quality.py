"""Inspect raw H2.csv without changing it. Requires pandas.

Run from the project folder:
    python src/data/check_data_quality.py

INVALID = violates a type, allowed value, or calendar rule.
MISSING = an empty/missing marker; a treatment still needs justification.
REVIEW = unusual or unexpected, but not proven incorrect.
Record numbers start at 1 and exclude the CSV header.
Definitions: https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/
"""

import argparse
import json
from pathlib import Path
from textwrap import fill

import pandas as pd


EXPECTED_COLUMNS = """
IsCanceled LeadTime ArrivalDateYear ArrivalDateMonth ArrivalDateWeekNumber
ArrivalDateDayOfMonth StaysInWeekendNights StaysInWeekNights Adults Children
Babies Meal Country MarketSegment DistributionChannel IsRepeatedGuest
PreviousCancellations PreviousBookingsNotCanceled ReservedRoomType
AssignedRoomType BookingChanges DepositType Agent Company DaysInWaitingList
CustomerType ADR RequiredCarParkingSpaces TotalOfSpecialRequests
ReservationStatus ReservationStatusDate
""".split()

COUNT_COLUMNS = """
LeadTime StaysInWeekendNights StaysInWeekNights Adults Children Babies
PreviousCancellations PreviousBookingsNotCanceled BookingChanges
DaysInWaitingList RequiredCarParkingSpaces TotalOfSpecialRequests
""".split()

MONTH_NAMES = """
January February March April May June July August September October
November December
""".split()
MONTH_NUMBERS = {name: str(i).zfill(2) for i, name in enumerate(MONTH_NAMES, 1)}

ALLOWED_VALUES = {
    "IsCanceled": {"0", "1"},
    "IsRepeatedGuest": {"0", "1"},
    "ArrivalDateMonth": set(MONTH_NAMES),
    "Meal": {"BB", "HB", "FB", "SC", "Undefined"},
    "DepositType": {"No Deposit", "Non Refund", "Refundable"},
    "CustomerType": {"Contract", "Group", "Transient", "Transient-Party"},
    "ReservationStatus": {"Canceled", "Check-Out", "No-Show"},
}


def load_country_reference(path):
    """Read the supplied reference, preserving numeric-code leading zeros.

    A current reference is not proof of a code's validity at an earlier date.
    Current matches are recognized; historical-only matches need review.
    No code is remapped to a successor country.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Country reference not found: {path}. Place Country_codes.json in "
            "data/reference/, or pass --country-codes followed by its path."
        )
    reference = json.loads(path.read_text(encoding="utf-8-sig"))
    if not reference.get("current") or not reference.get("historic"):
        raise ValueError("Country reference must contain current and historic lists.")

    def index_codes(records, fields):
        index = {}
        for record in records:
            for field in fields:
                code = str(record.get(field) or "").strip().upper()
                if code:
                    index.setdefault(code, set()).add(record["name"])
        return index

    current = index_codes(reference["current"], ["alpha2", "alpha3", "numeric"])
    historical = index_codes(
        reference["historic"],
        ["old_alpha2", "old_alpha3", "old_numeric", "alpha4", "preview2013_alpha4"],
    )
    return current, historical, str(reference.get("as_of", "not stated"))


def check_data_quality(csv_path, high_adr=1000.0,
                       country_codes_path=Path("data/reference/Country_codes.json")):
    """Return issue details and informational counts; never edit the source."""
    raw = pd.read_csv(csv_path, dtype="string", keep_default_na=False)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(raw.columns))
    extra_columns = sorted(set(raw.columns) - set(EXPECTED_COLUMNS))
    if missing_columns or extra_columns:
        raise ValueError(
            f"Source schema mismatch. Missing: {missing_columns}; "
            f"unexpected: {extra_columns}. Verify you are using original H2.csv."
        )

    # Work on a separate, whitespace-normalized inspection view.
    # Raw strings are retained in the issue report, and H2.csv is never saved.
    data = raw.apply(lambda column: column.str.strip())
    current_codes, historical_codes, reference_date = load_country_reference(
        country_codes_path
    )
    # Normalize only for the lookup; original source strings remain unchanged.
    country = data["Country"].str.upper()
    recognized_codes = set(current_codes) | set(historical_codes)
    issue_parts = []

    def flag(mask, column, severity, reason, values=None):
        indexes = data.index[mask.fillna(False)]
        if len(indexes) == 0:
            return
        source_values = raw[column] if values is None else values
        issue_parts.append(pd.DataFrame({
            "record_number": indexes + 1,
            "column": column,
            "severity": severity,
            "raw_value": source_values.loc[indexes].tolist(),
            "reason": reason,
        }))

    # NULL in Agent/Company is documented as not applicable, not unknown.
    missing = {}
    markers = {"", "NA", "N/A", "NAN", "NULL", "NONE", "?"}
    for column in data.columns:
        missing[column] = data[column].str.upper().isin(markers)
        if column in {"Agent", "Company"}:
            missing[column] &= data[column].str.upper().ne("NULL")
        if column == "Country":
            # NA is Namibia's alpha-2 code, so it must not become missing.
            missing[column] &= ~country.isin(recognized_codes)
        flag(missing[column], column, "MISSING", "Empty or missing-value marker")

    numeric_columns = COUNT_COLUMNS + [
        "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "ADR"
    ]
    numbers = {}
    numeric_valid = {}
    for column in numeric_columns:
        number = pd.to_numeric(data[column], errors="coerce")
        finite = number.notna() & ~number.isin([float("inf"), float("-inf")])
        present = ~missing[column]
        numbers[column] = number
        numeric_valid[column] = present & finite
        flag(present & ~finite, column, "INVALID", "Expected a finite number")
        if column != "ADR":
            flag(present & finite & number.mod(1).ne(0), column,
                 "INVALID", "Expected a whole number")
        if column in COUNT_COLUMNS:
            flag(present & finite & number.lt(0), column,
                 "INVALID", "A count/duration cannot be negative")

    for column, lower, upper in [
        ("ArrivalDateYear", 1, 9999),
        ("ArrivalDateWeekNumber", 1, 53),
        ("ArrivalDateDayOfMonth", 1, 31),
    ]:
        flag(numeric_valid[column] & ~numbers[column].between(lower, upper),
             column, "INVALID", f"Expected a value from {lower} to {upper}")

    for column, allowed in ALLOWED_VALUES.items():
        flag(~missing[column] & ~data[column].isin(allowed), column,
             "INVALID", "Outside the documented categories: "
             + ", ".join(sorted(allowed)))

    # Validate membership, rather than assuming every country has 3 letters.
    in_current = country.isin(current_codes)
    in_historical = country.isin(historical_codes)
    # Accept a recognized current code even if it also appears historically.
    # We validate code membership; we do not infer a historic identity or map
    # an old code to a modern country (reused codes make that unsafe).
    current_country = in_current
    historical_country = in_historical & ~in_current
    unrecognized_country = ~missing["Country"] & ~in_current & ~in_historical

    flag(unrecognized_country, "Country", "INVALID",
         "Not recognized in the supplied current or historical country-code lists")
    flag(historical_country, "Country", "REVIEW",
         "Recognized historical code; check its meaning and booking date")

    country_status = pd.Series("UNRECOGNIZED", index=data.index, dtype="string")
    country_status.loc[missing["Country"]] = "MISSING"
    country_status.loc[current_country] = "CURRENT"
    country_status.loc[historical_country] = "HISTORICAL_REVIEW"

    country_audit = []
    for (code, status), count in pd.DataFrame({
        "code": country, "status": country_status
    }).value_counts().items():
        names = current_codes.get(code, historical_codes.get(code, set()))
        country_audit.append({
            "country_code": code,
            "status": status,
            "reference_names": "; ".join(sorted(names)),
            "record_count": int(count),
            "reference_as_of": reference_date,
        })
    for column in ["Agent", "Company"]:
        valid_identifier = data[column].str.fullmatch(r"[0-9]+")
        not_applicable = data[column].str.upper().eq("NULL")
        flag(~missing[column] & ~valid_identifier & ~not_applicable,
             column, "REVIEW", "Expected a digit identifier or NULL; verify source")

    # An impossible date can pass individual day/month/year checks.
    arrival_text = (
        data["ArrivalDateYear"] + "-"
        + data["ArrivalDateMonth"].map(MONTH_NUMBERS).fillna("??") + "-"
        + data["ArrivalDateDayOfMonth"].str.zfill(2)
    )
    arrival = pd.to_datetime(arrival_text, format="%Y-%m-%d", errors="coerce")
    arrival_present = ~(
        missing["ArrivalDateYear"] | missing["ArrivalDateMonth"]
        | missing["ArrivalDateDayOfMonth"]
    )
    flag(arrival_present & arrival.isna(), "ArrivalDate", "INVALID",
         "Cannot form a supported, valid calendar date", values=arrival_text)
    flag(arrival.notna() & ~arrival.between("2015-07-01", "2017-08-31"),
         "ArrivalDate", "REVIEW", "Outside the documented H2 arrival period",
         values=arrival_text)

    status_date = pd.to_datetime(
        data["ReservationStatusDate"], format="%Y-%m-%d", errors="coerce"
    )
    flag(~missing["ReservationStatusDate"] & status_date.isna(),
         "ReservationStatusDate", "INVALID", "Invalid/unsupported date; expected YYYY-MM-DD")
    # A cancellation can precede arrival and checkout can follow it: neither is
    # automatically an error. Do not require the two dates to match.

    adr = numbers["ADR"]
    flag(numeric_valid["ADR"] & adr.le(0), "ADR", "REVIEW",
         "Zero/negative ADR: investigate; not automatically an error")
    flag(numeric_valid["ADR"] & adr.gt(high_adr), "ADR", "REVIEW",
         f"ADR exceeds illustrative review threshold {high_adr:g}; not a deletion rule")

    guest_columns = ["Adults", "Children", "Babies"]
    night_columns = ["StaysInWeekendNights", "StaysInWeekNights"]
    guest_total = sum(numbers[c] for c in guest_columns)
    night_total = sum(numbers[c] for c in night_columns)
    valid_guests = pd.concat([numeric_valid[c] for c in guest_columns], axis=1).all(axis=1)
    valid_nights = pd.concat([numeric_valid[c] for c in night_columns], axis=1).all(axis=1)
    zero_guests = valid_guests & guest_total.eq(0)
    zero_nights = valid_nights & night_total.eq(0)
    for columns, mask, reason in [
        (guest_columns, zero_guests, "Total recorded guests is zero; investigate"),
        (night_columns, zero_nights, "Total recorded nights is zero; investigate"),
    ]:
        values = data[columns[0]].copy()
        for column in columns[1:]:
            values = values + " + " + data[column]
        flag(mask, "+".join(columns), "REVIEW", reason, values)

    issue_columns = ["record_number", "column", "severity", "raw_value", "reason"]
    issues = (pd.concat(issue_parts, ignore_index=True) if issue_parts
              else pd.DataFrame(columns=issue_columns))
    if not issues.empty:
        issues = issues.sort_values(["record_number", "column", "severity"])
    information = {
        "rows": len(raw),
        "columns": len(raw.columns),
        "raw_duplicates_beyond_first": int(raw.duplicated().sum()),
        "trimmed_duplicates_beyond_first": int(data.duplicated().sum()),
        "rows_with_any_flag": int(issues["record_number"].nunique()),
        "country_current": int(current_country.sum()),
        "country_historical": int(historical_country.sum()),
        "country_missing": int(missing["Country"].sum()),
        "country_unrecognized": int(unrecognized_country.sum()),
        "country_reference_as_of": reference_date,
        "country_code_summary": country_audit,
    }
    return issues, information


def format_report(summary, information, high_adr):
    """Build a narrow terminal report with plain labels and grouped findings."""
    lines = ["DATA QUALITY REPORT", "=" * 50]

    def heading(title):
        lines.extend(["", title, "-" * len(title)])

    def count_line(label, count):
        lines.append(f"  {label:<39} {count:>8,}")

    heading("Dataset")
    count_line("Booking records", information["rows"])
    count_line("Columns", information["columns"])
    count_line("Records with at least one flag", information["rows_with_any_flag"])

    heading("Duplicates beyond the first occurrence")
    count_line("Before trimming spaces", information["raw_duplicates_beyond_first"])
    count_line("After trimming spaces", information["trimmed_duplicates_beyond_first"])
    lines.append("  These counts do not mean records should be deleted.")

    heading("Country-code reference checks")
    count_line("Recognized current codes", information["country_current"])
    count_line("Historical codes needing review", information["country_historical"])
    count_line("Missing countries", information["country_missing"])
    count_line("Unrecognized country codes", information["country_unrecognized"])
    lines.append(f"  Reference snapshot: {information['country_reference_as_of']}")
    lines.append("  Recognizes supplied 2-letter, 3-letter and numeric codes.")
    lines.append("  Historical codes are not automatically replaced.")

    for title, severity in [
        ("Missing values", "MISSING"),
        ("Invalid values", "INVALID"),
        ("Values to review", "REVIEW"),
    ]:
        heading(title)
        rows = summary[summary["severity"].eq(severity)]
        if rows.empty:
            lines.append("  None found by the implemented checks.")
            continue

        for row in rows.itertuples(index=False):
            label = row.column
            explain = None
            if severity == "REVIEW":
                if row.column == "ADR" and row.reason.startswith("ADR exceeds"):
                    label = f"ADR above {high_adr:,g}"
                elif row.column == "ADR" and row.reason.startswith("Zero/negative"):
                    label = "ADR equal to zero or negative"
                elif row.column == "Adults+Children+Babies":
                    label = "Total guests equal to zero"
                elif row.column == "StaysInWeekendNights+StaysInWeekNights":
                    label = "Total nights equal to zero"
                elif row.column == "Country":
                    label = "Historical country codes"
                    explain = row.reason
                else:
                    explain = row.reason
            elif severity == "INVALID":
                explain = row.reason

            count_line(label, row.flag_count)
            if explain:
                lines.append(fill(explain, width=70,
                                  initial_indent="    ", subsequent_indent="    "))

    heading("How to read this report")
    lines.extend([
        "  MISSING: information is unknown or empty.",
        "  INVALID: a value breaks one of the implemented rules.",
        "  REVIEW: a value needs investigation; it may be legitimate.",
        "",
        "  A booking can appear in more than one count.",
        "  The high-ADR threshold is a review prompt, not a deletion rule.",
        "  No invalid flags does not prove the data is error-free.",
        "  The source CSV has not been changed.",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/H2.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_quality"))
    parser.add_argument("--country-codes", type=Path,
                        default=Path("data/reference/Country_codes.json"),
                        help="Country_codes.json from the supplied country-code pack")
    parser.add_argument("--high-adr", type=float, default=1000.0,
                        help="Illustrative review threshold, NOT an invalid-value cutoff")
    args = parser.parse_args()
    issues, information = check_data_quality(
        args.data, args.high_adr, args.country_codes
    )
    summary = (issues.groupby(["severity", "column", "reason"], dropna=False)
               .size().reset_index(name="flag_count"))
    args.output.mkdir(parents=True, exist_ok=True)
    issues.to_csv(args.output / "flagged_values.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(information["country_code_summary"]).to_csv(
        args.output / "country_code_checks.csv", index=False
    )

    print(format_report(summary, information, args.high_adr))
    print("\nDetailed CSV reports:")
    print(f"  Folder: {args.output.resolve()}")
    print("  summary.csv        - counts by checking rule")
    print("  flagged_values.csv - individual records and reasons")
    print("  country_code_checks.csv - country codes, names and lookup results")


if __name__ == "__main__":
    main()
