from pathlib import Path

import pandas as pd


# Locate the project folder from this script's location.
project_folder = Path(__file__).resolve().parents[2]
csv_path = project_folder / "data" / "raw" / "H2.csv"

# Read the CSV into a table in memory.
# This does not modify the original file.
df = pd.read_csv(csv_path)

print("\n1. DATASET SIZE")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\n2. COLUMN NAMES")
for number, column in enumerate(df.columns, start=1):
    print(f"{number}. {column}")

print("\n3. FIRST FIVE BOOKINGS — FIRST SIX COLUMNS")
print(df.iloc[:5, :6].to_string(index=False))

print("\n4. TYPES DETECTED BY PANDAS")
print(df.dtypes.to_string())

print("\n5. VALUES PANDAS INTERPRETED AS MISSING")
missing_counts = df.isna().sum()
print(missing_counts.sort_values(ascending=False).to_string())

print("\n6. EXACT DUPLICATE ROWS")
print(f"Duplicates after the first occurrence: {df.duplicated().sum():,}")

# Second Run

print("\n7. ORIGINAL TEXT VALUES IN SELECTED COLUMNS")

# Read these columns again while preserving their text exactly.
# This helps us see what pandas may have interpreted as missing.
raw_text = pd.read_csv(
    csv_path,
    usecols=["Agent", "Company", "Country", "Children"],
    dtype="string",
    keep_default_na=False,
)

for column in raw_text.columns:
    print(f"\nColumn: {column}")
    print("Ten most frequent original values:")

    for value, count in raw_text[column].value_counts().head(10).items():
        # repr() makes surrounding spaces visible.
        print(f"  {value!r}: {count:,}")

    # These are inspection candidates, not automatic cleaning rules.
    normalized = raw_text[column].str.strip().str.upper()
    marker_mask = normalized.isin(
        ["", "NULL", "NA", "N/A", "NONE", "NAN", "?"]
    )

    print("Possible empty/special text markers:")
    marker_counts = raw_text.loc[marker_mask, column].value_counts()

    if marker_counts.empty:
        print("  None found.")
    else:
        for value, count in marker_counts.items():
            print(f"  {value!r}: {count:,}")


print("\n8. CHECK CHILDREN FOR FRACTIONAL VALUES")

known_children = df["Children"].dropna()
fractional_children = known_children[known_children % 1 != 0]

print(f"Non-missing fractional values: {len(fractional_children):,}")


print("\n9. ALL ROWS INVOLVED IN EXACT DUPLICATES")

print(
    "Rows belonging to repeated groups, including first occurrences:",
    f"{df.duplicated(keep=False).sum():,}",
)