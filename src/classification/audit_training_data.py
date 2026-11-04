"""Create a focused Milestone 1 audit for classification training data.

Run from the project folder:
    python src/classification/audit_training_data.py
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.classification.preprocessing import (
    CLASSIFICATION_OUTPUT,
    EXCLUDED_PREDICTORS,
    NUMERIC_COLUMNS,
    OHE_CATEGORICAL_COLUMNS,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    load_training_frame,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification"
AUDIT_SUMMARY_FILE = OUTPUT_DIR / "audit_summary.csv"
CANCELLATION_SUMMARY_FEATURES = [
    "ArrivalDateMonth",
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "CustomerType",
    "IsRepeatedGuest",
]


def _check_prepared_data_exists():
    if not CLASSIFICATION_OUTPUT.is_file():
        raise FileNotFoundError(
            "Prepared dataset not found. Run "
            "python src/preprocessing/classification/prepare_classification_data.py first."
        )


def _column_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        role = "target" if column == TARGET_COLUMN else "predictor"
        transform = "numeric passthrough"
        if column in OHE_CATEGORICAL_COLUMNS:
            transform = "one-hot encoding"
        rows.append({
            "column": column,
            "role": role,
            "dtype": str(df[column].dtype),
            "transformation": transform,
        })

    for column, reason in EXCLUDED_PREDICTORS.items():
        rows.append({
            "column": column,
            "role": "excluded predictor",
            "dtype": "not included",
            "transformation": reason,
        })

    return pd.DataFrame(rows)


def _cardinality_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in OHE_CATEGORICAL_COLUMNS:
        counts = df[column].value_counts(dropna=False)
        rows.append({
            "column": column,
            "n_unique": int(df[column].nunique(dropna=False)),
            "top_category": counts.index[0],
            "top_category_count": int(counts.iloc[0]),
        })
    return pd.DataFrame(rows)


def _repeated_input_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors_only = df[PREDICTOR_COLUMNS]
    grouped = (
        df.groupby(PREDICTOR_COLUMNS, dropna=False, sort=False)[TARGET_COLUMN]
        .agg(["size", "nunique"])
        .reset_index()
        .rename(columns={"size": "group_rows", "nunique": "distinct_target_labels"})
    )
    repeated = grouped[grouped["group_rows"] > 1].copy()
    conflicting = repeated[repeated["distinct_target_labels"] > 1].copy()

    repeated_rows_count = int(predictors_only.duplicated(keep=False).sum())
    summary = pd.DataFrame([{
        "repeated_selected_input_rows": repeated_rows_count,
        "repeated_selected_input_groups": int(len(repeated)),
        "conflicting_label_groups": int(len(conflicting)),
    }])

    return repeated, conflicting, summary


def _cancellation_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    part = (
        df.groupby(column, dropna=False)[TARGET_COLUMN]
        .agg(sample_count="count", canceled_rate="mean")
        .reset_index()
    )
    part["canceled_pct"] = (part["canceled_rate"] * 100).round(3)
    part["feature"] = column
    part = part.rename(columns={column: "category"})
    part["category"] = part["category"].astype("string")
    return part[["feature", "category", "sample_count", "canceled_pct"]].sort_values(
        by="sample_count", ascending=False
    )


def main():
    _check_prepared_data_exists()

    raw_train = load_training_frame()
    prepared = pd.read_csv(CLASSIFICATION_OUTPUT)
    prepared_labels = prepared[TARGET_COLUMN].astype("string").tolist()
    raw_labels = raw_train[TARGET_COLUMN].astype("string").str.strip().tolist()

    if len(prepared) != len(raw_train):
        raise ValueError("Prepared dataset row count differs from train.csv.")
    if prepared_labels != raw_labels:
        raise ValueError("Prepared target sequence does not match train.csv.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_counts = prepared[TARGET_COLUMN].value_counts().sort_index()
    target_dist = pd.DataFrame({
        "class": target_counts.index.astype(str),
        "count": target_counts.values,
        "percentage": (100 * target_counts.values / len(prepared)).round(3),
    })
    target_dist.to_csv(OUTPUT_DIR / "target_distribution.csv", index=False)

    inventory = _column_inventory(prepared)
    inventory.to_csv(OUTPUT_DIR / "column_inventory.csv", index=False)

    cardinality = _cardinality_table(prepared)
    cardinality.to_csv(OUTPUT_DIR / "categorical_cardinality.csv", index=False)

    repeated, conflicting, repeat_summary = _repeated_input_groups(prepared)
    repeated.to_csv(OUTPUT_DIR / "repeated_selected_input_groups.csv", index=False)
    conflicting.to_csv(OUTPUT_DIR / "conflicting_label_groups.csv", index=False)

    cancellation_tables = [
        _cancellation_summary(prepared, column) for column in CANCELLATION_SUMMARY_FEATURES
    ]
    pd.concat(cancellation_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "focused_cancellation_summary.csv",
        index=False,
    )

    numeric_dtypes = pd.DataFrame([{
        "column": column,
        "dtype": str(prepared[column].dtype),
    } for column in NUMERIC_COLUMNS + [TARGET_COLUMN]])
    numeric_dtypes.to_csv(OUTPUT_DIR / "numeric_dtype_check.csv", index=False)

    summary = pd.DataFrame([{
        "rows": len(prepared),
        "columns": len(prepared.columns),
        "predictor_columns": len(PREDICTOR_COLUMNS),
        "target_column": TARGET_COLUMN,
        "canceled_rows": int(target_counts.get(1, 0)),
        "not_canceled_rows": int(target_counts.get(0, 0)),
        "canceled_pct": round(float((prepared[TARGET_COLUMN] == 1).mean() * 100), 3),
    }]).join(repeat_summary)
    summary.to_csv(AUDIT_SUMMARY_FILE, index=False)

    print(f"Audit summary: {AUDIT_SUMMARY_FILE}")
    print(f"Prepared rows verified against shared train.csv: {len(prepared):,}")
    print(f"Conflicting-label predictor groups: {int(len(conflicting)):,}")
    print(f"Detailed tables saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
