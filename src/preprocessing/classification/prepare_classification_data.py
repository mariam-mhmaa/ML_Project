"""Prepare classification training data from the shared split.

Run from the project folder:
    python src/preprocessing/classification/prepare_classification_data.py
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.classification.preprocessing import (
    CLASSIFICATION_OUTPUT,
    TARGET_COLUMN,
    apply_missing_rules,
    build_classification_frame,
    collect_unexpected_missing_markers,
    convert_numeric_columns,
    load_training_frame,
    strip_text_whitespace,
    validate_input_schema,
    validate_predictor_list,
)


OUTPUTS_FOLDER = PROJECT_ROOT / "outputs" / "classification"


def main():
    raw_train = load_training_frame()
    expected_rows = len(raw_train)
    original_labels = raw_train[TARGET_COLUMN].astype("string").str.strip().tolist()

    missing_columns, unexpected_columns = validate_input_schema(raw_train)
    if missing_columns or unexpected_columns:
        raise ValueError(
            "Shared train.csv schema mismatch. "
            f"Missing: {missing_columns}; unexpected: {unexpected_columns}"
        )

    stripped, whitespace_changes = strip_text_whitespace(raw_train)
    cleaned, replacement_table = apply_missing_rules(stripped)

    class_df = build_classification_frame(cleaned)
    missing_predictors, unexpected_predictors = validate_predictor_list(class_df)
    if missing_predictors or unexpected_predictors:
        raise ValueError(
            "Predictor mismatch against agreed 23-column list. "
            f"Missing: {missing_predictors}; unexpected: {unexpected_predictors}"
        )

    unexpected_markers = collect_unexpected_missing_markers(class_df)
    if not unexpected_markers.empty:
        OUTPUTS_FOLDER.mkdir(parents=True, exist_ok=True)
        unexpected_path = OUTPUTS_FOLDER / "unexpected_missing_markers.csv"
        unexpected_markers.to_csv(unexpected_path, index=False)
        raise ValueError(
            "Unexpected missing-value markers remain after agreed replacements. "
            f"See {unexpected_path}"
        )

    converted = convert_numeric_columns(class_df)

    if len(converted) != expected_rows:
        raise ValueError("Row count changed during preparation.")
    prepared_labels = converted[TARGET_COLUMN].astype("int64").astype("string").tolist()
    if prepared_labels != original_labels:
        raise ValueError("Target labels or row order changed during preparation.")

    CLASSIFICATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(CLASSIFICATION_OUTPUT, index=False)

    OUTPUTS_FOLDER.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"column": key, "trimmed_values": value} for key, value in whitespace_changes.items()]
    ).to_csv(OUTPUTS_FOLDER / "whitespace_cleanup_counts.csv", index=False)
    replacement_table.to_csv(OUTPUTS_FOLDER / "missing_value_replacements.csv", index=False)

    print(f"Prepared classification data: {CLASSIFICATION_OUTPUT}")
    print(f"Rows retained: {len(converted):,}")
    print(
        f"Predictors: {len(converted.columns) - 1} | Target column: {TARGET_COLUMN}"
    )
    print(
        "Saved preparation logs: "
        f"{OUTPUTS_FOLDER / 'whitespace_cleanup_counts.csv'} and "
        f"{OUTPUTS_FOLDER / 'missing_value_replacements.csv'}"
    )


if __name__ == "__main__":
    main()
