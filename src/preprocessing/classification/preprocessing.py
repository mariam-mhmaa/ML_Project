"""Reusable classification data preparation utilities for Milestone 1."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_INPUT = PROJECT_ROOT / "data" / "preprocessed" / "train.csv"
CLASSIFICATION_OUTPUT = (
    PROJECT_ROOT / "data" / "preprocessed" / "classification" / "train_classification.csv"
)

TARGET_COLUMN = "IsCanceled"
EXPECTED_INPUT_COLUMNS = [
    "IsCanceled",
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "AssignedRoomType",
    "BookingChanges",
    "DepositType",
    "Agent",
    "Company",
    "DaysInWaitingList",
    "CustomerType",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
    "ReservationStatus",
    "ReservationStatusDate",
]

EXCLUDED_PREDICTORS = {
    "ReservationStatus": "Reveals the eventual outcome",
    "ReservationStatusDate": "Outcome-related timing information",
    "AssignedRoomType": "Assignment may occur after booking creation",
    "BookingChanges": "Cumulative changes after creation",
    "DaysInWaitingList": "Completed waiting duration may be unavailable at creation",
    "DepositType": "Transaction-derived category with uncertain booking-time availability",
    "ADR": "Transaction-derived rate with uncertain booking-time availability",
}

PREDICTOR_COLUMNS = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "Agent",
    "Company",
    "CustomerType",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]

OHE_CATEGORICAL_COLUMNS = [
    "ArrivalDateMonth",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "Agent",
    "Company",
    "CustomerType",
]

NUMERIC_COLUMNS = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]

MISSING_MARKERS = {"", "NA", "N/A", "NAN", "NULL", "NONE", "?"}
CHILDREN_MARKERS = MISSING_MARKERS
COUNTRY_MARKERS = {"", "NULL", "N/A", "NAN", "NONE", "?"}
AGENT_COMPANY_MARKERS = {"NULL"}


def _marker_mask(series: pd.Series, markers: set[str]) -> pd.Series:
    normalized = series.astype("string").str.strip().str.upper()
    return normalized.isin(markers)


def load_training_frame(path: Path = TRAIN_INPUT) -> pd.DataFrame:
    """Load the shared training split as strings without changing any values."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_input_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (missing_columns, unexpected_columns) for the shared train schema."""
    missing_columns = sorted(set(EXPECTED_INPUT_COLUMNS) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(EXPECTED_INPUT_COLUMNS))
    return missing_columns, unexpected_columns


def strip_text_whitespace(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Strip surrounding spaces from text and return per-column cleanup counts."""
    cleaned = df.copy()
    changes: dict[str, int] = {}

    for column in cleaned.columns:
        original = cleaned[column].astype("string")
        stripped = original.str.strip()
        changes[column] = int((original != stripped).sum())
        cleaned[column] = stripped

    return cleaned, changes


def apply_missing_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply agreed column-specific missing-value rules."""
    clean = df.copy()
    replacements: list[dict[str, object]] = []

    children_mask = _marker_mask(clean["Children"], CHILDREN_MARKERS)
    replacements.append({
        "column": "Children",
        "source_markers": "|".join(sorted(CHILDREN_MARKERS)),
        "replacement": "0",
        "replaced_rows": int(children_mask.sum()),
    })
    clean.loc[children_mask, "Children"] = "0"

    country_mask = _marker_mask(clean["Country"], COUNTRY_MARKERS)
    replacements.append({
        "column": "Country",
        "source_markers": "|".join(sorted(COUNTRY_MARKERS)),
        "replacement": "Unknown",
        "replaced_rows": int(country_mask.sum()),
    })
    clean.loc[country_mask, "Country"] = "Unknown"

    for column in ["Agent", "Company"]:
        marker_mask = _marker_mask(clean[column], AGENT_COMPANY_MARKERS)
        replacements.append({
            "column": column,
            "source_markers": "NULL",
            "replacement": "NotApplicable",
            "replaced_rows": int(marker_mask.sum()),
        })
        clean.loc[marker_mask, column] = "NotApplicable"

    return clean, pd.DataFrame(replacements)


def collect_unexpected_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Report remaining marker-like values after expected replacements."""
    issues: list[dict[str, object]] = []
    for column in df.columns:
        mask = _marker_mask(df[column], MISSING_MARKERS)
        count = int(mask.sum())
        if count:
            sample_values = (
                df.loc[mask, column].astype("string").value_counts().head(5).index.tolist()
            )
            issues.append({
                "column": column,
                "rows_with_marker_values": count,
                "sample_values": " | ".join(sample_values),
            })
    return pd.DataFrame(issues)


def build_classification_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only agreed predictors and the target, preserving row order."""
    selected_columns = PREDICTOR_COLUMNS + [TARGET_COLUMN]
    missing_selected = sorted(set(selected_columns) - set(df.columns))
    if missing_selected:
        raise ValueError(
            f"Missing required classification columns: {missing_selected}"
        )
    return df.loc[:, selected_columns].copy()


def validate_predictor_list(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check whether predictors exactly match the agreed 23-column list."""
    available_predictors = [c for c in df.columns if c != TARGET_COLUMN]
    missing_predictors = sorted(set(PREDICTOR_COLUMNS) - set(available_predictors))
    unexpected_predictors = sorted(set(available_predictors) - set(PREDICTOR_COLUMNS))
    return missing_predictors, unexpected_predictors


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric predictors and target, failing on parsing issues."""
    converted = df.copy()
    integer_columns = NUMERIC_COLUMNS + [TARGET_COLUMN]

    for column in integer_columns:
        parsed = pd.to_numeric(converted[column], errors="coerce")
        failed = converted[column].notna() & parsed.isna()
        if failed.any():
            examples = (
                converted.loc[failed, column].astype("string").value_counts().head(5).index
            )
            raise ValueError(
                f"Numeric parsing failed for {column}. Sample values: {list(examples)}"
            )
        if (parsed.mod(1) != 0).any():
            raise ValueError(f"Expected whole numbers in {column}, found non-integers.")
        converted[column] = parsed.astype("int64")

    for binary_column in [TARGET_COLUMN, "IsRepeatedGuest"]:
        values = set(converted[binary_column].unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(
                f"{binary_column} must be binary 0/1 after conversion, found {sorted(values)}"
            )

    return converted


def build_preprocessor() -> ColumnTransformer:
    """Create reusable preprocessing with one-hot encoding and passthrough numerics."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                OHE_CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                "passthrough",
                [col for col in NUMERIC_COLUMNS if col != "IsRepeatedGuest"]
                + ["IsRepeatedGuest"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_modeling_pipeline(estimator) -> Pipeline:
    """Compose fold-safe preprocessing with a downstream estimator."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])
