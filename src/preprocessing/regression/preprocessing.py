"""Regression-only input contract and fold-safe preprocessing for ADR."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_INPUT = PROJECT_ROOT / "data/preprocessed/train.csv"
PREPARED_OUTPUT = PROJECT_ROOT / "data/preprocessed/regression/train_regression.csv"
TARGET_COLUMN = "ADR"
RANDOM_STATE = 42

CATEGORICAL_COLUMNS = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "Agent", "Company", "CustomerType",
]
NUMERIC_COLUMNS = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
]
PREDICTOR_COLUMNS = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights", "Adults",
    "Children", "Babies", "Meal", "Country", "MarketSegment", "DistributionChannel",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "ReservedRoomType", "Agent", "Company", "CustomerType",
    "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
]
EXCLUDED_PREDICTORS = {
    "IsCanceled": "Eventual cancellation outcome, unavailable at booking creation",
    "ReservationStatus": "Final reservation outcome",
    "ReservationStatusDate": "Date of a later reservation outcome",
    "AssignedRoomType": "Ultimate room assignment may happen after booking creation",
    "BookingChanges": "Cumulative amendments through check-in or cancellation",
    "DaysInWaitingList": "Completed waiting duration is unknown at creation",
    "DepositType": "Derived from payments and total stay cost; timing and rate proxy risk",
}
EXPECTED_INPUT_COLUMNS = set(PREDICTOR_COLUMNS) | set(EXCLUDED_PREDICTORS) | {TARGET_COLUMN}
MISSING_MARKERS = {"", "NA", "N/A", "NAN", "NULL", "NONE", "?"}
# NA can be a country identifier; do not treat it as missing in Country.
COUNTRY_MISSING_MARKERS = MISSING_MARKERS - {"NA"}
MONTHS = "January February March April May June July August September October November December".split()


def load_training_frame() -> pd.DataFrame:
    """Read only the shared training split, preserving literal NULL categories."""
    return pd.read_csv(TRAIN_INPUT, dtype=str, keep_default_na=False)


def clean_predictors(frame: pd.DataFrame, *, require_positive_nights=True) -> pd.DataFrame:
    """Deterministic cleanup only: no means, medians, vocabulary, or scaling fit."""
    missing = sorted(set(PREDICTOR_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing regression predictors: {missing}")
    clean = frame.loc[:, PREDICTOR_COLUMNS].copy()
    for column in PREDICTOR_COLUMNS:
        text = clean[column].astype("string").str.strip()
        markers = COUNTRY_MISSING_MARKERS if column == "Country" else MISSING_MARKERS
        absent = text.isna() | text.str.upper().isin(markers)
        if column in {"Agent", "Company"}:
            # The source paper defines NULL as not applicable, not missing.
            not_applicable = text.str.upper().eq("NULL").fillna(False)
            text = text.mask(not_applicable, "NotApplicable")
            absent = absent & ~not_applicable
        if column not in {"Children", "Country"} and absent.any():
            raise ValueError(f"Unexpected missing values in {column}: {int(absent.sum())}")
        if column in NUMERIC_COLUMNS:
            parsed = pd.to_numeric(text.mask(absent), errors="coerce").astype(float)
            invalid = (~absent & parsed.isna()) | (parsed.notna() & ~np.isfinite(parsed))
            if invalid.any():
                raise ValueError(f"Invalid numeric values in {column}")
            if ((parsed.dropna() < 0) | (parsed.dropna() % 1 != 0)).any():
                raise ValueError(f"{column} must contain nonnegative whole numbers")
            clean[column] = parsed
        else:
            # Object strings plus np.nan work with sklearn categorical imputers.
            clean[column] = text.astype(object).mask(absent, np.nan)
    if not clean["IsRepeatedGuest"].isin([0, 1]).all():
        raise ValueError("IsRepeatedGuest must be binary 0/1")
    if not clean["ArrivalDateMonth"].isin(MONTHS).all():
        raise ValueError("ArrivalDateMonth must be a full English month name")
    if not clean["ArrivalDateWeekNumber"].between(1, 53).all():
        raise ValueError("ArrivalDateWeekNumber must be between 1 and 53")
    if not clean["ArrivalDateDayOfMonth"].between(1, 31).all():
        raise ValueError("ArrivalDateDayOfMonth must be between 1 and 31")
    nights = clean["StaysInWeekendNights"] + clean["StaysInWeekNights"]
    if require_positive_nights and nights.eq(0).any():
        raise ValueError("ADR regression requires at least one booked night")
    return clean


def prepare_training_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select positive-night bookings without modifying the shared training data."""
    if set(raw.columns) != EXPECTED_INPUT_COLUMNS or raw.columns.duplicated().any():
        raise ValueError("Shared training schema differs from the expected 31 H2 columns")
    predictors = clean_predictors(raw, require_positive_nights=False)
    target = pd.to_numeric(raw[TARGET_COLUMN].astype("string").str.strip(), errors="coerce").astype(float)
    if not np.isfinite(target).all() or target.lt(0).any():
        raise ValueError("ADR must be present, finite, and nonnegative; never impute the target")
    eligible = (predictors["StaysInWeekendNights"] + predictors["StaysInWeekNights"]).gt(0)
    exclusions = raw.loc[~eligible, [TARGET_COLUMN, "StaysInWeekendNights", "StaysInWeekNights"]].copy()
    exclusions.insert(0, "shared_train_record_number", np.flatnonzero(~eligible) + 1)
    exclusions.insert(1, "reason", "zero_booked_nights_ADR_denominator_undefined")
    prepared = predictors.loc[eligible].copy()
    prepared[TARGET_COLUMN] = target.loc[eligible]
    return prepared, exclusions


class RegressionInputCleaner(TransformerMixin, BaseEstimator):
    """Keep deterministic input validation inside the persisted inference pipeline."""

    def fit(self, X, y=None):
        clean_predictors(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        return clean_predictors(X)


def build_preprocessor() -> ColumnTransformer:
    """All learned quantities are fitted by Pipeline.fit within a training fold."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ], remainder="drop", sparse_threshold=1.0)


def build_modeling_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("cleaner", RegressionInputCleaner()),
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])
