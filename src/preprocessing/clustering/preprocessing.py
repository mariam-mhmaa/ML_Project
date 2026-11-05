"""Training-only feature contract for the Milestone 1 clustering baseline.

The clustering analysis describes booking behaviour known at booking creation.
Final outcomes and fields that can change later in the booking lifecycle are never
passed to the clustering algorithm.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_INPUT = PROJECT_ROOT / "data/preprocessed/train.csv"
RANDOM_STATE = 42

# These source fields are available at booking creation and describe timing,
# party/stay size, repeat history, and explicitly requested services.
SOURCE_FEATURES = [
    "LeadTime",
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

# The model uses compact, interpretable derived features rather than allowing
# weekend/weekday nights or adult/child counts to receive duplicated influence.
MODEL_FEATURES = [
    "LeadTime",
    "TotalNights",
    "TotalGuests",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]

# Log1p reduces the leverage of long-tailed count variables before Euclidean
# distance is calculated. Binary and small bounded counts remain untransformed.
LOG_FEATURES = [
    "LeadTime",
    "TotalNights",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
]
LINEAR_FEATURES = [
    "TotalGuests",
    "IsRepeatedGuest",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]

# This explicit audit inventory explains why every unused H2 source variable is
# outside the baseline. Outcome/future fields are separated from benign fields
# excluded only to keep the initial segmentation focused and interpretable.
LEAKAGE_EXCLUSIONS = {
    "IsCanceled": "Final cancellation outcome.",
    "ReservationStatus": "Final reservation outcome.",
    "ReservationStatusDate": "Date on which the final status was recorded.",
    "AssignedRoomType": "Ultimate room assignment can occur after booking creation.",
    "BookingChanges": "Cumulative amendments before check-in or cancellation.",
    "DaysInWaitingList": "Completed waiting duration is not known at booking creation.",
    "ADR": "Price/outcome field reserved for regression; it could dominate segment identity.",
    "DepositType": "Payment-related state and timing are not verified at booking creation.",
}
FOCUS_EXCLUSIONS = {
    "ArrivalDateYear": "Calendar field omitted from the compact behavioural baseline.",
    "ArrivalDateMonth": "Calendar field omitted from the compact behavioural baseline.",
    "ArrivalDateWeekNumber": "Calendar field omitted from the compact behavioural baseline.",
    "ArrivalDateDayOfMonth": "Calendar field omitted from the compact behavioural baseline.",
    "Meal": "Categorical preference deferred to later mixed-feature experiments.",
    "Country": "High-cardinality geography deferred to later experiments.",
    "MarketSegment": "Business-source category deferred to later mixed-feature experiments.",
    "DistributionChannel": "Business-source category deferred to later experiments.",
    "ReservedRoomType": "Room category deferred to later mixed-feature experiments.",
    "Agent": "High-cardinality identifier-like field excluded from this baseline.",
    "Company": "High-cardinality identifier-like field excluded from this baseline.",
    "CustomerType": "Categorical field deferred to later mixed-feature experiments.",
}

MISSING_MARKERS = {"", "NA", "N/A", "NAN", "NULL", "NONE", "?"}


def load_training_frame() -> pd.DataFrame:
    """Load only the shared training split and preserve literal text markers."""
    return pd.read_csv(TRAIN_INPUT, dtype=str, keep_default_na=False)


def prepare_clustering_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate source values and derive the model's eight behavioural inputs.

    This function performs deterministic row-wise work only. Median imputation,
    log transformation, and scaling remain inside the fitted sklearn pipeline.
    """
    missing = sorted(set(SOURCE_FEATURES) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing clustering source features: {missing}")

    numeric = pd.DataFrame(index=raw.index)
    for column in SOURCE_FEATURES:
        text = raw[column].astype("string").str.strip()
        absent = text.isna() | text.str.upper().isin(MISSING_MARKERS)
        parsed = pd.to_numeric(text.mask(absent), errors="coerce").astype(float)
        invalid = (~absent & parsed.isna()) | (parsed.notna() & ~np.isfinite(parsed))
        if invalid.any():
            raise ValueError(f"Invalid numeric values in clustering feature {column}")
        if (parsed.dropna() < 0).any():
            raise ValueError(f"{column} must be nonnegative")
        numeric[column] = parsed

    # Only Children is expected to be missing in the cleaned shared split. The
    # pipeline imputes it through TotalGuests; unexpected gaps fail loudly.
    unexpected_missing = numeric.drop(columns="Children").isna().sum()
    if unexpected_missing.any():
        details = unexpected_missing[unexpected_missing.gt(0)].to_dict()
        raise ValueError(f"Unexpected clustering missing values: {details}")
    if not numeric["IsRepeatedGuest"].isin([0, 1]).all():
        raise ValueError("IsRepeatedGuest must be binary 0/1")

    features = pd.DataFrame(index=raw.index)
    features["LeadTime"] = numeric["LeadTime"]
    features["TotalNights"] = (
        numeric["StaysInWeekendNights"] + numeric["StaysInWeekNights"]
    )
    # min_count keeps TotalGuests missing when Children is missing; the learned
    # median imputer then treats those four records consistently in the pipeline.
    features["TotalGuests"] = numeric[["Adults", "Children", "Babies"]].sum(
        axis=1, min_count=3
    )
    for column in MODEL_FEATURES[3:]:
        features[column] = numeric[column]
    return features.loc[:, MODEL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Build imputation, skew reduction, and scaling for K-Means distances."""
    log_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])
    linear_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("log_counts", log_pipeline, LOG_FEATURES),
        ("linear_counts", linear_pipeline, LINEAR_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)
