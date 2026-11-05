"""Run the training-only Milestone 1 K-Means booking segmentation baseline.

Run from the repository root:
    python src/clustering/run_milestone1_baseline.py

The script never opens the reserved test split. It saves assignments, internal
metrics, cluster sizes, interpretable profiles, a leakage audit, a figure, and
reproducibility metadata under outputs/clustering/.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cse382-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.clustering.preprocessing import (  # noqa: E402
    FOCUS_EXCLUSIONS,
    LEAKAGE_EXCLUSIONS,
    MODEL_FEATURES,
    RANDOM_STATE,
    SOURCE_FEATURES,
    TRAIN_INPUT,
    build_preprocessor,
    load_training_frame,
    prepare_clustering_features,
)


OUTPUT = PROJECT_ROOT / "outputs/clustering"
N_CLUSTERS = 4
N_INIT = 10
SILHOUETTE_SAMPLE_SIZE = 10_000


def build_pipeline() -> Pipeline:
    """Create the full reusable preprocessing-plus-clustering pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", KMeans(
            n_clusters=N_CLUSTERS,
            init="k-means++",
            n_init=N_INIT,
            random_state=RANDOM_STATE,
        )),
    ])


def _stable_cluster_names(profiles: pd.DataFrame) -> dict[int, str]:
    """Give clusters deterministic descriptive names from dominant profiles."""
    remaining = set(profiles.index.astype(int))
    names: dict[int, str] = {}
    rules = [
        ("repeat-history guests", "PreviousBookingsNotCanceled_mean"),
        ("parking-space request bookings", "RequiredCarParkingSpaces_mean"),
        ("special-request bookings", "TotalOfSpecialRequests_mean"),
    ]
    for name, column in rules:
        candidate = int(profiles.loc[list(remaining), column].idxmax())
        names[candidate] = name
        remaining.remove(candidate)
    # With four clusters one residual group remains. Keeping the naming rule
    # explicit prevents subjective manual relabeling after seeing results.
    for cluster in sorted(remaining):
        names[cluster] = "low-request standard bookings"
    return names


def _build_profiles(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Summarize cluster characteristics in original, human-readable units."""
    labeled = features.copy()
    labeled.insert(0, "cluster", labels)
    grouped = labeled.groupby("cluster", sort=True)
    parts = [grouped.size().rename("booking_count")]
    for column in MODEL_FEATURES:
        parts.extend([
            grouped[column].mean().rename(f"{column}_mean"),
            grouped[column].median().rename(f"{column}_median"),
        ])
    profiles = pd.concat(parts, axis=1).reset_index()
    profiles.insert(2, "booking_share", profiles["booking_count"] / len(features))
    names = _stable_cluster_names(profiles.set_index("cluster"))
    profiles.insert(1, "profile_name", profiles["cluster"].map(names))
    return profiles


def _save_profile_figure(profiles: pd.DataFrame) -> None:
    """Plot standardized profile means so differently scaled fields compare."""
    means = profiles.set_index("profile_name")[
        [f"{column}_mean" for column in MODEL_FEATURES]
    ].copy()
    means.columns = MODEL_FEATURES
    standardized = (means - means.mean(axis=0)) / means.std(axis=0, ddof=0)
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    image = ax.imshow(standardized, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(MODEL_FEATURES)), MODEL_FEATURES, rotation=35, ha="right")
    ax.set_yticks(range(len(standardized)), standardized.index)
    ax.set_title("K-Means cluster profiles (standardized cluster means)")
    ax.set_xlabel("Booking attribute")
    ax.set_ylabel("Descriptive cluster profile")
    fig.colorbar(image, ax=ax, label="Mean relative to the four cluster means")
    fig.savefig(OUTPUT / "cluster_profile_heatmap.png", dpi=170)
    plt.close(fig)


def main() -> None:
    """Fit the baseline, validate it, and write every required result."""
    raw = load_training_frame()
    features = prepare_clustering_features(raw)
    pipeline = build_pipeline()

    started = perf_counter()
    labels = pipeline.fit_predict(features)
    fit_seconds = perf_counter() - started
    transformed = pipeline.named_steps["preprocessor"].transform(features)

    if len(labels) != len(features) or np.unique(labels).size != N_CLUSTERS:
        raise RuntimeError("K-Means did not assign every row to the requested four clusters")
    if not np.isfinite(transformed).all():
        raise RuntimeError("Clustering representation contains non-finite values")

    # A fixed random sample makes the O(n^2) Silhouette calculation reproducible
    # and practical. Davies-Bouldin is linear enough to use all training rows.
    silhouette_rows = min(SILHOUETTE_SAMPLE_SIZE, len(features))
    silhouette = silhouette_score(
        transformed,
        labels,
        sample_size=silhouette_rows,
        random_state=RANDOM_STATE,
    )
    davies_bouldin = davies_bouldin_score(transformed, labels)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    assignments = pd.DataFrame({
        "shared_train_record_number": np.arange(1, len(features) + 1),
        "cluster": labels,
    })
    assignments.to_csv(OUTPUT / "cluster_assignments.csv", index=False)

    profiles = _build_profiles(features, labels)
    profiles.to_csv(OUTPUT / "cluster_profiles.csv", index=False)
    profiles[["cluster", "profile_name", "booking_count", "booking_share"]].to_csv(
        OUTPUT / "cluster_sizes.csv", index=False
    )

    metrics = pd.DataFrame([{
        "algorithm": "KMeans",
        "configuration": "milestone1_untuned_baseline",
        "n_clusters": N_CLUSTERS,
        "training_rows": len(features),
        "silhouette_score": silhouette,
        "silhouette_sample_rows": silhouette_rows,
        "davies_bouldin_index": davies_bouldin,
        "inertia": pipeline.named_steps["model"].inertia_,
        "fit_seconds": fit_seconds,
    }])
    metrics.to_csv(OUTPUT / "baseline_metrics.csv", index=False)

    audit_rows = []
    for column in SOURCE_FEATURES:
        audit_rows.append({
            "source_column": column,
            "decision": "included_or_used_in_derived_feature",
            "reason": "Booking-creation attribute used in the behavioural segmentation.",
        })
    for column, reason in LEAKAGE_EXCLUSIONS.items():
        audit_rows.append({"source_column": column, "decision": "excluded_leakage_risk", "reason": reason})
    for column, reason in FOCUS_EXCLUSIONS.items():
        audit_rows.append({"source_column": column, "decision": "excluded_baseline_focus", "reason": reason})
    audit = pd.DataFrame(audit_rows).sort_values("source_column")
    if set(audit["source_column"]) != set(raw.columns):
        raise RuntimeError("Leakage audit must account for all 31 H2 source columns")
    audit.to_csv(OUTPUT / "leakage_audit.csv", index=False)
    _save_profile_figure(profiles)

    metadata = {
        "milestone": 1,
        "purpose": "Discover training-set booking behaviour segments without outcome labels.",
        "training_input": str(TRAIN_INPUT.relative_to(PROJECT_ROOT)),
        "training_input_sha256": hashlib.sha256(TRAIN_INPUT.read_bytes()).hexdigest(),
        "test_data_used": False,
        "source_features": SOURCE_FEATURES,
        "model_features": MODEL_FEATURES,
        "derived_features": {
            "TotalNights": "StaysInWeekendNights + StaysInWeekNights",
            "TotalGuests": "Adults + Children + Babies",
        },
        "preprocessing": {
            "missing_values": "Median imputation inside the fitted pipeline; affects four missing Children values.",
            "skew": "log1p for LeadTime, TotalNights, PreviousCancellations, and PreviousBookingsNotCanceled.",
            "scaling": "StandardScaler on all eight model features before Euclidean K-Means.",
        },
        "algorithm": {
            "name": "KMeans",
            "parameters": pipeline.named_steps["model"].get_params(deep=True),
            "parameter_status": "Initial untuned baseline; k=4 chosen a priori for a compact interpretable segmentation.",
        },
        "metrics": {
            "silhouette_score": silhouette,
            "silhouette_sample_rows": silhouette_rows,
            "silhouette_sampling": "Fixed random sample from all training rows, random_state=42.",
            "davies_bouldin_index": davies_bouldin,
            "inertia": pipeline.named_steps["model"].inertia_,
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (OUTPUT / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(metrics.to_string(index=False))
    print("\nCluster sizes and descriptive names:")
    print(profiles[["cluster", "profile_name", "booking_count", "booking_share"]].to_string(index=False))
    print("\nFinal test split was not opened. This is an untuned Milestone 1 baseline.")


if __name__ == "__main__":
    main()
