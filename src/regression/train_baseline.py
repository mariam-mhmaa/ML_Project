"""Untuned Linear Regression and a mean dummy using grouped training-only CV.

No test path is accepted or opened. Final holdout evaluation belongs to Milestone 2.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cse382-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.regression.preprocessing import (
    CATEGORICAL_COLUMNS, EXCLUDED_PREDICTORS, NUMERIC_COLUMNS, PREDICTOR_COLUMNS,
    RANDOM_STATE, TARGET_COLUMN, TRAIN_INPUT, build_modeling_pipeline,
    load_training_frame, prepare_training_frame,
)

OUTPUT = PROJECT_ROOT / "outputs/regression"
ARTIFACTS = PROJECT_ROOT / "artifacts/regression"
N_SPLITS = 5


def regression_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {"mae": float(mean_absolute_error(actual, predicted)), "mse": float(mse),
            "rmse": float(np.sqrt(mse)), "r2": float(r2_score(actual, predicted))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-artifact", action="store_true",
                        help="Refit the preliminary linear pipeline on all eligible training rows and save locally")
    args = parser.parse_args()
    raw = load_training_frame()
    frame, exclusions = prepare_training_frame(raw)
    X, y = frame[PREDICTOR_COLUMNS], frame[TARGET_COLUMN]
    # Target excluded. Preserve matching cleaned selected inputs as one group,
    # including when their ADR values differ. No learned imputation before CV.
    groups = X.groupby(PREDICTOR_COLUMNS, dropna=False, sort=False).ngroup().to_numpy()
    folds = list(GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X, y, groups))
    fold_assignment = np.full(len(X), -1, dtype=int)
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        if np.intersect1d(groups[train_idx], groups[val_idx]).size:
            raise RuntimeError("Matching selected inputs cross a validation fold")
        fold_assignment[val_idx] = fold
    if np.any(fold_assignment < 1):
        raise RuntimeError("Every training row must receive one out-of-fold prediction")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({"shared_train_record_number": frame.index.to_numpy() + 1,
                                "input_group": groups, "validation_fold": fold_assignment,
                                "actual_ADR": y.to_numpy()})
    estimators = {"dummy_mean": DummyRegressor(strategy="mean"), "linear_regression": LinearRegression()}
    fold_rows, summary_rows, fitted = [], [], None
    for name, estimator in estimators.items():
        oof = np.full(len(X), np.nan)
        for fold, (train_idx, val_idx) in enumerate(folds, 1):
            pipeline = build_modeling_pipeline(sklearn.base.clone(estimator))
            start = perf_counter()
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            fit_seconds = perf_counter() - start
            start = perf_counter()
            predicted = pipeline.predict(X.iloc[val_idx])
            predict_seconds = perf_counter() - start
            if not np.isfinite(predicted).all():
                raise RuntimeError("Non-finite validation predictions")
            oof[val_idx] = predicted
            row = {"model": name, "fold": fold, "train_rows": len(train_idx),
                   "validation_rows": len(val_idx), "train_groups": len(np.unique(groups[train_idx])),
                   "validation_groups": len(np.unique(groups[val_idx])), "shared_groups": 0,
                   "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                   **regression_metrics(y.iloc[val_idx], predicted)}
            fold_rows.append(row)
            print(f"{name} fold {fold}: MAE {row['mae']:.3f}, RMSE {row['rmse']:.3f}, R2 {row['r2']:.3f}", flush=True)
        predictions[f"predicted_{name}"] = oof
        model_folds = pd.DataFrame([row for row in fold_rows if row["model"] == name])
        summary = {"model": name, "evaluation": "training_only_grouped_5_fold_oof",
                   "rows": len(X), **regression_metrics(y, oof)}
        for metric in ["mae", "mse", "rmse", "r2"]:
            summary[f"fold_mean_{metric}"] = model_folds[metric].mean()
            summary[f"fold_std_{metric}"] = model_folds[metric].std(ddof=1)
        summary_rows.append(summary)
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "cv_fold_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT / "cv_summary.csv", index=False)
    predictions.to_csv(OUTPUT / "oof_predictions.csv", index=False)
    actual = predictions.actual_ADR.to_numpy()
    predicted = predictions.predicted_linear_regression.to_numpy()
    # Hexbin shows every OOF row without selecting a flattering subset.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].hexbin(actual, predicted, gridsize=55, mincnt=1, bins="log", cmap="Blues")
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[0].plot(limits, limits, "--", color="#d97706", linewidth=1)
    axes[0].set(xlabel="Recorded ADR (EUR)", ylabel="Predicted ADR (EUR)", title="Linear Regression: out-of-fold predictions")
    axes[1].hexbin(predicted, actual - predicted, gridsize=55, mincnt=1, bins="log", cmap="Blues")
    axes[1].axhline(0, color="#d97706", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Predicted ADR (EUR)", ylabel="Actual - predicted (EUR)", title="Out-of-fold residuals; predictions not clipped")
    (OUTPUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUTPUT / "figures/baseline_oof_diagnostics.png", dpi=160)
    plt.close(fig)
    subgroup_rows = []
    for name, mask in [("all", np.ones(len(y), dtype=bool)), ("zero_adr", actual == 0), ("positive_adr", actual > 0)]:
        if mask.any():
            subgroup_rows.append({"subset": name, "rows": int(mask.sum()),
                                  "mae": float(mean_absolute_error(actual[mask], predicted[mask])),
                                  "rmse": float(np.sqrt(mean_squared_error(actual[mask], predicted[mask])))})
    pd.DataFrame(subgroup_rows).to_csv(OUTPUT / "oof_subgroup_metrics.csv", index=False)

    if args.save_artifact:
        fitted = build_modeling_pipeline(LinearRegression()).fit(X, y)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        artifact_path = ARTIFACTS / "linear_regression_baseline.joblib"
        joblib.dump(fitted, artifact_path)
        # Verify serialization immediately; fresh-process loading was also checked during review.
        np.testing.assert_allclose(joblib.load(artifact_path).predict(X.iloc[:10]), fitted.predict(X.iloc[:10]))
        print(f"Saved preliminary training-only pipeline: {artifact_path.relative_to(PROJECT_ROOT)}")
    metadata = {
        "milestone": 1, "target": TARGET_COLUMN, "prediction_point": "booking_creation_oriented_proxy",
        "training_input": str(TRAIN_INPUT.relative_to(PROJECT_ROOT)),
        "training_input_sha256": hashlib.sha256(TRAIN_INPUT.read_bytes()).hexdigest(),
        "training_rows": len(raw), "eligible_rows": len(frame), "zero_night_rows_excluded": len(exclusions),
        "predictors": PREDICTOR_COLUMNS, "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS, "excluded_predictors": EXCLUDED_PREDICTORS,
        "cv": {"splitter": "GroupKFold", "n_splits": N_SPLITS, "shuffle": True,
               "random_state": RANDOM_STATE, "grouping": "identical deterministic cleaned selected predictors; ADR excluded",
               "preprocessing_fit": "independently inside each training fold"},
        "estimators": {name: estimator.get_params() for name, estimator in estimators.items()},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__, "joblib": joblib.__version__},
        "test_data_used": False, "hyperparameter_search": False, "target_clipping": False,
        "negative_oof_predictions": int((predicted < 0).sum()),
        "artifact": "artifacts/regression/linear_regression_baseline.joblib" if fitted is not None else None,
    }
    (OUTPUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows)[["model", "mae", "rmse", "r2"]].to_string(index=False))
    print("Final test split was not opened. No tuning or final model selection was performed.")


if __name__ == "__main__":
    main()
