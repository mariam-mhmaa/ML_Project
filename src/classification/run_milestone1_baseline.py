"""Run the Milestone 1 untuned classification baseline on training data only.

Run from the project folder:
    python src/classification/run_milestone1_baseline.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.classification.preprocessing import (  # noqa: E402
    CLASSIFICATION_OUTPUT,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    build_modeling_pipeline,
)


RANDOM_STATE = 42
N_SPLITS = 5
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "classification"
MODEL_ARTIFACT = ARTIFACT_DIR / "random_forest_baseline_pipeline.joblib"


@dataclass
class ModelSpec:
    name: str
    estimator_factory: callable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _portable_text_sha256_utf8_lf(path: Path) -> str:
    """Hash text after UTF-8 decode and LF normalization.

    Normalization definition:
    - decode bytes as UTF-8
    - replace CRLF and CR with LF
    - encode normalized text as UTF-8 and hash
    """
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head_info() -> dict[str, object]:
    try:
        import subprocess

        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
        return {"available": True, "sha": sha, "failure_reason": None}
    except Exception as exc:
        return {"available": False, "sha": None, "failure_reason": str(exc)}


def _git_status_info() -> dict[str, object]:
    try:
        import subprocess

        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
        )
        lines = [line for line in output.splitlines() if line.strip()]
        return {
            "available": True,
            "lines": lines,
            "failure_reason": None,
            "is_clean": len(lines) == 0,
        }
    except Exception as exc:
        return {
            "available": False,
            "lines": None,
            "failure_reason": str(exc),
            "is_clean": None,
        }


def _to_posix_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _source_file_hashes(source_files: dict[str, Path]) -> dict[str, dict[str, object]]:
    hashes: dict[str, dict[str, object]] = {}
    for key, path in source_files.items():
        try:
            hashes[key] = {
                "path": str(path),
                "relative_path": _to_posix_relative(path),
                "sha256_bytes": _sha256(path),
                "sha256_utf8_lf_normalized": _portable_text_sha256_utf8_lf(path),
                "text_hash_normalization": "UTF-8 text with LF line endings",
                "read_error": None,
            }
        except Exception as exc:
            hashes[key] = {
                "path": str(path),
                "relative_path": _to_posix_relative(path),
                "sha256_bytes": None,
                "sha256_utf8_lf_normalized": None,
                "text_hash_normalization": "UTF-8 text with LF line endings",
                "read_error": str(exc),
            }
    return hashes


def _classify_worktree_changes(
    status_lines: list[str],
    relevant_source_rel_paths: set[str],
) -> dict[str, object]:
    tracked_changes = []
    untracked_paths = []
    for line in status_lines:
        status = line[:2]
        rel_path = line[3:].strip() if len(line) > 3 else ""
        if status == "??":
            untracked_paths.append(rel_path)
        else:
            tracked_changes.append(rel_path)

    tracked_source = [p for p in tracked_changes if p in relevant_source_rel_paths]
    untracked_source = [p for p in untracked_paths if p in relevant_source_rel_paths]
    tracked_non_source = [p for p in tracked_changes if p not in relevant_source_rel_paths]
    untracked_non_source = [p for p in untracked_paths if p not in relevant_source_rel_paths]

    return {
        "has_any_changes": bool(tracked_changes or untracked_paths),
        "has_tracked_changes": bool(tracked_changes),
        "has_untracked_changes": bool(untracked_paths),
        "has_relevant_source_changes": bool(tracked_source or untracked_source),
        "has_non_source_changes": bool(tracked_non_source or untracked_non_source),
        "tracked_change_paths": tracked_changes,
        "untracked_change_paths": untracked_paths,
        "relevant_tracked_source_paths": tracked_source,
        "relevant_untracked_source_paths": untracked_source,
        "tracked_non_source_paths": tracked_non_source,
        "untracked_non_source_paths": untracked_non_source,
    }


def _collect_source_state_snapshot(source_files: dict[str, Path]) -> dict[str, object]:
    git_head = _git_head_info()
    git_status = _git_status_info()
    source_hashes = _source_file_hashes(source_files)
    relevant_source_paths = {entry["relative_path"] for entry in source_hashes.values()}

    if git_status["available"]:
        worktree = _classify_worktree_changes(git_status["lines"], relevant_source_paths)
        worktree_state = "clean" if git_status["is_clean"] else "dirty"
    else:
        worktree = {
            "has_any_changes": None,
            "has_tracked_changes": None,
            "has_untracked_changes": None,
            "has_relevant_source_changes": None,
            "has_non_source_changes": None,
            "tracked_change_paths": None,
            "untracked_change_paths": None,
            "relevant_tracked_source_paths": None,
            "relevant_untracked_source_paths": None,
            "tracked_non_source_paths": None,
            "untracked_non_source_paths": None,
        }
        worktree_state = "unavailable"

    return {
        "captured_at_utc": _utc_now_iso(),
        "git_head": git_head,
        "git_status": {
            "available": git_status["available"],
            "state": worktree_state,
            "failure_reason": git_status["failure_reason"],
        },
        "worktree": worktree,
        "source_code": source_hashes,
    }


def _compare_source_snapshots(
    start_snapshot: dict[str, object],
    end_snapshot: dict[str, object],
) -> dict[str, object]:
    changed_keys = []
    for key in start_snapshot["source_code"]:
        start_hash = start_snapshot["source_code"][key]["sha256_bytes"]
        end_hash = end_snapshot["source_code"][key]["sha256_bytes"]
        if start_hash != end_hash:
            changed_keys.append(key)
    return {
        "relevant_source_files_changed_during_run": bool(changed_keys),
        "changed_source_file_keys": changed_keys,
        "source_files_unchanged_during_run": sorted(
            [key for key in start_snapshot["source_code"] if key not in changed_keys]
        ),
    }


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_avg_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_canceled": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_canceled": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_canceled": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(y_true, y_prob),
        "average_precision": _safe_avg_precision(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support_total": int(len(y_true)),
        "support_canceled": int((y_true == 1).sum()),
        "support_not_canceled": int((y_true == 0).sum()),
    }


def _build_group_ids(data: pd.DataFrame) -> np.ndarray:
    return data.groupby(PREDICTOR_COLUMNS, sort=False, dropna=False).ngroup().to_numpy()


def _validate_fold_integrity(
    fold_assignments: pd.DataFrame,
    groups: np.ndarray,
    n_rows: int,
) -> None:
    if len(fold_assignments) != n_rows:
        raise ValueError("Fold assignment table length mismatch.")
    if fold_assignments["fold"].isna().any():
        raise ValueError("At least one row did not receive a held-out fold assignment.")
    if not fold_assignments["row_index"].is_unique:
        raise ValueError("Fold assignment row_index values are not unique.")

    assignment_counts = fold_assignments["row_index"].value_counts()
    if assignment_counts.max() != 1 or assignment_counts.min() != 1:
        raise ValueError("Each row must appear exactly once in held-out fold assignments.")

    by_group = fold_assignments.groupby("group_id")["fold"].nunique()
    if (by_group > 1).any():
        raise ValueError("At least one group appears in multiple validation folds.")

    original_group_ids = pd.Series(groups, index=np.arange(n_rows))
    mapped = fold_assignments.set_index("row_index")["group_id"]
    if not original_group_ids.equals(mapped.sort_index()):
        raise ValueError("Fold assignment group IDs do not match source groups.")


def _make_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="random_forest",
            estimator_factory=lambda: RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            name="dummy_prior",
            estimator_factory=lambda: DummyClassifier(strategy="prior"),
        ),
    ]


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not canceled (0)", "Canceled (1)"],
    )
    display.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_curve_comparison(
    curves: list[dict[str, object]],
    x_label: str,
    y_label: str,
    title: str,
    chance_line: bool,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for item in curves:
        ax.plot(item["x"], item["y"], label=item["label"], linewidth=2)
    if chance_line:
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Chance")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    if not CLASSIFICATION_OUTPUT.is_file():
        raise FileNotFoundError(
            "Missing prerequisite prepared dataset. Run "
            "python src/preprocessing/classification/prepare_classification_data.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_started_utc = _utc_now_iso()
    source_files = {
        "training_script": Path(__file__).resolve(),
        "classification_preprocessing": (
            PROJECT_ROOT / "src" / "preprocessing" / "classification" / "preprocessing.py"
        ),
        "classification_prepare_script": (
            PROJECT_ROOT
            / "src"
            / "preprocessing"
            / "classification"
            / "prepare_classification_data.py"
        ),
    }
    source_state_start = _collect_source_state_snapshot(source_files)

    data = pd.read_csv(CLASSIFICATION_OUTPUT)
    expected_columns = PREDICTOR_COLUMNS + [TARGET_COLUMN]
    if data.columns.tolist() != expected_columns:
        raise ValueError(
            "Prepared dataset columns do not match expected predictor + target order."
        )

    X = data[PREDICTOR_COLUMNS].copy()
    y = data[TARGET_COLUMN].to_numpy(dtype=int)
    groups = _build_group_ids(data)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    folds = list(splitter.split(X, y, groups))

    fold_assignment = pd.DataFrame({
        "row_index": np.arange(len(data), dtype=int),
        "group_id": groups.astype(int),
        "fold": np.nan,
    })
    for fold_idx, (_, valid_idx) in enumerate(folds):
        fold_assignment.loc[valid_idx, "fold"] = fold_idx
    fold_assignment["fold"] = fold_assignment["fold"].astype(int)
    _validate_fold_integrity(fold_assignment, groups, len(data))
    fold_assignment.to_csv(OUTPUT_DIR / "cv_fold_assignments.csv", index=False)

    model_specs = _make_model_specs()
    per_fold_validation_rows: list[dict[str, object]] = []
    per_fold_training_rows: list[dict[str, object]] = []
    all_oof_predictions: list[pd.DataFrame] = []

    for spec in model_specs:
        oof_pred = np.zeros(len(data), dtype=int)
        oof_prob = np.zeros(len(data), dtype=float)

        for fold_idx, (train_idx, valid_idx) in enumerate(folds):
            train_groups = set(groups[train_idx].tolist())
            valid_groups = set(groups[valid_idx].tolist())
            if train_groups.intersection(valid_groups):
                raise ValueError(f"Group overlap detected in fold {fold_idx} for {spec.name}.")

            model = build_modeling_pipeline(spec.estimator_factory())
            x_train = X.iloc[train_idx]
            y_train = y[train_idx]
            x_valid = X.iloc[valid_idx]
            y_valid = y[valid_idx]

            start = time.time()
            model.fit(x_train, y_train)
            fit_seconds = time.time() - start

            valid_pred = model.predict(x_valid).astype(int)
            valid_prob = model.predict_proba(x_valid)[:, 1]
            oof_pred[valid_idx] = valid_pred
            oof_prob[valid_idx] = valid_prob

            train_pred = model.predict(x_train).astype(int)
            train_prob = model.predict_proba(x_train)[:, 1]

            valid_metrics = _metric_row(y_valid, valid_pred, valid_prob)
            train_metrics = _metric_row(y_train, train_pred, train_prob)
            valid_metrics.update({
                "model": spec.name,
                "fold": fold_idx,
                "fit_seconds": round(fit_seconds, 6),
            })
            train_metrics.update({
                "model": spec.name,
                "fold": fold_idx,
                "fit_seconds": round(fit_seconds, 6),
            })
            per_fold_validation_rows.append(valid_metrics)
            per_fold_training_rows.append(train_metrics)

        if np.isnan(oof_prob).any():
            raise ValueError(f"OOF probabilities contain NaN for {spec.name}.")

        oof_frame = pd.DataFrame({
            "row_index": np.arange(len(data), dtype=int),
            "fold": fold_assignment["fold"].to_numpy(dtype=int),
            "group_id": groups.astype(int),
            "y_true": y,
            "y_pred": oof_pred,
            "y_prob_canceled": oof_prob,
            "model": spec.name,
        })
        all_oof_predictions.append(oof_frame)
        oof_frame.to_csv(
            OUTPUT_DIR / f"oof_predictions_{spec.name}.csv",
            index=False,
        )

    validation_df = pd.DataFrame(per_fold_validation_rows)
    training_df = pd.DataFrame(per_fold_training_rows)
    validation_df.to_csv(OUTPUT_DIR / "cv_per_fold_validation_metrics.csv", index=False)
    training_df.to_csv(OUTPUT_DIR / "cv_per_fold_training_metrics.csv", index=False)

    metric_columns = [
        "accuracy",
        "precision_canceled",
        "recall_canceled",
        "f1_canceled",
        "roc_auc",
        "average_precision",
        "fit_seconds",
    ]
    summary_rows = []
    for model_name, part in validation_df.groupby("model", sort=False):
        row = {"model": model_name}
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "cv_validation_summary_mean_std.csv", index=False)

    oof_pooled_rows: list[dict[str, object]] = []
    roc_curves = []
    pr_curves = []
    comparison_rows = []
    for oof_part in all_oof_predictions:
        model_name = oof_part["model"].iloc[0]
        y_true_part = oof_part["y_true"].to_numpy(dtype=int)
        y_pred_part = oof_part["y_pred"].to_numpy(dtype=int)
        y_prob_part = oof_part["y_prob_canceled"].to_numpy(dtype=float)
        pooled = _metric_row(y_true_part, y_pred_part, y_prob_part)
        pooled["model"] = model_name
        oof_pooled_rows.append(pooled)

        fpr, tpr, _ = roc_curve(y_true_part, y_prob_part, pos_label=1)
        precision, recall, _ = precision_recall_curve(y_true_part, y_prob_part, pos_label=1)
        roc_curves.append({"x": fpr, "y": tpr, "label": model_name})
        pr_curves.append({"x": recall, "y": precision, "label": model_name})

        comparison_rows.append({
            "model": model_name,
            "f1_canceled_fold_mean": float(
                validation_df.loc[validation_df["model"] == model_name, "f1_canceled"].mean()
            ),
            "f1_canceled_fold_std": float(
                validation_df.loc[validation_df["model"] == model_name, "f1_canceled"].std(ddof=1)
            ),
            "oof_f1_canceled": float(pooled["f1_canceled"]),
            "oof_accuracy": float(pooled["accuracy"]),
            "oof_roc_auc": float(pooled["roc_auc"]),
            "oof_average_precision": float(pooled["average_precision"]),
        })

        _plot_confusion_matrix(
            y_true_part,
            y_pred_part,
            OUTPUT_DIR / f"oof_confusion_matrix_{model_name}.png",
            f"OOF confusion matrix ({model_name})",
        )

    pooled_df = pd.DataFrame(oof_pooled_rows)
    pooled_df.to_csv(OUTPUT_DIR / "cv_oof_pooled_metrics.csv", index=False)

    _plot_curve_comparison(
        roc_curves,
        x_label="False positive rate",
        y_label="True positive rate",
        title="OOF ROC curve comparison",
        chance_line=True,
        path=OUTPUT_DIR / "oof_roc_curve_comparison.png",
    )
    _plot_curve_comparison(
        pr_curves,
        x_label="Recall (cancellation class)",
        y_label="Precision (cancellation class)",
        title="OOF precision-recall curve comparison",
        chance_line=False,
        path=OUTPUT_DIR / "oof_precision_recall_curve_comparison.png",
    )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUTPUT_DIR / "rf_vs_dummy_comparison.csv", index=False)

    rf_pipeline = build_modeling_pipeline(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    )
    rf_pipeline.fit(X, y)
    joblib.dump(rf_pipeline, MODEL_ARTIFACT)
    reloaded = joblib.load(MODEL_ARTIFACT)

    sample_idx = np.arange(min(25, len(X)))
    original_pred = rf_pipeline.predict(X.iloc[sample_idx]).astype(int)
    reloaded_pred = reloaded.predict(X.iloc[sample_idx]).astype(int)
    original_prob = rf_pipeline.predict_proba(X.iloc[sample_idx])[:, 1]
    reloaded_prob = reloaded.predict_proba(X.iloc[sample_idx])[:, 1]
    if not np.array_equal(original_pred, reloaded_pred):
        raise ValueError("Reloaded artifact predictions differ from in-memory model.")
    if not np.allclose(original_prob, reloaded_prob, atol=0.0, rtol=0.0):
        raise ValueError("Reloaded artifact probabilities differ from in-memory model.")

    artifact_check = pd.DataFrame({
        "row_index": sample_idx,
        "prediction": original_pred,
        "probability_canceled": original_prob,
    })
    artifact_check.to_csv(OUTPUT_DIR / "artifact_reload_check_sample.csv", index=False)

    source_state_end = _collect_source_state_snapshot(source_files)
    run_completed_utc = _utc_now_iso()
    source_state_diff = _compare_source_snapshots(source_state_start, source_state_end)

    metadata = {
        "run_timestamps_utc": {
            "started": run_started_utc,
            "completed": run_completed_utc,
        },
        "input_data": {
            "path": str(CLASSIFICATION_OUTPUT),
            "sha256": _sha256(CLASSIFICATION_OUTPUT),
            "sha256_bytes": _sha256(CLASSIFICATION_OUTPUT),
            "sha256_utf8_lf_normalized": _portable_text_sha256_utf8_lf(CLASSIFICATION_OUTPUT),
            "text_hash_normalization": "UTF-8 text with LF line endings",
            "rows": int(len(data)),
            "columns": expected_columns,
        },
        "source_state": {
            "start": source_state_start,
            "end": source_state_end,
            "diff": source_state_diff,
        },
        "random_seed": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "grouping_rule": "Identical cleaned selected predictor rows, target excluded.",
        "model_config": {
            "random_forest": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ).get_params(deep=True),
            "dummy_prior": DummyClassifier(strategy="prior").get_params(deep=True),
        },
        "package_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "source_commit": source_state_start["git_head"]["sha"],
        "notes": {
            "primary_metric": "f1_canceled",
            "probability_metrics_use_predicted_probability_of_class_1": True,
            "zero_division_policy": 0,
            "provenance_interpretation": (
                "Start and end source snapshots distinguish relevant source-code "
                "changes from ordinary generated-output changes during a run. "
                "Git metadata may be unavailable when running from a downloaded ZIP."
            ),
        },
    }
    with (OUTPUT_DIR / "experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved fold assignments: {OUTPUT_DIR / 'cv_fold_assignments.csv'}")
    print(f"Saved per-fold validation metrics: {OUTPUT_DIR / 'cv_per_fold_validation_metrics.csv'}")
    print(f"Saved per-fold training metrics: {OUTPUT_DIR / 'cv_per_fold_training_metrics.csv'}")
    print(f"Saved pooled OOF metrics: {OUTPUT_DIR / 'cv_oof_pooled_metrics.csv'}")
    print(f"Saved RF vs Dummy comparison: {OUTPUT_DIR / 'rf_vs_dummy_comparison.csv'}")
    print(f"Saved curves and confusion matrices in: {OUTPUT_DIR}")
    print(f"Saved full-training pipeline artifact: {MODEL_ARTIFACT}")
    print(f"Saved metadata: {OUTPUT_DIR / 'experiment_metadata.json'}")


if __name__ == "__main__":
    main()
