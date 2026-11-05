"""Write regression-specific Milestone 1 diagnostics using train.csv only."""

from pathlib import Path
import os
import sys
import tempfile

# Keep Matplotlib's machine-specific cache out of the user's profile/repository.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cse382-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.regression.preprocessing import (
    CATEGORICAL_COLUMNS, EXCLUDED_PREDICTORS, MONTHS, NUMERIC_COLUMNS,
    PREDICTOR_COLUMNS, TARGET_COLUMN, load_training_frame, prepare_training_frame,
)

OUTPUT = PROJECT_ROOT / "outputs/regression"
FEATURE_REASONS = {
    "LeadTime": "Assume booking entry date and intended arrival date are known",
    "ArrivalDateYear": "Planned arrival year is known; random split does not test future years",
    "ArrivalDateMonth": "Planned arrival month captures seasonality",
    "ArrivalDateWeekNumber": "Planned arrival week captures seasonality; overlaps month",
    "ArrivalDateDayOfMonth": "Planned arrival day is known",
    "StaysInWeekendNights": "Assume recorded nights approximate originally booked weekend nights",
    "StaysInWeekNights": "Assume recorded nights approximate originally booked weekday nights",
    "Adults": "Assume recorded party size approximates booking-time party size",
    "Children": "Assume booking-time party size; missing values are not assumed to be zero",
    "Babies": "Assume recorded party size approximates booking-time party size",
    "Meal": "Assume recorded meal package approximates package chosen at booking",
    "Country": "Assume supplied at booking; paper warns true nationality may be known only at check-in",
    "MarketSegment": "Assume segment is identifiable from booking origin at creation",
    "DistributionChannel": "Assume channel is known when booking is created",
    "IsRepeatedGuest": "Assume guest profile predates booking and can be matched at creation",
    "PreviousCancellations": "Assume only outcomes already resolved at current booking creation",
    "PreviousBookingsNotCanceled": "Assume historical outcomes are known at current booking creation",
    "ReservedRoomType": "Use originally reserved room category, not ultimate room assignment",
    "Agent": "Agency identity is known from booking origin; nominal ID, never an ordered number",
    "Company": "Assume company payer is known at creation; nominal ID",
    "CustomerType": "Assume contract/group/transient arrangement is known at creation",
    "RequiredCarParkingSpaces": "Assume recorded parking requests were present at booking creation",
    "TotalOfSpecialRequests": "Assume recorded requests were present at booking creation",
}


def main():
    raw = load_training_frame()
    frame, exclusions = prepare_training_frame(raw)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    y = frame[TARGET_COLUMN]
    groups = frame.groupby(PREDICTOR_COLUMNS, dropna=False, sort=False)[TARGET_COLUMN].agg(
        group_rows="size", distinct_adr="nunique", minimum_adr="min", maximum_adr="max",
    ).reset_index()
    repeated = groups.loc[groups.group_rows.gt(1)]
    repeated.to_csv(OUTPUT / "repeated_selected_input_groups.csv", index=False)
    summary = {
        "shared_train_rows": len(raw), "excluded_zero_night_rows": len(exclusions),
        "regression_rows": len(frame), "predictors": len(PREDICTOR_COLUMNS),
        "zero_adr_rows_retained": int(y.eq(0).sum()),
        "missing_children": int(frame.Children.isna().sum()),
        "missing_country": int(frame.Country.isna().sum()),
        "selected_input_groups": len(groups), "repeated_input_groups": len(repeated),
        "repeated_input_rows": int(repeated.group_rows.sum()),
        "groups_with_multiple_adr_values": int(repeated.distinct_adr.gt(1).sum()),
    }
    pd.DataFrame([summary]).to_csv(OUTPUT / "audit_summary.csv", index=False)
    y.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).rename_axis("statistic").to_csv(
        OUTPUT / "target_summary.csv", header=["ADR"],
    )
    inventory = []
    for col in PREDICTOR_COLUMNS:
        inventory.append({
            "column": col, "role": "predictor", "dtype": str(frame[col].dtype),
            "missing_rows": int(frame[col].isna().sum()),
            "transformation": "median imputation + missing indicator where observed + standard scaling"
            if col in NUMERIC_COLUMNS else "Unknown imputation + one-hot encoding; IDs NULL -> NotApplicable",
            "prediction_time_justification": FEATURE_REASONS[col],
        })
    inventory.append({"column": TARGET_COLUMN, "role": "target", "dtype": "float64",
                      "missing_rows": 0, "transformation": "unchanged; no imputation, clipping, or log transform",
                      "prediction_time_justification": "Predict recorded ADR, which may differ from initial quoted rate"})
    inventory.extend({"column": col, "role": "excluded", "dtype": "not used",
                      "missing_rows": "", "transformation": "drop",
                      "prediction_time_justification": reason}
                     for col, reason in EXCLUDED_PREDICTORS.items())
    pd.DataFrame(inventory).to_csv(OUTPUT / "column_inventory.csv", index=False)
    whitespace = [{"column": col, "trimmed_values": int(raw[col].ne(raw[col].str.strip()).sum())}
                  for col in PREDICTOR_COLUMNS]
    pd.DataFrame(whitespace).to_csv(OUTPUT / "whitespace_cleanup_counts.csv", index=False)
    cardinality = [{"column": col, "distinct_nonmissing_categories": frame[col].nunique(),
                    "missing_rows": int(frame[col].isna().sum())} for col in CATEGORICAL_COLUMNS]
    pd.DataFrame(cardinality).to_csv(OUTPUT / "categorical_cardinality.csv", index=False)
    numeric = frame[NUMERIC_COLUMNS + [TARGET_COLUMN]]
    stats = numeric.describe().T
    stats["missing_rows"] = numeric.isna().sum()
    stats["skewness"] = numeric.skew()
    stats["excess_kurtosis"] = numeric.kurt()
    stats["iqr_lower_fence"] = stats["25%"] - 1.5 * (stats["75%"] - stats["25%"])
    stats["iqr_upper_fence"] = stats["75%"] + 1.5 * (stats["75%"] - stats["25%"])
    stats["rows_outside_iqr_fences"] = [int((numeric[col].lt(stats.loc[col, "iqr_lower_fence"]) |
                                           numeric[col].gt(stats.loc[col, "iqr_upper_fence"])).sum())
                                        for col in stats.index]
    stats.rename_axis("column").to_csv(OUTPUT / "numeric_summary.csv")
    correlations = numeric.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values()
    correlations.rename("pearson_correlation_with_ADR").rename_axis("column").to_csv(
        OUTPUT / "numeric_target_correlations.csv")
    profiles = []
    for col in ["ArrivalDateMonth", "ReservedRoomType", "MarketSegment"]:
        part = frame.groupby(col)[TARGET_COLUMN].agg(rows="size", mean_adr="mean", median_adr="median").reset_index()
        part = part.rename(columns={col: "category"})
        part.insert(0, "feature", col)
        profiles.append(part)
    pd.concat(profiles).to_csv(OUTPUT / "adr_by_booking_attributes.csv", index=False)
    zero = frame.loc[y.eq(0)].groupby("MarketSegment").size().rename("zero_adr_positive_night_rows")
    zero.to_csv(OUTPUT / "zero_adr_by_market_segment.csv")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].hist(y, bins=60, color="#2563eb", edgecolor="white")
    axes[0].set(xlabel="ADR (EUR per night)", ylabel="Training bookings", title="Recorded ADR distribution")
    axes[1].boxplot(y, orientation="horizontal", widths=.45)
    axes[1].set(xlabel="ADR (EUR per night)", yticks=[], title="Extremes retained; no target clipping")
    fig.suptitle(f"Regression training population: {len(frame):,} positive-night bookings")
    fig.savefig(figures / "adr_distribution.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    ax.barh(correlations.index, correlations.values, color="#2563eb")
    ax.axvline(0, color="black", linewidth=.7)
    ax.set(xlim=(-1, 1), xlabel="Pearson correlation with ADR", title="Training-only numeric associations (not causal effects)")
    fig.savefig(figures / "numeric_target_correlations.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.boxplot([frame.loc[frame.ArrivalDateMonth.eq(month), TARGET_COLUMN] for month in MONTHS],
               tick_labels=[month[:3] for month in MONTHS], showfliers=False)
    ax.set(xlabel="Planned arrival month", ylabel="ADR (EUR per night)",
           title="Training ADR by month (boxplot fliers hidden for readability; rows retained)")
    fig.savefig(figures / "adr_by_arrival_month.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Saved training-only audit tables and figures in {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
