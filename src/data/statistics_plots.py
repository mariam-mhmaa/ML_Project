"""H2 EDA: summary statistics, distributions, skewness and outliers.

Run from the project folder:
    python src/data/statistics_plots.py

Figures are saved to outputs/figures and tables to outputs/tables.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Locate the project folder from this script's location.
project_folder = Path(__file__).resolve().parents[2]
OUTDIR = project_folder / "outputs"
(OUTDIR / "figures").mkdir(parents=True, exist_ok=True)
(OUTDIR / "tables").mkdir(parents=True, exist_ok=True)

# ---- variable groups (H2 column names) ----
BINARY_COLS = ["IsCanceled", "IsRepeatedGuest"]
NUMERIC_COLS = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "PreviousCancellations",
    "PreviousBookingsNotCanceled", "BookingChanges", "DaysInWaitingList",
    "ADR", "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
]
CATEGORICAL_COLS = [
    "ArrivalDateMonth", "Meal", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "AssignedRoomType", "DepositType", "CustomerType",
    "ReservationStatus",
]
ID_COLS = ["Agent", "Company"]
MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
MONTH_ABBR = {m: m[:3] for m in MONTH_ORDER}

# ---- colours ----
ACCENT, ICE, NEG = "#3B82F6", "#DBEAFE", "#EF4444"
G900, G700, G500, G400, G300 = "#111827", "#374151", "#6B7280", "#9CA3AF", "#D1D5DB"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.unicode_minus": False,
    "axes.edgecolor": "#E5E7EB", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlepad": 12,
    "legend.frameon": False, "legend.fontsize": 9,
    "savefig.dpi": 200, "savefig.facecolor": "white", "savefig.pad_inches": 0.3,
})


# ---- helpers ----
def _grid(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).grid(True, alpha=0.12, color=G300)
    ax.set_axisbelow(True)


def save_fig(fig, name):
    """Save to outputs/figures."""
    path = OUTDIR / "figures" / name
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"figure -> {path}")


def save_table(df, name, index=False):
    path = OUTDIR / "tables" / name
    df.to_csv(path, index=index)
    print(f"table  -> {path}")


def _label_vbars(ax, values, fmt="{:,}", color=G700, fontsize=9):
    ymax = max(values)
    for i, v in enumerate(values):
        ax.text(i, v + ymax * 0.02, fmt.format(v), ha="center", va="bottom",
                fontsize=fontsize, color=color)
    ax.set_ylim(0, ymax * 1.18)


def _label_vbars_log(ax, values, fmt="{:,}", color=G700, fontsize=9):
    for i, v in enumerate(values):
        ax.text(i, v * 1.35, fmt.format(v), ha="center", va="bottom",
                fontsize=fontsize, color=color)
    ax.set_ylim(0.8, max(values) * 80)


def _ecdf(values, max_points=2000):
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    idx = (np.unique(np.linspace(0, n - 1, max_points).astype(int))
           if n > max_points else np.arange(n))
    return v[idx], (idx + 1) / n


def summary_statistics(df):
    num = df[NUMERIC_COLS]
    desc = num.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    desc.insert(0, "n_missing", num.isna().sum())
    desc["iqr"] = desc["75%"] - desc["25%"]
    desc = desc.rename(columns={"count": "n", "1%": "p01", "25%": "p25",
                                "50%": "median", "75%": "p75", "99%": "p99"})
    desc = desc[["n", "n_missing", "mean", "std", "min", "p01", "p25",
                 "median", "p75", "p99", "max", "iqr"]]
    desc = desc.rename_axis("variable").reset_index().round(3)
    save_table(desc, "05_numeric_summary.csv")

    cat_rows = []
    for col in CATEGORICAL_COLS + ID_COLS + BINARY_COLS:
        vc = df[col].value_counts(dropna=True)
        cat_rows.append({
            "variable": col,
            "n_unique": int(df[col].nunique(dropna=True)),
            "n_missing": int(df[col].isna().sum()),
            "mode": vc.index[0] if len(vc) else None,
            "mode_count": int(vc.iloc[0]) if len(vc) else 0,
            "mode_pct": round(100 * vc.iloc[0] / len(df), 2) if len(vc) else 0,
        })
    save_table(pd.DataFrame(cat_rows), "06_categorical_summary.csv")

    return {"adr": {k: round(float(v), 2) for k, v in df["ADR"].describe().items()},
            "cancel_rate": round(float(df["IsCanceled"].mean()), 4)}


def numeric_distributions(df):
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), constrained_layout=True)
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        data = df[col].dropna()
        ax.hist(data, bins=40, color=ACCENT, alpha=0.85,
                edgecolor="white", linewidth=0.3, zorder=3)
        if float((data == 0).mean()) > 0.8 or abs(float(data.skew())) > 5:
            ax.set_yscale("log")
        med = data.median()
        ax.axvline(med, color=G900, linewidth=1.2, linestyle="--", zorder=4)
        if col == "ArrivalDateYear":
            ax.set_xticks(sorted(data.unique()))
        ax.set_title(f"{col}  (median {med:g})", loc="left", fontsize=12)
        _grid(ax)
    for ax in axes.flat[len(NUMERIC_COLS):]:
        ax.axis("off")
    fig.suptitle("Numeric variable distributions — dashed line marks the median "
                 "(log y-axis where the variable is zero-dominated)",
                 fontsize=16, fontweight="bold")
    save_fig(fig, "fig03_numeric_distributions.png")
    return {"n_numeric_plotted": len(NUMERIC_COLS)}


def categorical_distributions(df):
    panels = [
        ("ArrivalDateMonth", "v"), ("Meal", "v"), ("MarketSegment", "h"),
        ("DistributionChannel", "h"), ("DepositType", "h"), ("CustomerType", "h"),
        ("ReservedRoomType", "v"), ("AssignedRoomType", "v"),
        ("ReservationStatus", "h"), ("IsCanceled", "v"), ("IsRepeatedGuest", "v"),
    ]
    n = len(df)
    fig, axes = plt.subplots(3, 4, figsize=(20, 16), constrained_layout=True)
    for ax, (col, orient) in zip(axes.flat, panels):
        if col == "ArrivalDateMonth":
            vc = df[col].value_counts().reindex(MONTH_ORDER).fillna(0)
            bars = ax.bar([MONTH_ABBR[m] for m in vc.index], vc.values,
                          color=G300, width=0.7, zorder=3, edgecolor="white")
            bars[int(np.argmax(vc.values))].set_color(ACCENT)
            _label_vbars(ax, [int(v) for v in vc.values], fontsize=8)
            ax.set_title(f"{col}  (chronological)", loc="left", fontsize=12)
        elif orient == "v":
            if col in BINARY_COLS:
                names = {"IsCanceled": ("0 — not canceled", "1 — canceled"),
                         "IsRepeatedGuest": ("0 — new guest", "1 — repeated")}[col]
                vc = df[col].value_counts().reindex([0, 1]).fillna(0)
                labels, colors = list(names), [G300, ACCENT]
            else:
                vc = df[col].value_counts()
                labels = [str(i) for i in vc.index]
                colors = [G300] * len(vc)
                colors[0] = ACCENT
            ax.bar(labels, vc.values, color=colors, width=0.6, zorder=3,
                   edgecolor="white")
            vals = [int(v) for v in vc.values]
            ymax = max(vals)
            for i, v in enumerate(vals):
                ax.text(i, v + ymax * 0.02, f"{v:,}\n({v / n:.0%})", ha="center",
                        va="bottom", fontsize=9, color=G700)
            ax.set_ylim(0, ymax * 1.30)
            ax.set_title(col, loc="left", fontsize=12)
        else:
            vc = df[col].value_counts().iloc[::-1]
            bars = ax.barh([str(i) for i in vc.index], vc.values, color=G300,
                           height=0.6, zorder=3, edgecolor="white")
            bars[-1].set_color(ACCENT)
            mx = vc.values.max()
            for b, v in zip(bars, vc.values):
                ax.text(v + mx * 0.02, b.get_y() + b.get_height() / 2,
                        f"{v:,} ({v / n:.0%})", va="center", fontsize=9, color=G700)
            ax.set_xlim(0, mx * 1.35)
            ax.set_title(col, loc="left", fontsize=12)
        _grid(ax)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("Categorical variable distributions — accent marks the modal category",
                 fontsize=16, fontweight="bold")
    save_fig(fig, "fig04_categorical_distributions.png")

    return {"modal_categories": {
        c: (str(df[c].value_counts().index[0]),
            round(float(100 * df[c].value_counts().iloc[0] / n), 1))
        for c in ["MarketSegment", "CustomerType", "DepositType", "Meal"]}}


def skewness_analysis(df):
    rows = []
    for col in NUMERIC_COLS:
        data = df[col].dropna()
        rows.append({
            "variable": col,
            "mean": round(float(data.mean()), 3),
            "median": round(float(data.median()), 3),
            "std": round(float(data.std()), 3),
            "min": round(float(data.min()), 3),
            "max": round(float(data.max()), 3),
            "skewness": round(float(data.skew()), 3),
            "excess_kurtosis": round(float(data.kurt()), 3),
            "zero_fraction_pct": round(100 * float((data == 0).mean()), 2),
        })
    table = pd.DataFrame(rows)

    def classify(s):
        if abs(s) > 1:
            return "highly skewed"
        if abs(s) > 0.5:
            return "moderately skewed"
        return "approximately symmetric"

    table["classification"] = table["skewness"].apply(classify)
    table = (table.sort_values("skewness", key=lambda s: s.abs(), ascending=False)
                  .reset_index(drop=True))
    save_table(table, "09_skewness_kurtosis.csv")
    n_high = int((table["classification"] == "highly skewed").sum())

    # ---- ranking figure ----
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    t = table.iloc[::-1]
    colors = [ACCENT if abs(s) > 1 else G300 for s in t["skewness"]]
    bars = ax.barh([str(v) for v in t["variable"]], t["skewness"], color=colors,
                   height=0.62, zorder=3, edgecolor="white")
    for b, v in zip(bars, t["skewness"]):
        side = dict(ha="left") if v >= 0 else dict(ha="right")
        x = v * 1.25 + (0.05 if v >= 0 else -0.05)
        ax.text(x, b.get_y() + b.get_height() / 2, f"{v:.1f}",
                va="center", fontsize=9, color=G900, **side)
    ax.axvline(1, color=G500, linewidth=1, linestyle="--", zorder=2)
    ax.axvline(-1, color=G500, linewidth=1, linestyle="--", zorder=2)
    ax.set_xscale("symlog", linthresh=0.5)
    lim = max(abs(table["skewness"]).max() * 2, 10)
    ax.set_xlim(-6, lim)
    ax.set_xlabel("skewness coefficient (symlog scale) — dashed lines: |skew| = 1")
    ax.set_title(f"{n_high} of {len(table)} numeric variables are highly skewed "
                 f"(|skew| > 1)", loc="left")
    _grid(ax, axis="x")
    save_fig(fig, "fig10_skewness_ranking.png")

    # ---- top-4 skewed histograms ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, (_, row) in zip(axes.flat, table.head(4).iterrows()):
        col = row["variable"]
        ax.hist(df[col].dropna(), bins=40, color=ACCENT, alpha=0.85,
                edgecolor="white", linewidth=0.3, zorder=3)
        ax.set_yscale("log")
        ax.axvline(row["mean"], color=G900, linewidth=1.4, zorder=4)
        ax.axvline(row["median"], color="#B45309", linewidth=1.4, linestyle="--", zorder=4)
        ax.legend([f"mean {row['mean']:g}", f"median {row['median']:g}"],
                  loc="upper right")
        ax.set_title(f"{col} — skew {row['skewness']:.1f}, "
                     f"{row['zero_fraction_pct']:.1f}% zeros", loc="left", fontsize=12)
        _grid(ax)
    fig.suptitle("The four most skewed variables — zero-inflation and long tails dominate",
                 fontsize=15, fontweight="bold")
    save_fig(fig, "fig11_skewed_variables.png")

    adr = df["ADR"].dropna()
    adr_trim = adr[adr <= 400]
    n_above = int((adr > 400).sum())
    return {
        "n_highly_skewed": n_high,
        "n_moderately_skewed": int((table["classification"] == "moderately skewed").sum()),
        "n_symmetric": int((table["classification"] == "approximately symmetric").sum()),
        "most_skewed": table.head(6)[["variable", "skewness", "zero_fraction_pct"]]
                            .to_dict("records"),
        "adr_skew_full": round(float(adr.skew()), 2),
        "adr_skew_excluding_above_400": round(float(adr_trim.skew()), 2),
        "n_adr_above_400": n_above,
    }


RECOMMENDED_HANDLING = {
    "ADR": "Cap/winsorise at a documented percentile or use a robust loss; decide explicitly on zero-ADR rows.",
    "LeadTime": "Keep; log1p transform for linear / distance-based models.",
    "Adults": "Flag zero-guest rows; impute or exclude with a documented rule.",
    "Children": "Impute missing with 0 + indicator; keep range as-is.",
    "Babies": "Near-constant; consider binarising (baby yes/no).",
    "StaysInWeekendNights": "Keep; tail is genuine long-stay behaviour.",
    "StaysInWeekNights": "Keep; consider total-nights feature + log1p.",
    "PreviousCancellations": "Keep; binarise or cap at a small value for linear models.",
    "PreviousBookingsNotCanceled": "Keep; combine with IsRepeatedGuest.",
    "BookingChanges": "Keep; note it is post-creation information (leakage audit).",
    "DaysInWaitingList": "Keep; log1p or binarise (waitlisted yes/no).",
    "RequiredCarParkingSpaces": "Keep; binarise (parking requested).",
    "TotalOfSpecialRequests": "Keep as-is; no treatment needed.",
}


def outlier_analysis(df):
    iqr_rows, z_rows = [], []
    for col in NUMERIC_COLS:
        data = df[col].dropna()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_lo, n_hi = int((data < lo).sum()), int((data > hi).sum())
        iqr_rows.append({
            "variable": col, "q1": round(q1, 2), "q3": round(q3, 2),
            "iqr": round(iqr, 2), "lower_fence": round(lo, 2),
            "upper_fence": round(hi, 2), "n_below": n_lo, "n_above": n_hi,
            "n_outliers": n_lo + n_hi,
            "pct_outliers": round(100 * (n_lo + n_hi) / len(data), 2),
            "iqr_degenerate": bool(iqr == 0),
        })
        z = (data - data.mean()) / data.std()
        z_rows.append({
            "variable": col, "mean": round(float(data.mean()), 3),
            "std": round(float(data.std()), 3),
            "n_abs_z_gt_3": int((z.abs() > 3).sum()),
            "pct_abs_z_gt_3": round(100 * float((z.abs() > 3).mean()), 3),
        })
    iqr_tab, z_tab = pd.DataFrame(iqr_rows), pd.DataFrame(z_rows)
    save_table(iqr_tab, "10_outliers_iqr.csv")
    save_table(z_tab, "11_outliers_zscore.csv")

    # ---- boxplot grid ----
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), constrained_layout=True)
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        data = df[col].dropna()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.45,
                   boxprops=dict(facecolor=ICE, edgecolor=ACCENT, linewidth=1.1),
                   whiskerprops=dict(color=G500, linewidth=1),
                   capprops=dict(color=G500, linewidth=1),
                   medianprops=dict(color=G900, linewidth=1.4),
                   flierprops=dict(marker="o", markersize=2, markerfacecolor=G400,
                                   markeredgecolor="none", alpha=0.35))
        if (iqr == 0 and data.max() > 5) or (q3 > 0 and data.max() > 8 * q3):
            ax.set_xscale("symlog", linthresh=1)
        row = iqr_tab.loc[iqr_tab["variable"] == col].iloc[0]
        deg = " [IQR=0]" if row["iqr_degenerate"] else ""
        ax.set_title(f"{col}: {int(row['n_outliers']):,} outliers "
                     f"({row['pct_outliers']:.1f}%){deg}", loc="left", fontsize=10)
        ax.set_yticks([])
        _grid(ax, axis="x")
    for ax in axes.flat[len(NUMERIC_COLS):]:
        ax.axis("off")
    fig.suptitle("Outlier screening per numeric variable (Tukey 1.5xIQR fences; "
                 "symlog axis where the tail is extreme; [IQR=0] = middle 50% "
                 "all zero, so every nonzero value is flagged)",
                 fontsize=15, fontweight="bold")
    save_fig(fig, "fig12_outlier_boxplots.png")

    # ---- deep dives ----
    nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    ax = axes[0, 0]
    adr = df["ADR"].dropna()
    x, y = _ecdf(adr)
    ax.plot(x, y, color=ACCENT, linewidth=2.2, zorder=3)
    ax.set_xscale("symlog", linthresh=10)
    q1, q3 = adr.quantile([0.25, 0.75])
    fence = q3 + 1.5 * (q3 - q1)
    n_above_fence = int((adr > fence).sum())
    ax.axvline(fence, color=G500, linewidth=1.2, linestyle="--")
    ax.annotate(f"upper fence €{fence:.0f}", xy=(fence, 0.35),
                xytext=(fence * 8, 0.3), fontsize=10, color=G900,
                arrowprops=dict(arrowstyle="->", color=G500, lw=1))
    top3 = adr.nlargest(3)
    top3_txt = ", ".join("€{:,.0f}".format(v) for v in top3)
    ax.annotate(f"top rates: {top3_txt}", xy=(float(top3.min()), 1.0),
                xytext=(30, 0.72), fontsize=10, color=G900,
                arrowprops=dict(arrowstyle="->", color=G500, lw=1))
    ax.set_title(f"ADR — {n_above_fence:,} of {len(adr):,} bookings above the fence",
                 loc="left")
    ax.set_xlabel("ADR (EUR) — symlog")
    _grid(ax)

    ax = axes[0, 1]
    lt = df["LeadTime"].dropna()
    x, y = _ecdf(lt)
    ax.plot(x, y, color=ACCENT, linewidth=2.2, zorder=3)
    for p in (0.5, 0.95, 0.99):
        v = lt.quantile(p)
        ax.axvline(v, color=G400, linewidth=1, linestyle="--", zorder=2)
        ax.text(v, 0.04 + (0.16 if p == 0.5 else 0), f"P{int(p * 100)} = {v:.0f}",
                fontsize=9, color=G700, ha="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    ax.set_title(f"LeadTime — 95% of bookings are within {lt.quantile(0.95):.0f} "
                 f"days, tail reaches {lt.max():.0f}", loc="left")
    ax.set_xlabel("LeadTime (days)")
    _grid(ax)

    ax = axes[1, 0]
    vc = df["Adults"].value_counts().sort_index()
    colors = [NEG if k == 0 else (ACCENT if k == vc.idxmax() else G300) for k in vc.index]
    ax.bar([str(k) for k in vc.index], vc.values, color=colors, width=0.6,
           zorder=3, edgecolor="white")
    _label_vbars_log(ax, [int(v) for v in vc.values])
    ax.set_yscale("log")
    n_zero = int((df["Adults"] == 0).sum())
    ax.set_title(f"Adults — {n_zero} bookings record no adult (highlighted red)",
                 loc="left", fontsize=12)
    _grid(ax)

    ax = axes[1, 1]
    vc = nights.value_counts()
    head = vc.loc[[v for v in range(0, 15) if v in vc.index]]
    tail_n = int(vc.loc[[v for v in vc.index if v >= 15]].sum())
    labels = [str(v) for v in head.index] + ["15+"]
    vals = [int(v) for v in head.values] + [tail_n]
    colors = ([NEG if k == 0 else (ACCENT if k == head.idxmax() else G300)
               for k in head.index] + [G300])
    ax.bar(labels, vals, color=colors, width=0.7, zorder=3, edgecolor="white")
    vals_plot = [max(v, 1) for v in vals]          # keep zero-height bars off log axis
    _label_vbars_log(ax, vals_plot, fontsize=8)
    ax.set_yscale("log")
    n_zero_nights = int((nights == 0).sum())
    ax.set_title(f"Total nights — {n_zero_nights} bookings have 0 nights, "
                 f"max {int(nights.max())}", loc="left", fontsize=12)
    ax.set_xlabel("weekend + week nights")
    _grid(ax)

    fig.suptitle("Outlier deep dives — extremes are few, genuine, and must be "
                 "handled by rule, not silently", fontsize=15, fontweight="bold")
    save_fig(fig, "fig13_outlier_deep_dives.png")

    # ---- deep-dive table (interpretation computed from the data) ----
    rows = []
    for col in NUMERIC_COLS:
        r = iqr_tab.loc[iqr_tab["variable"] == col].iloc[0]
        d = df[col].dropna()
        interp = (f"{int(r['n_outliers']):,} flagged; median {d.median():g}, "
                  f"max {d.max():g}, {100 * (d == 0).mean():.1f}% zeros, "
                  f"{int(df[col].isna().sum())} missing")
        rows.append({
            "variable": col, "n_outliers": int(r["n_outliers"]),
            "pct_outliers": r["pct_outliers"],
            "lower_fence": r["lower_fence"], "upper_fence": r["upper_fence"],
            "interpretation": interp,
            "recommended_handling": RECOMMENDED_HANDLING.get(col, "Keep; no specific action."),
        })
    save_table(pd.DataFrame(rows), "12_outlier_deep_dive.csv")

    guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    return {
        "n_adr_above_fence": n_above_fence,
        "adr_top3": [round(v, 2) for v in df["ADR"].nlargest(3)],
        "n_zero_adults": n_zero,
        "n_zero_guests": int((guests == 0).sum()),
        "n_zero_nights": n_zero_nights,
        "n_leadtime_over_365": int((df["LeadTime"] > 365).sum()),
        "n_babies_3_or_more": int((df["Babies"] >= 3).sum()),
    }


# Text values are padded with spaces (e.g. "       NULL", "HB       "),
# so strip them first, otherwise "NULL" isn't recognised as missing.
df = pd.read_csv(project_folder / "data" / "raw" / "H2.csv")
text_cols = df.select_dtypes(exclude="number").columns
df[text_cols] = df[text_cols].apply(lambda c: c.str.strip())
df = df.replace("NULL", np.nan)
print(df.shape)

results = {}
results["summary_statistics"] = summary_statistics(df)
results["numeric_distributions"] = numeric_distributions(df)
results["categorical_distributions"] = categorical_distributions(df)
results["skewness_analysis"] = skewness_analysis(df)
results["outlier_analysis"] = outlier_analysis(df)

for name, values in results.items():
    print(f"\n{name}: {values}")
