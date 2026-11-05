# Member 5 - Clustering Milestone 1 Explanation

## Scope and purpose

This section implements the single initial clustering algorithm required for
Milestone 1. Its purpose is to discover interpretable booking-behaviour segments
among City Hotel records without supplying a supervised outcome. It is not the
final three-algorithm comparison: optimized K-Means, hierarchical clustering, a
third algorithm, and the required PCA-based two-dimensional visualization remain
Milestone 2 work.

The analysis uses only `data/preprocessed/train.csv` (63,329 records). The shared
15,833-row `test.csv` is not opened by the clustering script. Although clustering
does not have a supervised target, keeping the existing holdout untouched avoids
using it to shape feature or parameter decisions and preserves a consistent team
experimental boundary.

## Selected booking attributes

The baseline focuses on behaviour known at booking creation:

| Model feature | Source field(s) | Why it is useful |
| --- | --- | --- |
| `LeadTime` | `LeadTime` | Distinguishes advance planners from short-notice bookings. |
| `TotalNights` | Weekend + weekday nights | Summarizes intended stay length without giving two related fields double influence. |
| `TotalGuests` | Adults + children + babies | Describes party size in one interpretable value. |
| `IsRepeatedGuest` | Same source label | Separates returning-customer behaviour. |
| `PreviousCancellations` | Same source label | Describes historical, not current-booking, behaviour. |
| `PreviousBookingsNotCanceled` | Same source label | Describes prior successful booking history. |
| `RequiredCarParkingSpaces` | Same source label | Captures a requested service. |
| `TotalOfSpecialRequests` | Same source label | Captures the level of stated booking requirements. |

The model intentionally starts with a compact numeric representation. Categorical
fields such as `Meal`, `Country`, `MarketSegment`, and `CustomerType` are deferred
to later mixed-feature experiments, where their encoding and weighting can be
compared deliberately. This avoids allowing high-cardinality identifiers such as
`Agent` and `Company` to dominate an initial distance-based model.

## Leakage audit

The complete column-by-column decision table is generated as
`outputs/clustering/leakage_audit.csv`. The most important exclusions are:

- `IsCanceled`, `ReservationStatus`, and `ReservationStatusDate`: final outcomes.
- `AssignedRoomType`: the room ultimately assigned, which may be decided later.
- `BookingChanges`: cumulative changes through check-in or cancellation.
- `DaysInWaitingList`: completed waiting duration rather than information known
  when the booking first entered the system.
- `ADR`: a price/outcome field and the regression target; including it could make
  segments primarily price bands instead of booking-behaviour groups.
- `DepositType`: excluded until its timing and availability at the stated analysis
  point can be verified.

No outcome label is used for fitting, selecting parameters, naming clusters, or
calculating the internal evaluation metrics.

## Preprocessing and missing values

All model-dependent preprocessing is inside the sklearn pipeline:

1. The source fields are parsed as nonnegative numeric values.
2. `TotalNights` and `TotalGuests` are derived deterministically.
3. The four records with missing `Children` produce missing `TotalGuests`; the
   fitted pipeline replaces these with the training median. No row is removed.
4. `log1p` is applied to the long-tailed nonnegative counts `LeadTime`,
   `TotalNights`, `PreviousCancellations`, and `PreviousBookingsNotCanceled`.
5. Every model feature is standardized. This is essential because K-Means uses
   Euclidean distance and otherwise large-unit variables would dominate.

## Initial algorithm and parameters

The baseline is `KMeans` with:

- `n_clusters=4`;
- `init="k-means++"`;
- `n_init=10`;
- `max_iter=300` and `tol=0.0001` (sklearn defaults);
- `random_state=42`;
- Lloyd's algorithm.

Four clusters were chosen **a priori** as a compact, interpretable starting point,
not as a claim that four is optimal. This preserves the distinction between an
untuned Milestone 1 baseline and the systematic cluster-count optimization required
later. `n_init=10` reduces dependence on one unlucky centroid initialization, and
the fixed seed makes the result reproducible.

## Initial results

| Measure | Result | Interpretation |
| --- | ---: | --- |
| Training records assigned | 63,329 | K-Means was fitted on all shared training rows. |
| Silhouette Score | 0.2122 | Positive but modest separation; clusters overlap in the selected feature space. |
| Silhouette sample | 10,000 | Fixed random sample with seed 42; avoids the exact metric's quadratic all-pairs cost. |
| Davies-Bouldin Index | 1.3107 | Moderate within/between-cluster separation; lower values will be preferred in later comparisons. |
| Inertia | 290,877.7761 | Baseline within-cluster squared distance; meaningful mainly when compared with other K values on the same representation. |

These are internal descriptive metrics, not supervised accuracy and not final test
results. The modest Silhouette Score is evidence against overstating that the data
contains four sharply separated natural populations.

## Cluster sizes and profiles

| Cluster | Descriptive profile | Records | Share | Key characteristics (means) |
| ---: | --- | ---: | ---: | --- |
| 0 | Special-request bookings | 22,323 | 35.25% | 1.29 special requests, 3.60 nights, 2.32 guests, 99-day lead time. |
| 1 | Low-request standard bookings | 38,151 | 60.24% | 0.09 special requests, 2.68 nights, 1.74 guests, 120-day lead time. |
| 2 | Parking-space request bookings | 1,348 | 2.13% | 1.01 parking spaces, 1.01 special requests, 2.27 guests. |
| 3 | Repeat-history guests | 1,507 | 2.38% | 98.9% repeated guests, 5.36 prior non-canceled bookings, 39-day lead time. |

Names are short descriptive summaries, not ground-truth customer classes. They are
assigned by fixed profile rules in code: highest prior successful-booking mean,
highest parking mean, highest special-request mean, then the residual low-request
group. Full means and medians are in `cluster_profiles.csv`; the standardized
profile heatmap is `cluster_profile_heatmap.png`.

## Generated files

Running the baseline writes the following reproducible outputs:

- `baseline_metrics.csv`: internal metrics, runtime, and configuration label;
- `cluster_assignments.csv`: training-record-to-cluster mapping;
- `cluster_sizes.csv`: counts and shares;
- `cluster_profiles.csv`: original-unit means and medians;
- `leakage_audit.csv`: a decision for every H2 source column;
- `cluster_profile_heatmap.png`: relative profile comparison;
- `experiment_metadata.json`: input hash, package versions, features, parameters,
  transformations, and confirmation that test data was not used.

## Reproduction

From the repository root, after installing `requirements.txt`:

```powershell
python src/clustering/run_milestone1_baseline.py
python -m unittest discover -s tests -v
```

## Milestone 2 plan

1. Investigate a justified K-Means cluster-count/search space using Silhouette,
   Davies-Bouldin, inertia, stability, minimum cluster size, and interpretability.
2. Add agglomerative hierarchical clustering on a documented representative
   sample and produce the required dendrogram where practical.
3. Add a third family such as Gaussian Mixture Model or DBSCAN, with a search
   appropriate to its parameters and scale.
4. Compare baseline and optimized settings for every algorithm without using
   `IsCanceled`, `ADR`, or other outcomes as selection criteria.
5. Test an appropriately encoded mixed numeric/categorical representation.
6. Add the required PCA-based two-dimensional cluster visualization while keeping
   PCA fitting inside the experimental pipeline.
7. Report stability, sensitivity, runtime, cluster sizes, profiles, and limitations
   rather than selecting an algorithm from one metric alone.
