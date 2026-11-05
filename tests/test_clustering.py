"""Behavioral tests for the clustering feature and leakage contract."""

import unittest

import numpy as np
import pandas as pd

from src.clustering.run_milestone1_baseline import N_CLUSTERS, build_pipeline
from src.preprocessing.clustering.preprocessing import (
    MODEL_FEATURES,
    prepare_clustering_features,
)


def sample_frame(rows=16):
    """Create small valid source data with varied, clusterable behaviour."""
    data = []
    for i in range(rows):
        data.append({
            "LeadTime": str(5 + 30 * (i % 4)),
            "StaysInWeekendNights": str(i % 3),
            "StaysInWeekNights": str(1 + i % 4),
            "Adults": "2",
            "Children": "" if i == 0 else str(i % 2),
            "Babies": "0",
            "IsRepeatedGuest": str(int(i % 5 == 0)),
            "PreviousCancellations": str(i % 3),
            "PreviousBookingsNotCanceled": str(i % 6),
            "RequiredCarParkingSpaces": str(int(i % 7 == 0)),
            "TotalOfSpecialRequests": str(i % 4),
            # These outcome/future values must have no effect on model inputs.
            "IsCanceled": str(i % 2),
            "ADR": str(80 + i),
            "AssignedRoomType": "A",
            "ReservationStatus": "Check-Out",
        })
    return pd.DataFrame(data)


class ClusteringContractTests(unittest.TestCase):
    def test_derived_features_and_missing_children(self):
        features = prepare_clustering_features(sample_frame())
        self.assertEqual(features.columns.tolist(), MODEL_FEATURES)
        self.assertEqual(features.loc[1, "TotalNights"], 3)
        self.assertEqual(features.loc[1, "TotalGuests"], 3)
        self.assertTrue(np.isnan(features.loc[0, "TotalGuests"]))

    def test_outcomes_and_future_fields_cannot_change_inputs(self):
        raw = sample_frame()
        expected = prepare_clustering_features(raw)
        changed = raw.copy()
        changed["IsCanceled"] = "1"
        changed["ADR"] = "9999"
        changed["AssignedRoomType"] = "Z"
        changed["ReservationStatus"] = "Canceled"
        pd.testing.assert_frame_equal(prepare_clustering_features(changed), expected)

    def test_pipeline_is_deterministic_and_assigns_requested_clusters(self):
        features = prepare_clustering_features(sample_frame(40))
        first = build_pipeline().fit_predict(features)
        second = build_pipeline().fit_predict(features)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(np.unique(first).size, N_CLUSTERS)

    def test_invalid_or_missing_source_values_fail_loudly(self):
        with self.assertRaises(ValueError):
            prepare_clustering_features(sample_frame().drop(columns="LeadTime"))
        invalid = sample_frame()
        invalid.loc[0, "Adults"] = "not-a-number"
        with self.assertRaises(ValueError):
            prepare_clustering_features(invalid)


if __name__ == "__main__":
    unittest.main()
