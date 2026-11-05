"""Behavioral checks for regression leakage boundaries and inference inputs."""

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from src.preprocessing.regression.preprocessing import (
    NUMERIC_COLUMNS, PREDICTOR_COLUMNS, build_modeling_pipeline,
    clean_predictors, prepare_training_frame,
)


def sample_frame():
    rows = []
    for i in range(12):
        rows.append({
            "IsCanceled": str(i % 2), "LeadTime": str(i + 1),
            "ArrivalDateYear": "2016", "ArrivalDateMonth": "January",
            "ArrivalDateWeekNumber": "2", "ArrivalDateDayOfMonth": "10",
            "StaysInWeekendNights": "1", "StaysInWeekNights": "2",
            "Adults": "2", "Children": "0" if i % 2 else "2", "Babies": "0",
            "Meal": "BB", "Country": "PRT", "MarketSegment": "Direct",
            "DistributionChannel": "Direct", "IsRepeatedGuest": "0",
            "PreviousCancellations": "0", "PreviousBookingsNotCanceled": "0",
            "ReservedRoomType": "A", "AssignedRoomType": "A", "BookingChanges": "0",
            "DepositType": "No Deposit", "Agent": "NULL", "Company": "NULL",
            "DaysInWaitingList": "0", "CustomerType": "Transient", "ADR": str(80 + i),
            "RequiredCarParkingSpaces": "0", "TotalOfSpecialRequests": "0",
            "ReservationStatus": "Check-Out", "ReservationStatusDate": "2016-01-13",
        })
    return pd.DataFrame(rows)


class RegressionContractTests(unittest.TestCase):
    def test_population_rule_preserves_zero_rate_and_original_row_identity(self):
        raw = sample_frame()
        before = raw.copy(deep=True)
        raw.loc[0, ["StaysInWeekendNights", "StaysInWeekNights", "ADR"]] = "0"
        raw.loc[1, "ADR"] = "0"
        expected_input = raw.copy(deep=True)
        prepared, excluded = prepare_training_frame(raw)
        self.assertEqual(excluded.shared_train_record_number.tolist(), [1])
        self.assertNotIn(0, prepared.index)
        self.assertEqual(prepared.loc[1, "ADR"], 0)
        self.assertEqual(len(prepared), len(before) - 1)
        pd.testing.assert_frame_equal(raw, expected_input)

    def test_preprocessing_learns_training_statistics_only_and_handles_unseen_ids(self):
        raw = sample_frame()
        raw.loc[0, "Children"] = ""
        raw.loc[1, "Country"] = "NULL"
        prepared, _ = prepare_training_frame(raw)
        X, y = prepared[PREDICTOR_COLUMNS], prepared.ADR
        pipeline = build_modeling_pipeline(LinearRegression()).fit(X, y)
        numeric = pipeline.named_steps["preprocessor"].named_transformers_["numeric"]
        child_idx = NUMERIC_COLUMNS.index("Children")
        self.assertEqual(numeric.named_steps["imputer"].statistics_[child_idx], X.Children.median())
        lead_mean = numeric.named_steps["scaler"].mean_[NUMERIC_COLUMNS.index("LeadTime")]
        self.assertEqual(lead_mean, X.LeadTime.mean())
        validation = X.iloc[[0]].copy()
        validation.loc[:, "LeadTime"] = 600.0
        validation.loc[:, "Agent"] = "UNSEEN_AGENCY"
        validation.loc[:, "Country"] = "UNSEEN_COUNTRY"
        self.assertTrue(np.isfinite(pipeline.predict(validation)).all())
        encoder = pipeline.named_steps["preprocessor"].named_transformers_["categorical"].named_steps["encoder"]
        self.assertFalse(any("UNSEEN_AGENCY" in categories for categories in encoder.categories_))
        self.assertEqual(numeric.named_steps["scaler"].mean_[NUMERIC_COLUMNS.index("LeadTime")], lead_mean)

    def test_outcome_columns_cannot_enter_predictors(self):
        raw = sample_frame()
        clean = clean_predictors(raw)
        changed = raw.copy()
        changed["ADR"] = "9999"
        changed["IsCanceled"] = "1"
        changed["ReservationStatus"] = "Canceled"
        changed["DepositType"] = "Non Refund"
        pd.testing.assert_frame_equal(clean_predictors(changed), clean)
        self.assertEqual(clean.columns.tolist(), PREDICTOR_COLUMNS)

    def test_grouping_does_not_depend_on_target(self):
        raw = sample_frame()
        raw.loc[1, PREDICTOR_COLUMNS] = raw.loc[0, PREDICTOR_COLUMNS]
        prepared, _ = prepare_training_frame(raw)
        X = prepared[PREDICTOR_COLUMNS]
        groups = X.groupby(PREDICTOR_COLUMNS, dropna=False, sort=False).ngroup()
        self.assertEqual(groups.iloc[0], groups.iloc[1])
        for train_idx, val_idx in GroupKFold(n_splits=3, shuffle=True, random_state=42).split(X, groups=groups):
            self.assertFalse(set(groups.iloc[train_idx]) & set(groups.iloc[val_idx]))

    def test_reject_bad_numeric_target_or_inference_population(self):
        for column, value in [("ADR", ""), ("ADR", "inf"), ("LeadTime", "oops"), ("Adults", "-1")]:
            raw = sample_frame()
            raw.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                prepare_training_frame(raw)
        raw = sample_frame()
        raw.loc[0, ["StaysInWeekendNights", "StaysInWeekNights"]] = "0"
        with self.assertRaises(ValueError):
            clean_predictors(raw)
        with self.assertRaises(ValueError):
            clean_predictors(sample_frame().drop(columns="ReservedRoomType"))


if __name__ == "__main__":
    unittest.main()
