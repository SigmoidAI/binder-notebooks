"""
Unit test utilities for the Distribution Analysis assignment.

Each Battery class calls the learner's function with controlled inputs and
exposes assertion methods that unittests.py queries to produce feedback.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from helper_utils import get_orders_df


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TestBattery:
    """Base class for test batteries."""

    def __init__(self, learner_object):
        self.learner_object = learner_object
        self._get_reference_inputs()
        self.extract_info()
        self.get_reference_checks()

    def _get_reference_inputs(self):
        pass

    def extract_info(self):
        pass

    def get_reference_checks(self):
        pass

    def _check(self, check_name, got):
        """Compare *got* against the reference value stored in reference_checks."""
        want = self.reference_checks[check_name]

        if isinstance(want, float):
            if got is None:
                return got, want, True
            return got, want, not np.isclose(got, want, rtol=1e-2, atol=1e-6)

        if isinstance(want, np.ndarray):
            if got is None:
                return got, want, True
            try:
                failed = not np.allclose(got, want, rtol=1e-5, atol=1e-8)
            except Exception:
                failed = True
            return got, want, failed

        if isinstance(want, pd.DataFrame):
            if got is None:
                return got, want, True
            try:
                failed = not got.reset_index(drop=True).equals(
                    want.reset_index(drop=True)
                )
            except Exception:
                failed = True
            return got, want, failed

        if isinstance(want, pd.Series):
            if got is None:
                return got, want, True
            try:
                failed = not got.reset_index(drop=True).equals(
                    want.reset_index(drop=True)
                )
            except Exception:
                failed = True
            return got, want, failed

        condition = got != want
        return got, want, condition


# ---------------------------------------------------------------------------
# Exercise 1 – compute_distribution_stats
# ---------------------------------------------------------------------------

class ComputeDistributionStatsBattery(TestBattery):
    """Test battery for Exercise 1: compute_distribution_stats(df, column)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.column = "order_value"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.column)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        s = self.df[self.column]
        self.reference_checks = {
            "mean": float(s.mean()),
        }
        self.expected_keys = {"mean", "median", "std", "iqr", "skewness", "kurtosis"}

    # --- assertion methods ---

    def result_is_dict(self):
        got = isinstance(self.result, dict)
        return got, True, not got

    def all_keys_present(self):
        if not isinstance(self.result, dict):
            return None, self.expected_keys, True
        missing = self.expected_keys - set(self.result.keys())
        got = len(missing) == 0
        return (
            set(self.result.keys()),
            self.expected_keys,
            not got,
        )

    def mean_is_correct(self):
        if not isinstance(self.result, dict):
            return None, self.reference_checks["mean"], True
        got = self.result.get("mean")
        return self._check("mean", got)

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Exercise 2 – classify_distribution
# ---------------------------------------------------------------------------

class ClassifyDistributionBattery(TestBattery):
    """Test battery for Exercise 2: classify_distribution(skewness, kurtosis)."""

    def _get_reference_inputs(self):
        self.case1_skew = 1.7
        self.case1_kurt = 3.8
        self.case2_skew = 0.05
        self.case2_kurt = -0.2

    def extract_info(self):
        try:
            self.result1 = self.learner_object(self.case1_skew, self.case1_kurt)
        except Exception:
            self.result1 = None
        try:
            self.result2 = self.learner_object(self.case2_skew, self.case2_kurt)
        except Exception:
            self.result2 = None

    def get_reference_checks(self):
        self.reference_checks = {
            "case1": "strong_right_skew",
            "case2": "symmetric",
        }

    # --- assertion methods ---

    def strong_right_skew_correct(self):
        want = self.reference_checks["case1"]
        got = self.result1
        return got, want, got != want

    def symmetric_correct(self):
        want = self.reference_checks["case2"]
        got = self.result2
        return got, want, got != want

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Exercise 3 – plot_distribution_triple
# ---------------------------------------------------------------------------

class PlotDistributionTripleBattery(TestBattery):
    """Test battery for Exercise 3: plot_distribution_triple(df, column)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.column = "customer_age"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.column)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_tuple(self):
        got = isinstance(self.result, tuple)
        return got, True, not got

    def tuple_has_two_elements(self):
        if not isinstance(self.result, tuple):
            return None, 2, True
        got = len(self.result)
        return got, 2, got != 2

    def first_element_is_figure(self):
        if not isinstance(self.result, tuple) or len(self.result) < 1:
            return None, True, True
        got = isinstance(self.result[0], Figure)
        return got, True, not got

    def axes_has_three_panels(self):
        if not isinstance(self.result, tuple) or len(self.result) < 2:
            return None, 3, True
        try:
            length = len(np.atleast_1d(self.result[1]))
        except Exception:
            return None, 3, True
        return length, 3, length != 3

    def cleanup(self):
        if isinstance(self.result, tuple) and len(self.result) > 0:
            try:
                plt.close(self.result[0])
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Exercise 4 – summarize_all_numeric
# ---------------------------------------------------------------------------

class SummarizeAllNumericBattery(TestBattery):
    """Test battery for Exercise 4: summarize_all_numeric(df)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        return got, True, not got

    def has_skewness_column(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, True, True
        got = "skewness" in self.result.columns
        return got, True, not got

    def has_iqr_column(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, True, True
        got = "iqr" in self.result.columns
        return got, True, not got

    def has_enough_rows(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, True, True
        got = len(self.result)
        return got, True, got < 4

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Exercise 5 – detect_skewed_columns
# ---------------------------------------------------------------------------

class DetectSkewedColumnsBattery(TestBattery):
    """Test battery for Exercise 5: detect_skewed_columns(df, threshold=1.0)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.threshold = 1.0

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, threshold=self.threshold)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_list(self):
        got = isinstance(self.result, list)
        return got, True, not got

    def order_value_in_result(self):
        if not isinstance(self.result, list):
            return None, True, True
        got = "order_value" in self.result
        return got, True, not got

    def customer_age_not_in_result(self):
        if not isinstance(self.result, list):
            return None, True, True
        got = "customer_age" not in self.result
        return got, True, not got

    def cleanup(self):
        pass
