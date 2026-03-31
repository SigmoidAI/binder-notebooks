"""
Unit test utilities for the Target Variable Deep Dive assignment.

Each Battery class calls the learner's function with controlled inputs and
exposes assertion methods that unittests.py queries to produce feedback.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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
# Exercise 1 – get_class_balance
# ---------------------------------------------------------------------------

class GetClassBalanceBattery(TestBattery):
    """Test battery for Exercise 1: get_class_balance(df, target_col)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.target_col = "returned"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.target_col)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        return got, True, not got

    def has_count_column(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, True, True
        got = "count" in self.result.columns
        return got, True, not got

    def has_proportion_column(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, True, True
        got = "proportion" in self.result.columns
        return got, True, not got

    def has_two_rows(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, 2, True
        got = len(self.result)
        want = 2
        return got, want, got != want

    def majority_class_proportion_above_threshold(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, 0.7, True
        if "proportion" not in self.result.columns:
            return None, 0.7, True
        try:
            got = float(self.result.loc[0, "proportion"])
            want = 0.7
            return got, want, got <= want
        except Exception:
            return None, 0.7, True

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Exercise 2 – classify_imbalance
# ---------------------------------------------------------------------------

class ClassifyImbalanceBattery(TestBattery):
    """Test battery for Exercise 2: classify_imbalance(minority_proportion)."""

    def _get_reference_inputs(self):
        self.severe_input = 0.18
        self.balanced_input = 0.45

    def extract_info(self):
        try:
            self.result_severe = self.learner_object(self.severe_input)
        except Exception:
            self.result_severe = None
        try:
            self.result_balanced = self.learner_object(self.balanced_input)
        except Exception:
            self.result_balanced = None

    def get_reference_checks(self):
        self.reference_checks = {
            "severe": "severe_imbalance",
            "balanced": "balanced",
        }

    # --- assertion methods ---

    def severe_returns_correct_string(self):
        got = self.result_severe
        want = self.reference_checks["severe"]
        return got, want, got != want

    def balanced_returns_correct_string(self):
        got = self.result_balanced
        want = self.reference_checks["balanced"]
        return got, want, got != want

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Exercise 3 – plot_target_distribution
# ---------------------------------------------------------------------------

class PlotTargetDistributionBattery(TestBattery):
    """Test battery for Exercise 3: plot_target_distribution(df, target_col)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.target_col = "returned"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.target_col)
        except Exception:
            self.result = None
        plt.close("all")

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_axes(self):
        got = isinstance(self.result, Axes)
        return got, True, not got

    def has_enough_patches(self):
        if not isinstance(self.result, Axes):
            return None, 2, True
        got = len(self.result.patches)
        want = 2
        return got, want, got < want

    def title_contains_class_distribution(self):
        if not isinstance(self.result, Axes):
            return None, True, True
        got = "Class Distribution" in self.result.get_title()
        return got, True, not got

    def cleanup(self):
        plt.close("all")


# ---------------------------------------------------------------------------
# Exercise 4 – compute_target_stats
# ---------------------------------------------------------------------------

class ComputeTargetStatsBattery(TestBattery):
    """Test battery for Exercise 4: compute_target_stats(df, target_col, numeric_cols)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.target_col = "returned"
        self.numeric_cols = ["order_value", "customer_age"]

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.target_col, self.numeric_cols)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        return got, True, not got

    def has_correct_shape(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, (2, 2), True
        got = self.result.shape
        want = (2, 2)
        return got, want, got != want

    def has_order_value_column(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, True, True
        got = "order_value" in self.result.columns
        return got, True, not got

    def has_customer_age_column(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, True, True
        got = "customer_age" in self.result.columns
        return got, True, not got

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Exercise 5 – detect_target_anomalies
# ---------------------------------------------------------------------------

class DetectTargetAnomaliesBattery(TestBattery):
    """Test battery for Exercise 5: detect_target_anomalies(df, target_col)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.target_col = "returned"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.target_col)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_dict(self):
        got = isinstance(self.result, dict)
        return got, True, not got

    def has_null_count_key(self):
        if not isinstance(self.result, dict):
            return None, True, True
        got = "null_count" in self.result
        return got, True, not got

    def null_count_is_zero(self):
        if not isinstance(self.result, dict) or "null_count" not in self.result:
            return None, 0, True
        got = self.result["null_count"]
        want = 0
        return got, want, got != want

    def unexpected_count_is_zero(self):
        if not isinstance(self.result, dict) or "unexpected_count" not in self.result:
            return None, 0, True
        got = self.result["unexpected_count"]
        want = 0
        return got, want, got != want

    def cleanup(self):
        pass
