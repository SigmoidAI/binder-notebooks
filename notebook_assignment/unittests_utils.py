"""
Unit test utilities for the Feature-Target Relationship Mapping assignment.

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

    def cleanup(self):
        plt.close("all")


# ---------------------------------------------------------------------------
# Exercise 1 – compute_feature_target_correlations
# ---------------------------------------------------------------------------

class ComputeFeatureTargetCorrelationsBattery(TestBattery):
    """Test battery for Exercise 1: compute_feature_target_correlations(df, numeric_cols, target_col)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.numeric_cols = ["order_value", "customer_age", "days_to_ship"]
        self.target_col = "returned"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.numeric_cols, self.target_col)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {
            "length": 3,
            "contains_order_value": "order_value",
        }

    # --- assertion methods ---

    def result_is_series(self):
        got = isinstance(self.result, pd.Series)
        return got, True, not got

    def length_is_correct(self):
        if not isinstance(self.result, pd.Series):
            return None, self.reference_checks["length"], True
        got = len(self.result)
        want = self.reference_checks["length"]
        return got, want, got != want

    def index_contains_order_value(self):
        if not isinstance(self.result, pd.Series):
            return None, self.reference_checks["contains_order_value"], True
        got = "order_value" in self.result.index
        return got, True, not got

    def is_sorted_by_abs_descending(self):
        if not isinstance(self.result, pd.Series):
            return None, True, True
        abs_vals = self.result.abs().values
        got = all(abs_vals[i] >= abs_vals[i + 1] for i in range(len(abs_vals) - 1))
        return got, True, not got


# ---------------------------------------------------------------------------
# Exercise 2 – plot_feature_target_correlations
# ---------------------------------------------------------------------------

class PlotFeatureTargetCorrelationsBattery(TestBattery):
    """Test battery for Exercise 2: plot_feature_target_correlations(df, numeric_cols, target_col)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.numeric_cols = ["order_value", "customer_age", "days_to_ship"]
        self.target_col = "returned"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.numeric_cols, self.target_col)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {
            "title_contains": "Correlations",
        }

    # --- assertion methods ---

    def result_is_axes(self):
        got = isinstance(self.result, Axes)
        return got, True, not got

    def has_enough_patches(self):
        if not isinstance(self.result, Axes):
            return None, True, True
        got = len(self.result.patches) >= 3
        return got, True, not got

    def title_contains_correlations(self):
        if not isinstance(self.result, Axes):
            return None, self.reference_checks["title_contains"], True
        got = self.reference_checks["title_contains"] in self.result.get_title()
        return got, True, not got


# ---------------------------------------------------------------------------
# Exercise 3 – compute_group_means
# ---------------------------------------------------------------------------

class ComputeGroupMeansBattery(TestBattery):
    """Test battery for Exercise 3: compute_group_means(df, categorical_col, target_col)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.categorical_col = "region"
        self.target_col = "returned"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.categorical_col, self.target_col)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        ref = self.df.groupby(self.categorical_col)[self.target_col].mean().sort_values(ascending=False)
        self.reference_checks = {
            "length": 5,
            "first_index": ref.index[0],
        }

    # --- assertion methods ---

    def result_is_series(self):
        got = isinstance(self.result, pd.Series)
        return got, True, not got

    def length_is_correct(self):
        if not isinstance(self.result, pd.Series):
            return None, self.reference_checks["length"], True
        got = len(self.result)
        want = self.reference_checks["length"]
        return got, want, got != want

    def first_index_is_highest_mean(self):
        if not isinstance(self.result, pd.Series):
            return None, self.reference_checks["first_index"], True
        got = self.result.index[0]
        want = self.reference_checks["first_index"]
        return got, want, got != want


# ---------------------------------------------------------------------------
# Exercise 4 – plot_kde_by_target
# ---------------------------------------------------------------------------

class PlotKdeByTargetBattery(TestBattery):
    """Test battery for Exercise 4: plot_kde_by_target(df, feature_col, target_col)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.feature_col = "order_value"
        self.target_col = "returned"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.feature_col, self.target_col)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {
            "title_contains": "KDE",
        }

    # --- assertion methods ---

    def result_is_axes(self):
        got = isinstance(self.result, Axes)
        return got, True, not got

    def title_contains_kde(self):
        if not isinstance(self.result, Axes):
            return None, self.reference_checks["title_contains"], True
        got = self.reference_checks["title_contains"] in self.result.get_title()
        return got, True, not got

    def has_at_least_two_lines(self):
        if not isinstance(self.result, Axes):
            return None, True, True
        got = len(self.result.lines) >= 2
        return got, True, not got


# ---------------------------------------------------------------------------
# Exercise 5 – build_relationship_summary
# ---------------------------------------------------------------------------

class BuildRelationshipSummaryBattery(TestBattery):
    """Test battery for Exercise 5: build_relationship_summary(df, numeric_cols, categorical_col, target_col)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.numeric_cols = ["order_value", "customer_age", "days_to_ship"]
        self.categorical_col = "region"
        self.target_col = "returned"

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.df, self.numeric_cols, self.categorical_col, self.target_col
            )
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {
            "shape": (3, 4),
            "col_correlation": "correlation_with_target",
            "col_mean_diff": "mean_diff",
        }

    # --- assertion methods ---

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        return got, True, not got

    def shape_is_correct(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, self.reference_checks["shape"], True
        got = self.result.shape
        want = self.reference_checks["shape"]
        return got, want, got != want

    def has_correlation_column(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, self.reference_checks["col_correlation"], True
        got = self.reference_checks["col_correlation"] in self.result.columns
        return got, True, not got

    def has_mean_diff_column(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, self.reference_checks["col_mean_diff"], True
        got = self.reference_checks["col_mean_diff"] in self.result.columns
        return got, True, not got
