"""
Unit test utilities for the Bivariate Relationships assignment.

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
# Exercise 1 – plot_scatter
# ---------------------------------------------------------------------------

class PlotScatterBattery(TestBattery):
    """Test battery for Exercise 1: plot_scatter(df, x, y, hue_col=None)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.x = "customer_age"
        self.y = "order_value"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.x, self.y)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {
            "title": f"{self.y} vs {self.x}",
        }

    # --- assertion methods ---

    def result_is_axes(self):
        got = isinstance(self.result, Axes)
        return got, True, not got

    def has_collections(self):
        if not isinstance(self.result, Axes):
            return None, True, True
        got = len(self.result.collections) >= 1
        return got, True, not got

    def title_is_correct(self):
        if not isinstance(self.result, Axes):
            return None, self.reference_checks["title"], True
        got = self.result.get_title()
        want = self.reference_checks["title"]
        return got, want, got != want


# ---------------------------------------------------------------------------
# Exercise 2 – plot_violin_by_group
# ---------------------------------------------------------------------------

class PlotViolinByGroupBattery(TestBattery):
    """Test battery for Exercise 2: plot_violin_by_group(df, feature_col, group_col)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.feature_col = "order_value"
        self.group_col = "returned"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.feature_col, self.group_col)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_axes(self):
        got = isinstance(self.result, Axes)
        return got, True, not got

    def has_violin_bodies(self):
        if not isinstance(self.result, Axes):
            return None, True, True
        got = len(self.result.collections) >= 1
        return got, True, not got

    def title_is_correct(self):
        if not isinstance(self.result, Axes):
            return None, f"{self.feature_col} by {self.group_col}", True
        got = self.result.get_title()
        want = f"{self.feature_col} by {self.group_col}"
        return got, want, got != want


# ---------------------------------------------------------------------------
# Exercise 3 – plot_grouped_bar_rate
# ---------------------------------------------------------------------------

class PlotGroupedBarRateBattery(TestBattery):
    """Test battery for Exercise 3: plot_grouped_bar_rate(df, categorical_col, target_col)."""

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
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_axes(self):
        got = isinstance(self.result, Axes)
        return got, True, not got

    def has_enough_bars(self):
        if not isinstance(self.result, Axes):
            return None, True, True
        got = len(self.result.patches) >= 5
        return got, True, not got

    def title_contains_rate(self):
        if not isinstance(self.result, Axes):
            return None, True, True
        got = "Rate" in self.result.get_title()
        return got, True, not got


# ---------------------------------------------------------------------------
# Exercise 4 – compute_correlation_matrix
# ---------------------------------------------------------------------------

class ComputeCorrelationMatrixBattery(TestBattery):
    """Test battery for Exercise 4: compute_correlation_matrix(df, columns)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.columns = ["order_value", "customer_age", "days_to_ship"]

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.columns)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        return got, True, not got

    def shape_is_correct(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, (3, 3), True
        got = self.result.shape
        want = (3, 3)
        return got, want, got != want

    def diagonal_is_one(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, True, True
        try:
            diag = np.diag(self.result.values)
            got = np.allclose(diag, 1.0, atol=1e-9)
            return got, True, not got
        except Exception:
            return None, True, True


# ---------------------------------------------------------------------------
# Exercise 5 – plot_correlation_heatmap
# ---------------------------------------------------------------------------

class PlotCorrelationHeatmapBattery(TestBattery):
    """Test battery for Exercise 5: plot_correlation_heatmap(df, columns)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.columns = ["order_value", "customer_age", "days_to_ship"]

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.columns)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {
            "title": "Correlation Heatmap",
        }

    # --- assertion methods ---

    def result_is_axes(self):
        got = isinstance(self.result, Axes)
        return got, True, not got

    def title_is_correct(self):
        if not isinstance(self.result, Axes):
            return None, self.reference_checks["title"], True
        got = self.result.get_title()
        want = self.reference_checks["title"]
        return got, want, got != want

    def has_images(self):
        if not isinstance(self.result, Axes):
            return None, True, True
        got = len(self.result.images) >= 1
        return got, True, not got
