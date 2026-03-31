"""
Unit test utilities for the Categorical Feature Analysis assignment.

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
# Exercise 1 – get_value_counts
# ---------------------------------------------------------------------------

class GetValueCountsBattery(TestBattery):
    """Test battery for Exercise 1: get_value_counts(df, column)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.column = "region"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.column)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        counts = self.df[self.column].value_counts()
        proportions = counts / len(self.df)
        self.reference_checks = {
            "n_rows": 5,
            "top_count": int(counts.iloc[0]),
        }

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

    def has_correct_row_count(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, self.reference_checks["n_rows"], True
        got = len(self.result)
        want = self.reference_checks["n_rows"]
        return got, want, got != want

    def top_count_is_large_enough(self):
        if not isinstance(self.result, pd.DataFrame) or "count" not in self.result.columns:
            return None, 80, True
        got = int(self.result.iloc[0]["count"])
        want = 80
        return got, want, got < want

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Exercise 2 – identify_rare_categories
# ---------------------------------------------------------------------------

class IdentifyRareCategoriesBattery(TestBattery):
    """Test battery for Exercise 2: identify_rare_categories(df, column, threshold)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.column = "region"
        self.threshold = 0.15

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.column, self.threshold)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        proportions = self.df[self.column].value_counts(normalize=True)
        self.reference_checks = {
            "rare": sorted(proportions[proportions < self.threshold].index.tolist()),
        }

    # --- assertion methods ---

    def result_is_list(self):
        got = isinstance(self.result, list)
        return got, True, not got

    def central_in_result(self):
        if not isinstance(self.result, list):
            return None, True, True
        got = "Central" in self.result
        return got, True, not got

    def north_not_in_result(self):
        if not isinstance(self.result, list):
            return None, True, True
        # North has ~30% proportion, well above 0.15 threshold
        got = "North" not in self.result
        return got, True, not got

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Exercise 3 – plot_sorted_bar
# ---------------------------------------------------------------------------

class PlotSortedBarBattery(TestBattery):
    """Test battery for Exercise 3: plot_sorted_bar(df, column)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.column = "region"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.column)
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
            return None, 5, True
        got = len(self.result.patches)
        want = 5
        return got, want, got < want

    def title_contains_value_counts(self):
        if not isinstance(self.result, Axes):
            return None, True, True
        got = "Value Counts" in self.result.get_title()
        return got, True, not got

    def cleanup(self):
        plt.close("all")


# ---------------------------------------------------------------------------
# Exercise 4 – compute_categorical_summary
# ---------------------------------------------------------------------------

class ComputeCategoricalSummaryBattery(TestBattery):
    """Test battery for Exercise 4: compute_categorical_summary(df)."""

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

    def has_n_unique_column(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, True, True
        got = "n_unique" in self.result.columns
        return got, True, not got

    def has_most_frequent_column(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, True, True
        got = "most_frequent" in self.result.columns
        return got, True, not got

    def has_at_least_one_row(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, 1, True
        got = len(self.result)
        return got, 1, got < 1

    def region_in_index(self):
        if not isinstance(self.result, pd.DataFrame):
            return None, True, True
        got = "region" in self.result.index
        return got, True, not got

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Exercise 5 – plot_proportion_chart
# ---------------------------------------------------------------------------

class PlotProportionChartBattery(TestBattery):
    """Test battery for Exercise 5: plot_proportion_chart(df, column)."""

    def _get_reference_inputs(self):
        self.df = get_orders_df()
        self.column = "region"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.column)
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
            return None, 5, True
        got = len(self.result.patches)
        want = 5
        return got, want, got < want

    def xlabel_contains_proportion(self):
        if not isinstance(self.result, Axes):
            return None, True, True
        got = "Proportion" in self.result.get_xlabel()
        return got, True, not got

    def cleanup(self):
        plt.close("all")
