"""
Unit test utilities for the Seaborn Statistical Plots assignment.

Each Battery class calls the learner's function with controlled inputs and
exposes assertion methods that unittests.py queries to produce feedback.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from helper_utils import get_sales_df


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
            return got, want, not np.isclose(got, want, rtol=1e-5, atol=1e-8)

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
# Exercise 1 – plot_distribution
# ---------------------------------------------------------------------------

class PlotDistributionBattery(TestBattery):
    """Test battery for Exercise 1: plot_distribution(df, column, hue=None)."""

    def _get_reference_inputs(self):
        self.df = get_sales_df()
        self.column = "amount"
        self.hue = None

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.column, hue=self.hue)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_axes(self):
        got = isinstance(self.result, Axes)
        return got, True, not got

    def title_is_set(self):
        try:
            got = self.result.get_title() != ""
        except Exception:
            got = False
        return got, True, not got

    def axes_has_content(self):
        try:
            has_patches = len(self.result.patches) >= 1
            has_lines = len(self.result.get_lines()) >= 1
            got = has_patches or has_lines
        except Exception:
            got = False
        return got, True, not got

    def cleanup(self):
        try:
            plt.close("all")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Exercise 2 – plot_categorical_comparison
# ---------------------------------------------------------------------------

class PlotCategoricalComparisonBattery(TestBattery):
    """Test battery for Exercise 2: plot_categorical_comparison(df, x, y)."""

    def _get_reference_inputs(self):
        self.df = get_sales_df()
        self.x = "category"
        self.y = "amount"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.x, self.y)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_axes(self):
        got = isinstance(self.result, Axes)
        return got, True, not got

    def axes_has_artists(self):
        try:
            has_patches = len(self.result.patches) >= 1
            has_collections = len(self.result.collections) >= 1
            got = has_patches or has_collections
        except Exception:
            got = False
        return got, True, not got

    def xlabel_is_set(self):
        try:
            got = self.result.get_xlabel() != ""
        except Exception:
            got = False
        return got, True, not got

    def cleanup(self):
        try:
            plt.close("all")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Exercise 3 – create_pair_plot
# ---------------------------------------------------------------------------

class CreatePairPlotBattery(TestBattery):
    """Test battery for Exercise 3: create_pair_plot(df, columns, hue)."""

    def _get_reference_inputs(self):
        self.df = get_sales_df()
        self.columns = ["amount", "units"]
        self.hue = "category"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.columns, self.hue)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_not_none(self):
        got = self.result is not None
        return got, True, not got

    def result_is_pairgrid(self):
        try:
            got = type(self.result).__name__ == "PairGrid"
        except Exception:
            got = False
        return got, True, not got

    def cleanup(self):
        try:
            plt.close("all")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Exercise 4 – plot_heatmap
# ---------------------------------------------------------------------------

class PlotHeatmapBattery(TestBattery):
    """Test battery for Exercise 4: plot_heatmap(df)."""

    def _get_reference_inputs(self):
        self.df = get_sales_df()[["amount", "units"]]

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_axes(self):
        got = isinstance(self.result, Axes)
        return got, True, not got

    def heatmap_drawn(self):
        try:
            got = len(self.result.collections) >= 1
        except Exception:
            got = False
        return got, True, not got

    def cleanup(self):
        try:
            plt.close("all")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Exercise 5 – plot_regression
# ---------------------------------------------------------------------------

class PlotRegressionBattery(TestBattery):
    """Test battery for Exercise 5: plot_regression(df, x, y, hue)."""

    def _get_reference_inputs(self):
        self.df = get_sales_df()
        self.x = "units"
        self.y = "amount"
        self.hue = "category"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.df, self.x, self.y, self.hue)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_not_none(self):
        got = self.result is not None
        return got, True, not got

    def result_is_facetgrid(self):
        try:
            got = type(self.result).__name__ == "FacetGrid"
        except Exception:
            got = False
        return got, True, not got

    def cleanup(self):
        try:
            plt.close("all")
        except Exception:
            pass
