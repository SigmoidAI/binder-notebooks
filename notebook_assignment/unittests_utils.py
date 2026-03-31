"""
Unit test utilities for the Interactive Visualization (Plotly) assignment.

Each Battery class calls the learner's function with controlled inputs and
exposes assertion methods that unittests.py queries to produce feedback.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
# Exercise 1 – create_scatter
# ---------------------------------------------------------------------------

class CreateScatterBattery(TestBattery):
    """Test battery for Exercise 1: create_scatter(df, x, y, color, hover_data)."""

    def _get_reference_inputs(self):
        self.df = get_sales_df()
        self.x = "units"
        self.y = "amount"
        self.color = "category"
        self.hover_data = ["region"]

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.df,
                x=self.x,
                y=self.y,
                color=self.color,
                hover_data=self.hover_data,
            )
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_figure(self):
        got = isinstance(self.result, go.Figure)
        return got, True, not got

    def has_traces(self):
        try:
            got = len(self.result.data) >= 1
        except Exception:
            got = False
        return got, True, not got

    def title_is_set(self):
        try:
            title_text = self.result.layout.title.text
            got = title_text is not None and title_text != ""
        except Exception:
            got = False
        return got, True, not got


# ---------------------------------------------------------------------------
# Exercise 2 – create_time_series
# ---------------------------------------------------------------------------

class CreateTimeSeriesBattery(TestBattery):
    """Test battery for Exercise 2: create_time_series(df, date_col, value_col, color_col)."""

    def _get_reference_inputs(self):
        self.df = get_sales_df()
        self.date_col = "order_date"
        self.value_col = "amount"
        self.color_col = "category"

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.df,
                date_col=self.date_col,
                value_col=self.value_col,
                color_col=self.color_col,
            )
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_figure(self):
        got = isinstance(self.result, go.Figure)
        return got, True, not got

    def has_traces(self):
        try:
            got = len(self.result.data) >= 1
        except Exception:
            got = False
        return got, True, not got

    def xaxis_title_is_set(self):
        try:
            x_title = self.result.layout.xaxis.title.text
            got = x_title is not None and x_title != ""
        except Exception:
            got = False
        return got, True, not got


# ---------------------------------------------------------------------------
# Exercise 3 – create_grouped_bar
# ---------------------------------------------------------------------------

class CreateGroupedBarBattery(TestBattery):
    """Test battery for Exercise 3: create_grouped_bar(df, x, y, color)."""

    def _get_reference_inputs(self):
        self.df = get_sales_df()
        self.x = "category"
        self.y = "amount"
        self.color = "status"

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.df,
                x=self.x,
                y=self.y,
                color=self.color,
            )
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_figure(self):
        got = isinstance(self.result, go.Figure)
        return got, True, not got

    def has_traces(self):
        try:
            got = len(self.result.data) >= 1
        except Exception:
            got = False
        return got, True, not got

    def barmode_is_group(self):
        try:
            got = self.result.layout.barmode == "group"
        except Exception:
            got = False
        return got, True, not got


# ---------------------------------------------------------------------------
# Exercise 4 – customize_layout
# ---------------------------------------------------------------------------

class CustomizeLayoutBattery(TestBattery):
    """Test battery for Exercise 4: customize_layout(fig, title, xaxis_title, yaxis_title)."""

    def _get_reference_inputs(self):
        df = get_sales_df()
        self.base_fig = px.scatter(df, x="units", y="amount")
        self.title = "My Custom Title"
        self.xaxis_title = "Units Sold"
        self.yaxis_title = "Revenue (USD)"

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.base_fig,
                title=self.title,
                xaxis_title=self.xaxis_title,
                yaxis_title=self.yaxis_title,
            )
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_figure(self):
        got = isinstance(self.result, go.Figure)
        return got, True, not got

    def title_matches(self):
        try:
            got = self.result.layout.title.text == self.title
        except Exception:
            got = False
        return got, True, not got

    def xaxis_title_matches(self):
        try:
            got = self.result.layout.xaxis.title.text == self.xaxis_title
        except Exception:
            got = False
        return got, True, not got

    def yaxis_title_matches(self):
        try:
            got = self.result.layout.yaxis.title.text == self.yaxis_title
        except Exception:
            got = False
        return got, True, not got


# ---------------------------------------------------------------------------
# Exercise 5 – save_figure_html
# ---------------------------------------------------------------------------

class SaveFigureHtmlBattery(TestBattery):
    """Test battery for Exercise 5: save_figure_html(fig, path)."""

    def _get_reference_inputs(self):
        df = get_sales_df()
        self.fig = px.scatter(df, x="units", y="amount", title="Test Figure")
        self.path = "/tmp/test_L4_1_N3.html"
        # Remove pre-existing file so the test is clean
        if os.path.exists(self.path):
            os.remove(self.path)

    def extract_info(self):
        try:
            self.learner_object(self.fig, self.path)
            self.result = True
        except Exception:
            self.result = False

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def file_exists(self):
        got = os.path.exists(self.path)
        return got, True, not got

    def file_is_not_empty(self):
        try:
            got = os.path.getsize(self.path) > 0
        except Exception:
            got = False
        return got, True, not got
