"""
Unit test utilities for the Matplotlib Fundamentals assignment.

Each Battery class calls the learner's function with controlled inputs and
exposes assertion methods that unittests.py queries to produce feedback.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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
# Exercise 1 – create_subplots
# ---------------------------------------------------------------------------

class CreateSubplotsBattery(TestBattery):
    """Test battery for Exercise 1: create_subplots(nrows, ncols, figsize)."""

    def _get_reference_inputs(self):
        self.nrows = 1
        self.ncols = 2
        self.figsize = (10, 4)

    def extract_info(self):
        try:
            self.result = self.learner_object(self.nrows, self.ncols, self.figsize)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_tuple(self):
        got = isinstance(self.result, tuple)
        return got, True, not got

    def result_has_figure(self):
        if not isinstance(self.result, tuple) or len(self.result) < 1:
            return None, True, True
        got = isinstance(self.result[0], Figure)
        return got, True, not got

    def axes_has_correct_length(self):
        if not isinstance(self.result, tuple) or len(self.result) < 2:
            return None, self.ncols, True
        try:
            length = len(np.atleast_1d(self.result[1]))
        except Exception:
            return None, self.ncols, True
        return length, self.ncols, length != self.ncols

    def cleanup(self):
        if isinstance(self.result, tuple) and len(self.result) > 0:
            try:
                plt.close(self.result[0])
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Exercise 2 – add_line_chart
# ---------------------------------------------------------------------------

class AddLineChartBattery(TestBattery):
    """Test battery for Exercise 2: add_line_chart(ax, x, y, title, xlabel, ylabel)."""

    def _get_reference_inputs(self):
        self.fig, self.ax = plt.subplots()
        self.x = np.arange(10)
        self.y = np.array([3.0, 7.0, 2.0, 8.0, 5.0, 9.0, 4.0, 6.0, 1.0, 10.0])
        self.title = 'Test Chart Title'
        self.xlabel = 'X Axis Label'
        self.ylabel = 'Y Axis Label'

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.ax, self.x, self.y,
                self.title, self.xlabel, self.ylabel,
            )
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def result_is_axes(self):
        got = isinstance(self.result, Axes)
        return got, True, not got

    def ax_has_line(self):
        try:
            got = len(self.ax.get_lines()) >= 1
        except Exception:
            got = False
        return got, True, not got

    def title_is_set(self):
        try:
            got = self.ax.get_title() == self.title
        except Exception:
            got = False
        return got, True, not got

    def xlabel_is_set(self):
        try:
            got = self.ax.get_xlabel() == self.xlabel
        except Exception:
            got = False
        return got, True, not got

    def cleanup(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Exercise 3 – add_peak_annotation
# ---------------------------------------------------------------------------

class AddAnnotationBattery(TestBattery):
    """Test battery for Exercise 3: add_peak_annotation(ax, df, x_column, y_column)."""

    def _get_reference_inputs(self):
        self.fig, self.ax = plt.subplots()
        self.df = get_sales_df().sort_values('order_date').reset_index(drop=True)
        # pre-draw a line so the annotation has context
        self.ax.plot(self.df['order_date'], self.df['amount'])

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.ax, self.df, 'order_date', 'amount'
            )
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def ax_has_annotation(self):
        try:
            got = len(self.ax.texts) >= 1
        except Exception:
            got = False
        return got, True, not got

    def cleanup(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Exercise 4 – apply_clean_styling
# ---------------------------------------------------------------------------

class ApplyStylingBattery(TestBattery):
    """Test battery for Exercise 4: apply_clean_styling(ax, grid=True)."""

    def _get_reference_inputs(self):
        self.fig, self.ax = plt.subplots()
        # add a bar so the axes is not empty
        self.ax.bar(['A', 'B', 'C'], [3, 5, 2])

    def extract_info(self):
        try:
            self.result = self.learner_object(self.ax)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def right_spine_hidden(self):
        try:
            got = not self.ax.spines['right'].get_visible()
        except Exception:
            got = False
        return got, True, not got

    def top_spine_hidden(self):
        try:
            got = not self.ax.spines['top'].get_visible()
        except Exception:
            got = False
        return got, True, not got

    def cleanup(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Exercise 5 – save_figure
# ---------------------------------------------------------------------------

class SaveFigureBattery(TestBattery):
    """Test battery for Exercise 5: save_figure(fig, path, dpi=150)."""

    def _get_reference_inputs(self):
        self.fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        self.path = '/tmp/test_L4_1_N1.png'
        # ensure the file does not exist before the test
        if os.path.exists(self.path):
            os.remove(self.path)

    def extract_info(self):
        try:
            self.result = self.learner_object(self.fig, self.path, dpi=150)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {}

    # --- assertion methods ---

    def file_exists(self):
        try:
            got = os.path.exists(self.path)
        except Exception:
            got = False
        return got, True, not got

    def cleanup(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass
