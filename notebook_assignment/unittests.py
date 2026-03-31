"""
Unit tests for the Seaborn Statistical Plots assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

from unittests_utils import (
    PlotDistributionBattery,
    PlotCategoricalComparisonBattery,
    CreatePairPlotBattery,
    PlotHeatmapBattery,
    PlotRegressionBattery,
)


class test_case:
    """Simple container for a single test result."""

    def __init__(self):
        self.failed = False
        self.msg = ""
        self.want = ""
        self.got = ""


def print_feedback(cases: List[test_case]) -> None:
    """Print colour-coded pass / fail feedback for a list of test cases."""
    failed_cases = [c for c in cases if c.failed]

    if not failed_cases:
        print("\033[92m" + "All tests passed!" + "\033[0m")
    else:
        print(
            "\033[91m"
            + f"Failed {len(failed_cases)} of {len(cases)} tests:"
            + "\033[0m"
        )
        for case in failed_cases:
            print(f"\n  - {case.msg}")
            if case.want:
                print(f"     Expected: {case.want}")
            if case.got:
                print(f"     Got:      {case.got}")


# ---------------------------------------------------------------------------
# Exercise 1 – plot_distribution
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the plot_distribution function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_distribution must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotDistributionBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "plot_distribution must return a matplotlib Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.title_is_set()
            if failed:
                t.failed = True
                t.msg = "The Axes title must be set (e.g. 'amount Distribution')"
                t.got = f"Got title: '{battery.result.get_title() if battery.result else 'N/A'}'"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.axes_has_content()
            if failed:
                t.failed = True
                t.msg = (
                    "The Axes must contain at least one histogram bar or KDE line — "
                    "check that sns.histplot is called with kde=True"
                )
            cases.append(t)

            battery.cleanup()

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 2 – plot_categorical_comparison
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the plot_categorical_comparison function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_categorical_comparison must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotCategoricalComparisonBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "plot_categorical_comparison must return a matplotlib Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.axes_has_artists()
            if failed:
                t.failed = True
                t.msg = (
                    "The Axes must contain drawn artists (patches or collections) — "
                    "check that both violinplot and swarmplot are called"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.xlabel_is_set()
            if failed:
                t.failed = True
                t.msg = "The x-axis label must be set (use ax.set_xlabel(x))"
                t.got = f"Got xlabel: '{battery.result.get_xlabel() if battery.result else 'N/A'}'"
            cases.append(t)

            battery.cleanup()

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 3 – create_pair_plot
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the create_pair_plot function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_pair_plot must be a function"
            return [t]
        cases.append(t)

        try:
            battery = CreatePairPlotBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_not_none()
            if failed:
                t.failed = True
                t.msg = "create_pair_plot must return a non-None value"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.result_is_pairgrid()
            if failed:
                t.failed = True
                t.msg = "create_pair_plot must return a seaborn PairGrid object"
                t.got = f"Got type: {type(battery.result).__name__ if battery.result is not None else 'None'}"
                t.want = "PairGrid"
            cases.append(t)

            battery.cleanup()

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 4 – plot_heatmap
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the plot_heatmap function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_heatmap must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotHeatmapBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "plot_heatmap must return a matplotlib Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.heatmap_drawn()
            if failed:
                t.failed = True
                t.msg = (
                    "The Axes must contain a heatmap mesh (at least one collection) — "
                    "check that sns.heatmap is called with ax=ax"
                )
            cases.append(t)

            battery.cleanup()

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 5 – plot_regression
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the plot_regression function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_regression must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotRegressionBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_not_none()
            if failed:
                t.failed = True
                t.msg = "plot_regression must return a non-None value"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.result_is_facetgrid()
            if failed:
                t.failed = True
                t.msg = "plot_regression must return a seaborn FacetGrid object"
                t.got = f"Got type: {type(battery.result).__name__ if battery.result is not None else 'None'}"
                t.want = "FacetGrid"
            cases.append(t)

            battery.cleanup()

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)
