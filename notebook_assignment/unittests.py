"""
Unit tests for the Bivariate Relationships assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

from unittests_utils import (
    PlotScatterBattery,
    PlotViolinByGroupBattery,
    PlotGroupedBarRateBattery,
    ComputeCorrelationMatrixBattery,
    PlotCorrelationHeatmapBattery,
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
# Exercise 1 – plot_scatter
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the plot_scatter function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_scatter must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotScatterBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "plot_scatter must return a matplotlib Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_collections()
            if failed:
                t.failed = True
                t.msg = "plot_scatter must draw at least one scatter collection"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.title_is_correct()
            if failed:
                t.failed = True
                t.msg = "plot_scatter title must be 'order_value vs customer_age'"
                t.want = str(want)
                t.got = str(got)
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
# Exercise 2 – plot_violin_by_group
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the plot_violin_by_group function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_violin_by_group must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotViolinByGroupBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "plot_violin_by_group must return a matplotlib Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_violin_bodies()
            if failed:
                t.failed = True
                t.msg = "plot_violin_by_group must draw violin body collections (ax.collections must be non-empty)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.title_is_correct()
            if failed:
                t.failed = True
                t.msg = "plot_violin_by_group title must be 'order_value by returned'"
                t.want = str(want)
                t.got = str(got)
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
# Exercise 3 – plot_grouped_bar_rate
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the plot_grouped_bar_rate function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_grouped_bar_rate must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotGroupedBarRateBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "plot_grouped_bar_rate must return a matplotlib Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_enough_bars()
            if failed:
                t.failed = True
                t.msg = "plot_grouped_bar_rate must draw at least 5 bars (one per region)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.title_contains_rate()
            if failed:
                t.failed = True
                t.msg = "plot_grouped_bar_rate title must contain the word 'Rate'"
                t.got = str(battery.result.get_title() if battery.result else None)
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
# Exercise 4 – compute_correlation_matrix
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the compute_correlation_matrix function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_correlation_matrix must be a function"
            return [t]
        cases.append(t)

        try:
            battery = ComputeCorrelationMatrixBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "compute_correlation_matrix must return a pd.DataFrame"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.shape_is_correct()
            if failed:
                t.failed = True
                t.msg = "Returned DataFrame must have shape (3, 3) for 3 input columns"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.diagonal_is_one()
            if failed:
                t.failed = True
                t.msg = "Diagonal of the correlation matrix must be 1.0 (each variable correlates perfectly with itself)"
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
# Exercise 5 – plot_correlation_heatmap
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the plot_correlation_heatmap function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_correlation_heatmap must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotCorrelationHeatmapBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "plot_correlation_heatmap must return a matplotlib Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.title_is_correct()
            if failed:
                t.failed = True
                t.msg = "plot_correlation_heatmap title must be 'Correlation Heatmap'"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_images()
            if failed:
                t.failed = True
                t.msg = "plot_correlation_heatmap must use ax.imshow (ax.images must be non-empty)"
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
