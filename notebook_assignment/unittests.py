"""
Unit tests for the Categorical Feature Analysis assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

from unittests_utils import (
    GetValueCountsBattery,
    IdentifyRareCategoriesBattery,
    PlotSortedBarBattery,
    ComputeCategoricalSummaryBattery,
    PlotProportionChartBattery,
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
# Exercise 1 – get_value_counts
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the get_value_counts function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "get_value_counts must be a function"
            return [t]
        cases.append(t)

        try:
            battery = GetValueCountsBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "get_value_counts must return a pd.DataFrame"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_count_column()
            if failed:
                t.failed = True
                t.msg = "Returned DataFrame must have a 'count' column"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_proportion_column()
            if failed:
                t.failed = True
                t.msg = "Returned DataFrame must have a 'proportion' column"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_correct_row_count()
            if failed:
                t.failed = True
                t.msg = "DataFrame must have exactly 5 rows (one per region)"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.top_count_is_large_enough()
            if failed:
                t.failed = True
                t.msg = "The top category ('North') should have at least 80 occurrences"
                t.want = f">= {want}"
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
# Exercise 2 – identify_rare_categories
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the identify_rare_categories function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "identify_rare_categories must be a function"
            return [t]
        cases.append(t)

        try:
            battery = IdentifyRareCategoriesBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_list()
            if failed:
                t.failed = True
                t.msg = "identify_rare_categories must return a list"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.central_in_result()
            if failed:
                t.failed = True
                t.msg = "'Central' must be in the result (proportion ~10% < 15% threshold)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.north_not_in_result()
            if failed:
                t.failed = True
                t.msg = "'North' must NOT be in the result (proportion ~30% > 15% threshold)"
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
# Exercise 3 – plot_sorted_bar
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the plot_sorted_bar function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_sorted_bar must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotSortedBarBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "plot_sorted_bar must return a matplotlib Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_enough_patches()
            if failed:
                t.failed = True
                t.msg = "The bar chart must have at least 5 bars (one per region)"
                t.want = f">= {want}"
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.title_contains_value_counts()
            if failed:
                t.failed = True
                t.msg = "The plot title must contain 'Value Counts'"
                t.got = f"Title was: '{battery.result.get_title() if battery.result else 'N/A'}'"
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
# Exercise 4 – compute_categorical_summary
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the compute_categorical_summary function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_categorical_summary must be a function"
            return [t]
        cases.append(t)

        try:
            battery = ComputeCategoricalSummaryBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "compute_categorical_summary must return a pd.DataFrame"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_n_unique_column()
            if failed:
                t.failed = True
                t.msg = "Returned DataFrame must have an 'n_unique' column"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_most_frequent_column()
            if failed:
                t.failed = True
                t.msg = "Returned DataFrame must have a 'most_frequent' column"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_at_least_one_row()
            if failed:
                t.failed = True
                t.msg = "Returned DataFrame must have at least one row"
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.region_in_index()
            if failed:
                t.failed = True
                t.msg = "'region' must appear in the DataFrame index (it is a categorical column)"
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
# Exercise 5 – plot_proportion_chart
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the plot_proportion_chart function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_proportion_chart must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotProportionChartBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "plot_proportion_chart must return a matplotlib Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_enough_patches()
            if failed:
                t.failed = True
                t.msg = "The horizontal bar chart must have at least 5 bars (one per region)"
                t.want = f">= {want}"
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.xlabel_contains_proportion()
            if failed:
                t.failed = True
                t.msg = "The x-axis label must contain 'Proportion'"
                t.got = f"xlabel was: '{battery.result.get_xlabel() if battery.result else 'N/A'}'"
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
