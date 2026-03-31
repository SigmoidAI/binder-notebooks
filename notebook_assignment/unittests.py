"""
Unit tests for the Feature-Target Relationship Mapping assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

from unittests_utils import (
    ComputeFeatureTargetCorrelationsBattery,
    PlotFeatureTargetCorrelationsBattery,
    ComputeGroupMeansBattery,
    PlotKdeByTargetBattery,
    BuildRelationshipSummaryBattery,
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
# Exercise 1 – compute_feature_target_correlations
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the compute_feature_target_correlations function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_feature_target_correlations must be a function"
            return [t]
        cases.append(t)

        try:
            battery = ComputeFeatureTargetCorrelationsBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_series()
            if failed:
                t.failed = True
                t.msg = "compute_feature_target_correlations must return a pd.Series"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.length_is_correct()
            if failed:
                t.failed = True
                t.msg = "Returned Series must have one entry per numeric column (length 3)"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.index_contains_order_value()
            if failed:
                t.failed = True
                t.msg = "Returned Series index must contain 'order_value'"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.is_sorted_by_abs_descending()
            if failed:
                t.failed = True
                t.msg = "Series must be sorted by absolute value descending (strongest correlation first)"
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
# Exercise 2 – plot_feature_target_correlations
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the plot_feature_target_correlations function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_feature_target_correlations must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotFeatureTargetCorrelationsBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "plot_feature_target_correlations must return a matplotlib Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_enough_patches()
            if failed:
                t.failed = True
                t.msg = "plot_feature_target_correlations must draw at least 3 bars (one per numeric column)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.title_contains_correlations()
            if failed:
                t.failed = True
                t.msg = "Plot title must contain the word 'Correlations'"
                t.got = f"Got title: '{battery.result.get_title() if battery.result else None}'"
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
# Exercise 3 – compute_group_means
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the compute_group_means function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_group_means must be a function"
            return [t]
        cases.append(t)

        try:
            battery = ComputeGroupMeansBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_series()
            if failed:
                t.failed = True
                t.msg = "compute_group_means must return a pd.Series"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.length_is_correct()
            if failed:
                t.failed = True
                t.msg = "Returned Series must have one entry per region (length 5)"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.first_index_is_highest_mean()
            if failed:
                t.failed = True
                t.msg = "Series must be sorted descending — first element should be the region with the highest return rate"
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
# Exercise 4 – plot_kde_by_target
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the plot_kde_by_target function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_kde_by_target must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotKdeByTargetBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "plot_kde_by_target must return a matplotlib Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.title_contains_kde()
            if failed:
                t.failed = True
                t.msg = "Plot title must contain 'KDE'"
                t.got = f"Got title: '{battery.result.get_title() if battery.result else None}'"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_at_least_two_lines()
            if failed:
                t.failed = True
                t.msg = "Plot must contain at least 2 lines (one KDE curve per target class)"
                t.got = f"Got {len(battery.result.lines) if battery.result else 0} lines"
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
# Exercise 5 – build_relationship_summary
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the build_relationship_summary function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "build_relationship_summary must be a function"
            return [t]
        cases.append(t)

        try:
            battery = BuildRelationshipSummaryBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "build_relationship_summary must return a pd.DataFrame"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.shape_is_correct()
            if failed:
                t.failed = True
                t.msg = "DataFrame must have shape (3, 4) — 3 features × 4 metric columns"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_correlation_column()
            if failed:
                t.failed = True
                t.msg = "DataFrame must contain column 'correlation_with_target'"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_mean_diff_column()
            if failed:
                t.failed = True
                t.msg = "DataFrame must contain column 'mean_diff'"
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
