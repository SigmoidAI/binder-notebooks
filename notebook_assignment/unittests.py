"""
Unit tests for the Target Variable Deep Dive assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

from unittests_utils import (
    GetClassBalanceBattery,
    ClassifyImbalanceBattery,
    PlotTargetDistributionBattery,
    ComputeTargetStatsBattery,
    DetectTargetAnomaliesBattery,
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
# Exercise 1 – get_class_balance
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the get_class_balance function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "get_class_balance must be a function"
            return [t]
        cases.append(t)

        try:
            battery = GetClassBalanceBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "get_class_balance must return a pd.DataFrame"
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
            got, want, failed = battery.has_two_rows()
            if failed:
                t.failed = True
                t.msg = "DataFrame must have exactly 2 rows (one per target class)"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.majority_class_proportion_above_threshold()
            if failed:
                t.failed = True
                t.msg = "Majority class (returned=0) proportion must be > 0.7"
                t.want = f"> {want}"
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
# Exercise 2 – classify_imbalance
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the classify_imbalance function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "classify_imbalance must be a function"
            return [t]
        cases.append(t)

        try:
            battery = ClassifyImbalanceBattery(learner_func)

            t = test_case()
            got, want, failed = battery.severe_returns_correct_string()
            if failed:
                t.failed = True
                t.msg = "classify_imbalance(0.18) must return 'severe_imbalance'"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.balanced_returns_correct_string()
            if failed:
                t.failed = True
                t.msg = "classify_imbalance(0.45) must return 'balanced'"
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
# Exercise 3 – plot_target_distribution
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the plot_target_distribution function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_target_distribution must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotTargetDistributionBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "plot_target_distribution must return a matplotlib Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_enough_patches()
            if failed:
                t.failed = True
                t.msg = "The bar chart must have at least 2 bars (one per target class)"
                t.want = f">= {want}"
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.title_contains_class_distribution()
            if failed:
                t.failed = True
                t.msg = "The plot title must contain 'Class Distribution'"
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
# Exercise 4 – compute_target_stats
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the compute_target_stats function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_target_stats must be a function"
            return [t]
        cases.append(t)

        try:
            battery = ComputeTargetStatsBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "compute_target_stats must return a pd.DataFrame"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_correct_shape()
            if failed:
                t.failed = True
                t.msg = "DataFrame must have shape (2, 2) for 2 target classes × 2 numeric columns"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_order_value_column()
            if failed:
                t.failed = True
                t.msg = "Returned DataFrame must contain 'order_value' column"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_customer_age_column()
            if failed:
                t.failed = True
                t.msg = "Returned DataFrame must contain 'customer_age' column"
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
# Exercise 5 – detect_target_anomalies
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the detect_target_anomalies function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "detect_target_anomalies must be a function"
            return [t]
        cases.append(t)

        try:
            battery = DetectTargetAnomaliesBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dict()
            if failed:
                t.failed = True
                t.msg = "detect_target_anomalies must return a dict"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_null_count_key()
            if failed:
                t.failed = True
                t.msg = "Returned dict must have a 'null_count' key"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.null_count_is_zero()
            if failed:
                t.failed = True
                t.msg = "null_count must be 0 for the clean orders DataFrame"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.unexpected_count_is_zero()
            if failed:
                t.failed = True
                t.msg = "unexpected_count must be 0 — all values in 'returned' are 0 or 1"
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
