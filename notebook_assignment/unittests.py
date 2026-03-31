"""
Unit tests for the Distribution Analysis assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

from unittests_utils import (
    ComputeDistributionStatsBattery,
    ClassifyDistributionBattery,
    PlotDistributionTripleBattery,
    SummarizeAllNumericBattery,
    DetectSkewedColumnsBattery,
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
# Exercise 1 – compute_distribution_stats
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the compute_distribution_stats function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_distribution_stats must be a function"
            return [t]
        cases.append(t)

        try:
            battery = ComputeDistributionStatsBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dict()
            if failed:
                t.failed = True
                t.msg = "compute_distribution_stats must return a dict"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.all_keys_present()
            if failed:
                t.failed = True
                t.msg = "Returned dict is missing expected keys"
                t.want = str(battery.expected_keys)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.mean_is_correct()
            if failed:
                t.failed = True
                t.msg = "The 'mean' value is incorrect"
                t.want = str(round(want, 4))
                t.got = str(round(got, 4)) if got is not None else "None"
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
# Exercise 2 – classify_distribution
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the classify_distribution function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "classify_distribution must be a function"
            return [t]
        cases.append(t)

        try:
            battery = ClassifyDistributionBattery(learner_func)

            t = test_case()
            got, want, failed = battery.strong_right_skew_correct()
            if failed:
                t.failed = True
                t.msg = "classify_distribution(1.7, 3.8) returned wrong classification"
                t.want = want
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.symmetric_correct()
            if failed:
                t.failed = True
                t.msg = "classify_distribution(0.05, -0.2) returned wrong classification"
                t.want = want
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
# Exercise 3 – plot_distribution_triple
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the plot_distribution_triple function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "plot_distribution_triple must be a function"
            return [t]
        cases.append(t)

        try:
            battery = PlotDistributionTripleBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_tuple()
            if failed:
                t.failed = True
                t.msg = "plot_distribution_triple must return a tuple (fig, axes)"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.tuple_has_two_elements()
            if failed:
                t.failed = True
                t.msg = "Returned tuple must contain exactly 2 elements (fig, axes)"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.first_element_is_figure()
            if failed:
                t.failed = True
                t.msg = "First element of the tuple must be a matplotlib Figure"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.axes_has_three_panels()
            if failed:
                t.failed = True
                t.msg = "The axes array must contain exactly 3 panels"
                t.want = f"{want} panels"
                t.got = f"{got} panels"
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
# Exercise 4 – summarize_all_numeric
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the summarize_all_numeric function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "summarize_all_numeric must be a function"
            return [t]
        cases.append(t)

        try:
            battery = SummarizeAllNumericBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "summarize_all_numeric must return a pd.DataFrame"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_skewness_column()
            if failed:
                t.failed = True
                t.msg = "Returned DataFrame must have a 'skewness' column"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_iqr_column()
            if failed:
                t.failed = True
                t.msg = "Returned DataFrame must have an 'iqr' column"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_enough_rows()
            if failed:
                t.failed = True
                t.msg = "Returned DataFrame must have at least 4 rows (one per numeric column)"
                t.got = f"{got} rows"
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
# Exercise 5 – detect_skewed_columns
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the detect_skewed_columns function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "detect_skewed_columns must be a function"
            return [t]
        cases.append(t)

        try:
            battery = DetectSkewedColumnsBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_list()
            if failed:
                t.failed = True
                t.msg = "detect_skewed_columns must return a list"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.order_value_in_result()
            if failed:
                t.failed = True
                t.msg = "'order_value' must be in the result (skewness \u2248 1.66 > 1.0)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.customer_age_not_in_result()
            if failed:
                t.failed = True
                t.msg = "'customer_age' must NOT be in the result (skewness \u2248 0.05 < 1.0)"
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
