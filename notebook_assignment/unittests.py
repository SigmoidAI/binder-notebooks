"""
Unit tests for the Outlier Investigation assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

import pandas as pd

from unittests_utils import (
    DetectIQROutliersBattery,
    DetectZScoreOutliersBattery,
    DetectIsolationForestBattery,
    RemoveOutliersBattery,
    OutlierSummaryBattery,
)


class test_case:
    """Simple test case class for tracking test results."""

    def __init__(self):
        self.failed = False
        self.msg = ""
        self.want = ""
        self.got = ""


def print_feedback(cases: List[test_case]):
    """Print feedback for all test cases."""
    failed_cases = [c for c in cases if c.failed]

    if not failed_cases:
        print("\033[92m" + "All tests passed!" + "\033[0m")
    else:
        print("\033[91m" + f"Failed {len(failed_cases)} of {len(cases)} tests:" + "\033[0m")
        for case in failed_cases:
            print(f"\n  - {case.msg}")
            if case.want:
                print(f"     Expected: {case.want}")
            if case.got:
                print(f"     Got: {case.got}")


# ---------------------------------------------------------------------------
# Exercise 1 – detect_iqr_outliers
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the detect_iqr_outliers function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "detect_iqr_outliers must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = DetectIQROutliersBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_boolean_series()
            if failed:
                t.failed = True
                t.msg = "detect_iqr_outliers must return a boolean pandas Series"
                t.want = "pandas Series with dtype=bool"
                t.got = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_length()
            if failed:
                t.failed = True
                t.msg = "Result Series must have the same length as the input DataFrame"
                t.want = f"{want} elements"
                t.got = f"{got} elements"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.outlier_100_flagged()
            if failed:
                t.failed = True
                t.msg = "The extreme value 100 should be detected as an IQR outlier"
                t.want = "True for the value 100"
                t.got = f"{got}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.non_outliers_clean()
            if failed:
                t.failed = True
                t.msg = "Values 1–9 should not be flagged as outliers"
                t.want = f"0 outliers among values 1–9"
                t.got = f"{got} value(s) incorrectly flagged"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 2 – detect_zscore_outliers
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the detect_zscore_outliers function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "detect_zscore_outliers must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = DetectZScoreOutliersBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_boolean_series()
            if failed:
                t.failed = True
                t.msg = "detect_zscore_outliers must return a boolean pandas Series"
                t.want = "pandas Series with dtype=bool"
                t.got = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_length()
            if failed:
                t.failed = True
                t.msg = "Result Series must have the same length as the input DataFrame"
                t.want = f"{want} elements"
                t.got = f"{got} elements"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.outlier_100_flagged()
            if failed:
                t.failed = True
                t.msg = (
                    "The extreme value 100 should be detected as a z-score outlier "
                    "(tested at threshold=2.0)"
                )
                t.want = "True for the value 100"
                t.got = f"{got}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 3 – detect_isolation_forest_outliers
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the detect_isolation_forest_outliers function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "detect_isolation_forest_outliers must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = DetectIsolationForestBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_boolean_series()
            if failed:
                t.failed = True
                t.msg = "detect_isolation_forest_outliers must return a boolean pandas Series"
                t.want = "pandas Series with dtype=bool"
                t.got = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_length()
            if failed:
                t.failed = True
                t.msg = "Result Series must have the same length as the input DataFrame"
                t.want = f"{want} elements"
                t.got = f"{got} elements"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.extreme_rows_flagged()
            if failed:
                t.failed = True
                t.msg = "Extreme rows with values [100, -100] should be flagged as outliers"
                t.want = "At least one of the two extreme rows flagged True"
                t.got = f"{got}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 4 – remove_outliers
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the remove_outliers function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "remove_outliers must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = RemoveOutliersBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "remove_outliers must return a pandas DataFrame"
                t.want = "pandas DataFrame"
                t.got = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_row_count()
            if failed:
                t.failed = True
                t.msg = "Incorrect number of rows after removing outliers"
                t.want = f"{want} rows"
                t.got = f"{got} rows"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.index_is_reset()
            if failed:
                t.failed = True
                t.msg = "Result index should be reset (0, 1, 2, ...)"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 5 – create_outlier_summary
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the create_outlier_summary function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_outlier_summary must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = OutlierSummaryBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dict()
            if failed:
                t.failed = True
                t.msg = "create_outlier_summary must return a dictionary"
                t.want = "dict"
                t.got = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_required_keys()
            if failed:
                t.failed = True
                t.msg = (
                    "Result dictionary must contain keys: "
                    "'total_outliers', 'clean_rows', 'outlier_pct'"
                )
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.total_outliers_at_least_one()
            if failed:
                t.failed = True
                t.msg = "Expected at least 1 outlier detected in the test data"
                t.want = "total_outliers >= 1"
                t.got = f"total_outliers = {got}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)
