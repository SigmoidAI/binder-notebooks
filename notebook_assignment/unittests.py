"""
Unit tests for the Missing Value Audit assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

import pandas as pd

from unittests_utils import (
    CountMissingBattery,
    MissingPercentageBattery,
    MissingPatternsBattery,
    MissingSummaryBattery,
    HighMissingnessBattery,
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
            print(f"\n  X {case.msg}")
            if case.want:
                print(f"     Expected: {case.want}")
            if case.got:
                print(f"     Got: {case.got}")


def exercise_1(learner_func):
    """Test the count_missing_values function."""
    def g():
        cases: List[test_case] = []

        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "count_missing_values must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = CountMissingBattery(learner_func)

            # Check return type
            t = test_case()
            got, want, failed = battery.result_is_series()
            if failed:
                t.failed = True
                t.msg = "Function must return a pandas Series"
                t.want = "pandas Series"
                t.got = str(type(battery.result))
            cases.append(t)

            # Check count for column A
            t = test_case()
            got, want, failed = battery.count_A_check()
            if failed:
                t.failed = True
                t.msg = "Missing count for column A is incorrect"
                t.want = f"{want}"
                t.got = f"{got}"
            cases.append(t)

            # Check count for column B
            t = test_case()
            got, want, failed = battery.count_B_check()
            if failed:
                t.failed = True
                t.msg = "Missing count for column B is incorrect"
                t.want = f"{want}"
                t.got = f"{got}"
            cases.append(t)

            # Check count for column C (no missing)
            t = test_case()
            got, want, failed = battery.count_C_check()
            if failed:
                t.failed = True
                t.msg = "Missing count for column C (no missing values) is incorrect"
                t.want = f"{want}"
                t.got = f"{got}"
            cases.append(t)

            # Check count for column D
            t = test_case()
            got, want, failed = battery.count_D_check()
            if failed:
                t.failed = True
                t.msg = "Missing count for column D is incorrect"
                t.want = f"{want}"
                t.got = f"{got}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = "An exception was raised when testing count_missing_values"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    """Test the calculate_missing_percentage function."""
    def g():
        cases: List[test_case] = []

        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "calculate_missing_percentage must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = MissingPercentageBattery(learner_func)

            # Check return type
            t = test_case()
            got, want, failed = battery.result_is_series()
            if failed:
                t.failed = True
                t.msg = "Function must return a pandas Series"
                t.want = "pandas Series"
                t.got = str(type(battery.result))
            cases.append(t)

            # Check percentage for column A (10%)
            t = test_case()
            got, want, failed = battery.pct_A_check()
            if failed:
                t.failed = True
                t.msg = "Missing percentage for column A is incorrect"
                t.want = f"{want}%"
                t.got = f"{got}"
            cases.append(t)

            # Check percentage for column B (20%)
            t = test_case()
            got, want, failed = battery.pct_B_check()
            if failed:
                t.failed = True
                t.msg = "Missing percentage for column B is incorrect"
                t.want = f"{want}%"
                t.got = f"{got}"
            cases.append(t)

            # Check percentage for column C (0%)
            t = test_case()
            got, want, failed = battery.pct_C_check()
            if failed:
                t.failed = True
                t.msg = "Missing percentage for column C (no missing) is incorrect"
                t.want = f"{want}%"
                t.got = f"{got}"
            cases.append(t)

            # Check percentage for column D (50%)
            t = test_case()
            got, want, failed = battery.pct_D_check()
            if failed:
                t.failed = True
                t.msg = "Missing percentage for column D is incorrect"
                t.want = f"{want}%"
                t.got = f"{got}"
            cases.append(t)

            # Check values are percentages (0-100)
            t = test_case()
            got, want, failed = battery.values_are_percentages()
            if failed:
                t.failed = True
                t.msg = "Percentage values should be between 0 and 100"
                t.want = "values in range [0, 100]"
                t.got = f"some values out of range"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = "An exception was raised when testing calculate_missing_percentage"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    """Test the identify_missing_patterns function."""
    def g():
        cases: List[test_case] = []

        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "identify_missing_patterns must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = MissingPatternsBattery(learner_func)

            # Check return type
            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "Function must return a pandas DataFrame"
                t.want = "pandas DataFrame"
                t.got = str(type(battery.result))
            cases.append(t)

            # Check for required columns
            t = test_case()
            got, want, failed = battery.has_required_columns()
            if failed:
                t.failed = True
                t.msg = "Result should have columns describing missing patterns"
                t.want = "column with 'pattern' or 'missing' in name"
                t.got = f"columns: {list(battery.result.columns) if battery.result is not None else None}"
            cases.append(t)

            # Check that missing rows are identified
            t = test_case()
            got, want, failed = battery.identifies_missing_rows()
            if failed:
                t.failed = True
                t.msg = "Function should identify rows with missing values"
                t.want = "at least 3 rows with missing patterns"
                t.got = f"{len(battery.result) if battery.result is not None else 0} rows identified"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = "An exception was raised when testing identify_missing_patterns"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    """Test the create_missing_summary function."""
    def g():
        cases: List[test_case] = []

        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_missing_summary must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = MissingSummaryBattery(learner_func)

            # Check return type
            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "Function must return a pandas DataFrame"
                t.want = "pandas DataFrame"
                t.got = str(type(battery.result))
            cases.append(t)

            # Check for count information
            t = test_case()
            got, want, failed = battery.has_count_info()
            if failed:
                t.failed = True
                t.msg = "Summary should include missing count information"
                t.want = "column with 'count' or 'missing' in name"
                t.got = f"columns: {list(battery.result.columns) if battery.result is not None else None}"
            cases.append(t)

            # Check for percentage information
            t = test_case()
            got, want, failed = battery.has_percentage_info()
            if failed:
                t.failed = True
                t.msg = "Summary should include percentage information"
                t.want = "column with 'percent', 'pct', or '%' in name"
                t.got = f"columns: {list(battery.result.columns) if battery.result is not None else None}"
            cases.append(t)

            # Check that all columns are covered
            t = test_case()
            got, want, failed = battery.covers_all_columns()
            if failed:
                t.failed = True
                t.msg = "Summary should cover all columns in the input DataFrame"
                t.want = "at least 4 entries (one per column)"
                t.got = f"{len(battery.result) if battery.result is not None else 0} entries"
            cases.append(t)

            # Check correct count values
            t = test_case()
            got, want, failed = battery.correct_count_values()
            if failed:
                t.failed = True
                t.msg = "Missing count values are incorrect"
                t.want = "total missing count of 8"
                t.got = f"got different total"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = "An exception was raised when testing create_missing_summary"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    """Test the detect_high_missingness function."""
    def g():
        cases: List[test_case] = []

        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "detect_high_missingness must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = HighMissingnessBattery(learner_func)

            # Check return type
            t = test_case()
            got, want, failed = battery.result_is_list()
            if failed:
                t.failed = True
                t.msg = "Function must return a list of column names"
                t.want = "list"
                t.got = str(type(battery.result_20))
            cases.append(t)

            # Check threshold 20% results
            t = test_case()
            got, want, failed = battery.threshold_20_check()
            if failed:
                t.failed = True
                t.msg = "Columns above 20% threshold are incorrect"
                t.want = f"{want}"
                t.got = f"{got}"
            cases.append(t)

            # Check threshold 50% results
            t = test_case()
            got, want, failed = battery.threshold_50_check()
            if failed:
                t.failed = True
                t.msg = "Columns above 50% threshold are incorrect"
                t.want = f"{want}"
                t.got = f"{got}"
            cases.append(t)

            # Check that low missing columns are excluded
            t = test_case()
            got, want, failed = battery.excludes_low_missing()
            if failed:
                t.failed = True
                t.msg = "Columns with low missingness should be excluded"
                t.want = "columns A and C excluded from 20% threshold results"
                t.got = f"found A or C in results"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = "An exception was raised when testing detect_high_missingness"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)
