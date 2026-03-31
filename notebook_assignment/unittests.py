"""
Unit tests for the Data Type Conversion assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

import pandas as pd

from unittests_utils import (
    ConvertToNumericBattery,
    ConvertToDatetimeBattery,
    ConvertToCategoricalBattery,
    SafeNumericConversionBattery,
    ConversionReportBattery,
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
# Exercise 1 – convert_to_numeric
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the convert_to_numeric function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "convert_to_numeric must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = ConvertToNumericBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_series()
            if failed:
                t.failed = True
                t.msg = "convert_to_numeric must return a pandas Series"
                t.want = "pandas Series"
                t.got = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.result_is_numeric()
            if failed:
                t.failed = True
                t.msg = "Returned Series must have a numeric dtype"
                t.want = "numeric dtype (e.g. float64)"
                t.got = str(battery.result.dtype) if battery.result is not None else "None"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.first_value_check()
            if failed:
                t.failed = True
                t.msg = "First value '1.0' should convert to the float 1.0"
                t.want = f"{want}"
                t.got = f"{got}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.abc_is_nan_check()
            if failed:
                t.failed = True
                t.msg = "Non-numeric string 'abc' should be coerced to NaN"
                t.want = "NaN at position 2"
                t.got = f"{got}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.nan_count_check()
            if failed:
                t.failed = True
                t.msg = "Both 'abc' and 'N/A' should result in NaN (2 total)"
                t.want = f"{want} NaN value(s)"
                t.got = f"{got} NaN value(s)"
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
# Exercise 2 – convert_to_datetime
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the convert_to_datetime function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "convert_to_datetime must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = ConvertToDatetimeBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_series()
            if failed:
                t.failed = True
                t.msg = "convert_to_datetime must return a pandas Series"
                t.want = "pandas Series"
                t.got = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.result_is_datetime()
            if failed:
                t.failed = True
                t.msg = "Returned Series must have datetime64 dtype"
                t.want = "datetime64[ns] dtype"
                t.got = str(battery.result.dtype) if battery.result is not None else "None"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.invalid_is_nat_check()
            if failed:
                t.failed = True
                t.msg = "Invalid date string 'not-a-date' should become NaT"
                t.want = "NaT at position 2"
                t.got = f"{got}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.valid_dates_not_nat()
            if failed:
                t.failed = True
                t.msg = "Valid dates '2023-01-15' and '2023-06-30' must NOT be NaT"
                t.want = "0 NaT among valid dates"
                t.got = f"{got} NaT value(s) among valid dates"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.nat_count_check()
            if failed:
                t.failed = True
                t.msg = "Exactly 1 NaT expected (from 'not-a-date')"
                t.want = f"{want} NaT value(s)"
                t.got = f"{got} NaT value(s)"
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
# Exercise 3 – convert_to_categorical
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the convert_to_categorical function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "convert_to_categorical must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = ConvertToCategoricalBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_series()
            if failed:
                t.failed = True
                t.msg = "convert_to_categorical must return a pandas Series"
                t.want = "pandas Series"
                t.got = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.result_is_categorical()
            if failed:
                t.failed = True
                t.msg = "Returned Series must have CategoricalDtype"
                t.want = "CategoricalDtype"
                t.got = str(battery.result.dtype) if battery.result is not None else "None"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.no_nan_introduced_check()
            if failed:
                t.failed = True
                t.msg = "No NaN values should be introduced during categorical conversion"
                t.want = f"{want} NaN value(s)"
                t.got = f"{got} NaN value(s)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.n_categories_check()
            if failed:
                t.failed = True
                t.msg = "Categorical column should have 3 unique categories: 'low', 'medium', 'high'"
                t.want = f"{want} categories"
                t.got = f"{got} categories"
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
# Exercise 4 – safe_numeric_conversion
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the safe_numeric_conversion function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "safe_numeric_conversion must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = SafeNumericConversionBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_tuple()
            if failed:
                t.failed = True
                t.msg = "safe_numeric_conversion must return a tuple (df_converted, error_counts)"
                t.want = "tuple"
                t.got = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.first_element_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "First element of the returned tuple must be a DataFrame"
                t.want = "pandas DataFrame"
                t.got = (
                    str(type(battery.result[0]))
                    if isinstance(battery.result, tuple)
                    else "N/A"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.second_element_is_dict()
            if failed:
                t.failed = True
                t.msg = "Second element of the returned tuple must be a dict of error counts"
                t.want = "dict"
                t.got = (
                    str(type(battery.result[1]))
                    if isinstance(battery.result, tuple)
                    else "N/A"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.a_errors_check()
            if failed:
                t.failed = True
                t.msg = "Column 'a' contains 'x' which is non-numeric → 1 error expected"
                t.want = f"{want} error(s) for column 'a'"
                t.got = f"{got} error(s)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.b_errors_check()
            if failed:
                t.failed = True
                t.msg = "Column 'b' contains only valid numbers → 0 errors expected"
                t.want = f"{want} error(s) for column 'b'"
                t.got = f"{got} error(s)"
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
# Exercise 5 – create_conversion_report
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the create_conversion_report function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_conversion_report must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = ConversionReportBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dict()
            if failed:
                t.failed = True
                t.msg = "create_conversion_report must return a dict"
                t.want = "dict"
                t.got = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_dtype_changes_key()
            if failed:
                t.failed = True
                t.msg = "Result dict must contain the key 'dtype_changes'"
                t.want = "key 'dtype_changes' present"
                t.got = (
                    f"keys present: {list(battery.result.keys())}"
                    if isinstance(battery.result, dict)
                    else "N/A"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_columns_converted_key()
            if failed:
                t.failed = True
                t.msg = "Result dict must contain the key 'columns_converted'"
                t.want = "key 'columns_converted' present"
                t.got = (
                    f"keys present: {list(battery.result.keys())}"
                    if isinstance(battery.result, dict)
                    else "N/A"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.dtype_changes_is_list()
            if failed:
                t.failed = True
                t.msg = "'dtype_changes' value must be a list"
                t.want = "list"
                t.got = (
                    str(type(battery.result.get("dtype_changes")))
                    if isinstance(battery.result, dict)
                    else "N/A"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.columns_converted_at_least_one()
            if failed:
                t.failed = True
                t.msg = (
                    f"'columns_converted' must be >= {want} "
                    "(at least two object columns become numeric)"
                )
                t.want = f">= {want}"
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
