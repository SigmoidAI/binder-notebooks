"""
Unit tests for the Inconsistency Resolution assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

import pandas as pd

from unittests_utils import (
    StandardizeCountryBattery,
    ParseCurrencyBattery,
    StandardizeDateBattery,
    ParseWeightBattery,
    StandardizeTextBattery,
)


class test_case:
    """Simple test case class for tracking test results."""

    def __init__(self):
        self.failed = False
        self.msg    = ""
        self.want   = ""
        self.got    = ""


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
# Exercise 1 – standardize_country
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the standardize_country function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg  = "standardize_country must be a function"
            t.want = "a Python function"
            t.got  = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = StandardizeCountryBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_series()
            if failed:
                t.failed = True
                t.msg  = "standardize_country must return a pandas Series"
                t.want = "pandas Series"
                t.got  = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.length_preserved_check()
            if failed:
                t.failed = True
                t.msg  = "Output Series length must match input Series length"
                t.want = f"{want} row(s)"
                t.got  = f"{got} row(s)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.all_values_standardized_check()
            if failed:
                t.failed = True
                t.msg  = "All USA variants must map to a single standardized value (1 unique value expected)"
                t.want = f"{want} unique value(s)"
                t.got  = f"{got} unique value(s)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.value_is_united_states_check()
            if failed:
                t.failed = True
                t.msg  = "All USA variants must map to 'United States'"
                t.want = f"'{want}'"
                t.got  = f"'{got}'"
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
# Exercise 2 – parse_currency
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the parse_currency function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg  = "parse_currency must be a function"
            t.want = "a Python function"
            t.got  = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = ParseCurrencyBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_series()
            if failed:
                t.failed = True
                t.msg  = "parse_currency must return a pandas Series"
                t.want = "pandas Series"
                t.got  = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.result_is_numeric()
            if failed:
                t.failed = True
                t.msg  = "Returned Series must have a numeric dtype (float64)"
                t.want = "numeric dtype (e.g. float64)"
                t.got  = str(battery.result.dtype) if battery.result is not None else "None"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.first_value_check()
            if failed:
                t.failed = True
                t.msg  = "'$1,200.50' should parse to the float 1200.50"
                t.want = f"{want}"
                t.got  = f"{got}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.length_preserved_check()
            if failed:
                t.failed = True
                t.msg  = "Output Series length must match input Series length"
                t.want = f"{want} row(s)"
                t.got  = f"{got} row(s)"
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
# Exercise 3 – standardize_date_format
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the standardize_date_format function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg  = "standardize_date_format must be a function"
            t.want = "a Python function"
            t.got  = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = StandardizeDateBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_series()
            if failed:
                t.failed = True
                t.msg  = "standardize_date_format must return a pandas Series"
                t.want = "pandas Series"
                t.got  = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.length_preserved_check()
            if failed:
                t.failed = True
                t.msg  = "Output Series length must match input Series length"
                t.want = f"{want} row(s)"
                t.got  = f"{got} row(s)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.first_value_check()
            if failed:
                t.failed = True
                t.msg  = "'2023-01-15' (already correct format) should remain '2023-01-15'"
                t.want = f"'{want}'"
                t.got  = f"'{got}'"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.valid_format_check()
            if failed:
                t.failed = True
                t.msg  = "'01/15/2023' should be converted to YYYY-MM-DD format"
                t.want = "string matching YYYY-MM-DD"
                t.got  = (
                    str(battery.result.iloc[1])
                    if battery.result is not None
                    else "None"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.invalid_date_handled_check()
            if failed:
                t.failed = True
                t.msg  = "'not-a-date' should result in NaN/NaT or an empty string"
                t.want = "NaN, NaT, or empty string"
                t.got  = (
                    str(battery.result.iloc[2])
                    if battery.result is not None
                    else "None"
                )
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
# Exercise 4 – parse_weight
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the parse_weight function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg  = "parse_weight must be a function"
            t.want = "a Python function"
            t.got  = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = ParseWeightBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_series()
            if failed:
                t.failed = True
                t.msg  = "parse_weight must return a pandas Series"
                t.want = "pandas Series"
                t.got  = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.first_value_check()
            if failed:
                t.failed = True
                t.msg  = "'72.5kg' should parse to the float 72.5"
                t.want = f"{want}"
                t.got  = f"{got}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.second_value_check()
            if failed:
                t.failed = True
                t.msg  = "'80.0 KG' should parse to the float 80.0"
                t.want = f"{want}"
                t.got  = f"{got}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.empty_string_is_nan_check()
            if failed:
                t.failed = True
                t.msg  = "Empty string '' should result in NaN"
                t.want = "NaN at position 3"
                t.got  = (
                    str(battery.result.iloc[3])
                    if battery.result is not None
                    else "None"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.abc_is_nan_check()
            if failed:
                t.failed = True
                t.msg  = "Non-numeric 'abc' should result in NaN"
                t.want = "NaN at position 4"
                t.got  = (
                    str(battery.result.iloc[4])
                    if battery.result is not None
                    else "None"
                )
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
# Exercise 5 – standardize_text_case
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the standardize_text_case function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg  = "standardize_text_case must be a function"
            t.want = "a Python function"
            t.got  = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = StandardizeTextBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_series()
            if failed:
                t.failed = True
                t.msg  = "standardize_text_case must return a pandas Series"
                t.want = "pandas Series"
                t.got  = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.length_preserved_check()
            if failed:
                t.failed = True
                t.msg  = "Output Series length must match input Series length"
                t.want = f"{want} row(s)"
                t.got  = f"{got} row(s)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.all_lowercase_check()
            if failed:
                t.failed = True
                t.msg  = "With case='lower', all values must be lowercase"
                t.want = "all lowercase values"
                t.got  = (
                    str(battery.result.tolist())
                    if battery.result is not None
                    else "None"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.n_unique_check()
            if failed:
                t.failed = True
                t.msg  = "After lowercasing, 'Active'/'ACTIVE'/'active' collapse to 1 value → 2 unique values total"
                t.want = f"{want} unique value(s)"
                t.got  = f"{got} unique value(s)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.unique_values_check()
            if failed:
                t.failed = True
                t.msg  = "Unique values must be exactly {'active', 'inactive'}"
                t.want = f"{want}"
                t.got  = f"{got}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)
