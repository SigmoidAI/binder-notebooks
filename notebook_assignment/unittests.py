"""
Unit tests for the Automated Data Validation assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

import pandas as pd

from unittests_utils import (
    ValidateNoNullsBattery,
    ValidateRangeBattery,
    ValidateUniquenessBattery,
    ValidateRegexBattery,
    CreateValidationReportBattery,
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
# Exercise 1 – validate_no_nulls
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the validate_no_nulls function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg  = "validate_no_nulls must be a function"
            t.want = "a Python function"
            t.got  = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = ValidateNoNullsBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dict_check()
            if failed:
                t.failed = True
                t.msg  = "validate_no_nulls must return a dict"
                t.want = "dict"
                t.got  = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.a_passed_check()
            if failed:
                t.failed = True
                t.msg  = "Column 'a' (contains None) should have passed=False"
                t.want = str(want)
                t.got  = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.a_null_count_check()
            if failed:
                t.failed = True
                t.msg  = "Column 'a' should report null_count=1"
                t.want = f"{want} null(s)"
                t.got  = f"{got} null(s)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.b_passed_check()
            if failed:
                t.failed = True
                t.msg  = "Column 'b' (no nulls) should have passed=True"
                t.want = str(want)
                t.got  = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.b_null_count_check()
            if failed:
                t.failed = True
                t.msg  = "Column 'b' should report null_count=0"
                t.want = f"{want} null(s)"
                t.got  = f"{got} null(s)"
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
# Exercise 2 – validate_range
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the validate_range function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg  = "validate_range must be a function"
            t.want = "a Python function"
            t.got  = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = ValidateRangeBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dict_check()
            if failed:
                t.failed = True
                t.msg  = "validate_range must return a dict"
                t.want = "dict"
                t.got  = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_required_keys_check()
            if failed:
                t.failed = True
                t.msg  = "Result dict must contain keys: 'passed', 'violations', 'violation_pct'"
                t.want = "{'passed', 'violations', 'violation_pct'} ⊆ result.keys()"
                t.got  = (
                    str(set(battery.result.keys()))
                    if battery.result is not None
                    else "None"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.violations_check()
            if failed:
                t.failed = True
                t.msg  = "scores [10, 50, 110, -5, 70] with range [0, 100] should yield 2 violations (110 and -5)"
                t.want = f"{want} violation(s)"
                t.got  = f"{got} violation(s)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.passed_check()
            if failed:
                t.failed = True
                t.msg  = "Result should have passed=False because there are range violations"
                t.want = str(want)
                t.got  = str(got)
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
# Exercise 3 – validate_uniqueness
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the validate_uniqueness function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg  = "validate_uniqueness must be a function"
            t.want = "a Python function"
            t.got  = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = ValidateUniquenessBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dict_check()
            if failed:
                t.failed = True
                t.msg  = "validate_uniqueness must return a dict"
                t.want = "dict"
                t.got  = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.passed_check()
            if failed:
                t.failed = True
                t.msg  = "Column 'id' has duplicates so passed should be False"
                t.want = str(want)
                t.got  = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.duplicate_count_check()
            if failed:
                t.failed = True
                t.msg  = "id=[1,2,2,3,4] has 1 duplicate value (the extra '2'), so duplicate_count should be 1"
                t.want = f"{want} duplicate(s)"
                t.got  = f"{got} duplicate(s)"
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
# Exercise 4 – validate_regex
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the validate_regex function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg  = "validate_regex must be a function"
            t.want = "a Python function"
            t.got  = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = ValidateRegexBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dict_check()
            if failed:
                t.failed = True
                t.msg  = "validate_regex must return a dict"
                t.want = "dict"
                t.got  = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.violations_check()
            if failed:
                t.failed = True
                t.msg  = (
                    "emails=['a@b.com', 'invalid', 'c@d.org'] with pattern r'.+@.+\\..+' "
                    "should yield 1 violation ('invalid')"
                )
                t.want = f"{want} violation(s)"
                t.got  = f"{got} violation(s)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.passed_check()
            if failed:
                t.failed = True
                t.msg  = "Result should have passed=False because 'invalid' does not match the email pattern"
                t.want = str(want)
                t.got  = str(got)
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
# Exercise 5 – create_validation_report
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the create_validation_report function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg  = "create_validation_report must be a function"
            t.want = "a Python function"
            t.got  = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = CreateValidationReportBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_list_check()
            if failed:
                t.failed = True
                t.msg  = "create_validation_report must return a list"
                t.want = "list"
                t.got  = str(type(battery.result))
            cases.append(t)

            t = test_case()
            got, want, failed = battery.length_check()
            if failed:
                t.failed = True
                t.msg  = "Report must contain one entry per rule (2 rules → 2 entries)"
                t.want = f"{want} item(s)"
                t.got  = f"{got} item(s)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.items_have_required_keys_check()
            if failed:
                t.failed = True
                t.msg  = "Each report item must have keys: 'rule', 'column', 'passed', 'violations'"
                t.want = "{'rule', 'column', 'passed', 'violations'} ⊆ item.keys()"
                t.got  = (
                    str([set(item.keys()) for item in battery.result])
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
