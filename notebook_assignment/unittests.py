"""
Unit tests for the Duplicate Detection & Removal assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

import pandas as pd

from unittests_utils import (
    DetectExactDuplicatesBattery,
    RemoveExactDuplicatesBattery,
    DetectFuzzyDuplicatesBattery,
    FlagDuplicatesBattery,
    DeduplicationReportBattery,
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
# Exercise 1 – detect_exact_duplicates
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the detect_exact_duplicates function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "detect_exact_duplicates must be a function"
            return [t]
        cases.append(t)

        try:
            battery = DetectExactDuplicatesBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "detect_exact_duplicates must return a pandas DataFrame"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_row_count()
            if failed:
                t.failed = True
                t.msg = "Incorrect number of duplicate rows detected"
                t.want = f"{want} rows"
                t.got = f"{got} rows"
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
# Exercise 2 – remove_exact_duplicates
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the remove_exact_duplicates function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "remove_exact_duplicates must be a function"
            return [t]
        cases.append(t)

        try:
            battery = RemoveExactDuplicatesBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "remove_exact_duplicates must return a pandas DataFrame"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_row_count()
            if failed:
                t.failed = True
                t.msg = "Incorrect number of rows after removing exact duplicates"
                t.want = f"{want} rows"
                t.got = f"{got} rows"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.no_exact_duplicates_remain()
            if failed:
                t.failed = True
                t.msg = "Result still contains exact duplicate rows"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.index_is_reset()
            if failed:
                t.failed = True
                t.msg = "Result index should be reset (0, 1, 2, ...)"
                t.want = str(want[:5]) + "..."
                t.got = str(got[:5]) + "..."
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
# Exercise 3 – detect_fuzzy_duplicates
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the detect_fuzzy_duplicates function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "detect_fuzzy_duplicates must be a function"
            return [t]
        cases.append(t)

        try:
            battery = DetectFuzzyDuplicatesBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "detect_fuzzy_duplicates must return a pandas DataFrame"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_required_columns()
            if failed:
                t.failed = True
                t.msg = "Result must have columns: index_1, index_2, value_1, value_2, similarity"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.finds_enough_pairs()
            if failed:
                t.failed = True
                t.msg = "Not enough fuzzy duplicate pairs detected"
                t.want = f"at least {want} pairs"
                t.got = f"{got} pairs"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.similarity_in_range()
            if failed:
                t.failed = True
                t.msg = "Similarity scores must be between 0 and 100"
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
# Exercise 4 – flag_duplicates
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the flag_duplicates function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "flag_duplicates must be a function"
            return [t]
        cases.append(t)

        try:
            battery = FlagDuplicatesBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "flag_duplicates must return a pandas DataFrame"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_is_duplicate_column()
            if failed:
                t.failed = True
                t.msg = "Result must contain a column named 'is_duplicate'"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_flagged_count()
            if failed:
                t.failed = True
                t.msg = "Incorrect number of rows flagged as duplicates"
                t.want = f"{want} rows flagged"
                t.got = f"{got} rows flagged"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.same_shape_as_input()
            if failed:
                t.failed = True
                t.msg = "Result must have the same number of rows as the input"
                t.want = f"{want} rows"
                t.got = f"{got} rows"
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
# Exercise 5 – create_deduplication_report
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the create_deduplication_report function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_deduplication_report must be a function"
            return [t]
        cases.append(t)

        try:
            battery = DeduplicationReportBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dict()
            if failed:
                t.failed = True
                t.msg = "create_deduplication_report must return a dict"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_required_keys()
            if failed:
                t.failed = True
                t.msg = "Report dict must contain keys: 'exact_count', 'total_rows', 'clean_rows'"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_exact_count()
            if failed:
                t.failed = True
                t.msg = "Incorrect value for 'exact_count'"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_total_rows()
            if failed:
                t.failed = True
                t.msg = "Incorrect value for 'total_rows'"
                t.want = str(want)
                t.got = str(got)
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_clean_rows()
            if failed:
                t.failed = True
                t.msg = "Incorrect value for 'clean_rows'"
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
