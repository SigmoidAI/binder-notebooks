"""
Unit tests for the Imputation Strategy Comparison assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

import pandas as pd

from unittests_utils import (
    ListwiseDeletionBattery,
    MeanMedianImputationBattery,
    ModeImputationBattery,
    KNNImputationBattery,
    IterativeImputationBattery,
    CompareStrategiesBattery,
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


def exercise_1(learner_func):
    """Test the listwise_deletion function."""
    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "listwise_deletion must be a function"
            return [t]
        cases.append(t)

        try:
            battery = ListwiseDeletionBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "Function must return a pandas DataFrame"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_row_count()
            if failed:
                t.failed = True
                t.msg = "Incorrect number of rows after listwise deletion"
                t.want = f"{want} rows"
                t.got = f"{got} rows"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.no_missing_values()
            if failed:
                t.failed = True
                t.msg = "Result should have no missing values"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    """Test the mean_median_imputation function."""
    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "mean_median_imputation must be a function"
            return [t]
        cases.append(t)

        try:
            battery = MeanMedianImputationBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "Function must return a pandas DataFrame"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.no_missing_values_mean()
            if failed:
                t.failed = True
                t.msg = "Result should have no missing values after mean imputation"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_mean_imputation_A()
            if failed:
                t.failed = True
                t.msg = "Mean imputation for column A is incorrect"
                t.want = f"{want}"
                t.got = f"{got}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_mean_imputation_B()
            if failed:
                t.failed = True
                t.msg = "Mean imputation for column B is incorrect"
                t.want = f"{want}"
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


def exercise_3(learner_func):
    """Test the mode_imputation function."""
    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "mode_imputation must be a function"
            return [t]
        cases.append(t)

        try:
            battery = ModeImputationBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "Function must return a pandas DataFrame"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.no_missing_values()
            if failed:
                t.failed = True
                t.msg = "Categorical column should have no missing values after mode imputation"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.correct_mode_imputation()
            if failed:
                t.failed = True
                t.msg = "Mode imputation is incorrect"
                t.want = f"'{want}'"
                t.got = f"'{got}'"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    """Test the knn_imputation function."""
    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "knn_imputation must be a function"
            return [t]
        cases.append(t)

        try:
            battery = KNNImputationBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "Function must return a pandas DataFrame"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.no_missing_values()
            if failed:
                t.failed = True
                t.msg = "Result should have no missing values after KNN imputation"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.imputed_value_in_reasonable_range()
            if failed:
                t.failed = True
                t.msg = "Imputed value is outside reasonable range"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    """Test the iterative_imputation function."""
    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "iterative_imputation must be a function"
            return [t]
        cases.append(t)

        try:
            battery = IterativeImputationBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "Function must return a pandas DataFrame"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.no_missing_values()
            if failed:
                t.failed = True
                t.msg = "Result should have no missing values after iterative imputation"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.imputed_values_reasonable()
            if failed:
                t.failed = True
                t.msg = "Imputed values are outside reasonable range"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_func):
    """Test the compare_imputation_strategies function."""
    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compare_imputation_strategies must be a function"
            return [t]
        cases.append(t)

        try:
            battery = CompareStrategiesBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_dict()
            if failed:
                t.failed = True
                t.msg = "Function must return a dictionary"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_required_methods()
            if failed:
                t.failed = True
                t.msg = "Dictionary must contain keys: 'mean', 'median', 'knn'"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.all_results_are_dataframes()
            if failed:
                t.failed = True
                t.msg = "All dictionary values must be pandas DataFrames"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.all_results_no_missing()
            if failed:
                t.failed = True
                t.msg = "All imputed DataFrames should have no missing values"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)
