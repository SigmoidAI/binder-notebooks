"""
Unit tests for the Missingness Pattern Analysis assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

import numpy as np
import pandas as pd

from unittests_utils import (
    CreateMissingnessIndicatorsBattery,
    SimplifiedMCARTestBattery,
    TestMARPatternBattery,
    MissingnessCorrelationBattery,
    ClassifyMissingnessTypeBattery,
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
    """Test the create_missingness_indicators function."""
    def g():
        cases: List[test_case] = []

        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_missingness_indicators must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = CreateMissingnessIndicatorsBattery(learner_func)

            # Check return type
            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "Function must return a pandas DataFrame"
                t.want = "pandas DataFrame"
                t.got = f"{type(battery.result)}"
                return cases + [t]
            cases.append(t)

            # Check shape
            t = test_case()
            got, want, failed = battery.correct_shape_check()
            if failed:
                t.failed = True
                t.msg = "Output DataFrame has incorrect shape"
                t.want = f"Shape {want}"
                t.got = f"Shape {got}"
            cases.append(t)

            # Check column names
            t = test_case()
            got, want, failed = battery.correct_columns_check()
            if failed:
                t.failed = True
                t.msg = "Column names are incorrect"
                t.want = f"Columns: {want}"
                t.got = f"Columns: {got}"
            cases.append(t)

            # Check A_missing values
            t = test_case()
            got, want, failed = battery.a_missing_values_check()
            if failed:
                t.failed = True
                t.msg = "A_missing column has incorrect values"
                t.want = f"{want}"
                t.got = f"{got}"
            cases.append(t)

            # Check B_missing values
            t = test_case()
            got, want, failed = battery.b_missing_values_check()
            if failed:
                t.failed = True
                t.msg = "B_missing column has incorrect values"
                t.want = f"{want}"
                t.got = f"{got}"
            cases.append(t)

            # Check all values are binary
            t = test_case()
            got, want, failed = battery.all_binary_check()
            if failed:
                t.failed = True
                t.msg = "All values must be binary (0 or 1)"
                t.want = "All values in {0, 1}"
                t.got = "Non-binary values found"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing create_missingness_indicators"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    """Test the simplified_mcar_test function."""
    def g():
        cases: List[test_case] = []

        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "simplified_mcar_test must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = SimplifiedMCARTestBattery(learner_func)

            # Check return type is tuple
            t = test_case()
            got, want, failed = battery.result_is_tuple_mcar()
            if failed:
                t.failed = True
                t.msg = "Function must return a tuple"
                t.want = "tuple"
                t.got = f"{type(battery.result_mcar)}"
                return cases + [t]
            cases.append(t)

            # Check tuple has 2 elements
            t = test_case()
            got, want, failed = battery.result_has_two_elements()
            if failed:
                t.failed = True
                t.msg = "Function must return tuple with 2 elements (statistic, p_value)"
                t.want = "2 elements"
                t.got = f"{got} elements"
                return cases + [t]
            cases.append(t)

            # Check statistic is numeric
            t = test_case()
            got, want, failed = battery.statistic_is_numeric()
            if failed:
                t.failed = True
                t.msg = "Test statistic must be numeric"
                t.want = "numeric value"
                t.got = f"{type(battery.result_mcar[0])}"
            cases.append(t)

            # Check p-value is numeric
            t = test_case()
            got, want, failed = battery.pvalue_is_numeric()
            if failed:
                t.failed = True
                t.msg = "P-value must be numeric"
                t.want = "numeric value"
                t.got = f"{type(battery.result_mcar[1])}"
            cases.append(t)

            # Check p-value is in valid range
            t = test_case()
            got, want, failed = battery.pvalue_in_valid_range()
            if failed:
                t.failed = True
                t.msg = "P-value must be between 0 and 1"
                t.want = want
                t.got = f"{got}"
            cases.append(t)

            # Check MCAR data gives high p-value
            t = test_case()
            got, want, failed = battery.mcar_high_pvalue()
            if failed:
                t.failed = True
                t.msg = "For MCAR data, p-value should typically be high"
                t.want = want
                t.got = f"p-value = {got}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing simplified_mcar_test"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    """Test the test_mar_pattern function."""
    def g():
        cases: List[test_case] = []

        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "test_mar_pattern must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = TestMARPatternBattery(learner_func)

            # Check return type is tuple
            t = test_case()
            got, want, failed = battery.result_is_tuple_mar()
            if failed:
                t.failed = True
                t.msg = "Function must return a tuple"
                t.want = "tuple"
                t.got = f"{type(battery.result_mar)}"
                return cases + [t]
            cases.append(t)

            # Check tuple has 2 elements
            t = test_case()
            got, want, failed = battery.result_has_two_elements()
            if failed:
                t.failed = True
                t.msg = "Function must return tuple with 2 elements (statistic, p_value)"
                t.want = "2 elements"
                t.got = f"{got} elements"
                return cases + [t]
            cases.append(t)

            # Check statistic is numeric
            t = test_case()
            got, want, failed = battery.statistic_is_numeric()
            if failed:
                t.failed = True
                t.msg = "Test statistic must be numeric"
                t.want = "numeric value"
                t.got = f"{type(battery.result_mar[0])}"
            cases.append(t)

            # Check p-value is numeric
            t = test_case()
            got, want, failed = battery.pvalue_is_numeric()
            if failed:
                t.failed = True
                t.msg = "P-value must be numeric"
                t.want = "numeric value"
                t.got = f"{type(battery.result_mar[1])}"
            cases.append(t)

            # Check p-value is in valid range
            t = test_case()
            got, want, failed = battery.pvalue_in_valid_range()
            if failed:
                t.failed = True
                t.msg = "P-value must be between 0 and 1"
                t.want = want
                t.got = f"{got}"
            cases.append(t)

            # Check MAR data gives low p-value
            t = test_case()
            got, want, failed = battery.mar_low_pvalue()
            if failed:
                t.failed = True
                t.msg = "For MAR data, p-value should typically be low (significant relationship)"
                t.want = want
                t.got = f"p-value = {got}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing test_mar_pattern"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    """Test the compute_missingness_correlation function."""
    def g():
        cases: List[test_case] = []

        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_missingness_correlation must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = MissingnessCorrelationBattery(learner_func)

            # Check return type is DataFrame
            t = test_case()
            got, want, failed = battery.result_is_dataframe()
            if failed:
                t.failed = True
                t.msg = "Function must return a pandas DataFrame"
                t.want = "pandas DataFrame"
                t.got = f"{type(battery.result)}"
                return cases + [t]
            cases.append(t)

            # Check result is square
            t = test_case()
            got, want, failed = battery.result_is_square()
            if failed:
                t.failed = True
                t.msg = "Correlation matrix must be square"
                t.want = want
                t.got = f"Shape {got}"
            cases.append(t)

            # Check diagonal is 1.0
            t = test_case()
            got, want, failed = battery.diagonal_is_one()
            if failed:
                t.failed = True
                t.msg = "Diagonal elements must be 1.0 (correlation with itself)"
                t.want = want
                t.got = f"{got}"
            cases.append(t)

            # Check values are in valid range
            t = test_case()
            got, want, failed = battery.values_in_valid_range()
            if failed:
                t.failed = True
                t.msg = "All correlation values must be between -1 and 1"
                t.want = want
                t.got = got
            cases.append(t)

            # Check correlated missingness is detected
            t = test_case()
            got, want, failed = battery.correlated_missingness_detected()
            if failed:
                t.failed = True
                t.msg = "Correlated missingness between A and B should be detected"
                t.want = want
                t.got = f"Correlation = {got}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing compute_missingness_correlation"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    """Test the classify_missingness_type function."""
    def g():
        cases: List[test_case] = []

        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "classify_missingness_type must be a function"
            t.want = "a Python function"
            t.got = str(type(learner_func))
            return [t]
        cases.append(t)

        try:
            battery = ClassifyMissingnessTypeBattery(learner_func)

            # Check return type is dict
            t = test_case()
            got, want, failed = battery.result_is_dict_mcar()
            if failed:
                t.failed = True
                t.msg = "Function must return a dictionary"
                t.want = "dict"
                t.got = f"{type(battery.result_mcar)}"
                return cases + [t]
            cases.append(t)

            # Check has 'classification' key
            t = test_case()
            got, want, failed = battery.result_has_classification_key()
            if failed:
                t.failed = True
                t.msg = "Result dictionary must contain 'classification' key"
                t.want = want
                t.got = f"Keys: {list(battery.result_mcar.keys()) if battery.result_mcar else None}"
            cases.append(t)

            # Check classification is valid type
            t = test_case()
            got, want, failed = battery.classification_is_valid_type()
            if failed:
                t.failed = True
                t.msg = "Classification must be one of: MCAR, MAR, MNAR, Unknown"
                t.want = want
                t.got = f"{got}"
            cases.append(t)

            # Check MCAR is classified correctly
            t = test_case()
            got, want, failed = battery.mcar_classified_correctly()
            if failed:
                t.failed = True
                t.msg = "MCAR data should be classified as MCAR or Unknown"
                t.want = want
                t.got = f"{got}"
            cases.append(t)

            # Check MAR is classified correctly
            t = test_case()
            got, want, failed = battery.mar_classified_correctly()
            if failed:
                t.failed = True
                t.msg = "MAR data should not be classified as MCAR"
                t.want = want
                t.got = f"{got}"
            cases.append(t)

            # Check has evidence/p_values key
            t = test_case()
            got, want, failed = battery.result_has_evidence_key()
            if failed:
                t.failed = True
                t.msg = "Result should include evidence (p_values or evidence key)"
                t.want = want
                t.got = f"Keys: {list(battery.result_mcar.keys()) if battery.result_mcar else None}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing classify_missingness_type"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)
