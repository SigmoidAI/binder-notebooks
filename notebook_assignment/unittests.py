"""
Unit tests for the Basic Statistics assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

import numpy as np

from unittests_utils import (
    ComputeMeanBattery,
    VarianceStdBattery,
    CovarianceBattery,
    CorrelationBattery,
    StatisticsSummaryBattery,
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
            print(f"\n  ❌ {case.msg}")
            if case.want:
                print(f"     Expected: {case.want}")
            if case.got:
                print(f"     Got: {case.got}")


def exercise_1(learner_func):
    """Test the compute_mean function."""
    def g():
        cases: List[test_case] = []
        
        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_mean must be a function"
            t.want = "a Python function"
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        
        try:
            battery = ComputeMeanBattery(learner_func)
            
            # Check 1D mean
            t = test_case()
            got, want, failed = battery.mean_1d_check()
            if failed:
                t.failed = True
                t.msg = "Mean of 1D array is incorrect"
                t.want = f"For [1,2,3,4,5], mean should be {want}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check 2D mean (all elements)
            t = test_case()
            got, want, failed = battery.mean_2d_all_check()
            if failed:
                t.failed = True
                t.msg = "Mean of all elements (axis=None) is incorrect"
                t.want = f"Mean should be {want}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check 2D mean axis=0
            t = test_case()
            got, want, failed = battery.mean_2d_axis0_check()
            if failed:
                t.failed = True
                t.msg = "Mean along axis=0 (column means) is incorrect"
                t.want = f"Column means should be {want}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check 2D mean axis=1
            t = test_case()
            got, want, failed = battery.mean_2d_axis1_check()
            if failed:
                t.failed = True
                t.msg = "Mean along axis=1 (row means) is incorrect"
                t.want = f"Row means should be {want}"
                t.got = f"{got}"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing compute_mean"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    """Test the compute_variance_std function."""
    def g():
        cases: List[test_case] = []
        
        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_variance_std must be a function"
            t.want = "a Python function"
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        
        try:
            battery = VarianceStdBattery(learner_func)
            
            # Check return type (population)
            t = test_case()
            got, want, failed = battery.result_is_tuple_pop()
            if failed:
                t.failed = True
                t.msg = "Function must return a tuple of (variance, std)"
                t.want = "tuple of 2 values"
                t.got = f"{type(battery.result_pop)}"
                return cases + [t]
            cases.append(t)
            
            # Check population variance
            t = test_case()
            got, want, failed = battery.pop_variance_check()
            if failed:
                t.failed = True
                t.msg = "Population variance (ddof=0) is incorrect"
                t.want = f"Variance should be {want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check population std
            t = test_case()
            got, want, failed = battery.pop_std_check()
            if failed:
                t.failed = True
                t.msg = "Population standard deviation (ddof=0) is incorrect"
                t.want = f"Std should be {want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check sample variance
            t = test_case()
            got, want, failed = battery.sample_variance_check()
            if failed:
                t.failed = True
                t.msg = "Sample variance (ddof=1) is incorrect"
                t.want = f"Variance should be {want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check sample std
            t = test_case()
            got, want, failed = battery.sample_std_check()
            if failed:
                t.failed = True
                t.msg = "Sample standard deviation (ddof=1) is incorrect"
                t.want = f"Std should be {want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing compute_variance_std"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    """Test the compute_covariance function."""
    def g():
        cases: List[test_case] = []
        
        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_covariance must be a function"
            t.want = "a Python function"
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        
        try:
            battery = CovarianceBattery(learner_func)
            
            # Check positive covariance value
            t = test_case()
            got, want, failed = battery.cov_positive_check()
            if failed:
                t.failed = True
                t.msg = "Covariance calculation is incorrect (positive case)"
                t.want = f"Covariance should be {want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check that positive covariance is actually positive
            t = test_case()
            got, want, failed = battery.cov_is_positive()
            if failed:
                t.failed = True
                t.msg = "Covariance should be positive when variables move together"
                t.want = "Positive value"
                t.got = f"{battery.result_pos}"
            cases.append(t)
            
            # Check negative covariance value
            t = test_case()
            got, want, failed = battery.cov_negative_check()
            if failed:
                t.failed = True
                t.msg = "Covariance calculation is incorrect (negative case)"
                t.want = f"Covariance should be {want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check that negative covariance is actually negative
            t = test_case()
            got, want, failed = battery.cov_is_negative()
            if failed:
                t.failed = True
                t.msg = "Covariance should be negative when variables move oppositely"
                t.want = "Negative value"
                t.got = f"{battery.result_neg}"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing compute_covariance"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    """Test the compute_correlation function."""
    def g():
        cases: List[test_case] = []
        
        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_correlation must be a function"
            t.want = "a Python function"
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        
        try:
            battery = CorrelationBattery(learner_func)
            
            # Check correlation values are in valid range
            t = test_case()
            got, want, failed = battery.corr_in_range()
            if failed:
                t.failed = True
                t.msg = "Correlation must be between -1 and 1"
                t.want = "Value in range [-1, 1]"
                t.got = f"{got}"
                return cases + [t]
            cases.append(t)
            
            # Check strong positive correlation
            t = test_case()
            got, want, failed = battery.corr_strong_check()
            if failed:
                t.failed = True
                t.msg = "Strong positive correlation is incorrect"
                t.want = f"Correlation should be approximately {want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check negative correlation
            t = test_case()
            got, want, failed = battery.corr_negative_check()
            if failed:
                t.failed = True
                t.msg = "Negative correlation is incorrect"
                t.want = f"Correlation should be {want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check perfect positive correlation
            t = test_case()
            got, want, failed = battery.corr_perfect_check()
            if failed:
                t.failed = True
                t.msg = "Perfect positive correlation is incorrect"
                t.want = f"Correlation should be {want:.6f} (perfect positive)"
                t.got = f"{got}"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing compute_correlation"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    """Test the compute_statistics_summary function."""
    def g():
        cases: List[test_case] = []
        
        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_statistics_summary must be a function"
            t.want = "a Python function"
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        
        try:
            battery = StatisticsSummaryBattery(learner_func)
            
            # Check return type
            t = test_case()
            got, want, failed = battery.result_is_dict()
            if failed:
                t.failed = True
                t.msg = "Function must return a dictionary"
                t.want = "dict"
                t.got = f"{type(battery.result)}"
                return cases + [t]
            cases.append(t)
            
            # Check required keys
            t = test_case()
            got, want, failed = battery.has_required_keys()
            if failed:
                t.failed = True
                t.msg = "Dictionary is missing required keys"
                t.want = "Keys: mean_x, mean_y, var_x, var_y, std_x, std_y, covariance, correlation"
                t.got = f"{got}"
                return cases + [t]
            cases.append(t)
            
            # Check mean_x
            t = test_case()
            got, want, failed = battery.mean_x_check()
            if failed:
                t.failed = True
                t.msg = "mean_x is incorrect"
                t.want = f"{want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check mean_y
            t = test_case()
            got, want, failed = battery.mean_y_check()
            if failed:
                t.failed = True
                t.msg = "mean_y is incorrect"
                t.want = f"{want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check var_x (sample variance)
            t = test_case()
            got, want, failed = battery.var_x_check()
            if failed:
                t.failed = True
                t.msg = "var_x is incorrect (should be sample variance, ddof=1)"
                t.want = f"{want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check var_y (sample variance)
            t = test_case()
            got, want, failed = battery.var_y_check()
            if failed:
                t.failed = True
                t.msg = "var_y is incorrect (should be sample variance, ddof=1)"
                t.want = f"{want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check std_x
            t = test_case()
            got, want, failed = battery.std_x_check()
            if failed:
                t.failed = True
                t.msg = "std_x is incorrect (should be sample std, ddof=1)"
                t.want = f"{want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check std_y
            t = test_case()
            got, want, failed = battery.std_y_check()
            if failed:
                t.failed = True
                t.msg = "std_y is incorrect (should be sample std, ddof=1)"
                t.want = f"{want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check covariance
            t = test_case()
            got, want, failed = battery.covariance_check()
            if failed:
                t.failed = True
                t.msg = "covariance is incorrect"
                t.want = f"{want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check correlation
            t = test_case()
            got, want, failed = battery.correlation_check()
            if failed:
                t.failed = True
                t.msg = "correlation is incorrect"
                t.want = f"{want:.6f}"
                t.got = f"{got}"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing compute_statistics_summary"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)
