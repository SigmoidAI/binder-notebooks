"""
Unit tests for the NumPy Array Operations assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

import numpy as np

from unittests_utils import (
    CreateArraysBattery,
    VectorizedOpsBattery,
    NormalizeBattery,
    LinearLayerBattery,
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
    """Test the create_arrays function."""
    def g():
        cases: List[test_case] = []
        
        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_arrays must be a function"
            t.want = "a Python function"
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        
        try:
            battery = CreateArraysBattery(learner_func)
            
            # Check return type
            t = test_case()
            got, want, failed = battery.result_is_tuple()
            if failed:
                t.failed = True
                t.msg = "Function must return a tuple"
                t.want = "tuple"
                t.got = f"got {type(battery.learner_result)}"
                return cases + [t]
            cases.append(t)
            
            # Check return length
            t = test_case()
            got, want, failed = battery.result_length()
            if failed:
                t.failed = True
                t.msg = "Function must return exactly 5 arrays"
                t.want = f"{want} arrays"
                t.got = f"{got} elements"
                return cases + [t]
            cases.append(t)
            
            # Check arr1 values
            t = test_case()
            got, want, failed = battery.arr1_values()
            if failed:
                t.failed = True
                t.msg = "arr1 does not have the correct values"
                t.want = f"Array with values [10, 20, 30, 40, 50]"
                t.got = f"Array with values {got}"
            cases.append(t)
            
            # Check arr2 shape
            t = test_case()
            got, want, failed = battery.arr2_shape()
            if failed:
                t.failed = True
                t.msg = "arr2 does not have the correct shape"
                t.want = f"Shape {want} (3x3 array of zeros)"
                t.got = f"Shape {got}"
            cases.append(t)
            
            # Check arr2 values (all zeros)
            t = test_case()
            got, want, failed = battery.arr2_values()
            if failed:
                t.failed = True
                t.msg = "arr2 should be filled with zeros"
                t.want = "All elements should be 0"
                t.got = f"Array contains non-zero values"
            cases.append(t)
            
            # Check arr3 shape
            t = test_case()
            got, want, failed = battery.arr3_shape()
            if failed:
                t.failed = True
                t.msg = "arr3 does not have the correct shape"
                t.want = f"Shape {want} (2x4 array of ones)"
                t.got = f"Shape {got}"
            cases.append(t)
            
            # Check arr3 values (all ones)
            t = test_case()
            got, want, failed = battery.arr3_values()
            if failed:
                t.failed = True
                t.msg = "arr3 should be filled with ones"
                t.want = "All elements should be 1"
                t.got = f"Array contains values other than 1"
            cases.append(t)
            
            # Check arr4 values
            t = test_case()
            got, want, failed = battery.arr4_values()
            if failed:
                t.failed = True
                t.msg = "arr4 does not have the correct values"
                t.want = f"Array [0, 1, 2, ..., 9] using np.arange"
                t.got = f"Array with values {got}"
            cases.append(t)
            
            # Check arr5 values
            t = test_case()
            got, want, failed = battery.arr5_values()
            if failed:
                t.failed = True
                t.msg = "arr5 does not have the correct values"
                t.want = f"6 evenly spaced values from 0 to 5 using np.linspace"
                t.got = f"Array with values {got}"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing create_arrays"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    """Test the apply_vectorized_operations function."""
    def g():
        cases: List[test_case] = []
        
        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "apply_vectorized_operations must be a function"
            t.want = "a Python function"
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        
        try:
            battery = VectorizedOpsBattery(learner_func)
            
            # Check return type
            t = test_case()
            got, want, failed = battery.result_is_tuple()
            if failed:
                t.failed = True
                t.msg = "Function must return a tuple"
                t.want = "tuple of 4 arrays"
                t.got = f"got {type(battery.learner_result)}"
                return cases + [t]
            cases.append(t)
            
            # Check return length
            t = test_case()
            got, want, failed = battery.result_length()
            if failed:
                t.failed = True
                t.msg = "Function must return exactly 4 results"
                t.want = f"{want} arrays"
                t.got = f"{got} elements"
                return cases + [t]
            cases.append(t)
            
            # Check squared
            t = test_case()
            got, want, failed = battery.squared_check()
            if failed:
                t.failed = True
                t.msg = "squared (a^2) is not computed correctly"
                t.want = f"For a=[1,4,9,16], squared should be {want}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check sqrt
            t = test_case()
            got, want, failed = battery.sqrt_a_check()
            if failed:
                t.failed = True
                t.msg = "sqrt_a (square root) is not computed correctly"
                t.want = f"For a=[1,4,9,16], sqrt should be {want}"
                t.got = f"{got}"
            cases.append(t)
            
            # Check exp
            t = test_case()
            got, want, failed = battery.exp_a_check()
            if failed:
                t.failed = True
                t.msg = "exp_a (exponential) is not computed correctly"
                t.want = f"Use np.exp(a)"
                t.got = f"Values don't match expected exponential"
            cases.append(t)
            
            # Check sum
            t = test_case()
            got, want, failed = battery.sum_ab_check()
            if failed:
                t.failed = True
                t.msg = "sum_ab (element-wise sum) is not computed correctly"
                t.want = f"For a=[1,4,9,16] and b=[10,20,30,40], sum should be {want}"
                t.got = f"{got}"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing apply_vectorized_operations"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    """Test the normalize_data function."""
    def g():
        cases: List[test_case] = []
        
        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "normalize_data must be a function"
            t.want = "a Python function"
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        
        try:
            battery = NormalizeBattery(learner_func)
            
            # Check return type
            t = test_case()
            got, want, failed = battery.result_is_array()
            if failed:
                t.failed = True
                t.msg = "Function must return a numpy array"
                t.want = "np.ndarray"
                t.got = f"got {type(battery.learner_result)}"
                return cases + [t]
            cases.append(t)
            
            # Check shape
            t = test_case()
            got, want, failed = battery.shape_check()
            if failed:
                t.failed = True
                t.msg = "Output shape does not match input shape"
                t.want = f"Shape {want}"
                t.got = f"Shape {got}"
                return cases + [t]
            cases.append(t)
            
            # Check normalized values
            t = test_case()
            got, want, failed = battery.normalized_check()
            if failed:
                t.failed = True
                t.msg = "Normalized values are not correct"
                t.want = "Use (data - row_mean) / row_std with axis=1 and keepdims=True"
                t.got = "Values don't match expected normalization"
            cases.append(t)
            
            # Check row means are ~0
            t = test_case()
            got, want, failed = battery.row_means_check()
            if failed:
                t.failed = True
                t.msg = "After normalization, row means should be approximately 0"
                t.want = f"Row means close to {want}"
                t.got = f"Row means are {got}"
            cases.append(t)
            
            # Check row stds are ~1
            t = test_case()
            got, want, failed = battery.row_stds_check()
            if failed:
                t.failed = True
                t.msg = "After normalization, row standard deviations should be approximately 1"
                t.want = f"Row stds close to {want}"
                t.got = f"Row stds are {got}"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing normalize_data"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    """Test the linear_layer_forward function."""
    def g():
        cases: List[test_case] = []
        
        # Check if it's a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "linear_layer_forward must be a function"
            t.want = "a Python function"
            t.got = type(learner_func)
            return [t]
        cases.append(t)
        
        try:
            battery = LinearLayerBattery(learner_func)
            
            # Check return type
            t = test_case()
            got, want, failed = battery.result_is_array()
            if failed:
                t.failed = True
                t.msg = "Function must return a numpy array"
                t.want = "np.ndarray"
                t.got = f"got {type(battery.learner_result)}"
                return cases + [t]
            cases.append(t)
            
            # Check shape
            t = test_case()
            got, want, failed = battery.shape_check()
            if failed:
                t.failed = True
                t.msg = "Output shape is not correct"
                t.want = f"Shape {want} (n_samples x n_output_features)"
                t.got = f"Shape {got}"
                return cases + [t]
            cases.append(t)
            
            # Check output values
            t = test_case()
            got, want, failed = battery.output_check()
            if failed:
                t.failed = True
                t.msg = "Output values are not correct"
                t.want = "Y = X @ W + b"
                t.got = "Values don't match expected linear transformation"
            cases.append(t)
            
            # Check first sample specifically
            t = test_case()
            got, want, failed = battery.first_sample_check()
            if failed:
                t.failed = True
                t.msg = "First sample output is not correct"
                t.want = f"First row should be {want}"
                t.got = f"First row is {got}"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing linear_layer_forward"
            t.want = "Function to execute without errors"
            t.got = f"Exception: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)
