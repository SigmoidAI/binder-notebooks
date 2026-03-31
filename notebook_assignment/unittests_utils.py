"""
Unit test utilities for the NumPy Array Operations assignment.

This module provides battery test classes that define test cases
and reference values for each exercise.
"""

import numpy as np


class TestBattery:
    """Base class for test batteries."""
    
    def __init__(self, learner_object):
        self.learner_object = learner_object
        self._get_reference_inputs()
        self.extract_info()
        self.get_reference_checks()

    def _get_reference_inputs(self):
        pass

    def extract_info(self):
        pass

    def get_reference_checks(self):
        pass

    def _check(self, check_name, got):
        want = self.reference_checks[check_name]
        if isinstance(want, np.ndarray):
            condition = not np.allclose(got, want, rtol=1e-5, atol=1e-8)
        else:
            condition = got != want
        return got, want, condition


class CreateArraysBattery(TestBattery):
    """Test battery for Exercise 1: create_arrays function."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def extract_info(self):
        """Execute the learner's function and store results."""
        self.learner_result = self.learner_object()
        if self.learner_result is not None and len(self.learner_result) == 5:
            self.arr1, self.arr2, self.arr3, self.arr4, self.arr5 = self.learner_result
        else:
            self.arr1 = self.arr2 = self.arr3 = self.arr4 = self.arr5 = None

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {
            "arr1_values": np.array([10, 20, 30, 40, 50]),
            "arr1_shape": (5,),
            "arr2_shape": (3, 3),
            "arr2_values": np.zeros((3, 3)),
            "arr3_shape": (2, 4),
            "arr3_values": np.ones((2, 4)),
            "arr4_values": np.arange(0, 10),
            "arr4_shape": (10,),
            "arr5_values": np.linspace(0, 5, 6),
            "arr5_shape": (6,),
        }

    def result_is_tuple(self):
        got = isinstance(self.learner_result, tuple)
        want = True
        return got, want, not got

    def result_length(self):
        got = len(self.learner_result) if self.learner_result else 0
        want = 5
        return got, want, got != want

    def arr1_values(self):
        got = self.arr1
        return self._check("arr1_values", got)

    def arr1_shape(self):
        got = self.arr1.shape if self.arr1 is not None else None
        return self._check("arr1_shape", got)

    def arr2_shape(self):
        got = self.arr2.shape if self.arr2 is not None else None
        return self._check("arr2_shape", got)

    def arr2_values(self):
        got = self.arr2
        return self._check("arr2_values", got)

    def arr3_shape(self):
        got = self.arr3.shape if self.arr3 is not None else None
        return self._check("arr3_shape", got)

    def arr3_values(self):
        got = self.arr3
        return self._check("arr3_values", got)

    def arr4_values(self):
        got = self.arr4
        return self._check("arr4_values", got)

    def arr4_shape(self):
        got = self.arr4.shape if self.arr4 is not None else None
        return self._check("arr4_shape", got)

    def arr5_values(self):
        got = self.arr5
        return self._check("arr5_values", got)

    def arr5_shape(self):
        got = self.arr5.shape if self.arr5 is not None else None
        return self._check("arr5_shape", got)


class VectorizedOpsBattery(TestBattery):
    """Test battery for Exercise 2: apply_vectorized_operations function."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        self.test_a = np.array([1.0, 4.0, 9.0, 16.0])
        self.test_b = np.array([10.0, 20.0, 30.0, 40.0])

    def extract_info(self):
        """Execute the learner's function and store results."""
        self.learner_result = self.learner_object(self.test_a, self.test_b)
        if self.learner_result is not None and len(self.learner_result) == 4:
            self.squared, self.sqrt_a, self.exp_a, self.sum_ab = self.learner_result
        else:
            self.squared = self.sqrt_a = self.exp_a = self.sum_ab = None

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {
            "squared": self.test_a ** 2,
            "sqrt_a": np.sqrt(self.test_a),
            "exp_a": np.exp(self.test_a),
            "sum_ab": self.test_a + self.test_b,
        }

    def result_is_tuple(self):
        got = isinstance(self.learner_result, tuple)
        want = True
        return got, want, not got

    def result_length(self):
        got = len(self.learner_result) if self.learner_result else 0
        want = 4
        return got, want, got != want

    def squared_check(self):
        got = self.squared
        return self._check("squared", got)

    def sqrt_a_check(self):
        got = self.sqrt_a
        return self._check("sqrt_a", got)

    def exp_a_check(self):
        got = self.exp_a
        return self._check("exp_a", got)

    def sum_ab_check(self):
        got = self.sum_ab
        return self._check("sum_ab", got)


class NormalizeBattery(TestBattery):
    """Test battery for Exercise 3: normalize_data function."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        self.test_data = np.array([
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0],
            [5.0, 10.0, 15.0, 20.0]
        ])

    def extract_info(self):
        """Execute the learner's function and store results."""
        self.learner_result = self.learner_object(self.test_data)

    def get_reference_checks(self):
        """Define expected values for each check."""
        row_mean = self.test_data.mean(axis=1, keepdims=True)
        row_std = self.test_data.std(axis=1, keepdims=True)
        self.expected_normalized = (self.test_data - row_mean) / row_std
        
        self.reference_checks = {
            "shape": self.test_data.shape,
            "normalized": self.expected_normalized,
            "row_means": np.zeros(3),
            "row_stds": np.ones(3),
        }

    def result_is_array(self):
        got = isinstance(self.learner_result, np.ndarray)
        want = True
        return got, want, not got

    def shape_check(self):
        got = self.learner_result.shape if self.learner_result is not None else None
        return self._check("shape", got)

    def normalized_check(self):
        got = self.learner_result
        return self._check("normalized", got)

    def row_means_check(self):
        if self.learner_result is None:
            return None, np.zeros(3), True
        got = self.learner_result.mean(axis=1)
        want = self.reference_checks["row_means"]
        condition = not np.allclose(got, want, atol=1e-10)
        return got, want, condition

    def row_stds_check(self):
        if self.learner_result is None:
            return None, np.ones(3), True
        got = self.learner_result.std(axis=1)
        want = self.reference_checks["row_stds"]
        condition = not np.allclose(got, want, atol=1e-10)
        return got, want, condition


class LinearLayerBattery(TestBattery):
    """Test battery for Exercise 4: linear_layer_forward function."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        self.test_X = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])
        self.test_W = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ])
        self.test_b = np.array([0.1, 0.2])

    def extract_info(self):
        """Execute the learner's function and store results."""
        self.learner_result = self.learner_object(self.test_X, self.test_W, self.test_b)

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.expected_output = self.test_X @ self.test_W + self.test_b
        
        self.reference_checks = {
            "shape": (4, 2),
            "output": self.expected_output,
        }

    def result_is_array(self):
        got = isinstance(self.learner_result, np.ndarray)
        want = True
        return got, want, not got

    def shape_check(self):
        got = self.learner_result.shape if self.learner_result is not None else None
        return self._check("shape", got)

    def output_check(self):
        got = self.learner_result
        return self._check("output", got)

    def first_sample_check(self):
        """Check the first sample's output specifically."""
        if self.learner_result is None:
            return None, self.expected_output[0], True
        got = self.learner_result[0]
        want = self.expected_output[0]
        condition = not np.allclose(got, want, rtol=1e-5)
        return got, want, condition
