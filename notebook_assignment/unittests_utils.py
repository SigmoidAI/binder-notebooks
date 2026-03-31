"""
Unit test utilities for the Imputation Strategy Comparison assignment.

This module provides battery test classes that define test cases
and reference values for each exercise.
"""

import numpy as np
import pandas as pd


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
            if got is None:
                condition = True
            else:
                condition = not np.allclose(got, want, rtol=1e-5, atol=1e-8)
        elif isinstance(want, float):
            if got is None:
                condition = True
            else:
                condition = not np.isclose(got, want, rtol=1e-5, atol=1e-8)
        else:
            condition = got != want
        return got, want, condition


class ListwiseDeletionBattery(TestBattery):
    """Test battery for Exercise 1: listwise_deletion function."""

    def _get_reference_inputs(self):
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0],
            'B': [np.nan, 2.0, 3.0, 4.0, 5.0],
            'C': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy())
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {
            "n_rows": 3,
            "no_missing": True,
        }

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        return got, True, not got

    def correct_row_count(self):
        if self.result is None:
            return None, self.reference_checks["n_rows"], True
        got = len(self.result)
        want = self.reference_checks["n_rows"]
        return got, want, got != want

    def no_missing_values(self):
        if self.result is None:
            return None, True, True
        got = self.result.isna().sum().sum() == 0
        return got, True, not got


class MeanMedianImputationBattery(TestBattery):
    """Test battery for Exercise 2: mean_median_imputation function."""

    def _get_reference_inputs(self):
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0],
            'B': [10.0, np.nan, 30.0, 40.0, 50.0],
        })
        self.strategy = 'mean'

    def extract_info(self):
        try:
            self.result_mean = self.learner_object(self.test_df.copy(), strategy='mean')
            self.result_median = self.learner_object(self.test_df.copy(), strategy='median')
        except Exception:
            self.result_mean = None
            self.result_median = None

    def get_reference_checks(self):
        self.reference_checks = {
            "mean_A": 3.0,
            "mean_B": 32.5,
            "median_A": 3.0,
            "median_B": 35.0,
        }

    def result_is_dataframe(self):
        got = isinstance(self.result_mean, pd.DataFrame)
        return got, True, not got

    def no_missing_values_mean(self):
        if self.result_mean is None:
            return None, True, True
        got = self.result_mean.isna().sum().sum() == 0
        return got, True, not got

    def correct_mean_imputation_A(self):
        if self.result_mean is None:
            return None, self.reference_checks["mean_A"], True
        got = self.result_mean.loc[2, 'A']
        want = self.reference_checks["mean_A"]
        return got, want, not np.isclose(got, want, rtol=1e-5)

    def correct_mean_imputation_B(self):
        if self.result_mean is None:
            return None, self.reference_checks["mean_B"], True
        got = self.result_mean.loc[1, 'B']
        want = self.reference_checks["mean_B"]
        return got, want, not np.isclose(got, want, rtol=1e-5)

    def correct_median_imputation_A(self):
        if self.result_median is None:
            return None, self.reference_checks["median_A"], True
        got = self.result_median.loc[2, 'A']
        want = self.reference_checks["median_A"]
        return got, want, not np.isclose(got, want, rtol=1e-5)


class ModeImputationBattery(TestBattery):
    """Test battery for Exercise 3: mode_imputation function."""

    def _get_reference_inputs(self):
        self.test_df = pd.DataFrame({
            'category': ['A', 'B', 'A', np.nan, 'A', 'B', np.nan],
        })

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy(), 'category')
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {
            "mode_value": 'A',
        }

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        return got, True, not got

    def no_missing_values(self):
        if self.result is None:
            return None, True, True
        got = self.result['category'].isna().sum() == 0
        return got, True, not got

    def correct_mode_imputation(self):
        if self.result is None:
            return None, self.reference_checks["mode_value"], True
        got = self.result.loc[3, 'category']
        want = self.reference_checks["mode_value"]
        return got, want, got != want


class KNNImputationBattery(TestBattery):
    """Test battery for Exercise 4: knn_imputation function."""

    def _get_reference_inputs(self):
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0],
            'B': [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        self.n_neighbors = 2

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy(), n_neighbors=self.n_neighbors)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {
            "no_missing": True,
            "imputed_in_range": True,
        }

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        return got, True, not got

    def no_missing_values(self):
        if self.result is None:
            return None, True, True
        got = self.result.isna().sum().sum() == 0
        return got, True, not got

    def imputed_value_in_reasonable_range(self):
        if self.result is None:
            return None, True, True
        imputed_val = self.result.loc[2, 'A']
        got = 1.0 <= imputed_val <= 5.0
        return got, True, not got


class IterativeImputationBattery(TestBattery):
    """Test battery for Exercise 5: iterative_imputation function."""

    def _get_reference_inputs(self):
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            'B': [2.0, 4.0, 6.0, np.nan, 10.0, 12.0, 14.0, 16.0],
            'C': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        })

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy(), max_iter=10)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {
            "no_missing": True,
        }

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        return got, True, not got

    def no_missing_values(self):
        if self.result is None:
            return None, True, True
        got = self.result.isna().sum().sum() == 0
        return got, True, not got

    def imputed_values_reasonable(self):
        if self.result is None:
            return None, True, True
        imputed_A = self.result.loc[2, 'A']
        imputed_B = self.result.loc[3, 'B']
        got = (0 <= imputed_A <= 10) and (0 <= imputed_B <= 20)
        return got, True, not got


class CompareStrategiesBattery(TestBattery):
    """Test battery for Exercise 6: compare_imputation_strategies function."""

    def _get_reference_inputs(self):
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0],
            'B': [10.0, np.nan, 30.0, 40.0, 50.0],
        })

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy())
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {
            "is_dict": True,
            "has_required_keys": True,
        }

    def result_is_dict(self):
        got = isinstance(self.result, dict)
        return got, True, not got

    def has_required_methods(self):
        if self.result is None:
            return None, True, True
        required = ['mean', 'median', 'knn']
        got = all(key in self.result for key in required)
        return got, True, not got

    def all_results_are_dataframes(self):
        if self.result is None:
            return None, True, True
        got = all(isinstance(v, pd.DataFrame) for v in self.result.values())
        return got, True, not got

    def all_results_no_missing(self):
        if self.result is None:
            return None, True, True
        got = all(df.isna().sum().sum() == 0 for df in self.result.values())
        return got, True, not got
