"""
Unit test utilities for the Basic Statistics assignment.

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


class ComputeMeanBattery(TestBattery):
    """Test battery for Exercise 1: compute_mean function."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        self.test_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.test_2d = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

    def extract_info(self):
        """Execute the learner's function and store results."""
        try:
            self.result_1d = self.learner_object(self.test_1d)
            self.result_2d_all = self.learner_object(self.test_2d)
            self.result_2d_axis0 = self.learner_object(self.test_2d, axis=0)
            self.result_2d_axis1 = self.learner_object(self.test_2d, axis=1)
        except Exception:
            self.result_1d = None
            self.result_2d_all = None
            self.result_2d_axis0 = None
            self.result_2d_axis1 = None

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {
            "mean_1d": np.mean(self.test_1d),
            "mean_2d_all": np.mean(self.test_2d),
            "mean_2d_axis0": np.mean(self.test_2d, axis=0),
            "mean_2d_axis1": np.mean(self.test_2d, axis=1),
        }

    def mean_1d_check(self):
        got = self.result_1d
        return self._check("mean_1d", got)

    def mean_2d_all_check(self):
        got = self.result_2d_all
        return self._check("mean_2d_all", got)

    def mean_2d_axis0_check(self):
        got = self.result_2d_axis0
        return self._check("mean_2d_axis0", got)

    def mean_2d_axis1_check(self):
        got = self.result_2d_axis1
        return self._check("mean_2d_axis1", got)


class VarianceStdBattery(TestBattery):
    """Test battery for Exercise 2: compute_variance_std function."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        self.test_data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

    def extract_info(self):
        """Execute the learner's function and store results."""
        try:
            self.result_pop = self.learner_object(self.test_data, ddof=0)
            self.result_sample = self.learner_object(self.test_data, ddof=1)
        except Exception:
            self.result_pop = None
            self.result_sample = None

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {
            "pop_variance": np.var(self.test_data, ddof=0),
            "pop_std": np.std(self.test_data, ddof=0),
            "sample_variance": np.var(self.test_data, ddof=1),
            "sample_std": np.std(self.test_data, ddof=1),
        }

    def result_is_tuple_pop(self):
        got = isinstance(self.result_pop, tuple) and len(self.result_pop) == 2
        want = True
        return got, want, not got

    def result_is_tuple_sample(self):
        got = isinstance(self.result_sample, tuple) and len(self.result_sample) == 2
        want = True
        return got, want, not got

    def pop_variance_check(self):
        if self.result_pop is None:
            return None, self.reference_checks["pop_variance"], True
        got = self.result_pop[0]
        return self._check("pop_variance", got)

    def pop_std_check(self):
        if self.result_pop is None:
            return None, self.reference_checks["pop_std"], True
        got = self.result_pop[1]
        return self._check("pop_std", got)

    def sample_variance_check(self):
        if self.result_sample is None:
            return None, self.reference_checks["sample_variance"], True
        got = self.result_sample[0]
        return self._check("sample_variance", got)

    def sample_std_check(self):
        if self.result_sample is None:
            return None, self.reference_checks["sample_std"], True
        got = self.result_sample[1]
        return self._check("sample_std", got)


class CovarianceBattery(TestBattery):
    """Test battery for Exercise 3: compute_covariance function."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        self.x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pos = np.array([2.0, 4.0, 5.0, 4.0, 5.0])
        self.y_neg = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    def extract_info(self):
        """Execute the learner's function and store results."""
        try:
            self.result_pos = self.learner_object(self.x, self.y_pos)
            self.result_neg = self.learner_object(self.x, self.y_neg)
        except Exception:
            self.result_pos = None
            self.result_neg = None

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {
            "cov_positive": np.cov(self.x, self.y_pos)[0, 1],
            "cov_negative": np.cov(self.x, self.y_neg)[0, 1],
        }

    def cov_positive_check(self):
        got = self.result_pos
        return self._check("cov_positive", got)

    def cov_negative_check(self):
        got = self.result_neg
        return self._check("cov_negative", got)

    def cov_is_positive(self):
        if self.result_pos is None:
            return None, "positive value", True
        got = self.result_pos > 0
        want = True
        return got, want, not got

    def cov_is_negative(self):
        if self.result_neg is None:
            return None, "negative value", True
        got = self.result_neg < 0
        want = True
        return got, want, not got


class CorrelationBattery(TestBattery):
    """Test battery for Exercise 4: compute_correlation function."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        self.x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_strong = np.array([2.2, 4.1, 5.9, 8.0, 9.8])
        self.y_negative = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
        self.y_perfect = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    def extract_info(self):
        """Execute the learner's function and store results."""
        try:
            self.result_strong = self.learner_object(self.x, self.y_strong)
            self.result_negative = self.learner_object(self.x, self.y_negative)
            self.result_perfect = self.learner_object(self.x, self.y_perfect)
        except Exception:
            self.result_strong = None
            self.result_negative = None
            self.result_perfect = None

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {
            "corr_strong": np.corrcoef(self.x, self.y_strong)[0, 1],
            "corr_negative": np.corrcoef(self.x, self.y_negative)[0, 1],
            "corr_perfect": np.corrcoef(self.x, self.y_perfect)[0, 1],
        }

    def corr_strong_check(self):
        got = self.result_strong
        return self._check("corr_strong", got)

    def corr_negative_check(self):
        got = self.result_negative
        return self._check("corr_negative", got)

    def corr_perfect_check(self):
        got = self.result_perfect
        return self._check("corr_perfect", got)

    def corr_in_range(self):
        """Check that all correlations are between -1 and 1."""
        results = [self.result_strong, self.result_negative, self.result_perfect]
        for r in results:
            if r is None:
                return None, "values between -1 and 1", True
            if not (-1 <= r <= 1):
                return r, "value between -1 and 1", True
        return True, True, False


class StatisticsSummaryBattery(TestBattery):
    """Test battery for Exercise 5: compute_statistics_summary function."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        self.x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        self.y = np.array([2.1, 4.0, 5.8, 8.1, 10.2, 11.9, 14.0, 15.8, 18.1, 20.0])

    def extract_info(self):
        """Execute the learner's function and store results."""
        try:
            self.result = self.learner_object(self.x, self.y)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {
            "mean_x": np.mean(self.x),
            "mean_y": np.mean(self.y),
            "var_x": np.var(self.x, ddof=1),
            "var_y": np.var(self.y, ddof=1),
            "std_x": np.std(self.x, ddof=1),
            "std_y": np.std(self.y, ddof=1),
            "covariance": np.cov(self.x, self.y)[0, 1],
            "correlation": np.corrcoef(self.x, self.y)[0, 1],
        }

    def result_is_dict(self):
        got = isinstance(self.result, dict)
        want = True
        return got, want, not got

    def has_required_keys(self):
        required_keys = ['mean_x', 'mean_y', 'var_x', 'var_y', 
                         'std_x', 'std_y', 'covariance', 'correlation']
        if self.result is None:
            return None, required_keys, True
        missing = [k for k in required_keys if k not in self.result]
        if missing:
            return f"Missing: {missing}", "All keys present", True
        return True, True, False

    def mean_x_check(self):
        if self.result is None or 'mean_x' not in self.result:
            return None, self.reference_checks["mean_x"], True
        got = self.result['mean_x']
        return self._check("mean_x", got)

    def mean_y_check(self):
        if self.result is None or 'mean_y' not in self.result:
            return None, self.reference_checks["mean_y"], True
        got = self.result['mean_y']
        return self._check("mean_y", got)

    def var_x_check(self):
        if self.result is None or 'var_x' not in self.result:
            return None, self.reference_checks["var_x"], True
        got = self.result['var_x']
        return self._check("var_x", got)

    def var_y_check(self):
        if self.result is None or 'var_y' not in self.result:
            return None, self.reference_checks["var_y"], True
        got = self.result['var_y']
        return self._check("var_y", got)

    def std_x_check(self):
        if self.result is None or 'std_x' not in self.result:
            return None, self.reference_checks["std_x"], True
        got = self.result['std_x']
        return self._check("std_x", got)

    def std_y_check(self):
        if self.result is None or 'std_y' not in self.result:
            return None, self.reference_checks["std_y"], True
        got = self.result['std_y']
        return self._check("std_y", got)

    def covariance_check(self):
        if self.result is None or 'covariance' not in self.result:
            return None, self.reference_checks["covariance"], True
        got = self.result['covariance']
        return self._check("covariance", got)

    def correlation_check(self):
        if self.result is None or 'correlation' not in self.result:
            return None, self.reference_checks["correlation"], True
        got = self.result['correlation']
        return self._check("correlation", got)
