"""
Unit test utilities for the Missingness Pattern Analysis assignment.

This module provides battery test classes that define test cases
and reference values for each exercise.
"""

import numpy as np
import pandas as pd
from scipy import stats


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
        elif isinstance(want, pd.DataFrame):
            if got is None:
                condition = True
            else:
                try:
                    pd.testing.assert_frame_equal(got, want, rtol=1e-5, atol=1e-8)
                    condition = False
                except AssertionError:
                    condition = True
        elif isinstance(want, float):
            if got is None:
                condition = True
            else:
                condition = not np.isclose(got, want, rtol=1e-5, atol=1e-8)
        else:
            condition = got != want
        return got, want, condition


class CreateMissingnessIndicatorsBattery(TestBattery):
    """Test battery for Exercise 1: create_missingness_indicators function."""

    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        # Create test DataFrame with known missing values
        self.test_df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0],
            'B': [np.nan, 2.0, 3.0, np.nan, 5.0],
            'C': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

    def extract_info(self):
        """Execute the learner's function and store results."""
        try:
            self.result = self.learner_object(self.test_df)
        except Exception:
            self.result = None

    def get_reference_checks(self):
        """Define expected values for each check."""
        # Expected missingness indicators
        expected_indicators = pd.DataFrame({
            'A_missing': [0, 0, 1, 0, 0],
            'B_missing': [1, 0, 0, 1, 0],
            'C_missing': [0, 0, 0, 0, 0]
        })

        self.reference_checks = {
            "is_dataframe": True,
            "correct_shape": (5, 3),
            "correct_columns": ['A_missing', 'B_missing', 'C_missing'],
            "A_missing_values": np.array([0, 0, 1, 0, 0]),
            "B_missing_values": np.array([1, 0, 0, 1, 0]),
            "all_binary": True,
        }

    def result_is_dataframe(self):
        if self.result is None:
            return None, True, True
        got = isinstance(self.result, pd.DataFrame)
        want = True
        return got, want, not got

    def correct_shape_check(self):
        if self.result is None:
            return None, (5, 3), True
        got = self.result.shape
        want = (5, 3)
        return got, want, got != want

    def correct_columns_check(self):
        if self.result is None:
            return None, ['A_missing', 'B_missing', 'C_missing'], True
        got = list(self.result.columns)
        want = ['A_missing', 'B_missing', 'C_missing']
        return got, want, got != want

    def a_missing_values_check(self):
        if self.result is None or 'A_missing' not in self.result.columns:
            return None, self.reference_checks["A_missing_values"], True
        got = self.result['A_missing'].values
        want = self.reference_checks["A_missing_values"]
        return got, want, not np.array_equal(got, want)

    def b_missing_values_check(self):
        if self.result is None or 'B_missing' not in self.result.columns:
            return None, self.reference_checks["B_missing_values"], True
        got = self.result['B_missing'].values
        want = self.reference_checks["B_missing_values"]
        return got, want, not np.array_equal(got, want)

    def all_binary_check(self):
        if self.result is None:
            return None, True, True
        # Check all values are 0 or 1
        all_binary = self.result.isin([0, 1]).all().all()
        return all_binary, True, not all_binary


class SimplifiedMCARTestBattery(TestBattery):
    """Test battery for Exercise 2: simplified_mcar_test function."""

    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        np.random.seed(42)
        n = 200

        # MCAR data - missingness is random
        self.mcar_df = pd.DataFrame({
            'age': np.random.normal(40, 10, n),
            'income': np.random.normal(50000, 15000, n),
            'score': np.random.normal(75, 10, n)
        })
        # Apply random MCAR missingness
        mcar_mask = np.random.random(n) < 0.2
        self.mcar_df.loc[mcar_mask, 'income'] = np.nan

        # Non-MCAR data - missingness depends on age
        self.non_mcar_df = pd.DataFrame({
            'age': np.random.normal(40, 10, n),
            'income': np.random.normal(50000, 15000, n),
            'score': np.random.normal(75, 10, n)
        })
        # Higher age = more likely to be missing
        age_normalized = (self.non_mcar_df['age'] - self.non_mcar_df['age'].min()) / \
                        (self.non_mcar_df['age'].max() - self.non_mcar_df['age'].min())
        non_mcar_mask = np.random.random(n) < (0.1 + 0.5 * age_normalized)
        self.non_mcar_df.loc[non_mcar_mask, 'income'] = np.nan

    def extract_info(self):
        """Execute the learner's function and store results."""
        try:
            self.result_mcar = self.learner_object(self.mcar_df, 'income', 'age')
            self.result_non_mcar = self.learner_object(self.non_mcar_df, 'income', 'age')
        except Exception as e:
            self.result_mcar = None
            self.result_non_mcar = None
            self.error = str(e)

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {
            "is_tuple": True,
            "has_two_elements": 2,
        }

    def result_is_tuple_mcar(self):
        if self.result_mcar is None:
            return None, True, True
        got = isinstance(self.result_mcar, tuple)
        want = True
        return got, want, not got

    def result_has_two_elements(self):
        if self.result_mcar is None or not isinstance(self.result_mcar, tuple):
            return None, 2, True
        got = len(self.result_mcar)
        want = 2
        return got, want, got != want

    def statistic_is_numeric(self):
        if self.result_mcar is None or not isinstance(self.result_mcar, tuple):
            return None, "numeric", True
        got = isinstance(self.result_mcar[0], (int, float, np.number))
        want = True
        return got, want, not got

    def pvalue_is_numeric(self):
        if self.result_mcar is None or not isinstance(self.result_mcar, tuple):
            return None, "numeric", True
        got = isinstance(self.result_mcar[1], (int, float, np.number))
        want = True
        return got, want, not got

    def pvalue_in_valid_range(self):
        if self.result_mcar is None or not isinstance(self.result_mcar, tuple):
            return None, "between 0 and 1", True
        p_value = self.result_mcar[1]
        got = 0 <= p_value <= 1
        want = True
        return p_value, "p-value in [0, 1]", not got

    def mcar_high_pvalue(self):
        """For MCAR data, p-value should typically be high (> 0.05)."""
        if self.result_mcar is None or not isinstance(self.result_mcar, tuple):
            return None, "> 0.05", True
        p_value = self.result_mcar[1]
        # We check if it's reasonably high for MCAR
        got = p_value
        want = "> 0.05 (typically for MCAR)"
        # Note: This is a soft check since p-values can vary
        return got, want, p_value < 0.01  # Only fail if very low


class TestMARPatternBattery(TestBattery):
    """Test battery for Exercise 3: test_mar_pattern function."""

    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        np.random.seed(42)
        n = 300

        # MAR data - missingness in income depends on age
        self.mar_df = pd.DataFrame({
            'age': np.random.normal(40, 10, n),
            'income': np.random.normal(50000, 15000, n),
            'education': np.random.normal(14, 3, n)
        })
        age_normalized = (self.mar_df['age'] - self.mar_df['age'].min()) / \
                        (self.mar_df['age'].max() - self.mar_df['age'].min())
        mar_mask = np.random.random(n) < (0.05 + 0.85 * age_normalized)
        self.mar_df.loc[mar_mask, 'income'] = np.nan

        # Non-MAR data (MCAR) - missingness is random
        self.non_mar_df = pd.DataFrame({
            'age': np.random.normal(40, 10, n),
            'income': np.random.normal(50000, 15000, n),
            'education': np.random.normal(14, 3, n)
        })
        mcar_mask = np.random.random(n) < 0.2
        self.non_mar_df.loc[mcar_mask, 'income'] = np.nan

    def extract_info(self):
        """Execute the learner's function and store results."""
        try:
            self.result_mar = self.learner_object(self.mar_df, 'income', 'age')
            self.result_non_mar = self.learner_object(self.non_mar_df, 'income', 'age')
        except Exception as e:
            self.result_mar = None
            self.result_non_mar = None
            self.error = str(e)

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {}

    def result_is_tuple_mar(self):
        if self.result_mar is None:
            return None, True, True
        got = isinstance(self.result_mar, tuple)
        want = True
        return got, want, not got

    def result_has_two_elements(self):
        if self.result_mar is None or not isinstance(self.result_mar, tuple):
            return None, 2, True
        got = len(self.result_mar)
        want = 2
        return got, want, got != want

    def statistic_is_numeric(self):
        if self.result_mar is None or not isinstance(self.result_mar, tuple):
            return None, "numeric", True
        got = isinstance(self.result_mar[0], (int, float, np.number))
        want = True
        return got, want, not got

    def pvalue_is_numeric(self):
        if self.result_mar is None or not isinstance(self.result_mar, tuple):
            return None, "numeric", True
        got = isinstance(self.result_mar[1], (int, float, np.number))
        want = True
        return got, want, not got

    def pvalue_in_valid_range(self):
        if self.result_mar is None or not isinstance(self.result_mar, tuple):
            return None, "between 0 and 1", True
        p_value = self.result_mar[1]
        got = 0 <= p_value <= 1
        want = True
        return p_value, "p-value in [0, 1]", not got

    def mar_low_pvalue(self):
        """For MAR data, p-value should typically be low (< 0.05)."""
        if self.result_mar is None or not isinstance(self.result_mar, tuple):
            return None, "< 0.05", True
        p_value = self.result_mar[1]
        got = p_value
        want = "< 0.05 (typically for MAR)"
        # MAR should show significant relationship
        return got, want, p_value > 0.1  # Only fail if very high


class MissingnessCorrelationBattery(TestBattery):
    """Test battery for Exercise 4: compute_missingness_correlation function."""

    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        np.random.seed(42)
        n = 200

        # Create data with correlated missingness
        self.test_df = pd.DataFrame({
            'A': np.random.normal(50, 10, n),
            'B': np.random.normal(50, 10, n),
            'C': np.random.normal(50, 10, n),
            'D': np.random.normal(50, 10, n)
        })

        # A and B have correlated missingness (both missing for same rows)
        shared_mask = np.random.random(n) < 0.2
        self.test_df.loc[shared_mask, 'A'] = np.nan
        self.test_df.loc[shared_mask, 'B'] = np.nan

        # C has independent missingness
        c_mask = np.random.random(n) < 0.15
        self.test_df.loc[c_mask, 'C'] = np.nan

        # D has no missing values
        # (already complete)

    def extract_info(self):
        """Execute the learner's function and store results."""
        try:
            self.result = self.learner_object(self.test_df)
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {
            "is_dataframe": True,
            "is_square": True,
        }

    def result_is_dataframe(self):
        if self.result is None:
            return None, True, True
        got = isinstance(self.result, pd.DataFrame)
        want = True
        return got, want, not got

    def result_is_square(self):
        if self.result is None or not isinstance(self.result, pd.DataFrame):
            return None, True, True
        got = self.result.shape[0] == self.result.shape[1]
        want = True
        return self.result.shape, "square matrix", not got

    def diagonal_is_one(self):
        """Diagonal should be 1.0 (correlation with itself)."""
        if self.result is None or not isinstance(self.result, pd.DataFrame):
            return None, 1.0, True
        diagonal = np.diag(self.result.values)
        all_ones = np.allclose(diagonal, 1.0)
        return diagonal, "all 1.0", not all_ones

    def values_in_valid_range(self):
        """All correlations should be between -1 and 1."""
        if self.result is None or not isinstance(self.result, pd.DataFrame):
            return None, "[-1, 1]", True
        values = self.result.values
        in_range = np.all((values >= -1) & (values <= 1))
        return "correlation values", "in [-1, 1]", not in_range

    def correlated_missingness_detected(self):
        """A and B should have high positive correlation (correlated missingness)."""
        if self.result is None or not isinstance(self.result, pd.DataFrame):
            return None, "> 0.5", True

        if 'A' not in self.result.columns or 'B' not in self.result.columns:
            # Check with _missing suffix
            if 'A_missing' in self.result.columns and 'B_missing' in self.result.columns:
                corr_ab = self.result.loc['A_missing', 'B_missing']
            else:
                return None, "> 0.5", True
        else:
            corr_ab = self.result.loc['A', 'B']

        got = corr_ab
        want = "> 0.5 (A and B have correlated missingness)"
        return got, want, corr_ab < 0.5


class ClassifyMissingnessTypeBattery(TestBattery):
    """Test battery for Exercise 5: classify_missingness_type function."""

    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        np.random.seed(42)
        n = 300

        # MCAR data
        self.mcar_df = pd.DataFrame({
            'age': np.random.normal(40, 10, n),
            'income': np.random.normal(50000, 15000, n),
            'education': np.random.normal(14, 3, n)
        })
        mcar_mask = np.random.random(n) < 0.2
        self.mcar_df.loc[mcar_mask, 'income'] = np.nan

        # MAR data
        self.mar_df = pd.DataFrame({
            'age': np.random.normal(40, 10, n),
            'income': np.random.normal(50000, 15000, n),
            'education': np.random.normal(14, 3, n)
        })
        age_normalized = (self.mar_df['age'] - self.mar_df['age'].min()) / \
                        (self.mar_df['age'].max() - self.mar_df['age'].min())
        mar_mask = np.random.random(n) < (0.1 + 0.6 * age_normalized)
        self.mar_df.loc[mar_mask, 'income'] = np.nan

    def extract_info(self):
        """Execute the learner's function and store results."""
        try:
            self.result_mcar = self.learner_object(
                self.mcar_df, 'income', ['age', 'education']
            )
            self.result_mar = self.learner_object(
                self.mar_df, 'income', ['age', 'education']
            )
        except Exception as e:
            self.result_mcar = None
            self.result_mar = None
            self.error = str(e)

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {
            "valid_types": ['MCAR', 'MAR', 'MNAR', 'Unknown'],
        }

    def result_is_dict_mcar(self):
        if self.result_mcar is None:
            return None, True, True
        got = isinstance(self.result_mcar, dict)
        want = True
        return got, want, not got

    def result_has_classification_key(self):
        if self.result_mcar is None or not isinstance(self.result_mcar, dict):
            return None, "'classification' key", True
        got = 'classification' in self.result_mcar
        want = True
        return got, want, not got

    def classification_is_valid_type(self):
        if self.result_mcar is None or not isinstance(self.result_mcar, dict):
            return None, "valid type", True
        if 'classification' not in self.result_mcar:
            return None, "valid type", True
        classification = self.result_mcar['classification']
        valid_types = ['MCAR', 'MAR', 'MNAR', 'Unknown']
        got = classification in valid_types
        return classification, f"one of {valid_types}", not got

    def mcar_classified_correctly(self):
        """MCAR data should be classified as MCAR."""
        if self.result_mcar is None or not isinstance(self.result_mcar, dict):
            return None, "MCAR", True
        if 'classification' not in self.result_mcar:
            return None, "MCAR", True
        got = self.result_mcar['classification']
        want = "MCAR"
        # Allow some flexibility since statistical tests can vary
        return got, want, got not in ['MCAR', 'Unknown']

    def mar_classified_correctly(self):
        """MAR data should be classified as MAR."""
        if self.result_mar is None or not isinstance(self.result_mar, dict):
            return None, "MAR", True
        if 'classification' not in self.result_mar:
            return None, "MAR", True
        got = self.result_mar['classification']
        want = "MAR"
        # MAR should definitely not be classified as MCAR
        return got, want, got == 'MCAR'

    def result_has_evidence_key(self):
        if self.result_mcar is None or not isinstance(self.result_mcar, dict):
            return None, "'evidence' key", True
        got = 'evidence' in self.result_mcar or 'p_values' in self.result_mcar
        want = True
        return got, want, not got
