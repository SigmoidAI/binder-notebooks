"""
Unit test utilities for the Missing Value Audit assignment.

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
        if isinstance(want, pd.Series):
            if got is None:
                condition = True
            elif not isinstance(got, pd.Series):
                condition = True
            else:
                try:
                    condition = not got.equals(want)
                except:
                    condition = True
        elif isinstance(want, pd.DataFrame):
            if got is None:
                condition = True
            elif not isinstance(got, pd.DataFrame):
                condition = True
            else:
                try:
                    # Check if DataFrames have same shape and columns
                    if got.shape != want.shape:
                        condition = True
                    else:
                        # Compare with tolerance for numeric columns
                        condition = not np.allclose(
                            got.select_dtypes(include=[np.number]).values,
                            want.select_dtypes(include=[np.number]).values,
                            rtol=1e-5, atol=1e-8, equal_nan=True
                        )
                except:
                    condition = True
        elif isinstance(want, np.ndarray):
            if got is None:
                condition = True
            else:
                condition = not np.allclose(got, want, rtol=1e-5, atol=1e-8, equal_nan=True)
        elif isinstance(want, float):
            if got is None:
                condition = True
            else:
                condition = not np.isclose(got, want, rtol=1e-5, atol=1e-8)
        elif isinstance(want, int):
            if got is None:
                condition = True
            else:
                condition = got != want
        else:
            condition = got != want
        return got, want, condition


class CountMissingBattery(TestBattery):
    """Test battery for Exercise 1: count_missing_values function."""

    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        # Create test DataFrame with known missing values
        self.test_df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0],
            'B': [np.nan, np.nan, 3.0, 4.0, 5.0],
            'C': [1.0, 2.0, 3.0, 4.0, 5.0],  # No missing
            'D': [np.nan, np.nan, np.nan, 4.0, 5.0],
        })

        # Test with empty DataFrame
        self.empty_df = pd.DataFrame({'A': [], 'B': []})

        # Test with all missing column
        self.all_missing_df = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': [1.0, 2.0, 3.0]
        })

    def extract_info(self):
        """Execute the learner's function and store results."""
        try:
            self.result = self.learner_object(self.test_df)
            self.result_empty = self.learner_object(self.empty_df)
            self.result_all_missing = self.learner_object(self.all_missing_df)
        except Exception as e:
            self.result = None
            self.result_empty = None
            self.result_all_missing = None
            self.error = str(e)

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {
            "count_A": 1,
            "count_B": 2,
            "count_C": 0,
            "count_D": 3,
            "total_missing": 6,
            "all_missing_A": 3,
        }

    def result_is_series(self):
        got = isinstance(self.result, pd.Series)
        want = True
        return got, want, not got

    def count_A_check(self):
        if self.result is None:
            return None, self.reference_checks["count_A"], True
        try:
            got = self.result['A']
        except:
            got = None
        return self._check("count_A", got)

    def count_B_check(self):
        if self.result is None:
            return None, self.reference_checks["count_B"], True
        try:
            got = self.result['B']
        except:
            got = None
        return self._check("count_B", got)

    def count_C_check(self):
        if self.result is None:
            return None, self.reference_checks["count_C"], True
        try:
            got = self.result['C']
        except:
            got = None
        return self._check("count_C", got)

    def count_D_check(self):
        if self.result is None:
            return None, self.reference_checks["count_D"], True
        try:
            got = self.result['D']
        except:
            got = None
        return self._check("count_D", got)

    def total_missing_check(self):
        if self.result is None:
            return None, self.reference_checks["total_missing"], True
        try:
            got = self.result.sum()
        except:
            got = None
        return self._check("total_missing", got)


class MissingPercentageBattery(TestBattery):
    """Test battery for Exercise 2: calculate_missing_percentage function."""

    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        self.test_df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 1 missing = 10%
            'B': [np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 2 missing = 20%
            'C': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 0 missing = 0%
            'D': [np.nan, np.nan, np.nan, np.nan, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0],  # 5 missing = 50%
        })

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
            "pct_A": 10.0,
            "pct_B": 20.0,
            "pct_C": 0.0,
            "pct_D": 50.0,
        }

    def result_is_series(self):
        got = isinstance(self.result, pd.Series)
        want = True
        return got, want, not got

    def pct_A_check(self):
        if self.result is None:
            return None, self.reference_checks["pct_A"], True
        try:
            got = self.result['A']
        except:
            got = None
        return self._check("pct_A", got)

    def pct_B_check(self):
        if self.result is None:
            return None, self.reference_checks["pct_B"], True
        try:
            got = self.result['B']
        except:
            got = None
        return self._check("pct_B", got)

    def pct_C_check(self):
        if self.result is None:
            return None, self.reference_checks["pct_C"], True
        try:
            got = self.result['C']
        except:
            got = None
        return self._check("pct_C", got)

    def pct_D_check(self):
        if self.result is None:
            return None, self.reference_checks["pct_D"], True
        try:
            got = self.result['D']
        except:
            got = None
        return self._check("pct_D", got)

    def values_are_percentages(self):
        """Check that values are between 0 and 100."""
        if self.result is None:
            return None, "values between 0 and 100", True
        try:
            all_valid = (self.result >= 0).all() and (self.result <= 100).all()
            return all_valid, True, not all_valid
        except:
            return None, "values between 0 and 100", True


class MissingPatternsBattery(TestBattery):
    """Test battery for Exercise 3: identify_missing_patterns function."""

    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        self.test_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [np.nan, np.nan, 3.0, 4.0, 5.0],
            'C': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        # Row 0: A present, B missing, C present -> pattern: B
        # Row 1: A missing, B missing, C present -> pattern: A, B
        # Row 2: all present -> no missing
        # Row 3: A missing, B present, C present -> pattern: A
        # Row 4: all present -> no missing

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
            "has_row_index": True,
            "has_pattern_column": True,
            "rows_with_missing": 4,  # Rows 0, 1, 3 have missing values, plus possibly row 4 depends on impl
        }

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        want = True
        return got, want, not got

    def has_required_columns(self):
        """Check that result has required structure."""
        if self.result is None:
            return None, "DataFrame with pattern info", True
        try:
            # Check for reasonable column names
            cols = self.result.columns.tolist()
            has_pattern = any('pattern' in str(c).lower() or 'missing' in str(c).lower() for c in cols)
            return has_pattern, True, not has_pattern
        except:
            return None, "DataFrame with pattern info", True

    def identifies_missing_rows(self):
        """Check that rows with missing values are identified."""
        if self.result is None:
            return None, "identifies rows with missing values", True
        try:
            # Should have at least some rows identified
            num_rows = len(self.result)
            # Rows 0, 1, 3 have missing values
            has_rows = num_rows >= 3
            return has_rows, True, not has_rows
        except:
            return None, "identifies rows with missing values", True


class MissingSummaryBattery(TestBattery):
    """Test battery for Exercise 4: create_missing_summary function."""

    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        self.test_df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'B': [np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'C': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'D': [np.nan, np.nan, np.nan, np.nan, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0],
        })

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
            "has_count_column": True,
            "has_percentage_column": True,
            "num_columns": 4,  # Should have 4 rows (one per column)
        }

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        want = True
        return got, want, not got

    def has_count_info(self):
        """Check that result has count information."""
        if self.result is None:
            return None, "DataFrame with count info", True
        try:
            cols = [str(c).lower() for c in self.result.columns]
            has_count = any('count' in c or 'missing' in c for c in cols)
            return has_count, True, not has_count
        except:
            return None, "DataFrame with count info", True

    def has_percentage_info(self):
        """Check that result has percentage information."""
        if self.result is None:
            return None, "DataFrame with percentage info", True
        try:
            cols = [str(c).lower() for c in self.result.columns]
            has_pct = any('percent' in c or 'pct' in c or '%' in c for c in cols)
            return has_pct, True, not has_pct
        except:
            return None, "DataFrame with percentage info", True

    def covers_all_columns(self):
        """Check that all original columns are in the summary."""
        if self.result is None:
            return None, "summary covers all columns", True
        try:
            # Should have one row per column or columns as index
            num_entries = len(self.result)
            has_all = num_entries >= 4
            return has_all, True, not has_all
        except:
            return None, "summary covers all columns", True

    def correct_count_values(self):
        """Check that count values are correct."""
        if self.result is None:
            return None, "correct missing counts", True
        try:
            # Find the count column
            count_col = None
            for col in self.result.columns:
                if 'count' in str(col).lower():
                    count_col = col
                    break

            if count_col is None:
                return None, "count column exists", True

            # Check if sum of counts matches expected (1+2+0+5=8)
            total_count = self.result[count_col].sum()
            expected_total = 8
            is_correct = abs(total_count - expected_total) < 0.01
            return is_correct, True, not is_correct
        except:
            return None, "correct missing counts", True


class HighMissingnessBattery(TestBattery):
    """Test battery for Exercise 5: detect_high_missingness function."""

    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        """Define test inputs."""
        self.test_df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 10%
            'B': [np.nan, np.nan, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 30%
            'C': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 0%
            'D': [np.nan, np.nan, np.nan, np.nan, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0],  # 50%
            'E': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0, 9.0, 10.0],  # 70%
        })
        self.threshold_20 = 20.0
        self.threshold_50 = 50.0

    def extract_info(self):
        """Execute the learner's function and store results."""
        try:
            self.result_20 = self.learner_object(self.test_df, self.threshold_20)
            self.result_50 = self.learner_object(self.test_df, self.threshold_50)
        except Exception as e:
            self.result_20 = None
            self.result_50 = None
            self.error = str(e)

    def get_reference_checks(self):
        """Define expected values for each check."""
        self.reference_checks = {
            "cols_above_20": ['B', 'D', 'E'],  # 30%, 50%, 70% > 20%
            "cols_above_50": ['E'],  # Only 70% > 50%
        }

    def result_is_list(self):
        got = isinstance(self.result_20, list)
        want = True
        return got, want, not got

    def threshold_20_check(self):
        """Check columns above 20% threshold."""
        if self.result_20 is None:
            return None, self.reference_checks["cols_above_20"], True
        try:
            expected = set(self.reference_checks["cols_above_20"])
            got_set = set(self.result_20)
            is_correct = expected == got_set
            return sorted(self.result_20), sorted(expected), not is_correct
        except:
            return None, self.reference_checks["cols_above_20"], True

    def threshold_50_check(self):
        """Check columns above 50% threshold."""
        if self.result_50 is None:
            return None, self.reference_checks["cols_above_50"], True
        try:
            expected = set(self.reference_checks["cols_above_50"])
            got_set = set(self.result_50)
            is_correct = expected == got_set
            return sorted(self.result_50), sorted(expected), not is_correct
        except:
            return None, self.reference_checks["cols_above_50"], True

    def excludes_low_missing(self):
        """Check that columns with low missingness are excluded."""
        if self.result_20 is None:
            return None, "excludes low missingness columns", True
        try:
            # A (10%) and C (0%) should not be in results
            has_A = 'A' in self.result_20
            has_C = 'C' in self.result_20
            is_correct = not has_A and not has_C
            return is_correct, True, not is_correct
        except:
            return None, "excludes low missingness columns", True
