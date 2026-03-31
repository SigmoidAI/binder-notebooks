"""
Unit test utilities for the Outlier Investigation assignment.

This module provides battery test classes that define test cases
and reference values for each exercise.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

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
                except Exception:
                    condition = True
        elif isinstance(want, pd.DataFrame):
            if got is None:
                condition = True
            elif not isinstance(got, pd.DataFrame):
                condition = True
            else:
                try:
                    if got.shape != want.shape:
                        condition = True
                    else:
                        condition = not np.allclose(
                            got.select_dtypes(include=[np.number]).values,
                            want.select_dtypes(include=[np.number]).values,
                            rtol=1e-5,
                            atol=1e-8,
                            equal_nan=True,
                        )
                except Exception:
                    condition = True
        elif isinstance(want, np.ndarray):
            if got is None:
                condition = True
            else:
                condition = not np.allclose(
                    got, want, rtol=1e-5, atol=1e-8, equal_nan=True
                )
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


# ---------------------------------------------------------------------------
# Exercise 1 – detect_iqr_outliers
# ---------------------------------------------------------------------------

class DetectIQROutliersBattery(TestBattery):
    """Test battery for Exercise 1: detect_iqr_outliers(df, column, factor=1.5)."""

    def _get_reference_inputs(self):
        # 100 is a clear IQR outlier among [1..9]
        # Q1≈3.25, Q3≈7.75, IQR=4.5  →  upper fence = 7.75 + 1.5*4.5 = 14.5
        self.test_df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.test_df.copy(), "values", factor=1.5
            )
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "n_elements": 10,
        }

    def result_is_boolean_series(self):
        if self.result is None:
            return False, True, True
        got = isinstance(self.result, pd.Series) and pd.api.types.is_bool_dtype(
            self.result
        )
        return got, True, not got

    def correct_length(self):
        if self.result is None:
            return None, self.reference_checks["n_elements"], True
        got = len(self.result)
        return self._check("n_elements", got)

    def outlier_100_flagged(self):
        """Value 100 (last element) must be True."""
        if self.result is None:
            return None, True, True
        try:
            got = bool(self.result.iloc[-1])
        except Exception:
            got = None
        return got, True, got is not True

    def non_outliers_clean(self):
        """Values 1–9 should not be flagged as outliers."""
        if self.result is None:
            return None, 0, True
        try:
            got = int(self.result.iloc[:-1].sum())
        except Exception:
            got = None
        return got, 0, got != 0


# ---------------------------------------------------------------------------
# Exercise 2 – detect_zscore_outliers
# ---------------------------------------------------------------------------

class DetectZScoreOutliersBattery(TestBattery):
    """Test battery for Exercise 2: detect_zscore_outliers(df, column, threshold=3.0).

    The test calls the function with threshold=2.0 so that the value 100
    (z ≈ 2.47 with ddof=1, or ≈ 2.65 with ddof=0) is reliably detected.
    """

    def _get_reference_inputs(self):
        self.test_df = pd.DataFrame(
            {"values": [10, 11, 10, 12, 10, 11, 10, 100]}
        )

    def extract_info(self):
        try:
            # threshold=2.0 ensures 100 is flagged regardless of ddof choice
            self.result = self.learner_object(
                self.test_df.copy(), "values", threshold=2.0
            )
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "n_elements": 8,
        }

    def result_is_boolean_series(self):
        if self.result is None:
            return False, True, True
        got = isinstance(self.result, pd.Series) and pd.api.types.is_bool_dtype(
            self.result
        )
        return got, True, not got

    def correct_length(self):
        if self.result is None:
            return None, self.reference_checks["n_elements"], True
        got = len(self.result)
        return self._check("n_elements", got)

    def outlier_100_flagged(self):
        """Value 100 (last element) must be True."""
        if self.result is None:
            return None, True, True
        try:
            got = bool(self.result.iloc[-1])
        except Exception:
            got = None
        return got, True, got is not True


# ---------------------------------------------------------------------------
# Exercise 3 – detect_isolation_forest_outliers
# ---------------------------------------------------------------------------

class DetectIsolationForestBattery(TestBattery):
    """Test battery for Exercise 3: detect_isolation_forest_outliers(
        df, columns=None, contamination=0.1, random_state=42)."""

    def _get_reference_inputs(self):
        rng = np.random.default_rng(0)
        normal_rows = pd.DataFrame(
            {
                "feature_1": rng.normal(0, 1, 20),
                "feature_2": rng.normal(0, 1, 20),
            }
        )
        extreme_rows = pd.DataFrame(
            {
                "feature_1": [100.0, -100.0],
                "feature_2": [100.0, -100.0],
            }
        )
        self.test_df = pd.concat(
            [normal_rows, extreme_rows], ignore_index=True
        )

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.test_df.copy(),
                columns=None,
                contamination=0.1,
                random_state=42,
            )
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "n_elements": 22,
        }

    def result_is_boolean_series(self):
        if self.result is None:
            return False, True, True
        got = isinstance(self.result, pd.Series) and pd.api.types.is_bool_dtype(
            self.result
        )
        return got, True, not got

    def correct_length(self):
        if self.result is None:
            return None, self.reference_checks["n_elements"], True
        got = len(self.result)
        return self._check("n_elements", got)

    def extreme_rows_flagged(self):
        """At least one of the two extreme rows (indices 20, 21) must be True."""
        if self.result is None:
            return None, True, True
        try:
            got = bool(self.result.iloc[20]) or bool(self.result.iloc[21])
        except Exception:
            got = False
        return got, True, not got


# ---------------------------------------------------------------------------
# Exercise 4 – remove_outliers
# ---------------------------------------------------------------------------

class RemoveOutliersBattery(TestBattery):
    """Test battery for Exercise 4: remove_outliers(df, outlier_mask)."""

    def _get_reference_inputs(self):
        self.test_df = pd.DataFrame(
            {
                "a": [10, 20, 30, 40, 50],
                "b": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        # Rows at index 1 and 4 are "outliers"
        self.mask = pd.Series([False, True, False, False, True])

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.test_df.copy(), self.mask.copy()
            )
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        # 5 rows − 2 outliers = 3 clean rows
        self.reference_checks = {
            "n_rows": 3,
        }

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        return got, True, not got

    def correct_row_count(self):
        if self.result is None:
            return None, self.reference_checks["n_rows"], True
        got = len(self.result)
        return self._check("n_rows", got)

    def index_is_reset(self):
        if self.result is None:
            return None, True, True
        expected = list(range(len(self.result)))
        got = list(self.result.index)
        return got, expected, got != expected


# ---------------------------------------------------------------------------
# Exercise 5 – create_outlier_summary
# ---------------------------------------------------------------------------

class OutlierSummaryBattery(TestBattery):
    """Test battery for Exercise 5: create_outlier_summary(df, columns=None)."""

    def _get_reference_inputs(self):
        # 10 rows; last two have extreme values in 'values' column
        self.test_df = pd.DataFrame(
            {
                "values": [10, 11, 10, 12, 10, 11, 10, 12, 1000, -1000],
                "other": [1.0, 2.0, 1.5, 1.8, 1.2, 2.1, 1.7, 2.0, 1.3, 1.9],
            }
        )

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy(), columns=None)
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {}

    def result_is_dict(self):
        got = isinstance(self.result, dict)
        return got, True, not got

    def has_required_keys(self):
        if self.result is None or not isinstance(self.result, dict):
            return None, True, True
        required = {"total_outliers", "clean_rows", "outlier_pct"}
        got = set(self.result.keys())
        failed = not required.issubset(got)
        return got, required, failed

    def total_outliers_at_least_one(self):
        """Expect at least 1 outlier detected in the test data."""
        if self.result is None or not isinstance(self.result, dict):
            return None, ">=1", True
        try:
            got = self.result.get("total_outliers", 0)
            failed = got < 1
        except Exception:
            got = None
            failed = True
        return got, ">=1", failed
