"""
Unit test utilities for the Data Type Conversion assignment.

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
# Exercise 1 – convert_to_numeric
# ---------------------------------------------------------------------------

class ConvertToNumericBattery(TestBattery):
    """Test battery for Exercise 1: convert_to_numeric(df, column, errors='coerce')."""

    def _get_reference_inputs(self):
        # '1.0', '2.5' → floats; 'abc' → NaN; '4' → 4.0; 'N/A' → NaN
        self.test_df = pd.DataFrame({"vals": ["1.0", "2.5", "abc", "4", "N/A"]})

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy(), "vals", errors="coerce")
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "first_value": 1.0,
            "n_nan": 2,
            "n_elements": 5,
        }

    def result_is_series(self):
        got = isinstance(self.result, pd.Series)
        return got, True, not got

    def result_is_numeric(self):
        if self.result is None:
            return False, True, True
        got = pd.api.types.is_numeric_dtype(self.result)
        return got, True, not got

    def first_value_check(self):
        """'1.0' should convert to 1.0."""
        if self.result is None:
            return None, self.reference_checks["first_value"], True
        try:
            got = float(self.result.iloc[0])
        except Exception:
            got = None
        return self._check("first_value", got)

    def abc_is_nan_check(self):
        """'abc' at position 2 should be coerced to NaN."""
        if self.result is None:
            return None, True, True
        try:
            got = bool(pd.isna(self.result.iloc[2]))
        except Exception:
            got = False
        return got, True, got is not True

    def nan_count_check(self):
        """Both 'abc' and 'N/A' should become NaN → 2 total NaN values."""
        if self.result is None:
            return None, self.reference_checks["n_nan"], True
        try:
            got = int(self.result.isna().sum())
        except Exception:
            got = None
        return self._check("n_nan", got)


# ---------------------------------------------------------------------------
# Exercise 2 – convert_to_datetime
# ---------------------------------------------------------------------------

class ConvertToDatetimeBattery(TestBattery):
    """Test battery for Exercise 2: convert_to_datetime(df, column, dayfirst=False)."""

    def _get_reference_inputs(self):
        self.test_df = pd.DataFrame(
            {"dates": ["2023-01-15", "2023-06-30", "not-a-date"]}
        )

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.test_df.copy(), "dates", dayfirst=False
            )
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "n_nat": 1,
        }

    def result_is_series(self):
        got = isinstance(self.result, pd.Series)
        return got, True, not got

    def result_is_datetime(self):
        if self.result is None:
            return False, True, True
        got = pd.api.types.is_datetime64_any_dtype(self.result)
        return got, True, not got

    def invalid_is_nat_check(self):
        """'not-a-date' at position 2 should become NaT."""
        if self.result is None:
            return None, True, True
        try:
            got = bool(pd.isna(self.result.iloc[2]))
        except Exception:
            got = False
        return got, True, got is not True

    def valid_dates_not_nat(self):
        """First two valid dates must NOT be NaT."""
        if self.result is None:
            return None, 0, True
        try:
            got = int(self.result.iloc[:2].isna().sum())
        except Exception:
            got = None
        return got, 0, got != 0

    def nat_count_check(self):
        """Exactly 1 NaT expected (from 'not-a-date')."""
        if self.result is None:
            return None, self.reference_checks["n_nat"], True
        try:
            got = int(self.result.isna().sum())
        except Exception:
            got = None
        return self._check("n_nat", got)


# ---------------------------------------------------------------------------
# Exercise 3 – convert_to_categorical
# ---------------------------------------------------------------------------

class ConvertToCategoricalBattery(TestBattery):
    """Test battery for Exercise 3: convert_to_categorical(df, column, categories=None, ordered=False)."""

    def _get_reference_inputs(self):
        self.test_df = pd.DataFrame(
            {"cat": ["low", "high", "medium", "low", "high"]}
        )

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.test_df.copy(), "cat", categories=None, ordered=False
            )
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "n_categories": 3,
            "n_nan": 0,
        }

    def result_is_series(self):
        got = isinstance(self.result, pd.Series)
        return got, True, not got

    def result_is_categorical(self):
        if self.result is None:
            return False, True, True
        try:
            got = pd.api.types.is_categorical_dtype(self.result)
        except Exception:
            got = False
        return got, True, not got

    def no_nan_introduced_check(self):
        """Conversion must not introduce NaN values."""
        if self.result is None:
            return None, self.reference_checks["n_nan"], True
        try:
            got = int(self.result.isna().sum())
        except Exception:
            got = None
        return self._check("n_nan", got)

    def n_categories_check(self):
        """3 unique categories: 'low', 'medium', 'high'."""
        if self.result is None:
            return None, self.reference_checks["n_categories"], True
        try:
            got = len(self.result.cat.categories)
        except Exception:
            got = None
        return self._check("n_categories", got)


# ---------------------------------------------------------------------------
# Exercise 4 – safe_numeric_conversion
# ---------------------------------------------------------------------------

class SafeNumericConversionBattery(TestBattery):
    """Test battery for Exercise 4: safe_numeric_conversion(df, columns) → (df_converted, error_counts)."""

    def _get_reference_inputs(self):
        # column 'a' has one non-numeric value 'x'; column 'b' is all numeric
        self.test_df = pd.DataFrame(
            {
                "a": ["1", "2", "x"],
                "b": ["10", "20", "30"],
            }
        )
        self.test_columns = ["a", "b"]

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.test_df.copy(), self.test_columns
            )
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "a_errors": 1,
            "b_errors": 0,
        }

    def result_is_tuple(self):
        got = isinstance(self.result, tuple)
        return got, True, not got

    def first_element_is_dataframe(self):
        if not isinstance(self.result, tuple):
            return False, True, True
        got = isinstance(self.result[0], pd.DataFrame)
        return got, True, not got

    def second_element_is_dict(self):
        if not isinstance(self.result, tuple):
            return False, True, True
        got = isinstance(self.result[1], dict)
        return got, True, not got

    def a_errors_check(self):
        """Column 'a' has 'x' which cannot be converted → 1 error."""
        if not isinstance(self.result, tuple):
            return None, self.reference_checks["a_errors"], True
        try:
            got = int(self.result[1]["a"])
        except Exception:
            got = None
        return self._check("a_errors", got)

    def b_errors_check(self):
        """Column 'b' has only valid numerics → 0 errors."""
        if not isinstance(self.result, tuple):
            return None, self.reference_checks["b_errors"], True
        try:
            got = int(self.result[1]["b"])
        except Exception:
            got = None
        return self._check("b_errors", got)


# ---------------------------------------------------------------------------
# Exercise 5 – create_conversion_report
# ---------------------------------------------------------------------------

class ConversionReportBattery(TestBattery):
    """Test battery for Exercise 5: create_conversion_report(original_df, converted_df)."""

    def _get_reference_inputs(self):
        # original: all object dtype; converted: two numeric columns
        self.original_df = pd.DataFrame(
            {
                "age":   ["25", "30", "45"],
                "price": ["1.99", "2.50", "3.00"],
                "label": ["a", "b", "c"],
            }
        )
        self.converted_df = pd.DataFrame(
            {
                "age":   [25.0, 30.0, 45.0],
                "price": [1.99, 2.50, 3.00],
                "label": ["a", "b", "c"],
            }
        )

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.original_df.copy(), self.converted_df.copy()
            )
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "min_columns_converted": 1,
        }

    def result_is_dict(self):
        got = isinstance(self.result, dict)
        return got, True, not got

    def has_dtype_changes_key(self):
        if not isinstance(self.result, dict):
            return False, True, True
        got = "dtype_changes" in self.result
        return got, True, not got

    def has_columns_converted_key(self):
        if not isinstance(self.result, dict):
            return False, True, True
        got = "columns_converted" in self.result
        return got, True, not got

    def dtype_changes_is_list(self):
        if not isinstance(self.result, dict):
            return False, True, True
        got = isinstance(self.result.get("dtype_changes"), list)
        return got, True, not got

    def columns_converted_at_least_one(self):
        """'columns_converted' must be >= 1 (two object cols become float)."""
        if not isinstance(self.result, dict):
            return None, self.reference_checks["min_columns_converted"], True
        try:
            got = self.result.get("columns_converted", 0)
            failed = got < self.reference_checks["min_columns_converted"]
        except Exception:
            got = None
            failed = True
        return got, self.reference_checks["min_columns_converted"], failed
