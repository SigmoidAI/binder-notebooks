"""
Unit test utilities for the Inconsistency Resolution assignment.

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
# Exercise 1 – standardize_country
# ---------------------------------------------------------------------------

class StandardizeCountryBattery(TestBattery):
    """Test battery for Exercise 1: standardize_country(series) → pd.Series."""

    def _get_reference_inputs(self):
        self.test_series = pd.Series(["USA", "US", "United States", "u.s.a"])

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_series.copy())
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "length":         4,
            "n_unique":       1,
            "expected_value": "United States",
        }

    def result_is_series(self):
        got = isinstance(self.result, pd.Series)
        return got, True, not got

    def length_preserved_check(self):
        """Output length must equal input length."""
        if self.result is None:
            return None, self.reference_checks["length"], True
        try:
            got = len(self.result)
        except Exception:
            got = None
        return self._check("length", got)

    def all_values_standardized_check(self):
        """All values must collapse to a single unique value."""
        if self.result is None:
            return None, self.reference_checks["n_unique"], True
        try:
            got = int(self.result.nunique())
        except Exception:
            got = None
        return self._check("n_unique", got)

    def value_is_united_states_check(self):
        """First value must be 'United States'."""
        if self.result is None:
            return None, self.reference_checks["expected_value"], True
        try:
            got = str(self.result.iloc[0])
        except Exception:
            got = None
        return self._check("expected_value", got)


# ---------------------------------------------------------------------------
# Exercise 2 – parse_currency
# ---------------------------------------------------------------------------

class ParseCurrencyBattery(TestBattery):
    """Test battery for Exercise 2: parse_currency(series) → pd.Series of float."""

    def _get_reference_inputs(self):
        self.test_series = pd.Series(["$1,200.50", "900.00 USD", "1200", "$50"])

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_series.copy())
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "first_value": 1200.50,
            "length":      4,
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
        """'$1,200.50' should parse to 1200.50."""
        if self.result is None:
            return None, self.reference_checks["first_value"], True
        try:
            got = float(self.result.iloc[0])
        except Exception:
            got = None
        return self._check("first_value", got)

    def length_preserved_check(self):
        """Output length must equal input length."""
        if self.result is None:
            return None, self.reference_checks["length"], True
        try:
            got = len(self.result)
        except Exception:
            got = None
        return self._check("length", got)


# ---------------------------------------------------------------------------
# Exercise 3 – standardize_date_format
# ---------------------------------------------------------------------------

class StandardizeDateBattery(TestBattery):
    """Test battery for Exercise 3: standardize_date_format(series, output_format='%Y-%m-%d') → pd.Series of str."""

    def _get_reference_inputs(self):
        self.test_series = pd.Series(["2023-01-15", "01/15/2023", "not-a-date"])

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.test_series.copy(), output_format="%Y-%m-%d"
            )
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "length":      3,
            "first_value": "2023-01-15",
        }

    def result_is_series(self):
        got = isinstance(self.result, pd.Series)
        return got, True, not got

    def length_preserved_check(self):
        """Output length must equal input length."""
        if self.result is None:
            return None, self.reference_checks["length"], True
        try:
            got = len(self.result)
        except Exception:
            got = None
        return self._check("length", got)

    def first_value_check(self):
        """'2023-01-15' (already in correct format) should remain '2023-01-15'."""
        if self.result is None:
            return None, self.reference_checks["first_value"], True
        try:
            got = str(self.result.iloc[0])
        except Exception:
            got = None
        return self._check("first_value", got)

    def valid_format_check(self):
        """'01/15/2023' at position 1 should be converted to YYYY-MM-DD."""
        if self.result is None:
            return None, True, True
        try:
            val = self.result.iloc[1]
            if pd.isna(val):
                got = False
            else:
                import re
                got = bool(re.match(r"^\d{4}-\d{2}-\d{2}$", str(val)))
        except Exception:
            got = False
        return got, True, not got

    def invalid_date_handled_check(self):
        """'not-a-date' at position 2 should result in NaN/NaT or empty string."""
        if self.result is None:
            return None, True, True
        try:
            val = self.result.iloc[2]
            if pd.isna(val):
                got = True
            else:
                got = str(val).strip() == ""
        except Exception:
            got = False
        return got, True, not got


# ---------------------------------------------------------------------------
# Exercise 4 – parse_weight
# ---------------------------------------------------------------------------

class ParseWeightBattery(TestBattery):
    """Test battery for Exercise 4: parse_weight(series) → pd.Series of float."""

    def _get_reference_inputs(self):
        self.test_series = pd.Series(["72.5kg", "80.0 KG", "65", "", "abc"])

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_series.copy())
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "first_value":  72.5,
            "second_value": 80.0,
            "length":       5,
        }

    def result_is_series(self):
        got = isinstance(self.result, pd.Series)
        return got, True, not got

    def first_value_check(self):
        """'72.5kg' should parse to 72.5."""
        if self.result is None:
            return None, self.reference_checks["first_value"], True
        try:
            got = float(self.result.iloc[0])
        except Exception:
            got = None
        return self._check("first_value", got)

    def second_value_check(self):
        """'80.0 KG' should parse to 80.0."""
        if self.result is None:
            return None, self.reference_checks["second_value"], True
        try:
            got = float(self.result.iloc[1])
        except Exception:
            got = None
        return self._check("second_value", got)

    def empty_string_is_nan_check(self):
        """Empty string '' at position 3 should become NaN."""
        if self.result is None:
            return None, True, True
        try:
            got = bool(pd.isna(self.result.iloc[3]))
        except Exception:
            got = False
        return got, True, not got

    def abc_is_nan_check(self):
        """Non-numeric 'abc' at position 4 should become NaN."""
        if self.result is None:
            return None, True, True
        try:
            got = bool(pd.isna(self.result.iloc[4]))
        except Exception:
            got = False
        return got, True, not got


# ---------------------------------------------------------------------------
# Exercise 5 – standardize_text_case
# ---------------------------------------------------------------------------

class StandardizeTextBattery(TestBattery):
    """Test battery for Exercise 5: standardize_text_case(series, case='lower') → pd.Series."""

    def _get_reference_inputs(self):
        self.test_series = pd.Series(["Active", "ACTIVE", "active", "Inactive"])

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_series.copy(), case="lower")
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "n_unique": 2,
            "length":   4,
        }

    def result_is_series(self):
        got = isinstance(self.result, pd.Series)
        return got, True, not got

    def length_preserved_check(self):
        """Output length must equal input length."""
        if self.result is None:
            return None, self.reference_checks["length"], True
        try:
            got = len(self.result)
        except Exception:
            got = None
        return self._check("length", got)

    def all_lowercase_check(self):
        """All values must be lowercase when case='lower'."""
        if self.result is None:
            return None, True, True
        try:
            got = bool((self.result == self.result.str.lower()).all())
        except Exception:
            got = False
        return got, True, not got

    def n_unique_check(self):
        """case='lower' must collapse 3 Active-variants and Inactive to 2 unique values."""
        if self.result is None:
            return None, self.reference_checks["n_unique"], True
        try:
            got = int(self.result.nunique())
        except Exception:
            got = None
        return self._check("n_unique", got)

    def unique_values_check(self):
        """Unique values must be exactly {'active', 'inactive'}."""
        if self.result is None:
            return None, True, True
        try:
            got  = set(self.result.unique())
            want = {"active", "inactive"}
            return got, want, got != want
        except Exception:
            return None, {"active", "inactive"}, True
