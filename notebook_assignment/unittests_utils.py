"""
Unit test utilities for the Automated Data Validation assignment.

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
# Exercise 1 – validate_no_nulls
# ---------------------------------------------------------------------------

class ValidateNoNullsBattery(TestBattery):
    """
    Test battery for Exercise 1:
        validate_no_nulls(df, columns=None)
        → dict {col: {'passed': bool, 'null_count': int}}
    """

    def _get_reference_inputs(self):
        self.test_df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy())
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "result_is_dict":       True,
            "a_passed":             False,
            "a_null_count":         1,
            "b_passed":             True,
            "b_null_count":         0,
        }

    # ── individual checks ────────────────────────────────────────────────────

    def result_is_dict_check(self):
        got = isinstance(self.result, dict)
        return got, True, not got

    def a_passed_check(self):
        if self.result is None:
            return None, self.reference_checks["a_passed"], True
        try:
            got = bool(self.result["a"]["passed"])
        except Exception:
            got = None
        return self._check("a_passed", got)

    def a_null_count_check(self):
        if self.result is None:
            return None, self.reference_checks["a_null_count"], True
        try:
            got = int(self.result["a"]["null_count"])
        except Exception:
            got = None
        return self._check("a_null_count", got)

    def b_passed_check(self):
        if self.result is None:
            return None, self.reference_checks["b_passed"], True
        try:
            got = bool(self.result["b"]["passed"])
        except Exception:
            got = None
        return self._check("b_passed", got)

    def b_null_count_check(self):
        if self.result is None:
            return None, self.reference_checks["b_null_count"], True
        try:
            got = int(self.result["b"]["null_count"])
        except Exception:
            got = None
        return self._check("b_null_count", got)


# ---------------------------------------------------------------------------
# Exercise 2 – validate_range
# ---------------------------------------------------------------------------

class ValidateRangeBattery(TestBattery):
    """
    Test battery for Exercise 2:
        validate_range(df, column, min_val=None, max_val=None)
        → dict {'passed': bool, 'violations': int, 'violation_pct': float}
    """

    def _get_reference_inputs(self):
        self.test_df  = pd.DataFrame({"score": [10, 50, 110, -5, 70]})
        self.column   = "score"
        self.min_val  = 0
        self.max_val  = 100

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.test_df.copy(),
                self.column,
                min_val=self.min_val,
                max_val=self.max_val,
            )
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "result_is_dict":  True,
            "passed":          False,
            "violations":      2,
        }

    # ── individual checks ────────────────────────────────────────────────────

    def result_is_dict_check(self):
        got = isinstance(self.result, dict)
        return got, True, not got

    def has_required_keys_check(self):
        if self.result is None:
            return False, True, True
        required = {"passed", "violations", "violation_pct"}
        got = required.issubset(set(self.result.keys()))
        return got, True, not got

    def violations_check(self):
        if self.result is None:
            return None, self.reference_checks["violations"], True
        try:
            got = int(self.result["violations"])
        except Exception:
            got = None
        return self._check("violations", got)

    def passed_check(self):
        if self.result is None:
            return None, self.reference_checks["passed"], True
        try:
            got = bool(self.result["passed"])
        except Exception:
            got = None
        return self._check("passed", got)


# ---------------------------------------------------------------------------
# Exercise 3 – validate_uniqueness
# ---------------------------------------------------------------------------

class ValidateUniquenessBattery(TestBattery):
    """
    Test battery for Exercise 3:
        validate_uniqueness(df, column)
        → dict {'passed': bool, 'duplicate_count': int}
    """

    def _get_reference_inputs(self):
        self.test_df = pd.DataFrame({"id": [1, 2, 2, 3, 4]})
        self.column  = "id"

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy(), self.column)
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "result_is_dict":  True,
            "passed":          False,
            "duplicate_count": 1,
        }

    # ── individual checks ────────────────────────────────────────────────────

    def result_is_dict_check(self):
        got = isinstance(self.result, dict)
        return got, True, not got

    def passed_check(self):
        if self.result is None:
            return None, self.reference_checks["passed"], True
        try:
            got = bool(self.result["passed"])
        except Exception:
            got = None
        return self._check("passed", got)

    def duplicate_count_check(self):
        if self.result is None:
            return None, self.reference_checks["duplicate_count"], True
        try:
            got = int(self.result["duplicate_count"])
        except Exception:
            got = None
        return self._check("duplicate_count", got)


# ---------------------------------------------------------------------------
# Exercise 4 – validate_regex
# ---------------------------------------------------------------------------

class ValidateRegexBattery(TestBattery):
    """
    Test battery for Exercise 4:
        validate_regex(df, column, pattern)
        → dict {'passed': bool, 'violations': int}
    """

    def _get_reference_inputs(self):
        self.test_df = pd.DataFrame(
            {"email": ["a@b.com", "invalid", "c@d.org"]}
        )
        self.column  = "email"
        self.pattern = r".+@.+\..+"

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.test_df.copy(), self.column, self.pattern
            )
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "result_is_dict": True,
            "passed":         False,
            "violations":     1,
        }

    # ── individual checks ────────────────────────────────────────────────────

    def result_is_dict_check(self):
        got = isinstance(self.result, dict)
        return got, True, not got

    def violations_check(self):
        if self.result is None:
            return None, self.reference_checks["violations"], True
        try:
            got = int(self.result["violations"])
        except Exception:
            got = None
        return self._check("violations", got)

    def passed_check(self):
        if self.result is None:
            return None, self.reference_checks["passed"], True
        try:
            got = bool(self.result["passed"])
        except Exception:
            got = None
        return self._check("passed", got)


# ---------------------------------------------------------------------------
# Exercise 5 – create_validation_report
# ---------------------------------------------------------------------------

class CreateValidationReportBattery(TestBattery):
    """
    Test battery for Exercise 5:
        create_validation_report(df, rules)
        → list of dicts, each with keys: 'rule', 'column', 'passed', 'violations'

    Rules list format:
        [{'type': 'no_nulls',  'column': 'a'},
         {'type': 'range',     'column': 'score', 'min_val': 0, 'max_val': 100}]
    """

    def _get_reference_inputs(self):
        self.test_df = pd.DataFrame(
            {
                "a":     [1, None, 3],
                "score": [10, 50, 110],
            }
        )
        self.rules = [
            {"type": "no_nulls", "column": "a"},
            {"type": "range",    "column": "score", "min_val": 0, "max_val": 100},
        ]

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy(), self.rules)
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "result_is_list":  True,
            "length":          2,
        }

    # ── individual checks ────────────────────────────────────────────────────

    def result_is_list_check(self):
        got = isinstance(self.result, list)
        return got, True, not got

    def length_check(self):
        if self.result is None:
            return None, self.reference_checks["length"], True
        try:
            got = int(len(self.result))
        except Exception:
            got = None
        return self._check("length", got)

    def items_have_required_keys_check(self):
        """Every item in the report must have: 'rule', 'column', 'passed', 'violations'."""
        required = {"rule", "column", "passed", "violations"}
        if self.result is None or not isinstance(self.result, list) or len(self.result) == 0:
            return False, True, True
        got = all(required.issubset(set(item.keys())) for item in self.result)
        return got, True, not got
