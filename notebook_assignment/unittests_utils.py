"""Unit test utilities for the Duplicate Detection & Removal assignment."""
import difflib

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Standard customer dataset used by all batteries
# ---------------------------------------------------------------------------

def _make_test_df() -> pd.DataFrame:
    """Return a small, deterministic test DataFrame with known duplicates."""
    rows = [
        # unique rows
        {"customer_id": "C001", "name": "Alice Smith",  "email": "alice@mail.com",  "age": 30, "purchase_amount": 150.0},
        {"customer_id": "C002", "name": "Bob Jones",    "email": "bob@mail.com",    "age": 25, "purchase_amount": 200.0},
        {"customer_id": "C003", "name": "Carol White",  "email": "carol@mail.com",  "age": 35, "purchase_amount": 300.0},
        {"customer_id": "C004", "name": "David Black",  "email": "david@mail.com",  "age": 40, "purchase_amount": 400.0},
        {"customer_id": "C005", "name": "Eve Green",    "email": "eve@mail.com",    "age": 28, "purchase_amount": 120.0},
        # exact duplicates of C001 and C002
        {"customer_id": "C001", "name": "Alice Smith",  "email": "alice@mail.com",  "age": 30, "purchase_amount": 150.0},
        {"customer_id": "C002", "name": "Bob Jones",    "email": "bob@mail.com",    "age": 25, "purchase_amount": 200.0},
        # near-duplicate of C003 (name differs in casing)
        {"customer_id": "C003", "name": "CAROL WHITE",  "email": "carol@mail.com",  "age": 35, "purchase_amount": 300.0},
    ]
    return pd.DataFrame(rows)


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
        if isinstance(want, float):
            if got is None:
                return got, want, True
            return got, want, not np.isclose(got, want, rtol=1e-5, atol=1e-8)
        else:
            condition = got != want
        return got, want, condition


# ---------------------------------------------------------------------------
# Exercise 1 – detect_exact_duplicates
# ---------------------------------------------------------------------------

class DetectExactDuplicatesBattery(TestBattery):
    """Test battery for Exercise 1: detect_exact_duplicates(df)."""

    def _get_reference_inputs(self):
        self.test_df = _make_test_df()

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy())
        except Exception:
            self.result = None

    def get_reference_checks(self):
        # C001 and C002 each appear twice → all four occurrences are "duplicates"
        self.reference_checks = {
            "n_rows": 4,
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

    def no_extra_rows(self):
        """All returned rows must actually be duplicates (present ≥2 times)."""
        if self.result is None:
            return None, True, True
        ref = self.test_df.duplicated(keep=False)
        expected_rows = set(self.test_df[ref].index.tolist())
        got_rows = set(self.result.index.tolist())
        failed = not got_rows.issubset(set(self.test_df.index.tolist()))
        return got_rows, expected_rows, failed


# ---------------------------------------------------------------------------
# Exercise 2 – remove_exact_duplicates
# ---------------------------------------------------------------------------

class RemoveExactDuplicatesBattery(TestBattery):
    """Test battery for Exercise 2: remove_exact_duplicates(df, keep='first')."""

    def _get_reference_inputs(self):
        self.test_df = _make_test_df()

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy(), keep="first")
        except Exception:
            self.result = None

    def get_reference_checks(self):
        # 8 rows – 2 exact dupes removed → 6 unique rows remain
        self.reference_checks = {
            "n_rows": 6,
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

    def no_exact_duplicates_remain(self):
        if self.result is None:
            return None, True, True
        got = self.result.duplicated().sum() == 0
        return got, True, not got

    def index_is_reset(self):
        if self.result is None:
            return None, True, True
        expected_index = list(range(len(self.result)))
        got = list(self.result.index)
        return got, expected_index, got != expected_index


# ---------------------------------------------------------------------------
# Exercise 3 – detect_fuzzy_duplicates
# ---------------------------------------------------------------------------

class DetectFuzzyDuplicatesBattery(TestBattery):
    """Test battery for Exercise 3: detect_fuzzy_duplicates(df, column, threshold)."""

    def _get_reference_inputs(self):
        self.test_df = _make_test_df()
        self.column = "name"
        self.threshold = 80

    def extract_info(self):
        try:
            self.result = self.learner_object(
                self.test_df.copy(), self.column, self.threshold
            )
        except Exception:
            self.result = None

    def get_reference_checks(self):
        # "Alice Smith" / "Alice Smith" = 100 %, "Carol White"/"CAROL WHITE" ≈ 100 %
        # so at least 2 pairs expected
        self.reference_checks = {
            "min_pairs": 2,
            "has_similarity_col": True,
        }

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        return got, True, not got

    def has_required_columns(self):
        required = {"index_1", "index_2", "value_1", "value_2", "similarity"}
        if self.result is None:
            return None, required, True
        got = set(self.result.columns)
        missing = required - got
        return got, required, len(missing) > 0

    def finds_enough_pairs(self):
        if self.result is None:
            return None, self.reference_checks["min_pairs"], True
        got = len(self.result)
        want = self.reference_checks["min_pairs"]
        return got, want, got < want

    def similarity_in_range(self):
        if self.result is None or len(self.result) == 0:
            return None, True, True
        if "similarity" not in self.result.columns:
            return None, True, True
        all_valid = ((self.result["similarity"] >= 0) & (self.result["similarity"] <= 100)).all()
        return bool(all_valid), True, not all_valid


# ---------------------------------------------------------------------------
# Exercise 4 – flag_duplicates
# ---------------------------------------------------------------------------

class FlagDuplicatesBattery(TestBattery):
    """Test battery for Exercise 4: flag_duplicates(df)."""

    def _get_reference_inputs(self):
        self.test_df = _make_test_df()

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy())
        except Exception:
            self.result = None

    def get_reference_checks(self):
        # rows 5 and 6 (index 5, 6) are later-occurring exact duplicates
        self.reference_checks = {
            "flagged_count": 2,
            "has_column": True,
        }

    def result_is_dataframe(self):
        got = isinstance(self.result, pd.DataFrame)
        return got, True, not got

    def has_is_duplicate_column(self):
        if self.result is None:
            return None, True, True
        got = "is_duplicate" in self.result.columns
        return got, True, not got

    def correct_flagged_count(self):
        if self.result is None or "is_duplicate" not in (self.result.columns if self.result is not None else []):
            return None, self.reference_checks["flagged_count"], True
        got = int(self.result["is_duplicate"].sum())
        want = self.reference_checks["flagged_count"]
        return got, want, got != want

    def same_shape_as_input(self):
        if self.result is None:
            return None, True, True
        got_rows = len(self.result)
        want_rows = len(self.test_df)
        return got_rows, want_rows, got_rows != want_rows


# ---------------------------------------------------------------------------
# Exercise 5 – create_deduplication_report
# ---------------------------------------------------------------------------

class DeduplicationReportBattery(TestBattery):
    """Test battery for Exercise 5: create_deduplication_report(df)."""

    def _get_reference_inputs(self):
        self.test_df = _make_test_df()

    def extract_info(self):
        try:
            self.result = self.learner_object(self.test_df.copy())
        except Exception:
            self.result = None

    def get_reference_checks(self):
        self.reference_checks = {
            "exact_count": 2,
            "total_rows": 8,
            "clean_rows": 6,
        }

    def result_is_dict(self):
        got = isinstance(self.result, dict)
        return got, True, not got

    def has_required_keys(self):
        required = {"exact_count", "total_rows", "clean_rows"}
        if self.result is None:
            return None, required, True
        got = set(self.result.keys())
        missing = required - got
        return got, required, len(missing) > 0

    def correct_exact_count(self):
        if self.result is None or "exact_count" not in (self.result or {}):
            return None, self.reference_checks["exact_count"], True
        got = self.result["exact_count"]
        want = self.reference_checks["exact_count"]
        return got, want, got != want

    def correct_total_rows(self):
        if self.result is None or "total_rows" not in (self.result or {}):
            return None, self.reference_checks["total_rows"], True
        got = self.result["total_rows"]
        want = self.reference_checks["total_rows"]
        return got, want, got != want

    def correct_clean_rows(self):
        if self.result is None or "clean_rows" not in (self.result or {}):
            return None, self.reference_checks["clean_rows"], True
        got = self.result["clean_rows"]
        want = self.reference_checks["clean_rows"]
        return got, want, got != want
