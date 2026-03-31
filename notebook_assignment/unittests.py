import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures
from dlai_grader.grading import test_case, print_feedback


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_test_df():
    """Small, fully deterministic DataFrame for testing."""
    return pd.DataFrame({
        'customer_id': [1, 2, 3],
        'last_purchase_date': pd.to_datetime([
            '2023-12-01',   # 31 days before reference
            '2023-06-01',   # 214 days before reference
            '2023-01-15',   # 351 days before reference
        ]),
        'purchase_count': [10, 5, 20],
        'total_spent':    [1000.0, 250.0, 2000.0],
        'n_products':     [2, 5, 8],
        'product_name': [
            'laptop computer',
            'wireless headphones premium',
            'coffee maker deluxe',
        ],
        'signup_date': pd.to_datetime([
            '2022-01-01',   # 730 days before reference
            '2021-01-01',   # 1095 days before reference
            '2020-06-01',   # 1309 days before reference
        ]),
    })


REFERENCE_DATE = pd.Timestamp('2024-01-01')


# ── Exercise 1: create_domain_features ───────────────────────────────────────

def exercise_1(learner_func):
    cases = []

    # --- test 1: required columns are added ---
    def test_columns_added():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        required = {'recency_days', 'avg_order_value',
                    'customer_lifetime_days', 'purchase_frequency'}
        missing = required - set(result.columns)
        if missing:
            t.failed = True
            t.msg = f"Missing columns: {missing}"
        return t

    # --- test 2: recency_days values ---
    def test_recency_days():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [31, 214, 351]
        actual = result['recency_days'].tolist()
        if actual != expected:
            t.failed = True
            t.msg = f"recency_days: expected {expected}, got {actual}"
        return t

    # --- test 3: avg_order_value ---
    def test_avg_order_value():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [100.0, 50.0, 100.0]
        actual = result['avg_order_value'].tolist()
        if not np.allclose(actual, expected):
            t.failed = True
            t.msg = f"avg_order_value: expected {expected}, got {actual}"
        return t

    # --- test 4: customer_lifetime_days ---
    def test_lifetime():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [730, 1095, 1309]
        actual = result['customer_lifetime_days'].tolist()
        if actual != expected:
            t.failed = True
            t.msg = f"customer_lifetime_days: expected {expected}, got {actual}"
        return t

    # --- test 5: purchase_frequency ---
    def test_frequency():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [10/730, 5/1095, 20/1309]
        actual = result['purchase_frequency'].tolist()
        if not np.allclose(actual, expected):
            t.failed = True
            t.msg = f"purchase_frequency: expected {expected}, got {actual}"
        return t

    # --- test 6: original columns are preserved ---
    def test_original_preserved():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        original_cols = set(df.columns)
        if not original_cols.issubset(set(result.columns)):
            t.failed = True
            t.msg = "Original DataFrame columns were removed from the result."
        return t

    cases.extend([
        test_columns_added(),
        test_recency_days(),
        test_avg_order_value(),
        test_lifetime(),
        test_frequency(),
        test_original_preserved(),
    ])
    print_feedback(cases)


# ── Exercise 2: create_mathematical_features ─────────────────────────────────

def exercise_2(learner_func):
    cases = []

    # --- test 1: required columns are added ---
    def test_columns_added():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        required = {'spend_per_product', 'purchase_spend_interaction',
                    'log_total_spent', 'total_spent_squared'}
        missing = required - set(result.columns)
        if missing:
            t.failed = True
            t.msg = f"Missing columns: {missing}"
        return t

    # --- test 2: spend_per_product ---
    def test_spend_per_product():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [1000.0/2, 250.0/5, 2000.0/8]
        actual = result['spend_per_product'].tolist()
        if not np.allclose(actual, expected):
            t.failed = True
            t.msg = f"spend_per_product: expected {expected}, got {actual}"
        return t

    # --- test 3: purchase_spend_interaction ---
    def test_interaction():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [10*1000.0, 5*250.0, 20*2000.0]
        actual = result['purchase_spend_interaction'].tolist()
        if not np.allclose(actual, expected):
            t.failed = True
            t.msg = f"purchase_spend_interaction: expected {expected}, got {actual}"
        return t

    # --- test 4: log_total_spent ---
    def test_log():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = np.log1p([1000.0, 250.0, 2000.0]).tolist()
        actual = result['log_total_spent'].tolist()
        if not np.allclose(actual, expected):
            t.failed = True
            t.msg = f"log_total_spent: expected {expected}, got {actual}"
        return t

    # --- test 5: total_spent_squared ---
    def test_squared():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [1000.0**2, 250.0**2, 2000.0**2]
        actual = result['total_spent_squared'].tolist()
        if not np.allclose(actual, expected):
            t.failed = True
            t.msg = f"total_spent_squared: expected {expected}, got {actual}"
        return t

    cases.extend([
        test_columns_added(),
        test_spend_per_product(),
        test_interaction(),
        test_log(),
        test_squared(),
    ])
    print_feedback(cases)


# ── Exercise 3: create_datetime_features ─────────────────────────────────────

def exercise_3(learner_func):
    cases = []

    # known day-of-week for test dates:
    # 2023-12-01 = Friday  = 4
    # 2023-06-01 = Thursday = 3
    # 2023-01-15 = Sunday   = 6

    # --- test 1: required columns are added ---
    def test_columns_added():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        required = {'purchase_day_of_week', 'purchase_month',
                    'purchase_quarter', 'is_weekend_purchase', 'signup_year'}
        missing = required - set(result.columns)
        if missing:
            t.failed = True
            t.msg = f"Missing columns: {missing}"
        return t

    # --- test 2: day of week ---
    def test_day_of_week():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [4, 3, 6]  # Fri, Thu, Sun
        actual = result['purchase_day_of_week'].tolist()
        if actual != expected:
            t.failed = True
            t.msg = f"purchase_day_of_week: expected {expected}, got {actual}"
        return t

    # --- test 3: month ---
    def test_month():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [12, 6, 1]
        actual = result['purchase_month'].tolist()
        if actual != expected:
            t.failed = True
            t.msg = f"purchase_month: expected {expected}, got {actual}"
        return t

    # --- test 4: quarter ---
    def test_quarter():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [4, 2, 1]
        actual = result['purchase_quarter'].tolist()
        if actual != expected:
            t.failed = True
            t.msg = f"purchase_quarter: expected {expected}, got {actual}"
        return t

    # --- test 5: is_weekend ---
    def test_weekend():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [0, 0, 1]   # Fri=no, Thu=no, Sun=yes
        actual = result['is_weekend_purchase'].tolist()
        if actual != expected:
            t.failed = True
            t.msg = f"is_weekend_purchase: expected {expected}, got {actual}"
        return t

    # --- test 6: signup_year ---
    def test_signup_year():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [2022, 2021, 2020]
        actual = result['signup_year'].tolist()
        if actual != expected:
            t.failed = True
            t.msg = f"signup_year: expected {expected}, got {actual}"
        return t

    cases.extend([
        test_columns_added(),
        test_day_of_week(),
        test_month(),
        test_quarter(),
        test_weekend(),
        test_signup_year(),
    ])
    print_feedback(cases)


# ── Exercise 4: create_text_features ─────────────────────────────────────────

def exercise_4(learner_func):
    cases = []

    # 'laptop computer'             → len=15, words=2, premium=0, wireless=0
    # 'wireless headphones premium' → len=27, words=3, premium=1, wireless=1
    # 'coffee maker deluxe'         → len=19, words=3, premium=0, wireless=0

    # --- test 1: required columns ---
    def test_columns_added():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        required = {'name_length', 'word_count', 'has_premium', 'has_wireless'}
        missing = required - set(result.columns)
        if missing:
            t.failed = True
            t.msg = f"Missing columns: {missing}"
        return t

    # --- test 2: name_length ---
    def test_name_length():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [15, 27, 19]
        actual = result['name_length'].tolist()
        if actual != expected:
            t.failed = True
            t.msg = f"name_length: expected {expected}, got {actual}"
        return t

    # --- test 3: word_count ---
    def test_word_count():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [2, 3, 3]
        actual = result['word_count'].tolist()
        if actual != expected:
            t.failed = True
            t.msg = f"word_count: expected {expected}, got {actual}"
        return t

    # --- test 4: has_premium ---
    def test_has_premium():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [0, 1, 0]
        actual = result['has_premium'].tolist()
        if actual != expected:
            t.failed = True
            t.msg = f"has_premium: expected {expected}, got {actual}"
        return t

    # --- test 5: has_wireless ---
    def test_has_wireless():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df)
        expected = [0, 1, 0]
        actual = result['has_wireless'].tolist()
        if actual != expected:
            t.failed = True
            t.msg = f"has_wireless: expected {expected}, got {actual}"
        return t

    cases.extend([
        test_columns_added(),
        test_name_length(),
        test_word_count(),
        test_has_premium(),
        test_has_wireless(),
    ])
    print_feedback(cases)


# ── Exercise 5: create_polynomial_features ───────────────────────────────────

def exercise_5(learner_func):
    cases = []

    X_test = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
    # degree=2, include_bias=False → 9 features for 3 inputs

    # --- test 1: output shape ---
    def test_shape():
        t = test_case()
        result = learner_func(X_test, degree=2)
        expected_shape = (2, 9)
        if result.shape != expected_shape:
            t.failed = True
            t.msg = f"Shape: expected {expected_shape}, got {result.shape}"
        return t

    # --- test 2: matches sklearn ---
    def test_matches_sklearn():
        t = test_case()
        X_sklearn = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_test)
        result = learner_func(X_test, degree=2)
        if not np.allclose(result, X_sklearn):
            t.failed = True
            t.msg = "Output does not match sklearn PolynomialFeatures."
        return t

    # --- test 3: degree=3 shape ---
    def test_degree3_shape():
        t = test_case()
        result = learner_func(X_test, degree=3)
        # degree=3 for 3 features: C(3+3-1, 3) + C(3+2-1, 2) + 3 = 10 + 6 + 3 = 19
        expected_shape = (2, 19)
        if result.shape != expected_shape:
            t.failed = True
            t.msg = f"degree=3 shape: expected {expected_shape}, got {result.shape}"
        return t

    # --- test 4: original features are included ---
    def test_original_included():
        t = test_case()
        result = learner_func(X_test, degree=2)
        # First 3 columns should be the original features
        if not np.allclose(result[:, :3], X_test):
            t.failed = True
            t.msg = "Original features are not in the first columns of the output."
        return t

    cases.extend([
        test_shape(),
        test_matches_sklearn(),
        test_degree3_shape(),
        test_original_included(),
    ])
    print_feedback(cases)


# ── Exercise 6: polynomial_features_from_scratch ─────────────────────────────

def exercise_6(learner_func):
    cases = []

    X_test = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])

    # --- test 1: output shape matches sklearn ---
    def test_shape():
        t = test_case()
        result = learner_func(X_test, degree=2)
        expected_shape = (2, 9)
        if result.shape != expected_shape:
            t.failed = True
            t.msg = f"Shape: expected {expected_shape}, got {result.shape}"
        return t

    # --- test 2: values match sklearn ---
    def test_values_match_sklearn():
        t = test_case()
        X_sklearn = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_test)
        result = learner_func(X_test, degree=2)
        if not np.allclose(result, X_sklearn):
            t.failed = True
            t.msg = "Values do not match sklearn PolynomialFeatures(degree=2)."
        return t

    # --- test 3: degree=3 matches sklearn ---
    def test_degree3_matches_sklearn():
        t = test_case()
        X_sklearn = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X_test)
        result = learner_func(X_test, degree=3)
        if not np.allclose(result, X_sklearn):
            t.failed = True
            t.msg = "Values do not match sklearn PolynomialFeatures(degree=3)."
        return t

    # --- test 4: specific known value (x0*x1 for first row = 1*2 = 2) ---
    def test_known_value():
        t = test_case()
        result = learner_func(X_test, degree=2)
        # sklearn order: x0, x1, x2, x0^2, x0*x1, x0*x2, x1^2, x1*x2, x2^2
        # index 4 = x0*x1 = 1*2 = 2
        if not np.isclose(result[0, 4], 2.0):
            t.failed = True
            t.msg = f"Expected result[0, 4] (x0*x1) = 2.0, got {result[0, 4]}"
        return t

    # --- test 5: does not use sklearn internally ---
    def test_no_sklearn():
        t = test_case()
        import inspect
        try:
            src = inspect.getsource(learner_func)
            if 'PolynomialFeatures' in src:
                t.failed = True
                t.msg = "The from-scratch implementation should not use sklearn's PolynomialFeatures."
        except OSError:
            pass  # source not available in this context; skip check
        return t

    cases.extend([
        test_shape(),
        test_values_match_sklearn(),
        test_degree3_matches_sklearn(),
        test_known_value(),
        test_no_sklearn(),
    ])
    print_feedback(cases)
