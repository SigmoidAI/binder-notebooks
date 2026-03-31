import numpy as np
import pandas as pd
from dlai_grader.grading import test_case, print_feedback


# ── Test DataFrame ───────────────────────────────────────────────────────────

def _make_test_df():
    """Small, fully deterministic DataFrame for all exercises."""
    return pd.DataFrame({
        "category": ["Electronics", "Clothing", "Food", "Electronics", "Books", "Clothing"],
        "education": ["High School", "Bachelor", "Master", "PhD", "Bachelor", "Master"],
        "city":      ["NYC", "LA", "NYC", "Chicago", "LA", "NYC"],
        "price":     [100.0, 200.0, 150.0, 300.0, 250.0, 180.0],
        "sales":     [1000.0, 800.0, 600.0, 1200.0, 900.0, 750.0],
    })


EDUCATION_ORDER = ["High School", "Bachelor", "Master", "PhD"]


# ── Exercise 1: apply_one_hot_encoding ──────────────────────────────────────

def exercise_1(learner_func):
    cases = []

    def test_ohe_columns_created():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df, ["category"])
        cat_cols = [c for c in result.columns if c.startswith("category_")]
        # 4 unique values: Books, Clothing, Electronics, Food
        if len(cat_cols) != 4:
            t.failed = True
            t.msg = (f"Expected 4 category_* columns for 4 unique categories; "
                     f"got {len(cat_cols)}: {cat_cols}")
        return t

    def test_original_col_dropped():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df, ["category"])
        if "category" in result.columns:
            t.failed = True
            t.msg = "Original 'category' column should be removed after OHE"
        return t

    def test_values_binary():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df, ["category"])
        cat_cols = [c for c in result.columns if c.startswith("category_")]
        vals = set(result[cat_cols].values.ravel().tolist())
        if not vals.issubset({0, 1}):
            t.failed = True
            t.msg = f"OHE values must be 0 or 1 only; found {vals - {0, 1}}"
        return t

    def test_row_sums_equal_one():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df, ["category"])
        cat_cols = [c for c in result.columns if c.startswith("category_")]
        sums = result[cat_cols].sum(axis=1).tolist()
        if not all(s == 1 for s in sums):
            t.failed = True
            t.msg = f"Each row should sum to 1 (exactly one 1 per row); got: {sums}"
        return t

    def test_other_cols_preserved():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df, ["category"])
        for col in ["education", "city", "price", "sales"]:
            if col not in result.columns:
                t.failed = True
                t.msg = f"Column '{col}' should be preserved in output"
                return t
        return t

    cases.extend([
        test_ohe_columns_created(),
        test_original_col_dropped(),
        test_values_binary(),
        test_row_sums_equal_one(),
        test_other_cols_preserved(),
    ])
    print_feedback(cases)


# ── Exercise 2: apply_ordinal_encoding ──────────────────────────────────────

def exercise_2(learner_func):
    cases = []

    def test_encoded_column_added():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df, "education", EDUCATION_ORDER)
        if "education_encoded" not in result.columns:
            t.failed = True
            t.msg = "Expected 'education_encoded' column in result"
        return t

    def test_correct_mapping():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df, "education", EDUCATION_ORDER)
        # education values: [High School, Bachelor, Master, PhD, Bachelor, Master]
        # → [0, 1, 2, 3, 1, 2]
        expected = [0, 1, 2, 3, 1, 2]
        actual = result["education_encoded"].tolist()
        if actual != expected:
            t.failed = True
            t.msg = f"Incorrect ordinal mapping: expected {expected}, got {actual}"
        return t

    def test_original_preserved():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df, "education", EDUCATION_ORDER)
        if "education" not in result.columns:
            t.failed = True
            t.msg = "Original 'education' column should be preserved"
        return t

    def test_dtype_numeric():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df, "education", EDUCATION_ORDER)
        if "education_encoded" in result.columns and not np.issubdtype(
            result["education_encoded"].dtype, np.number
        ):
            t.failed = True
            t.msg = "education_encoded should have a numeric dtype"
        return t

    cases.extend([
        test_encoded_column_added(),
        test_correct_mapping(),
        test_original_preserved(),
        test_dtype_numeric(),
    ])
    print_feedback(cases)


# ── Exercise 3: apply_frequency_encoding ────────────────────────────────────

def exercise_3(learner_func):
    cases = []

    def test_freq_col_added():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df, "city")
        if "city_freq" not in result.columns:
            t.failed = True
            t.msg = "Expected 'city_freq' column in result"
        return t

    def test_correct_frequencies():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df, "city")
        # city counts: NYC=3, LA=2, Chicago=1  → freq: 3/6, 2/6, 3/6, 1/6, 2/6, 3/6
        expected = [3/6, 2/6, 3/6, 1/6, 2/6, 3/6]
        actual = result["city_freq"].tolist()
        if not np.allclose(actual, expected, atol=1e-9):
            t.failed = True
            t.msg = (f"Incorrect frequencies; "
                     f"expected {[round(e, 4) for e in expected]}, "
                     f"got {[round(a, 4) for a in actual]}")
        return t

    def test_values_in_zero_one():
        t = test_case()
        df = _make_test_df()
        result = learner_func(df, "city")
        if not result["city_freq"].between(0, 1).all():
            t.failed = True
            t.msg = "Frequency values must be in [0, 1]"
        return t

    cases.extend([
        test_freq_col_added(),
        test_correct_frequencies(),
        test_values_in_zero_one(),
    ])
    print_feedback(cases)


# ── Exercise 4: apply_target_encoding ───────────────────────────────────────

def exercise_4(learner_func):
    cases = []

    df_all = _make_test_df()
    X_train = df_all.iloc[:4].copy().reset_index(drop=True)
    y_train = np.array([1000.0, 800.0, 600.0, 1200.0])
    X_test  = df_all.iloc[4:].copy().reset_index(drop=True)

    def test_output_types():
        t = test_case()
        X_tr, X_te = learner_func(X_train.copy(), y_train.copy(),
                                  X_test.copy(), "city", alpha=0)
        if not isinstance(X_tr, pd.DataFrame):
            t.failed = True
            t.msg = "First return value should be a DataFrame (X_train_enc)"
            return t
        if not isinstance(X_te, pd.DataFrame):
            t.failed = True
            t.msg = "Second return value should be a DataFrame (X_test_enc)"
        return t

    def test_col_added_to_train():
        t = test_case()
        X_tr, _ = learner_func(X_train.copy(), y_train.copy(),
                               X_test.copy(), "city", alpha=0)
        if "city_target_enc" not in X_tr.columns:
            t.failed = True
            t.msg = "Expected 'city_target_enc' column in training output"
        return t

    def test_train_values_alpha0():
        t = test_case()
        # With alpha=0: exact category means from y_train
        # train cities: NYC(row 0), LA(row 1), NYC(row 2), Chicago(row 3)
        # NYC → mean([1000, 600]) = 800.0
        # LA  → mean([800])       = 800.0
        # Chicago → mean([1200])  = 1200.0
        X_tr, _ = learner_func(X_train.copy(), y_train.copy(),
                               X_test.copy(), "city", alpha=0)
        expected = [800.0, 800.0, 800.0, 1200.0]
        actual = X_tr["city_target_enc"].tolist()
        if not np.allclose(actual, expected, atol=1e-6):
            t.failed = True
            t.msg = f"Train target enc (alpha=0): expected {expected}, got {actual}"
        return t

    def test_test_values_known():
        t = test_case()
        # test cities: LA(row 4), NYC(row 5)
        # LA mean from train = 800, NYC mean from train = 800
        _, X_te = learner_func(X_train.copy(), y_train.copy(),
                               X_test.copy(), "city", alpha=0)
        expected = [800.0, 800.0]
        actual = X_te["city_target_enc"].tolist()
        if not np.allclose(actual, expected, atol=1e-6):
            t.failed = True
            t.msg = f"Test target enc (alpha=0): expected {expected}, got {actual}"
        return t

    def test_smoothing_pulls_toward_global_mean():
        t = test_case()
        global_mean = float(np.mean(y_train))
        X_tr_raw, _ = learner_func(X_train.copy(), y_train.copy(),
                                   X_test.copy(), "city", alpha=0)
        X_tr_sm,  _ = learner_func(X_train.copy(), y_train.copy(),
                                   X_test.copy(), "city", alpha=500)
        # Chicago (n=1, raw=1200) should be pulled close to global mean with high alpha
        chicago_raw = float(X_tr_raw["city_target_enc"].iloc[3])
        chicago_sm  = float(X_tr_sm["city_target_enc"].iloc[3])
        if not (abs(chicago_sm - global_mean) < abs(chicago_raw - global_mean)):
            t.failed = True
            t.msg = (f"With alpha=500, Chicago should be pulled closer to global mean "
                     f"({global_mean:.1f}). Raw: {chicago_raw:.1f}, Smoothed: {chicago_sm:.1f}")
        return t

    cases.extend([
        test_output_types(),
        test_col_added_to_train(),
        test_train_values_alpha0(),
        test_test_values_known(),
        test_smoothing_pulls_toward_global_mean(),
    ])
    print_feedback(cases)


# ── Exercise 5: create_encoding_pipeline ────────────────────────────────────

def exercise_5(learner_func):
    cases = []

    def test_returns_pipeline():
        t = test_case()
        from sklearn.pipeline import Pipeline
        result = learner_func(
            nominal_cols=["category"],
            ordinal_col="education",
            ordinal_order=EDUCATION_ORDER,
        )
        if not isinstance(result, Pipeline):
            t.failed = True
            t.msg = f"Expected sklearn Pipeline, got {type(result).__name__}"
        return t

    def test_pipeline_has_steps():
        t = test_case()
        from sklearn.pipeline import Pipeline
        pipeline = learner_func(
            nominal_cols=["category"],
            ordinal_col="education",
            ordinal_order=EDUCATION_ORDER,
        )
        if len(pipeline.steps) < 2:
            t.failed = True
            t.msg = "Pipeline must have at least 2 steps (preprocessor + estimator)"
        return t

    def test_pipeline_fits_and_predicts():
        t = test_case()
        pipeline = learner_func(
            nominal_cols=["category"],
            ordinal_col="education",
            ordinal_order=EDUCATION_ORDER,
        )
        df = _make_test_df()
        X = df[["category", "education", "price"]]
        y = df["sales"]
        try:
            pipeline.fit(X, y)
        except Exception as e:
            t.failed = True
            t.msg = f"pipeline.fit() raised: {e}"
            return t
        try:
            preds = pipeline.predict(X)
            if len(preds) != len(y):
                t.failed = True
                t.msg = f"Expected {len(y)} predictions, got {len(preds)}"
        except Exception as e:
            t.failed = True
            t.msg = f"pipeline.predict() raised: {e}"
        return t

    cases.extend([
        test_returns_pipeline(),
        test_pipeline_has_steps(),
        test_pipeline_fits_and_predicts(),
    ])
    print_feedback(cases)


# ── Exercise 6: one_hot_encode_from_scratch ──────────────────────────────────

def exercise_6(learner_func):
    cases = []

    def test_return_type():
        t = test_case()
        series = pd.Series(["A", "B", "A", "C"])
        result = learner_func(series)
        if not isinstance(result, tuple) or len(result) != 2:
            t.failed = True
            t.msg = "Expected a 2-tuple (matrix, categories)"
            return t
        matrix, cats = result
        if not isinstance(matrix, np.ndarray):
            t.failed = True
            t.msg = f"First element must be np.ndarray, got {type(matrix).__name__}"
        return t

    def test_shape():
        t = test_case()
        series = pd.Series(["A", "B", "A", "C"])
        matrix, cats = learner_func(series)
        if matrix.shape != (4, 3):
            t.failed = True
            t.msg = f"Expected shape (4, 3) for 4 rows and 3 unique values; got {matrix.shape}"
        return t

    def test_correct_encoding():
        t = test_case()
        series = pd.Series(["A", "B", "A", "C"])
        matrix, cats = learner_func(series)
        sorted_cats = sorted(cats)
        a_idx = sorted_cats.index("A")
        b_idx = sorted_cats.index("B")
        c_idx = sorted_cats.index("C")
        # Row 0: A → [1,0,0];  Row 1: B → [0,1,0];  Row 3: C → [0,0,1]
        expected_row0 = [0, 0, 0]; expected_row0[a_idx] = 1
        expected_row1 = [0, 0, 0]; expected_row1[b_idx] = 1
        expected_row3 = [0, 0, 0]; expected_row3[c_idx] = 1
        checks = [
            (matrix[0].tolist(), expected_row0, "row 0 (A)"),
            (matrix[1].tolist(), expected_row1, "row 1 (B)"),
            (matrix[3].tolist(), expected_row3, "row 3 (C)"),
        ]
        for actual, expected, label in checks:
            if actual != expected:
                t.failed = True
                t.msg = f"Encoding error for {label}: expected {expected}, got {actual}"
                return t
        return t

    def test_row_sums():
        t = test_case()
        series = pd.Series(["A", "B", "A", "C"])
        matrix, _ = learner_func(series)
        sums = matrix.sum(axis=1).tolist()
        if sums != [1, 1, 1, 1]:
            t.failed = True
            t.msg = f"Each row must sum to 1; got {sums}"
        return t

    cases.extend([
        test_return_type(),
        test_shape(),
        test_correct_encoding(),
        test_row_sums(),
    ])
    print_feedback(cases)


# ── Exercise 7: target_encode_from_scratch ──────────────────────────────────

def exercise_7(learner_func):
    cases = []

    train_col = pd.Series(["NYC", "LA", "NYC", "Chicago"])
    y_train   = np.array([1000.0, 800.0, 600.0, 1200.0])
    test_col  = pd.Series(["LA", "NYC", "Berlin"])  # Berlin unseen

    def test_output_shapes():
        t = test_case()
        tr, te = learner_func(train_col.copy(), y_train.copy(),
                              test_col.copy(), alpha=0)
        if len(tr) != 4:
            t.failed = True
            t.msg = f"Train output should have length 4; got {len(tr)}"
            return t
        if len(te) != 3:
            t.failed = True
            t.msg = f"Test output should have length 3; got {len(te)}"
        return t

    def test_train_encoding_alpha0():
        t = test_case()
        tr, _ = learner_func(train_col.copy(), y_train.copy(),
                             test_col.copy(), alpha=0)
        # NYC → (1000+600)/2 = 800; LA → 800; Chicago → 1200
        expected = [800.0, 800.0, 800.0, 1200.0]
        if not np.allclose(list(tr), expected, atol=1e-6):
            t.failed = True
            t.msg = f"Train target enc (alpha=0): expected {expected}, got {list(tr)}"
        return t

    def test_unknown_gets_global_mean():
        t = test_case()
        _, te = learner_func(train_col.copy(), y_train.copy(),
                             test_col.copy(), alpha=0)
        global_mean = float(np.mean(y_train))  # = 900.0
        berlin_enc = float(te[2])
        if not np.isclose(berlin_enc, global_mean, atol=1e-6):
            t.failed = True
            t.msg = (f"Unknown category 'Berlin' should encode to global mean "
                     f"{global_mean}; got {berlin_enc}")
        return t

    def test_smoothing_effect():
        t = test_case()
        tr_raw, _ = learner_func(train_col.copy(), y_train.copy(),
                                 test_col.copy(), alpha=0)
        tr_sm, _  = learner_func(train_col.copy(), y_train.copy(),
                                 test_col.copy(), alpha=200)
        global_mean = float(np.mean(y_train))
        # Chicago (n=1, raw=1200) must move closer to global mean with high alpha
        chicago_raw = float(tr_raw[3])
        chicago_sm  = float(tr_sm[3])
        if not (abs(chicago_sm - global_mean) < abs(chicago_raw - global_mean)):
            t.failed = True
            t.msg = (f"With alpha=200, Chicago should be pulled toward global mean "
                     f"({global_mean}). Raw: {chicago_raw}, Smoothed: {chicago_sm}")
        return t

    cases.extend([
        test_output_shapes(),
        test_train_encoding_alpha0(),
        test_unknown_gets_global_mean(),
        test_smoothing_effect(),
    ])
    print_feedback(cases)
