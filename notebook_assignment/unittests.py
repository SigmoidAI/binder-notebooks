import numpy as np
import pandas as pd
from dlai_grader.grading import test_case, print_feedback


# ── Shared test data ─────────────────────────────────────────────────────────

TRAIN = np.array([
    [1.0, 100.0],
    [2.0, 200.0],
    [3.0, 300.0],
    [4.0, 400.0],
    [5.0, 500.0],
], dtype=float)

TEST = np.array([
    [3.0, 150.0],
    [6.0, 600.0],
], dtype=float)

TRAIN_DF = pd.DataFrame(TRAIN, columns=["feat1", "feat2"])
TEST_DF  = pd.DataFrame(TEST,  columns=["feat1", "feat2"])
COLS     = ["feat1", "feat2"]

# Series of perfect squares — all positive, clean for boxcox/log
TRANSFORM_SERIES = pd.Series([1.0, 4.0, 9.0, 16.0, 25.0])

BIN_SERIES = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

# Dataset where scaling dramatically helps KNN
# Feature 1: class signal (gap of 5σ between classes)
# Feature 2: high-scale noise (range ~3000, no class information)
np.random.seed(42)
_N = 200
_X = np.column_stack([
    np.concatenate([np.random.randn(_N // 2),       np.random.randn(_N // 2) + 5]),
    np.concatenate([np.random.randn(_N // 2) * 600, np.random.randn(_N // 2) * 600]),
])
_y = np.array([0] * (_N // 2) + [1] * (_N // 2))
KNN_XTRAIN, KNN_XTEST = _X[:160], _X[160:]
KNN_YTRAIN, KNN_YTEST = _y[:160], _y[160:]


# ── Exercise 1: apply_scalers ────────────────────────────────────────────────

def exercise_1(learner_func):
    cases = []

    def test_returns_dict_with_keys():
        t = test_case()
        result = learner_func(TRAIN_DF.copy(), TEST_DF.copy(), COLS)
        if not isinstance(result, dict):
            t.failed = True
            t.msg = f"Expected a dict; got {type(result).__name__}"
            return t
        missing = {"standard", "minmax", "robust"} - set(result.keys())
        if missing:
            t.failed = True
            t.msg = f"Dict is missing keys: {missing}"
        return t

    def test_each_value_is_tuple_of_dataframes():
        t = test_case()
        result = learner_func(TRAIN_DF.copy(), TEST_DF.copy(), COLS)
        for key in ("standard", "minmax", "robust"):
            val = result.get(key)
            if not isinstance(val, tuple) or len(val) != 2:
                t.failed = True
                t.msg = f"result['{key}'] should be a 2-tuple (X_train_scaled, X_test_scaled)"
                return t
            X_tr, X_te = val
            if not isinstance(X_tr, pd.DataFrame) or not isinstance(X_te, pd.DataFrame):
                t.failed = True
                t.msg = f"result['{key}'] elements should be DataFrames"
                return t
        return t

    def test_standard_scaler_train_mean_zero():
        t = test_case()
        result = learner_func(TRAIN_DF.copy(), TEST_DF.copy(), COLS)
        X_tr, _ = result["standard"]
        means = X_tr[COLS].mean()
        if not np.allclose(means.values, 0, atol=1e-9):
            t.failed = True
            t.msg = (f"StandardScaler train columns must have mean ≈ 0; "
                     f"got means {means.values.round(6).tolist()}")
        return t

    def test_standard_scaler_train_std_one():
        t = test_case()
        result = learner_func(TRAIN_DF.copy(), TEST_DF.copy(), COLS)
        X_tr, _ = result["standard"]
        # population std (ddof=0), consistent with sklearn
        stds = X_tr[COLS].std(ddof=0)
        if not np.allclose(stds.values, 1, atol=1e-9):
            t.failed = True
            t.msg = (f"StandardScaler train columns must have std ≈ 1 (ddof=0); "
                     f"got stds {stds.values.round(6).tolist()}")
        return t

    def test_minmax_train_range_zero_one():
        t = test_case()
        result = learner_func(TRAIN_DF.copy(), TEST_DF.copy(), COLS)
        X_tr, _ = result["minmax"]
        for col in COLS:
            lo, hi = X_tr[col].min(), X_tr[col].max()
            if not (np.isclose(lo, 0, atol=1e-9) and np.isclose(hi, 1, atol=1e-9)):
                t.failed = True
                t.msg = (f"MinMaxScaler train '{col}' must span [0, 1]; "
                         f"got [{lo:.6f}, {hi:.6f}]")
                return t
        return t

    def test_standard_test_values():
        t = test_case()
        result = learner_func(TRAIN_DF.copy(), TEST_DF.copy(), COLS)
        _, X_te = result["standard"]
        # feat1: mean=3, std=sqrt(2); test[0]=(3-3)/sqrt(2)=0, test[1]=(6-3)/sqrt(2)=2.121
        expected_f1 = [0.0, 3.0 / np.sqrt(2)]
        actual_f1 = X_te["feat1"].tolist()
        if not np.allclose(actual_f1, expected_f1, atol=1e-6):
            t.failed = True
            t.msg = (f"StandardScaler test feat1: expected {[round(v, 4) for v in expected_f1]}, "
                     f"got {[round(v, 4) for v in actual_f1]}")
        return t

    cases.extend([
        test_returns_dict_with_keys(),
        test_each_value_is_tuple_of_dataframes(),
        test_standard_scaler_train_mean_zero(),
        test_standard_scaler_train_std_one(),
        test_minmax_train_range_zero_one(),
        test_standard_test_values(),
    ])
    print_feedback(cases)


# ── Exercise 2: apply_distribution_transforms ────────────────────────────────

def exercise_2(learner_func):
    cases = []

    def test_returns_dict_with_keys():
        t = test_case()
        result = learner_func(TRANSFORM_SERIES.copy())
        if not isinstance(result, dict):
            t.failed = True
            t.msg = f"Expected a dict; got {type(result).__name__}"
            return t
        required = {"log", "sqrt", "boxcox", "yeojohnson"}
        missing = required - set(result.keys())
        if missing:
            t.failed = True
            t.msg = f"Dict missing keys: {missing}"
        return t

    def test_log_values_correct():
        t = test_case()
        result = learner_func(TRANSFORM_SERIES.copy())
        expected = np.log([1.0, 4.0, 9.0, 16.0, 25.0])
        actual = np.array(result["log"]).flatten()
        if not np.allclose(actual, expected, atol=1e-9):
            t.failed = True
            t.msg = (f"log transform: expected {expected.round(4).tolist()}, "
                     f"got {actual.round(4).tolist()}")
        return t

    def test_sqrt_values_correct():
        t = test_case()
        result = learner_func(TRANSFORM_SERIES.copy())
        expected = [1.0, 2.0, 3.0, 4.0, 5.0]
        actual = np.array(result["sqrt"]).flatten().tolist()
        if not np.allclose(actual, expected, atol=1e-9):
            t.failed = True
            t.msg = f"sqrt transform: expected {expected}, got {[round(v,4) for v in actual]}"
        return t

    def test_boxcox_reduces_skewness():
        t = test_case()
        from scipy import stats
        result = learner_func(TRANSFORM_SERIES.copy())
        bc_data = np.array(result["boxcox"]).flatten()
        original_skew = float(stats.skew(TRANSFORM_SERIES.values))
        bc_skew = float(stats.skew(bc_data))
        if abs(bc_skew) > abs(original_skew) + 0.01:
            t.failed = True
            t.msg = (f"Box-Cox should not increase skewness; "
                     f"original={original_skew:.3f}, boxcox={bc_skew:.3f}")
        return t

    def test_yeojohnson_reduces_skewness():
        t = test_case()
        from scipy import stats
        result = learner_func(TRANSFORM_SERIES.copy())
        yj_data = np.array(result["yeojohnson"]).flatten()
        original_skew = float(stats.skew(TRANSFORM_SERIES.values))
        yj_skew = float(stats.skew(yj_data))
        if abs(yj_skew) > abs(original_skew) + 0.01:
            t.failed = True
            t.msg = (f"Yeo-Johnson should not increase skewness; "
                     f"original={original_skew:.3f}, yeojohnson={yj_skew:.3f}")
        return t

    def test_output_length_preserved():
        t = test_case()
        result = learner_func(TRANSFORM_SERIES.copy())
        for key in ("log", "sqrt", "boxcox", "yeojohnson"):
            arr = np.array(result[key]).flatten()
            if len(arr) != len(TRANSFORM_SERIES):
                t.failed = True
                t.msg = (f"'{key}' output must have same length as input "
                         f"({len(TRANSFORM_SERIES)}); got {len(arr)}")
                return t
        return t

    cases.extend([
        test_returns_dict_with_keys(),
        test_log_values_correct(),
        test_sqrt_values_correct(),
        test_boxcox_reduces_skewness(),
        test_yeojohnson_reduces_skewness(),
        test_output_length_preserved(),
    ])
    print_feedback(cases)


# ── Exercise 3: apply_binning ─────────────────────────────────────────────────

def exercise_3(learner_func):
    cases = []

    def test_equal_width_bin_count():
        t = test_case()
        result = learner_func(BIN_SERIES.copy(), strategy="equal_width", n_bins=2)
        n_unique = result.nunique()
        if n_unique != 2:
            t.failed = True
            t.msg = f"equal_width with n_bins=2 should produce 2 unique bins; got {n_unique}"
        return t

    def test_equal_width_values():
        t = test_case()
        result = learner_func(BIN_SERIES.copy(), strategy="equal_width", n_bins=2)
        expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        actual = result.tolist()
        if actual != expected:
            t.failed = True
            t.msg = (f"equal_width n_bins=2 on [1..10]: "
                     f"expected {expected}, got {actual}")
        return t

    def test_equal_freq_bin_sizes():
        t = test_case()
        result = learner_func(BIN_SERIES.copy(), strategy="equal_freq", n_bins=2)
        counts = result.value_counts().sort_index().tolist()
        # Should split [1..10] into two groups of 5
        if sum(counts) != 10 or len(counts) != 2:
            t.failed = True
            t.msg = (f"equal_freq n_bins=2: expected 2 bins with 5 items each; "
                     f"got counts {counts}")
            return t
        if not all(c == 5 for c in counts):
            t.failed = True
            t.msg = f"equal_freq bins should have equal counts; got {counts}"
        return t

    def test_custom_bins():
        t = test_case()
        result = learner_func(
            BIN_SERIES.copy(), strategy="custom",
            n_bins=2, custom_edges=[0.0, 5.5, 10.0]
        )
        expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        actual = result.tolist()
        if actual != expected:
            t.failed = True
            t.msg = (f"custom bins [0, 5.5, 10] on [1..10]: "
                     f"expected {expected}, got {actual}")
        return t

    def test_returns_integer_series():
        t = test_case()
        result = learner_func(BIN_SERIES.copy(), strategy="equal_width", n_bins=5)
        if not np.issubdtype(result.dtype, np.integer):
            t.failed = True
            t.msg = f"apply_binning must return an integer dtype Series; got {result.dtype}"
        return t

    def test_equal_width_n5_values():
        t = test_case()
        result = learner_func(BIN_SERIES.copy(), strategy="equal_width", n_bins=5)
        expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        actual = result.tolist()
        if actual != expected:
            t.failed = True
            t.msg = (f"equal_width n_bins=5 on [1..10]: "
                     f"expected {expected}, got {actual}")
        return t

    cases.extend([
        test_equal_width_bin_count(),
        test_equal_width_values(),
        test_equal_freq_bin_sizes(),
        test_custom_bins(),
        test_returns_integer_series(),
        test_equal_width_n5_values(),
    ])
    print_feedback(cases)


# ── Exercise 4: compare_knn_scaling ──────────────────────────────────────────

def exercise_4(learner_func):
    cases = []

    def test_returns_dict_with_keys():
        t = test_case()
        result = learner_func(
            KNN_XTRAIN, KNN_YTRAIN, KNN_XTEST, KNN_YTEST
        )
        if not isinstance(result, dict):
            t.failed = True
            t.msg = f"Expected a dict; got {type(result).__name__}"
            return t
        required = {"unscaled_accuracy", "scaled_accuracy"}
        missing = required - set(result.keys())
        if missing:
            t.failed = True
            t.msg = f"Missing keys: {missing}"
        return t

    def test_accuracy_values_in_range():
        t = test_case()
        result = learner_func(
            KNN_XTRAIN, KNN_YTRAIN, KNN_XTEST, KNN_YTEST
        )
        for key in ("unscaled_accuracy", "scaled_accuracy"):
            val = result.get(key, None)
            if val is None:
                continue
            if not (0.0 <= float(val) <= 1.0):
                t.failed = True
                t.msg = f"'{key}' must be in [0, 1]; got {val}"
                return t
        return t

    def test_scaling_improves_knn():
        t = test_case()
        result = learner_func(
            KNN_XTRAIN, KNN_YTRAIN, KNN_XTEST, KNN_YTEST
        )
        unscaled = float(result.get("unscaled_accuracy", 0))
        scaled   = float(result.get("scaled_accuracy",   0))
        # On this dataset (feat1=signal at scale ~1, feat2=noise at scale ~600)
        # KNN with scaling should clearly outperform KNN without scaling
        if scaled < unscaled:
            t.failed = True
            t.msg = (f"Scaled KNN should outperform unscaled KNN on this dataset. "
                     f"Unscaled: {unscaled:.3f}, Scaled: {scaled:.3f}. "
                     f"Check that you are using StandardScaler fitted on X_train only.")
        return t

    cases.extend([
        test_returns_dict_with_keys(),
        test_accuracy_values_in_range(),
        test_scaling_improves_knn(),
    ])
    print_feedback(cases)


# ── Exercise 5: create_transformation_pipeline ───────────────────────────────

def _make_pipeline_data():
    return pd.DataFrame({
        "income":  [1000.0, 5000.0, 10000.0, 50000.0, 100000.0, 2000.0],
        "age":     [25.0,   30.0,   35.0,    40.0,    45.0,     28.0],
        "score":   [0.5,    0.6,    0.7,     0.8,     0.9,      0.55],
    })


def exercise_5(learner_func):
    cases = []

    def test_returns_pipeline():
        t = test_case()
        from sklearn.pipeline import Pipeline
        result = learner_func(
            numeric_cols=["income", "age", "score"],
            log_cols=["income"],
        )
        if not isinstance(result, Pipeline):
            t.failed = True
            t.msg = f"Expected sklearn Pipeline; got {type(result).__name__}"
        return t

    def test_pipeline_has_two_steps():
        t = test_case()
        from sklearn.pipeline import Pipeline
        pipeline = learner_func(
            numeric_cols=["income", "age", "score"],
            log_cols=["income"],
        )
        if len(pipeline.steps) < 2:
            t.failed = True
            t.msg = "Pipeline must have at least 2 steps (preprocessor + estimator)"
        return t

    def test_pipeline_fits():
        t = test_case()
        pipeline = learner_func(
            numeric_cols=["income", "age", "score"],
            log_cols=["income"],
        )
        X = _make_pipeline_data()
        y = np.array([0, 0, 1, 1, 0, 1])
        try:
            pipeline.fit(X, y)
        except Exception as e:
            t.failed = True
            t.msg = f"pipeline.fit() raised an error: {e}"
        return t

    def test_pipeline_predicts_correct_shape():
        t = test_case()
        pipeline = learner_func(
            numeric_cols=["income", "age", "score"],
            log_cols=["income"],
        )
        X = _make_pipeline_data()
        y = np.array([0, 0, 1, 1, 0, 1])
        pipeline.fit(X, y)
        try:
            preds = pipeline.predict(X)
            if len(preds) != len(y):
                t.failed = True
                t.msg = f"Expected {len(y)} predictions; got {len(preds)}"
        except Exception as e:
            t.failed = True
            t.msg = f"pipeline.predict() raised an error: {e}"
        return t

    cases.extend([
        test_returns_pipeline(),
        test_pipeline_has_two_steps(),
        test_pipeline_fits(),
        test_pipeline_predicts_correct_shape(),
    ])
    print_feedback(cases)


# ── Exercise 6: standard_scaler_from_scratch ─────────────────────────────────

def exercise_6(learner_func):
    cases = []

    def test_output_shapes():
        t = test_case()
        X_tr, X_te = learner_func(TRAIN.copy(), TEST.copy())
        if X_tr.shape != TRAIN.shape:
            t.failed = True
            t.msg = f"Train output shape: expected {TRAIN.shape}, got {X_tr.shape}"
            return t
        if X_te.shape != TEST.shape:
            t.failed = True
            t.msg = f"Test output shape: expected {TEST.shape}, got {X_te.shape}"
        return t

    def test_train_mean_zero():
        t = test_case()
        X_tr, _ = learner_func(TRAIN.copy(), TEST.copy())
        means = X_tr.mean(axis=0)
        if not np.allclose(means, 0, atol=1e-9):
            t.failed = True
            t.msg = (f"Train output mean must be ≈ 0 per column; "
                     f"got {means.round(6).tolist()}")
        return t

    def test_train_std_one():
        t = test_case()
        X_tr, _ = learner_func(TRAIN.copy(), TEST.copy())
        stds = X_tr.std(axis=0, ddof=0)
        if not np.allclose(stds, 1, atol=1e-9):
            t.failed = True
            t.msg = (f"Train output std (ddof=0) must be ≈ 1 per column; "
                     f"got {stds.round(6).tolist()}")
        return t

    def test_test_uses_train_params():
        t = test_case()
        _, X_te = learner_func(TRAIN.copy(), TEST.copy())
        # feat1: mean=3, std=sqrt(2); test row 0: (3-3)/sqrt(2) = 0
        #                              test row 1: (6-3)/sqrt(2) = 2.1213
        expected_f1_0 = 0.0
        expected_f1_1 = 3.0 / np.sqrt(2)
        actual = X_te[:, 0]
        if not (np.isclose(actual[0], expected_f1_0, atol=1e-6) and
                np.isclose(actual[1], expected_f1_1, atol=1e-6)):
            t.failed = True
            t.msg = (f"Test feat1: expected [{expected_f1_0:.4f}, {expected_f1_1:.4f}], "
                     f"got {actual.round(4).tolist()}. "
                     f"Make sure you use the training mean and std, not the test set's.")
        return t

    def test_returns_numpy_arrays():
        t = test_case()
        X_tr, X_te = learner_func(TRAIN.copy(), TEST.copy())
        if not isinstance(X_tr, np.ndarray):
            t.failed = True
            t.msg = f"Train output must be np.ndarray; got {type(X_tr).__name__}"
            return t
        if not isinstance(X_te, np.ndarray):
            t.failed = True
            t.msg = f"Test output must be np.ndarray; got {type(X_te).__name__}"
        return t

    cases.extend([
        test_output_shapes(),
        test_train_mean_zero(),
        test_train_std_one(),
        test_test_uses_train_params(),
        test_returns_numpy_arrays(),
    ])
    print_feedback(cases)


# ── Exercise 7: minmax_scaler_from_scratch ────────────────────────────────────

def exercise_7(learner_func):
    cases = []

    def test_output_shapes():
        t = test_case()
        X_tr, X_te = learner_func(TRAIN.copy(), TEST.copy())
        if X_tr.shape != TRAIN.shape:
            t.failed = True
            t.msg = f"Train output shape: expected {TRAIN.shape}, got {X_tr.shape}"
            return t
        if X_te.shape != TEST.shape:
            t.failed = True
            t.msg = f"Test output shape: expected {TEST.shape}, got {X_te.shape}"
        return t

    def test_train_min_zero():
        t = test_case()
        X_tr, _ = learner_func(TRAIN.copy(), TEST.copy())
        mins = X_tr.min(axis=0)
        if not np.allclose(mins, 0, atol=1e-9):
            t.failed = True
            t.msg = f"Train output min must be ≈ 0 per column; got {mins.round(6).tolist()}"
        return t

    def test_train_max_one():
        t = test_case()
        X_tr, _ = learner_func(TRAIN.copy(), TEST.copy())
        maxs = X_tr.max(axis=0)
        if not np.allclose(maxs, 1, atol=1e-9):
            t.failed = True
            t.msg = f"Train output max must be ≈ 1 per column; got {maxs.round(6).tolist()}"
        return t

    def test_train_values():
        t = test_case()
        X_tr, _ = learner_func(TRAIN.copy(), TEST.copy())
        # feat1: min=1, max=5, range=4 → [0, 0.25, 0.5, 0.75, 1.0]
        expected_f1 = [0.0, 0.25, 0.5, 0.75, 1.0]
        actual_f1 = X_tr[:, 0].tolist()
        if not np.allclose(actual_f1, expected_f1, atol=1e-9):
            t.failed = True
            t.msg = (f"Train feat1: expected {expected_f1}, "
                     f"got {[round(v, 4) for v in actual_f1]}")
        return t

    def test_test_uses_train_params():
        t = test_case()
        _, X_te = learner_func(TRAIN.copy(), TEST.copy())
        # feat1: min=1, max=5; test=[3, 6] → [(3-1)/4, (6-1)/4] = [0.5, 1.25]
        expected_f1 = [0.5, 1.25]
        actual_f1 = X_te[:, 0].tolist()
        if not np.allclose(actual_f1, expected_f1, atol=1e-9):
            t.failed = True
            t.msg = (f"Test feat1: expected {expected_f1}, "
                     f"got {[round(v, 4) for v in actual_f1]}. "
                     f"Make sure you use the training min and max, not the test set's.")
        return t

    def test_returns_numpy_arrays():
        t = test_case()
        X_tr, X_te = learner_func(TRAIN.copy(), TEST.copy())
        if not isinstance(X_tr, np.ndarray):
            t.failed = True
            t.msg = f"Train output must be np.ndarray; got {type(X_tr).__name__}"
            return t
        if not isinstance(X_te, np.ndarray):
            t.failed = True
            t.msg = f"Test output must be np.ndarray; got {type(X_te).__name__}"
        return t

    cases.extend([
        test_output_shapes(),
        test_train_min_zero(),
        test_train_max_one(),
        test_train_values(),
        test_test_uses_train_params(),
        test_returns_numpy_arrays(),
    ])
    print_feedback(cases)
