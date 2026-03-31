from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif


def _make_clf_data():
    X, y = make_classification(
        n_samples=300, n_features=20, n_informative=10, n_redundant=5, random_state=42
    )
    return X, y


def _make_reg_data():
    X, y = make_regression(n_samples=300, n_features=20, n_informative=10, random_state=42)
    return X, y


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "apply_variance_threshold"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_clf_data()

        try:
            result = learner_func(X, threshold=0.1)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, tuple) or len(result) != 2:
            t.failed = True
            t.msg = "Must return a tuple of length 2: (selector, X_filtered)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        selector, X_filtered = result

        t = test_case()
        if not isinstance(selector, VarianceThreshold):
            t.failed = True
            t.msg = "selector must be a VarianceThreshold instance"
            t.want = VarianceThreshold
            t.got = type(selector)
        cases.append(t)

        t = test_case()
        if not isinstance(X_filtered, np.ndarray):
            t.failed = True
            t.msg = "X_filtered must be a numpy array"
            t.want = np.ndarray
            t.got = type(X_filtered)
        cases.append(t)

        t = test_case()
        if X_filtered.shape[0] != X.shape[0]:
            t.failed = True
            t.msg = "X_filtered must have same number of rows as X"
            t.want = X.shape[0]
            t.got = X_filtered.shape[0]
        cases.append(t)

        t = test_case()
        if X_filtered.shape[1] > X.shape[1]:
            t.failed = True
            t.msg = "X_filtered must have equal or fewer columns than X"
            t.want = f"<= {X.shape[1]}"
            t.got = X_filtered.shape[1]
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "plot_variance_scores"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_clf_data()
        selector = VarianceThreshold(threshold=0.1)
        selector.fit(X)
        variances = selector.variances_
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        try:
            import matplotlib
            matplotlib.use("Agg")
            learner_func(variances, feature_names, threshold=0.1, top_n=10)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        cases.append(t)
        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "select_with_mutual_information"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_clf_data()
        k = 8

        try:
            result = learner_func(X, y, k, task_type="classification")
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, tuple) or len(result) != 2:
            t.failed = True
            t.msg = "Must return a tuple of length 2: (selector, X_selected)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        selector, X_selected = result

        t = test_case()
        if not isinstance(X_selected, np.ndarray) or X_selected.shape != (X.shape[0], k):
            t.failed = True
            t.msg = "X_selected shape must be (n_samples, k)"
            t.want = (X.shape[0], k)
            t.got = getattr(X_selected, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "select_with_f_classif"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_clf_data()
        k = 8
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        try:
            result = learner_func(X, y, k, feature_names)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, tuple) or len(result) != 3:
            t.failed = True
            t.msg = "Must return a tuple of length 3: (selector, X_selected, scores_df)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        selector, X_selected, scores_df = result

        t = test_case()
        if not isinstance(X_selected, np.ndarray) or X_selected.shape != (X.shape[0], k):
            t.failed = True
            t.msg = "X_selected shape must be (n_samples, k)"
            t.want = (X.shape[0], k)
            t.got = getattr(X_selected, "shape", None)
        cases.append(t)

        t = test_case()
        if not isinstance(scores_df, pd.DataFrame):
            t.failed = True
            t.msg = "scores_df must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(scores_df)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "select_with_chi2"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_clf_data()
        # chi2 requires non-negative input
        scaler = MinMaxScaler()
        X_nonneg = scaler.fit_transform(X)
        k = 8

        try:
            result = learner_func(X_nonneg, y, k)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, tuple) or len(result) != 2:
            t.failed = True
            t.msg = "Must return a tuple of length 2: (selector, X_selected)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        _, X_selected = result

        t = test_case()
        if not isinstance(X_selected, np.ndarray) or X_selected.shape != (X.shape[0], k):
            t.failed = True
            t.msg = "X_selected shape must be (n_samples, k)"
            t.want = (X.shape[0], k)
            t.got = getattr(X_selected, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "select_for_regression"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_reg_data()
        k = 8
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        try:
            result = learner_func(X, y, k, feature_names)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, tuple) or len(result) != 3:
            t.failed = True
            t.msg = "Must return a tuple of length 3: (selector, X_selected, scores_df)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        selector, X_selected, scores_df = result

        t = test_case()
        if not isinstance(X_selected, np.ndarray) or X_selected.shape != (X.shape[0], k):
            t.failed = True
            t.msg = "X_selected shape must be (n_samples, k)"
            t.want = (X.shape[0], k)
            t.got = getattr(X_selected, "shape", None)
        cases.append(t)

        t = test_case()
        if not isinstance(scores_df, pd.DataFrame):
            t.failed = True
            t.msg = "scores_df must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(scores_df)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_7(learner_func):
    def g():
        cases = []
        func_name = "compare_selection_methods"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_clf_data()
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        try:
            result = learner_func(X, y, feature_names)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, pd.DataFrame):
            t.failed = True
            t.msg = "Result must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        t = test_case()
        required_cols = {"f_rank", "mi_rank", "rank_diff"}
        if not required_cols.issubset(set(result.columns)):
            t.failed = True
            t.msg = "DataFrame must contain columns: f_rank, mi_rank, rank_diff"
            t.want = required_cols
            t.got = set(result.columns)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_8(learner_func):
    def g():
        cases = []
        func_name = "find_correlated_pairs"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_clf_data()
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        try:
            result = learner_func(X_df, threshold=0.85)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, list):
            t.failed = True
            t.msg = "Result must be a list of correlated pairs"
            t.want = list
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if result and not isinstance(result[0], tuple):
            t.failed = True
            t.msg = "Each element must be a tuple (feature_a, feature_b, ...)"
            t.want = "list of tuples"
            t.got = type(result[0])
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_9(learner_func):
    def g():
        cases = []
        func_name = "remove_correlated_features"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_clf_data()
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        # dummy target scores (higher = better)
        target_scores = {f"feature_{i}": float(np.random.default_rng(i).random()) for i in range(X.shape[1])}

        try:
            result = learner_func(X_df, target_scores, threshold=0.85)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, tuple) or len(result) != 2:
            t.failed = True
            t.msg = "Must return a tuple of length 2: (features_to_keep, features_removed)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        features_to_keep, features_removed = result

        t = test_case()
        if not isinstance(features_to_keep, (list, set)):
            t.failed = True
            t.msg = "features_to_keep must be a list or set"
            t.want = "list or set"
            t.got = type(features_to_keep)
        cases.append(t)

        t = test_case()
        total = len(set(features_to_keep)) + len(set(features_removed))
        if total != X_df.shape[1]:
            t.failed = True
            t.msg = "features_to_keep + features_removed must cover all original features"
            t.want = X_df.shape[1]
            t.got = total
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_10(learner_func):
    def g():
        cases = []
        func_name = "create_selection_pipeline"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        from sklearn.feature_selection import f_classif

        try:
            result = learner_func(score_func=f_classif, k=5)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, Pipeline):
            t.failed = True
            t.msg = "Result must be an sklearn Pipeline"
            t.want = Pipeline
            t.got = type(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_11(learner_func):
    def g():
        cases = []
        func_name = "compute_f_statistic_from_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_clf_data()

        try:
            result = learner_func(X, y)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, np.ndarray):
            t.failed = True
            t.msg = "Result must be a numpy array"
            t.want = np.ndarray
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        t = test_case()
        if result.shape != (X.shape[1],):
            t.failed = True
            t.msg = f"Result must have shape ({X.shape[1]},)"
            t.want = (X.shape[1],)
            t.got = result.shape
        cases.append(t)

        t = test_case()
        sklearn_scores, _ = f_classif(X, y)
        if not np.allclose(result, sklearn_scores, rtol=1e-3, atol=1e-3):
            t.failed = True
            t.msg = "F-scores do not match sklearn's f_classif"
            t.want = "Values close to sklearn's f_classif output"
            t.got = result[:5].tolist()
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
