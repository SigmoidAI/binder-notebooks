from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _make_clf_data(n_features=20):
    X, y = make_classification(
        n_samples=300, n_features=n_features, n_informative=10, n_redundant=4, random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def _make_reg_data(n_features=20):
    X, y = make_regression(n_samples=300, n_features=n_features, n_informative=10, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "lasso_feature_selection"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_reg_data()
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        try:
            result = learner_func(X, y, alpha=0.01, feature_names=feature_names)
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
            t.msg = "Must return a tuple of length 3: (selector, X_selected, coef_df)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        selector, X_selected, coef_df = result

        t = test_case()
        if not isinstance(X_selected, np.ndarray) or X_selected.shape[0] != X.shape[0]:
            t.failed = True
            t.msg = "X_selected must be a numpy array with same number of rows as X"
            t.want = f"shape ({X.shape[0]}, n_selected)"
            t.got = getattr(X_selected, "shape", None)
        cases.append(t)

        t = test_case()
        if not isinstance(coef_df, pd.DataFrame):
            t.failed = True
            t.msg = "coef_df must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(coef_df)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "tune_lasso_alpha"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_reg_data()
        alphas = [0.001, 0.01, 0.1, 1.0]

        try:
            result = learner_func(X, y, alphas=alphas)
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
        required_cols = {"alpha", "n_selected"}
        if not required_cols.issubset(set(result.columns)):
            t.failed = True
            t.msg = "DataFrame must have columns 'alpha' and 'n_selected'"
            t.want = required_cols
            t.got = set(result.columns)
        cases.append(t)

        t = test_case()
        if len(result) != len(alphas):
            t.failed = True
            t.msg = "DataFrame must have one row per alpha value"
            t.want = len(alphas)
            t.got = len(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "get_tree_gini_importance"

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
            result = learner_func(X, y, feature_names=feature_names)
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
            t.msg = "Must return a tuple of length 2: (imp_df, rf)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        imp_df, rf = result

        t = test_case()
        if not isinstance(imp_df, pd.DataFrame):
            t.failed = True
            t.msg = "imp_df must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(imp_df)
        cases.append(t)

        t = test_case()
        if not isinstance(rf, RandomForestClassifier):
            t.failed = True
            t.msg = "rf must be a RandomForestClassifier"
            t.want = RandomForestClassifier
            t.got = type(rf)
        cases.append(t)

        t = test_case()
        if not hasattr(rf, "feature_importances_"):
            t.failed = True
            t.msg = "RandomForestClassifier must be fitted"
            t.want = "fitted RandomForestClassifier"
            t.got = "unfitted"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "get_permutation_importance"

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
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        try:
            result = learner_func(model, X, y, feature_names=feature_names)
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
        required_cols = {"feature", "importance_mean", "importance_std"}
        if not required_cols.issubset(set(result.columns)):
            t.failed = True
            t.msg = "DataFrame must have columns: feature, importance_mean, importance_std"
            t.want = required_cols
            t.got = set(result.columns)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "apply_rfe"

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
        estimator = LogisticRegression(max_iter=500, random_state=42)
        n_features = 8

        try:
            result = learner_func(X, y, estimator=estimator, n_features=n_features, feature_names=feature_names)
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
            t.msg = "Must return a tuple of length 3: (rfe, X_selected, ranking_df)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        rfe, X_selected, ranking_df = result

        t = test_case()
        if not isinstance(rfe, RFE):
            t.failed = True
            t.msg = "rfe must be an sklearn RFE instance"
            t.want = RFE
            t.got = type(rfe)
        cases.append(t)

        t = test_case()
        if not isinstance(X_selected, np.ndarray) or X_selected.shape != (X.shape[0], n_features):
            t.failed = True
            t.msg = "X_selected shape must be (n_samples, n_features)"
            t.want = (X.shape[0], n_features)
            t.got = getattr(X_selected, "shape", None)
        cases.append(t)

        t = test_case()
        if not isinstance(ranking_df, pd.DataFrame):
            t.failed = True
            t.msg = "ranking_df must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(ranking_df)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "check_rfe_stability"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_clf_data()
        estimator = LogisticRegression(max_iter=500, random_state=42)

        try:
            result = learner_func(X, y, estimator=estimator, n_features=8, n_splits=5)
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
            t.msg = "Must return a tuple of length 2: (mean_jaccard, selected_sets)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        mean_jaccard, selected_sets = result

        t = test_case()
        try:
            mj = float(mean_jaccard)
            if not (0.0 <= mj <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "mean_jaccard must be a float in [0.0, 1.0]"
            t.want = "float in [0.0, 1.0]"
            t.got = mean_jaccard
        cases.append(t)

        t = test_case()
        if not isinstance(selected_sets, list):
            t.failed = True
            t.msg = "selected_sets must be a list"
            t.want = list
            t.got = type(selected_sets)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_7(learner_func):
    def g():
        cases = []
        func_name = "apply_sequential_selection"

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
        estimator = LogisticRegression(max_iter=500, random_state=42)

        try:
            result = learner_func(
                X, y, estimator=estimator, n_features=5,
                feature_names=feature_names, direction="forward"
            )
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
            t.msg = "Must return a tuple of length 2: (sfs, selected_features)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        sfs, selected_features = result

        t = test_case()
        if not isinstance(sfs, SequentialFeatureSelector):
            t.failed = True
            t.msg = "sfs must be a SequentialFeatureSelector instance"
            t.want = SequentialFeatureSelector
            t.got = type(sfs)
        cases.append(t)

        t = test_case()
        if not isinstance(selected_features, (list, np.ndarray)) or len(selected_features) != 5:
            t.failed = True
            t.msg = "selected_features must be a list of length n_features=5"
            t.want = "list of length 5"
            t.got = f"type={type(selected_features)}, len={len(selected_features) if hasattr(selected_features, '__len__') else 'N/A'}"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_8(learner_func):
    def g():
        cases = []
        func_name = "ensemble_feature_selection"

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
        k = 8

        try:
            result = learner_func(X, y, feature_names=feature_names, k=k)
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
            t.msg = "Must return a tuple of length 2: (rank_df, top_k)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        rank_df, top_k = result

        t = test_case()
        if not isinstance(rank_df, pd.DataFrame):
            t.failed = True
            t.msg = "rank_df must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(rank_df)
        cases.append(t)

        t = test_case()
        if not isinstance(top_k, (list, np.ndarray)) or len(top_k) != k:
            t.failed = True
            t.msg = f"top_k must be a list of length {k}"
            t.want = f"list of length {k}"
            t.got = f"type={type(top_k)}, len={len(top_k) if hasattr(top_k, '__len__') else 'N/A'}"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_9(learner_func):
    def g():
        cases = []
        func_name = "create_model_based_pipeline"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        try:
            result = learner_func(method="lasso", k=5)
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


def exercise_10(learner_func):
    def g():
        cases = []
        func_name = "forward_selection_from_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_clf_data()
        k = 5

        try:
            result = learner_func(X, y, k=k)
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
            t.msg = "Must return a tuple of length 2: (selected_indices, score_history)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        selected_indices, score_history = result

        t = test_case()
        if not isinstance(selected_indices, (list, np.ndarray)) or len(selected_indices) != k:
            t.failed = True
            t.msg = f"selected_indices must have length {k}"
            t.want = k
            t.got = len(selected_indices) if hasattr(selected_indices, "__len__") else "N/A"
        cases.append(t)

        t = test_case()
        if not isinstance(score_history, (list, np.ndarray)) or len(score_history) != k:
            t.failed = True
            t.msg = f"score_history must have length {k}"
            t.want = k
            t.got = len(score_history) if hasattr(score_history, "__len__") else "N/A"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
