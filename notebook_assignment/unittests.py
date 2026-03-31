from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import load_digits, make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _make_reg_data(n_features=5):
    X, y = make_regression(n_samples=200, n_features=n_features, noise=10, random_state=42)
    return X, y


def _make_clf_data():
    X, y = load_digits(return_X_y=True)
    return X, y


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "generate_learning_curve"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_reg_data()
        estimator = Ridge(alpha=1.0)

        try:
            result = learner_func(estimator, X, y, cv=3)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, dict):
            t.failed = True
            t.msg = "Result must be a dict"
            t.want = dict
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        required_keys = {"train_sizes", "train_mean", "val_mean", "train_scores", "val_scores"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        n = len(result.get("train_sizes", []))
        if n < 1:
            t.failed = True
            t.msg = "train_sizes must not be empty"
            t.want = "non-empty"
            t.got = n
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "evaluate_poly_degrees"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
        degrees = [1, 2, 3]

        try:
            result = learner_func(X, y, degrees=degrees)
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
            t.msg = "Result must be a list"
            t.want = list
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        t = test_case()
        if len(result) != len(degrees):
            t.failed = True
            t.msg = "Result must have one entry per degree"
            t.want = len(degrees)
            t.got = len(result)
        cases.append(t)

        t = test_case()
        for item in result:
            if not isinstance(item, dict) or "degree" not in item or "rmse" not in item:
                t.failed = True
                t.msg = "Each item must be a dict with 'degree' and 'rmse' keys"
                t.want = "{'degree': int, 'rmse': float}"
                t.got = item
                break
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "test_regularization"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
        alphas = [0.01, 1.0, 100.0]

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
        if not isinstance(result, list):
            t.failed = True
            t.msg = "Result must be a list"
            t.want = list
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        t = test_case()
        if len(result) != len(alphas):
            t.failed = True
            t.msg = "Result must have one entry per alpha"
            t.want = len(alphas)
            t.got = len(result)
        cases.append(t)

        t = test_case()
        for item in result:
            if not isinstance(item, dict) or "alpha" not in item:
                t.failed = True
                t.msg = "Each item must be a dict with 'alpha' key"
                t.want = "{'alpha': float, 'rmse': float}"
                t.got = item
                break
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "analyze_cv_stability"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_clf_data()
        model = DecisionTreeClassifier(random_state=42)

        try:
            result = learner_func(model, X, y, cv=5)
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
            t.msg = "Result must be a tuple (mean_score, std_score, scores)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        mean_score, std_score, scores = result

        t = test_case()
        try:
            if not (0.0 <= float(mean_score) <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "mean_score must be a float in [0.0, 1.0]"
            t.want = "float in [0.0, 1.0]"
            t.got = mean_score
        cases.append(t)

        t = test_case()
        if not isinstance(scores, np.ndarray) or scores.shape != (5,):
            t.failed = True
            t.msg = "scores must be a numpy array of shape (cv,)"
            t.want = (5,)
            t.got = getattr(scores, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "compare_models"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_clf_data()
        from sklearn.tree import DecisionTreeClassifier as DTC
        from sklearn.ensemble import RandomForestClassifier as RFC
        models_dict = {
            "DecisionTree(d=3)": DTC(max_depth=3, random_state=42),
            "DecisionTree(d=5)": DTC(max_depth=5, random_state=42),
            "RandomForest": RFC(n_estimators=20, random_state=42),
        }

        try:
            result = learner_func(models_dict, X, y, cv=3)
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
        if len(result) != len(models_dict):
            t.failed = True
            t.msg = "DataFrame must have one row per model"
            t.want = len(models_dict)
            t.got = len(result)
        cases.append(t)

        required_cols = {"Model", "Mean Accuracy"}
        t = test_case()
        if not required_cols.issubset(set(result.columns)):
            t.failed = True
            t.msg = f"DataFrame must have columns: {required_cols}"
            t.want = required_cols
            t.got = set(result.columns)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "learning_curve_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_clf_data()
        estimator = DecisionTreeClassifier(max_depth=5, random_state=42)

        try:
            result = learner_func(estimator, X, y)
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
            t.msg = "Result must be a tuple (sizes, train_scores, val_scores)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        sizes, train_scores, val_scores = result

        t = test_case()
        if not (len(sizes) == len(train_scores) == len(val_scores)):
            t.failed = True
            t.msg = "sizes, train_scores, val_scores must have the same length"
            t.want = "equal lengths"
            t.got = (len(sizes), len(train_scores), len(val_scores))
        cases.append(t)

        t = test_case()
        for score in train_scores:
            try:
                if not (0.0 <= float(score) <= 1.0):
                    raise ValueError
            except (TypeError, ValueError):
                t.failed = True
                t.msg = "All train_scores must be floats in [0.0, 1.0]"
                t.want = "float in [0.0, 1.0]"
                t.got = score
                break
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
