from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.linear_model import LogisticRegression


def _make_split_data():
    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.8, 0.2],
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    return X, y


def _make_cv_data():
    return load_breast_cancer(return_X_y=True, as_frame=True)


def _make_imbalanced_data():
    X, y = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=6,
        n_redundant=1,
        weights=[0.92, 0.08],
        random_state=42,
    )
    return pd.DataFrame(X), pd.Series(y)


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "create_stratified_splits"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_split_data()

        try:
            result = learner_func(X, y, test_size=0.2, val_size=0.25, random_state=42)
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

        required = {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test"}
        t = test_case()
        if not required.issubset(result.keys()):
            t.failed = True
            t.msg = f"Result must contain keys: {required}"
            t.want = required
            t.got = set(result.keys())
        cases.append(t)

        total = len(result["X_train"]) + len(result["X_val"]) + len(result["X_test"])
        t = test_case()
        if total != len(X):
            t.failed = True
            t.msg = "Split sizes must add up to original dataset size"
            t.want = len(X)
            t.got = total
        cases.append(t)

        base_rate = y.mean()
        t = test_case()
        for split_name in ("y_train", "y_val", "y_test"):
            if abs(result[split_name].mean() - base_rate) > 0.05:
                t.failed = True
                t.msg = "Class balance must remain close to the original ratio"
                t.want = f"within 0.05 of {base_rate:.3f}"
                t.got = {name: float(result[name].mean()) for name in ("y_train", "y_val", "y_test")}
                break
        cases.append(t)

        return cases

    print_feedback(g())


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "summarize_class_proportions"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_split_data()
        y_train = y.iloc[:300]
        y_val = y.iloc[300:400]
        y_test = y.iloc[400:]

        try:
            result = learner_func(y_train, y_val, y_test)
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
        expected_cols = {"train", "validation", "test"}
        if not expected_cols.issubset(result.columns):
            t.failed = True
            t.msg = f"DataFrame must contain columns: {expected_cols}"
            t.want = expected_cols
            t.got = set(result.columns)
        cases.append(t)

        return cases

    print_feedback(g())


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "run_kfold_cv"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_cv_data()
        model = LogisticRegression(max_iter=2000, solver="liblinear")

        try:
            result = learner_func(model, X, y, n_splits=5, shuffle=True, random_state=42, scoring="accuracy")
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

        required = {"fold_scores", "mean_score", "std_score"}
        t = test_case()
        if not required.issubset(result.keys()):
            t.failed = True
            t.msg = f"Result must contain keys: {required}"
            t.want = required
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        scores = result.get("fold_scores")
        if len(scores) != 5:
            t.failed = True
            t.msg = "fold_scores must contain one score per fold"
            t.want = 5
            t.got = len(scores)
        cases.append(t)

        return cases

    print_feedback(g())


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "analyze_fold_variance"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        scores = np.array([0.91, 0.94, 0.92, 0.90, 0.95])

        try:
            result = learner_func(scores)
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

        required = {"mean", "std", "min", "max", "range"}
        t = test_case()
        if not required.issubset(result.keys()):
            t.failed = True
            t.msg = f"Result must contain keys: {required}"
            t.want = required
            t.got = set(result.keys())
        cases.append(t)

        return cases

    print_feedback(g())


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "compare_kfold_strategies"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_imbalanced_data()
        model = LogisticRegression(max_iter=2000, solver="liblinear")

        try:
            result = learner_func(model, X, y, n_splits=5, random_state=42)
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

        required = {"kfold_scores", "stratified_scores", "kfold_mean", "stratified_mean"}
        t = test_case()
        if not required.issubset(result.keys()):
            t.failed = True
            t.msg = f"Result must contain keys: {required}"
            t.want = required
            t.got = set(result.keys())
        cases.append(t)

        return cases

    print_feedback(g())


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "inspect_fold_class_balance"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        _, y = _make_imbalanced_data()

        try:
            result = learner_func(y, n_splits=5, random_state=42)
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

        required_cols = {"kfold_positive_rate", "stratified_positive_rate"}
        t = test_case()
        if not required_cols.issubset(result.columns):
            t.failed = True
            t.msg = f"DataFrame must contain columns: {required_cols}"
            t.want = required_cols
            t.got = set(result.columns)
        cases.append(t)

        return cases

    print_feedback(g())
