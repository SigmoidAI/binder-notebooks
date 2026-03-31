from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression


def _make_time_frame(n=240):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    signal = np.sin(np.arange(n) / 12.0) + np.arange(n) * 0.01 + rng.normal(0, 0.1, n)
    df = pd.DataFrame(
        {
            "date": dates,
            "lag_1": pd.Series(signal).shift(1).bfill(),
            "lag_7": pd.Series(signal).shift(7).bfill(),
            "target": signal,
        }
    )
    return df


def _make_nested_data():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return X.iloc[:250].copy(), y.iloc[:250].copy()


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "create_time_split"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        df = _make_time_frame()

        try:
            result = learner_func(df, target_col="target", date_col="date", test_size=0.2)
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

        required = {"X_train", "X_test", "y_train", "y_test"}
        t = test_case()
        if not required.issubset(result.keys()):
            t.failed = True
            t.msg = f"Result must contain keys: {required}"
            t.want = required
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        train_max = pd.to_datetime(result["X_train"]["date"]).max()
        test_min = pd.to_datetime(result["X_test"]["date"]).min()
        if not (train_max < test_min):
            t.failed = True
            t.msg = "Temporal split must preserve chronology"
            t.want = "max(train dates) < min(test dates)"
            t.got = (str(train_max), str(test_min))
        cases.append(t)

        return cases

    print_feedback(g())


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "create_walk_forward_splits"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        try:
            result = learner_func(n_samples=60, initial_window=30, horizon=10, step=10)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, list) or len(result) == 0:
            t.failed = True
            t.msg = "Result must be a non-empty list of splits"
            t.want = "list[(train_idx, test_idx)]"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        t = test_case()
        train_idx, test_idx = result[0]
        if max(train_idx) >= min(test_idx):
            t.failed = True
            t.msg = "Each walk-forward split must train on the past and test on the future"
            t.want = "max(train_idx) < min(test_idx)"
            t.got = (max(train_idx), min(test_idx))
        cases.append(t)

        return cases

    print_feedback(g())


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "run_nested_cv"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_nested_data()
        model = LogisticRegression(max_iter=2000, solver="liblinear")
        param_grid = {"C": [0.1, 1.0, 10.0]}

        try:
            result = learner_func(model, param_grid, X, y, outer_splits=3, inner_splits=2, scoring="accuracy", random_state=42)
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

        required = {"outer_scores", "mean_outer_score", "std_outer_score", "best_params_per_fold"}
        t = test_case()
        if not required.issubset(result.keys()):
            t.failed = True
            t.msg = f"Result must contain keys: {required}"
            t.want = required
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        if len(result["outer_scores"]) != 3:
            t.failed = True
            t.msg = "outer_scores must contain one value per outer fold"
            t.want = 3
            t.got = len(result["outer_scores"])
        cases.append(t)

        return cases

    print_feedback(g())


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "compare_nested_vs_non_nested"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_nested_data()
        model = LogisticRegression(max_iter=2000, solver="liblinear")
        param_grid = {"C": [0.1, 1.0, 10.0]}

        try:
            result = learner_func(model, param_grid, X, y, cv_splits=3, scoring="accuracy", random_state=42)
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

        required = {"nested_score", "non_nested_score", "optimism_gap"}
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
        func_name = "kfold_indices"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        try:
            result = learner_func(n_samples=20, n_splits=5, shuffle=False, random_state=None)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, list) or len(result) != 5:
            t.failed = True
            t.msg = "Result must be a list with one split per fold"
            t.want = 5
            t.got = len(result) if isinstance(result, list) else type(result)
            return cases + [t]
        cases.append(t)

        all_val = []
        t = test_case()
        for train_idx, val_idx in result:
            all_val.extend(list(val_idx))
            if len(set(train_idx).intersection(set(val_idx))) != 0:
                t.failed = True
                t.msg = "Train and validation indices must be disjoint"
                t.want = "disjoint sets"
                t.got = "overlap detected"
                break
        cases.append(t)

        t = test_case()
        if sorted(all_val) != list(range(20)):
            t.failed = True
            t.msg = "Validation folds must cover each sample exactly once"
            t.want = list(range(20))
            t.got = sorted(all_val)
        cases.append(t)

        return cases

    print_feedback(g())


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "stratified_kfold_indices"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y = np.array([0] * 16 + [1] * 4)

        try:
            result = learner_func(y, n_splits=4, shuffle=False, random_state=None)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, list) or len(result) != 4:
            t.failed = True
            t.msg = "Result must be a list with one split per fold"
            t.want = 4
            t.got = len(result) if isinstance(result, list) else type(result)
            return cases + [t]
        cases.append(t)

        overall_rate = y.mean()
        t = test_case()
        for _, val_idx in result:
            if abs(y[list(val_idx)].mean() - overall_rate) > 0.15:
                t.failed = True
                t.msg = "Each validation fold must roughly preserve class balance"
                t.want = f"within 0.15 of {overall_rate:.2f}"
                t.got = [float(y[list(v)].mean()) for _, v in result]
                break
        cases.append(t)

        return cases

    print_feedback(g())
