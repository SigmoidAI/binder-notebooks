from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _make_data():
    X, y = load_digits(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "perform_grid_search"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, _, y_train, _ = _make_data()

        try:
            result = learner_func(X_train, y_train, cv_folds=2, verbose=False)
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

        required_keys = {"best_params", "best_score", "results_df", "total_time", "n_iterations"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        try:
            score = float(result.get("best_score", -1))
            if not (0.0 <= score <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "best_score must be a float in [0.0, 1.0]"
            t.want = "float in [0.0, 1.0]"
            t.got = result.get("best_score")
        cases.append(t)

        t = test_case()
        if not isinstance(result.get("results_df"), pd.DataFrame):
            t.failed = True
            t.msg = "results_df must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(result.get("results_df"))
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "perform_random_search"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, _, y_train, _ = _make_data()

        try:
            result = learner_func(X_train, y_train, n_iter=10, cv_folds=2, verbose=False)
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

        required_keys = {"best_params", "best_score", "results_df", "total_time", "n_iterations"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        try:
            score = float(result.get("best_score", -1))
            if not (0.0 <= score <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "best_score must be a float in [0.0, 1.0]"
            t.want = "float in [0.0, 1.0]"
            t.got = result.get("best_score")
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "perform_bayesian_optimization"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, _, y_train, _ = _make_data()

        try:
            result = learner_func(X_train, y_train, n_trials=5, cv_folds=2, verbose=False)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        # result is None if optuna not installed — acceptable
        t = test_case()
        if result is not None:
            if not isinstance(result, dict):
                t.failed = True
                t.msg = "Result must be a dict (or None if optuna is unavailable)"
                t.want = dict
                t.got = type(result)
        cases.append(t)

        t = test_case()
        if result is not None:
            required_keys = {"best_params", "best_score"}
            if not required_keys.issubset(set(result.keys())):
                t.failed = True
                t.msg = f"Result must have keys: {required_keys}"
                t.want = required_keys
                t.got = set(result.keys())
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "analyze_parameter_importance"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # Build a small results_df
        rng = np.random.default_rng(42)
        n = 20
        results_df = pd.DataFrame({
            "param_n_estimators": rng.choice([50, 100, 200], n),
            "param_max_depth": rng.choice([3, 5, None], n),
            "mean_score": rng.uniform(0.7, 0.95, n),
        })

        try:
            result = learner_func(results_df, method="correlation")
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
            t.msg = "Result must be a dict of {param_name: importance_float}"
            t.want = dict
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if len(result) < 1:
            t.failed = True
            t.msg = "Result must have at least one parameter importance entry"
            t.want = ">= 1 entry"
            t.got = len(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "compare_tuning_strategies"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, _, y_train, _ = _make_data()

        try:
            result = learner_func(X_train, y_train, n_iter_budget=5)
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
        cases.append(t)

        t = test_case()
        if len(result) < 2:
            t.failed = True
            t.msg = "Must compare at least 2 tuning strategies"
            t.want = ">= 2 keys"
            t.got = len(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "grid_search_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, _, y_train, _ = _make_data()
        param_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}

        try:
            result = learner_func(
                X_train, y_train,
                param_grid=param_grid,
                model_class=RandomForestClassifier,
                cv=2,
                scoring="accuracy"
            )
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

        required_keys = {"best_params", "best_score", "all_results"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        n_expected = 4  # 2 * 2 combinations
        if len(result.get("all_results", [])) != n_expected:
            t.failed = True
            t.msg = "all_results must contain one entry per parameter combination"
            t.want = n_expected
            t.got = len(result.get("all_results", []))
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
