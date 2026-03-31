from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _make_binary():
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    return X, y


def _make_multiclass():
    X, y = make_classification(
        n_samples=300, n_features=10, n_classes=3, n_informative=5, random_state=42
    )
    return X, y


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "binary_logistic_regression"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_binary()

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
        if not isinstance(result, LogisticRegression):
            t.failed = True
            t.msg = "Result must be a LogisticRegression"
            t.want = LogisticRegression
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if not hasattr(result, "coef_"):
            t.failed = True
            t.msg = "Model must be fitted"
            t.want = "fitted LogisticRegression with coef_"
            t.got = "unfitted"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "compute_classification_metrics"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_pred = rng.integers(0, 2, 100)

        try:
            result = learner_func(y_true, y_pred)
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

        required_keys = {"accuracy", "precision", "recall", "f1"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = "Dict must contain keys: accuracy, precision, recall, f1"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        for key in required_keys:
            if key in result:
                try:
                    v = float(result[key])
                    if not (0.0 <= v <= 1.0):
                        t.failed = True
                        t.msg = f"'{key}' value must be in [0.0, 1.0]"
                        t.want = "float in [0.0, 1.0]"
                        t.got = v
                        break
                except (TypeError, ValueError):
                    t.failed = True
                    t.msg = f"'{key}' must be numeric"
                    t.want = "float"
                    t.got = type(result[key])
                    break
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "compute_odds_ratios"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_binary()
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        try:
            result = learner_func(model, feature_names)
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

        t = test_case()
        if set(result.keys()) != set(feature_names):
            t.failed = True
            t.msg = "Keys must match feature_names"
            t.want = set(feature_names)
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        for k, v in result.items():
            try:
                if float(v) <= 0:
                    t.failed = True
                    t.msg = f"Odds ratio for '{k}' must be positive (exp(coef) > 0)"
                    t.want = "> 0"
                    t.got = v
                    break
            except (TypeError, ValueError):
                t.failed = True
                t.msg = "Odds ratio values must be numeric"
                t.want = "positive float"
                t.got = type(v)
                break
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "plot_decision_boundary"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        try:
            import matplotlib
            _prev_backend = matplotlib.get_backend()
            matplotlib.use("Agg")
            learner_func(model, X, y, title="Test Decision Boundary")
            matplotlib.use(_prev_backend)
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


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "multiclass_logistic_regression"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_multiclass()

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
        if not isinstance(result, LogisticRegression):
            t.failed = True
            t.msg = "Result must be a LogisticRegression"
            t.want = LogisticRegression
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if not hasattr(result, "coef_") or result.coef_.shape[0] != 3:
            t.failed = True
            t.msg = "Model must be fitted for 3 classes"
            t.want = "fitted multiclass LogisticRegression"
            t.got = getattr(result, "coef_", {}).shape if hasattr(result, "coef_") else "unfitted"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "l2_logistic_regression"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_binary()

        try:
            result = learner_func(X, y, C=1.0)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, LogisticRegression):
            t.failed = True
            t.msg = "Result must be a LogisticRegression"
            t.want = LogisticRegression
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if float(result.l1_ratio) != 0.0:
            t.failed = True
            t.msg = "Model must use L2 regularization (l1_ratio=0)"
            t.want = 0.0
            t.got = result.l1_ratio
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_7(learner_func):
    def g():
        cases = []
        func_name = "l1_logistic_regression"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_binary()

        try:
            result = learner_func(X, y, C=1.0)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, LogisticRegression):
            t.failed = True
            t.msg = "Result must be a LogisticRegression"
            t.want = LogisticRegression
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if float(result.l1_ratio) != 1.0:
            t.failed = True
            t.msg = "Model must use L1 regularization (l1_ratio=1)"
            t.want = 1.0
            t.got = result.l1_ratio
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_8(learner_func):
    def g():
        cases = []
        func_name = "sigmoid"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        try:
            result_zero = learner_func(0)
            result_large = learner_func(100)
            result_neg = learner_func(-100)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        try:
            if not np.isclose(float(result_zero), 0.5, atol=1e-6):
                t.failed = True
                t.msg = "sigmoid(0) must equal 0.5"
                t.want = 0.5
                t.got = result_zero
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "sigmoid must return a numeric value"
            t.want = "float"
            t.got = type(result_zero)
        cases.append(t)

        t = test_case()
        try:
            if not np.isclose(float(result_large), 1.0, atol=1e-5):
                t.failed = True
                t.msg = "sigmoid(100) must be approximately 1.0"
                t.want = "~1.0"
                t.got = result_large
        except (TypeError, ValueError):
            pass
        cases.append(t)

        t = test_case()
        try:
            if not np.isclose(float(result_neg), 0.0, atol=1e-5):
                t.failed = True
                t.msg = "sigmoid(-100) must be approximately 0.0"
                t.want = "~0.0"
                t.got = result_neg
        except (TypeError, ValueError):
            pass
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_9(learner_func):
    def g():
        cases = []
        func_name = "compute_cost_logistic"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        y = rng.integers(0, 2, 50).astype(float)
        theta = np.zeros(3)

        try:
            result = learner_func(X, y, theta)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        try:
            cost = float(result)
            if cost < 0:
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "cost must be a non-negative float"
            t.want = ">= 0"
            t.got = result
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_10(learner_func):
    def g():
        cases = []
        func_name = "gradient_descent_logistic"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        y = rng.integers(0, 2, 50).astype(float)
        theta = np.zeros(3)
        iterations = 100

        try:
            result = learner_func(X, theta, y, learning_rate=0.1, iterations=iterations)
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
            t.msg = "Must return a tuple of length 2: (theta, costs)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        theta_out, costs = result

        t = test_case()
        if not isinstance(theta_out, np.ndarray) or theta_out.shape != (3,):
            t.failed = True
            t.msg = "theta must be a numpy array of shape (3,)"
            t.want = (3,)
            t.got = getattr(theta_out, "shape", None)
        cases.append(t)

        t = test_case()
        if not isinstance(costs, (list, np.ndarray)) or len(costs) != iterations:
            t.failed = True
            t.msg = f"costs must have length {iterations}"
            t.want = iterations
            t.got = len(costs) if hasattr(costs, "__len__") else "N/A"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
