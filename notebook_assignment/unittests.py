from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _make_data(n_features=10):
    X, y = make_regression(n_samples=300, n_features=n_features, noise=10, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _make_preds(n_features=5):
    X_train, X_test, y_train, y_test = _make_data(n_features)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred, n_features


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "calculate_error_metrics"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true, y_pred, _ = _make_preds()

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

        required_keys = {
            "mse_manual", "rmse_manual", "mae_manual",
            "mse_sklearn", "rmse_sklearn", "mae_sklearn",
        }
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        for key in ("mse_manual", "mae_manual"):
            t = test_case()
            try:
                val = float(result.get(key, -1))
                if val < 0:
                    raise ValueError
            except (TypeError, ValueError):
                t.failed = True
                t.msg = f"{key} must be a non-negative float"
                t.want = ">= 0"
                t.got = result.get(key)
            cases.append(t)

        t = test_case()
        if not np.isclose(
            float(result.get("rmse_manual", -1)),
            float(result.get("mse_manual", 0)) ** 0.5,
            rtol=1e-4,
        ):
            t.failed = True
            t.msg = "rmse_manual must equal sqrt(mse_manual)"
            t.want = f"sqrt({result.get('mse_manual')})"
            t.got = result.get("rmse_manual")
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "calculate_r_squared"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true, y_pred, n_features = _make_preds()

        try:
            result = learner_func(y_true, y_pred, n_features=n_features)
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

        required_keys = {"r2_manual", "r2_adjusted", "r2_sklearn", "variance_explained", "ss_res", "ss_tot"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        if not np.isclose(
            float(result.get("r2_manual", -99)),
            float(result.get("r2_sklearn", -99)),
            rtol=1e-3,
        ):
            t.failed = True
            t.msg = "r2_manual and r2_sklearn must be close"
            t.want = "close values"
            t.got = f"manual={result.get('r2_manual')}, sklearn={result.get('r2_sklearn')}"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "analyze_residuals"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true, y_pred, _ = _make_preds()

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

        required_keys = {"residuals", "mean", "std", "is_normal", "n_outliers"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        residuals = result.get("residuals")
        if not isinstance(residuals, np.ndarray) or residuals.shape != (len(y_true),):
            t.failed = True
            t.msg = "residuals must be a numpy array of shape (n_samples,)"
            t.want = (len(y_true),)
            t.got = getattr(residuals, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
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

        X_train, X_test, y_train, y_test = _make_data()
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge(alpha=1)": Ridge(alpha=1.0),
            "Lasso(alpha=1)": Lasso(alpha=1.0),
        }

        try:
            result = learner_func(X_train, X_test, y_train, y_test, models_dict=models)
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
        if len(result) != len(models):
            t.failed = True
            t.msg = "Result must have one entry per model"
            t.want = len(models)
            t.got = len(result)
        cases.append(t)

        t = test_case()
        for name, sub in result.items():
            if not isinstance(sub, dict):
                t.failed = True
                t.msg = f"result['{name}'] must be a dict of metrics"
                t.want = dict
                t.got = type(sub)
                break
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5a(learner_func):
    def g():
        cases = []
        func_name = "asymmetric_loss"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(100)
        y_pred = rng.standard_normal(100)

        try:
            result = learner_func(y_true, y_pred, cost_under=2.0, cost_over=1.0)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        try:
            if float(result) < 0:
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "asymmetric_loss must return a non-negative float"
            t.want = ">= 0"
            t.got = result
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5b(learner_func):
    def g():
        cases = []
        func_name = "quantile_loss"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(100)
        y_pred = rng.standard_normal(100)

        try:
            result = learner_func(y_true, y_pred, quantile=0.5)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        try:
            if float(result) < 0:
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "quantile_loss must return a non-negative float"
            t.want = ">= 0"
            t.got = result
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5c(learner_func):
    def g():
        cases = []
        func_name = "percentage_error"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true = np.array([100.0, 200.0, 150.0])
        y_pred = np.array([110.0, 190.0, 150.0])

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
        try:
            if float(result) < 0:
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "percentage_error must return a non-negative float"
            t.want = ">= 0"
            t.got = result
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5d(learner_func):
    def g():
        cases = []
        func_name = "threshold_accuracy"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.2, 1.8, 3.3, 3.6])

        try:
            result = learner_func(y_true, y_pred, threshold=0.5)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        try:
            val = float(result)
            if not (0.0 <= val <= 100.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "threshold_accuracy must return a float in [0.0, 100.0]"
            t.want = "float in [0.0, 100.0]"
            t.got = result
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6a(learner_func):
    def g():
        cases = []
        func_name = "mse_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

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
        try:
            if not np.isclose(float(result), 0.0, atol=1e-10):
                t.failed = True
                t.msg = "MSE of identical arrays must be 0.0"
                t.want = 0.0
                t.got = result
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "mse_scratch must return a float"
            t.want = "float"
            t.got = type(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6b(learner_func):
    def g():
        cases = []
        func_name = "rmse_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])

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
        try:
            if not np.isclose(float(result), 1.0, atol=1e-6):
                t.failed = True
                t.msg = "RMSE of unit-offset arrays must be 1.0"
                t.want = 1.0
                t.got = result
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "rmse_scratch must return a float"
            t.want = "float"
            t.got = type(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6c(learner_func):
    def g():
        cases = []
        func_name = "mae_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

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
        try:
            if not np.isclose(float(result), 0.0, atol=1e-10):
                t.failed = True
                t.msg = "MAE of identical arrays must be 0.0"
                t.want = 0.0
                t.got = result
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "mae_scratch must return a float"
            t.want = "float"
            t.got = type(result)
        cases.append(t)

        y_true2 = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred2 = np.array([2.0, 3.0, 4.0, 5.0])
        try:
            result2 = learner_func(y_true2, y_pred2)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name} with offset arrays"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        try:
            if not np.isclose(float(result2), 1.0, atol=1e-6):
                t.failed = True
                t.msg = "MAE of unit-offset arrays must be 1.0"
                t.want = 1.0
                t.got = result2
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "mae_scratch must return a float"
            t.want = "float"
            t.got = type(result2)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6d(learner_func):
    def g():
        cases = []
        func_name = "r2_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

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
        try:
            if not np.isclose(float(result), 1.0, atol=1e-6):
                t.failed = True
                t.msg = "R\u00b2 of perfect predictions must be 1.0"
                t.want = 1.0
                t.got = result
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "r2_scratch must return a float"
            t.want = "float"
            t.got = type(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6e(learner_func):
    def g():
        cases = []
        func_name = "adjusted_r2_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

        try:
            result = learner_func(y_true, y_pred, n_features=2)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        try:
            if not np.isclose(float(result), 1.0, atol=1e-6):
                t.failed = True
                t.msg = "Adjusted R² of perfect predictions must be 1.0"
                t.want = 1.0
                t.got = result
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "adjusted_r2_scratch must return a float"
            t.want = "float"
            t.got = type(result)
        cases.append(t)

        y_true2, y_pred2, n_features = _make_preds(n_features=3)
        try:
            result2 = learner_func(y_true2, y_pred2, n_features=n_features)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name} with real data"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        try:
            val = float(result2)
            if not (val <= 1.0):
                t.failed = True
                t.msg = "Adjusted R² must be <= 1.0"
                t.want = "<= 1.0"
                t.got = val
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "adjusted_r2_scratch must return a float"
            t.want = "float"
            t.got = type(result2)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
