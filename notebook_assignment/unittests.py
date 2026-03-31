from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures


def _make_simple():
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, (100, 1))
    y = 2.5 * X.ravel() + 3.0 + rng.normal(0, 1, 100)
    return X, y


def _make_multi(n_features=5):
    X, y = make_regression(n_samples=200, n_features=n_features, n_informative=n_features, random_state=42)
    return X, y


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "simple_linear_regression"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_simple()

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
        if not isinstance(result, tuple) or len(result) != 3:
            t.failed = True
            t.msg = "Must return a tuple of length 3: (model, slope, intercept)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        model, slope, intercept = result

        t = test_case()
        if not isinstance(model, LinearRegression):
            t.failed = True
            t.msg = "model must be an sklearn LinearRegression"
            t.want = LinearRegression
            t.got = type(model)
        cases.append(t)

        t = test_case()
        if not hasattr(model, "coef_"):
            t.failed = True
            t.msg = "model must be fitted"
            t.want = "fitted LinearRegression with coef_"
            t.got = "unfitted model"
        cases.append(t)

        t = test_case()
        try:
            s = float(slope)
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "slope must be a numeric value"
            t.want = "float"
            t.got = type(slope)
        cases.append(t)

        t = test_case()
        try:
            ic = float(intercept)
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "intercept must be a numeric value"
            t.want = "float"
            t.got = type(intercept)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "compute_metrics"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_simple()
        model = LinearRegression()
        model.fit(X, y)

        try:
            result = learner_func(model, X, y)
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
            t.msg = "Must return a tuple of length 2: (mse, r2)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        mse, r2 = result

        t = test_case()
        try:
            if float(mse) < 0:
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "mse must be a non-negative float"
            t.want = ">= 0"
            t.got = mse
        cases.append(t)

        t = test_case()
        try:
            r = float(r2)
            if r > 1.001:
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "r2 must be a float <= 1"
            t.want = "<= 1"
            t.got = r2
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "multiple_linear_regression"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_multi()
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]

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
        if not isinstance(result, tuple) or len(result) != 2:
            t.failed = True
            t.msg = "Must return a tuple of length 2: (model, coefficients)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        model, coefficients = result

        t = test_case()
        if not isinstance(model, LinearRegression):
            t.failed = True
            t.msg = "model must be a LinearRegression"
            t.want = LinearRegression
            t.got = type(model)
        cases.append(t)

        t = test_case()
        if not isinstance(coefficients, dict):
            t.failed = True
            t.msg = "coefficients must be a dict mapping feature names to values"
            t.want = dict
            t.got = type(coefficients)
        cases.append(t)

        t = test_case()
        if set(coefficients.keys()) != set(feature_names):
            t.failed = True
            t.msg = "coefficients keys must match feature_names"
            t.want = set(feature_names)
            t.got = set(coefficients.keys())
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "compute_vif"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_multi()
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]

        try:
            result = learner_func(X, feature_names)
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
            t.msg = "vif_dict must be a dict"
            t.want = dict
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        t = test_case()
        if set(result.keys()) != set(feature_names):
            t.failed = True
            t.msg = "vif_dict keys must match feature_names"
            t.want = set(feature_names)
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        for k, v in result.items():
            try:
                if float(v) < 1.0:
                    t.failed = True
                    t.msg = f"VIF value for '{k}' must be >= 1.0"
                    t.want = ">= 1.0"
                    t.got = v
                    break
            except (TypeError, ValueError):
                t.failed = True
                t.msg = "VIF values must be numeric"
                t.want = "float >= 1"
                t.got = type(v)
                break
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "polynomial_regression"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_simple()
        degree = 3

        try:
            result = learner_func(X, y, degree)
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
            t.msg = "Must return a tuple of length 3: (model, poly, X_poly)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        model, poly, X_poly = result

        t = test_case()
        if not isinstance(model, LinearRegression):
            t.failed = True
            t.msg = "model must be a LinearRegression"
            t.want = LinearRegression
            t.got = type(model)
        cases.append(t)

        t = test_case()
        if not isinstance(poly, PolynomialFeatures):
            t.failed = True
            t.msg = "poly must be a PolynomialFeatures"
            t.want = PolynomialFeatures
            t.got = type(poly)
        cases.append(t)

        t = test_case()
        if not isinstance(X_poly, np.ndarray) or X_poly.shape[0] != X.shape[0]:
            t.failed = True
            t.msg = "X_poly must be a numpy array with same number of rows as X"
            t.want = f"array of shape ({X.shape[0]}, ...)"
            t.got = getattr(X_poly, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "ridge_regression"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_multi()

        try:
            result = learner_func(X, y, alpha=1.0)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, Ridge):
            t.failed = True
            t.msg = "Result must be a Ridge model"
            t.want = Ridge
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if not hasattr(result, "coef_"):
            t.failed = True
            t.msg = "Ridge model must be fitted"
            t.want = "fitted Ridge with coef_"
            t.got = "unfitted"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_7(learner_func):
    def g():
        cases = []
        func_name = "lasso_regression"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_multi()

        try:
            result = learner_func(X, y, alpha=0.1)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, Lasso):
            t.failed = True
            t.msg = "Result must be a Lasso model"
            t.want = Lasso
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if not hasattr(result, "coef_"):
            t.failed = True
            t.msg = "Lasso model must be fitted"
            t.want = "fitted Lasso with coef_"
            t.got = "unfitted"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_8(learner_func):
    def g():
        cases = []
        func_name = "compute_cost"

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
        y = rng.standard_normal(50)
        theta = np.zeros(X.shape[1])

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
            t.msg = "cost must be a non-negative float (MSE cost)"
            t.want = ">= 0"
            t.got = result
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_9(learner_func):
    def g():
        cases = []
        func_name = "gradient_descent"

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
        y = rng.standard_normal(50)
        theta = np.zeros(3)
        iterations = 100

        try:
            result = learner_func(X, theta, y, learning_rate=0.01, iterations=iterations)
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
            t.msg = f"costs must have length {iterations} (one per iteration)"
            t.want = iterations
            t.got = len(costs) if hasattr(costs, "__len__") else "N/A"
        cases.append(t)

        t = test_case()
        costs_arr = np.array(costs, dtype=float)
        if not (float(costs_arr[-1]) < float(costs_arr[0])):
            t.failed = True
            t.msg = "Cost must decrease after gradient descent iterations"
            t.want = f"costs[-1] < costs[0]"
            t.got = f"costs[0]={costs_arr[0]:.4f}, costs[-1]={costs_arr[-1]:.4f}"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
