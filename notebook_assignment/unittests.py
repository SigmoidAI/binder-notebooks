from types import FunctionType

import numpy as np
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _make_binary():
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def _make_split():
    X, y = _make_binary()
    return train_test_split(X, y, test_size=0.3, random_state=42)


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "train_linear_svm"

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
        if not isinstance(result, SVC):
            t.failed = True
            t.msg = "Result must be an sklearn SVC"
            t.want = SVC
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if not hasattr(result, "support_vectors_"):
            t.failed = True
            t.msg = "SVC must be fitted (support_vectors_ attribute missing)"
            t.want = "fitted SVC"
            t.got = "unfitted"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "compare_scaled_unscaled_svm"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_split()

        try:
            result = learner_func(X_train, X_test, y_train, y_test, C=1.0, gamma=0.1)
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

        required_keys = {"unscaled_acc", "scaled_acc", "scaler"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result dict must contain keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "tune_rbf_svm"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_binary()
        C_values = [0.1, 1.0, 10.0]
        gamma_values = [0.01, 0.1]

        try:
            result = learner_func(X, y, C_values=C_values, gamma_values=gamma_values, test_size=0.3)
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
            t.msg = "Must return a tuple of length 3: (best_model, best_params, best_score)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        best_model, best_params, best_score = result

        t = test_case()
        if not hasattr(best_model, "predict"):
            t.failed = True
            t.msg = "best_model must have a predict method"
            t.want = "object with predict()"
            t.got = type(best_model)
        cases.append(t)

        t = test_case()
        if not isinstance(best_params, dict):
            t.failed = True
            t.msg = "best_params must be a dict"
            t.want = dict
            t.got = type(best_params)
        cases.append(t)

        t = test_case()
        try:
            s = float(best_score)
            if not (0.0 <= s <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "best_score must be a float in [0.0, 1.0]"
            t.want = "float in [0.0, 1.0]"
            t.got = best_score
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "compare_kernels"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_binary()
        datasets = {
            "set_a": (X[:150], y[:150]),
            "set_b": (X[150:], y[150:]),
        }

        try:
            result = learner_func(datasets)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        import pandas as pd
        t = test_case()
        if not isinstance(result, pd.DataFrame):
            t.failed = True
            t.msg = "Result must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        t = test_case()
        if result.shape[1] < 2:
            t.failed = True
            t.msg = "DataFrame must have at least 2 kernel columns"
            t.want = ">= 2 columns"
            t.got = result.shape[1]
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "train_multiclass_svm_strategies"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = make_classification(
            n_samples=300, n_features=10, n_classes=3, n_informative=5, random_state=42
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        try:
            result = learner_func(X_train, y_train, X_test, y_test)
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
        if len(result) < 2:
            t.failed = True
            t.msg = "Must include at least 2 strategies (OvR and OvO)"
            t.want = ">= 2 keys"
            t.got = len(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6a(learner_func):
    def g():
        cases = []
        func_name = "hinge_loss"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        rng = np.random.default_rng(42)
        y_true = np.array([1, -1, 1, -1, 1], dtype=float)
        y_pred = rng.standard_normal(5)

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
            loss = float(result)
            if loss < 0:
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "hinge_loss must return a non-negative float"
            t.want = ">= 0"
            t.got = result
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6b(learner_func):
    def g():
        cases = []
        func_name = "svm_primal_gradients"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        y = rng.choice([-1, 1], 50).astype(float)
        w = rng.standard_normal(5)
        b = float(rng.standard_normal())

        try:
            result = learner_func(X, y, w, b, C=1.0)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not (isinstance(result, tuple) and len(result) == 2):
            t.failed = True
            t.msg = "Must return a tuple (grad_w, grad_b)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        grad_w, grad_b = result
        t = test_case()
        if not isinstance(grad_w, np.ndarray) or grad_w.shape != w.shape:
            t.failed = True
            t.msg = "grad_w must be a numpy array of same shape as w"
            t.want = w.shape
            t.got = getattr(grad_w, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6c(learner_func):
    def g():
        cases = []
        func_name = "train_linear_svm_from_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = rng.choice([-1, 1], 100).astype(float)
        iterations = 100

        try:
            result = learner_func(X, y, C=1.0, lr=0.001, num_iters=iterations)
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
            t.msg = "Must return a tuple of length 3: (w, b, history)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        w, b, history = result

        t = test_case()
        if not isinstance(w, np.ndarray) or w.shape != (5,):
            t.failed = True
            t.msg = "w must be a numpy array of shape (n_features,)"
            t.want = (5,)
            t.got = getattr(w, "shape", None)
        cases.append(t)

        t = test_case()
        if not isinstance(history, (list, np.ndarray)) or len(history) != iterations:
            t.failed = True
            t.msg = f"history must have length {iterations}"
            t.want = iterations
            t.got = len(history) if hasattr(history, "__len__") else "N/A"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
