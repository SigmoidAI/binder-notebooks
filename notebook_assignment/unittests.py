from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _make_data():
    X, y = make_classification(
        n_samples=200, n_features=15, n_informative=10, random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "apply_pca"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()
        n_components = 5

        try:
            result = learner_func(X, n_components)
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
            t.msg = "Must return a tuple of length 2: (pca, X_pca)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        pca, X_pca = result

        t = test_case()
        if not isinstance(pca, PCA):
            t.failed = True
            t.msg = "First element must be a fitted sklearn PCA object"
            t.want = PCA
            t.got = type(pca)
        cases.append(t)

        t = test_case()
        if not hasattr(pca, "explained_variance_ratio_"):
            t.failed = True
            t.msg = "PCA object must be fitted (call pca.fit or pca.fit_transform)"
            t.want = "fitted PCA with explained_variance_ratio_"
            t.got = "unfitted PCA"
        cases.append(t)

        t = test_case()
        if not isinstance(X_pca, np.ndarray):
            t.failed = True
            t.msg = "X_pca must be a numpy array"
            t.want = np.ndarray
            t.got = type(X_pca)
        cases.append(t)

        t = test_case()
        if X_pca.shape != (X.shape[0], n_components):
            t.failed = True
            t.msg = f"X_pca shape is incorrect"
            t.want = (X.shape[0], n_components)
            t.got = X_pca.shape
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "plot_explained_variance"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()
        pca = PCA(n_components=10, random_state=42)
        pca.fit(X)

        try:
            learner_func(pca, threshold=0.95)
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
        func_name = "select_n_components"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()
        pca = PCA(n_components=15, random_state=42)
        pca.fit(X)

        try:
            result = learner_func(pca, variance_threshold=0.90)
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
            t.msg = "Must return a tuple of length 2: (n_components, actual_variance)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        n_components, actual_variance = result

        t = test_case()
        if not isinstance(n_components, (int, np.integer)):
            t.failed = True
            t.msg = "n_components must be an integer"
            t.want = "int"
            t.got = type(n_components)
        cases.append(t)

        t = test_case()
        if not (0.0 <= float(actual_variance) <= 1.0):
            t.failed = True
            t.msg = "actual_variance must be in [0.0, 1.0]"
            t.want = "float in [0.0, 1.0]"
            t.got = actual_variance
        cases.append(t)

        t = test_case()
        if float(actual_variance) < 0.90:
            t.failed = True
            t.msg = "actual_variance must be >= variance_threshold=0.90"
            t.want = ">= 0.90"
            t.got = actual_variance
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "visualize_pca_2d"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_data()
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)

        try:
            import matplotlib
            matplotlib.use("Agg")
            learner_func(X_2d, y, title="Test PCA 2D")
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
        func_name = "visualize_pca_3d"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_data()
        pca = PCA(n_components=3, random_state=42)
        X_3d = pca.fit_transform(X)

        try:
            import matplotlib
            matplotlib.use("Agg")
            learner_func(X_3d, y, title="Test PCA 3D")
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


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "select_k_best_features"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_data()
        k = 5

        try:
            result = learner_func(X, y, k)
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
        if not isinstance(selector, SelectKBest):
            t.failed = True
            t.msg = "selector must be a SelectKBest instance"
            t.want = SelectKBest
            t.got = type(selector)
        cases.append(t)

        t = test_case()
        if not isinstance(X_selected, np.ndarray):
            t.failed = True
            t.msg = "X_selected must be a numpy array"
            t.want = np.ndarray
            t.got = type(X_selected)
        cases.append(t)

        t = test_case()
        if X_selected.shape != (X.shape[0], k):
            t.failed = True
            t.msg = "X_selected shape is incorrect"
            t.want = (X.shape[0], k)
            t.got = X_selected.shape
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_7(learner_func):
    def g():
        cases = []
        func_name = "compare_dimensionality_reduction"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_data()

        try:
            result = learner_func(X, y, n_components=5)
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
        if len(result) == 0:
            t.failed = True
            t.msg = "Result dict must not be empty"
            t.want = "dict with accuracy scores"
            t.got = {}
        cases.append(t)

        t = test_case()
        for key, val in result.items():
            try:
                v = float(val)
                if not (0.0 <= v <= 1.0):
                    t.failed = True
                    t.msg = f"Accuracy value for '{key}' must be in [0.0, 1.0]"
                    t.want = "float in [0.0, 1.0]"
                    t.got = v
                    break
            except (TypeError, ValueError):
                pass
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_8(learner_func):
    def g():
        cases = []
        func_name = "create_pca_pipeline"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        try:
            result = learner_func(n_components=5)
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
            return cases + [t]
        cases.append(t)

        t = test_case()
        has_pca = any(isinstance(step, PCA) for _, step in result.steps)
        if not has_pca:
            t.failed = True
            t.msg = "Pipeline must contain a PCA step"
            t.want = "Pipeline with PCA"
            t.got = [type(s).__name__ for _, s in result.steps]
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_9(learner_func):
    def g():
        cases = []
        func_name = "pca_from_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()
        n_components = 5

        try:
            result = learner_func(X, n_components)
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
            t.msg = "Must return a tuple of length 3: (X_pca, components, explained_variance_ratio)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        X_pca, components, evr = result

        t = test_case()
        if not isinstance(X_pca, np.ndarray) or X_pca.shape != (X.shape[0], n_components):
            t.failed = True
            t.msg = "X_pca shape is incorrect"
            t.want = (X.shape[0], n_components)
            t.got = getattr(X_pca, "shape", None)
        cases.append(t)

        t = test_case()
        if not isinstance(components, np.ndarray) or components.shape != (n_components, X.shape[1]):
            t.failed = True
            t.msg = "components shape is incorrect"
            t.want = (n_components, X.shape[1])
            t.got = getattr(components, "shape", None)
        cases.append(t)

        t = test_case()
        evr_arr = np.array(evr)
        if not (len(evr_arr) == n_components and np.all(evr_arr >= 0) and np.sum(evr_arr) <= 1.001):
            t.failed = True
            t.msg = "explained_variance_ratio must have length n_components, all >= 0, sum <= 1"
            t.want = f"array of length {n_components} summing to <= 1"
            t.got = evr_arr
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_10(learner_func):
    def g():
        cases = []
        func_name = "pca_using_svd"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()
        n_components = 5

        try:
            result = learner_func(X, n_components)
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
            t.msg = "Must return a tuple of length 3: (X_pca, components, explained_variance_ratio)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        X_pca, components, evr = result

        t = test_case()
        if not isinstance(X_pca, np.ndarray) or X_pca.shape != (X.shape[0], n_components):
            t.failed = True
            t.msg = "X_pca shape is incorrect"
            t.want = (X.shape[0], n_components)
            t.got = getattr(X_pca, "shape", None)
        cases.append(t)

        t = test_case()
        if not isinstance(components, np.ndarray) or components.shape != (n_components, X.shape[1]):
            t.failed = True
            t.msg = "components shape is incorrect"
            t.want = (n_components, X.shape[1])
            t.got = getattr(components, "shape", None)
        cases.append(t)

        t = test_case()
        evr_arr = np.array(evr)
        if not (len(evr_arr) == n_components and np.all(evr_arr >= 0) and np.sum(evr_arr) <= 1.001):
            t.failed = True
            t.msg = "explained_variance_ratio must have length n_components, all >= 0, sum <= 1"
            t.want = f"array of length {n_components} summing to <= 1"
            t.got = evr_arr
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
