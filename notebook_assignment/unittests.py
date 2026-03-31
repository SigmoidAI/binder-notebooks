from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def _make_data(n_clusters=4):
    X, y = make_blobs(n_samples=300, n_features=4, centers=n_clusters, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "apply_kmeans"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()

        try:
            result = learner_func(X, n_clusters=4, random_state=42)
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

        required_keys = {"labels", "centers", "inertia", "silhouette"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        try:
            s = float(result.get("silhouette", -99))
            if not (-1.0 <= s <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "silhouette score must be a float in [-1.0, 1.0]"
            t.want = "float in [-1.0, 1.0]"
            t.got = result.get("silhouette")
        cases.append(t)

        t = test_case()
        labels = result.get("labels")
        if not isinstance(labels, np.ndarray) or labels.shape != (X.shape[0],):
            t.failed = True
            t.msg = "labels must be a numpy array of shape (n_samples,)"
            t.want = (X.shape[0],)
            t.got = getattr(labels, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "find_optimal_k"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()
        max_k = 10

        try:
            result = learner_func(X, max_k=max_k)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, (int, np.integer)):
            t.failed = True
            t.msg = "optimal_k must be an integer"
            t.want = "int"
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if not (2 <= int(result) <= max_k):
            t.failed = True
            t.msg = f"optimal_k must be in range [2, {max_k}]"
            t.want = f"int in [2, {max_k}]"
            t.got = result
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "apply_gmm"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()

        try:
            result = learner_func(X, n_components=4, random_state=42)
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

        required_keys = {"labels", "probabilities", "gmm", "bic", "aic", "silhouette"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        if not isinstance(result.get("gmm"), GaussianMixture):
            t.failed = True
            t.msg = "result['gmm'] must be a GaussianMixture"
            t.want = GaussianMixture
            t.got = type(result.get("gmm"))
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "apply_hierarchical"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()

        try:
            result = learner_func(X, n_clusters=4, linkage_method="ward")
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

        required_keys = {"labels", "silhouette"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "compare_linkage_methods"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()

        try:
            result = learner_func(X, n_clusters=4)
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
            t.msg = "Result must be a dict mapping linkage method to results"
            t.want = dict
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if len(result) < 2:
            t.failed = True
            t.msg = "Must compare at least 2 linkage methods"
            t.want = ">= 2 methods"
            t.got = len(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "evaluate_clustering"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y_true = _make_data()
        km = KMeans(n_clusters=4, random_state=42, n_init="auto")
        labels = km.fit_predict(X)

        try:
            result = learner_func(X, labels, y_true=y_true)
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

        required_keys = {"silhouette", "davies_bouldin", "calinski_harabasz"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must contain keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)

def exercise_7(learner_func):
    def g():
        cases = []
        func_name = "initialize_centroids"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()
        k = 4

        try:
            result = learner_func(X, k=k, random_state=42)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, np.ndarray) or result.shape != (k, X.shape[1]):
            t.failed = True
            t.msg = "centroids must be a numpy array of shape (k, n_features)"
            t.want = (k, X.shape[1])
            t.got = getattr(result, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_8(learner_func):
    def g():
        cases = []
        func_name = "assign_clusters"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()
        k = 4
        rng = np.random.default_rng(42)
        centroids = X[rng.choice(len(X), k, replace=False)]

        try:
            result = learner_func(X, centroids)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, np.ndarray) or result.shape != (X.shape[0],):
            t.failed = True
            t.msg = "labels must be a numpy array of shape (n_samples,)"
            t.want = (X.shape[0],)
            t.got = getattr(result, "shape", None)
        cases.append(t)

        t = test_case()
        if not (np.all(result >= 0) and np.all(result < k)):
            t.failed = True
            t.msg = "All label values must be in [0, k-1]"
            t.want = f"values in [0, {k-1}]"
            t.got = f"min={result.min()}, max={result.max()}"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_9(learner_func):
    def g():
        cases = []
        func_name = "update_centroids"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()
        k = 4
        rng = np.random.default_rng(42)
        labels = rng.integers(0, k, X.shape[0])

        try:
            result = learner_func(X, labels, k=k)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, np.ndarray) or result.shape != (k, X.shape[1]):
            t.failed = True
            t.msg = "centroids must be a numpy array of shape (k, n_features)"
            t.want = (k, X.shape[1])
            t.got = getattr(result, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_10(learner_func):
    def g():
        cases = []
        func_name = "kmeans_from_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()

        try:
            result = learner_func(X, k=4, max_iters=50, random_state=42)
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

        required_keys = {"labels", "centroids", "inertia", "n_iters"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_11(learner_func):
    def g():
        cases = []
        func_name = "gmm_from_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, _ = _make_data()

        try:
            result = learner_func(X, k=4, max_iters=30, random_state=42)
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

        required_keys = {"labels", "means", "covariances", "weights", "responsibilities", "n_iters"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)

