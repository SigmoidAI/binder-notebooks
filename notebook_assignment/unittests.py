from types import FunctionType

import numpy as np
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler


def _make_data():
    X, y = make_classification(n_samples=300, n_features=10, n_informative=5, random_state=42)
    return X, y


def _make_precomputed_stats():
    """Returns means, variances, priors from a fixed dataset."""
    X, y = _make_data()
    classes = np.unique(y)
    means = {c: X[y == c].mean(axis=0) for c in classes}
    variances = {c: X[y == c].var(axis=0) + 1e-9 for c in classes}
    priors = {c: np.mean(y == c) for c in classes}
    return X, y, means, variances, priors


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "calculate_statistics"

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
            t.msg = "Must return a tuple of length 3: (means, variances, priors)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        means, variances, priors = result

        for name, val in [("means", means), ("variances", variances), ("priors", priors)]:
            t = test_case()
            if not isinstance(val, dict):
                t.failed = True
                t.msg = f"{name} must be a dict"
                t.want = dict
                t.got = type(val)
            cases.append(t)

        t = test_case()
        prior_sum = sum(float(v) for v in priors.values())
        if not np.isclose(prior_sum, 1.0, atol=1e-5):
            t.failed = True
            t.msg = "Prior probabilities must sum to 1.0"
            t.want = 1.0
            t.got = prior_sum
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "gaussian_likelihood"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # Test: likelihood of x == mean should be maximum
        mean = 5.0
        variance = 2.0
        x_at_mean = 5.0
        x_away = 100.0

        try:
            like_center = learner_func(x_at_mean, mean, variance)
            like_away = learner_func(x_away, mean, variance)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        try:
            lc = float(like_center)
            if lc < 0:
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "Likelihood must be a non-negative float"
            t.want = ">= 0"
            t.got = like_center
        cases.append(t)

        t = test_case()
        try:
            if not (float(like_center) > float(like_away)):
                t.failed = True
                t.msg = "Likelihood at x=mean must be greater than at x far from mean"
                t.want = "like_center > like_away"
                t.got = f"like_center={like_center}, like_away={like_away}"
        except (TypeError, ValueError):
            pass
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "predict_with_probabilities"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y, means, variances, priors = _make_precomputed_stats()

        try:
            result = learner_func(X, means, variances, priors)
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
            t.msg = "Must return a tuple of length 2: (predictions, probabilities)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        predictions, probabilities = result

        t = test_case()
        if not isinstance(predictions, np.ndarray) or predictions.shape != (X.shape[0],):
            t.failed = True
            t.msg = "predictions must be a numpy array of shape (n_samples,)"
            t.want = (X.shape[0],)
            t.got = getattr(predictions, "shape", None)
        cases.append(t)

        t = test_case()
        if not isinstance(probabilities, (list, np.ndarray)) or len(probabilities) != X.shape[0]:
            t.failed = True
            t.msg = "probabilities must have length n_samples"
            t.want = X.shape[0]
            t.got = len(probabilities) if hasattr(probabilities, "__len__") else "N/A"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "analyze_predictions"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y, means, variances, priors = _make_precomputed_stats()
        classes = np.unique(y)
        predictions = np.array([classes[0]] * len(y))
        probabilities = [{c: 1.0 / len(classes) for c in classes} for _ in range(len(y))]

        try:
            result = learner_func(y, predictions, probabilities, top_n=5)
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
        required_keys = {"avg_confidence", "most_confident"}
        missing = required_keys - set(result.keys())
        if missing:
            t.failed = True
            t.msg = f"Result dict missing keys: {missing}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "calibrate_probabilities"

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
        y_pred = y_true.copy()
        probabilities = rng.uniform(0.4, 0.9, 100)

        try:
            result = learner_func(y_true, y_pred, probabilities)
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
            t.msg = "Must return a tuple of length 2: (calibrated_probs, calibration_model)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        calibrated_probs, calibration_model = result

        t = test_case()
        if not isinstance(calibrated_probs, np.ndarray) or calibrated_probs.shape != (100,):
            t.failed = True
            t.msg = "calibrated_probs must be a numpy array of length 100"
            t.want = (100,)
            t.got = getattr(calibrated_probs, "shape", None)
        cases.append(t)

        t = test_case()
        if calibration_model is None:
            t.failed = True
            t.msg = "calibration_model must not be None"
            t.want = "fitted calibration model"
            t.got = None
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_class):
    def g():
        cases = []
        class_name = "MultinomialNB"

        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} must be a class"
            t.want = type
            t.got = type(learner_class)
            return [t]
        cases.append(t)

        X, y = _make_data()
        scaler = MinMaxScaler()
        X_nonneg = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_nonneg, y, test_size=0.3, random_state=42)

        try:
            obj = learner_class(alpha=1.0)
            obj.fit(X_train, y_train)
            predictions = obj.predict(X_test)
            proba = obj.predict_proba(X_test)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when using {class_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(predictions, np.ndarray) or predictions.shape != (X_test.shape[0],):
            t.failed = True
            t.msg = "predict() must return array of shape (n_test_samples,)"
            t.want = (X_test.shape[0],)
            t.got = getattr(predictions, "shape", None)
        cases.append(t)

        t = test_case()
        if not isinstance(proba, np.ndarray) or proba.shape[0] != X_test.shape[0]:
            t.failed = True
            t.msg = "predict_proba() must return array with n_test_samples rows"
            t.want = f"array with {X_test.shape[0]} rows"
            t.got = getattr(proba, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_7(learner_func):
    def g():
        cases = []
        func_name = "compare_classifiers"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        try:
            result = learner_func(X_train, X_test, y_train, y_test)
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
        for key, val in result.items():
            try:
                v = float(val)
                if not (0.0 <= v <= 1.0):
                    t.failed = True
                    t.msg = f"Accuracy for '{key}' must be in [0.0, 1.0]"
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
        func_name = "build_spam_classifier"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = _make_data()
        scaler = MinMaxScaler()
        X_nn = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_nn, y, test_size=0.3, random_state=42)

        try:
            result = learner_func(X_train, X_test, y_train, y_test)
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
            t.msg = "Result must be a dict with classifier results"
            t.want = dict
            t.got = type(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
