from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _make_binary_preds():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return y_test, y_pred, y_proba


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "analyze_confusion_matrix"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true, y_pred, _ = _make_binary_preds()

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

        required_keys = {"confusion_matrix", "tn", "fp", "fn", "tp", "tpr", "tnr", "fpr", "fnr"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        cm = result.get("confusion_matrix")
        if not isinstance(cm, np.ndarray) or cm.shape != (2, 2):
            t.failed = True
            t.msg = "confusion_matrix must be a (2, 2) ndarray"
            t.want = (2, 2)
            t.got = getattr(cm, "shape", None)
        cases.append(t)

        for key in ("tpr", "tnr", "fpr", "fnr"):
            t = test_case()
            try:
                val = float(result.get(key, -1))
                if not (0.0 <= val <= 1.0):
                    raise ValueError
            except (TypeError, ValueError):
                t.failed = True
                t.msg = f"{key} must be a float in [0.0, 1.0]"
                t.want = "float in [0.0, 1.0]"
                t.got = result.get(key)
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "calculate_classification_metrics"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true, y_pred, y_proba = _make_binary_preds()

        try:
            result = learner_func(y_true, y_pred, y_pred_proba=y_proba)
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
            "precision_manual", "recall_manual", "f1_manual",
            "precision_sklearn", "recall_sklearn", "f1_sklearn",
        }
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        for key in ("precision_manual", "precision_sklearn"):
            t = test_case()
            try:
                val = float(result.get(key, -1))
                if not (0.0 <= val <= 1.0):
                    raise ValueError
            except (TypeError, ValueError):
                t.failed = True
                t.msg = f"{key} must be a float in [0.0, 1.0]"
                t.want = "float in [0.0, 1.0]"
                t.got = result.get(key)
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "compare_roc_pr_curves"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_s, y_train)

        try:
            import matplotlib
            matplotlib.use("Agg")
            result = learner_func(X_train_s, X_test_s, y_train, y_test, model1=clf)
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
        if "model1" not in result:
            t.failed = True
            t.msg = "Result must have key 'model1'"
            t.want = "'model1' in result"
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        sub = result.get("model1", {})
        required_sub = {"fpr", "tpr", "auc", "precision", "recall", "ap"}
        if not required_sub.issubset(set(sub.keys())):
            t.failed = True
            t.msg = f"result['model1'] must have keys: {required_sub}"
            t.want = required_sub
            t.got = set(sub.keys())
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "optimize_threshold"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true, _, y_proba = _make_binary_preds()

        try:
            result = learner_func(y_true, y_proba, cost_fp=1, cost_fn=2)
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

        required_keys = {"optimal_threshold", "optimal_metrics", "all_metrics"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        try:
            thr = float(result.get("optimal_threshold", -1))
            if not (0.0 <= thr <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "optimal_threshold must be a float in [0.0, 1.0]"
            t.want = "float in [0.0, 1.0]"
            t.got = result.get("optimal_threshold")
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "evaluate_multiclass"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        try:
            result = learner_func(model_class=LogisticRegression)
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

        required_keys = {"confusion_matrix", "metrics", "classification_report"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        cm = result.get("confusion_matrix")
        if not isinstance(cm, np.ndarray) or cm.ndim != 2:
            t.failed = True
            t.msg = "confusion_matrix must be a 2D ndarray"
            t.want = "2D ndarray"
            t.got = getattr(cm, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6a(learner_func):
    def g():
        cases = []
        func_name = "confusion_matrix_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0])

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
        if not isinstance(result, np.ndarray) or result.shape != (2, 2):
            t.failed = True
            t.msg = "Result must be a (2, 2) ndarray"
            t.want = (2, 2)
            t.got = getattr(result, "shape", None)
        cases.append(t)

        t = test_case()
        # [[TN,FP],[FN,TP]] → TN=2,FP=1,FN=1,TP=2
        expected = np.array([[2, 1], [1, 2]])
        if not np.array_equal(result, expected):
            t.failed = True
            t.msg = "Confusion matrix values are incorrect"
            t.want = expected.tolist()
            t.got = result.tolist()
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6b(learner_func):
    def g():
        cases = []
        func_name = "accuracy_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

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
            if not np.isclose(float(result), 0.8, atol=1e-6):
                t.failed = True
                t.msg = "Accuracy must be 0.8 for this input"
                t.want = 0.8
                t.got = result
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "accuracy must be a numeric scalar"
            t.want = "float"
            t.got = type(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6c(learner_func):
    def g():
        cases = []
        func_name = "precision_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])

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
            if not np.isclose(float(result), 2.0 / 3.0, atol=1e-6):
                t.failed = True
                t.msg = "Precision must be ~0.667 for this input"
                t.want = round(2 / 3, 4)
                t.got = result
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "precision must be a numeric scalar"
            t.want = "float"
            t.got = type(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6d(learner_func):
    def g():
        cases = []
        func_name = "recall_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])

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
            if not np.isclose(float(result), 2.0 / 3.0, atol=1e-6):
                t.failed = True
                t.msg = "Recall must be ~0.667 for this input"
                t.want = round(2 / 3, 4)
                t.got = result
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "recall must be a numeric scalar"
            t.want = "float"
            t.got = type(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6e(learner_func):
    def g():
        cases = []
        func_name = "f1_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])

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
                t.msg = "F1 of perfect predictions must be 1.0"
                t.want = 1.0
                t.got = result
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "f1 must be a numeric scalar"
            t.want = "float"
            t.got = type(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
