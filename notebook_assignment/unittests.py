from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def _make_data():
    X, y = make_classification(n_samples=400, n_features=15, n_informative=8, random_state=42)
    return X, y


def _make_split():
    X, y = _make_data()
    return train_test_split(X, y, test_size=0.25, random_state=42)


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "train_decision_tree"

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
            result = learner_func(X, y, criterion="gini", max_depth=5, min_samples_split=2)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, DecisionTreeClassifier):
            t.failed = True
            t.msg = "Result must be a DecisionTreeClassifier"
            t.want = DecisionTreeClassifier
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if not hasattr(result, "tree_"):
            t.failed = True
            t.msg = "DecisionTree must be fitted"
            t.want = "fitted tree with tree_ attribute"
            t.got = "unfitted"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "compare_tree_depths"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_split()
        max_depths = [1, 3, 5, 10]

        try:
            result = learner_func(X_train, y_train, X_test, y_test, max_depths=max_depths)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, pd.DataFrame):
            t.failed = True
            t.msg = "Result must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        t = test_case()
        if len(result) != len(max_depths):
            t.failed = True
            t.msg = f"DataFrame must have {len(max_depths)} rows (one per depth)"
            t.want = len(max_depths)
            t.got = len(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "train_random_forest"

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
            result = learner_func(
                X_train, y_train, X_test, y_test,
                n_estimators=50, max_features="sqrt", max_depth=5, min_samples_split=2
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

        required_keys = {"model", "test_acc", "feature_importances"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must contain keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        if not isinstance(result.get("model"), RandomForestClassifier):
            t.failed = True
            t.msg = "result['model'] must be a RandomForestClassifier"
            t.want = RandomForestClassifier
            t.got = type(result.get("model"))
        cases.append(t)

        t = test_case()
        try:
            acc = float(result.get("test_acc", -1))
            if not (0.0 <= acc <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "test_acc must be a float in [0.0, 1.0]"
            t.want = "float in [0.0, 1.0]"
            t.got = result.get("test_acc")
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "train_gradient_boosting"

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
            result = learner_func(
                X_train, y_train, X_test, y_test,
                n_estimators=50, learning_rate=0.1, max_depth=3, subsample=0.8
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
        cases.append(t)

        t = test_case()
        if len(result) < 1:
            t.failed = True
            t.msg = "Result dict must not be empty"
            t.want = "dict with at least one model"
            t.got = {}
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "compare_ensemble_methods"

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
            result = learner_func(X_train, y_train, X_test, y_test)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, pd.DataFrame):
            t.failed = True
            t.msg = "Result must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        t = test_case()
        if len(result) < 2:
            t.failed = True
            t.msg = "Must compare at least 2 ensemble methods"
            t.want = ">= 2 rows"
            t.got = len(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6a(learner_func):
    def g():
        cases = []
        func_name = "calculate_gini"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        pure = np.array([1, 1, 1, 1])
        mixed = np.array([0, 0, 1, 1])

        try:
            gini_pure = learner_func(pure)
            gini_mixed = learner_func(mixed)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        try:
            if not np.isclose(float(gini_pure), 0.0, atol=1e-6):
                t.failed = True
                t.msg = "Gini of a pure node [1,1,1,1] must be 0.0"
                t.want = 0.0
                t.got = gini_pure
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "Gini must be a numeric value"
            t.want = "float"
            t.got = type(gini_pure)
        cases.append(t)

        t = test_case()
        try:
            if not np.isclose(float(gini_mixed), 0.5, atol=1e-6):
                t.failed = True
                t.msg = "Gini of [0,0,1,1] must be 0.5"
                t.want = 0.5
                t.got = gini_mixed
        except (TypeError, ValueError):
            pass
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6b(learner_func):
    def g():
        cases = []
        func_name = "calculate_entropy"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        pure = np.array([1, 1, 1, 1])
        mixed = np.array([0, 1, 0, 1])

        try:
            ent_pure = learner_func(pure)
            ent_mixed = learner_func(mixed)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        try:
            if not np.isclose(float(ent_pure), 0.0, atol=1e-6):
                t.failed = True
                t.msg = "Entropy of a pure node must be 0.0"
                t.want = 0.0
                t.got = ent_pure
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "Entropy must be a numeric value"
            t.want = "float >= 0"
            t.got = type(ent_pure)
        cases.append(t)

        t = test_case()
        try:
            if float(ent_pure) > float(ent_mixed):
                t.failed = True
                t.msg = "Entropy should be higher for mixed class distributions"
                t.want = "ent_mixed > ent_pure"
                t.got = f"ent_pure={ent_pure}, ent_mixed={ent_mixed}"
        except (TypeError, ValueError):
            pass
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6c(learner_func):
    def g():
        cases = []
        func_name = "find_best_split"

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
        y = rng.integers(0, 2, 100)

        try:
            result = learner_func(X, y, criterion='gini')
        except Exception as e:
            t = test_case()            
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not (isinstance(result, tuple) and len(result) == 3):
            t.failed = True
            t.msg = "Must return a tuple of (best_feature, best_threshold, best_gain)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        best_feature, best_threshold, best_gain = result

        t = test_case()
        try:
            gain = float(best_gain)
            if gain < 0:
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "best_gain must be a non-negative float"
            t.want = ">= 0"
            t.got = best_gain
        cases.append(t)

        # Verify correctness on a linearly separable dataset
        X_simple = np.array([[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]])
        y_simple = np.array([0, 0, 0, 1, 1, 1])
        feat, thr, gain = learner_func(X_simple, y_simple, criterion='gini')

        t = test_case()
        if feat != 0:
            t.failed = True
            t.msg = "Should find feature 0 as best split for linearly separable data"
            t.want = 0
            t.got = feat
        cases.append(t)

        t = test_case()
        if thr is None or not (2.5 < thr < 7.5):
            t.failed = True
            t.msg = "Threshold should separate the two groups"
            t.want = "between 3.0 and 7.0"
            t.got = thr
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6d(learner_func):
    def g():
        cases = []
        func_name = "build_decision_tree_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = rng.integers(0, 2, 80)

        try:
            result = learner_func(X, y, max_depth=3, min_samples_split=5)
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
            t.msg = "Tree must be represented as a dict"
            t.want = dict
            t.got = type(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_7(learner_func):
    def g():
        cases = []
        func_name = "random_forest_from_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, _ = _make_split()

        try:
            result = learner_func(X_train, y_train, n_estimators=10, max_depth=3)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, list):
            t.failed = True
            t.msg = "Must return a list of decision trees"
            t.want = list
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        t = test_case()
        if len(result) != 10:
            t.failed = True
            t.msg = "Must return exactly n_estimators trees"
            t.want = 10
            t.got = len(result)
        cases.append(t)

        t = test_case()
        if not all(isinstance(tree, dict) for tree in result):
            t.failed = True
            t.msg = "Each tree in the forest must be a dict"
            t.want = "list of dicts"
            t.got = type(result[0]) if result else "empty list"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
