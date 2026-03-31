from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _make_diabetes_data():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _make_breast_cancer_data():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "interpret_linear_model"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_diabetes_data()

        try:
            result = learner_func(X_train, y_train, X_test, y_test, alpha=1.0)
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
            t.msg = "Result must be a tuple (coef_df, r2)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        coef_df, r2 = result

        t = test_case()
        if not isinstance(coef_df, pd.DataFrame):
            t.failed = True
            t.msg = "coef_df must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(coef_df)
        cases.append(t)

        required_cols = {"feature", "coefficient"}
        t = test_case()
        if not required_cols.issubset(set(coef_df.columns)):
            t.failed = True
            t.msg = f"coef_df must have columns: {required_cols}"
            t.want = required_cols
            t.got = set(coef_df.columns)
        cases.append(t)

        t = test_case()
        try:
            if not (-1.0 <= float(r2) <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "r2 must be a float in [-1.0, 1.0]"
            t.want = "float in [-1.0, 1.0]"
            t.got = r2
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "interpret_tree_model"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_breast_cancer_data()

        try:
            result = learner_func(X_train, y_train, X_test, y_test, max_depth=4)
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
            t.msg = "Result must be a tuple (importance_df, text_rules, accuracy)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        importance_df, text_rules, accuracy = result

        t = test_case()
        if not isinstance(importance_df, pd.DataFrame):
            t.failed = True
            t.msg = "importance_df must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(importance_df)
        cases.append(t)

        required_cols = {"feature", "importance"}
        t = test_case()
        if not required_cols.issubset(set(importance_df.columns)):
            t.failed = True
            t.msg = f"importance_df must have columns: {required_cols}"
            t.want = required_cols
            t.got = set(importance_df.columns)
        cases.append(t)

        t = test_case()
        try:
            if not (0.0 <= float(accuracy) <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "accuracy must be a float in [0.0, 1.0]"
            t.want = "float in [0.0, 1.0]"
            t.got = accuracy
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "compute_permutation_importance"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_breast_cancer_data()

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
        if not isinstance(result, tuple) or len(result) != 2:
            t.failed = True
            t.msg = "Result must be a tuple (perm_df, accuracy)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        perm_df, accuracy = result

        t = test_case()
        if not isinstance(perm_df, pd.DataFrame):
            t.failed = True
            t.msg = "perm_df must be a pandas DataFrame"
            t.want = pd.DataFrame
            t.got = type(perm_df)
        cases.append(t)

        required_cols = {"feature", "importance_mean", "importance_std"}
        t = test_case()
        if not required_cols.issubset(set(perm_df.columns)):
            t.failed = True
            t.msg = f"perm_df must have columns: {required_cols}"
            t.want = required_cols
            t.got = set(perm_df.columns)
        cases.append(t)

        t = test_case()
        try:
            if not (0.0 <= float(accuracy) <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "accuracy must be a float in [0.0, 1.0]"
            t.want = "float in [0.0, 1.0]"
            t.got = accuracy
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "compute_shap_values"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_breast_cancer_data()

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
        if not isinstance(result, tuple) or len(result) != 3:
            t.failed = True
            t.msg = "Result must be a tuple (explainer, shap_values, accuracy)"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        _explainer, _shap_values, accuracy = result

        t = test_case()
        try:
            if not (0.0 <= float(accuracy) <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "accuracy must be a float in [0.0, 1.0]"
            t.want = "float in [0.0, 1.0]"
            t.got = accuracy
        cases.append(t)

        # shap/explainer may be None if shap not installed — acceptable
        t = test_case()
        if _shap_values is not None:
            n_test = X_test.shape[0]
            n_feats = X_test.shape[1]
            vals = np.array(_shap_values)
            if vals.ndim < 2:
                t.failed = True
                t.msg = "shap_values must have at least 2 dimensions (n_samples, n_features)"
                t.want = "(n_samples, n_features)"
                t.got = vals.shape
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "compute_lime_explanation"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_breast_cancer_data()
        feature_names = list(X_train.columns)
        class_names = ["malignant", "benign"]

        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)

        try:
            result = learner_func(
                rf, X_train, X_test,
                feature_names=feature_names,
                class_names=class_names,
                instance_idx=0,
            )
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
            t.msg = "Result must be a tuple (explanation_list, lime_explainer)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        explanation_list, lime_explainer = result

        # Both may be None/empty if lime not installed — acceptable
        t = test_case()
        if explanation_list is not None and len(explanation_list) > 0:
            first = explanation_list[0]
            if not (isinstance(first, tuple) and len(first) == 2):
                t.failed = True
                t.msg = "explanation_list items must be (feature_name, weight) tuples"
                t.want = "(str, float) tuples"
                t.got = first
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "permutation_importance_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_breast_cancer_data()
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)

        def scoring_fn(model, X, y):
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, model.predict(X))

        try:
            result = learner_func(rf, X_test, y_test, scoring_fn=scoring_fn, random_state=42)
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

        required_cols = {"feature", "importance"}
        t = test_case()
        if not required_cols.issubset(set(result.columns)):
            t.failed = True
            t.msg = f"Result must have columns: {required_cols}"
            t.want = required_cols
            t.got = set(result.columns)
        cases.append(t)

        t = test_case()
        if len(result) != X_test.shape[1]:
            t.failed = True
            t.msg = "Must compute importance for every feature"
            t.want = X_test.shape[1]
            t.got = len(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
