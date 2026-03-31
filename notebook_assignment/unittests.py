from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _make_numeric_data():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _make_mixed_data():
    """Small synthetic dataset with both numeric and categorical columns."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "age": rng.uniform(20, 70, n),
        "income": rng.uniform(20000, 100000, n),
        "gender": rng.choice(["M", "F"], n),
        "category": rng.choice(["A", "B", "C"], n),
    })
    y = pd.Series((df["income"] > 60000).astype(int), name="label")
    return train_test_split(df, y, test_size=0.2, random_state=42)


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "create_basic_pipeline"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_numeric_data()

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

        required_keys = {"pipeline", "train_score", "test_score"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        if not isinstance(result.get("pipeline"), Pipeline):
            t.failed = True
            t.msg = "result['pipeline'] must be a sklearn Pipeline"
            t.want = Pipeline
            t.got = type(result.get("pipeline"))
        cases.append(t)

        t = test_case()
        try:
            score = float(result.get("test_score", -1))
            if not (0.0 <= score <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "test_score must be a float in [0.0, 1.0]"
            t.want = "float in [0.0, 1.0]"
            t.got = result.get("test_score")
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "create_column_transformer_pipeline"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_mixed_data()

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

        required_keys = {"pipeline", "train_score", "test_score", "numeric_features", "categorical_features"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        if not isinstance(result.get("pipeline"), Pipeline):
            t.failed = True
            t.msg = "result['pipeline'] must be a sklearn Pipeline"
            t.want = Pipeline
            t.got = type(result.get("pipeline"))
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "identify_and_fix_leakage"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        try:
            result = learner_func(X, y, test_size=0.2)
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

        required_keys = {"leaky_score", "correct_score", "score_difference"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        for key in ("leaky_score", "correct_score"):
            try:
                score = float(result.get(key, -1))
                if not (0.0 <= score <= 1.0):
                    raise ValueError
            except (TypeError, ValueError):
                t.failed = True
                t.msg = f"{key} must be a float in [0.0, 1.0]"
                t.want = "float in [0.0, 1.0]"
                t.got = result.get(key)
                break
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "tune_pipeline_hyperparameters"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_numeric_data()

        try:
            result = learner_func(X_train, X_test, y_train, y_test, cv_folds=2)
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

        required_keys = {"best_pipeline", "best_params", "best_cv_score", "test_score"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        if not isinstance(result.get("best_pipeline"), Pipeline):
            t.failed = True
            t.msg = "result['best_pipeline'] must be a sklearn Pipeline"
            t.want = Pipeline
            t.got = type(result.get("best_pipeline"))
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "create_production_pipeline"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_numeric_data()

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

        required_keys = {"pipeline", "train_score", "test_score", "serialization_valid"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        if not result.get("serialization_valid", False):
            t.failed = True
            t.msg = "serialization_valid must be True"
            t.want = True
            t.got = result.get("serialization_valid")
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_class):
    def g():
        cases = []
        class_name = "SimplePipeline"

        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} must be a class (type)"
            t.want = type
            t.got = type(learner_class)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_numeric_data()

        try:
            pipeline = learner_class(steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ])
            pipeline.fit(X_train, y_train)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception during {class_name}.fit()"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        try:
            predictions = pipeline.predict(X_test)
            if not isinstance(predictions, np.ndarray) or predictions.shape != (X_test.shape[0],):
                t.failed = True
                t.msg = "predict() must return array of shape (n_samples,)"
                t.want = (X_test.shape[0],)
                t.got = getattr(predictions, "shape", None)
        except Exception as e:
            t.failed = True
            t.msg = f"Exception during predict(): {e}"
            t.want = "np.ndarray"
            t.got = str(e)
        cases.append(t)

        t = test_case()
        try:
            score = pipeline.score(X_test, y_test)
            if not (0.0 <= float(score) <= 1.0):
                t.failed = True
                t.msg = "score() must return a float in [0.0, 1.0]"
                t.want = "float in [0.0, 1.0]"
                t.got = score
        except Exception as e:
            t.failed = True
            t.msg = f"Exception during score(): {e}"
            t.want = "float"
            t.got = str(e)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
