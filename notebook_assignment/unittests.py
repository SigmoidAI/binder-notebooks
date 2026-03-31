from types import FunctionType

import numpy as np
from dlai_grader.grading import print_feedback, test_case
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _make_imbalanced(weights=(0.90, 0.10), n_samples=1000, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        weights=list(weights),
        flip_y=0.01,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "diagnose_imbalance"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, _, y_train, _ = _make_imbalanced()

        try:
            result = learner_func(y_train)
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

        required_keys = {"class_counts", "imbalance_ratio", "majority_class", "minority_class"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result dict must contain keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        if not isinstance(result.get("class_counts"), dict):
            t.failed = True
            t.msg = "class_counts must be a dict"
            t.want = dict
            t.got = type(result.get("class_counts"))
        cases.append(t)

        t = test_case()
        ratio = result.get("imbalance_ratio")
        try:
            ratio_val = float(ratio)
            if not (ratio_val > 1.0):
                raise ValueError
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "imbalance_ratio must be a float > 1.0 for an imbalanced dataset"
            t.want = "float > 1.0"
            t.got = ratio
        cases.append(t)

        t = test_case()
        classes = np.unique(y_train)
        counts = np.array([np.sum(y_train == c) for c in classes])
        expected_majority = classes[np.argmax(counts)]
        if result.get("majority_class") != expected_majority:
            t.failed = True
            t.msg = "majority_class is incorrect"
            t.want = expected_majority
            t.got = result.get("majority_class")
        cases.append(t)

        t = test_case()
        expected_minority = classes[np.argmin(counts)]
        if result.get("minority_class") != expected_minority:
            t.failed = True
            t.msg = "minority_class is incorrect"
            t.want = expected_minority
            t.got = result.get("minority_class")
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "train_with_class_weights"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_imbalanced()

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

        required_keys = {"f1_none", "f1_balanced", "model_none", "model_balanced"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result dict must contain keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        for key in ("f1_none", "f1_balanced"):
            t = test_case()
            try:
                val = float(result.get(key, None))
                if not (0.0 <= val <= 1.0):
                    raise ValueError
            except (TypeError, ValueError):
                t.failed = True
                t.msg = f"{key} must be a float in [0.0, 1.0]"
                t.want = "float in [0.0, 1.0]"
                t.got = result.get(key)
            cases.append(t)

        t = test_case()
        f1_none = float(result.get("f1_none", -1))
        f1_bal = float(result.get("f1_balanced", -1))
        if f1_bal < f1_none:
            t.failed = True
            t.msg = "f1_balanced should be >= f1_none (balanced weights should help on imbalanced data)"
            t.want = f"f1_balanced >= f1_none ({f1_none:.3f})"
            t.got = f"f1_balanced = {f1_bal:.3f}"
        cases.append(t)

        for key in ("model_none", "model_balanced"):
            t = test_case()
            model = result.get(key)
            if not hasattr(model, "predict"):
                t.failed = True
                t.msg = f"{key} must be a fitted sklearn model with a predict method"
                t.want = "fitted model with predict()"
                t.got = type(model)
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "apply_smote"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, _, y_train, _ = _make_imbalanced()

        try:
            result = learner_func(X_train, y_train, k_neighbors=5, random_state=42)
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
            t.msg = "Must return a tuple of (X_resampled, y_resampled)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        X_res, y_res = result

        t = test_case()
        if not isinstance(X_res, np.ndarray):
            t.failed = True
            t.msg = "X_resampled must be a numpy array"
            t.want = np.ndarray
            t.got = type(X_res)
        cases.append(t)

        t = test_case()
        if not isinstance(y_res, np.ndarray):
            t.failed = True
            t.msg = "y_resampled must be a numpy array"
            t.want = np.ndarray
            t.got = type(y_res)
        cases.append(t)

        classes_orig, counts_orig = np.unique(y_train, return_counts=True)
        classes_res, counts_res = np.unique(y_res, return_counts=True)

        t = test_case()
        if len(X_res) <= len(X_train):
            t.failed = True
            t.msg = "SMOTE must produce more samples than the original training set"
            t.want = f"> {len(X_train)} samples"
            t.got = len(X_res)
        cases.append(t)

        t = test_case()
        if len(set(counts_res)) > 1:
            # counts_res should be equal (balanced) after SMOTE
            t.failed = True
            t.msg = "After SMOTE, class counts should be equal (balanced)"
            t.want = "equal counts per class"
            t.got = dict(zip(classes_res, counts_res))
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "apply_undersampling"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, _, y_train, _ = _make_imbalanced()

        for strategy in ("random", "tomek", "nearmiss"):
            try:
                result = learner_func(X_train, y_train, strategy=strategy, random_state=42)
            except Exception as e:
                t = test_case()
                t.failed = True
                t.msg = f"Exception raised with strategy='{strategy}'"
                t.want = "No exception"
                t.got = str(e)
                cases.append(t)
                continue

            t = test_case()
            if not isinstance(result, tuple) or len(result) != 2:
                t.failed = True
                t.msg = f"Strategy '{strategy}': must return tuple of (X_resampled, y_resampled)"
                t.want = "tuple of length 2"
                t.got = type(result)
            cases.append(t)

            X_res, y_res = result

            t = test_case()
            if not isinstance(X_res, np.ndarray):
                t.failed = True
                t.msg = f"Strategy '{strategy}': X_resampled must be a numpy array"
                t.want = np.ndarray
                t.got = type(X_res)
            cases.append(t)

            t = test_case()
            if len(X_res) >= len(X_train) and strategy in ("random", "nearmiss"):
                t.failed = True
                t.msg = f"Strategy '{strategy}': undersampling should reduce the dataset size"
                t.want = f"< {len(X_train)} samples"
                t.got = len(X_res)
            cases.append(t)

        # Test that ValueError is raised for unknown strategy
        t = test_case()
        try:
            learner_func(X_train, y_train, strategy="unknown_strategy")
            t.failed = True
            t.msg = "Should raise ValueError for unknown strategy"
            t.want = "ValueError"
            t.got = "No exception raised"
        except ValueError:
            pass  # expected
        except Exception as e:
            t.failed = True
            t.msg = "Should raise ValueError (not another exception) for unknown strategy"
            t.want = "ValueError"
            t.got = type(e).__name__
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "evaluate_imbalanced_classifier"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, X_test, y_train, y_test = _make_imbalanced()
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        try:
            result = learner_func(y_test, y_pred, y_prob)
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

        required_keys = {"precision", "recall", "f1", "roc_auc", "pr_auc"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result dict must contain keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        for key in required_keys:
            t = test_case()
            try:
                val = float(result.get(key, None))
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


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "smote_from_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        X_train, _, y_train, _ = _make_imbalanced()

        try:
            result = learner_func(X_train, y_train, k_neighbors=5, random_state=42)
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
            t.msg = "Must return a tuple of (X_resampled, y_resampled)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        X_res, y_res = result

        t = test_case()
        if not isinstance(X_res, np.ndarray):
            t.failed = True
            t.msg = "X_resampled must be a numpy array"
            t.want = np.ndarray
            t.got = type(X_res)
        cases.append(t)

        t = test_case()
        if X_res.shape[1] != X_train.shape[1]:
            t.failed = True
            t.msg = "X_resampled must have the same number of features as X_train"
            t.want = X_train.shape[1]
            t.got = X_res.shape[1]
        cases.append(t)

        classes_res, counts_res = np.unique(y_res, return_counts=True)

        t = test_case()
        if len(X_res) <= len(X_train):
            t.failed = True
            t.msg = "smote_from_scratch must add synthetic samples (result should be larger than input)"
            t.want = f"> {len(X_train)} total samples"
            t.got = len(X_res)
        cases.append(t)

        t = test_case()
        if len(set(counts_res)) > 1:
            t.failed = True
            t.msg = "After from-scratch SMOTE, class counts should be equal (balanced)"
            t.want = "equal counts per class"
            t.got = dict(zip(classes_res, counts_res))
        cases.append(t)

        # Verify synthetic samples are in valid feature range (interpolation should stay within bounds)
        t = test_case()
        min_orig = X_train.min(axis=0)
        max_orig = X_train.max(axis=0)
        synthetic = X_res[len(X_train):]
        if np.any(synthetic < min_orig - 1e-6) or np.any(synthetic > max_orig + 1e-6):
            t.failed = True
            t.msg = "Synthetic samples should lie within the range of the original training data (they are interpolations)"
            t.want = "samples within original feature range"
            t.got = "samples outside range"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
