from types import FunctionType

import numpy as np
import pandas as pd
from dlai_grader.grading import print_feedback, test_case


def _make_series(n=365):
    rng = np.random.default_rng(42)
    t = np.arange(n)
    values = 0.5 * t + 10 * np.sin(2 * np.pi * t / 30) + rng.standard_normal(n) * 2
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(values, index=idx, name="test_series")


def _make_train_test():
    series = _make_series(300)
    split = int(len(series) * 0.8)
    return series.iloc[:split], series.iloc[split:]


def exercise_1(learner_func):
    def g():
        cases = []
        func_name = "create_synthetic_series"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        try:
            result = learner_func(n_periods=365, start_date="2020-01-01")
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, pd.Series):
            t.failed = True
            t.msg = "Result must be a pandas Series"
            t.want = pd.Series
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        t = test_case()
        if len(result) != 365:
            t.failed = True
            t.msg = "Series must have n_periods elements"
            t.want = 365
            t.got = len(result)
        cases.append(t)

        t = test_case()
        if not isinstance(result.index, pd.DatetimeIndex):
            t.failed = True
            t.msg = "Series index must be a DatetimeIndex"
            t.want = pd.DatetimeIndex
            t.got = type(result.index)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        func_name = "explore_time_series"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        series = _make_series()

        try:
            result = learner_func(series, window=30)
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

        required_keys = {"mean", "std", "min", "max"}
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


def exercise_3(learner_func):
    def g():
        cases = []
        func_name = "decompose_series"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        series = _make_series()

        try:
            result = learner_func(series, model="additive", period=30)
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

        required_keys = {"trend", "seasonal", "residual", "strength_trend", "strength_seasonal"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        try:
            st = float(result.get("strength_trend", -1))
            if not (0.0 <= st <= 1.0):
                t.failed = True
                t.msg = "strength_trend must be in [0.0, 1.0]"
                t.want = "float in [0.0, 1.0]"
                t.got = st
        except (TypeError, ValueError):
            t.failed = True
            t.msg = "strength_trend must be a float"
            t.want = "float in [0.0, 1.0]"
            t.got = type(result.get("strength_trend"))
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []
        func_name = "test_stationarity"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        series = _make_series()

        try:
            result = learner_func(series)
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

        required_keys = {"adf_stat", "p_value", "critical_values", "is_stationary"}
        t = test_case()
        if not required_keys.issubset(set(result.keys())):
            t.failed = True
            t.msg = f"Result must have keys: {required_keys}"
            t.want = required_keys
            t.got = set(result.keys())
        cases.append(t)

        t = test_case()
        if not isinstance(result.get("is_stationary"), bool):
            t.failed = True
            t.msg = "is_stationary must be a bool"
            t.want = bool
            t.got = type(result.get("is_stationary"))
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    def g():
        cases = []
        func_name = "make_stationary"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        series = _make_series()

        try:
            result = learner_func(series, method="diff", order=1)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, pd.Series):
            t.failed = True
            t.msg = "Result must be a pandas Series"
            t.want = pd.Series
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if len(result) < 1:
            t.failed = True
            t.msg = "Result must not be empty"
            t.want = "non-empty Series"
            t.got = "empty Series"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_func):
    def g():
        cases = []
        func_name = "analyze_autocorrelation"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        series = _make_series()

        try:
            result = learner_func(series, nlags=40)
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

        required_keys = {"acf_values", "pacf_values"}
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


def exercise_7(learner_func):
    def g():
        cases = []
        func_name = "forecast_moving_average"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        train, test = _make_train_test()

        try:
            result = learner_func(train, test, window=7)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, pd.Series):
            t.failed = True
            t.msg = "Result must be a pandas Series"
            t.want = pd.Series
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if len(result) != len(test):
            t.failed = True
            t.msg = "Forecast length must match test length"
            t.want = len(test)
            t.got = len(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_8(learner_func):
    def g():
        cases = []
        func_name = "forecast_exponential_smoothing"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        train, test = _make_train_test()

        try:
            result = learner_func(train, test, alpha=0.3)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, pd.Series):
            t.failed = True
            t.msg = "Result must be a pandas Series"
            t.want = pd.Series
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if len(result) != len(test):
            t.failed = True
            t.msg = "Forecast length must match test length"
            t.want = len(test)
            t.got = len(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_9(learner_func):
    def g():
        cases = []
        func_name = "temporal_train_test_split"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        series = _make_series()

        try:
            result = learner_func(series, train_ratio=0.8)
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
            t.msg = "Result must be a tuple of (train, test)"
            t.want = "tuple of length 2"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        train, test = result
        t = test_case()
        if not isinstance(train, pd.Series) or not isinstance(test, pd.Series):
            t.failed = True
            t.msg = "Both train and test must be pandas Series"
            t.want = pd.Series
            t.got = (type(train), type(test))
        cases.append(t)

        t = test_case()
        if len(train) + len(test) != len(series):
            t.failed = True
            t.msg = "train + test lengths must equal original series length"
            t.want = len(series)
            t.got = len(train) + len(test)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_10(learner_func):
    def g():
        cases = []
        func_name = "time_series_cv"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        series = _make_series()

        try:
            result = learner_func(series, n_splits=5, test_size=30)
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
            t.msg = "Result must be a list of (train, test) tuples"
            t.want = list
            t.got = type(result)
        cases.append(t)

        t = test_case()
        if len(result) != 5:
            t.failed = True
            t.msg = "Number of splits must equal n_splits"
            t.want = 5
            t.got = len(result)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_11(learner_func):
    def g():
        cases = []
        func_name = "simple_moving_average_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        series = _make_series(50)

        try:
            result = learner_func(series, window=7)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, np.ndarray) or result.shape != (len(series),):
            t.failed = True
            t.msg = "Result must be a numpy array of same length as series"
            t.want = (len(series),)
            t.got = getattr(result, "shape", None)
        cases.append(t)

        t = test_case()
        if not np.isnan(result[0]):
            t.failed = True
            t.msg = "First window-1 values must be NaN"
            t.want = "NaN"
            t.got = result[0]
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)



def exercise_12(learner_func):
    def g():
        cases = []
        func_name = "weighted_moving_average_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        series = _make_series(50)

        try:
            result = learner_func(series, window=7)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, np.ndarray) or result.shape != (len(series),):
            t.failed = True
            t.msg = "Result must be a numpy array of same length as series"
            t.want = (len(series),)
            t.got = getattr(result, "shape", None)
        cases.append(t)

        t = test_case()
        if not np.isnan(result[0]):
            t.failed = True
            t.msg = "First window-1 values must be NaN"
            t.want = "NaN"
            t.got = result[0]
        cases.append(t)

        t = test_case()
        # Verify weighted MA differs from simple MA (it should weight recent values more)
        sma_last = float(np.mean(np.array(series)[-7:]))
        wma_val = float(result[-1]) if not np.isnan(result[-1]) else None
        if wma_val is None:
            t.failed = True
            t.msg = "Last value of weighted MA must not be NaN"
            t.want = "float"
            t.got = "NaN"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)

def exercise_13(learner_func):
    def g():
        cases = []
        func_name = "exponential_smoothing_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        series = _make_series(50)

        try:
            result = learner_func(series, alpha=0.3)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, np.ndarray) or result.shape != (len(series),):
            t.failed = True
            t.msg = "Result must be a numpy array of same length as series"
            t.want = (len(series),)
            t.got = getattr(result, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_14(learner_func):
    def g():
        cases = []
        func_name = "forecast_with_exponential_smoothing_scratch"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        series = _make_series(100)
        steps = 10

        try:
            result = learner_func(series, alpha=0.3, steps=steps)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised when calling {func_name}"
            t.want = "No exception"
            t.got = str(e)
            return cases + [t]

        t = test_case()
        if not isinstance(result, np.ndarray) or result.shape != (steps,):
            t.failed = True
            t.msg = "Result must be numpy array of shape (steps,)"
            t.want = (steps,)
            t.got = getattr(result, "shape", None)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
