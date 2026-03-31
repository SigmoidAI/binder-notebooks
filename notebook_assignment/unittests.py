"""Unit tests for Imputation Impact Assessment exercises."""

from types import FunctionType
from typing import List


class test_case:
    """Simple test case class for feedback."""
    def __init__(self):
        self.failed = False
        self.msg = ""
        self.want = ""
        self.got = ""


def print_feedback(cases: List[test_case]):
    """Print colored feedback for test cases."""
    failed_cases = [c for c in cases if c.failed]
    if not failed_cases:
        print("\033[92m All tests passed!\033[0m")
    else:
        print(f"\033[91mFailed {len(failed_cases)} of {len(cases)} tests:\033[0m\n")
        for case in failed_cases:
            print(f"  \u274c {case.msg}")
            print(f"     Expected: {case.want}")
            print(f"     Got: {case.got}\n")


def exercise_1(learner_func):
    """Test Exercise 1: Prepare baseline model."""
    from unittests_utils import BaselineModelBattery
    
    def g():
        cases: List[test_case] = []
        func_name = "prepare_baseline_model"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        try:
            battery = BaselineModelBattery(learner_func)

            t = test_case()
            got, want, failed = battery.returns_dict()
            if failed:
                t.failed = True
                t.msg = "Function must return a dictionary"
                t.want = "A dictionary with 'rmse', 'mae', 'r2' keys"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_rmse()
            if failed:
                t.failed = True
                t.msg = "Result dictionary must contain 'rmse' key"
                t.want = "Key 'rmse' in result"
                t.got = "Key 'rmse' not found"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_mae()
            if failed:
                t.failed = True
                t.msg = "Result dictionary must contain 'mae' key"
                t.want = "Key 'mae' in result"
                t.got = "Key 'mae' not found"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_r2()
            if failed:
                t.failed = True
                t.msg = "Result dictionary must contain 'r2' key"
                t.want = "Key 'r2' in result"
                t.got = "Key 'r2' not found"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.rmse_reasonable()
            if failed:
                t.failed = True
                t.msg = "RMSE value seems unreasonable (too high)"
                t.want = "RMSE < 100"
                t.got = f"RMSE = {battery.result.get('rmse', 'N/A')}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.r2_positive()
            if failed:
                t.failed = True
                t.msg = "R2 score should be positive for a good model"
                t.want = "R2 > 0"
                t.got = f"R2 = {battery.result.get('r2', 'N/A')}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing {func_name}"
            t.want = "Function to execute without exceptions"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    """Test Exercise 2: Evaluate deletion impact."""
    from unittests_utils import DeletionImpactBattery
    
    def g():
        cases: List[test_case] = []
        func_name = "apply_listwise_deletion"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        try:
            battery = DeletionImpactBattery(learner_func)

            t = test_case()
            got, want, failed = battery.returns_dataframe()
            if failed:
                t.failed = True
                t.msg = "Function must return a DataFrame as first element"
                t.want = "pd.DataFrame"
                t.got = f"Got type: {type(battery.X_clean)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.no_missing_values()
            if failed:
                t.failed = True
                t.msg = "Returned DataFrame should have no missing values"
                t.want = "No NaN values"
                t.got = "DataFrame still contains missing values"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.deleted_count_positive()
            if failed:
                t.failed = True
                t.msg = "Function should return the count of deleted rows"
                t.want = "Positive integer for deleted_count"
                t.got = f"Got: {battery.deleted_count}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.y_length_matches()
            if failed:
                t.failed = True
                t.msg = "X and y should have the same length after deletion"
                t.want = "len(X_clean) == len(y_clean)"
                t.got = "Lengths do not match"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.samples_reduced()
            if failed:
                t.failed = True
                t.msg = "Sample count should be reduced after deletion"
                t.want = "Fewer samples after deletion"
                t.got = "Sample count not reduced"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing {func_name}"
            t.want = "Function to execute without exceptions"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    """Test Exercise 3: Evaluate simple imputation impact."""
    from unittests_utils import SimpleImputationBattery
    
    def g():
        cases: List[test_case] = []
        func_name = "apply_simple_imputation"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        try:
            battery = SimpleImputationBattery(learner_func)

            t = test_case()
            got, want, failed = battery.returns_dataframe()
            if failed:
                t.failed = True
                t.msg = "Function must return a DataFrame as first element"
                t.want = "pd.DataFrame"
                t.got = f"Got type: {type(battery.X_imputed)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.no_missing_values()
            if failed:
                t.failed = True
                t.msg = "Imputed DataFrame should have no missing values"
                t.want = "No NaN values"
                t.got = "DataFrame still contains missing values"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.same_shape()
            if failed:
                t.failed = True
                t.msg = "Imputed DataFrame should have same shape as training input"
                t.want = f"Shape: {battery.X_train.shape}"
                t.got = f"Shape: {battery.X_imputed.shape if battery.X_imputed is not None else 'N/A'}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.columns_preserved()
            if failed:
                t.failed = True
                t.msg = "Column names should be preserved after imputation"
                t.want = "Same column names"
                t.got = "Column names differ"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing {func_name}"
            t.want = "Function to execute without exceptions"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    """Test Exercise 4: Evaluate advanced imputation impact."""
    from unittests_utils import AdvancedImputationBattery
    
    def g():
        cases: List[test_case] = []
        func_name = "apply_advanced_imputation"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        try:
            battery = AdvancedImputationBattery(learner_func)

            t = test_case()
            got, want, failed = battery.knn_returns_dataframe()
            if failed:
                t.failed = True
                t.msg = "KNN imputation must return a DataFrame"
                t.want = "pd.DataFrame"
                t.got = f"Got type: {type(battery.X_knn)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.iterative_returns_dataframe()
            if failed:
                t.failed = True
                t.msg = "Iterative imputation must return a DataFrame"
                t.want = "pd.DataFrame"
                t.got = f"Got type: {type(battery.X_iterative)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.knn_no_missing()
            if failed:
                t.failed = True
                t.msg = "KNN imputed DataFrame should have no missing values"
                t.want = "No NaN values"
                t.got = "DataFrame still contains missing values"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.iterative_no_missing()
            if failed:
                t.failed = True
                t.msg = "Iterative imputed DataFrame should have no missing values"
                t.want = "No NaN values"
                t.got = "DataFrame still contains missing values"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.knn_same_shape()
            if failed:
                t.failed = True
                t.msg = "KNN imputed DataFrame should have same shape as training input"
                t.want = f"Shape: {battery.X_train.shape}"
                t.got = f"Shape: {battery.X_knn.shape if battery.X_knn is not None else 'N/A'}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.iterative_same_shape()
            if failed:
                t.failed = True
                t.msg = "Iterative imputed DataFrame should have same shape as training input"
                t.want = f"Shape: {battery.X_train.shape}"
                t.got = f"Shape: {battery.X_iterative.shape if battery.X_iterative is not None else 'N/A'}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing {func_name}"
            t.want = "Function to execute without exceptions"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(learner_func):
    """Test Exercise 5: Create comprehensive impact report."""
    from unittests_utils import ImpactReportBattery
    
    def g():
        cases: List[test_case] = []
        func_name = "create_impact_report"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        try:
            battery = ImpactReportBattery(learner_func)

            t = test_case()
            got, want, failed = battery.returns_dataframe()
            if failed:
                t.failed = True
                t.msg = "Function must return a pd.DataFrame"
                t.want = "pd.DataFrame"
                t.got = f"Got type: {type(battery.report)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_strategy_col()
            if failed:
                t.failed = True
                t.msg = "Report DataFrame must contain 'strategy' column"
                t.want = "Column 'strategy'"
                t.got = f"Columns: {list(battery.report.columns) if battery.report is not None else 'N/A'}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_rmse_col()
            if failed:
                t.failed = True
                t.msg = "Report DataFrame must contain 'rmse' column"
                t.want = "Column 'rmse'"
                t.got = f"Columns: {list(battery.report.columns) if battery.report is not None else 'N/A'}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_r2_col()
            if failed:
                t.failed = True
                t.msg = "Report DataFrame must contain 'r2' column"
                t.want = "Column 'r2'"
                t.got = f"Columns: {list(battery.report.columns) if battery.report is not None else 'N/A'}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_n_samples_col()
            if failed:
                t.failed = True
                t.msg = "Report DataFrame must contain 'n_samples' column"
                t.want = "Column 'n_samples'"
                t.got = f"Columns: {list(battery.report.columns) if battery.report is not None else 'N/A'}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_multiple_rows()
            if failed:
                t.failed = True
                t.msg = "Report must have at least 3 strategy rows"
                t.want = ">= 3 rows"
                t.got = f"Got {len(battery.report)} rows" if battery.report is not None else "N/A"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when testing {func_name}"
            t.want = "Function to execute without exceptions"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)
