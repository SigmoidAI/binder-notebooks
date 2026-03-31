"""Unit tests for the Production-Ready Imputation Pipeline assignment."""
from types import FunctionType
from typing import List

from unittests_utils import (
    ConfigurableImputerBattery,
    EdgeCaseHandlerBattery,
    SklearnTransformerBattery,
    FitTransformPatternBattery,
    ImputationLoggerBattery,
    CompletePipelineBattery,
)


class test_case:
    """Simple test case class for tracking test results."""
    def __init__(self):
        self.failed = False
        self.msg = ""
        self.want = ""
        self.got = ""


def print_feedback(cases: List[test_case]):
    """Print feedback for all test cases."""
    failed_cases = [c for c in cases if c.failed]
    
    if not failed_cases:
        print("\033[92m" + "All tests passed!" + "\033[0m")
    else:
        print("\033[91m" + f"Failed {len(failed_cases)} of {len(cases)} tests:" + "\033[0m")
        for case in failed_cases:
            print(f"\n  - {case.msg}")
            if case.want:
                print(f"     Expected: {case.want}")
            if case.got:
                print(f"     Got: {case.got}")


def exercise_1(learner_class):
    """Test the ConfigurableImputer class."""
    def g():
        cases: List[test_case] = []
        
        t = test_case()
        if not callable(learner_class):
            t.failed = True
            t.msg = "ConfigurableImputer must be a class"
            t.want = "a callable class"
            t.got = type(learner_class)
            return [t]
        cases.append(t)
        
        try:
            battery = ConfigurableImputerBattery(learner_class)
            
            t = test_case()
            got, want, failed = battery.is_class_check()
            if failed:
                t.failed = True
                t.msg = "ConfigurableImputer must be instantiable"
                t.want = "Class that can be instantiated"
                t.got = f"Error during instantiation"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.has_strategy_check()
            if failed:
                t.failed = True
                t.msg = "ConfigurableImputer must have a 'strategy' attribute"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.has_fit_check()
            if failed:
                t.failed = True
                t.msg = "ConfigurableImputer must have a 'fit' method"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.has_transform_check()
            if failed:
                t.failed = True
                t.msg = "ConfigurableImputer must have a 'transform' method"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.has_statistics_check()
            if failed:
                t.failed = True
                t.msg = "ConfigurableImputer must store 'statistics_' after fitting"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.no_missing_after_transform()
            if failed:
                t.failed = True
                t.msg = "Transform should produce output with no missing values"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    """Test the handle_edge_cases function."""
    def g():
        cases: List[test_case] = []
        
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "handle_edge_cases must be a function"
            return [t]
        cases.append(t)
        
        try:
            battery = EdgeCaseHandlerBattery(learner_func)
            
            t = test_case()
            got, want, failed = battery.returns_dict_check()
            if failed:
                t.failed = True
                t.msg = "Function must return a dictionary"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.has_is_valid_key()
            if failed:
                t.failed = True
                t.msg = "Return dict must have 'is_valid' key"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.has_issue_type_key()
            if failed:
                t.failed = True
                t.msg = "Return dict must have 'issue_type' key"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.detects_all_missing()
            if failed:
                t.failed = True
                t.msg = "Function should detect columns with all missing values"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.detects_constant()
            if failed:
                t.failed = True
                t.msg = "Function should detect constant columns"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.normal_is_valid()
            if failed:
                t.failed = True
                t.msg = "Normal columns should be marked as valid"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)


def exercise_3(learner_class):
    """Test the sklearn-compatible transformer."""
    def g():
        cases: List[test_case] = []
        
        t = test_case()
        if not callable(learner_class):
            t.failed = True
            t.msg = "Transformer must be a class"
            return [t]
        cases.append(t)
        
        try:
            battery = SklearnTransformerBattery(learner_class)
            
            t = test_case()
            got, want, failed = battery.inherits_base_estimator()
            if failed:
                t.failed = True
                t.msg = "Transformer must inherit from BaseEstimator"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.inherits_transformer_mixin()
            if failed:
                t.failed = True
                t.msg = "Transformer must inherit from TransformerMixin"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.has_fit_method()
            if failed:
                t.failed = True
                t.msg = "Transformer must have a 'fit' method"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.has_transform_method()
            if failed:
                t.failed = True
                t.msg = "Transformer must have a 'transform' method"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.train_no_missing()
            if failed:
                t.failed = True
                t.msg = "Transformed training data should have no missing values"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.test_no_missing()
            if failed:
                t.failed = True
                t.msg = "Transformed test data should have no missing values"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)


def exercise_4(learner_class):
    """Test the fit/transform pattern implementation."""
    def g():
        cases: List[test_case] = []
        
        t = test_case()
        if not callable(learner_class):
            t.failed = True
            t.msg = "Imputer must be a class"
            return [t]
        cases.append(t)
        
        try:
            battery = FitTransformPatternBattery(learner_class)
            
            t = test_case()
            got, want, failed = battery.uses_training_statistics()
            if failed:
                t.failed = True
                t.msg = "Test data must be imputed using training statistics (not test statistics)"
                t.want = "Imputed value based on training mean"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.statistics_stored()
            if failed:
                t.failed = True
                t.msg = "Imputer must store statistics_ after fitting"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.correct_train_imputation()
            if failed:
                t.failed = True
                t.msg = "Training data should be imputed with training statistics"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)


def exercise_5(learner_class):
    """Test the imputation logger."""
    def g():
        cases: List[test_case] = []
        
        t = test_case()
        if not callable(learner_class):
            t.failed = True
            t.msg = "ImputationLogger must be a class"
            return [t]
        cases.append(t)
        
        try:
            battery = ImputationLoggerBattery(learner_class)
            
            t = test_case()
            got, want, failed = battery.has_imputation_log()
            if failed:
                t.failed = True
                t.msg = "ImputationLogger must have 'imputation_log_' attribute after transform"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.log_is_dataframe_or_dict()
            if failed:
                t.failed = True
                t.msg = "imputation_log_ should be a DataFrame, dict, or list"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.log_tracks_positions()
            if failed:
                t.failed = True
                t.msg = "Log should track positions (row/column) of imputed values"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.log_has_correct_count()
            if failed:
                t.failed = True
                t.msg = "Log should contain entries for all imputed values (5 total)"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)


def exercise_6(learner_func):
    """Test the complete preprocessing pipeline."""
    def g():
        cases: List[test_case] = []
        
        t = test_case()
        if not callable(learner_func):
            t.failed = True
            t.msg = "create_preprocessing_pipeline must be a callable"
            return [t]
        cases.append(t)
        
        try:
            battery = CompletePipelineBattery(learner_func)
            
            t = test_case()
            got, want, failed = battery.is_sklearn_pipeline()
            if failed:
                t.failed = True
                t.msg = "Function must return an sklearn Pipeline object"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.has_multiple_steps()
            if failed:
                t.failed = True
                t.msg = "Pipeline should have at least 2 steps (imputation + other preprocessing)"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.train_no_missing_numeric()
            if failed:
                t.failed = True
                t.msg = "Transformed training data should have no missing numeric values"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.test_no_missing_numeric()
            if failed:
                t.failed = True
                t.msg = "Transformed test data should have no missing numeric values"
            cases.append(t)
            
            t = test_case()
            got, want, failed = battery.output_shape_correct()
            if failed:
                t.failed = True
                t.msg = "Output should have same number of rows as input"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised: {e}"
            return cases + [t]
        
        return cases
    
    cases = g()
    print_feedback(cases)
