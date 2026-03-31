"""Unit test utilities for the Production-Ready Imputation Pipeline assignment."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TestBattery:
    """Base class for test batteries."""
    
    def __init__(self, learner_object):
        self.learner_object = learner_object
        self._get_reference_inputs()
        self.extract_info()
        self.get_reference_checks()

    def _get_reference_inputs(self):
        pass

    def extract_info(self):
        pass

    def get_reference_checks(self):
        pass

    def _check(self, check_name, got):
        want = self.reference_checks[check_name]
        if isinstance(want, np.ndarray):
            if got is None:
                condition = True
            else:
                condition = not np.allclose(got, want, rtol=1e-5, atol=1e-8)
        elif isinstance(want, float):
            if got is None:
                condition = True
            else:
                condition = not np.isclose(got, want, rtol=1e-5, atol=1e-8)
        else:
            condition = got != want
        return got, want, condition


class ConfigurableImputerBattery(TestBattery):
    """Test battery for Exercise 1: ConfigurableImputer class."""
    
    def _get_reference_inputs(self):
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0],
            'B': [np.nan, 2.0, 3.0, np.nan, 5.0],
            'C': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        self.numeric_cols = ['A', 'B', 'C']

    def extract_info(self):
        try:
            self.instance = self.learner_object(strategy='mean')
            self.is_class = True
            self.has_strategy = hasattr(self.instance, 'strategy')
            self.has_fit = hasattr(self.instance, 'fit')
            self.has_transform = hasattr(self.instance, 'transform')
            
            if self.has_fit and self.has_transform:
                self.instance.fit(self.test_data[self.numeric_cols])
                self.result = self.instance.transform(self.test_data[self.numeric_cols])
                self.has_statistics = hasattr(self.instance, 'statistics_')
            else:
                self.result = None
                self.has_statistics = False
        except Exception as e:
            self.is_class = False
            self.has_strategy = False
            self.has_fit = False
            self.has_transform = False
            self.has_statistics = False
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            "mean_A": 3.0,
            "mean_B": 10.0 / 3.0,
            "mean_C": 3.0,
        }

    def is_class_check(self):
        return self.is_class, True, not self.is_class

    def has_strategy_check(self):
        return self.has_strategy, True, not self.has_strategy

    def has_fit_check(self):
        return self.has_fit, True, not self.has_fit

    def has_transform_check(self):
        return self.has_transform, True, not self.has_transform

    def has_statistics_check(self):
        return self.has_statistics, True, not self.has_statistics

    def no_missing_after_transform(self):
        if self.result is None:
            return None, "No missing values", True
        if isinstance(self.result, pd.DataFrame):
            has_missing = self.result.isnull().any().any()
        else:
            has_missing = np.isnan(self.result).any()
        return not has_missing, True, has_missing


class EdgeCaseHandlerBattery(TestBattery):
    """Test battery for Exercise 2: handle_edge_cases function."""
    
    def _get_reference_inputs(self):
        self.all_missing_col = pd.Series([np.nan, np.nan, np.nan, np.nan])
        self.constant_col = pd.Series([5.0, 5.0, 5.0, 5.0])
        self.normal_col = pd.Series([1.0, np.nan, 3.0, 4.0])

    def extract_info(self):
        try:
            self.result_all_missing = self.learner_object(self.all_missing_col)
            self.result_constant = self.learner_object(self.constant_col)
            self.result_normal = self.learner_object(self.normal_col)
        except Exception as e:
            self.result_all_missing = None
            self.result_constant = None
            self.result_normal = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {}

    def returns_dict_check(self):
        is_dict = isinstance(self.result_all_missing, dict)
        return is_dict, True, not is_dict

    def has_is_valid_key(self):
        if not isinstance(self.result_all_missing, dict):
            return False, True, True
        has_key = 'is_valid' in self.result_all_missing
        return has_key, True, not has_key

    def has_issue_type_key(self):
        if not isinstance(self.result_all_missing, dict):
            return False, True, True
        has_key = 'issue_type' in self.result_all_missing
        return has_key, True, not has_key

    def detects_all_missing(self):
        if not isinstance(self.result_all_missing, dict):
            return None, "all_missing detected", True
        is_valid = self.result_all_missing.get('is_valid', True)
        issue = self.result_all_missing.get('issue_type', '')
        detected = not is_valid and 'all_missing' in str(issue).lower()
        return detected, True, not detected

    def detects_constant(self):
        if not isinstance(self.result_constant, dict):
            return None, "constant detected", True
        is_valid = self.result_constant.get('is_valid', True)
        issue = self.result_constant.get('issue_type', '')
        detected = not is_valid and 'constant' in str(issue).lower()
        return detected, True, not detected

    def normal_is_valid(self):
        if not isinstance(self.result_normal, dict):
            return None, True, True
        is_valid = self.result_normal.get('is_valid', False)
        return is_valid, True, not is_valid


class SklearnTransformerBattery(TestBattery):
    """Test battery for Exercise 3: sklearn-compatible transformer."""
    
    def _get_reference_inputs(self):
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0],
            'B': [np.nan, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0]
        })
        self.X_test = pd.DataFrame({
            'A': [np.nan, 2.0, 3.0],
            'B': [1.0, np.nan, 3.0]
        })

    def extract_info(self):
        try:
            self.instance = self.learner_object()
            self.is_base_estimator = isinstance(self.instance, BaseEstimator)
            self.is_transformer_mixin = isinstance(self.instance, TransformerMixin)
            self.has_fit = hasattr(self.instance, 'fit')
            self.has_transform = hasattr(self.instance, 'transform')
            self.has_fit_transform = hasattr(self.instance, 'fit_transform')
            
            if self.has_fit and self.has_transform:
                self.instance.fit(self.X_train)
                self.train_result = self.instance.transform(self.X_train)
                self.test_result = self.instance.transform(self.X_test)
            else:
                self.train_result = None
                self.test_result = None
        except Exception as e:
            self.is_base_estimator = False
            self.is_transformer_mixin = False
            self.has_fit = False
            self.has_transform = False
            self.has_fit_transform = False
            self.train_result = None
            self.test_result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {}

    def inherits_base_estimator(self):
        return self.is_base_estimator, True, not self.is_base_estimator

    def inherits_transformer_mixin(self):
        return self.is_transformer_mixin, True, not self.is_transformer_mixin

    def has_fit_method(self):
        return self.has_fit, True, not self.has_fit

    def has_transform_method(self):
        return self.has_transform, True, not self.has_transform

    def train_no_missing(self):
        if self.train_result is None:
            return None, "No missing", True
        if isinstance(self.train_result, pd.DataFrame):
            has_missing = self.train_result.isnull().any().any()
        else:
            has_missing = np.isnan(self.train_result).any()
        return not has_missing, True, has_missing

    def test_no_missing(self):
        if self.test_result is None:
            return None, "No missing", True
        if isinstance(self.test_result, pd.DataFrame):
            has_missing = self.test_result.isnull().any().any()
        else:
            has_missing = np.isnan(self.test_result).any()
        return not has_missing, True, has_missing


class FitTransformPatternBattery(TestBattery):
    """Test battery for Exercise 4: proper fit/transform pattern."""
    
    def _get_reference_inputs(self):
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0],
            'B': [10.0, 20.0, 30.0, np.nan, 50.0]
        })
        self.X_test = pd.DataFrame({
            'A': [np.nan, 7.0, 8.0],
            'B': [60.0, np.nan, 80.0]
        })

    def extract_info(self):
        try:
            self.imputer = self.learner_object()
            self.imputer.fit(self.X_train)
            self.train_result = self.imputer.transform(self.X_train)
            self.test_result = self.imputer.transform(self.X_test)
            
            self.has_statistics = hasattr(self.imputer, 'statistics_')
            if self.has_statistics:
                self.statistics = self.imputer.statistics_
            else:
                self.statistics = None
        except Exception as e:
            self.train_result = None
            self.test_result = None
            self.has_statistics = False
            self.statistics = None
            self.error = str(e)

    def get_reference_checks(self):
        train_mean_A = self.X_train['A'].mean()
        train_mean_B = self.X_train['B'].mean()
        self.reference_checks = {
            "train_mean_A": train_mean_A,
            "train_mean_B": train_mean_B,
        }

    def uses_training_statistics(self):
        if self.test_result is None:
            return None, "Uses training stats", True
        
        train_mean_A = self.X_train['A'].mean()
        if isinstance(self.test_result, pd.DataFrame):
            test_A_imputed = self.test_result['A'].iloc[0]
        else:
            test_A_imputed = self.test_result[0, 0]
        
        uses_train = np.isclose(test_A_imputed, train_mean_A, rtol=1e-5)
        return uses_train, True, not uses_train

    def statistics_stored(self):
        return self.has_statistics, True, not self.has_statistics

    def correct_train_imputation(self):
        if self.train_result is None:
            return None, "Correct imputation", True
        
        train_mean_A = self.X_train['A'].mean()
        if isinstance(self.train_result, pd.DataFrame):
            imputed_value = self.train_result['A'].iloc[2]
        else:
            imputed_value = self.train_result[2, 0]
        
        correct = np.isclose(imputed_value, train_mean_A, rtol=1e-5)
        return correct, True, not correct


class ImputationLoggerBattery(TestBattery):
    """Test battery for Exercise 5: imputation logging/tracking."""
    
    def _get_reference_inputs(self):
        self.X = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [np.nan, 2.0, np.nan, 4.0, np.nan]
        })

    def extract_info(self):
        try:
            self.logger = self.learner_object()
            self.result = self.logger.fit_transform(self.X)
            
            self.has_log = hasattr(self.logger, 'imputation_log_')
            if self.has_log:
                self.log = self.logger.imputation_log_
            else:
                self.log = None
        except Exception as e:
            self.result = None
            self.has_log = False
            self.log = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {}

    def has_imputation_log(self):
        return self.has_log, True, not self.has_log

    def log_is_dataframe_or_dict(self):
        if self.log is None:
            return None, "DataFrame or dict", True
        is_valid = isinstance(self.log, (pd.DataFrame, dict, list))
        return is_valid, True, not is_valid

    def log_tracks_positions(self):
        if self.log is None:
            return None, "Tracks positions", True
        
        if isinstance(self.log, pd.DataFrame):
            has_row_info = 'row' in self.log.columns or 'index' in self.log.columns
            has_col_info = 'column' in self.log.columns or 'col' in self.log.columns
            tracks = has_row_info and has_col_info
        elif isinstance(self.log, dict):
            tracks = len(self.log) > 0
        elif isinstance(self.log, list):
            tracks = len(self.log) > 0
        else:
            tracks = False
        
        return tracks, True, not tracks

    def log_has_correct_count(self):
        if self.log is None:
            return None, "Correct count", True
        
        expected_count = 5
        
        if isinstance(self.log, pd.DataFrame):
            actual_count = len(self.log)
        elif isinstance(self.log, dict):
            actual_count = sum(len(v) if isinstance(v, list) else 1 for v in self.log.values())
        elif isinstance(self.log, list):
            actual_count = len(self.log)
        else:
            actual_count = 0
        
        correct = actual_count == expected_count
        return correct, True, not correct


class CompletePipelineBattery(TestBattery):
    """Test battery for Exercise 6: complete preprocessing pipeline."""
    
    def _get_reference_inputs(self):
        np.random.seed(42)
        n = 100
        self.X_train = pd.DataFrame({
            'age': np.random.normal(40, 10, n),
            'income': np.random.lognormal(10, 0.5, n),
            'category': np.random.choice(['A', 'B', 'C'], n)
        })
        self.X_test = pd.DataFrame({
            'age': np.random.normal(40, 10, 20),
            'income': np.random.lognormal(10, 0.5, 20),
            'category': np.random.choice(['A', 'B', 'C'], 20)
        })
        
        mask = np.random.random(n) < 0.15
        self.X_train.loc[mask, 'age'] = np.nan
        mask = np.random.random(n) < 0.10
        self.X_train.loc[mask, 'income'] = np.nan
        
        mask = np.random.random(20) < 0.20
        self.X_test.loc[mask, 'age'] = np.nan

    def extract_info(self):
        try:
            self.pipeline = self.learner_object()
            
            from sklearn.pipeline import Pipeline
            self.is_pipeline = isinstance(self.pipeline, Pipeline)
            
            if self.is_pipeline:
                self.pipeline.fit(self.X_train)
                self.train_result = self.pipeline.transform(self.X_train)
                self.test_result = self.pipeline.transform(self.X_test)
                self.n_steps = len(self.pipeline.steps)
            else:
                self.train_result = None
                self.test_result = None
                self.n_steps = 0
        except Exception as e:
            self.is_pipeline = False
            self.train_result = None
            self.test_result = None
            self.n_steps = 0
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {}

    def is_sklearn_pipeline(self):
        return self.is_pipeline, True, not self.is_pipeline

    def has_multiple_steps(self):
        has_steps = self.n_steps >= 2
        return has_steps, True, not has_steps

    def train_no_missing_numeric(self):
        if self.train_result is None:
            return None, "No missing", True
        
        if isinstance(self.train_result, pd.DataFrame):
            numeric_cols = self.train_result.select_dtypes(include=[np.number]).columns
            has_missing = self.train_result[numeric_cols].isnull().any().any()
        else:
            has_missing = np.isnan(self.train_result).any()
        
        return not has_missing, True, has_missing

    def test_no_missing_numeric(self):
        if self.test_result is None:
            return None, "No missing", True
        
        if isinstance(self.test_result, pd.DataFrame):
            numeric_cols = self.test_result.select_dtypes(include=[np.number]).columns
            has_missing = self.test_result[numeric_cols].isnull().any().any()
        else:
            has_missing = np.isnan(self.test_result).any()
        
        return not has_missing, True, has_missing

    def output_shape_correct(self):
        if self.train_result is None or self.test_result is None:
            return None, "Correct shape", True
        
        if isinstance(self.train_result, pd.DataFrame):
            train_rows = len(self.train_result)
            test_rows = len(self.test_result)
        else:
            train_rows = self.train_result.shape[0]
            test_rows = self.test_result.shape[0]
        
        train_correct = train_rows == len(self.X_train)
        test_correct = test_rows == len(self.X_test)
        
        return train_correct and test_correct, True, not (train_correct and test_correct)
