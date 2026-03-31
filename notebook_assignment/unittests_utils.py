"""Battery test classes for Imputation Impact Assessment exercises."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


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
        if isinstance(want, float):
            condition = abs(got - want) > 0.01  # tolerance for floats
        elif isinstance(want, np.ndarray):
            condition = not np.allclose(got, want, rtol=0.01)
        else:
            condition = got != want
        return got, want, condition


class BaselineModelBattery(TestBattery):
    """Battery for Exercise 1: Prepare baseline model."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        np.random.seed(42)
        n_samples = 500
        self.X = pd.DataFrame({
            'feature_0': np.random.normal(50, 10, n_samples),
            'feature_1': np.random.uniform(0, 100, n_samples),
            'feature_2': np.random.exponential(20, n_samples) + 10,
            'feature_3': np.random.normal(50, 20, n_samples),
            'feature_4': np.random.normal(100, 25, n_samples)
        })
        self.y = 2.0 * self.X['feature_0'] + 0.5 * self.X['feature_1'] - 1.5 * self.X['feature_2']
        self.y += np.random.normal(0, 5, n_samples)
        self.y = pd.Series(self.y, name='target')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def extract_info(self):
        try:
            self.result = self.learner_object(self.X_train, self.X_test, self.y_train, self.y_test)
        except Exception as e:
            self.result = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            'returns_dict': True,
            'has_rmse': True,
            'has_mae': True,
            'has_r2': True,
            'rmse_reasonable': True,
            'r2_positive': True
        }

    def returns_dict(self):
        got = isinstance(self.result, dict)
        return self._check('returns_dict', got)

    def has_rmse(self):
        got = 'rmse' in self.result if self.result else False
        return self._check('has_rmse', got)

    def has_mae(self):
        got = 'mae' in self.result if self.result else False
        return self._check('has_mae', got)

    def has_r2(self):
        got = 'r2' in self.result if self.result else False
        return self._check('has_r2', got)

    def rmse_reasonable(self):
        got = self.result['rmse'] < 100 if self.result and 'rmse' in self.result else False
        return self._check('rmse_reasonable', got)

    def r2_positive(self):
        got = self.result['r2'] > 0 if self.result and 'r2' in self.result else False
        return self._check('r2_positive', got)


class DeletionImpactBattery(TestBattery):
    """Battery for Exercise 2: Evaluate deletion impact."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        np.random.seed(42)
        n_samples = 500
        self.X = pd.DataFrame({
            'feature_0': np.random.normal(50, 10, n_samples),
            'feature_1': np.random.uniform(0, 100, n_samples),
            'feature_2': np.random.exponential(20, n_samples) + 10,
            'feature_3': np.random.normal(50, 20, n_samples),
            'feature_4': np.random.normal(100, 25, n_samples)
        })
        self.y = 2.0 * self.X['feature_0'] + 0.5 * self.X['feature_1'] - 1.5 * self.X['feature_2']
        self.y += np.random.normal(0, 5, n_samples)
        self.y = pd.Series(self.y, name='target')
        # Introduce missing values
        np.random.seed(42)
        mask = np.random.random(self.X.shape) < 0.2
        for i in range(len(self.X)):
            if mask[i].sum() == self.X.shape[1]:
                mask[i, np.random.randint(self.X.shape[1])] = False
        self.X_missing = self.X.mask(mask)

    def extract_info(self):
        try:
            self.X_clean, self.y_clean, self.deleted_count = self.learner_object(
                self.X_missing.copy(), self.y.copy()
            )
        except Exception as e:
            self.X_clean = None
            self.y_clean = None
            self.deleted_count = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            'returns_dataframe': True,
            'no_missing_values': True,
            'deleted_count_positive': True,
            'y_length_matches': True,
            'samples_reduced': True
        }

    def returns_dataframe(self):
        got = isinstance(self.X_clean, pd.DataFrame)
        return self._check('returns_dataframe', got)

    def no_missing_values(self):
        got = self.X_clean.isnull().sum().sum() == 0 if self.X_clean is not None else False
        return self._check('no_missing_values', got)

    def deleted_count_positive(self):
        got = self.deleted_count > 0 if self.deleted_count is not None else False
        return self._check('deleted_count_positive', got)

    def y_length_matches(self):
        got = len(self.X_clean) == len(self.y_clean) if self.X_clean is not None and self.y_clean is not None else False
        return self._check('y_length_matches', got)

    def samples_reduced(self):
        got = len(self.X_clean) < len(self.X_missing) if self.X_clean is not None else False
        return self._check('samples_reduced', got)


class SimpleImputationBattery(TestBattery):
    """Battery for Exercise 3: Evaluate simple imputation impact."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        np.random.seed(42)
        n_samples = 500
        self.X = pd.DataFrame({
            'feature_0': np.random.normal(50, 10, n_samples),
            'feature_1': np.random.uniform(0, 100, n_samples),
            'feature_2': np.random.exponential(20, n_samples) + 10,
            'feature_3': np.random.normal(50, 20, n_samples),
            'feature_4': np.random.normal(100, 25, n_samples)
        })
        # Introduce missing values
        np.random.seed(42)
        mask = np.random.random(self.X.shape) < 0.2
        for i in range(len(self.X)):
            if mask[i].sum() == self.X.shape[1]:
                mask[i, np.random.randint(self.X.shape[1])] = False
        self.X_missing = self.X.mask(mask)

    def extract_info(self):
        try:
            from sklearn.model_selection import train_test_split as _tts
            self.X_train, self.X_test, _, _ = _tts(
                self.X_missing, self.X_missing.iloc[:, 0], test_size=0.2, random_state=42
            )
            self.X_train_imp, self.X_test_imp = self.learner_object(
                self.X_train.copy(), self.X_test.copy(), strategy='mean'
            )
            # For compatibility, expose primary result
            self.X_imputed = self.X_train_imp
        except Exception as e:
            self.X_train_imp = None
            self.X_test_imp = None
            self.X_imputed = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            'returns_dataframe': True,
            'no_missing_values': True,
            'same_shape': True,
            'columns_preserved': True
        }

    def returns_dataframe(self):
        got = isinstance(self.X_imputed, pd.DataFrame)
        return self._check('returns_dataframe', got)

    def no_missing_values(self):
        got = self.X_imputed.isnull().sum().sum() == 0 if self.X_imputed is not None else False
        return self._check('no_missing_values', got)

    def same_shape(self):
        got = self.X_imputed.shape == self.X_train.shape if self.X_imputed is not None else False
        return self._check('same_shape', got)

    def columns_preserved(self):
        got = list(self.X_imputed.columns) == list(self.X_train.columns) if self.X_imputed is not None else False
        return self._check('columns_preserved', got)


class AdvancedImputationBattery(TestBattery):
    """Battery for Exercise 4: Evaluate advanced imputation impact."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        np.random.seed(42)
        n_samples = 200  # Smaller for faster KNN
        self.X = pd.DataFrame({
            'feature_0': np.random.normal(50, 10, n_samples),
            'feature_1': np.random.uniform(0, 100, n_samples),
            'feature_2': np.random.exponential(20, n_samples) + 10,
            'feature_3': np.random.normal(50, 20, n_samples),
            'feature_4': np.random.normal(100, 25, n_samples)
        })
        # Introduce missing values
        np.random.seed(42)
        mask = np.random.random(self.X.shape) < 0.15
        for i in range(len(self.X)):
            if mask[i].sum() == self.X.shape[1]:
                mask[i, np.random.randint(self.X.shape[1])] = False
        self.X_missing = self.X.mask(mask)

    def extract_info(self):
        try:
            from sklearn.model_selection import train_test_split as _tts
            self.X_train, self.X_test, _, _ = _tts(
                self.X_missing, self.X_missing.iloc[:, 0], test_size=0.2, random_state=42
            )
            self.X_knn, self.X_te_knn = self.learner_object(
                self.X_train.copy(), self.X_test.copy(), method='knn'
            )
            self.X_iterative, self.X_te_iterative = self.learner_object(
                self.X_train.copy(), self.X_test.copy(), method='iterative'
            )
        except Exception as e:
            self.X_knn = None
            self.X_iterative = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            'knn_returns_dataframe': True,
            'iterative_returns_dataframe': True,
            'knn_no_missing': True,
            'iterative_no_missing': True,
            'knn_same_shape': True,
            'iterative_same_shape': True
        }

    def knn_returns_dataframe(self):
        got = isinstance(self.X_knn, pd.DataFrame)
        return self._check('knn_returns_dataframe', got)

    def iterative_returns_dataframe(self):
        got = isinstance(self.X_iterative, pd.DataFrame)
        return self._check('iterative_returns_dataframe', got)

    def knn_no_missing(self):
        got = self.X_knn.isnull().sum().sum() == 0 if self.X_knn is not None else False
        return self._check('knn_no_missing', got)

    def iterative_no_missing(self):
        got = self.X_iterative.isnull().sum().sum() == 0 if self.X_iterative is not None else False
        return self._check('iterative_no_missing', got)

    def knn_same_shape(self):
        got = self.X_knn.shape == self.X_train.shape if self.X_knn is not None else False
        return self._check('knn_same_shape', got)

    def iterative_same_shape(self):
        got = self.X_iterative.shape == self.X_train.shape if self.X_iterative is not None else False
        return self._check('iterative_same_shape', got)


class ImpactReportBattery(TestBattery):
    """Battery for Exercise 5: Create comprehensive impact report."""
    
    def __init__(self, learner_func):
        super().__init__(learner_func)

    def _get_reference_inputs(self):
        np.random.seed(42)
        n_samples = 300
        self.X_complete = pd.DataFrame({
            'feature_0': np.random.normal(50, 10, n_samples),
            'feature_1': np.random.uniform(0, 100, n_samples),
            'feature_2': np.random.exponential(20, n_samples) + 10,
            'feature_3': np.random.normal(50, 20, n_samples),
            'feature_4': np.random.normal(100, 25, n_samples)
        })
        self.y = (2.0 * self.X_complete['feature_0']
                  + 0.5 * self.X_complete['feature_1']
                  - 1.5 * self.X_complete['feature_2']
                  + np.random.normal(0, 5, n_samples))
        self.y = pd.Series(self.y, name='target')
        self.X_missing = self.X_complete.copy()
        np.random.seed(42)
        mask = np.random.random(self.X_complete.shape) < 0.15
        self.X_missing = self.X_complete.mask(mask)

    def extract_info(self):
        try:
            self.report = self.learner_object(
                self.X_complete.copy(), self.X_missing.copy(), self.y.copy()
            )
        except Exception as e:
            self.report = None
            self.error = str(e)

    def get_reference_checks(self):
        self.reference_checks = {
            'returns_dataframe': True,
            'has_strategy_col': True,
            'has_rmse_col': True,
            'has_r2_col': True,
            'has_n_samples_col': True,
            'has_multiple_rows': True,
        }

    def returns_dataframe(self):
        got = isinstance(self.report, pd.DataFrame)
        return self._check('returns_dataframe', got)

    def has_strategy_col(self):
        got = 'strategy' in self.report.columns if self.report is not None else False
        return self._check('has_strategy_col', got)

    def has_rmse_col(self):
        got = 'rmse' in self.report.columns if self.report is not None else False
        return self._check('has_rmse_col', got)

    def has_r2_col(self):
        got = 'r2' in self.report.columns if self.report is not None else False
        return self._check('has_r2_col', got)

    def has_n_samples_col(self):
        got = 'n_samples' in self.report.columns if self.report is not None else False
        return self._check('has_n_samples_col', got)

    def has_multiple_rows(self):
        got = len(self.report) >= 3 if self.report is not None else False
        return self._check('has_multiple_rows', got)
