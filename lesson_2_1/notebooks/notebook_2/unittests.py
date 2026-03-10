from types import FunctionType

import helper_utils
import numpy as np
import pandas as pd
import pandas.testing as pdt
from dlai_grader.grading import print_feedback, test_case


def exercise_1(learner_func):
    def g():
        cases: list[test_case] = []
        func_name = "explore_numeric_series"

        # --- check function type ---
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # --- call function safely ---
        try:
            result = learner_func()
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when calling {func_name}"
            t.want = f"{func_name} to execute without exceptions."
            t.got = f"Exception raised: {e}"
            return cases + [t]

        # --- check tuple length ---
        t = test_case()
        if not isinstance(result, tuple) or len(result) != 11:
            t.failed = True
            t.msg = "Returned object must be a tuple of length 11"
            t.want = "tuple with 11 pd.Series/scalar elements"
            t.got = f"type={type(result)}, len={len(result) if isinstance(result, tuple) else 'N/A'}"
            return cases + [t]
        cases.append(t)

        head_5, tail_3, dtype_result, null_count, mean_val, median_val, min_val, max_val, std_val, var_val, describe_result = result

        # Reference data (same as in the exercise function)
        data = [22.5, 27.2, 31.8, None, 28.6, 29.0, None, 32.1, 35.0, 24.1, 30.5, 26.7, 28.9, None, 33.2]
        series = pd.Series(data=data, name="TEMPERATURE_DATA")

        # --- head_5 ---
        t = test_case()
        try:
            expected = series.head(5)
            cond = isinstance(head_5, pd.Series)
            if cond:
                pdt.assert_series_equal(head_5.reset_index(drop=True), expected.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "head_5 is incorrect - use series.head(5)"
            t.want = series.head(5)
            t.got = head_5
        cases.append(t)

        # --- tail_3 ---
        t = test_case()
        try:
            expected = series.tail(3)
            cond = isinstance(tail_3, pd.Series)
            if cond:
                pdt.assert_series_equal(tail_3.reset_index(drop=True), expected.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "tail_3 is incorrect - use series.tail(3)"
            t.want = series.tail(3)
            t.got = tail_3
        cases.append(t)

        # --- dtype_result ---
        t = test_case()
        if dtype_result != series.dtype:
            t.failed = True
            t.msg = "dtype_result is incorrect - use series.dtype"
            t.want = series.dtype
            t.got = dtype_result
        cases.append(t)

        # --- null_count ---
        t = test_case()
        if null_count != series.isna().sum():
            t.failed = True
            t.msg = "null_count is incorrect - use series.isna().sum()"
            t.want = series.isna().sum()
            t.got = null_count
        cases.append(t)

        # --- mean_val ---
        t = test_case()
        try:
            cond = np.isclose(mean_val, series.mean())
        except Exception:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "mean_val is incorrect - use series.mean()"
            t.want = series.mean()
            t.got = mean_val
        cases.append(t)

        # --- median_val ---
        t = test_case()
        try:
            cond = np.isclose(median_val, series.median())
        except Exception:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "median_val is incorrect - use series.median()"
            t.want = series.median()
            t.got = median_val
        cases.append(t)

        # --- min_val ---
        t = test_case()
        try:
            cond = np.isclose(min_val, series.min())
        except Exception:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "min_val is incorrect - use series.min()"
            t.want = series.min()
            t.got = min_val
        cases.append(t)

        # --- max_val ---
        t = test_case()
        try:
            cond = np.isclose(max_val, series.max())
        except Exception:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "max_val is incorrect - use series.max()"
            t.want = series.max()
            t.got = max_val
        cases.append(t)

        # --- std_val ---
        t = test_case()
        try:
            cond = np.isclose(std_val, series.std())
        except Exception:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "std_val is incorrect - use series.std()"
            t.want = series.std()
            t.got = std_val
        cases.append(t)

        # --- var_val ---
        t = test_case()
        try:
            cond = np.isclose(var_val, series.var())
        except Exception:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "var_val is incorrect - use series.var()"
            t.want = series.var()
            t.got = var_val
        cases.append(t)

        # --- describe_result ---
        t = test_case()
        try:
            expected = series.describe()
            cond = isinstance(describe_result, pd.Series)
            if cond:
                pdt.assert_series_equal(describe_result, expected)
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "describe_result is incorrect - use series.describe()"
            t.want = series.describe()
            t.got = describe_result
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases: list[test_case] = []
        func_name = "explore_pandas_dataframe"

        # --- check function type ---
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # --- call function safely ---
        try:
            result = learner_func()
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when calling {func_name}"
            t.want = f"{func_name} to execute without exceptions."
            t.got = f"Exception raised: {e}"
            return cases + [t]

        # --- check tuple length ---
        t = test_case()
        if not isinstance(result, tuple) or len(result) != 8:
            t.failed = True
            t.msg = "Returned object must be a tuple of length 8"
            t.want = "tuple with 8 elements"
            t.got = f"type={type(result)}, len={len(result) if isinstance(result, tuple) else 'N/A'}"
            return cases + [t]
        cases.append(t)

        head_3, tail_2, shape, col_names, dtypes, null_per_col, describe_result, mem_total = result

        # Reference DataFrame (same as in the exercise function)
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
            "age": [25.0, 30.0, 35.0, 28.0, None, 22.0],
            "salary": [50000.0, 60000.0, 70000.0, None, 48000.0, 52000.0],
            "department": ["HR", "Engineering", "Engineering", "Marketing", "HR", None],
            "score": [88.0, 95.0, None, 76.0, 82.0, 90.0],
        })

        # --- head_3 ---
        t = test_case()
        try:
            expected = df.head(3)
            cond = isinstance(head_3, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(head_3.reset_index(drop=True), expected.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "head_3 is incorrect - use df.head(3)"
            t.want = df.head(3)
            t.got = head_3
        cases.append(t)

        # --- tail_2 ---
        t = test_case()
        try:
            expected = df.tail(2)
            cond = isinstance(tail_2, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(tail_2.reset_index(drop=True), expected.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "tail_2 is incorrect - use df.tail(2)"
            t.want = df.tail(2)
            t.got = tail_2
        cases.append(t)

        # --- shape ---
        t = test_case()
        if shape != df.shape:
            t.failed = True
            t.msg = "shape is incorrect - use df.shape"
            t.want = df.shape
            t.got = shape
        cases.append(t)

        # --- col_names ---
        t = test_case()
        if col_names != df.columns.tolist():
            t.failed = True
            t.msg = "col_names is incorrect - use df.columns.tolist()"
            t.want = df.columns.tolist()
            t.got = col_names
        cases.append(t)

        # --- dtypes ---
        t = test_case()
        try:
            expected = df.dtypes
            cond = isinstance(dtypes, pd.Series)
            if cond:
                pdt.assert_series_equal(dtypes, expected)
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "dtypes is incorrect - use df.dtypes"
            t.want = df.dtypes
            t.got = dtypes
        cases.append(t)

        # --- null_per_col ---
        t = test_case()
        try:
            expected = df.isna().sum()
            cond = isinstance(null_per_col, pd.Series)
            if cond:
                pdt.assert_series_equal(null_per_col, expected)
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "null_per_col is incorrect - use df.isna().sum()"
            t.want = df.isna().sum()
            t.got = null_per_col
        cases.append(t)

        # --- describe_result ---
        t = test_case()
        try:
            expected = df.describe()
            cond = isinstance(describe_result, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(describe_result, expected)
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "describe_result is incorrect - use df.describe()"
            t.want = df.describe()
            t.got = describe_result
        cases.append(t)

        # --- mem_total ---
        t = test_case()
        if mem_total != df.memory_usage(deep=True).sum():
            t.failed = True
            t.msg = "mem_total is incorrect - use df.memory_usage(deep=True).sum()"
            t.want = df.memory_usage(deep=True).sum()
            t.got = mem_total
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases: list[test_case] = []
        func_name = "select_from_dataframe"

        # --- check function type ---
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # --- call function safely ---
        try:
            result = learner_func()
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when calling {func_name}"
            t.want = f"{func_name} to execute without exceptions."
            t.got = f"Exception raised: {e}"
            return cases + [t]

        # --- check tuple length ---
        t = test_case()
        if not isinstance(result, tuple) or len(result) != 7:
            t.failed = True
            t.msg = "Returned object must be a tuple of length 7"
            t.want = "tuple with 7 elements"
            t.got = f"type={type(result)}, len={len(result) if isinstance(result, tuple) else 'N/A'}"
            return cases + [t]
        cases.append(t)

        species_col, two_cols, first_5_rows, rows_10_to_20, specific_value, setosa_rows, last3_first2 = result

        # Reference DataFrame
        df = pd.read_json(path_or_buf=helper_utils.__get_path_to_json_dataset())

        # --- species_col ---
        t = test_case()
        try:
            expected = df["species"]
            cond = isinstance(species_col, pd.Series)
            if cond:
                pdt.assert_series_equal(species_col.reset_index(drop=True), expected.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "species_col is incorrect - use df['species']"
            t.want = df["species"]
            t.got = species_col
        cases.append(t)

        # --- two_cols ---
        t = test_case()
        try:
            expected = df[["sepalLength", "sepalWidth"]]
            cond = isinstance(two_cols, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(two_cols.reset_index(drop=True), expected.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "two_cols is incorrect - use df[['sepalLength', 'sepalWidth']]"
            t.want = df[["sepalLength", "sepalWidth"]]
            t.got = two_cols
        cases.append(t)

        # --- first_5_rows ---
        t = test_case()
        try:
            expected = df.iloc[:5]
            cond = isinstance(first_5_rows, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(first_5_rows.reset_index(drop=True), expected.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "first_5_rows is incorrect - use df.iloc[:5]"
            t.want = df.iloc[:5]
            t.got = first_5_rows
        cases.append(t)

        # --- rows_10_to_20 ---
        t = test_case()
        try:
            expected = df.iloc[10:21]
            cond = isinstance(rows_10_to_20, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(rows_10_to_20.reset_index(drop=True), expected.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "rows_10_to_20 is incorrect - use df.iloc[10:21]"
            t.want = df.iloc[10:21]
            t.got = rows_10_to_20
        cases.append(t)

        # --- specific_value ---
        t = test_case()
        expected_val = df.loc[0, "sepalLength"]
        try:
            cond = np.isclose(specific_value, expected_val)
        except Exception:
            cond = specific_value == expected_val
        if not cond:
            t.failed = True
            t.msg = "specific_value is incorrect - use df.loc[0, 'sepalLength']"
            t.want = expected_val
            t.got = specific_value
        cases.append(t)

        # --- setosa_rows ---
        t = test_case()
        try:
            expected = df[df["species"] == "setosa"]
            cond = isinstance(setosa_rows, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(setosa_rows.reset_index(drop=True), expected.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "setosa_rows is incorrect - use df[df['species'] == 'setosa']"
            t.want = df[df["species"] == "setosa"]
            t.got = setosa_rows
        cases.append(t)

        # --- last3_first2 ---
        t = test_case()
        try:
            expected = df.iloc[-3:, :2]
            cond = isinstance(last3_first2, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(last3_first2.reset_index(drop=True), expected.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "last3_first2 is incorrect - use df.iloc[-3:, :2]"
            t.want = df.iloc[-3:, :2]
            t.got = last3_first2
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases: list[test_case] = []
        func_name = "groupby_and_aggregate"

        # --- check function type ---
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # --- call function safely ---
        try:
            result = learner_func()
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when calling {func_name}"
            t.want = f"{func_name} to execute without exceptions."
            t.got = f"Exception raised: {e}"
            return cases + [t]

        # --- check tuple length ---
        t = test_case()
        if not isinstance(result, tuple) or len(result) != 4:
            t.failed = True
            t.msg = "Returned object must be a tuple of length 4"
            t.want = "tuple with 4 elements"
            t.got = f"type={type(result)}, len={len(result) if isinstance(result, tuple) else 'N/A'}"
            return cases + [t]
        cases.append(t)

        mean_sepal_by_species, count_per_species, agg_sepal_length, mean_all_numeric = result

        # Reference DataFrame
        df = pd.read_json(path_or_buf=helper_utils.__get_path_to_json_dataset())

        # --- mean_sepal_by_species ---
        t = test_case()
        try:
            expected = df.groupby("species")["sepalLength"].mean()
            cond = isinstance(mean_sepal_by_species, pd.Series)
            if cond:
                pdt.assert_series_equal(mean_sepal_by_species, expected)
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "mean_sepal_by_species is incorrect - use df.groupby('species')['sepalLength'].mean()"
            t.want = df.groupby("species")["sepalLength"].mean()
            t.got = mean_sepal_by_species
        cases.append(t)

        # --- count_per_species ---
        t = test_case()
        try:
            expected = df.groupby("species").size()
            cond = isinstance(count_per_species, pd.Series)
            if cond:
                pdt.assert_series_equal(count_per_species, expected)
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "count_per_species is incorrect - use df.groupby('species').size()"
            t.want = df.groupby("species").size()
            t.got = count_per_species
        cases.append(t)

        # --- agg_sepal_length ---
        t = test_case()
        try:
            expected = df.groupby("species")["sepalLength"].agg(["mean", "std"])
            cond = isinstance(agg_sepal_length, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(agg_sepal_length, expected)
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "agg_sepal_length is incorrect - use df.groupby('species')['sepalLength'].agg(['mean', 'std'])"
            t.want = df.groupby("species")["sepalLength"].agg(["mean", "std"])
            t.got = agg_sepal_length
        cases.append(t)

        # --- mean_all_numeric ---
        t = test_case()
        try:
            expected = df.groupby("species").mean(numeric_only=True)
            cond = isinstance(mean_all_numeric, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(mean_all_numeric, expected)
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "mean_all_numeric is incorrect - use df.groupby('species').mean(numeric_only=True)"
            t.want = df.groupby("species").mean(numeric_only=True)
            t.got = mean_all_numeric
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
