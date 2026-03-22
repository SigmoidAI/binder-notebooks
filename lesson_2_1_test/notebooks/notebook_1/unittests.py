from types import FunctionType

import helper_utils
import numpy as np
import pandas as pd
import pandas.testing as pdt
from dlai_grader.grading import print_feedback, test_case


def exercise_1(learner_func):
    def g():
        cases: list[test_case] = []
        func_name = "create_pandas_series"

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
        if not isinstance(result, tuple) or len(result) != 5:
            t.failed = True
            t.msg = "Returned object must be a tuple of length 5"
            t.want = "tuple with 5 pd.Series objects"
            t.got = f"type={type(result)}, len={len(result) if isinstance(result, tuple) else 'N/A'}"
            return cases + [t]
        cases.append(t)

        ser_1, ser_2, ser_3, ser_4, ser_5 = result

        # --- ser_1 checks ---
        t = test_case()
        expected_1 = pd.Series(data=[10, 20, 30, 40, 50], name="SERIES_FROM_LIST")
        if not isinstance(ser_1, pd.Series) or not ser_1.equals(expected_1):
            t.failed = True
            t.msg = "ser_1 is incorrect"
            t.want = expected_1
            t.got = ser_1
        cases.append(t)

        # --- ser_2 checks ---
        t = test_case()
        expected_2 = pd.Series(
            data=0,
            index=["VAL_1", "VAL_2", "VAL_3"],
            name="SERIES_WITH_ZEROS",
        )
        if not isinstance(ser_2, pd.Series) or not ser_2.equals(expected_2):
            t.failed = True
            t.msg = "ser_2 is incorrect"
            t.want = expected_2
            t.got = ser_2
        cases.append(t)

        # --- ser_3 checks ---
        t = test_case()
        cond = (
            isinstance(ser_3, pd.Series)
            and ser_3.name == "SERIES_FROM_NP_ARRAY"
            and len(ser_3) == 5
            and all(isinstance(x, str) and x.isalpha() for x in ser_3.values)
        )
        if not cond:
            t.failed = True
            t.msg = "ser_3 must be a Series of 5 alphabetic characters from numpy array"
            t.want = "5 alphabetic characters, name='SERIES_FROM_NP_ARRAY'"
            t.got = ser_3
        cases.append(t)

        # --- ser_4 checks ---
        t = test_case()
        expected_4 = pd.Series(
            data={"INDEX_1": "VAL_A", "INDEX_2": "VAL_B", "INDEX_3": "VAL_C"},
            name="SERIES_FROM_PY_DICT",
        )
        if not isinstance(ser_4, pd.Series) or not ser_4.equals(expected_4):
            t.failed = True
            t.msg = "ser_4 is incorrect"
            t.want = expected_4
            t.got = ser_4
        cases.append(t)

        # --- ser_5 checks ---
        t = test_case()
        defined_df = pd.DataFrame(
            data=[
                ["A", "B", "C"],
                [1, 2, 3],
                [None, 0.2, "D"],
            ],
            columns=["Feature_1", "Feature_2", "Feature_3"],
        )

        expected_5 = pd.Series(
            data=defined_df["Feature_2"],
            name="SERIES_FROM_DF_COL",
        ).rename(index={0: "CHANGED_INDEX"})

        if not isinstance(ser_5, pd.Series) or not ser_5.equals(expected_5):
            t.failed = True
            t.msg = "ser_5 is incorrect"
            t.want = expected_5
            t.got = ser_5
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)

def exercise_2(learner_func):
    def g():
        cases: list[test_case] = []
        func_name = "create_pandas_dataframes"

        # --- 1. Function type ---
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # --- 2. Safe execution ---
        try:
            result = learner_func()
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{func_name} raised an exception"
            t.want = "Function to execute without errors"
            t.got = str(e)
            return cases + [t]

        # --- 3. Structure check ---
        t = test_case()
        if not isinstance(result, tuple) or len(result) != 4:
            t.failed = True
            t.msg = "Function must return a tuple of 4 DataFrames"
            t.want = "tuple of length 4"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        df_1, df_2, df_3, df_4 = result

        # ======================================================
        # df_1: dictionary construction
        # ======================================================
        t = test_case()
        expected_df1 = pd.DataFrame({
            "FEAT_1": [1, 2, 3],
            "FEAT_2": [20, 50, 80],
            "FEAT_3": [5, 10, 15],
        })
        
        if not isinstance(df_1, pd.DataFrame) or not df_1.equals(expected_df1):
            t.failed = True
            t.msg = "df_1 is incorrect"
            t.want = expected_df1
            t.got = df_1
        cases.append(t)

        # ======================================================
        # df_2: list of dicts with alpha chars
        # ======================================================
        t = test_case()
        cond = (
            isinstance(df_2, pd.DataFrame)
            and set(df_2.columns) == set(["FEAT_1", "FEAT_2", "FEAT_3"])
            and df_2.map(lambda x: isinstance(x, str) and x.isalpha()).all().all()
        )
        if not cond:
            t.failed = True
            t.msg = "df_2 must contain only alphabetic string values with columns FEAT_1, FEAT_2, FEAT_3"
            t.want = "DataFrame of alphabetic characters"
            t.got = df_2
        cases.append(t)

        # ======================================================
        # df_3: 2D numpy float64 array (3 rows, 4 columns)
        # ======================================================
        t = test_case()
        cond = (
            isinstance(df_3, pd.DataFrame)
            and df_3.shape == (3, 4)
            and all(dtype == np.float64 for dtype in df_3.dtypes)
        )
        if not cond:
            t.failed = True
            t.msg = "df_3 must be a 3x4 DataFrame with float64 dtype"
            t.want = "(3,4) float64 DataFrame"
            t.got = (type(df_3), getattr(df_3, "shape", None), getattr(df_3, "dtypes", None))
        cases.append(t)

        # ======================================================
        # df_4: from defined series with renaming rules
        # ======================================================
        t = test_case()

        defined_ser_1 = pd.Series(["A", "B", "C"], name="Feature_1")
        defined_ser_2 = pd.Series([1, 2, 3], name="Feature_2")
        defined_ser_3 = pd.Series([None, 4.3, None], name="Feature_3")
        defined_ser_4 = pd.Series(["D", "S", None], name="Feature_3")

        expected_df4 = pd.concat(
            [
                defined_ser_1.rename("Feature_1"),
                defined_ser_2.rename("NEW_COL_NAME_2"),
                defined_ser_3.rename("Feature_3"),
                defined_ser_4.rename("NEW_COL_NAME_4"),
            ],
            axis=1,
        )

        if not isinstance(df_4, pd.DataFrame) or not df_4.equals(expected_df4):
            t.failed = True
            t.msg = "df_4 is incorrect (column naming or concatenation logic wrong)"
            t.want = expected_df4
            t.got = df_4
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
    
def exercise_3(learner_func):
    def g():
        cases: list[test_case] = []
        func_name = "read_from_csv"

        # --- 1. Function type ---
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # --- 2. Safe execution ---
        try:
            result = learner_func()
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{func_name} raised an exception"
            t.want = "Function to execute without errors"
            t.got = str(e)
            return cases + [t]

        # --- 3. Structure check ---
        t = test_case()
        if not isinstance(result, tuple) or len(result) != 4:
            t.failed = True
            t.msg = "Function must return a tuple of 4 DataFrames"
            t.want = "tuple of length 4"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        df_1, df_2, df_3, df_4 = result

        # Load expected base dataset
        path_to_csv = helper_utils.get_path_to_dataset_by_type(dataset_type="csv")
        full_df = pd.read_csv(path_to_csv)

        # ======================================================
        # df_1: entire dataset as-is
        # ======================================================
        t = test_case()
        try:
            cond = isinstance(df_1, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(df_1.reset_index(drop=True), full_df.reset_index(drop=True))
        except AssertionError as e:
            cond = False
            err = str(e)

        if not cond:
            t.failed = True
            t.msg = "df_from_csv_1 must contain the entire dataset as-is"
            t.want = "DataFrame identical to pd.read_csv(path)"
            t.got = df_1
        cases.append(t)

        # ======================================================
        # df_2: first 10 non-header rows
        # ======================================================
        t = test_case()
        expected_df2 = full_df.head(10)
        try:
            cond = isinstance(df_2, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(df_2.reset_index(drop=True), expected_df2.reset_index(drop=True))
        except AssertionError as e:
            cond = False
            err = str(e)

        if not cond:
            t.failed = True
            t.msg = "df_from_csv_2 must contain first 10 rows"
            t.want = expected_df2
            t.got = df_2
        cases.append(t)

        # ======================================================
        # df_3: 8 rows, last 5 columns capitalized
        # ======================================================
        t = test_case()
        col_names = pd.read_csv(path_to_csv, nrows=0).columns  # get all column names
        last6_cols = col_names[-6:]  # select last 6 columns

        expected_df3: pd.DataFrame = (
            pd.read_csv(path_to_csv, usecols=last6_cols, nrows=8)
            .rename(columns=str.capitalize)
        )
        
        try:
            cond = isinstance(df_3, pd.DataFrame) and df_3.shape[0] == 8
            if cond:
                pdt.assert_frame_equal(df_3.reset_index(drop=True), expected_df3.reset_index(drop=True))
        except AssertionError as e:
            cond = False
            err = str(e)

        if not cond:
            t.failed = True
            t.msg = "df_from_csv_3 must have 8 rows and last 5 columns capitalized"
            t.want = expected_df3
            t.got = df_3
        cases.append(t)

        # ======================================================
        # df_4: skip first 1000 rows, 5 rows, index_col=0,
        #       odd-indexed columns uppercase (1-based indexing)
        # ======================================================
        t = test_case()
        expected_df4 = pd.read_csv(path_to_csv, skiprows=range(1, 1001), nrows=5, index_col=0)

        # Apply renaming rule: odd-indexed column names uppercase (1-based)
        new_columns = []
        for i, col in enumerate(expected_df4.columns):
            if i % 2 == 1:  # odd position (1-based)
                new_columns.append(col.upper())
            else:
                new_columns.append(col)
        expected_df4.columns = new_columns

        try:
            cond = isinstance(df_4, pd.DataFrame) and df_4.shape[0] == 5
            if cond:
                pdt.assert_frame_equal(df_4, expected_df4)
        except AssertionError as e:
            cond = False
            err = str(e)

        if not cond:
            t.failed = True
            t.msg = (
                "df_from_csv_4 must: skip first 1000 rows, have 5 rows, use column 0 as index, "
                "and uppercase odd-indexed column names"
            )
            t.want = expected_df4
            t.got = df_4
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases: list[test_case] = []
        func_name = "read_from_excel"

        # --- 1. Function type ---
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # --- 2. Safe execution ---
        try:
            result = learner_func()
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{func_name} raised an exception"
            t.want = "Function to execute without errors"
            t.got = str(e)
            return cases + [t]

        # --- 3. Structure check ---
        t = test_case()
        if not isinstance(result, tuple) or len(result) != 4:
            t.failed = True
            t.msg = "Function must return a tuple of 4 DataFrames"
            t.want = "tuple of length 4"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        df_1, df_2, df_3, df_4 = result

        # Load expected base dataset
        path_to_excel = helper_utils.get_path_to_dataset_by_type(dataset_type="excel")
        full_df = pd.read_excel(path_to_excel)

        # ======================================================
        # df_1: entire dataset as-is
        # ======================================================
        t = test_case()
        try:
            cond = isinstance(df_1, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(df_1.reset_index(drop=True), full_df.reset_index(drop=True))
        except AssertionError:
            cond = False

        if not cond:
            t.failed = True
            t.msg = "df_from_excel_1 must contain the entire dataset as-is"
            t.want = "DataFrame identical to pd.read_excel(path)"
            t.got = df_1
        cases.append(t)

        # ======================================================
        # df_2: last 15 rows
        # ======================================================
        t = test_case()
        expected_df2 = full_df.tail(15)
        try:
            cond = isinstance(df_2, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(df_2.reset_index(drop=True), expected_df2.reset_index(drop=True))
        except AssertionError:
            cond = False

        if not cond:
            t.failed = True
            t.msg = "df_from_excel_2 must contain last 15 rows"
            t.want = expected_df2
            t.got = df_2
        cases.append(t)

        # ======================================================
        # df_3: first 3 columns, 5 rows, renamed Feat_1..3
        # ======================================================
        t = test_case()
        expected_df3 = full_df.iloc[:5, :3].copy()
        expected_df3.columns = [f"Feat_{i+1}" for i in range(len(expected_df3.columns))]
        try:
            cond = isinstance(df_3, pd.DataFrame) and df_3.shape == expected_df3.shape
            if cond:
                pdt.assert_frame_equal(df_3.reset_index(drop=True), expected_df3.reset_index(drop=True))
        except AssertionError:
            cond = False

        if not cond:
            t.failed = True
            t.msg = "df_from_excel_3 must have first 3 columns, 5 rows, renamed as Feat_1..3"
            t.want = expected_df3
            t.got = df_3
        cases.append(t)

        # ======================================================
        # df_4: skip 50 rows, 10 rows, index_col=0, rename columns Feat_1..
        # ======================================================
        t = test_case()
        expected_df4 = pd.read_excel(path_to_excel, skiprows=50, nrows=10, index_col=0)
        expected_df4.columns = [f"Feat_{i+1}" for i in range(len(expected_df4.columns))]

        try:
            cond = isinstance(df_4, pd.DataFrame) and df_4.shape == expected_df4.shape
            if cond:
                pdt.assert_frame_equal(df_4, expected_df4)
        except AssertionError:
            cond = False

        if not cond:
            t.failed = True
            t.msg = "df_from_excel_4 must skip 50 rows, have 10 rows, use first column as index, rename columns Feat_1.."
            t.want = expected_df4
            t.got = df_4
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
    
def exercise_5(learner_func):
    def g():
        cases: list[test_case] = []
        func_name = "read_from_json"

        # --- 1. Function type ---
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # --- 2. Safe execution ---
        try:
            result = learner_func()
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{func_name} raised an exception"
            t.want = "Function to execute without errors"
            t.got = str(e)
            return cases + [t]

        # --- 3. Structure check ---
        t = test_case()
        if not isinstance(result, tuple) or len(result) != 5:
            t.failed = True
            t.msg = "Function must return a tuple of 5 DataFrames"
            t.want = "tuple of length 5"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        df_1, df_2, df_3, df_4, df_5 = result

        # Load expected base dataset
        path_to_json_file = helper_utils.__get_path_to_json_dataset()
        full_df = pd.read_json(path_to_json_file, orient="records")

        # ======================================================
        # df_1: entire dataset from records
        # ======================================================
        t = test_case()
        try:
            cond = isinstance(df_1, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(df_1.reset_index(drop=True), full_df.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "df_from_json_1 must contain the entire dataset as-is"
            t.want = "DataFrame identical to pd.read_json(records)"
            t.got = df_1
        cases.append(t)

        # ======================================================
        # df_2: from split
        # ======================================================
        t = test_case()
        expected_df2 = pd.read_json(path_to_json_file.parent / "iris_split.json", orient="split")
        try:
            cond = isinstance(df_2, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(df_2.reset_index(drop=True), expected_df2.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "df_from_json_2 must match split-orient DataFrame"
            t.want = expected_df2
            t.got = df_2
        cases.append(t)

        # ======================================================
        # df_3: from index
        # ======================================================
        t = test_case()
        expected_df3 = pd.read_json(path_to_json_file.parent / "iris_index.json", orient="index")
        try:
            cond = isinstance(df_3, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(df_3.reset_index(drop=True), expected_df3.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "df_from_json_3 must match index-orient DataFrame"
            t.want = expected_df3
            t.got = df_3
        cases.append(t)

        # ======================================================
        # df_4: last 30 rows from columns-orient
        # ======================================================
        t = test_case()
        expected_df4 = pd.read_json(path_to_json_file.parent / "iris_columns.json", orient="columns").tail(30)
        try:
            cond = isinstance(df_4, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(df_4.reset_index(drop=True), expected_df4.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "df_from_json_4 must contain last 30 rows from columns-orient JSON"
            t.want = expected_df4
            t.got = df_4
        cases.append(t)

        # ======================================================
        # df_5: first 50 rows from lines JSON
        # ======================================================
        t = test_case()
        expected_df5 = pd.read_json(path_to_json_file.parent / "iris_lines.jsonl", lines=True).head(50)
        try:
            cond = isinstance(df_5, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(df_5.reset_index(drop=True), expected_df5.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "df_from_json_5 must contain first 50 rows from lines JSON"
            t.want = expected_df5
            t.got = df_5
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_6(learner_func):
    def g():
        cases: list[test_case] = []
        func_name = "read_from_sql_db"

        # --- 1. Function type ---
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # --- 2. Safe execution ---
        try:
            result = learner_func()
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"{func_name} raised an exception"
            t.want = "Function to execute without errors"
            t.got = str(e)
            return cases + [t]

        # --- 3. Structure check ---
        t = test_case()
        if not isinstance(result, tuple) or len(result) != 3:
            t.failed = True
            t.msg = "Function must return a tuple of 3 DataFrames"
            t.want = "tuple of length 3"
            t.got = type(result)
            return cases + [t]
        cases.append(t)

        df_1, df_2, df_3 = result

        # Load expected base dataset from the SQLite DB
        path_to_sqlite_db_file = helper_utils.get_path_to_dataset_by_type(dataset_type="sql_db")
        conn = helper_utils.create_sqlite_conn(db_path=path_to_sqlite_db_file)
        table_name = "IRIS_DATASET_TABLE"
        full_df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        helper_utils.close_sqlite_conn(conn=conn)

        # --- df_1: entire dataset ---
        t = test_case()
        try:
            cond = isinstance(df_1, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(df_1.reset_index(drop=True), full_df.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "df_from_sql_db_1 must contain the entire dataset as-is"
            t.want = "DataFrame identical to full dataset from DB"
            t.got = df_1
        cases.append(t)

        # --- df_2: last 10 rows, last 2 columns + id as index ---
        t = test_case()
        try:
            all_cols = full_df.columns.tolist()
            last_two_cols = all_cols[-2:]
            expected_df2 = full_df[["id"] + last_two_cols].tail(10).set_index("id")
            cond = isinstance(df_2, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(df_2.reset_index(drop=True), expected_df2.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "df_from_sql_db_2 must contain last 10 rows with last 2 columns and id as index"
            t.want = expected_df2
            t.got = df_2
        cases.append(t)

        # --- df_3: filtered dataset ---
        t = test_case()
        try:
            expected_df3 = full_df[
                (full_df["species"].isin(["setosa", "versicolor"])) &
                (full_df["sepalLength"].between(4.7, 5.0))
            ].set_index("id")
            cond = isinstance(df_3, pd.DataFrame)
            if cond:
                pdt.assert_frame_equal(df_3.reset_index(drop=True), expected_df3.reset_index(drop=True))
        except AssertionError:
            cond = False
        if not cond:
            t.failed = True
            t.msg = "df_from_sql_db_3 must contain filtered rows with species in setosa/versicolor and sepalLength between 4.7 and 5.0"
            t.want = expected_df3
            t.got = df_3
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
    # https: // mybinder.org / v2 / gh / malmilo6 / test - binder / main?filepath = lesson_2_1 / notebooks / notebook_1 / L2_1_N1_Assignment_EN.ipynb