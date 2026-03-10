# Data Loading & Initial Exploration Quiz

## Question 1

Which `pandas` function would you use to load a tab-separated values (TSV) file?

    A. pd.read_tsv('data.tsv')
    B. pd.read_csv('data.tsv', sep=',')
    C. pd.read_csv('data.tsv', sep='\t')
    D. pd.read_table('data.tsv', delimiter=',')

### Correct Answer: C

**Explanation:**
`pd.read_csv()` handles any delimiter-separated file, not just commas. To read a TSV file, you pass `sep='\t'` to override the default comma delimiter. There is no dedicated `pd.read_tsv()` function in `pandas`.

---

## Question 2

You receive an Excel workbook with five sheets. You want to load all of them at once into a single Python object. Which call achieves this?

    A. pd.read_excel('data.xlsx', sheet_name='all')
    B. pd.read_excel('data.xlsx', sheet_name=None)
    C. pd.read_excel('data.xlsx', sheet_name=0)
    D. pd.read_excel('data.xlsx', sheets=True)

### Correct Answer: B

**Explanation:**
Passing `sheet_name=None` tells `pd.read_excel()` to read every sheet and return a dictionary where each key is a sheet name and each value is the corresponding `pd.DataFrame`. Using `sheet_name=0` loads only the first sheet.

---

## Question 3

What does `.info()` display that `.dtypes` alone does not?

    A. The data type of each column
    B. The first five rows of the DataFrame
    C. The number of non-null values per column and total memory usage
    D. Summary statistics such as mean and standard deviation

### Correct Answer: C

**Explanation:**
`.dtypes` only shows the data type of each column. `.info()` provides a richer overview: number of rows, column names, non-null counts per column, data types, and total memory usage — making it the go-to first-look method for a newly loaded DataFrame.

---

## Question 4

A colleague hands you a JSON file where each line is a separate JSON object (JSONL format). Which argument must you set when calling `pd.read_json()`?

    A. lines=True
    B. multiline=True
    C. sep='\n'
    D. orient='records'

### Correct Answer: A

**Explanation:**
Setting `lines=True` tells `pd.read_json()` to treat each line of the file as an independent JSON object, which is the correct approach for newline-delimited JSON (`.jsonl`) files. The `orient` parameter controls the structural layout of the JSON, not whether it is line-delimited.

---

## Question 5

What is the key difference between a `pd.Series` and a `pd.DataFrame`?

    A. A `pd.Series` is one-dimensional (a single column); a `pd.DataFrame` is two-dimensional (rows and columns)
    B. A `pd.Series` can only store integers; a `pd.DataFrame` can store mixed types
    C. A `pd.DataFrame` has an index; a `pd.Series` does not
    D. A `pd.Series` is faster to create but cannot be used inside a `pd.DataFrame`

### Correct Answer: A

**Explanation:**
A `pd.Series` is a one-dimensional labeled array representing a single column of data. A `pd.DataFrame` is a two-dimensional structure composed of multiple aligned `pd.Series` objects. In fact, each column of a `pd.DataFrame` is internally stored as a `pd.Series`.

---

## Question 6

You load a large CSV file but only need three of its fifty columns. Which argument helps you avoid loading unnecessary data?

    A. nrows
    B. skiprows
    C. dtype
    D. usecols

### Correct Answer: D

**Explanation:**
`usecols` accepts a list of column names (or positions) and instructs `pd.read_csv()` to parse and load only those columns, significantly reducing memory usage when working with wide files. `nrows` limits rows, not columns.

---

## Question 7

You query a REST API that returns JSON and want to load the response into a DataFrame. Which approach is correct?

    A. pd.read_json(url) directly without any HTTP library
    B. Use pd.read_api(url)
    C. Use requests.get(url) to fetch the response, then pass response.json() to pd.DataFrame()
    D. Use pd.read_sql() with the API endpoint as the connection string

### Correct Answer: C

**Explanation:**
`pandas` has no dedicated API reader. The standard pattern is to use the `requests` library to make the HTTP call, call `response.raise_for_status()` to catch errors, then pass `response.json()` — which returns a Python list or dict — to `pd.DataFrame()` for conversion.

---

## Question 8

Which method gives you the number of rows and columns of a DataFrame as a tuple?

    A. df.size
    B. df.ndim
    C. df.count()
    D. df.shape

### Correct Answer: D

**Explanation:**
`df.shape` returns a tuple `(n_rows, n_columns)`. `df.size` returns the total number of elements (rows × columns). `df.ndim` returns the number of dimensions (always 2 for a DataFrame). `df.count()` returns the number of non-null values per column.

---

## Question 9

What is the correct way to count missing values per column in a DataFrame?

    A. df.missing()
    B. df.isna().sum()
    C. df.null().sum()
    D. df.count_na()

### Correct Answer: B

**Explanation:**
`.isna()` produces a boolean DataFrame where `True` marks each missing value. Chaining `.sum()` then counts the `True` values (each treated as `1`) per column, giving the total number of missing values in each column.

---

## Question 10

When creating a `pd.Series` from a dictionary, what becomes the index?

    A. The dictionary keys
    B. The dictionary values
    C. Integer positions starting from 0
    D. The data type of each value

### Correct Answer: A

**Explanation:**
When you pass a dictionary to `pd.Series()`, the keys are used as the index labels and the values become the data. For example, `pd.Series({'Alice': 25, 'Bob': 30})` produces a Series indexed by `'Alice'` and `'Bob'`.

---

## Question 11

You need to connect to a PostgreSQL database and load query results into a DataFrame. Which combination of tools is correct?

    A. pd.read_csv() with a database cursor
    B. pd.read_excel() with the database hostname
    C. pd.read_sql() with a SQLAlchemy engine as the connection
    D. pd.read_json() with the database URL

### Correct Answer: C

**Explanation:**
`pd.read_sql()` requires a live connection object. For PostgreSQL, the standard approach is to create a SQLAlchemy engine with `create_engine('postgresql+psycopg2://user:password@host/db')` and pass it as the `con` argument. `sqlite3` connections also work for SQLite databases.

---

## Question 12

What does `df.isna().mean()` return?

    A. The mean of all numeric values, excluding missing ones
    B. The proportion of missing values for each column
    C. The total number of missing values across the entire DataFrame
    D. A boolean indicating whether any value is missing

### Correct Answer: B

**Explanation:**
`.isna()` converts each cell to `True` (1) or `False` (0). Calling `.mean()` on that boolean DataFrame calculates the average of `True` values per column, which equals the proportion (fraction) of missing values in each column. A result of `0.4` means 40% of values in that column are missing.

---

## Question 13

Which `pd.read_csv()` argument lets you specify that `-`, `N/A`, and `missing` should all be treated as `NaN`?

    A. fillna=['-', 'N/A', 'missing']
    B. null_values=['-', 'N/A', 'missing']
    C. missing_values=['-', 'N/A', 'missing']
    D. na_values=['-', 'N/A', 'missing']

### Correct Answer: D

**Explanation:**
The `na_values` argument accepts a list of additional strings to recognize as `NaN` during parsing. This is particularly useful when datasets use custom placeholders for missing data instead of standard empty cells.

---

## Question 14

What type of data structure does `pd.read_excel('data.xlsx', sheet_name=None)` return?

    A. A list of DataFrames
    B. A dictionary mapping sheet names to DataFrames
    C. A tuple of DataFrames
    D. A single merged DataFrame

### Correct Answer: B

**Explanation:**
When `sheet_name=None`, `pandas` reads all sheets and returns a Python dictionary where each key is the sheet name (a string) and each value is the corresponding `pd.DataFrame`. You can then iterate over `.items()` or access individual sheets by name.

---

## Question 15

You have a `pd.Series` of country names. Which method returns the most frequently occurring value?

    A. .mode()
    B. .max()
    C. .value_counts()
    D. .describe()

### Correct Answer: A

**Explanation:**
`.mode()` returns the most frequently occurring value(s) in a Series. `.value_counts()` also shows frequencies but returns a full ranked count for every unique value rather than just the top one. `.max()` would return the alphabetically last string, not the most frequent.

---

## Question 16

What is the difference between `.count()` and `.isna().sum()` on a DataFrame column?

    A. They always return the same result
    B. `.count()` counts all rows including NaN; `.isna().sum()` only counts numeric rows
    C. `.count()` is for Series only; `.isna().sum()` works on DataFrames only
    D. `.count()` returns non-missing values; `.isna().sum()` returns the number of missing values

### Correct Answer: D

**Explanation:**
`.count()` returns the number of non-null values in a column, while `.isna().sum()` counts the number of null (missing) values. Together they are complementary: if a column has 100 rows and `.isna().sum()` is 3, then `.count()` will be 97.

---

## Question 17

When loading data from a SQL database using `pd.read_sql()`, what does the `params` argument allow you to do?

    A. Pass values into a parameterized query safely, avoiding SQL injection
    B. Specify which columns to load
    C. Set the data types of the resulting columns
    D. Define the number of rows to fetch

### Correct Answer: A

**Explanation:**
The `params` argument lets you bind values to placeholders in a parameterized SQL query (e.g., `WHERE department = ?`). This is safer than string-formatting values directly into the query, as it prevents SQL injection vulnerabilities.

---

## Question 18

Which of the following correctly describes what `df.describe(include='all')` does differently from `df.describe()`?

    A. It shows memory usage for each column
    B. It also computes statistics for non-numeric columns, adding fields like `unique`, `top`, and `freq`
    C. It returns the same result as `df.info()`
    D. It includes missing values in the count

### Correct Answer: B

**Explanation:**
By default, `df.describe()` only summarizes numeric columns. Passing `include='all'` forces it to also include non-numeric (e.g., string/categorical) columns, adding statistics such as `unique` (number of unique values), `top` (most frequent value), and `freq` (its count).

---

## Question 19

What are the four components that make up every `pd.Series` object?

    A. Data, Shape, Columns, Dtype
    B. Rows, Columns, Index, Name
    C. Data, Labels, Size, Type
    D. Values, Index, Name, Dtype

### Correct Answer: D

**Explanation:**
Every `pd.Series` consists of: **Values** (the underlying data array), **Index** (labels identifying each element), **Name** (an optional label for the Series as a whole, useful when it represents a column in a DataFrame), and **Dtype** (the data type of the stored values).

---

## Question 20

A dataset is stored in a structured format with strict rows and columns (like a table). Which dataset structurality category does this belong to, and which file format is a typical example?

    A. Structured — CSV
    B. Unstructured — TXT
    C. Semi-structured — CSV
    D. Semi-structured — JSON

### Correct Answer: A

**Explanation:**
Structured data follows a strict, predefined schema with clearly defined rows and columns. CSV files are a classic example of structured data, alongside relational databases and Excel files. JSON is typically semi-structured because its schema can vary between records, and plain text files are unstructured.

---

**Score Summary:**

- Total Questions: 20
- Topics Covered: Data types and structurality, `pd.Series` and `pd.DataFrame` fundamentals, DataFrame inspection methods (`.info()`, `.shape`, `.dtypes`, `.describe()`), missing value detection (`.isna()`, `.count()`), frequency statistics (`.value_counts()`, `.mode()`), data loading from CSV, JSON, Excel, SQL, and APIs
- Correct Answers Order: C, B, C, A, A, D, C, D, B, A, C, B, D, B, A, D, A, B, D, A
