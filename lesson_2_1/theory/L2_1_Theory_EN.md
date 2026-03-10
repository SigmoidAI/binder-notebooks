# Lesson 2.1 - Data Loading & Initial Exploration

## Table of Contents

1. [What is Data?](#1-what-is-data)
2. [What is Pandas?](#2-what-is-pandas)
3. [Where to find Datasets?](#3-where-to-find-datasets)
4. [Introduction to `pd.Series`](#4-introduction-to-ser)
5. [`pd.Series` Initial Exploration](#5-expl-series)
6. [Introduction to `pd.DataFrame`](#6-introduction-to-df)
7. [`pd.DataFrame` Initial Exploration](#7-expl-df)
8. [Multi-Format Data Loading](#8-data-loading)

---

<a id="1-what-is-data"></a>

## 1. What is Data?

Data is a raw or basic collection of observations, measurements, facts etc. that is used to represent and describe objects, phenomena and events. On the other hand, datasets are organized pieces of data, that are typically related to a specific domain or have some common logic behind their organization. Data and datasets provide probably the most important and crucial details without which machine learning could not evolve so much over the years. One of the most important things about high-quality data is that it provides a way for machine learning models to "learn", or find underlying patterns and relations between various pieces of data.

Datasets can originate from various sources, such as: experiments, surveys, systems logs, sensors, databases and many others. They vary in size and format. Usually, datasets are organized based on structurality. By this criterion, datasets can be:

1. **Structured** - highly organized/formatted data represented in strict predefined format (with a strict "schema"). Usually, this type of data is represented in Relational Databases, CSV/Excel files and others. Below is represented an example of a dataset organized in tabular format:

<div align="center">

| **ID** | **Age** | **Body_Mass_Index** | **Gender** | **Blood_Type** | **...** | **Is_Sick** |
|:------:|:-------:|:-------------------:|:----------:|:--------------:|:-------:|:-----------:|
|   _1_  |    18   |          40         |     "M"    |      "AB+"     |   ...   |     True    |
|   _2_  |    35   |          27         |    None    |       "O"      |   ...   |    False    |
|   _3_  |    24   |          31         |     "F"    |      None      |   ...   |    False    |
|  _..._ |   ...   |         ...         |     ...    |       ...      |   ...   |     ...     |
| _1000_ |    95   |          35         |     "F"    |      "A-"      |   ...   |     True    |

</div>

2. **Semi-Structured** - data that have some organizational properties, with flexible "schema", that may differ from dataset record to record. Most of the time, this type of datasets is represented in JSON, XML, system logs and other formats. Below is represented an example of such JSON dataset (Note differences in these records):

    ```json
    {
        "name": "John",
        "age": 20,
        "interests": ["Literature", "Music Production"],
        "num_requests": 201,
        "is_subscribed": "True"
    },
    {
        "name": "Jane",
        "age": "None",
        "interests": ["Literature", "Music Production"],
        "is_subscribed": "False",
        "location": "Europe"
    },
    ...
    ```

3. **Unstructured** - data that contain information that is does not conform to traditional data models, with no "schema". This type of datasets can be represented in various ways: videos/audio/text files, chat messages, etc. This category is the hardest to work with, but most of the time, they capture rich insights and information.

<a id="2-what-is-pandas"></a>

## 2. What is `pandas`?

Most of Machine Learning application involve working with structured datasets, however datasets of other two types are also used, but require more preparation and deeper understanding of complex loading and processing mechanisms. For the start, being able to work with datasets involve being able to load them properly into data structures that can be manipulated/actioned on via programming. One of the most popular libraries that offer this possibility is `pandas`. [`pandas`](https://pandas.pydata.org/docs/index.html) (commonly shortened to `pd`) is an open-source data analysis and manipulation tool, built on top of `Python` programming language, that is able to convert data from files into data structures that can be interacted with via code.

`pandas` library offers two main data structures - `pd.DataFrame` and `pd.Series`. They can be interpreted as two-dimensional array = `pd.DataFrame` and one-dimensional array - `pd.Series`. However, they have extensive utils that make them very useful when dealing with datasets.

1. `pd.Series` - one-dimensional labeled array holding data of any type. This structure can be interpreted as a basic list. For example:

<div align="center">

| **ID** |
|:------:|
|   _1_  |
|   _2_  |
|   _3_  |
|  _..._ |
| _1000_ |

</div>

2. `pd.DataFrame` - two-dimensional data structure that holds data like a two-dimensional array. In other words, this is a basic table with rows and columns. For example, recall previous structured dataset example:

<div align="center">

| **ID** | **Age** | **Body_Mass_Index** |
|:------:|:-------:|:-------------------:|
|   _1_  |    18   |          40         |
|   _2_  |    35   |          27         |
|   _3_  |    24   |          31         |
|  _..._ |   ...   |         ...         |
| _1000_ |    95   |          35         |

</div>

Both structures can hold any values, just like python lists, but are compatible with other libraries, like `numpy`, meaning that when instantiating objects of `pd.DataFrame` and `pd.Series`, as their inputs, `numpy` arrays can be used. Don't worry if you are not familiar with this library, next lectures will shed some light on it. As of now, take this as a sign of how `pandas` library has inherent compatibility with other libraries.

<a id="3-where-to-find-datasets"></a>

## 3. Where to find Datasets?

Before starting trying to load any dataset, it is required to first find (or produce) the dataset. This can be done via different services, as well as other sources - experiments, trials, personal observations. Since internet offers a way for people to share information with others, in the domain of Machine Learning, people use it to share different datasets as well, making them public and free-to-use. Take into consideration that, most of the time, to use qualitative and big datasets is required additional confirmation of how the dataset will be used, what are the goals that are trying to be achieved using a particular dataset and so on.

However, for most of the ML applications, public datasets are more than enough. Some of the most influential platforms in sharing and publishing datasets are:

1. [**Kaggle**](https://www.kaggle.com/) - AI/ML Community Web platform that offers a way to share datasets, host and attend competitions, post notebooks with solutions for different tasks and access and study a lot of learning material. In order to view the dedicated pages for datasets, follow the link: [Kaggle Datasets](https://www.kaggle.com/datasets). Beginner-friendly, suitable for both beginners and advanced ML enthusiasts, with high focus on learning and participating in competitions.
2. [**HuggingFace**](https://huggingface.co/) - Similar AI/ML Community Web platform offering a place to host and share datasets, in a similar way [Github](https://github.com/) does (repository structure, file-versioning, etc.), access AI/ML open-source models, share your achievements in ML field and so on. To access the dataset section of the platform, follow the link: [HuggingFace Datasets](https://huggingface.co/datasets). Steeper "learning" curve, with strong focus more on sharing advancements and hosting models, AI/ML apps and datasets, sharing them with the community.
3. [**Github**](https://github.com/) - Although [Github](https://github.com/) is more a cloud-based platform that uses [Git](https://git-scm.com/) as a Version Control System rather than focusing on datasets hosting, people may publish datasets on this platform as well. You may search across different repositories the ones that are specifically stated as for hosting datasets. Try your luck on this platform as well.

<a id="4-introduction-to-ser"></a>

## 4. Introduction to `pd.Series`

`pd.Series` represent a one-dimensional labeled array capable of holding any data type (e.g., _integers_, _floats_, _strings_, _Python objects_, etc.). Conceptually, it can be viewed as a single column of data paired with an index that provides meaningful labels for each observation.

A `pd.Series` consists of several main components:

* **Values** – the underlying data stored in a NumPy-like array structure,
* **Index** – a sequence of labels that uniquely identify each element,
* **Name** - a name of the `pd.Series` object.
* **Dtype** – the data type of the values (e.g., `int64`, `float64`, `object`, etc.).

The labeling mechanism is particularly important during the early stages of data analysis, as it allows data to be accessed, aligned, and filtered based on meaningful identifiers rather than only positional indices.

When loading data from external sources (e.g., _CSV_, _Excel_, _JSON_, _SQL databases_, etc.), each column of the dataset is internally represented as a `pd.Series`. This makes the `pd.Series` object the core abstraction for understanding how tabular data is structured and manipulated in `pandas`.

a `pd.Series` constructor looks the following way:

```py
class pandas.Series(data=None, index=None, dtype=None, name=None, copy=None):
    """
    One-dimensional ndarray with axis labels (including time series).

    ...
    """
```

<div align="center">

| Argument   | Description                                                                                                                  |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `data`     | The actual data of the Series. Can be a list, tuple, NumPy array, dictionary, or scalar.                                     |
| `index`    | Optional labels for each element. Must match the length of `data`. If omitted, integer indices `0..n-1` are used.            |
| `dtype`    | Optional data type for the values (e.g., `int64`, `float64`, `object`, `category`). If not provided, inferred automatically. |
| `name`     | Optional name of the Series. Useful when it represents a column in a DataFrame.                                              |
| `copy`     | Whether to copy the data. Default is `False`.                                                                                |

</div>

In order to instantiate a `pd.Series`, there are many ways. For example, the simplest one:

* Initialize empty `pd.Series`:

```py
import pandas as pd

sample_series: pd.Series = pd.Series()
print(sample_series)
#   OUTPUT:
#       Series([], dtype: object)
```

* Initialize from a list:

```py
import pandas as pd

numbers = [10, 20, 30, 40]
series_from_list = pd.Series(data=numbers)
print(series_from_list)
#   OUTPUT:
#       0    10
#       1    20
#       2    30
#       3    40
#       dtype: int64
```

* Initialize from a list with Custom Index:

```py
import pandas as pd

numbers = [10, 20, 30, 40]
letters = ['a', 'b', 'c', 'd']
series_with_index = pd.Series(data=numbers, index=letters)
print(series_with_index)
#   OUTPUT:
#       a    10
#       b    20
#       c    30
#       d    40
#       dtype: int64
```

* Initialize from a dictionary (`keys` are taken as `pd.Index`, `values` - as row data):

```py
import pandas as pd

data_dict = {'Alice': 25, 'Bob': 30, 'Charlie': 35}
series_from_dict = pd.Series(data_dict)
print(series_from_dict)
#   OUTPUT:
#       Alice      25
#       Bob        30
#       Charlie    35
#       dtype: int64
```

* Specify name and data type:

```py
series_named = pd.Series(data=[1, 2, 3], dtype='float64', name='SAMPLE_SERIES')
print(series_named)
#   OUTPUT:
#       0    1.0
#       1    2.0
#       2    3.0
#       Name: SAMPLE_SERIES, dtype: float64
```

* From a scalar value (the value is copied across all indices):

```py
scalar_series = pd.Series(0, index=['a', 'b', 'c'])
print(scalar_series)
#   OUTPUT:
#       a    0
#       b    0
#       c    0
#       dtype: int64
```

<a id="5-expl-series"></a>

## 5. `pd.Series` Initial Exploration

Since `pd.Series` represents a single variable, the exploration focuses on understanding its values, type, missing data and basic statistics. Analysts frequently investigate these objects via their methods in the following way:

### Inspect variables

```py
import pandas as pd

series_obj: pd.Series = pd.Series(
    data=[i + 1 for i in range(0, 100)],
    name="SAMPLE_DATA"
)
```

* Entire `pd.Series`:

```py
print(series_obj)
#   OUTPUT:
#       0       1
#       1       2
#       2       3
#       3       4
#       4       5
#            ...
#       95     96
#       96     97
#       97     98
#       98     99
#       99    100
#       Name: SAMPLE_DATA, Length: 100, dtype: int64
```

* First `n` rows - `.head()`:

```py
n: int = 3 # first 3 rows
print(series_obj.head(n=n))
#   OUTPUT:
#       0    1
#       1    2
#       2    3
#       Name: SAMPLE_DATA, dtype: int64
```

* Last `n` rows - `.tail()`:

```py
n: int = 3 # first 3 rows
print(series_obj.head(n=n))
#   OUTPUT:
#       97     98
#       98     99
#       99    100
#       Name: SAMPLE_DATA, dtype: int64
```

### Check data types

```py
import pandas as pd

series_obj = pd.Series([1, 2, 3, 4], name="NUMBERS")
```

* Use internal attribute - `.dtype`:

```py
print(series_obj.dtype)
#   OUTPUT:
#       int64
```

### Identify missing values

```py
import numpy as np
import pandas as pd

series_obj = pd.Series(data=[1, np.nan, 3, None, 5], name="WITH_MISSING")
```

* As a `pd.Series` with `bool` values for each row - `.isna()`:

```py
print(series_obj.isna())
#   OUTPUT:
#       0    False
#       1     True
#       2    False
#       3     True
#       4    False
#       Name: WITH_MISSING, dtype: bool
```

* Total number of missing values - `.isna().sum()`:

```py
print(f"NUMBER OF MISSING VALUES: {series_obj.isna().sum()}")
#   OUTPUT:
#       NUMBER OF MISSING VALUES: 2
```

### Compute Numerical Statistics

```py
import pandas as pd

numeric_series = pd.Series(
    data=[10, 10, 10, 20, 30, 100, 0, -100, -20, 20, 10, 45, 20, 30, 40, 50], 
    name="NUMERIC_STATS"
)
```

* Arithmetic Mean and Median values - `.mean()` and `.median()`:

```py
print(f"ARITHMETIC MEAN: {numeric_series.mean()}")
print(f"MEDIAN VALUE: {numeric_series.median()}")
#   OUTPUT:
#       ARITHMETIC MEAN: 17.1875
#       MEDIAN VALUE: 20.0
```

* Maximum and Minimum values - `.min()` and `.max()`:

```py
print(f"MINIMUM VALUE: {numeric_series.min()}")
print(f"MAXIMUM VALUE: {numeric_series.max()}")
#   OUTPUT:
#       MINIMUM VALUE: -100
#       MAXIMUM VALUE: 100
```

* Sum of all values - `.sum()`:

```py
print(f"SUM OF VALUES: {numeric_series.sum()}")
#   OUTPUT:
#       SUM OF VALUES: 275
```

* Number of Non-Missing values - `.count()`:

```py
print(f"NUMBER OF NON-MISSING VALUES: {numeric_series.count()}")
#   OUTPUT:
#       NUMBER OF NON-MISSING VALUES: 16
```

* Standard Deviation and Variance - `.std()` and `.var()`:

```py
print(f"STANDARD DEVIATION VALUE: {numeric_series.std()}")
print(f"VARIANCE VALUE: {numeric_series.var()}")
#   OUTPUT:
#       STANDARD DEVIATION VALUE: 40.98653234092064
#       VARIANCE VALUE: 1679.8958333333333
```

* Quantiles and Distribution insights - `.quantile()`:

```py
print(f"Q1 (25th percentile): {numeric_series.quantile(0.25)}")
print(f"Q2 (50th percentile / median): {numeric_series.quantile(0.50)}")
print(f"Q3 (75th percentile): {numeric_series.quantile(0.75)}")
#   OUTPUT:
#       Q1 (25th percentile): 10.0
#       Q2 (50th percentile / median): 20.0
#       Q3 (75th percentile): 32.5
```

* Aggregate multiple numeric statistics - `.agg()`:

```py
print("\nAGGREGATED NUMERIC STATS:")
print(numeric_series.agg(["mean", "median", "min", "max", "std", "count"]))
#   OUTPUT:
#       AGGREGATED NUMERIC STATS:
#       mean       17.187500
#       median     20.000000
#       min      -100.000000
#       max       100.000000
#       std        40.986532
#       count      16.000000
#       Name: NUMERIC_STATS, dtype: float64
```

* Compute Statistics Summary (yield previously mentioned statistics)

```py
import numpy as np
import pandas as pd

numeric_series = pd.Series(
    data=[22.5, 27.2, 31.8, np.nan, 28.6, 29.0, None, 32.1, 35.0],
    name="NUMERIC_SERIES"
)
```

* Compute summary statistics - `.describe()`:

```py
print(f"SUMMARY STATISTICS: {numeric_series.describe()}")
#   OUTPUT:
#       SUMMARY STATISTICS:
#       count     7.000000
#       mean     29.457143
#       std       4.033963
#       min      22.500000
#       25%      27.900000
#       50%      29.000000
#       75%      31.950000
#       max      35.000000
#       Name: NUMERIC_SERIES, dtype: float64
```

### Compute Frequency-based Statistics (valid for non-numeric data)

```py
import pandas as pd

categorical_series = pd.Series(
    ["CANADA", "AUSTRALIA", "USA", "JAPAN", "USA", "CANADA", "JAPAN", "JAPAN"],
    name="COUNTRY_STATS"
)
```

* Frequency of Unique Values - `.value_counts()`:

```py
print(f"FREQUENCY OF UNIQUE VALUES:\n{categorical_series.value_counts()}")
#   OUTPUT:
#       FREQUENCY OF UNIQUE VALUES:
#       COUNTRY_STATS
#       JAPAN        3
#       CANADA       2
#       USA          2
#       AUSTRALIA    1
#       Name: count, dtype: int64 
```

* Mode (most frequent value(s)) - `.mode()`:

```py
print(f"\nMOST FREQUENT VALUE(S):\n{categorical_series.mode()}")
#   OUTPUT:
#       MOST FREQUENT VALUE(S):        
#       0    JAPAN
#       Name: COUNTRY_STATS, dtype: str
```

* Number of Non-Missing Values - `.count()`:

```py
print(f"\nNUMBER OF NON-MISSING VALUES: {categorical_series.count()}")
#   OUTPUT:
#       NUMBER OF NON-MISSING VALUES: 8
```

* Number of Unique Values - `.nunique()`:

```py
print(f"NUMBER OF UNIQUE VALUES: {categorical_series.nunique()}")
#   OUTPUT:
#       NUMBER OF UNIQUE VALUES: 4
```

* Compute Statistics Summary (yield previously mentioned statistics)

```py
import numpy as np
import pandas as pd

mixed_series = pd.Series(
    data=[10, "CANADA", 30, 40, np.nan, None, 60, "USA", 80],
    name="MIXED_SERIES"
)
```

* Compute summary statistics - `.describe()`:

```py
print(f"SUMMARY STATISTICS: {mixed_series.describe()}")
#   OUTPUT:
#       SUMMARY STATISTICS:
#       count      7
#       unique     7
#       top       10
#       freq       1
#       Name: MIXED_SERIES, dtype: int64
```

### Find out Memory Occupied by `pd.Series` or `pd.DataFrame`

```py
import numpy as np
import pandas as pd

mixed_series = pd.Series(
    data=[10, "CANADA", 30, 40, np.nan, None, 60, "USA", 80],
    name="MIXED_SERIES"
)
```

* Compute memory consumption - `.memory_usage()`:

```py
print(f"MEMORY OCCUPIED BY mixed_series: {mixed_series.memory_usage()} bytes")
#   OUTPUT:
#       MEMORY OCCUPIED BY mixed_series: 204 bytes
```

Because `pd.Series` objects support vectorized operations, they enable efficient exploratory computations without explicit loops. This capability is essential when performing quick diagnostics on newly loaded datasets, such as: detecting anomalies, validating ranges or standardizing values before further processing (you will get to this in further lessons).

<a id="6-introduction-to-df"></a>

## 6. Introduction to `pd.DataFrame`

While a `pd.Series` models a single column, the primary structure used for data loading and exploratory analysis is the `pd.DataFrame`. A `pd.DataFrame` is a two-dimensional, labeled data structure composed of multiple aligned `pd.Series` objects sharing the same index. It can be thought of as an in-memory table where rows represent observations and columns represent variables.

Each `pd.DataFrame` contains:

* **Values** – stored as a collection of `pd.Series`, containing actual values in tabular form (rows and columns).
* **Row Index** – the row labels that uniquely identify each observation.
* **Column Labels** – the column names that identify each variable or feature.
* **Dtypes** – the data types of each column (e.g., `int64`, `float64`, `object`, etc.).

This tabular representation makes the `pd.DataFrame` ideal for importing and exploring real-world datasets. When data is read from files or databases (e.g., _CSV_, _Excel_, _JSON_, _SQL databases_, etc.), `pandas` automatically organizes the information into a `pd.DataFrame`, preserving column names and attempting to infer appropriate data types.

The `pd.DataFrame` constructor looks the following way:

```py
class pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None):
    """
    Two-dimensional, size-mutable, potentially heterogeneous tabular data.

    ...
    """
```

<div align="center">

| Argument  | Description                                                                                                                                |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `data`    | The actual data of the DataFrame. Can be a dict of lists/arrays/Series, a 2D NumPy array, another DataFrame, or a list of dicts.           |
| `index`   | Optional row labels. Must match the number of rows in `data`. If omitted, integer indices `0..n-1` are used.                               |
| `columns` | Optional column labels. If `data` is a dict, this can be used to reorder or select a subset of columns.                                    |
| `dtype`   | Optional data type to force on all columns. If not provided, each column's type is inferred automatically.                                 |
| `copy`    | Whether to copy the input data. Default is `False`.                                                                                        |

</div>

There are many ways to instantiate a `pd.DataFrame`. For example:

* Initialize an empty `pd.DataFrame`:

```py
import pandas as pd

empty_df: pd.DataFrame = pd.DataFrame()
print(empty_df)
#   OUTPUT:
#       Empty DataFrame
#       Columns: []
#       Index: []
```

* Initialize from a dictionary of lists (`keys` become column labels, `values` become column data):

```py
import pandas as pd

data_dict = {
    'name':   ['Alice', 'Bob', 'Charlie'],
    'age':    [25, 30, 35],
    'salary': [50000.0, 60000.0, 70000.0]
}
df_from_dict = pd.DataFrame(data=data_dict)
print(df_from_dict)
#   OUTPUT:
#             name  age   salary
#       0    Alice   25  50000.0
#       1      Bob   30  60000.0
#       2  Charlie   35  70000.0
```

* Initialize from a list of dictionaries (each dict represents one row):

```py
import pandas as pd

rows = [
    {'name': 'Alice',   'age': 25},
    {'name': 'Bob',     'age': 30},
    {'name': 'Charlie', 'age': 35},
]
df_from_records = pd.DataFrame(data=rows)
print(df_from_records)
#   OUTPUT:
#             name  age
#       0    Alice   25
#       1      Bob   30
#       2  Charlie   35
```

* Initialize with a custom row index:

```py
import pandas as pd

data_dict = {'score': [88, 95, 72], 'grade': ['B', 'A', 'C']}
custom_index = ['student_1', 'student_2', 'student_3']
df_custom_index = pd.DataFrame(data=data_dict, index=custom_index)
print(df_custom_index)
#   OUTPUT:
#                  score grade
#       student_1     88     B
#       student_2     95     A
#       student_3     72     C
```

* Initialize from a 2D NumPy array with explicit column names:

```py
import numpy as np
import pandas as pd

array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df_from_array = pd.DataFrame(data=array_2d, columns=['col_A', 'col_B', 'col_C'])
print(df_from_array)
#   OUTPUT:
#          col_A  col_B  col_C
#       0      1      2      3
#       1      4      5      6
#       2      7      8      9
```

* Initialize from a collection of `pd.Series` objects (each `Series` becomes a column, aligned by index):

```py
import pandas as pd

s1 = pd.Series([10, 20, 30], name='col_X')
s2 = pd.Series([40, 50, 60], name='col_Y')
df_from_series = pd.DataFrame(data={s1.name: s1, s2.name: s2})
print(df_from_series)
#   OUTPUT:
#          col_X  col_Y
#       0     10     40
#       1     20     50
#       2     30     60
```

<a id="7-expl-df"></a>

## 7. `pd.DataFrame` Initial Exploration

Since a `pd.DataFrame` is a two-dimensional structure, its exploration involves understanding not only the values but also the shape, column types, missing data patterns, and summary statistics across multiple variables simultaneously. Analysts typically investigate `pd.DataFrame` objects in the following ways:

### Inspect the DataFrame

```py
import numpy as np
import pandas as pd

df: pd.DataFrame = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 22],
    'salary': [50000.0, 60000.0, 70000.0, 55000.0, 48000.0],
    'department': ['HR', 'Engineering', 'Engineering', 'Marketing', 'HR'],
    'score': [88, 95, np.nan, 76, None],
})
```

* Entire `pd.DataFrame`:

```py
print(df)
#   OUTPUT:
#             name  age   salary   department  score
#       0    Alice   25  50000.0           HR   88.0
#       1      Bob   30  60000.0  Engineering   95.0
#       2  Charlie   35  70000.0  Engineering    NaN
#       3    Diana   28  55000.0    Marketing   76.0
#       4      Eve   22  48000.0           HR    NaN
```

* First `n` rows – `.head()`:

```py
n: int = 3
print(df.head(n=n))
#   OUTPUT:
#             name  age   salary   department  score
#       0    Alice   25  50000.0           HR   88.0
#       1      Bob   30  60000.0  Engineering   95.0
#       2  Charlie   35  70000.0  Engineering    NaN
```

* Last `n` rows – `.tail()`:

```py
n: int = 3
print(df.tail(n=n))
#   OUTPUT:
#             name  age   salary   department  score
#       2  Charlie   35  70000.0  Engineering    NaN
#       3    Diana   28  55000.0    Marketing   76.0
#       4      Eve   22  48000.0           HR    NaN
```

### Check Shape and Structure

* Dimensions of the DataFrame (rows, columns) – `.shape`:

```py
print(f"SHAPE: {df.shape}")
#   OUTPUT:
#       SHAPE: (5, 5)
```

* Column names – `.columns`:

```py
print(f"COLUMNS: {df.columns}")
print(f"COLUMNS AS A LIST: {df.columns.tolist()}")
#   OUTPUT:
#       COLUMNS: Index(['name', 'age', 'salary', 'department', 'score'], dtype='str')
#       COLUMNS AS A LIST: ['name', 'age', 'salary', 'department', 'score']
```

* Data type of each column – `.dtypes`:

```py
print(df.dtypes)
#   OUTPUT:
#       name              str
#       age             int64
#       salary        float64
#       department        str
#       score         float64
#       dtype: object
```

* Concise structural summary of the DataFrame – `.info()`:

This method is very useful in ML applications (as well as others). It displays complex information about structural components of the `pd.DataFrame`, such as number of rows, columns, data types, non-null elements count and memory usage.

```py
df.info()
#   OUTPUT:
#       <class 'pandas.DataFrame'>
#       RangeIndex: 5 entries, 0 to 4
#       Data columns (total 5 columns):
#        #   Column      Non-Null Count  Dtype
#       ---  ------      --------------  -----
#        0   name        5 non-null      str
#        1   age         5 non-null      int64
#        2   salary      5 non-null      float64
#        3   department  5 non-null      str
#        4   score       3 non-null      float64
#       dtypes: float64(2), int64(1), str(2)
#       memory usage: 332.0 bytes
```

### Identify Missing Values

* Boolean mask of missing values per cell – `.isna()`:

```py
print(df.isna())
#   OUTPUT:
#           name    age  salary  department  score
#       0  False  False   False       False  False
#       1  False  False   False       False  False
#       2  False  False   False       False   True
#       3  False  False   False       False  False
#       4  False  False   False       False   True
```

* Total number of missing values per column – `.isna().sum()`. This results in a `pd.Series` in which each Index is a column name from the `pd.DataFrame`:

```py
print(df.isna().sum())
#   OUTPUT:
#       name          0
#       age           0
#       salary        0
#       department    0
#       score         2
#       dtype: int64
```

* Proportion of missing values per column – `.isna().mean()`. This demonstrates that multiple operations may be stacked on top of each other, chaining them together, since `.isna()` results in a `pd.Series` and this object has `.mean()` method, as you saw in previous sections:

```py
print(df.isna().mean())
#   OUTPUT:
#       name          0.0
#       age           0.0
#       salary        0.0
#       department    0.0
#       score         0.4
#       dtype: float64
```

### Compute Numerical Statistics

* Aggregate summary statistics across all numeric columns – `.describe()`:

```py
print(df.describe())
#   OUTPUT:
#                    age        salary      score
#       count   5.000000      5.000000   3.000000
#       mean   28.000000  56600.000000  86.333333
#       std     4.949747   8820.430828   9.609024
#       min    22.000000  48000.000000  76.000000
#       25%    25.000000  50000.000000  82.000000
#       50%    28.000000  55000.000000  88.000000
#       75%    30.000000  60000.000000  91.500000
#       max    35.000000  70000.000000  95.000000
```

* Arithmetic mean of each numeric column – `.mean()`. This operation, as well as other numerical operations, is supported only for the columns that have numbers as values - with string or any other non-number value, it will raise an Error:

```py
print(df[['age', 'salary', 'score']].mean())
#   OUTPUT:
#       age          28.000000
#       salary    56600.000000
#       score        86.333333
#       dtype: float64
```

* Minimum and maximum of each numeric column – `.min()` and `.max()`:

```py
print(df[['age', 'salary', 'score']].min())
print(df[['age', 'salary', 'score']].max())
#   OUTPUT (min):
#       age          22.0
#       salary    48000.0
#       score        76.0
#       dtype: float64   
#   OUTPUT (max):
#       age          35.0
#       salary    70000.0
#       score        95.0
#       dtype: float64
```

### Compute Frequency-based Statistics (valid for non-numeric columns)

* Frequency of unique values in a categorical column – `.value_counts()`:

```py
print(df['department'].value_counts())
#   OUTPUT:
#       department
#       HR             2
#       Engineering    2
#       Marketing      1
#       Name: count, dtype: int64
```

* Number of unique values per column – `.nunique()`:

```py
print(df.nunique())
#   OUTPUT:
#       name          5
#       age           5
#       salary        5
#       department    3
#       score         3
#       dtype: int64
```

* Summary statistics for all columns including non-numeric ones – `.describe(include='all')`:

```py
print(df.describe(include='all'))
#   OUTPUT:
#                name        age        salary department      score
#       count       5   5.000000      5.000000          5   3.000000
#       unique      5        NaN           NaN          3        NaN
#       top     Alice        NaN           NaN         HR        NaN
#       freq        1        NaN           NaN          2        NaN
#       mean      NaN  28.000000  56600.000000        NaN  86.333333
#       std       NaN   4.949747   8820.430828        NaN   9.609024
#       min       NaN  22.000000  48000.000000        NaN  76.000000
#       25%       NaN  25.000000  50000.000000        NaN  82.000000
#       50%       NaN  28.000000  55000.000000        NaN  88.000000
#       75%       NaN  30.000000  60000.000000        NaN  91.500000
#       max       NaN  35.000000  70000.000000        NaN  95.000000
```

### Find out Memory Occupied by `pd.DataFrame`

* Compute total memory consumption – `.memory_usage()`:

```py
print(df.memory_usage(deep=True))
#   OUTPUT:
#       Index         132
#       name          268
#       age            40
#       salary         40
#       department    280
#       score          40
#       dtype: int64
```

* Compute total memory across all columns:

```py
print(f"TOTAL MEMORY: {df.memory_usage(deep=True).sum()} bytes")
#   OUTPUT:
#       TOTAL MEMORY: 800 bytes
```

Because `pd.DataFrame` objects support vectorized operations across all columns simultaneously, they enable efficient exploratory computations without explicit loops. This capability is essential when performing quick diagnostics on newly loaded datasets, such as detecting anomalies, validating column types, identifying missing data patterns, or computing distributions before further processing.

<a id="8-data-loading"></a>

## 8. Multi-Format Data Loading

Besides previously mentioned methods of instantiation of `pd.DataFrame` objects, there is also another way to create such object - **data loading**.

`pandas` supports reading data from a wide variety of external formats and sources directly into a `pd.DataFrame`:

* **CSV** (`.csv`) or any other Delimiter-Separated Values format - `pd.read_csv()`,
* **JSON** (`.json`, `.jsonl`, etc.) - `pd.read_json()`,
* **Excel** (`.xlsx`, `.xls`, `.xlsb`, `.xlsm`, etc.) - `pd.read_excel()`,
* **SQL** databases (via a connection or SQL query) - `pd.read_sql()`,
* **APIs** (via HTTP responses, typically JSON-based).

Each format has a dedicated `pandas` reader function that handles parsing, type inference, and index assignment automatically. However, it is possible to handle manually, using arguments in appropriate `pandas` function.

### 8.1 Loading CSV and Delimiter-Separated Files

CSV (Comma-Separated Values) is the most common flat-file format for tabular data. `pandas` reads it using `pd.read_csv()`:

```py
pandas.read_csv(filepath_or_buffer, sep=',', header='infer', names=None,
                index_col=None, usecols=None, dtype=None, na_values=None,
                skiprows=None, nrows=None, encoding='utf-8', ...)
```

<div align="center">

| Argument              | Description                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| `filepath_or_buffer`  | Path to the file, a URL, or any file-like object.                                                        |
| `sep`                 | Delimiter between fields. Default is `','`. Use `'\t'` for TSV files.                                    |
| `header`              | Row number(s) to use as column names. Default `'infer'` uses the first row.                              |
| `names`               | Explicit list of column names to assign. Useful when the file has no header row.                         |
| `index_col`           | Column(s) to use as the row index. Can be a column name or integer position.                             |
| `usecols`             | Subset of columns to load. Reduces memory usage on wide files.                                           |
| `dtype`               | Dictionary mapping column names to explicit data types.                                                  |
| `na_values`           | Additional strings to recognize as `NaN` (e.g., `['N/A', 'missing', '-']`).                              |
| `skiprows`            | Number of rows (or list of row indices) to skip at the start of the file.                                |
| `nrows`               | Maximum number of rows to read. Useful for previewing large files.                                       |
| `encoding`            | File encoding (e.g., `'utf-8'`, `'latin-1'`). Default is `'utf-8'`.                                      |

</div>

* Basic usage - read a CSV file with default settings:

```py
import pandas as pd

df = pd.read_csv('data/employees.csv')
print(df.head())
#   OUTPUT:
#             name  age   salary   department
#       0    Alice   25  50000.0           HR
#       1      Bob   30  60000.0  Engineering
#       2  Charlie   35  70000.0  Engineering
#       3    Diana   28  55000.0    Marketing
#       4      Eve   22  48000.0           HR
```

* Load only specific columns to reduce memory usage:

```py
df = pd.read_csv('data/employees.csv', usecols=['name', 'salary'])
print(df.head(3))
#   OUTPUT:
#             name   salary
#       0    Alice  50000.0
#       1      Bob  60000.0
#       2  Charlie  70000.0
```

* Load a TSV (Tab-Separated Values) file (or any other Delimiter-Spaced Values file - specify the separator that is used in the file as an argument to the function):

```py
df = pd.read_csv('data/employees.tsv', sep='\t')
print(df.head(3))
#             name   salary
#       0     John  10000.0
#       1   Daniel  20000.0
#       2     Rick  40000.0
```

* Preview a large file without loading it fully (for large datasets, there will be a better way to handle loading using other libraries, since `pandas` loads everything into RAM directly, which may not be enough - this will be covered in future lessons):

```py
df_preview = pd.read_csv('data/large_dataset.csv', nrows=100)
print(f"Preview shape: {df_preview.shape}")
#   OUTPUT:
#       Preview shape: (100, 4)
```

* Specify a column as the row index and define explicit data types:

```py
df = pd.read_csv(
    'data/employees.csv',
    index_col='name',
    dtype={'age': 'int32', 'salary': 'float32'}
)
print(df.dtypes)
#   OUTPUT:
#       age       int32
#       salary    float32
#       ...
```

### 8.2 Loading JSON Files

JSON (JavaScript Object Notation) is a widely used format for structured and semi-structured data, especially in web and API contexts. `pandas` reads it using `pd.read_json()`:

```py
pandas.read_json(path_or_buf, orient=None, dtype=None, encoding='utf-8',
                 lines=False, ...)
```

<div align="center">

| Argument       | Description                                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `path_or_buf`  | Path to the file, a URL, or a JSON string.                                                                               |
| `orient`       | Expected format of the JSON data. Common values: `'records'`, `'columns'`, `'index'`, `'values'`, `'split'`.             |
| `dtype`        | Dictionary of column name to data type mappings, or `False` to disable type inference.                                   |
| `encoding`     | File encoding. Default is `'utf-8'`.                                                                                     |
| `lines`        | If `True`, reads the file as newline-delimited JSON (`.jsonl`), where each line is a separate JSON object.               |

</div>

The `orient` parameter is the most important one to understand, as JSON can represent tabular data in several different structural layouts:

<div align="center">

| `orient`      | Expected JSON Structure                                                                 |
| ------------- | --------------------------------------------------------------------------------------- |
| `'records'`   | List of row dicts: `[{"col": val, ...}, ...]`                                           |
| `'columns'`   | Dict of column arrays: `{"col": {index: val, ...}, ...}`                                |
| `'index'`     | Dict of row dicts keyed by index: `{index: {"col": val, ...}, ...}`                     |
| `'values'`    | Bare 2D array of values (no column or index labels).                                    |
| `'split'`     | Dict with separate keys for `index`, `columns`, and `data`.                             |

</div>

* Load a standard records-oriented JSON file:

```py
import pandas as pd

df = pd.read_json('data/employees.json', orient='records')
print(df.head(3))
#   OUTPUT:
#             name  age   salary   department
#       0    Alice   25  50000.0           HR
#       1      Bob   30  60000.0  Engineering
#       2  Charlie   35  70000.0  Engineering
```

* Load a newline-delimited JSON file (`.jsonl`), where each line is one record:

```py
df = pd.read_json('data/employees.jsonl', lines=True)
print(df.head(3))
#   OUTPUT:
#             name  age   salary
#       0    Alice   25  50000.0
#       1      Bob   30  60000.0
#       2  Charlie   35  70000.0
```

* Parse a raw JSON string directly (useful when working with API responses):

```py
import json
import pandas as pd

json_string = '[{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]'
df = pd.read_json(json_string, orient='records')
print(df)
#   OUTPUT:
#           name  age
#       0  Alice   25
#       1    Bob   30
```

### 8.3 Loading Excel Files

Excel files are common in business and research contexts. `pandas` reads them using `pd.read_excel()`, which supports multiple engines and sheet selection:

```py
pandas.read_excel(io, sheet_name=0, header=0, names=None, index_col=None,
                  usecols=None, dtype=None, na_values=None, skiprows=None,
                  nrows=None, engine=None, ...)
```

<div align="center">

| Argument      | Description                                                                                                               |
| ------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `io`          | Path to the file or a URL.                                                                                                |
| `sheet_name`  | Sheet to read. Accepts a name (`'Sheet1'`), integer index (`0`), or `None` to load all sheets as a dict of DataFrames.    |
| `header`      | Row number to use as column names. Default is `0` (first row).                                                            |
| `names`       | Explicit list of column names to assign.                                                                                  |
| `index_col`   | Column to use as the row index.                                                                                           |
| `usecols`     | Subset of columns to load. Accepts column names, integer positions, or an Excel-style range like `'A:D'`.                 |
| `dtype`       | Dictionary mapping column names to data types.                                                                            |
| `na_values`   | Additional strings to interpret as `NaN`.                                                                                 |
| `skiprows`    | Rows to skip at the beginning of the sheet.                                                                               |
| `nrows`       | Maximum number of rows to read.                                                                                           |
| `engine`      | Parsing engine to use: `'openpyxl'` (default for `.xlsx`), `'xlrd'` (for legacy `.xls`), `'calamine'`, etc.               |

</div>

> **Note:** Reading Excel files requires an optional dependency. Install it with `pip install openpyxl` for `.xlsx` files or `pip install xlrd` for legacy `.xls` files.

* Basic usage - read the first sheet of an Excel file:

```py
import pandas as pd

df = pd.read_excel('data/employees.xlsx')
print(df.head(3))
#   OUTPUT:
#             name  age   salary   department
#       0    Alice   25  50000.0           HR
#       1      Bob   30  60000.0  Engineering
#       2  Charlie   35  70000.0  Engineering
```

* Read a specific sheet by name:

```py
df = pd.read_excel('data/employees.xlsx', sheet_name='Q1_Data')
```

* Read all sheets at once - returns a dictionary of DataFrames keyed by sheet name:

```py
all_sheets: dict = pd.read_excel('data/employees.xlsx', sheet_name=None)

for sheet_name, sheet_df in all_sheets.items():
    print(f"Sheet: {sheet_name}, Shape: {sheet_df.shape}")
#   OUTPUT:
#       Sheet: Q1_Data, Shape: (50, 4)
#       Sheet: Q2_Data, Shape: (48, 4)
#       Sheet: Q3_Data, Shape: (55, 4)
```

* Select specific columns using an Excel-style column range:

```py
df = pd.read_excel('data/employees.xlsx', usecols='A:C')
```

### 8.4 Loading from SQL Databases

`pandas` can query relational databases directly and load the results as a `pd.DataFrame` using `pd.read_sql()`, `pd.read_sql_query()`, or `pd.read_sql_table()`. Take into consideration that, as of now, `pandas` suggests using `pd.read_sql()`, that supports both table name and SQL queries (`pd.read_sql()` is a wrapper around `pd.read_sql_query()` and `pd.read_sql_table()`, and reroutes to proper function automatically). A database connection object is required and can be created using libraries such as `sqlite3`, `sqlalchemy`, `psycopg2` or others.

```py
pandas.read_sql(sql, con, index_col=None, params=None, dtype=None, ...)
```

<div align="center">

| Argument    | Description                                                                                                        |
| ----------- | ------------------------------------------------------------------------------------------------------------------ |
| `sql`       | SQL query string or table name to execute/read.                                                                    |
| `con`       | Database connection object (e.g., a `sqlalchemy` engine or a `sqlite3` connection).                                |
| `index_col` | Column(s) to set as the row index.                                                                                 |
| `params`    | Parameters to bind to the SQL query (for parameterized queries).                                                   |
| `dtype`     | Dictionary of column name to data type mappings.                                                                   |

</div>

* Query a SQLite database using `sqlite3`:

```py
import sqlite3
import pandas as pd

connection = sqlite3.connect('data/company.db')

df = pd.read_sql(sql="SELECT * FROM employees", con=connection)
print(df.head(3))
#   OUTPUT:
#             name  age   salary   department
#       0    Alice   25  50000.0           HR
#       1      Bob   30  60000.0  Engineering
#       2  Charlie   35  70000.0  Engineering

connection.close()
```

* Use a parameterized query to filter results safely:

```py
import sqlite3
import pandas as pd

connection = sqlite3.connect('data/company.db')

query = "SELECT name, salary FROM employees WHERE department = ?"
df = pd.read_sql(sql=query, con=connection, params=('Engineering',))
print(df)
#   OUTPUT:
#             name   salary
#       0      Bob  60000.0
#       1  Charlie  70000.0

connection.close()
```

* Connect to a PostgreSQL database using `sqlalchemy`:

```py
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql+psycopg2://user:password@localhost:5432/company_db')

df = pd.read_sql(sql="SELECT * FROM employees LIMIT 5", con=engine)
print(df.head())
```

### 8.5 Loading from APIs

APIs typically return data in JSON format over HTTP. While `pandas` has no dedicated `read_api()` function, the standard workflow combines the `requests` library for fetching data and `pd.DataFrame` or `pd.read_json()` for parsing the response.

> **Note:** The `requests` library is not bundled with `pandas`. Install it with `pip install requests`.

* Fetch JSON data from a REST API and load it into a DataFrame:

```py
import requests
import pandas as pd

url = 'https://jsonplaceholder.typicode.com/users'
response = requests.get(url)
response.raise_for_status()  # raises an error for non-2xx HTTP responses

df = pd.DataFrame(response.json())
print(df[['id', 'name', 'email', 'phone']].head(3))
#   OUTPUT:
#          id              name               email           phone
#       0   1     Leanne Graham   Sincere@april.biz  1-770-736-8031
#       1   2      Ervin Howell   Shanna@melissa.tv    010-692-6593
#       2   3  Clementine Bauch  Nathan@yesenia.net  1-463-123-4447
```

* Pass query parameters to filter the API response:

```py
import requests
import pandas as pd

url = 'https://jsonplaceholder.typicode.com/posts'
params = {'userId': 1}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json())
print(df[['id', 'title']].head(3))
#   OUTPUT:
#          id                                              title
#       0   1   sunt aut facere repellat provident occaecati ...
#       1   2                                       qui est esse
#       2   3  ea molestias quasi exercitationem repellat qui...
```

* Include authentication headers for protected APIs:

```py
import requests
import pandas as pd

url = 'https://api.example.com/v1/sales'
headers = {'Authorization': 'Bearer <YOUR_API_TOKEN>'}

response = requests.get(url, headers=headers)
response.raise_for_status()

df = pd.DataFrame(response.json()['data'])
print(df.head(3))
```

### Summary

The table below provides a quick reference for the loading function associated with each format:

<div align="center">

| Format       | Function               | Common Arguments                                             |
| ------------ | ---------------------- | ------------------------------------------------------------ |
| CSV / TSV    | `pd.read_csv()`        | `sep`, `usecols`, `dtype`, `na_values`, `nrows`              |
| JSON / JSONL | `pd.read_json()`       | `orient`, `lines`, `dtype`                                   |
| Excel        | `pd.read_excel()`      | `sheet_name`, `usecols`, `engine`, `nrows`                   |
| SQL          | `pd.read_sql()`        | `sql`, `con`, `index_col`, `params`                          |
| API          | `requests` + DataFrame | `response.json()`, `pd.DataFrame(...)`, `raise_for_status()` |

</div>

## Cheatsheet

| Concept                   | One-line reminder                                                                             |
|---------------------------|-----------------------------------------------------------------------------------------------|
| Data                      | Raw observations or measurements; datasets are organized, domain-specific collections of data |
| Structured data           | Tabular, schema-enforced format - rows and columns (CSV, Excel, SQL)                          |
| Semi-structured data      | Flexible schema that may vary per record - JSON, XML, logs                                    |
| Unstructured data         | No schema at all - text, images, audio, video                                                 |
| `pandas`                  | Python library for loading, representing, and manipulating tabular data                       |
| `pd.Series`               | One-dimensional labeled array - represents a single column of data                            |
| `pd.DataFrame`            | Two-dimensional labeled table - a collection of aligned `pd.Series` objects                   |
| `pd.Series` components    | Values, Index, Name, Dtype                                                                    |
| `pd.DataFrame` components | Values, Row Index, Column Labels, Dtypes                                                      |
| Series exploration        | `.head()`, `.tail()`, `.dtype`, `.isna()`, `.describe()`, `.value_counts()`                   |
| DataFrame exploration     | `.head()`, `.tail()`, `.shape`, `.columns`, `.dtypes`, `.info()`, `.describe()`               |
| Missing values            | `.isna().sum()` counts missing; `.isna().mean()` gives proportion per column                  |
| Numerical statistics      | `.mean()`, `.median()`, `.std()`, `.var()`, `.min()`, `.max()`, `.quantile()`                 |
| Categorical statistics    | `.value_counts()`, `.mode()`, `.nunique()`                                                    |
| Memory usage              | `.memory_usage(deep=True)` - use `.sum()` for total across all columns                        |
| `pd.read_csv()`           | Reads CSV / TSV; key args: `sep`, `usecols`, `dtype`, `na_values`, `nrows`                    |
| `pd.read_json()`          | Reads JSON / JSONL; key args: `orient`, `lines` - **match `orient` to your file layout**      |
| `pd.read_excel()`         | Reads Excel sheets; `sheet_name=None` loads all sheets as a dict of DataFrames                |
| `pd.read_sql()`           | Executes a SQL query via a connection object (`sqlite3`, `sqlalchemy`, etc.)                  |
| API loading               | No dedicated reader - use `requests.get()` + `pd.DataFrame(response.json())`                  |
| Dataset platforms         | Kaggle (competitions + learning), HuggingFace (models + datasets), GitHub (repos)             |

_Next: open the exercise notebooks to practice these ideas hands-on._
