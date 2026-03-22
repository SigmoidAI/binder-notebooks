# Lecția 2.1 - Încărcarea Datelor și Explorarea Inițială

## Cuprins

1. [Ce sunt Datele?](#1-what-is-data)
2. [Ce este Pandas?](#2-what-is-pandas)
3. [Unde găsim Seturi de Date?](#3-where-to-find-datasets)
4. [Introducere în `pd.Series`](#4-introduction-to-ser)
5. [Explorarea Inițială a `pd.Series`](#5-expl-series)
6. [Introducere în `pd.DataFrame`](#6-introduction-to-df)
7. [Explorarea Inițială a `pd.DataFrame`](#7-expl-df)
8. [Încărcarea Datelor în Formate Multiple](#8-data-loading)

---

<a id="1-what-is-data"></a>

## 1. Ce sunt Datele?

Datele reprezintă o colecție brută sau de bază de observații, măsurători, fapte etc., utilizată pentru a reprezenta și descrie obiecte, fenomene și evenimente. Pe de altă parte, seturile de date sunt colecții organizate de date, de obicei legate de un domeniu specific sau cu o logică comună în spatele organizării lor. Datele și seturile de date furnizează probabil cele mai importante și cruciale informații fără de care machine learning-ul nu ar fi putut evolua atât de mult de-a lungul anilor. Unul dintre cele mai importante aspecte ale datelor de înaltă calitate este că oferă modelelor de machine learning posibilitatea de a „învăța", adică de a identifica tipare și relații latente între diverse fragmente de date.

Seturile de date pot proveni din diverse surse, precum: experimente, sondaje, jurnale de sistem, senzori, baze de date și multe altele. Ele variază ca dimensiune și format. De regulă, seturile de date sunt organizate în funcție de structuralitate. După acest criteriu, seturile de date pot fi:

1. **Structurate** - date foarte organizate/formatate, reprezentate într-un format strict predefinit (cu o „schemă" rigidă). De obicei, acest tip de date este reprezentat în baze de date relaționale, fișiere CSV/Excel și altele. Mai jos este prezentat un exemplu de set de date organizat în format tabelar:

<div align="center">

| **ID** | **Age** | **Body_Mass_Index** | **Gender** | **Blood_Type** | **...** | **Is_Sick** |
|:------:|:-------:|:-------------------:|:----------:|:--------------:|:-------:|:-----------:|
|   _1_  |    18   |          40         |     "M"    |      "AB+"     |   ...   |     True    |
|   _2_  |    35   |          27         |    None    |       "O"      |   ...   |    False    |
|   _3_  |    24   |          31         |     "F"    |      None      |   ...   |    False    |
|  _..._ |   ...   |         ...         |     ...    |       ...      |   ...   |     ...     |
| _1000_ |    95   |          35         |     "F"    |      "A-"      |   ...   |     True    |

</div>

2. **Semi-Structurate** - date care au unele proprietăți organizaționale, cu o „schemă" flexibilă, ce poate diferi de la o înregistrare la alta. Cel mai adesea, acest tip de seturi de date este reprezentat în JSON, XML, jurnale de sistem și alte formate. Mai jos este prezentat un exemplu de astfel de set de date JSON (observați diferențele dintre înregistrări):

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

3. **Nestructurate** - date care conțin informații ce nu se conformează modelelor tradiționale de date, fără nicio „schemă". Acest tip de seturi de date poate fi reprezentat în diverse moduri: fișiere video/audio/text, mesaje de chat etc. Această categorie este cea mai dificil de lucrat, dar de cele mai multe ori captează perspective și informații bogate.

<a id="2-what-is-pandas"></a>

## 2. Ce este `pandas`?

Majoritatea aplicațiilor de Machine Learning implică lucrul cu seturi de date structurate; totuși, seturile de date din celelalte două categorii sunt de asemenea utilizate, dar necesită mai multă pregătire și o înțelegere mai profundă a mecanismelor complexe de încărcare și procesare. Pentru început, capacitatea de a lucra cu seturi de date presupune abilitatea de a le încărca corect în structuri de date ce pot fi manipulate/acționate prin programare. Una dintre cele mai populare biblioteci care oferă această posibilitate este `pandas`. [`pandas`](https://pandas.pydata.org/docs/index.html) (prescurtat uzual `pd`) este un instrument open-source de analiză și manipulare a datelor, construit pe baza limbajului de programare `Python`, capabil să convertească datele din fișiere în structuri de date cu care se poate interacționa prin cod.

Biblioteca `pandas` oferă două structuri principale de date - `pd.DataFrame` și `pd.Series`. Ele pot fi interpretate ca un array bidimensional = `pd.DataFrame` și un array unidimensional = `pd.Series`. Cu toate acestea, ele dispun de utilitare extinse care le fac foarte utile în lucrul cu seturi de date.

1. `pd.Series` - array etichetat unidimensional care conține date de orice tip. Această structură poate fi interpretată ca o listă de bază. De exemplu:

<div align="center">

| **ID** |
|:------:|
|   _1_  |
|   _2_  |
|   _3_  |
|  _..._ |
| _1000_ |

</div>

2. `pd.DataFrame` - structură de date bidimensională care conține date precum un array bidimensional. Cu alte cuvinte, este un tabel de bază cu rânduri și coloane. De exemplu, reamintim exemplul anterior cu setul de date structurat:

<div align="center">

| **ID** | **Age** | **Body_Mass_Index** |
|:------:|:-------:|:-------------------:|
|   _1_  |    18   |          40         |
|   _2_  |    35   |          27         |
|   _3_  |    24   |          31         |
|  _..._ |   ...   |         ...         |
| _1000_ |    95   |          35         |

</div>

Ambele structuri pot conține orice valori, la fel ca listele Python, dar sunt compatibile cu alte biblioteci, precum `numpy`, ceea ce înseamnă că la instanțierea obiectelor de tip `pd.DataFrame` și `pd.Series`, pot fi utilizate ca input array-uri `numpy`. Nu vă îngrijorați dacă nu sunteți familiarizat cu această bibliotecă - lecțiile viitoare vor clarifica acest aspect. Deocamdată, considerați aceasta un indiciu al compatibilității inerente a bibliotecii `pandas` cu alte biblioteci.

<a id="3-where-to-find-datasets"></a>

## 3. Unde găsim Seturi de Date?

Înainte de a încerca să încărcați orice set de date, este necesar să îl găsiți (sau să îl produceți) mai întâi. Acest lucru poate fi realizat prin intermediul diferitelor servicii, precum și al altor surse - experimente, teste, observații personale. Deoarece internetul oferă oamenilor posibilitatea de a partaja informații cu alții, în domeniul Machine Learning, oamenii îl folosesc și pentru a partaja diverse seturi de date, făcându-le publice și disponibile gratuit. Luați în considerare că, de cele mai multe ori, pentru a utiliza seturi de date calitative și de dimensiuni mari, este necesară o confirmare suplimentară privind modul în care va fi utilizat setul de date, care sunt obiectivele urmărite cu ajutorul acestuia și așa mai departe.

Totuși, pentru majoritatea aplicațiilor ML, seturile de date publice sunt mai mult decât suficiente. Câteva dintre cele mai influente platforme în partajarea și publicarea seturilor de date sunt:

1. [**Kaggle**](https://www.kaggle.com/) - Platformă web a comunității AI/ML ce oferă posibilitatea de a partaja seturi de date, a găzdui și participa la competiții, a posta notebook-uri cu soluții pentru diverse sarcini și a accesa și studia o cantitate mare de material de învățare. Pentru a vizualiza paginile dedicate seturilor de date, urmați link-ul: [Kaggle Datasets](https://www.kaggle.com/datasets). Accesibilă pentru începători, potrivită atât pentru novici cât și pentru pasionații avansați de ML, cu accent puternic pe învățare și participare la competiții.
2. [**HuggingFace**](https://huggingface.co/) - Platformă web similară a comunității AI/ML, ce oferă un loc pentru a găzdui și partaja seturi de date, într-un mod similar cu cel al [Github](https://github.com/) (structură de repository, versionare fișiere etc.), acces la modele open-source AI/ML, posibilitatea de a partaja realizările în domeniul ML și altele. Pentru a accesa secțiunea de seturi de date a platformei, urmați link-ul: [HuggingFace Datasets](https://huggingface.co/datasets). Curbă de învățare mai abruptă, cu accent mai puternic pe partajarea progreselor și găzduirea de modele, aplicații AI/ML și seturi de date, făcându-le disponibile comunității.
3. [**Github**](https://github.com/) - Deși [Github](https://github.com/) este mai degrabă o platformă cloud care utilizează [Git](https://git-scm.com/) ca Sistem de Control al Versiunilor decât o platformă axată pe găzduirea seturilor de date, oamenii pot publica și seturi de date pe această platformă. Puteți căuta printre diferite repository-uri pe cele declarate explicit ca destinate găzduirii de seturi de date. Încercați-vă norocul și pe această platformă.

<a id="4-introduction-to-ser"></a>

## 4. Introducere în `pd.Series`

`pd.Series` reprezintă un array etichetat unidimensional capabil să conțină orice tip de date (de ex., _întregi_, _numere cu virgulă mobilă_, _șiruri de caractere_, _obiecte Python_, etc.). Conceptual, poate fi privit ca o singură coloană de date asociată cu un index ce furnizează etichete semnificative pentru fiecare observație.

Un `pd.Series` este alcătuit din mai multe componente principale:

* **Values** – datele subiacente stocate într-o structură similară unui array NumPy,
* **Index** – o secvență de etichete ce identifică unic fiecare element,
* **Name** - numele obiectului `pd.Series`.
* **Dtype** – tipul de date al valorilor (de ex., `int64`, `float64`, `object`, etc.).

Mecanismul de etichetare este deosebit de important în etapele timpurii ale analizei datelor, deoarece permite accesarea, alinierea și filtrarea datelor pe baza unor identificatori semnificativi, nu doar a indicilor poziționale.

La încărcarea datelor din surse externe (de ex., _CSV_, _Excel_, _JSON_, _baze de date SQL_, etc.), fiecare coloană a setului de date este reprezentată intern ca un `pd.Series`. Aceasta face din obiectul `pd.Series` abstractizarea de bază pentru înțelegerea modului în care datele tabelare sunt structurate și manipulate în `pandas`.

Constructorul unui `pd.Series` arată astfel:

```py
class pandas.Series(data=None, index=None, dtype=None, name=None, copy=None):
    """
    One-dimensional ndarray with axis labels (including time series).

    ...
    """
```

<div align="center">

| Argument   | Descriere                                                                                                                    |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `data`     | Datele efective ale Series. Poate fi o listă, tuplu, array NumPy, dicționar sau scalar.                                      |
| `index`    | Etichete opționale pentru fiecare element. Trebuie să corespundă lungimii `data`. Dacă lipsește, se folosesc indicii întregi `0..n-1`. |
| `dtype`    | Tip de date opțional pentru valori (de ex., `int64`, `float64`, `object`, `category`). Dacă nu este furnizat, este inferit automat. |
| `name`     | Numele opțional al Series. Util când reprezintă o coloană într-un DataFrame.                                                 |
| `copy`     | Dacă se copiază datele. Implicit `False`.                                                                                    |

</div>

Pentru a instanția un `pd.Series` există mai multe modalități. De exemplu, cea mai simplă:

* Inițializare `pd.Series` gol:

```py
import pandas as pd

sample_series: pd.Series = pd.Series()
print(sample_series)
#   OUTPUT:
#       Series([], dtype: object)
```

* Inițializare dintr-o listă:

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

* Inițializare dintr-o listă cu Index Personalizat:

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

* Inițializare dintr-un dicționar (`keys` sunt preluate ca `pd.Index`, `values` - ca date pe rânduri):

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

* Specificarea numelui și a tipului de date:

```py
series_named = pd.Series(data=[1, 2, 3], dtype='float64', name='SAMPLE_SERIES')
print(series_named)
#   OUTPUT:
#       0    1.0
#       1    2.0
#       2    3.0
#       Name: SAMPLE_SERIES, dtype: float64
```

* Dintr-o valoare scalară (valoarea este copiată pe toți indicii):

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

## 5. Explorarea Inițială a `pd.Series`

Deoarece `pd.Series` reprezintă o singură variabilă, explorarea se concentrează pe înțelegerea valorilor, tipului de date, datelor lipsă și a statisticilor de bază. Analiștii investighează frecvent aceste obiecte prin metodele lor, în felul următor:

### Inspecția variabilelor

```py
import pandas as pd

series_obj: pd.Series = pd.Series(
    data=[i + 1 for i in range(0, 100)],
    name="SAMPLE_DATA"
)
```

* Întreg `pd.Series`-ul:

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

* Primele `n` rânduri - `.head()`:

```py
n: int = 3 # primele 3 rânduri
print(series_obj.head(n=n))
#   OUTPUT:
#       0    1
#       1    2
#       2    3
#       Name: SAMPLE_DATA, dtype: int64
```

* Ultimele `n` rânduri - `.tail()`:

```py
n: int = 3 # ultimele 3 rânduri
print(series_obj.tail(n=n))
#   OUTPUT:
#       97     98
#       98     99
#       99    100
#       Name: SAMPLE_DATA, dtype: int64
```

### Verificarea tipurilor de date

```py
import pandas as pd

series_obj = pd.Series([1, 2, 3, 4], name="NUMBERS")
```

* Utilizarea atributului intern - `.dtype`:

```py
print(series_obj.dtype)
#   OUTPUT:
#       int64
```

### Identificarea valorilor lipsă

```py
import numpy as np
import pandas as pd

series_obj = pd.Series(data=[1, np.nan, 3, None, 5], name="WITH_MISSING")
```

* Ca `pd.Series` cu valori `bool` pentru fiecare rând - `.isna()`:

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

* Numărul total de valori lipsă - `.isna().sum()`:

```py
print(f"NUMĂR DE VALORI LIPSĂ: {series_obj.isna().sum()}")
#   OUTPUT:
#       NUMĂR DE VALORI LIPSĂ: 2
```

### Calculul Statisticilor Numerice

```py
import pandas as pd

numeric_series = pd.Series(
    data=[10, 10, 10, 20, 30, 100, 0, -100, -20, 20, 10, 45, 20, 30, 40, 50], 
    name="NUMERIC_STATS"
)
```

* Media Aritmetică și Valoarea Mediană - `.mean()` și `.median()`:

```py
print(f"MEDIE ARITMETICĂ: {numeric_series.mean()}")
print(f"VALOARE MEDIANĂ: {numeric_series.median()}")
#   OUTPUT:
#       MEDIE ARITMETICĂ: 17.1875
#       VALOARE MEDIANĂ: 20.0
```

* Valorile Maxime și Minime - `.min()` și `.max()`:

```py
print(f"VALOARE MINIMĂ: {numeric_series.min()}")
print(f"VALOARE MAXIMĂ: {numeric_series.max()}")
#   OUTPUT:
#       VALOARE MINIMĂ: -100
#       VALOARE MAXIMĂ: 100
```

* Suma tuturor valorilor - `.sum()`:

```py
print(f"SUMA VALORILOR: {numeric_series.sum()}")
#   OUTPUT:
#       SUMA VALORILOR: 275
```

* Numărul de Valori Non-Lipsă - `.count()`:

```py
print(f"NUMĂR DE VALORI NON-LIPSĂ: {numeric_series.count()}")
#   OUTPUT:
#       NUMĂR DE VALORI NON-LIPSĂ: 16
```

* Deviația Standard și Varianța - `.std()` și `.var()`:

```py
print(f"VALOAREA DEVIAȚIEI STANDARD: {numeric_series.std()}")
print(f"VALOAREA VARIANȚEI: {numeric_series.var()}")
#   OUTPUT:
#       VALOAREA DEVIAȚIEI STANDARD: 40.98653234092064
#       VALOAREA VARIANȚEI: 1679.8958333333333
```

* Cuartile și perspective asupra distribuției - `.quantile()`:

```py
print(f"Q1 (percentila 25): {numeric_series.quantile(0.25)}")
print(f"Q2 (percentila 50 / mediană): {numeric_series.quantile(0.50)}")
print(f"Q3 (percentila 75): {numeric_series.quantile(0.75)}")
#   OUTPUT:
#       Q1 (percentila 25): 10.0
#       Q2 (percentila 50 / mediană): 20.0
#       Q3 (percentila 75): 32.5
```

* Agregarea mai multor statistici numerice - `.agg()`:

```py
print("\nSTATISTICI NUMERICE AGREGATE:")
print(numeric_series.agg(["mean", "median", "min", "max", "std", "count"]))
#   OUTPUT:
#       STATISTICI NUMERICE AGREGATE:
#       mean       17.187500
#       median     20.000000
#       min      -100.000000
#       max       100.000000
#       std        40.986532
#       count      16.000000
#       Name: NUMERIC_STATS, dtype: float64
```

* Calculul Rezumatului Statistic (producând statisticile menționate anterior)

```py
import numpy as np
import pandas as pd

numeric_series = pd.Series(
    data=[22.5, 27.2, 31.8, np.nan, 28.6, 29.0, None, 32.1, 35.0],
    name="NUMERIC_SERIES"
)
```

* Calculul statisticilor sumare - `.describe()`:

```py
print(f"STATISTICI SUMARE: {numeric_series.describe()}")
#   OUTPUT:
#       STATISTICI SUMARE:
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

### Calculul Statisticilor bazate pe Frecvență (valabil pentru date non-numerice)

```py
import pandas as pd

categorical_series = pd.Series(
    ["CANADA", "AUSTRALIA", "USA", "JAPAN", "USA", "CANADA", "JAPAN", "JAPAN"],
    name="COUNTRY_STATS"
)
```

* Frecvența Valorilor Unice - `.value_counts()`:

```py
print(f"FRECVENȚA VALORILOR UNICE:\n{categorical_series.value_counts()}")
#   OUTPUT:
#       FRECVENȚA VALORILOR UNICE:
#       COUNTRY_STATS
#       JAPAN        3
#       CANADA       2
#       USA          2
#       AUSTRALIA    1
#       Name: count, dtype: int64 
```

* Moda (valoarea/valorile cel mai frecvent întâlnite) - `.mode()`:

```py
print(f"\nVALOAREA/VALORILE CEL MAI FRECVENTE:\n{categorical_series.mode()}")
#   OUTPUT:
#       VALOAREA/VALORILE CEL MAI FRECVENTE:        
#       0    JAPAN
#       Name: COUNTRY_STATS, dtype: str
```

* Numărul de Valori Non-Lipsă - `.count()`:

```py
print(f"\nNUMĂR DE VALORI NON-LIPSĂ: {categorical_series.count()}")
#   OUTPUT:
#       NUMĂR DE VALORI NON-LIPSĂ: 8
```

* Numărul de Valori Unice - `.nunique()`:

```py
print(f"NUMĂR DE VALORI UNICE: {categorical_series.nunique()}")
#   OUTPUT:
#       NUMĂR DE VALORI UNICE: 4
```

* Calculul Rezumatului Statistic (producând statisticile menționate anterior)

```py
import numpy as np
import pandas as pd

mixed_series = pd.Series(
    data=[10, "CANADA", 30, 40, np.nan, None, 60, "USA", 80],
    name="MIXED_SERIES"
)
```

* Calculul statisticilor sumare - `.describe()`:

```py
print(f"STATISTICI SUMARE: {mixed_series.describe()}")
#   OUTPUT:
#       STATISTICI SUMARE:
#       count      7
#       unique     7
#       top       10
#       freq       1
#       Name: MIXED_SERIES, dtype: int64
```

### Determinarea Memoriei Ocupate de `pd.Series` sau `pd.DataFrame`

```py
import numpy as np
import pandas as pd

mixed_series = pd.Series(
    data=[10, "CANADA", 30, 40, np.nan, None, 60, "USA", 80],
    name="MIXED_SERIES"
)
```

* Calculul consumului de memorie - `.memory_usage()`:

```py
print(f"MEMORIE OCUPATĂ DE mixed_series: {mixed_series.memory_usage()} bytes")
#   OUTPUT:
#       MEMORIE OCUPATĂ DE mixed_series: 204 bytes
```

Deoarece obiectele `pd.Series` suportă operații vectorizate, ele permit calcule exploratorii eficiente fără bucle explicite. Această capacitate este esențială în diagnosticarea rapidă a seturilor de date nou încărcate, precum: detectarea anomaliilor, validarea intervalelor sau standardizarea valorilor înainte de procesarea ulterioară (veți ajunge la acestea în lecțiile viitoare).

<a id="6-introduction-to-df"></a>

## 6. Introducere în `pd.DataFrame`

În timp ce un `pd.Series` modelează o singură coloană, structura principală utilizată pentru încărcarea datelor și analiza exploratorie este `pd.DataFrame`. Un `pd.DataFrame` este o structură de date bidimensională, etichetată, compusă din mai multe obiecte `pd.Series` aliniate, care partajează același index. Poate fi gândit ca un tabel în memorie unde rândurile reprezintă observații, iar coloanele reprezintă variabile.

Fiecare `pd.DataFrame` conține:

* **Values** – stocate ca o colecție de `pd.Series`, conținând valorile efective în formă tabelară (rânduri și coloane).
* **Row Index** – etichetele rândurilor care identifică unic fiecare observație.
* **Column Labels** – numele coloanelor care identifică fiecare variabilă sau feature.
* **Dtypes** – tipurile de date ale fiecărei coloane (de ex., `int64`, `float64`, `object`, etc.).

Această reprezentare tabelară face `pd.DataFrame`-ul ideal pentru importul și explorarea seturilor de date din lumea reală. Când datele sunt citite din fișiere sau baze de date (de ex., _CSV_, _Excel_, _JSON_, _baze de date SQL_, etc.), `pandas` organizează automat informațiile într-un `pd.DataFrame`, păstrând numele coloanelor și încercând să infereze tipurile de date adecvate.

Constructorul `pd.DataFrame` arată astfel:

```py
class pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None):
    """
    Two-dimensional, size-mutable, potentially heterogeneous tabular data.

    ...
    """
```

<div align="center">

| Argument  | Descriere                                                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `data`    | Datele efective ale DataFrame-ului. Poate fi un dict de liste/array-uri/Series, un array NumPy 2D, un alt DataFrame, sau o listă de dict-uri. |
| `index`   | Etichete opționale pentru rânduri. Trebuie să corespundă numărului de rânduri din `data`. Dacă lipsește, se folosesc indicii întregi `0..n-1`. |
| `columns` | Etichete opționale pentru coloane. Dacă `data` este un dict, poate fi utilizat pentru a reordona sau selecta un subset de coloane.           |
| `dtype`   | Tip de date opțional de forțat pe toate coloanele. Dacă nu este furnizat, tipul fiecărei coloane este inferit automat.                      |
| `copy`    | Dacă se copiază datele de intrare. Implicit `False`.                                                                                       |

</div>

Există mai multe modalități de a instanția un `pd.DataFrame`. De exemplu:

* Inițializare `pd.DataFrame` gol:

```py
import pandas as pd

empty_df: pd.DataFrame = pd.DataFrame()
print(empty_df)
#   OUTPUT:
#       Empty DataFrame
#       Columns: []
#       Index: []
```

* Inițializare dintr-un dicționar de liste (`keys` devin etichete de coloane, `values` devin datele coloanelor):

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

* Inițializare dintr-o listă de dicționare (fiecare dict reprezintă un rând):

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

* Inițializare cu un index de rând personalizat:

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

* Inițializare dintr-un array NumPy 2D cu nume de coloane explicite:

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

* Inițializare dintr-o colecție de obiecte `pd.Series` (fiecare `Series` devine o coloană, aliniate după index):

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

## 7. Explorarea Inițială a `pd.DataFrame`

Deoarece un `pd.DataFrame` este o structură bidimensională, explorarea sa implică înțelegerea nu doar a valorilor, ci și a dimensiunii, tipurilor de coloane, tiparelor de date lipsă și a statisticilor sumare pe mai multe variabile simultan. Analiștii investighează de obicei obiectele `pd.DataFrame` în următoarele moduri:

### Inspecția DataFrame-ului

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

* Întreg `pd.DataFrame`-ul:

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

* Primele `n` rânduri – `.head()`:

```py
n: int = 3
print(df.head(n=n))
#   OUTPUT:
#             name  age   salary   department  score
#       0    Alice   25  50000.0           HR   88.0
#       1      Bob   30  60000.0  Engineering   95.0
#       2  Charlie   35  70000.0  Engineering    NaN
```

* Ultimele `n` rânduri – `.tail()`:

```py
n: int = 3
print(df.tail(n=n))
#   OUTPUT:
#             name  age   salary   department  score
#       2  Charlie   35  70000.0  Engineering    NaN
#       3    Diana   28  55000.0    Marketing   76.0
#       4      Eve   22  48000.0           HR    NaN
```

### Verificarea Dimensiunii și Structurii

* Dimensiunile DataFrame-ului (rânduri, coloane) – `.shape`:

```py
print(f"DIMENSIUNE: {df.shape}")
#   OUTPUT:
#       DIMENSIUNE: (5, 5)
```

* Numele coloanelor – `.columns`:

```py
print(f"COLOANE: {df.columns}")
print(f"COLOANE CA LISTĂ: {df.columns.tolist()}")
#   OUTPUT:
#       COLOANE: Index(['name', 'age', 'salary', 'department', 'score'], dtype='str')
#       COLOANE CA LISTĂ: ['name', 'age', 'salary', 'department', 'score']
```

* Tipul de date al fiecărei coloane – `.dtypes`:

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

* Rezumatul structural concis al DataFrame-ului – `.info()`:

Această metodă este foarte utilă în aplicațiile ML (și în altele). Afișează informații complexe despre componentele structurale ale `pd.DataFrame`-ului, cum ar fi numărul de rânduri, coloane, tipuri de date, numărul de elemente non-nule și utilizarea memoriei.

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

### Identificarea Valorilor Lipsă

* Mască booleană a valorilor lipsă per celulă – `.isna()`:

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

* Numărul total de valori lipsă per coloană – `.isna().sum()`. Aceasta produce un `pd.Series` în care fiecare Index este un nume de coloană din `pd.DataFrame`:

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

* Proporția valorilor lipsă per coloană – `.isna().mean()`. Aceasta demonstrează că mai multe operații pot fi înlănțuite una după alta, deoarece `.isna()` produce un `pd.Series`, iar acest obiect dispune de metoda `.mean()`, după cum ați văzut în secțiunile anterioare:

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

### Calculul Statisticilor Numerice

* Statistici sumare agregate pentru toate coloanele numerice – `.describe()`:

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

* Media aritmetică a fiecărei coloane numerice – `.mean()`. Această operație, ca și alte operații numerice, este suportată doar pentru coloanele cu valori numerice - cu valori de tip string sau orice alt tip non-numeric, va genera o eroare:

```py
print(df[['age', 'salary', 'score']].mean())
#   OUTPUT:
#       age          28.000000
#       salary    56600.000000
#       score        86.333333
#       dtype: float64
```

* Minimul și maximul fiecărei coloane numerice – `.min()` și `.max()`:

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

### Calculul Statisticilor bazate pe Frecvență (valabil pentru coloanele non-numerice)

* Frecvența valorilor unice dintr-o coloană categorială – `.value_counts()`:

```py
print(df['department'].value_counts())
#   OUTPUT:
#       department
#       HR             2
#       Engineering    2
#       Marketing      1
#       Name: count, dtype: int64
```

* Numărul de valori unice per coloană – `.nunique()`:

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

* Statistici sumare pentru toate coloanele, inclusiv cele non-numerice – `.describe(include='all')`:

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

### Determinarea Memoriei Ocupate de `pd.DataFrame`

* Calculul consumului total de memorie – `.memory_usage()`:

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

* Calculul memoriei totale pentru toate coloanele:

```py
print(f"MEMORIE TOTALĂ: {df.memory_usage(deep=True).sum()} bytes")
#   OUTPUT:
#       MEMORIE TOTALĂ: 800 bytes
```

Deoarece obiectele `pd.DataFrame` suportă operații vectorizate pe toate coloanele simultan, ele permit calcule exploratorii eficiente fără bucle explicite. Această capacitate este esențială în diagnosticarea rapidă a seturilor de date nou încărcate, cum ar fi: detectarea anomaliilor, validarea tipurilor de coloane, identificarea tiparelor de date lipsă sau calculul distribuțiilor înainte de procesarea ulterioară.

<a id="8-data-loading"></a>

## 8. Încărcarea Datelor în Formate Multiple

Pe lângă metodele de instanțiere a obiectelor `pd.DataFrame` menționate anterior, există și o altă modalitate de a crea astfel de obiecte - **încărcarea datelor**.

`pandas` suportă citirea datelor dintr-o mare varietate de formate și surse externe direct într-un `pd.DataFrame`:

* **CSV** (`.csv`) sau orice alt format cu valori delimitate - `pd.read_csv()`,
* **JSON** (`.json`, `.jsonl`, etc.) - `pd.read_json()`,
* **Excel** (`.xlsx`, `.xls`, `.xlsb`, `.xlsm`, etc.) - `pd.read_excel()`,
* **Baze de date SQL** (prin intermediul unei conexiuni sau al unei interogări SQL) - `pd.read_sql()`,
* **API-uri** (prin răspunsuri HTTP, de obicei în format JSON).

Fiecare format dispune de o funcție dedicată de citire în `pandas`, care gestionează automat parsarea, inferența tipurilor și atribuirea indexului. Totuși, este posibilă și gestionarea manuală, utilizând argumente în funcția `pandas` corespunzătoare.

### 8.1 Încărcarea Fișierelor CSV și a Fișierelor cu Valori Delimitate

CSV (Comma-Separated Values) este cel mai comun format de fișier plat pentru date tabelare. `pandas` îl citește folosind `pd.read_csv()`:

```py
pandas.read_csv(filepath_or_buffer, sep=',', header='infer', names=None,
                index_col=None, usecols=None, dtype=None, na_values=None,
                skiprows=None, nrows=None, encoding='utf-8', ...)
```

<div align="center">

| Argument              | Descriere                                                                                                |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| `filepath_or_buffer`  | Calea către fișier, un URL sau orice obiect de tip file-like.                                             |
| `sep`                 | Delimitatorul dintre câmpuri. Implicit `','`. Folosiți `'\t'` pentru fișiere TSV.                        |
| `header`              | Numărul rândului (rândurilor) de utilizat ca nume de coloane. Implicit `'infer'` folosește primul rând.  |
| `names`               | Listă explicită de nume de coloane de atribuit. Util când fișierul nu are rând de antet.                 |
| `index_col`           | Coloana (coloanele) de utilizat ca index de rând. Poate fi un nume de coloană sau o poziție întreagă.    |
| `usecols`             | Subset de coloane de încărcat. Reduce utilizarea memoriei pentru fișiere late.                           |
| `dtype`               | Dicționar care mapează numele coloanelor la tipuri de date explicite.                                     |
| `na_values`           | Șiruri suplimentare de recunoscut ca `NaN` (de ex., `['N/A', 'missing', '-']`).                          |
| `skiprows`            | Numărul de rânduri (sau lista de indici de rânduri) de omis la începutul fișierului.                     |
| `nrows`               | Numărul maxim de rânduri de citit. Util pentru previzualizarea fișierelor mari.                          |
| `encoding`            | Codificarea fișierului (de ex., `'utf-8'`, `'latin-1'`). Implicit `'utf-8'`.                             |

</div>

* Utilizare de bază - citirea unui fișier CSV cu setările implicite:

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

* Încărcarea doar a unor coloane specifice pentru a reduce utilizarea memoriei:

```py
df = pd.read_csv('data/employees.csv', usecols=['name', 'salary'])
print(df.head(3))
#   OUTPUT:
#             name   salary
#       0    Alice  50000.0
#       1      Bob  60000.0
#       2  Charlie  70000.0
```

* Încărcarea unui fișier TSV (Tab-Separated Values) sau a oricărui alt fișier cu valori delimitate - specificați separatorul utilizat în fișier ca argument al funcției:

```py
df = pd.read_csv('data/employees.tsv', sep='\t')
print(df.head(3))
#             name   salary
#       0     John  10000.0
#       1   Daniel  20000.0
#       2     Rick  40000.0
```

* Previzualizarea unui fișier mare fără a-l încărca complet (pentru seturi de date mari, vor exista modalități mai bune de a gestiona încărcarea folosind alte biblioteci, deoarece `pandas` încarcă totul direct în RAM, ceea ce poate fi insuficient - acest aspect va fi acoperit în lecțiile viitoare):

```py
df_preview = pd.read_csv('data/large_dataset.csv', nrows=100)
print(f"Dimensiune previzualizare: {df_preview.shape}")
#   OUTPUT:
#       Dimensiune previzualizare: (100, 4)
```

* Specificarea unei coloane ca index de rând și definirea explicită a tipurilor de date:

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

### 8.2 Încărcarea Fișierelor JSON

JSON (JavaScript Object Notation) este un format larg utilizat pentru date structurate și semi-structurate, în special în contexte web și API. `pandas` îl citește folosind `pd.read_json()`:

```py
pandas.read_json(path_or_buf, orient=None, dtype=None, encoding='utf-8',
                 lines=False, ...)
```

<div align="center">

| Argument       | Descriere                                                                                                                  |
| -------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `path_or_buf`  | Calea către fișier, un URL sau un șir JSON brut.                                                                           |
| `orient`       | Formatul așteptat al datelor JSON. Valori comune: `'records'`, `'columns'`, `'index'`, `'values'`, `'split'`.              |
| `dtype`        | Dicționar de mapări între numele coloanelor și tipuri de date, sau `False` pentru dezactivarea inferenței tipurilor.       |
| `encoding`     | Codificarea fișierului. Implicit `'utf-8'`.                                                                                |
| `lines`        | Dacă `True`, citește fișierul ca JSON delimitat prin newline (`.jsonl`), unde fiecare linie este un obiect JSON separat.   |

</div>

Parametrul `orient` este cel mai important de înțeles, deoarece JSON poate reprezenta date tabelare în mai multe layout-uri structurale diferite:

<div align="center">

| `orient`      | Structura JSON Așteptată                                                                |
| ------------- | --------------------------------------------------------------------------------------- |
| `'records'`   | Listă de dict-uri de rând: `[{"col": val, ...}, ...]`                                   |
| `'columns'`   | Dict de array-uri de coloane: `{"col": {index: val, ...}, ...}`                         |
| `'index'`     | Dict de dict-uri de rând indexate după index: `{index: {"col": val, ...}, ...}`         |
| `'values'`    | Array 2D pur de valori (fără etichete de coloane sau index).                            |
| `'split'`     | Dict cu chei separate pentru `index`, `columns` și `data`.                             |

</div>

* Încărcarea unui fișier JSON standard orientat pe înregistrări:

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

* Încărcarea unui fișier JSON delimitat prin newline (`.jsonl`), unde fiecare linie reprezintă o înregistrare:

```py
df = pd.read_json('data/employees.jsonl', lines=True)
print(df.head(3))
#   OUTPUT:
#             name  age   salary
#       0    Alice   25  50000.0
#       1      Bob   30  60000.0
#       2  Charlie   35  70000.0
```

* Parsarea directă a unui șir JSON brut (util la lucrul cu răspunsuri API):

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

### 8.3 Încărcarea Fișierelor Excel

Fișierele Excel sunt comune în contexte de business și cercetare. `pandas` le citește folosind `pd.read_excel()`, care suportă mai multe engine-uri și selecția foilor de lucru:

```py
pandas.read_excel(io, sheet_name=0, header=0, names=None, index_col=None,
                  usecols=None, dtype=None, na_values=None, skiprows=None,
                  nrows=None, engine=None, ...)
```

<div align="center">

| Argument      | Descriere                                                                                                               |
| ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `io`          | Calea către fișier sau un URL.                                                                                          |
| `sheet_name`  | Foaia de citit. Acceptă un nume (`'Sheet1'`), un index întreg (`0`) sau `None` pentru a încărca toate foile ca un dict de DataFrame-uri. |
| `header`      | Numărul rândului de utilizat ca nume de coloane. Implicit `0` (primul rând).                                            |
| `names`       | Listă explicită de nume de coloane de atribuit.                                                                         |
| `index_col`   | Coloana de utilizat ca index de rând.                                                                                   |
| `usecols`     | Subset de coloane de încărcat. Acceptă nume de coloane, poziții întregi sau un interval în stilul Excel, de ex. `'A:D'`. |
| `dtype`       | Dicționar care mapează numele coloanelor la tipuri de date.                                                             |
| `na_values`   | Șiruri suplimentare de interpretat ca `NaN`.                                                                            |
| `skiprows`    | Rânduri de omis la începutul foii.                                                                                      |
| `nrows`       | Numărul maxim de rânduri de citit.                                                                                      |
| `engine`      | Engine-ul de parsare de utilizat: `'openpyxl'` (implicit pentru `.xlsx`), `'xlrd'` (pentru `.xls` de versiuni vechi), `'calamine'`, etc. |

</div>

> **Notă:** Citirea fișierelor Excel necesită o dependență opțională. Instalați-o cu `pip install openpyxl` pentru fișiere `.xlsx` sau `pip install xlrd` pentru fișiere `.xls` de versiuni vechi.

* Utilizare de bază - citirea primei foi dintr-un fișier Excel:

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

* Citirea unei foi specifice după nume:

```py
df = pd.read_excel('data/employees.xlsx', sheet_name='Q1_Data')
```

* Citirea tuturor foilor odată - returnează un dicționar de DataFrame-uri indexat după numele foii:

```py
all_sheets: dict = pd.read_excel('data/employees.xlsx', sheet_name=None)

for sheet_name, sheet_df in all_sheets.items():
    print(f"Foaie: {sheet_name}, Dimensiune: {sheet_df.shape}")
#   OUTPUT:
#       Foaie: Q1_Data, Dimensiune: (50, 4)
#       Foaie: Q2_Data, Dimensiune: (48, 4)
#       Foaie: Q3_Data, Dimensiune: (55, 4)
```

* Selectarea unor coloane specifice folosind un interval de coloane în stilul Excel:

```py
df = pd.read_excel('data/employees.xlsx', usecols='A:C')
```

### 8.4 Încărcarea din Baze de Date SQL

`pandas` poate interoga direct baze de date relaționale și poate încărca rezultatele ca `pd.DataFrame` folosind `pd.read_sql()`, `pd.read_sql_query()` sau `pd.read_sql_table()`. Luați în considerare că, în prezent, `pandas` recomandă utilizarea `pd.read_sql()`, care suportă atât numele unui tabel cât și interogări SQL (`pd.read_sql()` este un wrapper peste `pd.read_sql_query()` și `pd.read_sql_table()`, redirecționând automat către funcția potrivită). Este necesar un obiect de conexiune la baza de date, care poate fi creat folosind biblioteci precum `sqlite3`, `sqlalchemy`, `psycopg2` sau altele.

```py
pandas.read_sql(sql, con, index_col=None, params=None, dtype=None, ...)
```

<div align="center">

| Argument    | Descriere                                                                                                       |
| ----------- | ---------------------------------------------------------------------------------------------------------------- |
| `sql`       | Șirul interogării SQL sau numele tabelului de executat/citit.                                                    |
| `con`       | Obiectul de conexiune la baza de date (de ex., un engine `sqlalchemy` sau o conexiune `sqlite3`).               |
| `index_col` | Coloana (coloanele) de setat ca index de rând.                                                                  |
| `params`    | Parametri de legat la interogarea SQL (pentru interogări parametrizate).                                        |
| `dtype`     | Dicționar de mapări între numele coloanelor și tipuri de date.                                                  |

</div>

* Interogarea unei baze de date SQLite folosind `sqlite3`:

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

* Utilizarea unei interogări parametrizate pentru filtrarea sigură a rezultatelor:

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

* Conectarea la o bază de date PostgreSQL folosind `sqlalchemy`:

```py
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql+psycopg2://user:password@localhost:5432/company_db')

df = pd.read_sql(sql="SELECT * FROM employees LIMIT 5", con=engine)
print(df.head())
```

### 8.5 Încărcarea din API-uri

API-urile returnează de obicei date în format JSON prin HTTP. Deși `pandas` nu are o funcție dedicată `read_api()`, fluxul standard combină biblioteca `requests` pentru preluarea datelor cu `pd.DataFrame` sau `pd.read_json()` pentru parsarea răspunsului.

> **Notă:** Biblioteca `requests` nu este inclusă în `pandas`. Instalați-o cu `pip install requests`.

* Preluarea datelor JSON de la un REST API și încărcarea lor într-un DataFrame:

```py
import requests
import pandas as pd

url = 'https://jsonplaceholder.typicode.com/users'
response = requests.get(url)
response.raise_for_status()  # generează o eroare pentru răspunsuri HTTP non-2xx

df = pd.DataFrame(response.json())
print(df[['id', 'name', 'email', 'phone']].head(3))
#   OUTPUT:
#          id              name               email           phone
#       0   1     Leanne Graham   Sincere@april.biz  1-770-736-8031
#       1   2      Ervin Howell   Shanna@melissa.tv    010-692-6593
#       2   3  Clementine Bauch  Nathan@yesenia.net  1-463-123-4447
```

* Transmiterea parametrilor de interogare pentru filtrarea răspunsului API:

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

* Includerea header-elor de autentificare pentru API-uri protejate:

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

### Sumar

Tabelul de mai jos oferă o referință rapidă pentru funcția de încărcare asociată fiecărui format:

<div align="center">

| Format       | Funcție                | Argumente Comune                                             |
| ------------ | ---------------------- | ------------------------------------------------------------ |
| CSV / TSV    | `pd.read_csv()`        | `sep`, `usecols`, `dtype`, `na_values`, `nrows`              |
| JSON / JSONL | `pd.read_json()`       | `orient`, `lines`, `dtype`                                   |
| Excel        | `pd.read_excel()`      | `sheet_name`, `usecols`, `engine`, `nrows`                   |
| SQL          | `pd.read_sql()`        | `sql`, `con`, `index_col`, `params`                          |
| API          | `requests` + DataFrame | `response.json()`, `pd.DataFrame(...)`, `raise_for_status()` |

</div>

## Cheatsheet

| Concept                      | Reminder pe scurt                                                                                                     |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| Date                         | Observații sau măsurători brute; seturile de date sunt colecții organizate, specifice unui domeniu                    |
| Date structurate             | Format tabelar, cu schemă impusă - rânduri și coloane (CSV, Excel, SQL)                                               |
| Date semi-structurate        | Schemă flexibilă, ce poate varia per înregistrare - JSON, XML, log-uri                                                |
| Date nestructurate           | Fără schemă - text, imagini, audio, video                                                                             |
| `pandas`                     | Bibliotecă Python pentru încărcarea, reprezentarea și manipularea datelor tabelare                                    |
| `pd.Series`                  | Array etichetat unidimensional - reprezintă o singură coloană de date                                                 |
| `pd.DataFrame`               | Tabel etichetat bidimensional - o colecție de obiecte `pd.Series` aliniate                                            |
| Componente `pd.Series`       | Values, Index, Name, Dtype                                                                                            |
| Componente `pd.DataFrame`    | Values, Row Index, Column Labels, Dtypes                                                                              |
| Explorare Series             | `.head()`, `.tail()`, `.dtype`, `.isna()`, `.describe()`, `.value_counts()`                                           |
| Explorare DataFrame          | `.head()`, `.tail()`, `.shape`, `.columns`, `.dtypes`, `.info()`, `.describe()`                                       |
| Valori lipsă                 | `.isna().sum()` numără valorile lipsă; `.isna().mean()` dă proporția per coloană                                      |
| Statistici numerice          | `.mean()`, `.median()`, `.std()`, `.var()`, `.min()`, `.max()`, `.quantile()`                                         |
| Statistici categoriale       | `.value_counts()`, `.mode()`, `.nunique()`                                                                            |
| Utilizare memorie            | `.memory_usage(deep=True)` - folosiți `.sum()` pentru total pe toate coloanele                                        |
| `pd.read_csv()`              | Citește CSV / TSV; argumente cheie: `sep`, `usecols`, `dtype`, `na_values`, `nrows`                                   |
| `pd.read_json()`             | Citește JSON / JSONL; argumente cheie: `orient`, `lines` - **potriviți `orient` cu layout-ul fișierului**             |
| `pd.read_excel()`            | Citește foi Excel; `sheet_name=None` încarcă toate foile ca dict de DataFrame-uri                                     |
| `pd.read_sql()`              | Execută o interogare SQL printr-un obiect de conexiune (`sqlite3`, `sqlalchemy`, etc.)                                |
| Încărcare API                | Fără cititor dedicat - folosiți `requests.get()` + `pd.DataFrame(response.json())`                                    |
| Platforme de seturi de date  | Kaggle (competiții + învățare), HuggingFace (modele + seturi de date), GitHub (repository-uri)                        |

_Urmează: deschideți notebook-urile de exerciții pentru a exersa aceste concepte în mod practic._
