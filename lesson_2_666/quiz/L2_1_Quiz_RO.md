# Chestionar — Încărcarea Datelor și Explorarea Inițială

## Întrebarea 1

Ce funcție din `pandas` ai folosi pentru a încărca un fișier cu valori separate prin tab (TSV)?

    A. pd.read_tsv('data.tsv')
    B. pd.read_csv('data.tsv', sep=',')
    C. pd.read_csv('data.tsv', sep='\t')
    D. pd.read_table('data.tsv', delimiter=',')

### Răspuns Corect: C

**Explicație:**
`pd.read_csv()` gestionează orice fișier cu separatori, nu doar virgule. Pentru a citi un fișier TSV, se transmite `sep='\t'` pentru a suprascrie separatorul implicit (virgula). Nu există o funcție dedicată `pd.read_tsv()` în `pandas`.

---

## Întrebarea 2

Primești un registru Excel cu cinci foi de lucru. Vrei să le încarci pe toate odată într-un singur obiect Python. Care apel realizează acest lucru?

    A. pd.read_excel('data.xlsx', sheet_name='all')
    B. pd.read_excel('data.xlsx', sheet_name=None)
    C. pd.read_excel('data.xlsx', sheet_name=0)
    D. pd.read_excel('data.xlsx', sheets=True)

### Răspuns Corect: B

**Explicație:**
Transmiterea `sheet_name=None` îi indică lui `pd.read_excel()` să citească toate foile de lucru și să returneze un dicționar în care fiecare cheie este numele foii, iar fiecare valoare este `pd.DataFrame`-ul corespunzător. Folosind `sheet_name=0` se încarcă doar prima foaie.

---

## Întrebarea 3

Ce afișează `.info()` față de `.dtypes` în mod individual?

    A. Tipul de date al fiecărei coloane
    B. Primele cinci rânduri ale DataFrame-ului
    C. Numărul de valori non-nule per coloană și utilizarea totală a memoriei
    D. Statistici sumare, precum media și deviația standard

### Răspuns Corect: C

**Explicație:**
`.dtypes` afișează doar tipul de date al fiecărei coloane. `.info()` oferă o privire de ansamblu mai bogată: numărul de rânduri, numele coloanelor, numărul de valori non-nule per coloană, tipurile de date și utilizarea totală a memoriei — fiind metoda de referință pentru o primă inspecție a unui DataFrame nou încărcat.

---

## Întrebarea 4

Un coleg îți oferă un fișier JSON în care fiecare linie este un obiect JSON separat (format JSONL). Ce argument trebuie să setezi când apelezi `pd.read_json()`?

    A. lines=True
    B. multiline=True
    C. sep='\n'
    D. orient='records'

### Răspuns Corect: A

**Explicație:**
Setarea `lines=True` îi indică lui `pd.read_json()` să trateze fiecare linie a fișierului ca un obiect JSON independent, ceea ce este abordarea corectă pentru fișierele JSON delimitate prin newline (`.jsonl`). Parametrul `orient` controlează structura layout-ului JSON, nu dacă acesta este delimitat prin linii.

---

## Întrebarea 5

Care este diferența principală dintre `pd.Series` și `pd.DataFrame`?

    A. `pd.Series` este unidimensional (o singură coloană); `pd.DataFrame` este bidimensional (rânduri și coloane)
    B. `pd.Series` poate stoca doar numere întregi; `pd.DataFrame` poate stoca tipuri mixte
    C. `pd.DataFrame` are un index; `pd.Series` nu are
    D. `pd.Series` se creează mai rapid, dar nu poate fi folosit în interiorul unui `pd.DataFrame`

### Răspuns Corect: A

**Explicație:**
Un `pd.Series` este un array etichetat unidimensional, reprezentând o singură coloană de date. Un `pd.DataFrame` este o structură bidimensională compusă din mai multe obiecte `pd.Series` aliniate. De fapt, fiecare coloană a unui `pd.DataFrame` este stocată intern ca un `pd.Series`.

---

## Întrebarea 6

Încarci un fișier CSV de mari dimensiuni, dar ai nevoie doar de trei din cele cincizeci de coloane ale sale. Ce argument te ajută să eviți încărcarea datelor inutile?

    A. nrows
    B. skiprows
    C. dtype
    D. usecols

### Răspuns Corect: D

**Explicație:**
`usecols` acceptă o listă de nume de coloane (sau poziții) și îi instruiește lui `pd.read_csv()` să parseze și să încarce doar acele coloane, reducând semnificativ utilizarea memoriei la lucrul cu fișiere late. `nrows` limitează rândurile, nu coloanele.

---

## Întrebarea 7

Interoghezi un REST API care returnează JSON și vrei să încarci răspunsul într-un DataFrame. Care abordare este corectă?

    A. pd.read_json(url) direct, fără nicio bibliotecă HTTP
    B. Folosești pd.read_api(url)
    C. Folosești requests.get(url) pentru a prelua răspunsul, apoi transmiți response.json() către pd.DataFrame()
    D. Folosești pd.read_sql() cu endpoint-ul API ca șir de conexiune

### Răspuns Corect: C

**Explicație:**
`pandas` nu are un cititor dedicat pentru API-uri. Tiparul standard este să folosești biblioteca `requests` pentru apelul HTTP, să apelezi `response.raise_for_status()` pentru a detecta erorile, apoi să transmiți `response.json()` — care returnează o listă sau un dicționar Python — către `pd.DataFrame()` pentru conversie.

---

## Întrebarea 8

Ce metodă îți returnează numărul de rânduri și coloane ale unui DataFrame sub forma unui tuplu?

    A. df.size
    B. df.ndim
    C. df.count()
    D. df.shape

### Răspuns Corect: D

**Explicație:**
`df.shape` returnează un tuplu `(nr_rânduri, nr_coloane)`. `df.size` returnează numărul total de elemente (rânduri × coloane). `df.ndim` returnează numărul de dimensiuni (întotdeauna 2 pentru un DataFrame). `df.count()` returnează numărul de valori non-nule per coloană.

---

## Întrebarea 9

Care este modul corect de a număra valorile lipsă per coloană într-un DataFrame?

    A. df.missing()
    B. df.isna().sum()
    C. df.null().sum()
    D. df.count_na()

### Răspuns Corect: B

**Explicație:**
`.isna()` produce un DataFrame boolean în care `True` marchează fiecare valoare lipsă. Înlănțuirea cu `.sum()` numără apoi valorile `True` (fiecare tratată ca `1`) per coloană, oferind numărul total de valori lipsă din fiecare coloană.

---

## Întrebarea 10

Când creezi un `pd.Series` dintr-un dicționar, ce devine indexul?

    A. Cheile dicționarului
    B. Valorile dicționarului
    C. Poziții întregi începând de la 0
    D. Tipul de date al fiecărei valori

### Răspuns Corect: A

**Explicație:**
Când transmiți un dicționar către `pd.Series()`, cheile sunt folosite ca etichete de index, iar valorile devin datele. De exemplu, `pd.Series({'Alice': 25, 'Bob': 30})` produce un Series indexat după `'Alice'` și `'Bob'`.

---

## Întrebarea 11

Trebuie să te conectezi la o bază de date PostgreSQL și să încarci rezultatele unei interogări într-un DataFrame. Care combinație de instrumente este corectă?

    A. pd.read_csv() cu un cursor de bază de date
    B. pd.read_excel() cu numele de gazdă al bazei de date
    C. pd.read_sql() cu un engine SQLAlchemy ca și conexiune
    D. pd.read_json() cu URL-ul bazei de date

### Răspuns Corect: C

**Explicație:**
`pd.read_sql()` necesită un obiect de conexiune activ. Pentru PostgreSQL, abordarea standard este să creezi un engine SQLAlchemy cu `create_engine('postgresql+psycopg2://user:parola@host/db')` și să îl transmiți ca argument `con`. Conexiunile `sqlite3` funcționează de asemenea pentru bazele de date SQLite.

---

## Întrebarea 12

Ce returnează `df.isna().mean()`?

    A. Media tuturor valorilor numerice, excluzând valorile lipsă
    B. Proporția valorilor lipsă pentru fiecare coloană
    C. Numărul total de valori lipsă din întregul DataFrame
    D. Un boolean care indică dacă există vreo valoare lipsă

### Răspuns Corect: B

**Explicație:**
`.isna()` convertește fiecare celulă în `True` (1) sau `False` (0). Apelarea `.mean()` pe acel DataFrame boolean calculează media valorilor `True` per coloană, ceea ce este echivalent cu proporția (fracția) valorilor lipsă din fiecare coloană. Un rezultat de `0.4` înseamnă că 40% din valorile acelei coloane sunt lipsă.

---

## Întrebarea 13

Ce argument al `pd.read_csv()` îți permite să specifici că `-`, `N/A` și `missing` să fie tratate toate ca `NaN`?

    A. fillna=['-', 'N/A', 'missing']
    B. null_values=['-', 'N/A', 'missing']
    C. missing_values=['-', 'N/A', 'missing']
    D. na_values=['-', 'N/A', 'missing']

### Răspuns Corect: D

**Explicație:**
Argumentul `na_values` acceptă o listă de șiruri suplimentare care să fie recunoscute ca `NaN` în timpul parsării. Acesta este deosebit de util când seturile de date folosesc substituenți personalizați pentru datele lipsă în loc de celule goale standard.

---

## Întrebarea 14

Ce tip de structură de date returnează `pd.read_excel('data.xlsx', sheet_name=None)`?

    A. O listă de DataFrame-uri
    B. Un dicționar care mapează numele foilor la DataFrame-uri
    C. Un tuplu de DataFrame-uri
    D. Un singur DataFrame îmbinat

### Răspuns Corect: B

**Explicație:**
Când `sheet_name=None`, `pandas` citește toate foile și returnează un dicționar Python în care fiecare cheie este numele foii (un șir de caractere) și fiecare valoare este `pd.DataFrame`-ul corespunzător. Poți apoi itera peste `.items()` sau accesa foile individuale după nume.

---

## Întrebarea 15

Ai un `pd.Series` cu nume de țări. Ce metodă returnează valoarea cel mai frecvent întâlnită?

    A. .mode()
    B. .max()
    C. .value_counts()
    D. .describe()

### Răspuns Corect: A

**Explicație:**
`.mode()` returnează valoarea (valorile) cel mai frecvent întâlnite într-un Series. `.value_counts()` afișează de asemenea frecvențele, dar returnează un număr complet clasificat pentru fiecare valoare unică, nu doar pe prima. `.max()` ar returna ultimul șir în ordine alfabetică, nu cel mai frecvent.

---

## Întrebarea 16

Care este diferența dintre `.count()` și `.isna().sum()` pe o coloană a unui DataFrame?

    A. Returnează întotdeauna același rezultat
    B. `.count()` numără toate rândurile, inclusiv NaN; `.isna().sum()` numără doar rândurile numerice
    C. `.count()` este pentru Series; `.isna().sum()` funcționează doar pe DataFrame-uri
    D. `.count()` returnează valorile non-nule; `.isna().sum()` returnează numărul de valori lipsă

### Răspuns Corect: D

**Explicație:**
`.count()` returnează numărul de valori non-nule dintr-o coloană, în timp ce `.isna().sum()` numără valorile nule (lipsă). Împreună sunt complementare: dacă o coloană are 100 de rânduri și `.isna().sum()` este 3, atunci `.count()` va fi 97.

---

## Întrebarea 17

Când încarci date dintr-o bază de date SQL folosind `pd.read_sql()`, ce îți permite să faci argumentul `params`?

    A. Să transmiți valori într-o interogare parametrizată în mod sigur, evitând SQL injection
    B. Să specifici ce coloane să încarci
    C. Să setezi tipurile de date ale coloanelor rezultante
    D. Să definești numărul de rânduri de preluat

### Răspuns Corect: A

**Explicație:**
Argumentul `params` îți permite să legi valori la substituenții dintr-o interogare SQL parametrizată (de ex., `WHERE departament = ?`). Aceasta este mai sigur decât formatarea directă a valorilor în interogare ca șiruri de caractere, deoarece previne vulnerabilitățile de tip SQL injection.

---

## Întrebarea 18

Care dintre următoarele descrie corect ce face `df.describe(include='all')` diferit față de `df.describe()`?

    A. Afișează utilizarea memoriei pentru fiecare coloană
    B. Calculează statistici și pentru coloanele non-numerice, adăugând câmpuri precum `unique`, `top` și `freq`
    C. Returnează același rezultat ca `df.info()`
    D. Include valorile lipsă în numărul total

### Răspuns Corect: B

**Explicație:**
Implicit, `df.describe()` rezumă doar coloanele numerice. Transmiterea `include='all'` o forțează să includă și coloanele non-numerice (de ex., șiruri de caractere/categoriale), adăugând statistici precum `unique` (numărul de valori unice), `top` (valoarea cea mai frecventă) și `freq` (numărul de apariții ale acesteia).

---

## Întrebarea 19

Care sunt cele patru componente care alcătuiesc orice obiect `pd.Series`?

    A. Data, Shape, Columns, Dtype
    B. Rows, Columns, Index, Name
    C. Data, Labels, Size, Type
    D. Values, Index, Name, Dtype

### Răspuns Corect: D

**Explicație:**
Orice `pd.Series` este alcătuit din: **Values** (array-ul de date subiacent), **Index** (etichetele care identifică fiecare element), **Name** (o etichetă opțională pentru Series în ansamblu, utilă când reprezintă o coloană într-un DataFrame) și **Dtype** (tipul de date al valorilor stocate).

---

## Întrebarea 20

Un set de date este stocat într-un format structurat, cu rânduri și coloane stricte (similar unui tabel). Din ce categorie de structuralitate a datelor face parte și care este un exemplu tipic de format de fișier?

    A. Structurat — CSV
    B. Nestructurat — TXT
    C. Semi-structurat — CSV
    D. Semi-structurat — JSON

### Răspuns Corect: A

**Explicație:**
Datele structurate urmează o schemă strictă, predefinită, cu rânduri și coloane clar definite. Fișierele CSV sunt un exemplu clasic de date structurate, alături de bazele de date relaționale și fișierele Excel. JSON este în mod tipic semi-structurat, deoarece schema sa poate varia între înregistrări, iar fișierele text simplu sunt nestructurate.

---

**Sumar Scoruri:**

- Total Întrebări: 20
- Teme Acoperite: Tipuri de date și structuralitate, fundamente `pd.Series` și `pd.DataFrame`, metode de inspecție a DataFrame-ului (`.info()`, `.shape`, `.dtypes`, `.describe()`), detectarea valorilor lipsă (`.isna()`, `.count()`), statistici de frecvență (`.value_counts()`, `.mode()`), încărcarea datelor din CSV, JSON, Excel, SQL și API-uri
- Ordine Răspunsuri Corecte: C, B, C, A, A, D, C, D, B, A, C, B, D, B, A, D, A, B, D, A
