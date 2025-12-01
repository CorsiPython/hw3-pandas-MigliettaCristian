[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AxQNHZ_p)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=21899012&assignment_repo_type=AssignmentRepo)
# Homework 3: Analisi Dati e Visualizzazione con Pandas e Matplotlib

Questo repository è il punto di partenza per l'homework "Analisi Dati con Pandas".
L'obiettivo è esercitarti con l'analisi esplorativa dei dati (EDA), utilizzando
Pandas per caricare, pulire e analizzare un dataset reale, e Matplotlib per
creare visualizzazioni grafiche.

In questa versione "starter" trovi lo scheletro delle funzioni da completare e un
setup di test basato su `pytest`.

---

## Il Dataset

Utilizzeremo il **California Housing Dataset** disponibile in scikit-learn.
Questo dataset contiene informazioni su blocchi residenziali in California
basate sul censimento del 1990.

### Colonne del Dataset

| Colonna | Descrizione |
|---------|-------------|
| `MedInc` | Reddito mediano nel blocco (in decine di migliaia di dollari) |
| `HouseAge` | Età mediana delle case nel blocco (anni) |
| `AveRooms` | Numero medio di stanze per abitazione |
| `AveBedrms` | Numero medio di camere da letto per abitazione |
| `Population` | Popolazione del blocco |
| `AveOccup` | Numero medio di occupanti per abitazione |
| `Latitude` | Latitudine del blocco |
| `Longitude` | Longitudine del blocco |
| `MedHouseVal` | **Target** - Valore mediano delle case (in centinaia di migliaia di $) |

---

## Cosa devi fare

Implementa le funzioni nel file `data_analysis.py`:

### 1. `load_california_housing_data() -> pd.DataFrame`

**Comportamento:**
- Carica il dataset usando `fetch_california_housing` di sklearn
- Crea un DataFrame con le feature e il target (`MedHouseVal`)
- Controlla la presenza di valori mancanti e, se presenti, rimuovi le righe
- I nomi delle colonne devono essere quelli originali del dataset più `MedHouseVal`

**Output:** DataFrame pulito con 9 colonne

### 2. `calculate_average_value_by_age(df: pd.DataFrame) -> pd.Series`

**Input:** Il DataFrame pulito

**Comportamento:**
- Raggruppa i dati per `HouseAge` (età della casa)
- Calcola la media di `MedHouseVal` per ogni gruppo
- Ordina l'indice in ordine crescente

**Output:** Serie Pandas con `HouseAge` come indice e valore medio come valori

### 3. `get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame`

**Input:** Il DataFrame pulito

**Comportamento:**
- Calcola la matrice di correlazione di Pearson per tutte le colonne numeriche

**Output:** Matrice di correlazione come DataFrame quadrato

### 4. `filter_by_location(df, lat_min, lat_max, lon_min, lon_max) -> pd.DataFrame`

**Input:**
- DataFrame pulito
- Limiti geografici (latitudine e longitudine min/max)

**Comportamento:**
- Filtra le righe dove `Latitude` è tra `lat_min` e `lat_max` (inclusi)
- Filtra le righe dove `Longitude` è tra `lon_min` e `lon_max` (inclusi)

**Output:** DataFrame filtrato

### 5. `save_value_vs_income_plot(df: pd.DataFrame, output_path: str) -> None`

**Input:**
- DataFrame pulito
- Percorso del file di output (es. `"scatter_plot.png"`)

**Comportamento:**
- Crea un grafico scatter con `MedInc` (asse X) e `MedHouseVal` (asse Y)
- Aggiunge titolo ed etichette agli assi
- Salva il grafico nel percorso specificato
- Chiude la figura dopo il salvataggio

**Output:** Nessuno (la funzione salva un file)

---

## Requisiti

- Python 3.10 o superiore (consigliato 3.11)
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `scikit-learn>=1.3.0`
- `pytest>=7.4.0`

---

## Struttura del repository

```
hw3-pandas-houses/
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # Fixtures pytest per i test
│   └── test_data_analysis_public.py     # Test pubblici
├── data_analysis.py                     # FILE DA COMPLETARE
├── requirements.txt                     # Dipendenze
├── pytest.ini                           # Configurazione pytest
└── README.md
```

---

## Setup locale (consigliato)

1) Naviga nella directory del progetto:

```bash
cd hw3-pandas-houses
```

2) Crea ed attiva un ambiente virtuale con `uv` (consigliato):

```bash
uv venv  # solo la prima volta
source .venv/bin/activate  # su Linux/macOS
# oppure .venv\Scripts\activate su Windows
```

Da Visual Studio Code, puoi selezionare il virtual environment `.venv` come interprete Python
dalla barra di stato in basso a destra.

3) Installa le dipendenze con `uv`:

```bash 
uv pip install -r requirements.txt  # solo la prima volta
```

4) Esegui i test in locale con `uv`:

```bash
uv run pytest  # ogni volta che vuoi verificare le tue implementazioni
```

Finché non implementi le funzioni, vedrai dei fallimenti nei test (FAIL).

---

## Suggerimenti per l'implementazione

### Caricare il dataset

```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
# housing.data contiene le feature (ndarray)
# housing.target contiene il target (ndarray)
# housing.feature_names contiene i nomi delle colonne
```

### Creare un DataFrame

```python
import pandas as pd

df = pd.DataFrame(data, columns=feature_names)
df['MedHouseVal'] = target
```

### Raggruppare e calcolare medie

```python
# Raggruppare per colonna e calcolare media di un'altra
result = df.groupby('colonna_gruppo')['colonna_valore'].mean()

# Ordinare l'indice
result = result.sort_index()
```

### Matrice di correlazione

```python
correlation_matrix = df.corr()
```

### Filtrare un DataFrame

```python
# Filtrare con condizioni multiple
filtered = df[(df['col1'] >= min_val) & (df['col1'] <= max_val)]
```

### Creare un grafico scatter

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['x_column'], df['y_column'], alpha=0.5)
plt.xlabel('Label X')
plt.ylabel('Label Y')
plt.title('Titolo del Grafico')
plt.savefig(output_path)
plt.close()
```

---

## Esecuzione dei test

Per eseguire tutti i test:

```bash
uv run pytest
```

Per eseguire test di una specifica classe:

```bash
uv run pytest tests/test_data_analysis_public.py::TestLoadCaliforniaHousingData
```

Per eseguire un singolo test:

```bash
uv run pytest tests/test_data_analysis_public.py::TestLoadCaliforniaHousingData::test_returns_dataframe
```

Per vedere output più dettagliato:

```bash
uv run pytest -v --tb=long
```

---

## Valutazione

Il tuo codice sarà valutato automaticamente tramite i test pytest forniti.
Assicurati che tutti i test passino prima di consegnare.

I test verificano:
- ✅ Corretto caricamento e pulizia dei dati
- ✅ Calcolo accurato delle medie raggruppate
- ✅ Matrice di correlazione valida
- ✅ Filtro geografico funzionante
- ✅ Creazione corretta dei grafici

---

## Note

- Il dataset California Housing non contiene valori mancanti di default,
  ma la tua funzione deve comunque gestire questa eventualità
- I valori di `MedHouseVal` sono in centinaia di migliaia di dollari
  (es. 2.5 = $250,000)
- Usa il backend non interattivo `'Agg'` per Matplotlib per evitare
  problemi in ambiente di test:
  ```python
  import matplotlib
  matplotlib.use('Agg')
  ```
