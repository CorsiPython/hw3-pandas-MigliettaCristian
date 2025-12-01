"""
Homework 3: Analisi Dati e Visualizzazione con Pandas e Matplotlib

Questo modulo implementa una pipeline di analisi dati esplorativa (EDA) sul dataset
California Housing. Utilizzerai Pandas per caricare, pulire e analizzare i dati,
e Matplotlib per creare visualizzazioni grafiche.

Il dataset California Housing contiene informazioni sulle case in California,
con le seguenti features principali:
- MedInc: Reddito mediano nel blocco (in decine di migliaia di dollari)
- HouseAge: Età mediana delle case nel blocco
- AveRooms: Numero medio di stanze per abitazione
- AveBedrms: Numero medio di camere da letto per abitazione
- Population: Popolazione del blocco
- AveOccup: Numero medio di occupanti per abitazione
- Latitude: Latitudine del blocco
- Longitude: Longitudine del blocco
- MedHouseVal: Valore mediano delle case (target, in centinaia di migliaia di dollari)

Le funzioni principali da implementare sono:
- load_california_housing_data: Carica e pulisce il dataset
- calculate_average_value_by_age: Calcola il valore medio per età della casa
- get_correlation_matrix: Calcola la matrice di correlazione
- filter_by_location: Filtra i dati per posizione geografica
- save_value_vs_income_plot: Crea e salva un grafico scatter

Mantieni le firme delle funzioni esattamente come definite: i test automatici le importano.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non interattivo per salvare grafici
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing


__all__ = [
    "load_california_housing_data",
    "calculate_average_value_by_age",
    "get_correlation_matrix",
    "filter_by_location",
    "save_value_vs_income_plot",
]


def load_california_housing_data() -> pd.DataFrame:
    """Carica il dataset California Housing e lo restituisce come DataFrame pulito.
    
    Comportamento:
    - Carica il dataset usando fetch_california_housing di sklearn
    - Crea un DataFrame con le feature e il target (MedHouseVal)
    - Controlla la presenza di valori mancanti e, se presenti, rimuove le righe
    - I nomi delle colonne devono essere quelli originali del dataset più 'MedHouseVal'
    
    Returns
    -------
    pd.DataFrame
        DataFrame pulito con le colonne: MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude, MedHouseVal
    """
    # TODO: Implementa questa funzione
    raise NotImplementedError("Implementa load_california_housing_data")


def calculate_average_value_by_age(df: pd.DataFrame) -> pd.Series:
    """Calcola il valore medio delle case raggruppando per età della casa.
    
    Parameters
    ----------
    df : pd.DataFrame
        Il DataFrame pulito ottenuto da load_california_housing_data()
    
    Comportamento:
    - Raggruppa i dati per 'HouseAge' (età della casa, arrotondato all'intero)
    - Calcola la media di 'MedHouseVal' per ogni gruppo
    - Ordina l'indice in ordine crescente
    
    Returns
    -------
    pd.Series
        Serie con 'HouseAge' come indice e il valore medio delle case come valori
    """
    # TODO: Implementa questa funzione
    raise NotImplementedError("Implementa calculate_average_value_by_age")


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola la matrice di correlazione tra tutte le colonne numeriche.
    
    Parameters
    ----------
    df : pd.DataFrame
        Il DataFrame pulito ottenuto da load_california_housing_data()
    
    Comportamento:
    - Calcola la matrice di correlazione di Pearson per tutte le colonne numeriche
    
    Returns
    -------
    pd.DataFrame
        Matrice di correlazione come DataFrame quadrato
    """
    # TODO: Implementa questa funzione
    raise NotImplementedError("Implementa get_correlation_matrix")


def filter_by_location(
    df: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> pd.DataFrame:
    """Filtra il DataFrame per area geografica specificata.
    
    Parameters
    ----------
    df : pd.DataFrame
        Il DataFrame pulito ottenuto da load_california_housing_data()
    lat_min : float
        Latitudine minima (inclusa)
    lat_max : float
        Latitudine massima (inclusa)
    lon_min : float
        Longitudine minima (inclusa)
    lon_max : float
        Longitudine massima (inclusa)
    
    Comportamento:
    - Filtra le righe dove Latitude è tra lat_min e lat_max (inclusi)
    - Filtra le righe dove Longitude è tra lon_min e lon_max (inclusi)
    
    Returns
    -------
    pd.DataFrame
        DataFrame filtrato contenente solo le righe nell'area specificata
    """
    # TODO: Implementa questa funzione
    raise NotImplementedError("Implementa filter_by_location")


def save_value_vs_income_plot(df: pd.DataFrame, output_path: str) -> None:
    """Crea e salva un grafico scatter del valore delle case vs reddito mediano.
    
    Parameters
    ----------
    df : pd.DataFrame
        Il DataFrame pulito ottenuto da load_california_housing_data()
    output_path : str
        Percorso dove salvare l'immagine (es. "scatter_plot.png")
    
    Comportamento:
    - Crea un grafico a dispersione (scatter plot) con:
      - Asse X: MedInc (reddito mediano)
      - Asse Y: MedHouseVal (valore mediano delle case)
    - Aggiunge titolo ed etichette agli assi
    - Salva il grafico nel percorso specificato
    - Chiude la figura dopo il salvataggio per liberare memoria
    
    Returns
    -------
    None
        La funzione salva un file, non restituisce nulla
    """
    # TODO: Implementa questa funzione
    raise NotImplementedError("Implementa save_value_vs_income_plot")
