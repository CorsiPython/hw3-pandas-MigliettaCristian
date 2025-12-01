"""
Configurazione pytest per i test dell'Homework 3.

Questo file definisce fixtures comuni utilizzate dai test.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_california_housing


@pytest.fixture(scope="session")
def california_data():
    """Carica il dataset California Housing una volta per tutta la sessione di test.
    
    Returns
    -------
    tuple
        (data, target, feature_names) dal dataset sklearn
    """
    housing = fetch_california_housing()
    return housing.data, housing.target, housing.feature_names


@pytest.fixture
def reference_df(california_data):
    """Crea un DataFrame di riferimento per i test.
    
    Questo DataFrame è costruito in modo indipendente dalla funzione dello studente,
    per poter verificare che la loro implementazione sia corretta.
    
    Returns
    -------
    pd.DataFrame
        DataFrame di riferimento con tutte le colonne corrette
    """
    data, target, feature_names = california_data
    df = pd.DataFrame(data, columns=feature_names)
    df['MedHouseVal'] = target
    df = df.dropna()
    return df


@pytest.fixture
def sample_df():
    """Crea un piccolo DataFrame di esempio per test unitari veloci.
    
    Returns
    -------
    pd.DataFrame
        DataFrame ridotto per test rapidi
    """
    return pd.DataFrame({
        'MedInc': [3.5, 5.0, 8.2, 2.1, 6.5],
        'HouseAge': [20.0, 35.0, 20.0, 15.0, 35.0],
        'AveRooms': [5.0, 6.0, 7.0, 4.0, 6.5],
        'AveBedrms': [1.0, 1.2, 1.1, 0.9, 1.3],
        'Population': [1000.0, 1500.0, 800.0, 2000.0, 1200.0],
        'AveOccup': [3.0, 2.5, 2.0, 4.0, 2.8],
        'Latitude': [34.0, 37.5, 38.0, 33.5, 36.0],
        'Longitude': [-118.0, -122.0, -121.5, -117.5, -120.0],
        'MedHouseVal': [2.5, 3.5, 5.0, 1.5, 4.0],
    })


@pytest.fixture
def plot_output_path(tmp_path):
    """Crea un percorso temporaneo per salvare i grafici di test.
    
    Yields
    ------
    Path
        Percorso al file temporaneo per il grafico
    """
    return str(tmp_path / "test_plot.png")
