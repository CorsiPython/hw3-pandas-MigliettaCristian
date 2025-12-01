"""
Test pubblici per l'Homework 3: Analisi Dati con Pandas e Matplotlib

Questi test verificano le funzionalità base della pipeline di analisi dati.
Gli studenti devono implementare il codice in modo che tutti i test passino.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Aggiungi la directory parent al path per importare il modulo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_analysis import (
    load_california_housing_data,
    calculate_average_value_by_age,
    get_correlation_matrix,
    filter_by_location,
    save_value_vs_income_plot,
)


# =============================================================================
# Test per load_california_housing_data
# =============================================================================

class TestLoadCaliforniaHousingData:
    """Test per la funzione load_california_housing_data."""
    
    def test_returns_dataframe(self):
        """Verifica che la funzione restituisca un DataFrame."""
        result = load_california_housing_data()
        assert isinstance(result, pd.DataFrame), \
            "La funzione deve restituire un pd.DataFrame"
    
    def test_correct_columns(self):
        """Verifica che il DataFrame abbia le colonne corrette."""
        df = load_california_housing_data()
        expected_columns = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal'
        ]
        assert list(df.columns) == expected_columns, \
            f"Colonne attese: {expected_columns}, trovate: {list(df.columns)}"
    
    def test_no_missing_values(self):
        """Verifica che non ci siano valori mancanti nel DataFrame."""
        df = load_california_housing_data()
        assert df.isna().sum().sum() == 0, \
            "Il DataFrame non deve contenere valori mancanti"
    
    def test_correct_number_of_rows(self, reference_df):
        """Verifica che il numero di righe sia corretto (uguale al dataset originale)."""
        df = load_california_housing_data()
        # Il dataset California Housing ha 20640 righe
        assert len(df) == len(reference_df), \
            f"Il DataFrame deve avere {len(reference_df)} righe, trovate {len(df)}"
    
    def test_correct_data_types(self):
        """Verifica che le colonne abbiano tipi numerici."""
        df = load_california_housing_data()
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), \
                f"La colonna '{col}' deve essere numerica"
    
    def test_medhouseval_values_reasonable(self):
        """Verifica che i valori di MedHouseVal siano nel range atteso."""
        df = load_california_housing_data()
        # I valori del target sono in centinaia di migliaia di dollari
        assert df['MedHouseVal'].min() >= 0, \
            "MedHouseVal non può essere negativo"
        assert df['MedHouseVal'].max() <= 6, \
            "MedHouseVal sembra fuori range (max atteso ~5)"


# =============================================================================
# Test per calculate_average_value_by_age
# =============================================================================

class TestCalculateAverageValueByAge:
    """Test per la funzione calculate_average_value_by_age."""
    
    def test_returns_series(self, reference_df):
        """Verifica che la funzione restituisca una Series."""
        result = calculate_average_value_by_age(reference_df)
        assert isinstance(result, pd.Series), \
            "La funzione deve restituire una pd.Series"
    
    def test_index_is_house_age(self, sample_df):
        """Verifica che l'indice della Series sia HouseAge."""
        result = calculate_average_value_by_age(sample_df)
        # Nel sample_df abbiamo età 15, 20, 35
        expected_ages = sorted(sample_df['HouseAge'].unique())
        assert list(result.index) == expected_ages, \
            f"L'indice deve contenere le età uniche ordinate: {expected_ages}"
    
    def test_correct_average_calculation(self, sample_df):
        """Verifica che la media sia calcolata correttamente."""
        result = calculate_average_value_by_age(sample_df)
        
        # Per età 20: MedHouseVal = [2.5, 5.0] -> media = 3.75
        # Per età 35: MedHouseVal = [3.5, 4.0] -> media = 3.75
        # Per età 15: MedHouseVal = [1.5] -> media = 1.5
        
        assert np.isclose(result[15.0], 1.5), \
            f"Media per età 15 attesa: 1.5, trovata: {result[15.0]}"
        assert np.isclose(result[20.0], 3.75), \
            f"Media per età 20 attesa: 3.75, trovata: {result[20.0]}"
        assert np.isclose(result[35.0], 3.75), \
            f"Media per età 35 attesa: 3.75, trovata: {result[35.0]}"
    
    def test_sorted_index(self, reference_df):
        """Verifica che l'indice sia ordinato in modo crescente."""
        result = calculate_average_value_by_age(reference_df)
        assert result.index.is_monotonic_increasing, \
            "L'indice deve essere ordinato in modo crescente"
    
    def test_no_nan_values(self, reference_df):
        """Verifica che non ci siano NaN nei risultati."""
        result = calculate_average_value_by_age(reference_df)
        assert result.isna().sum() == 0, \
            "La Series non deve contenere valori NaN"


# =============================================================================
# Test per get_correlation_matrix
# =============================================================================

class TestGetCorrelationMatrix:
    """Test per la funzione get_correlation_matrix."""
    
    def test_returns_dataframe(self, reference_df):
        """Verifica che la funzione restituisca un DataFrame."""
        result = get_correlation_matrix(reference_df)
        assert isinstance(result, pd.DataFrame), \
            "La funzione deve restituire un pd.DataFrame"
    
    def test_square_matrix(self, reference_df):
        """Verifica che la matrice sia quadrata."""
        result = get_correlation_matrix(reference_df)
        assert result.shape[0] == result.shape[1], \
            "La matrice di correlazione deve essere quadrata"
    
    def test_diagonal_is_one(self, reference_df):
        """Verifica che la diagonale sia composta da 1."""
        result = get_correlation_matrix(reference_df)
        diagonal = np.diag(result.values)
        assert np.allclose(diagonal, 1.0), \
            "La diagonale della matrice di correlazione deve essere 1"
    
    def test_symmetric(self, reference_df):
        """Verifica che la matrice sia simmetrica."""
        result = get_correlation_matrix(reference_df)
        assert np.allclose(result.values, result.values.T), \
            "La matrice di correlazione deve essere simmetrica"
    
    def test_values_in_range(self, reference_df):
        """Verifica che i valori siano nel range [-1, 1]."""
        result = get_correlation_matrix(reference_df)
        assert result.values.min() >= -1, \
            "I valori della correlazione devono essere >= -1"
        assert result.values.max() <= 1, \
            "I valori della correlazione devono essere <= 1"
    
    def test_correct_columns(self, reference_df):
        """Verifica che la matrice abbia le colonne corrette."""
        result = get_correlation_matrix(reference_df)
        assert list(result.columns) == list(reference_df.columns), \
            "Le colonne della matrice devono corrispondere a quelle del DataFrame"
    
    def test_known_correlation(self, sample_df):
        """Verifica una correlazione nota tra colonne del sample."""
        result = get_correlation_matrix(sample_df)
        # La correlazione tra MedInc e MedHouseVal nel sample dovrebbe essere positiva
        assert result.loc['MedInc', 'MedHouseVal'] > 0, \
            "La correlazione tra MedInc e MedHouseVal deve essere positiva"


# =============================================================================
# Test per filter_by_location
# =============================================================================

class TestFilterByLocation:
    """Test per la funzione filter_by_location."""
    
    def test_returns_dataframe(self, reference_df):
        """Verifica che la funzione restituisca un DataFrame."""
        result = filter_by_location(reference_df, 34.0, 35.0, -119.0, -118.0)
        assert isinstance(result, pd.DataFrame), \
            "La funzione deve restituire un pd.DataFrame"
    
    def test_same_columns(self, reference_df):
        """Verifica che il DataFrame filtrato abbia le stesse colonne."""
        result = filter_by_location(reference_df, 34.0, 35.0, -119.0, -118.0)
        assert list(result.columns) == list(reference_df.columns), \
            "Il DataFrame filtrato deve avere le stesse colonne dell'originale"
    
    def test_filter_latitude(self, sample_df):
        """Verifica che il filtro sulla latitudine funzioni."""
        # Nel sample: latitudini 33.5, 34.0, 36.0, 37.5, 38.0
        result = filter_by_location(sample_df, 36.0, 38.0, -125.0, -115.0)
        assert all(result['Latitude'] >= 36.0), \
            "Tutte le latitudini devono essere >= 36.0"
        assert all(result['Latitude'] <= 38.0), \
            "Tutte le latitudini devono essere <= 38.0"
    
    def test_filter_longitude(self, sample_df):
        """Verifica che il filtro sulla longitudine funzioni."""
        # Nel sample: longitudini -122.0, -121.5, -120.0, -118.0, -117.5
        result = filter_by_location(sample_df, 30.0, 40.0, -121.0, -118.0)
        assert all(result['Longitude'] >= -121.0), \
            "Tutte le longitudini devono essere >= -121.0"
        assert all(result['Longitude'] <= -118.0), \
            "Tutte le longitudini devono essere <= -118.0"
    
    def test_combined_filter(self, sample_df):
        """Verifica che i filtri combinati funzionino."""
        # Filtra per area Los Angeles (circa lat 33-35, lon -119 a -117)
        result = filter_by_location(sample_df, 33.0, 35.0, -119.0, -117.0)
        
        # Dovrebbe includere solo le righe con lat in [33, 35] e lon in [-119, -117]
        assert len(result) > 0, \
            "Il filtro dovrebbe restituire almeno una riga"
        assert all((result['Latitude'] >= 33.0) & (result['Latitude'] <= 35.0)), \
            "Le latitudini devono essere nel range specificato"
        assert all((result['Longitude'] >= -119.0) & (result['Longitude'] <= -117.0)), \
            "Le longitudini devono essere nel range specificato"
    
    def test_empty_result(self, sample_df):
        """Verifica che il filtro restituisca DataFrame vuoto se nessun match."""
        result = filter_by_location(sample_df, 50.0, 60.0, -100.0, -90.0)
        assert len(result) == 0, \
            "Il filtro deve restituire DataFrame vuoto se nessuna riga corrisponde"
    
    def test_inclusive_bounds(self, sample_df):
        """Verifica che i limiti siano inclusivi."""
        # C'è una riga con Latitude=34.0 e Longitude=-118.0
        result = filter_by_location(sample_df, 34.0, 34.0, -118.0, -118.0)
        assert len(result) == 1, \
            "I limiti devono essere inclusivi (trovare la riga con lat=34.0, lon=-118.0)"


# =============================================================================
# Test per save_value_vs_income_plot
# =============================================================================

class TestSaveValueVsIncomePlot:
    """Test per la funzione save_value_vs_income_plot."""
    
    def test_creates_file(self, sample_df, plot_output_path):
        """Verifica che la funzione crei il file del grafico."""
        save_value_vs_income_plot(sample_df, plot_output_path)
        assert Path(plot_output_path).exists(), \
            f"Il file {plot_output_path} non è stato creato"
    
    def test_returns_none(self, sample_df, plot_output_path):
        """Verifica che la funzione restituisca None."""
        result = save_value_vs_income_plot(sample_df, plot_output_path)
        assert result is None, \
            "La funzione deve restituire None"
    
    def test_file_not_empty(self, sample_df, plot_output_path):
        """Verifica che il file creato non sia vuoto."""
        save_value_vs_income_plot(sample_df, plot_output_path)
        file_size = Path(plot_output_path).stat().st_size
        assert file_size > 0, \
            "Il file del grafico non deve essere vuoto"
    
    def test_valid_image_file(self, sample_df, plot_output_path):
        """Verifica che il file sia un'immagine PNG valida."""
        save_value_vs_income_plot(sample_df, plot_output_path)
        
        # Verifica che il file abbia l'header PNG
        with open(plot_output_path, 'rb') as f:
            header = f.read(8)
        
        # PNG signature
        png_signature = b'\x89PNG\r\n\x1a\n'
        assert header == png_signature, \
            "Il file deve essere un'immagine PNG valida"
    
    def test_different_output_paths(self, sample_df, tmp_path):
        """Verifica che la funzione funzioni con percorsi diversi."""
        paths = [
            str(tmp_path / "plot1.png"),
            str(tmp_path / "subdir" / "plot2.png"),
        ]
        
        # Crea la sottodirectory
        (tmp_path / "subdir").mkdir()
        
        for path in paths:
            save_value_vs_income_plot(sample_df, path)
            assert Path(path).exists(), \
                f"Il file {path} non è stato creato"


# =============================================================================
# Test di integrazione
# =============================================================================

class TestIntegration:
    """Test di integrazione che verificano il flusso completo."""
    
    def test_full_pipeline(self, plot_output_path):
        """Verifica che l'intera pipeline funzioni correttamente."""
        # Carica i dati
        df = load_california_housing_data()
        assert len(df) > 0, "DataFrame vuoto"
        
        # Calcola media per età
        avg_by_age = calculate_average_value_by_age(df)
        assert len(avg_by_age) > 0, "Series vuota"
        
        # Calcola correlazione
        corr = get_correlation_matrix(df)
        assert corr.shape[0] == len(df.columns), "Dimensione correlazione errata"
        
        # Filtra per San Francisco (circa)
        sf_df = filter_by_location(df, 37.5, 38.0, -122.5, -122.0)
        assert len(sf_df) > 0, "Nessuna casa trovata nell'area di San Francisco"
        
        # Crea grafico
        save_value_vs_income_plot(df, plot_output_path)
        assert Path(plot_output_path).exists(), "Grafico non creato"
    
    def test_filtered_analysis(self, plot_output_path):
        """Verifica che si possa analizzare un subset filtrato."""
        df = load_california_housing_data()
        
        # Filtra per area di Los Angeles
        la_df = filter_by_location(df, 33.5, 34.5, -118.5, -117.5)
        
        # Calcola statistiche sul subset
        if len(la_df) > 0:
            avg_by_age = calculate_average_value_by_age(la_df)
            corr = get_correlation_matrix(la_df)
            
            assert len(avg_by_age) > 0, "Media per età vuota per LA"
            assert corr.shape[0] == len(df.columns), "Correlazione errata per LA"
