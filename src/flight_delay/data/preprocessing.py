"""
preprocessing.py

Módulo dedicado exclusivamente à limpeza e preparação de atributos
para o pipeline de previsão de atrasos (BTS).
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


NUMERIC_FEATURES: List[str] = [
    'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'CRSElapsedTime',
    'Distance', 'DistanceGroup', 'CRSDepHour', 'CRSDepMinute',
    'CRSArrHour', 'CRSArrMinute'
]

CATEGORICAL_FEATURES: List[str] = [
    'OriginAirportID', 'DestAirportID', 'Reporting_Airline',
    'OriginCityMarketID', 'DestCityMarketID', 'OriginStateFips', 'DestStateFips'
]

TARGET_COLUMN: str = 'Target'


def load_dataset(csv_path: Path | str) -> pd.DataFrame:
    """Carrega o ficheiro CSV a partir do caminho indicado."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro não encontrado: {path.resolve()}")
    return pd.read_csv(path)


def clean_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove inconsistências de integridade física (ex: CRSElapsedTime <= 0)
    e reseta o índice mantendo a ordenação temporal.
    """
    cleaned_df = df[df["CRSElapsedTime"] > 0].copy()
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df


def build_preprocessor(
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer do Scikit-Learn:
      - StandardScaler para atributos contínuos.
      - OrdinalEncoder com tolerância a categorias não vistas (-1) para atributos categóricos.
    """
    num_cols = numeric_cols if numeric_cols is not None else NUMERIC_FEATURES
    cat_cols = categorical_cols if categorical_cols is not None else CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
        ],
        remainder='drop'
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor