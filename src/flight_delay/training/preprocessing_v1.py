"""
preprocessing.py

Módulo dedicado exclusivamente à limpeza e preparação de atributos
para o pipeline de previsão de atrasos (BTS).
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


NUMERIC_FEATURES: List[str] = [
    'QUARTER',
    'MONTH',
    'DAYOFMONTH',
    'DAYOFWEEK',
    'CRSELAPSEDTIME',
    'DISTANCE',
    'DISTANCEGROUP',
    'CRSDEPHOUR',
    'CRSDEPMINUTE',
    'CRSARRHOUR',
    'CRSARRMINUTE',
    'MILESPERMINUTE'
]

CATEGORICAL_FEATURES: List[str] = [
    'ORIGINAIRPORTID',
    'DESTAIRPORTID',
    'REPORTING_AIRLINE',
    'ORIGINCITYMARKETID',
    'DESTCITYMARKETID',
    'ORIGINSTATEFIPS',
    'DESTSTATEFIPS'
]

TARGET_COLUMN: str = 'TARGET'


def load_dataset(path: Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    """Carrega o dataset a partir de um ficheiro CSV ou Parquet."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro não encontrado: {path.resolve()}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix == ".csv":
        return pd.read_csv(path, usecols=columns)

    raise ValueError(f"Formato não suportado: {suffix} (usa .csv ou .parquet)")


def upper_columns_names(df: pd.DataFrame) -> pd.DataFrame:
    """Converte os nomes das colunas para maiúsculas."""
    df_upper = df.copy()
    df_upper.columns = df_upper.columns.str.upper()
    return df_upper


# Fazer antes de dar UPPER às colunas
def add_target(
    df: pd.DataFrame,
    delay_column: str = 'ArrDelay'
) -> pd.DataFrame:
    """Adiciona a classe de atraso à coluna definida por TARGET_COLUMN."""
    if delay_column not in df.columns:
        #raise KeyError(f"Coluna de atraso não encontrada: {delay_column}")
        return df
    

    df_with_target = df.copy()
    df_with_target[TARGET_COLUMN] = np.select(
        [
            df_with_target[delay_column] < 15,
            df_with_target[delay_column] < 60
        ],
        [0, 1],
        default=2
    )
    df_with_target.drop(columns=['ArrDelay'],inplace=True)
    df_with_target.drop(columns=['Unnamed: 109'],inplace=True,errors='ignore')
    return df_with_target


def clean_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove inconsistências de integridade física (ex: CRSElapsedTime <= 0)
    e reseta o índice mantendo a ordenação temporal.
    """
    cleaned_df = df[df["CRSELAPSEDTIME"] > 0].copy()
    cleaned_df = cleaned_df[cleaned_df['CANCELLED']!=1]
    cleaned_df = cleaned_df[cleaned_df['DIVERTED']!=1]
    cleaned_df.drop(columns=['CANCELLED','DIVERTED'],inplace=True)
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df

def type_transformation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['DISTANCE'] = df['DISTANCE'].astype("int64")
    df['CRSELAPSEDTIME'] = df['CRSELAPSEDTIME'].astype("int64")
    df['REPORTING_AIRLINE'] = df['REPORTING_AIRLINE'].astype("category")
    df['ORIGINAIRPORTID'] = df['ORIGINAIRPORTID'].astype("category")
    df['ORIGINCITYMARKETID'] = df['ORIGINCITYMARKETID'].astype("category")
    df['ORIGINSTATEFIPS'] = df['ORIGINSTATEFIPS'].astype("category")
    df['DESTAIRPORTID'] = df['DESTAIRPORTID'].astype("category")
    df['DESTCITYMARKETID'] = df['DESTCITYMARKETID'].astype("category")
    df['DESTSTATEFIPS'] = df['DESTSTATEFIPS'].astype("category")
    return df


def data_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['MILESPERMINUTE'] = df['DISTANCE'] / df['CRSELAPSEDTIME']

    df['CRSARRHOUR'] = df['CRSARRTIME'] // 100
    df['CRSDEPHOUR'] = df['CRSDEPTIME'] // 100
    df['CRSARRMINUTE'] = df['CRSARRTIME'] % 100
    df['CRSDEPMINUTE'] = df['CRSDEPTIME'] % 100
    df['CRSDEPHOUR'] = df['CRSDEPHOUR'].astype("int64")
    df['CRSDEPMINUTE'] = df['CRSDEPMINUTE'].astype("int64")
    df['CRSARRHOUR'] = df['CRSARRHOUR'].astype("int64")
    df['CRSARRMINUTE'] = df['CRSARRMINUTE'].astype("int64")
    df.drop(columns=['CRSDEPTIME','CRSARRTIME'],inplace=True)
    return df

# calcular features com informação de voos anteriores (com classes ou atraso médio(continuo)) - fazer diferentes médias eventualmente (3 a 10 voos por exemplo)
# procurar features que capturem dimensao temporal - ver padrões habituais e depois ultimos valores


def prepare_data(df: pd.DataFrame,treino:bool=True) -> pd.DataFrame:
    """
    Executa a sequência completa de pré-processamento:
      1. Adiciona a coluna TARGET.
      2. Remove inconsistências de integridade física.
      3. Converte os tipos de dados para otimização de memória.
      4. Cria novos atributos derivados.
      5. Converte os nomes das colunas para maiúsculas.
    """
    if treino:
        df_prepared = add_target(df)
    df_prepared = upper_columns_names(df_prepared)
    df_prepared = clean_flight_data(df_prepared)
    df_prepared = type_transformation(df_prepared)
    df_prepared = data_engineering(df_prepared)
    
    return df_prepared

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