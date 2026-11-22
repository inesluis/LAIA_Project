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
from functools import lru_cache
import airportsdata


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
DELAY_COLUMN: str = 'ARRDELAY'

# Configuração das features de histórico: (prefixo, chaves de agrupamento, janelas)
HISTORY_GROUPS: List[tuple[str, List[str], List[int]]] = [
    ('ORIG', ['ORIGINAIRPORTID'], [5, 10, 20, 50]),
    ('AIRLINE', ['REPORTING_AIRLINE'], [5, 10, 20, 50]),
    ('ROUTE', ['ORIGINAIRPORTID', 'DESTAIRPORTID'], [5, 10, 20]),
]


def history_feature_names() -> List[str]:
    """Nomes de todas as features de histórico, pela ordem em que são criadas."""
    names: List[str] = []
    for prefix, _, windows in HISTORY_GROUPS:
        names += [f'{prefix}_SEVERE_LAST{w}' for w in windows]
        names += [f'{prefix}_MEANDELAY_LAST{w}' for w in windows]
        names += [f'{prefix}_MEANDELAY_DIFF5_{w}' for w in windows if w > 5]
    return names


# Atraso médio de todos os voos já aterrados com a mesma hora de partida prevista (CRSDEPHOUR);
# % de voos com atraso (TARGET >= 1) já aterrados, do mesmo aeroporto de origem, no dia do voo e no anterior
CONTEXT_FEATURES: List[str] = [
    'DEPHOUR_MEANDELAY_ALL',
    'ORIG_PCT_DELAYED_TODAY',
    'ORIG_PCT_DELAYED_YESTERDAY',
]

HISTORY_FEATURES: List[str] = history_feature_names() + CONTEXT_FEATURES

# Encodings cíclicos (seno/cosseno) do mês, dia da semana e hora de partida prevista
CYCLIC_FEATURES: List[str] = [
    'MONTH_SIN', 'MONTH_COS',
    'DAYOFWEEK_SIN', 'DAYOFWEEK_COS',
    'CRSDEPTIME_SIN', 'CRSDEPTIME_COS',
]

#CYCLIC FEATURES experimentar numa experiência 2.1

NUMERIC_FEATURES = NUMERIC_FEATURES + HISTORY_FEATURES 


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
    # O ArrDelay é mantido para as features de histórico; é removido no data_engineering.
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
    # Só no treino (na inferência não há ARRDELAY):
    #  - ARRDELAY em falta: o TARGET ficaria 2 por omissão no np.select;
    #  - chegada real antes da partida prevista (ARRDELAY < -CRSELAPSEDTIME): erro nos dados.
    if DELAY_COLUMN in cleaned_df.columns:
        delay = cleaned_df[DELAY_COLUMN]
        cleaned_df = cleaned_df[delay.notna() & (delay + cleaned_df['CRSELAPSEDTIME'] >= 0)]
    cleaned_df.drop(columns=['CANCELLED','DIVERTED'],inplace=True,errors='ignore')
    # ARRTIME é a hora real de chegada: só se sabe depois do voo -> não pode chegar ao modelo
    cleaned_df.drop(columns=['ARRTIME'],inplace=True,errors='ignore')
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


def _sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena os voos por data e hora de partida prevista (ordenação estável)."""
    if 'FLIGHTDATE' in df.columns:
        date_cols = ['FLIGHTDATE']
    else:
        date_cols = ['YEAR', 'MONTH', 'DAYOFMONTH']
    return df.sort_values(by=date_cols + ['CRSDEPTIME'], kind='mergesort').reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_airports() -> dict:
    """Base de dados de aeroportos (IATA -> info, incluindo o fuso 'tz')."""
    return airportsdata.load('IATA')


def _flight_dates(df: pd.DataFrame) -> pd.Series:
    """Data (local, da partida) de cada voo."""
    if 'FLIGHTDATE' in df.columns:
        return pd.to_datetime(df['FLIGHTDATE'])
    return pd.to_datetime(pd.DataFrame({'year': df['YEAR'], 'month': df['MONTH'], 'day': df['DAYOFMONTH']}))


def _flight_times_utc(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Devolve (partida prevista, chegada real) de cada voo em minutos UTC desde 2000-01-01.

    - A partida prevista (hora local da origem) é convertida para UTC com o fuso do aeroporto
      de origem (coluna ORIGIN, código IATA), via airportsdata -> trata mudanças de hora.
    - A chegada real = partida prevista + CRSELAPSEDTIME + ARRDELAY (não depende da hora
      local do destino nem de voos que passam a meia-noite).
    Voos com aeroporto sem fuso conhecido ficam a NaN.
    """
    dates = _flight_dates(df)
    hhmm = df['CRSDEPTIME'].astype('int64')
    # 2400 = meia-noite do dia seguinte, tratado naturalmente pelo timedelta
    dep_local = (dates + pd.to_timedelta(hhmm // 100, unit='h') + pd.to_timedelta(hhmm % 100, unit='m')).reset_index(drop=True)

    airports = _load_airports()
    origin_tz = df['ORIGIN'].astype(str).map(lambda code: airports.get(code, {}).get('tz')).reset_index(drop=True)

    epoch = pd.Timestamp('2000-01-01', tz='UTC')
    sched_dep = np.full(len(df), np.nan)
    for tz, idx in origin_tz.groupby(origin_tz).groups.items():
        local = dep_local.loc[idx]
        # Hora repetida na mudança para inverno: assume-se hora de inverno (erro máx. 1h, 1h/ano)
        utc = local.dt.tz_localize(tz, ambiguous=np.zeros(len(local), dtype=bool), nonexistent='shift_forward')
        sched_dep[idx.to_numpy()] = ((utc - epoch) / pd.Timedelta(minutes=1)).to_numpy()

    real_arr = sched_dep + df['CRSELAPSEDTIME'].to_numpy(dtype='float64') + df[DELAY_COLUMN].to_numpy(dtype='float64')
    # Salvaguarda (o clean_flight_data já remove estes casos): um voo nunca pode "aterrar" antes
    # da própria partida prevista, senão entraria na sua própria janela (fuga do próprio target).
    # np.maximum propaga NaN: voos sem ARRDELAY ou sem fuso ficam fora do histórico
    real_arr = np.maximum(real_arr, sched_dep)
    return sched_dep, real_arr


def _rolling_landed(
    values: dict[str, np.ndarray],
    group_id: np.ndarray,
    sched_dep: np.ndarray,
    real_arr: np.ndarray,
    windows: List[Optional[int]],
    query_group_id: Optional[np.ndarray] = None,
) -> dict[str, dict[Optional[int], tuple[np.ndarray, np.ndarray]]]:
    """
    Para cada voo i e janela w: (soma, nº de voos) dos últimos w voos do mesmo grupo que
    JÁ ATERRARAM antes da partida prevista de i (real_arr_j < sched_dep_i). Assim só se usa
    informação que estaria disponível no momento da previsão.
    w = None -> todos os voos do grupo já aterrados (média "expanding").
    query_group_id permite que o voo i consulte outro grupo que não o seu (ex.: o dia anterior);
    por omissão é o próprio group_id. Os ids têm de estar no mesmo espaço numérico.

    Algoritmo: ordenam-se as aterragens por (grupo, hora de chegada) e calculam-se somas
    cumulativas; para cada voo, um searchsorted dá quantas aterragens do grupo já ocorreram.
    """
    if query_group_id is None:
        query_group_id = group_id
    landed = ~np.isnan(real_arr)
    l_gid = group_id[landed]
    l_arr = real_arr[landed]
    order = np.lexsort((l_arr, l_gid))
    l_gid, l_arr = l_gid[order], l_arr[order]

    # Chave lexicográfica (grupo, tempo) num único float: tempos ficam bem abaixo de 1e8 minutos
    scale = 1e8
    l_key = l_gid * scale + l_arr
    q_key = query_group_id * scale + sched_dep

    group_start = np.searchsorted(l_key, query_group_id * scale, side='left')
    n_landed = np.searchsorted(l_key, q_key, side='left')   # aterragens estritamente antes da partida
    # Voos sem hora de partida conhecida (fuso desconhecido) não têm histórico
    n_landed = np.where(np.isnan(sched_dep), group_start, n_landed)

    result: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for name, vals in values.items():
        cum = np.concatenate([[0.0], np.cumsum(vals[landed][order])])
        result[name] = {}
        for w in windows:
            lo = group_start if w is None else np.maximum(n_landed - w, group_start)
            result[name][w] = (cum[n_landed] - cum[lo], (n_landed - lo).astype('float64'))
    return result


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona, para cada grupo em HISTORY_GROUPS (aeroporto de origem, companhia, rota):
      - nº de atrasos severos (TARGET == 2) nos últimos w voos já aterrados;
      - atraso médio (ARRDELAY) nos últimos w voos já aterrados;
      - diferença entre o atraso médio dos últimos 5 voos e o dos últimos w voos (w > 5).
    E as features de contexto (CONTEXT_FEATURES, ver _context_features).
    "Já aterrados" = chegada real anterior à partida prevista do voo atual.
    Se não houver ARRDELAY/TARGET (inferência), as colunas são criadas a NaN.
    """
    df = df.copy()
    has_history = DELAY_COLUMN in df.columns and TARGET_COLUMN in df.columns
    if not has_history:
        for name in HISTORY_FEATURES:
            df[name] = np.nan
        return df

    sched_dep, real_arr = _flight_times_utc(df)
    values = {
        'severe': (df[TARGET_COLUMN] == 2).to_numpy(dtype='float64'),
        'delay': df[DELAY_COLUMN].to_numpy(dtype='float64'),
    }

    new_cols = {}
    for prefix, group_cols, windows in HISTORY_GROUPS:
        group_id = df.groupby(group_cols, observed=True, sort=False).ngroup().to_numpy().astype('float64')
        rolled = _rolling_landed(values, group_id, sched_dep, real_arr, windows)

        means = {}
        for w in windows:
            sev_sum, cnt = rolled['severe'][w]
            d_sum, _ = rolled['delay'][w]
            has_data = cnt > 0                                  # NaN quando nenhum voo aterrou ainda
            new_cols[f'{prefix}_SEVERE_LAST{w}'] = np.where(has_data, sev_sum, np.nan)
            means[w] = np.where(has_data, d_sum / np.where(has_data, cnt, 1), np.nan)
            new_cols[f'{prefix}_MEANDELAY_LAST{w}'] = means[w]

        for w in windows:
            if w > 5:
                new_cols[f'{prefix}_MEANDELAY_DIFF5_{w}'] = means[5] - means[w]

    new_cols.update(_context_features(df, sched_dep, real_arr, values['delay']))

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)[HISTORY_FEATURES]], axis=1)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, com NaN quando den == 0 (ainda não há voos aterrados)."""
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _context_features(
    df: pd.DataFrame,
    sched_dep: np.ndarray,
    real_arr: np.ndarray,
    delay: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Features de contexto, sempre só com voos já aterrados antes da partida prevista do voo atual:
      - DEPHOUR_MEANDELAY_ALL: atraso médio de todos os voos (de qualquer aeroporto) com a mesma
        hora de partida prevista (CRSDEPHOUR, hora local), desde o início dos dados.
      - ORIG_PCT_DELAYED_TODAY / _YESTERDAY: % de voos com TARGET >= 1 entre os voos que partiram
        do mesmo aeroporto de origem no dia do voo / no dia anterior (data local de partida).
    """
    cols: dict[str, np.ndarray] = {}

    # 1) Atraso médio por hora de partida prevista (expanding)
    dep_hour = (df['CRSDEPTIME'].astype('int64') // 100).to_numpy().astype('float64')
    rolled = _rolling_landed({'delay': delay}, dep_hour, sched_dep, real_arr, [None])
    d_sum, cnt = rolled['delay'][None]
    cols['DEPHOUR_MEANDELAY_ALL'] = _safe_ratio(d_sum, cnt)

    # 2) % de voos atrasados por (aeroporto de origem, dia)
    delayed = (df[TARGET_COLUMN] >= 1).to_numpy(dtype='float64')
    day = (_flight_dates(df) - pd.Timestamp('2000-01-01')).dt.days.to_numpy()
    origin = df['ORIGINAIRPORTID'].astype('int64').to_numpy()
    # Chave (aeroporto, dia) -> id compacto partilhado pelas aterragens e pelas consultas
    own_key = origin * 100_000 + day
    prev_key = own_key - 1                       # mesmo aeroporto, dia anterior
    _, ids = np.unique(np.concatenate([own_key, prev_key]), return_inverse=True)
    own_id = ids[:len(df)].astype('float64')
    prev_id = ids[len(df):].astype('float64')

    for suffix, query_id in [('TODAY', own_id), ('YESTERDAY', prev_id)]:
        rolled = _rolling_landed({'delayed': delayed}, own_id, sched_dep, real_arr, [None],
                                 query_group_id=query_id)
        n_delayed, cnt = rolled['delayed'][None]
        cols[f'ORIG_PCT_DELAYED_{suffix}'] = 100 * _safe_ratio(n_delayed, cnt)
    return cols


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodings cíclicos (seno/cosseno), para que dezembro fique perto de janeiro, domingo perto de
    segunda e 23h perto de 0h. Requer CRSDEPHOUR/CRSDEPMINUTE já calculados.
    """
    df = df.copy()
    cycles = {
        'MONTH': (df['MONTH'].astype('float64') - 1, 12),                # 1..12 -> 0..11
        'DAYOFWEEK': (df['DAYOFWEEK'].astype('float64') - 1, 7),         # 1..7  -> 0..6
        'CRSDEPTIME': (df['CRSDEPHOUR'] + df['CRSDEPMINUTE'] / 60, 24),  # hora do dia, com minutos
    }
    for name, (value, period) in cycles.items():
        angle = 2 * np.pi * value / period
        df[f'{name}_SIN'] = np.sin(angle)
        df[f'{name}_COS'] = np.cos(angle)
    return df


def data_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = _sort_chronologically(df)
    df = add_history_features(df)
    df.drop(columns=[DELAY_COLUMN], inplace=True, errors='ignore')

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
    df = add_cyclic_features(df)
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
    else:
        df_prepared = upper_columns_names(df)
    
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