"""
finetune_v2.py

Otimização de hiperparâmetros do XGBoost (Optuna + TimeSeriesSplit) com o pré-processamento v2.
No fim, os melhores parâmetros são guardados em best_params_v2.json, que o train_v2.py lê.

Correr a partir de qualquer pasta com:  python src/flight_delay/training/finetune_v2.py
Variáveis de ambiente opcionais:
  DATA_PATH   caminho do dataset (por omissão <raiz>/data/dataset_completo.v2.parquet)
  XGB_DEVICE  'cuda' (omissão) ou 'cpu'
  N_TRIALS    nº de trials a correr nesta execução (omissão 10)
"""

import gc
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from preprocessing.preprocessing_v2 import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    load_dataset,
    prepare_data,
)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]


def version_from_filename(path: str | Path) -> str:
    """Extrai a versão do nome do ficheiro (ex.: 'finetune_v2.py' -> 'v2')."""
    match = re.search(r"_(v\d+)$", Path(path).stem)
    if match is None:
        raise ValueError(f"O ficheiro '{Path(path).name}' não segue o padrão '<nome>_v<N>.py'")
    return match.group(1)


# A versão vem do nome do ficheiro: estudo do Optuna e JSON de parâmetros ficam alinhados com o train_vN
VERSION = version_from_filename(__file__)
BEST_PARAMS_PATH = BASE_DIR / f"best_params_{VERSION}.json"

# Estudo próprio por versão: trials feitos com outras features não se misturam com estes
STORAGE = f"sqlite:///{(REPO_ROOT / 'bts_xgboost_optimization.db').as_posix()}"
STUDY_NAME = f"xgboost_timeseries_f1_{VERSION}"

N_SPLITS = 4
N_TRIALS = int(os.environ.get("N_TRIALS", 10))
# GPU por omissão para acelerar a pesquisa; os parâmetros encontrados servem igualmente para o
# treino final em CPU (train_v2.py). Usar XGB_DEVICE=cpu numa máquina sem GPU.
DEVICE = os.environ.get("XGB_DEVICE", "cuda")


# ==============================================================================
# 1. DADOS
# ==============================================================================

def resolve_data_path() -> Path:
    env_path = os.environ.get("DATA_PATH")
    if env_path:
        return Path(env_path)
    return REPO_ROOT / "data" / "dataset_completo.v2.parquet"


def select_dev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fica só com o período de desenvolvimento (2022 e 1.º semestre de 2023); os últimos 6 meses
    de 2023 são o teste cego final e nunca entram no fine-tune.
    ATENÇÃO: tem de coincidir com o corte do split_data do train_v2.py.
    O df já vem por ordem cronológica do prepare_data e a máscara mantém essa ordem
    (o TimeSeriesSplit depende dela).
    """
    mask_dev = (df["YEAR"] == 2022) | ((df["YEAR"] == 2023) & (df["MONTH"] <= 6))
    return df.loc[mask_dev].reset_index(drop=True)


def load_dev_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    t0 = time.time()
    # As features de histórico são calculadas sobre o dataset completo antes do corte:
    # cada valor só usa voos que já tinham aterrado, por isso não há fuga entre folds.
    df = prepare_data(load_dataset(path), treino=True)
    df_dev = select_dev(df)
    del df   # o dataset completo (com o período de teste) já não é preciso
    print(f"Pré-processamento concluído em {time.time() - t0:.0f}s")

    y_dev = df_dev[TARGET_COLUMN]
    # Só as colunas que o modelo usa: o ColumnTransformer descartava as outras de qualquer forma
    # (ex.: ORIGIN, com ~10M strings, ocupa muita memória sem nunca chegar ao modelo)
    X_dev = df_dev[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return X_dev, y_dev


# ==============================================================================
# 2. FUNÇÃO OBJETIVO COM TIMESERIESSPLIT E XGBOOST
# ==============================================================================

def make_objective(X_dev: pd.DataFrame, y_dev: pd.Series, n_splits: int = N_SPLITS, device: str = DEVICE):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": -1,
            "device": device,
        }

        fold_scores = []
        print(f"\n[Trial {trial.number + 1}] A iniciar avaliação com {n_splits} folds temporais...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_dev)):
            t_fold_start = time.time()

            X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
            X_va, y_va = X_dev.iloc[val_idx], y_dev.iloc[val_idx]

            # Pré-processador ajustado só no treino do fold atual (evita data leakage)
            preprocessor = build_preprocessor()
            # float32: o XGBoost converte para float32 internamente de qualquer forma, por isso os
            # resultados são os mesmos e evita-se ter uma cópia float64 em memória (~metade da RAM)
            X_tr_proc = preprocessor.fit_transform(X_tr).astype(np.float32)
            X_va_proc = preprocessor.transform(X_va).astype(np.float32)

            model = XGBClassifier(**params)
            model.fit(X_tr_proc, y_tr)

            score = f1_score(y_va, model.predict(X_va_proc), average="macro")
            fold_scores.append(score)
            print(f"  └─ Fold {fold + 1}/{n_splits} concluído em {time.time() - t_fold_start:.1f}s | F1-macro: {score:.4f}")

            # Libertar as cópias deste fold antes do seguinte (RAM e memória da GPU)
            del X_tr, X_va, X_tr_proc, X_va_proc, model, preprocessor
            gc.collect()

        return float(np.mean(fold_scores))

    return objective


# ==============================================================================
# 3. ESTUDO OPTUNA
# ==============================================================================

def format_elapsed(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"


def run_study(objective, n_trials: int = N_TRIALS, storage: str = STORAGE, study_name: str = STUDY_NAME) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )
    # Seed diferente em cada execução (depende dos trials já feitos): reprodutível e, ao retomar
    # o estudo, não repete os parâmetros dos primeiros trials aleatórios
    study.sampler = optuna.samplers.TPESampler(seed=42 + len(study.trials))

    start = time.time()

    def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        trial_duration = trial.duration.total_seconds() if trial.duration else 0.0
        value = f"{trial.value:.4f}" if trial.value is not None else "falhou"
        print("─" * 70)
        print(
            f"[RESUMO TRIAL {trial.number + 1:03d}] "
            f"Score Médio: {value} | Melhor até agora: {study.best_value:.4f}\n"
            f"Duração deste Trial: {trial_duration:.1f}s | Tempo Total Decorrido: {format_elapsed(time.time() - start)}"
        )
        print("─" * 70)

    print("=" * 70)
    print(f"A iniciar otimização do estudo '{study_name}' ({n_trials} trials, device={DEVICE})...")
    print(f"Base de dados: {storage}")
    print(f"Podes acompanhar em tempo real com: optuna-dashboard {storage}")
    print("=" * 70)

    study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])

    print("\n" + "=" * 70)
    print("--- OTIMIZAÇÃO CONCLUÍDA COM SUCESSO ---")
    print(f"Tempo total gasto: {format_elapsed(time.time() - start)}")
    print(f"Melhor F1-Macro médio obtido: {study.best_value:.4f}")
    print("Melhores Hiperparâmetros:")
    for param, val in study.best_params.items():
        print(f"  • {param}: {val}")
    print("=" * 70)
    return study


def save_best_params(study: optuna.Study, path: Path = BEST_PARAMS_PATH) -> None:
    """Guarda os melhores parâmetros no JSON que o train_vN.py lê."""
    with open(path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Melhores parâmetros guardados em {path}")


if __name__ == "__main__":
    X_dev, y_dev = load_dev_data(resolve_data_path())
    print(f"Total de registos para treino/validação no Optuna: {len(X_dev):,}")

    study = run_study(make_objective(X_dev, y_dev))
    save_best_params(study)
