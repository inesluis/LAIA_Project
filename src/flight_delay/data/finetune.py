import time
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# Importar as funções modularizadas criadas no preprocessing.py
from preprocessing import load_dataset, clean_flight_data, build_preprocessor, TARGET_COLUMN

# ==============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================
csv_path = Path(__file__).resolve().parents[3] / "data" / "dataset_completo_limpo.v3.csv" 

df = clean_flight_data(load_dataset(csv_path))

# Garantir a ordenação temporal antes de qualquer corte
if 'FlightDate' in df.columns:
    df = df.sort_values(by='FlightDate').reset_index(drop=True)
elif {'Year', 'Month', 'DayofMonth'}.issubset(df.columns):
    df = df.sort_values(by=['Year', 'Month', 'DayofMonth']).reset_index(drop=True)

# Separar os últimos 6 meses de 2023 para o teste cego final
mask_dev = (df['Year'] == 2022) | ((df['Year'] == 2023) & (df['Month'] <= 6))
df_dev = df.loc[mask_dev].reset_index(drop=True)

y_dev = df_dev[TARGET_COLUMN]
X_dev = df_dev.drop(columns=[TARGET_COLUMN])

# Remover Year dos atributos para evitar overfitting anual
if 'Year' in X_dev.columns:
    X_dev = X_dev.drop(columns=['Year'])

print(f"Total de registos para treino/validação no Optuna: {len(X_dev):,}")

# ==============================================================================
# 2. DEFINIÇÃO DA FUNÇÃO OBJETIVO COM TIMESERIESSPLIT E XGBOOST
# ==============================================================================
N_SPLITS = 4
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

def objective(trial):
    # Espaço de busca para o XGBoost
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 350, step=50),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'tree_method': 'hist',   # Essencial para velocidade com dados BTS
        'n_jobs': -1,
        'device':'cuda'
    }

    fold_scores = []

    print(f"\n[Trial {trial.number + 1}] A iniciar avaliação com {N_SPLITS} folds temporais...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_dev)):
        t_fold_start = time.time()

        X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_va, y_val = X_dev.iloc[val_idx], y_dev.iloc[val_idx]

        # Ajustar o pré-processador isolado no treino do fold atual para evitar data leakage
        preprocessor = build_preprocessor()
        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_va_proc = preprocessor.transform(X_va)

        model = XGBClassifier(**params)
        model.fit(X_tr_proc, y_tr)

        preds = model.predict(X_va_proc)
        score = f1_score(y_val, preds, average='macro')
        fold_scores.append(score)

        fold_time = time.time() - t_fold_start
        print(f"  └─ Fold {fold + 1}/{N_SPLITS} concluído em {fold_time:.1f}s | F1-macro: {score:.4f}")

    mean_score = float(np.mean(fold_scores))
    return mean_score

# ==============================================================================
# 3. CONFIGURAÇÃO DO ESTUDO OPTUNA (SQLITE) E CALLBACKS COM TIMER
# ==============================================================================
optuna.logging.set_verbosity(optuna.logging.ERROR)

storage_name = "sqlite:///bts_xgboost_optimization.db"
study_name = "xgboost_timeseries_f1"

sampler = optuna.samplers.TPESampler(seed=42)

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction="maximize",
    load_if_exists=True
)

start_global_time = time.time()
TOTAL_TRIALS = 50

def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """Callback chamado ao fim de cada trial para reportar progresso e cronómetro global."""
    elapsed_total = time.time() - start_global_time
    hours, remainder = divmod(int(elapsed_total), 3600)
    minutes, seconds = divmod(remainder, 60)
    tempo_decorrido = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

    trial_duration = trial.duration.total_seconds() if trial.duration else 0.0

    print("─" * 70)
    print(
        f"[RESUMO TRIAL {trial.number + 1:03d}/{TOTAL_TRIALS}] "
        f"Score Médio: {trial.value:.4f} | "
        f"Melhor até agora: {study.best_value:.4f}\n"
        f"Duração deste Trial: {trial_duration:.1f}s | "
        f"Tempo Total Decorrido: {tempo_decorrido}"
    )
    print("─" * 70)

# ==============================================================================
# 4. EXECUÇÃO DA OTIMIZAÇÃO
# ==============================================================================
print("=" * 70)
print(f"A iniciar otimização do estudo '{study_name}' ({TOTAL_TRIALS} trials)...")
print(f"Base de dados: {storage_name}")
print("Podes acompanhar em tempo real pelo terminal com: optuna-dashboard " + storage_name)
print("=" * 70)

study.optimize(
    objective,
    n_trials=TOTAL_TRIALS,
    callbacks=[trial_callback]
)

# ==============================================================================
# 5. RESULTADOS FINAIS
# ==============================================================================
total_elapsed = time.time() - start_global_time
tot_h, rem = divmod(int(total_elapsed), 3600)
tot_m, tot_s = divmod(rem, 60)

print("\n" + "=" * 70)
print("--- OTIMIZAÇÃO CONCLUÍDA COM SUCESSO ---")
print(f"Tempo total gasto: {tot_h:02d}h {tot_m:02d}m {tot_s:02d}s")
print(f"Melhor F1-Macro médio obtido: {study.best_value:.4f}")
print("Melhores Hiperparâmetros:")
for param, val in study.best_params.items():
    print(f"  • {param}: {val}")
print("=" * 70)