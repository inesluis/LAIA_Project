import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing_v1 import (
    build_preprocessor,
    load_dataset,
    prepare_data,
)

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# MLFLOW
# ---------------------------------------------------------------------

# O servidor corre com --serve-artifacts: os artefactos são enviados por
# HTTP para o servidor, em vez de escritos num caminho local.
ARTIFACT_URI = "mlflow-artifacts:/"

# Localizações que apontam para o disco de quem corre o código (ex.: /mlflow,
# file:///mlflow). Dentro do container não existem nem têm permissão de escrita.
LOCAL_ARTIFACT_PREFIXES = ("/", "file:")

DEFAULT_EXPERIMENT_NAME = "flight_delay"


def _is_local(artifact_location: str) -> bool:
    return artifact_location.startswith(LOCAL_ARTIFACT_PREFIXES)


def _ensure_not_deleted(experiment) -> None:
    if experiment.lifecycle_stage == "deleted":
        raise RuntimeError(
            f"A experiência '{experiment.name}' foi apagada e continua no lixo "
            "do MLflow. Restaura-a na UI ou define outro MLFLOW_EXPERIMENT_NAME."
        )


def _create_experiment(name: str) -> str:
    mlflow.create_experiment(name, artifact_location=ARTIFACT_URI)
    print(f"Experiência '{name}' criada com armazenamento remoto de artefactos")
    return name


def get_or_create_experiment(name: str) -> str:
    """Devolve o nome da experiência a usar, criando-a se necessário.

    - Não existe -> cria-a com artefactos remotos.
    - Existe com artefactos remotos -> usa-a.
    - Existe com artefactos num caminho local -> usa (ou cria) '<nome>_remote'.
    """
    experiment = mlflow.get_experiment_by_name(name)

    if experiment is None:
        return _create_experiment(name)

    _ensure_not_deleted(experiment)

    if not _is_local(experiment.artifact_location):
        print(f"A usar a experiência existente '{name}'")
        return name

    fallback = f"{name}_remote"
    print(
        f"A experiência '{name}' guarda artefactos num caminho local "
        f"({experiment.artifact_location}). A usar '{fallback}'."
    )

    fallback_experiment = mlflow.get_experiment_by_name(fallback)

    if fallback_experiment is None:
        return _create_experiment(fallback)

    _ensure_not_deleted(fallback_experiment)

    if _is_local(fallback_experiment.artifact_location):
        raise RuntimeError(
            f"A experiência '{fallback}' também usa um caminho local "
            f"({fallback_experiment.artifact_location}). "
            "Define outro MLFLOW_EXPERIMENT_NAME."
        )

    return fallback


def setup_mlflow() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        # Falhar já, em vez de gravar tudo numa pasta mlruns/ local sem aviso
        raise RuntimeError(
            "MLFLOW_TRACKING_URI não está definida (ver .env.example)."
        )

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME
    )
    mlflow.set_experiment(get_or_create_experiment(experiment_name))


# ---------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------


def load_params() -> dict:
    # Relativo ao script, e não à pasta de onde o script é corrido
    with open(BASE_DIR / "best_params_v1.json", "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------


def resolve_data_path() -> Path:
    # No container vem do DATA_PATH. Localmente, usa <raiz do repo>/data/.
    env_path = os.environ.get("DATA_PATH")
    if env_path:
        return Path(env_path)
    return BASE_DIR.parents[3] / "data" / "dataset_completo.v1.parquet"


def split_data(df):
    if "FLIGHTDATE" in df.columns:
        df = df.sort_values(by="FLIGHTDATE").reset_index(drop=True)
    elif {"YEAR", "MONTH", "DAYOFMONTH"}.issubset(df.columns):
        df = df.sort_values(by=["YEAR", "MONTH", "DAYOFMONTH"]).reset_index(drop=True)

    # Separar os últimos 6 meses de 2023 para o teste cego final
    mask_dev = (df["YEAR"] == 2022) | ((df["YEAR"] == 2023) & (df["MONTH"] <= 6))
    df_treino = df.loc[mask_dev].reset_index(drop=True)
    df_test = df.loc[~mask_dev].reset_index(drop=True)

    y_train = df_treino["TARGET"]
    X_train = df_treino.drop(columns=["TARGET"])
    X_test = df_test.drop(columns=["TARGET"])
    y_test = df_test["TARGET"]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train(df, best_params: dict, dataset_name: str):

    # preparação de dados
    df_prepared = prepare_data(df=df, treino=True)

    X_train, X_test, y_train, y_test = split_data(df_prepared)

    # modelo
    model = XGBClassifier(
        **best_params,
        random_state=42,
        device="cpu",
        tree_method="hist",
        num_class=3,
        n_jobs=-1,
    )

    # pipeline completo
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )

    # Se houver uma exceção dentro do "with", o MLflow marca o run como FAILED
    # automaticamente ao sair do bloco.
    with mlflow.start_run(run_name="xgb_v1"):
        try:
            # treino
            pipeline.fit(X_train, y_train)

            # previsão
            y_pred = pipeline.predict(X_test)

            # métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average=None, zero_division=0)
            precision = precision_score(y_test, y_pred, average=None, zero_division=0)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            # logging
            mlflow.log_param("model_type", "XGBClassifier")
            mlflow.log_param("dataset_version", dataset_name)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_params(best_params)

            mlflow.log_dict(
                y_train.value_counts().to_dict(), "class_distribution.json"
            )
            mlflow.log_dict(
                y_test.value_counts().to_dict(), "class_distribution_test.json"
            )

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_macro", f1)

            mlflow.log_metric("recall_ontime", recall[0])
            mlflow.log_metric("recall_moderate_delay", recall[1])
            mlflow.log_metric("recall_severe_delay", recall[2])

            mlflow.log_metric("precision_ontime", precision[0])
            mlflow.log_metric("precision_moderate_delay", precision[1])
            mlflow.log_metric("precision_severe_delay", precision[2])

            mlflow.log_dict(report, "classification_report.json")
            mlflow.log_dict(
                {"confusion_matrix": cm.tolist()}, "confusion_matrix.json"
            )

            sample = X_train.sample(100, random_state=42)
            signature = infer_signature(sample, pipeline.predict(sample))

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="pipeline",
                registered_model_name="xgboost_v1",
                signature=signature,
                input_example=X_train.head(3),
            )

            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Macro: {f1:.4f}")

        except Exception as e:
            print(f"Erro durante o treino: {e}")
            # Tag em vez de param: não tem o limite de tamanho dos params
            mlflow.set_tag("training_failed", str(e)[:5000])
            raise

    return {
        "pipeline": pipeline,
        "accuracy": accuracy,
        "f1": f1,
    }


if __name__ == "__main__":
    setup_mlflow()

    data_path = resolve_data_path()
    best_params = load_params()

    # carregar dataset
    df = load_dataset(csv_path=data_path)

    # treinar modelo
    result = train(df, best_params, dataset_name=data_path.name)

    print(result["accuracy"])
    print(result["f1"])
