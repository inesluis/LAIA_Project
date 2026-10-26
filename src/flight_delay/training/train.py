"""Esboco do treino da pessoa B. Completar os TODO antes de executar.

O treino contacta o MLflow pela rede; nao e executado pelo servidor MLflow.
"""

import os

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score


def preparar_dados():
    # TODO: integrar o codigo de dados/features da pessoa A.
    # Usar dados de 2022 e 2023, filtrados conforme o enunciado.
    # Fazer divisao temporal e separar entradas X do alvo y.
    # X deve conter apenas informacao disponivel antes da partida prevista.
    # Devolver: X_train, X_val, y_train, y_val.
    # Combinar as classes: on_time, moderate_delay, severe_delay.
    raise NotImplementedError("Integrar dados e divisao temporal.")


def construir_pipeline():
    # TODO: devolver uma sklearn Pipeline com preprocessamento + modelo.
    # Ajustar os transformadores apenas nos dados de treino.
    # Tratar categorias desconhecidas e combinar o formato com a pessoa C.
    raise NotImplementedError("Definir preprocessamento e modelo.")


def main():
    # Local: http://localhost:5050
    # Noutra maquina/contentor: http://<IP-ou-DNS-da-VM1>:5050
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("flight-delay")

    X_train, X_val, y_train, y_val = preparar_dados()
    pipeline = construir_pipeline()

    with mlflow.start_run():
        # TODO: registar hiperparametros, versao dos dados e commit Git.
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)

        mlflow.log_metrics({
            "val_accuracy": accuracy_score(y_val, predictions),
            "val_f1_macro": f1_score(
                y_val,
                predictions,
                labels=["on_time", "moderate_delay", "severe_delay"],
                average="macro",
                zero_division=0,
            ),
        })

        # TODO: recall por classe, matriz de confusao e metricas por
        # companhia operadora/aeroporto de origem.

        # Guardar preprocessamento E modelo, criando uma versao no registry.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="pipeline",
            registered_model_name="flight-delay",
            input_example=X_val.head(3),
        )

        # TODO: comparar candidatos com criterios definidos.
        # So depois aprovar uma versao e atribuir-lhe o alias "champion".
        # A API da pessoa C podera carregar models:/flight-delay@champion.


if __name__ == "__main__":
    main()
