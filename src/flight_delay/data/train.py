import json
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from xgboost import Booster, XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
)
from mlflow.models import infer_signature
from preprocessing import (
    load_dataset,
    prepare_data,
    build_preprocessor,
)

# ---------------------------------------------------------------------
# MLFLOW
# ---------------------------------------------------------------------

mlflow.set_tracking_uri("http://10.6.0.80:5000")
mlflow.set_experiment("flight_delay_prediction")


# ---------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------

with open("best_params_v1.json", "r") as f:
    best_params = json.load(f)



def split_data(df):
    if 'FLIGHTDATE' in df.columns:
        df = df.sort_values(by='FLIGHTDATE').reset_index(drop=True)
    elif {'YEAR', 'MONTH', 'DAYOFMONTH'}.issubset(df.columns):
        df = df.sort_values(by=['YEAR', 'MONTH', 'DAYOFMONTH']).reset_index(drop=True)

    # Separar os últimos 6 meses de 2023 para o teste cego final
    mask_dev = (df['YEAR'] == 2022) | ((df['YEAR'] == 2023) & (df['MONTH'] <= 6))
    df_treino = df.loc[mask_dev].reset_index(drop=True)
    df_test = df.loc[~mask_dev].reset_index(drop=True)

    y_train = df_treino['TARGET']
    X_train = df_treino.drop(columns=['TARGET'])
    X_test = df_test.drop(columns=['TARGET'])
    y_test = df_test['TARGET']
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train(df):

    # preparação de dados
    df_prepared = prepare_data(
        df=df,
        treino=True
    )

    X_train, X_test, y_train, y_test = split_data(
        df_prepared
    )

    # modelo
    model = XGBClassifier(
        **best_params,
        random_state=42,
        device='cuda',
        tree_method="hist",
        num_class=3,
        n_jobs=-1,
    )

    # pipeline completo
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                model,
            ),
        ]
    )
    try:
        with mlflow.start_run(run_name="xgb_v1"):

            # treino
            pipeline.fit(X_train, y_train)


            # previsão
            y_pred = pipeline.predict(X_test)


            # métricas
            accuracy = accuracy_score(y_test,y_pred,)

            f1 = f1_score(y_test,y_pred,average="macro",)

            recall = recall_score(y_test,y_pred,average=None,zero_division=0,)

            precision = precision_score(y_test,y_pred,average=None,zero_division=0,)

            report = classification_report(y_test,y_pred,output_dict=True,)

            cm = confusion_matrix(y_test,y_pred,)

            # logging
            mlflow.log_param("model_type","XGBClassifier",)
            mlflow.log_param(
        "dataset_version",
        "dataset_completo.v2.csv"
    )
            mlflow.log_param(
        "train_rows",
        len(X_train)
            )

            mlflow.log_param(
                "test_rows",
                len(X_test)
    )
            mlflow.log_dict(
        y_train.value_counts().to_dict(),
        "class_distribution.json"
    )

            mlflow.log_dict(
                y_test.value_counts().to_dict(),
                "class_distribution_test.json"
            )

            mlflow.log_params(best_params)

            mlflow.log_metric("accuracy",accuracy,)

            mlflow.log_metric("f1_macro",f1,)

            mlflow.log_metric("recall_ontime",recall[0],)
            mlflow.log_metric("recall_moderate_delay",recall[1],)
            mlflow.log_metric("recall_severe_delay",recall[2],)

            mlflow.log_metric("precision_ontime",precision[0],)
            mlflow.log_metric("precision_moderate_delay",precision[1],)
            mlflow.log_metric("precision_severe_delay",precision[2],)

            mlflow.log_dict(report,"classification_report.json",)

            mlflow.log_dict({"confusion_matrix": cm.tolist()},"confusion_matrix.json",)


            sample = X_train.sample(
                100,
                random_state=42
            )

            signature = infer_signature(
                sample,
                pipeline.predict(sample)
            )

            
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="pipeline",
                registered_model_name="xgboost_v1",
                signature=signature,
                input_example=X_train.head(3),
                )
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Macro: {f1:.4f}")


    except Exception as e:
        print(f"Erro durante o treino: {e}")
        mlflow.log_param(
            "training_failed",
            str(e)
        )

        mlflow.end_run(status="FAILED")
        raise e

    return {
    "pipeline": pipeline,
    "accuracy": accuracy,
    "f1": f1
}




if __name__ == "__main__":
    # carregar dataset
    csv_base = Path.cwd().parent.parent.parent / "data" 
    path = csv_base / "dataset_completo.v2.csv"
    df = load_dataset(
        csv_path=path
    )

    # treinar modelo
    result = train(df)

    print(result["accuracy"])
    print(result["f1"])


"""
            signature = infer_signature(
                X_train,
                pipeline.predict(
                    X_train.head(10)
                ),
            )

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="pipeline",
                registered_model_name="xgboost_v1",
                signature=signature,
                input_example=X_train.head(3),
                skops_trusted_types=[
                    XGBClassifier,
                    Booster,
                ],
            )"""