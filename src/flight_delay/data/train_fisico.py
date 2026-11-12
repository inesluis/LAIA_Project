import json
from pathlib import Path
import joblib

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
)

from flight_delay.data.preprocessing_v1 import (
    load_dataset,
    prepare_data,
    build_preprocessor,
)


with open("best_params_v1.json", "r") as f:
    best_params = json.load(f)


def split_data(df):

    if "FLIGHTDATE" in df.columns:
        df = df.sort_values(
            by="FLIGHTDATE"
        ).reset_index(drop=True)

    elif {"YEAR", "MONTH", "DAYOFMONTH"}.issubset(df.columns):
        df = df.sort_values(
            by=["YEAR", "MONTH", "DAYOFMONTH"]
        ).reset_index(drop=True)

    mask_dev = (
        (df["YEAR"] == 2022)
        |
        (
            (df["YEAR"] == 2023)
            &
            (df["MONTH"] <= 6)
        )
    )

    df_train = df.loc[mask_dev].reset_index(drop=True)
    df_test = df.loc[~mask_dev].reset_index(drop=True)

    X_train = df_train.drop(columns=["TARGET"])
    y_train = df_train["TARGET"]

    X_test = df_test.drop(columns=["TARGET"])
    y_test = df_test["TARGET"]

    return X_train, X_test, y_train, y_test


def train(df):

    df_prepared = prepare_data(
        df=df,
        treino=True,
    )

    X_train, X_test, y_train, y_test = split_data(
        df_prepared
    )

    model = XGBClassifier(
        **best_params,
        random_state=42,
        num_class=3,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                model,
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    y_pred = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    precision = precision_score(
        y_test,
        y_pred,
        average=None,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average=None,
        zero_division=0,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1:.4f}")

    print("\nPrecision:")
    print(precision)

    print("\nRecall:")
    print(recall)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # guardar pipeline completo

    output_dir = Path("models")
    output_dir.mkdir(
        exist_ok=True
    )

    model_path = output_dir / "xgboost_v1.pkl"

    joblib.dump(
        pipeline,
        model_path,
    )

    print(
        f"\nModelo guardado em: {model_path}"
    )

    return pipeline


if __name__ == "__main__":

    csv_base = (
        Path.cwd()
        .parent
        .parent
        .parent
        / "data"
    )

    csv_path = (
        csv_base
        / "dataset_completo.v2.csv"
    )

    df = load_dataset(
        csv_path=csv_path
    )

    train(df)