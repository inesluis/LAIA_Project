import types
import pytest
from fastapi.testclient import TestClient
from flight_delay.serving.app import app, state
from flight_delay.serving.model_loader import DummyModel, load_model

FLIGHT = {
    "FL_DATE": "2024-01-15",
    "OP_UNIQUE_CARRIER": "TP",
    "ORIGIN": "LIS",
    "DEST": "OPO",
    "CRS_DEP_TIME": 930,
    "CRS_ARR_TIME": 1030,
    "CRS_ELAPSED_TIME": 60.0,
    "DISTANCE": 274.0,
}

# ---------- false models: only for test ----------

class BoomModel:
    def predict(self, df):
        raise RuntimeError("internal-secret")

class BadLabelModel:
    def predict(self, df):
        return ["invented_label"] * len(df)

class ShortModel:
    def predict(self, df):
        return ["on_time"]  # sempre 1, seja qual for o nº de voos

class ByDistanceModel:
    # rótulo depende da DISTANCE -> permite verificar a ordem da resposta
    def predict(self, df):
        return ["severe_delay" if d > 1000 else "on_time" for d in df["DISTANCE"]]

@pytest.fixture
def client():
    # "with" corre o lifespan (carrega o modelo dummy); no fim limpa o state
    with TestClient(app) as c:
        yield c

def post(client, model, instances):
    # troca o modelo DEPOIS do arranque e faz o pedido
    state["model"] = model
    return client.post("/classify", json={"instances": instances})

# ---------- testes that already exist  ----------

def test_health_and_classify(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}

    r = client.post("/classify", json={"instances": [FLIGHT, FLIGHT]})
    assert r.status_code == 200
    assert r.json() == {"classifications": ["on_time", "on_time"]}

def test_classify_rejects_invalid_record(client):
    bad = {**FLIGHT, "DISTANCE": -1}
    assert client.post("/classify", json={"instances": [bad]}).status_code == 422

# ---------- branches that were missing in app.py ----------

def test_empty_batch_returns_empty_list(client):  # linha 29
    r = client.post("/classify", json={"instances": []})
    assert r.status_code == 200
    assert r.json() == {"classifications": []}

def test_model_exception_returns_500_without_leaking(client):  # linhas 36-38
    r = post(client, BoomModel(), [FLIGHT])
    assert r.status_code == 500
    assert r.json() == {"detail": "Erro interno na inferência."}
    assert "internal-secret" not in r.text  # a mensagem da exceção não chega ao cliente

def test_invalid_label_from_model_returns_500(client):  # linha 41, 1ª condição
    r = post(client, BadLabelModel(), [FLIGHT])
    assert r.status_code == 500
    assert r.json() == {"detail": "Saída do modelo inválida."}

def test_wrong_number_of_predictions_returns_500(client):  # linha 41, 2ª condição
    r = post(client, ShortModel(), [FLIGHT, FLIGHT])  # 2 voos, 1 previsão
    assert r.status_code == 500

def test_order_is_preserved(client):
    long_flight = {**FLIGHT, "DISTANCE": 5000.0}
    r = post(client, ByDistanceModel(), [FLIGHT, long_flight, FLIGHT])
    assert r.json() == {"classifications": ["on_time", "severe_delay", "on_time"]}

class NumericModel:
    # como o Target do treino: 0 = on_time, 1 = moderate, 2 = severe
    def predict(self, df):
        return [0, 1, 2]

def test_numeric_predictions_are_converted_to_labels(client):
    r = post(client, NumericModel(), [FLIGHT, FLIGHT, FLIGHT])
    assert r.status_code == 200
    assert r.json() == {"classifications": ["on_time", "moderate_delay", "severe_delay"]}

def test_unknown_number_returns_500(client):
    class OutOfRangeModel:
        def predict(self, df):
            return [3]
    assert post(client, OutOfRangeModel(), [FLIGHT]).status_code == 500

# ---------- model_loader.py ----------

def test_load_model_without_uri_gives_dummy():
    assert isinstance(load_model(""), DummyModel)

def test_load_model_with_uri_uses_mlflow(monkeypatch):  # linhas 18-19
    # mlflow falso: não precisa de estar instalado nem de servidor
    fake_mlflow = types.SimpleNamespace(
        pyfunc=types.SimpleNamespace(load_model=lambda uri: ("loaded", uri))
    )
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    assert load_model("models:/flight-delay@champion") == (
        "loaded",
        "models:/flight-delay@champion",
    )

# ---------- campos em falta e categorias nunca vistas ----------

def test_missing_required_field_returns_422(client):
    # a validação é do schema (Pydantic): não depende do modelo
    incomplete = {k: v for k, v in FLIGHT.items() if k != "ORIGIN"}
    r = client.post("/classify", json={"instances": [incomplete]})
    assert r.status_code == 422

@pytest.mark.parametrize(
    "field, value",
    [("DISTANCE", "abc"), ("CRS_DEP_TIME", "morning"), ("FL_DATE", "not-a-date")],
)
def test_wrong_type_returns_422(client, field, value):
    # tipo errado (texto onde se espera número/data) -> erro de validação, não 500
    r = client.post("/classify", json={"instances": [{**FLIGHT, field: value}]})
    assert r.status_code == 422

def test_body_without_instances_returns_422(client):
    # corpo sem a chave obrigatória "instances"
    assert client.post("/classify", json={}).status_code == 422

def test_unseen_category_is_not_rejected_by_api(client):
    # a API aceita aeroportos/companhias novos (str livre). Isto NÃO prova
    # que o modelo os aguenta: ver o teste seguinte.
    new_flight = {**FLIGHT, "ORIGIN": "ZZZ", "OP_UNIQUE_CARRIER": "XX"}
    assert client.post("/classify", json={"instances": [new_flight]}).status_code == 200

# ---------- formato do leaderboard ----------

# um voo com os 36 campos, tal como o leaderboard o mostra
LEADERBOARD_FLIGHT = {
    "FL_DATE": "2024-01-08", "YEAR": 2024, "QUARTER": 1, "MONTH": 1,
    "DAY_OF_MONTH": 8, "DAY_OF_WEEK": 1, "OP_UNIQUE_CARRIER": "9E",
    "OP_CARRIER_AIRLINE_ID": 20363, "OP_CARRIER": "9E", "TAIL_NUM": "N485PX",
    "OP_CARRIER_FL_NUM": 4801, "ORIGIN_AIRPORT_ID": 12953,
    "ORIGIN_AIRPORT_SEQ_ID": 1295304, "ORIGIN_CITY_MARKET_ID": 31703,
    "ORIGIN": "LGA", "ORIGIN_CITY_NAME": "New York, NY", "ORIGIN_STATE_ABR": "NY",
    "ORIGIN_STATE_FIPS": 36, "ORIGIN_STATE_NM": "New York", "ORIGIN_WAC": 22,
    "DEST_AIRPORT_ID": 13871, "DEST_AIRPORT_SEQ_ID": 1387102,
    "DEST_CITY_MARKET_ID": 33316, "DEST": "OMA", "DEST_CITY_NAME": "Omaha, NE",
    "DEST_STATE_ABR": "NE", "DEST_STATE_FIPS": 31, "DEST_STATE_NM": "Nebraska",
    "DEST_WAC": 65, "CRS_DEP_TIME": 856, "DEP_TIME_BLK": "0800-0859",
    "CRS_ARR_TIME": 1135, "ARR_TIME_BLK": "1100-1159",
    "CRS_ELAPSED_TIME": 219.0, "DISTANCE": 1148.0, "DISTANCE_GROUP": 5,
}

def test_leaderboard_batch_of_1000_is_accepted(client):
    body = {"request_id": "abc", "mode": "sandbox", "instances": [LEADERBOARD_FLIGHT] * 1000}
    r = client.post("/classify", json=body)
    assert r.status_code == 200
    assert len(r.json()["classifications"]) == 1000

@pytest.mark.parametrize("field", list(FLIGHT))
def test_null_value_is_accepted(client, field):
    # um null num voo não pode rejeitar o lote inteiro
    r = client.post("/classify", json={"instances": [FLIGHT, {**FLIGHT, field: None}]})
    assert r.status_code == 200
    assert len(r.json()["classifications"]) == 2

def test_unknown_mode_returns_422(client):
    assert client.post("/classify", json={"mode": "other", "instances": [FLIGHT]}).status_code == 422

# ---------- /labels ----------

LABELS_BODY = {
    "request_id": "abc",
    "mode": "live",
    "labels": ["on_time", "severe_delay"],
    "outcomes": [
        {"DEP_TIME": 851, "ARR_DELAY": -11.0, "CARRIER_DELAY": None},
        {"DEP_TIME": 1832, "ARR_DELAY": 88.0, "CARRIER_DELAY": 0.0},
    ],
}

def test_labels_are_acknowledged(client):
    r = client.post("/labels", json=LABELS_BODY)
    assert r.status_code == 200
    assert r.json() == {"status": "received"}

@pytest.mark.parametrize(
    "change",
    [{"labels": ["on_time", "invented_label"]}, {"labels": ["on_time"]}, {"request_id": None}],
)
def test_invalid_labels_body_returns_422(client, change):
    assert client.post("/labels", json={**LABELS_BODY, **change}).status_code == 422

# ---------- modelo real ----------

@pytest.mark.skip(reason="aguarda o modelo real (pipeline com handle_unknown)")
def test_unseen_category_with_real_model(monkeypatch):
    # TODO: com MODEL_URI a apontar para o modelo registado (ou um pipeline
    # guardado em tests/fixtures), enviar ORIGIN="ZZZ" e esperar 200 + rótulo válido.
    ...
