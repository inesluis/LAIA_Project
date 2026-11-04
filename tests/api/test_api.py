from fastapi.testclient import TestClient

from flight_delay.serving.app import app

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

def test_health_and_classify():
    # "with" corre o lifespan, que carrega o modelo (dummy: MODEL_URI vazio)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}

        r = client.post("/classify", json={"instances": [FLIGHT, FLIGHT]})
        assert r.status_code == 200
        assert r.json() == {"classifications": ["on_time", "on_time"]}


def test_classify_rejects_invalid_record():
    with TestClient(app) as client:
        bad = {**FLIGHT, "DISTANCE": -1}
        assert client.post("/classify", json={"instances": [bad]}).status_code == 422
