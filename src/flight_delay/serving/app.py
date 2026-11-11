# Serviço de inferência: classificação da severidade de atrasos de voos 

from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException
from .model_loader import load_model
from .schemas import ClassifyRequest, ClassifyResponse

VALID_LABELS = {"on_time", "moderate_delay", "severe_delay"}
# Target do treino: 0/1/2 -> rótulos pedidos pelos docentes. Texto passa tal e qual.
# shortcut: só inteiros ("0"), não floats ("0.0"); alargar se o modelo devolver floats.
NUMERIC_TO_LABEL = {"0": "on_time", "1": "moderate_delay", "2": "severe_delay"}
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # carrega o modelo uma única vez (no arranque) e não em cada pedido 
    state["model"] = load_model()
    yield
    state.clear()

app = FastAPI(title="Flight Delay Severity API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    # devolve lista vazia (um rótulo por registo = zero rótulos).
    if not req.instances:
        return ClassifyResponse(classifications=[])

    # uma linha por registo, pela ordem recebida -> ordem preservada.
    df = pd.DataFrame([r.model_dump() for r in req.instances])

    try:
        preds = [str(p) for p in state["model"].predict(df)]
        preds = [NUMERIC_TO_LABEL.get(p, p) for p in preds]
    except Exception:
        # nunca expor o traceback ao cliente.
        raise HTTPException(status_code=500, detail="Erro interno na inferência.")

    if len(preds) != len(req.instances) or not set(preds) <= VALID_LABELS:
        raise HTTPException(status_code=500, detail="Saída do modelo inválida.")

    return ClassifyResponse(classifications=preds)