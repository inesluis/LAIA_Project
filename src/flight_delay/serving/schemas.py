# Schemas de entrada e saída dos endpoints /classify e /labels

# nota: os nomes seguem a especificação do leaderboard (36 campos). Por agora só estão
# declarados 8; os restantes são ignorados até se ligar o modelo real.
# final version tem de seguir o data contract do pedro

from datetime import date
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

Label = Literal["on_time", "moderate_delay", "severe_delay"]
Mode = Literal["sandbox", "live"]

class FlightRecord(BaseModel): # um voo, com atributos disponíveis antes da partida programada  
     
    model_config = ConfigDict(extra="ignore") # se enviar campos a mais ignoram-se, não se rejeitam 

    # Optional sem valor por defeito = o campo tem de vir, mas pode ser null.
    # Os limites (ge, le, gt) continuam a aplicar-se quando há valor.
    FL_DATE: Optional[date]
    OP_UNIQUE_CARRIER: Optional[str]
    ORIGIN: Optional[str]           # str livre: aeroportos novos não dão erro
    DEST: Optional[str]
    CRS_DEP_TIME: Optional[int] = Field(..., ge=0, le=2400)
    CRS_ARR_TIME: Optional[int] = Field(..., ge=0, le=2400)
    CRS_ELAPSED_TIME: Optional[float] = None # se o modelo do cura souber lidar com missing data
    DISTANCE: Optional[float] = Field(..., gt=0)
# outra opção é ter um flightrecord gerado a partir de um ficheiro (util para mim, A e B) ver vantagens e desvantagens de cada abordagem.

"""
def build_flight_record(path: Path = SCHEMA_PATH) -> type[BaseModel]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    fields = {}
    for name, f in spec["fields"].items():
        py_type = TYPES[f["type"]]
        limits = {k: f[k] for k in ("min", "max", "gt") if k in f}
        limits = {{"min": "ge", "max": "le"}.get(k, k): v for k, v in limits.items()}
        if f.get("required", True):
            fields[name] = (py_type, Field(..., **limits))
        else:
            fields[name] = (Optional[py_type], Field(None, **limits))
    return create_model("FlightRecord", __config__=ConfigDict(extra="ignore"), **fields)

FlightRecord = build_flight_record()

"""
class ClassifyRequest(BaseModel):
    # opcionais: o exemplo do enunciado e o smoke test não os enviam
    request_id: Optional[str] = None
    mode: Optional[Mode] = None
    instances: List[FlightRecord]

class ClassifyResponse(BaseModel):
    classifications: List[Label]

class LabelsRequest(BaseModel):
    # resultados reais que o leaderboard envia para /labels
    request_id: str
    mode: Mode
    labels: List[Label]
    outcomes: List[Dict[str, Any]]  # dados pós-voo; nunca entram na previsão

    @model_validator(mode="after")
    def same_length(self):
        if len(self.labels) != len(self.outcomes):
            raise ValueError("labels e outcomes têm de ter o mesmo tamanho")
        return self