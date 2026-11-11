# Schemas de entrada e saída do endpoint /classify 

# nota: é PROVISÓRIO - os campos e os nomes vêm das colunas BTS permitidas (antes da partida)  
# final version tem de seguir o data contract do pedro

from datetime import date
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

Label = Literal["on_time", "moderate_delay", "severe_delay"]

class FlightRecord(BaseModel): # um voo, com atributos disponíveis antes da partida programada  
     
    model_config = ConfigDict(extra="ignore") # se enviar campos a mais ignoram-se, não se rejeitam 

    FL_DATE: date
    OP_UNIQUE_CARRIER: str          # nome exato a confirmar com A/B
    ORIGIN: str                     # str livre: aeroportos novos não dão erro
    DEST: str
    CRS_DEP_TIME: int = Field(ge=0, le=2400)    
    CRS_ARR_TIME: int = Field(ge=0, le=2400)
    CRS_ELAPSED_TIME: Optional[float] = None # se o modelo do cura souber lidar com missing data
    DISTANCE: float = Field(gt=0)
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
    instances: List[FlightRecord]

class ClassifyResponse(BaseModel):
    classifications: List[Label]