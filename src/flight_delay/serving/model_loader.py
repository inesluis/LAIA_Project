# Carregamento do modelo usado pelo serviço de inferência 

import pandas as pd
from .config import MODEL_URI

class DummyModel:
    # Modelo de teste: prevê sempre a classe maioritária (on_time) 
  
    def predict(self, df: pd.DataFrame) -> list[str]:
        return ["on_time"] * len(df)


def load_model(uri: str = MODEL_URI):
    # Carrega o modelo do MLflow se houver URI, senão, devolve o dummy 
    
    if not uri:
        return DummyModel()
    import mlflow  # import aqui: só é preciso quando há modelo real
    return mlflow.pyfunc.load_model(uri)