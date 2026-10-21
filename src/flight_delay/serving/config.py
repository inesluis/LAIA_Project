# Configuração do serviço, lida de variáveis de ambiente 
import os

# URI do modelo no MLflow, ex.: "models:/flight-delay-severity@champion".
# Vazio -> usa o modelo dummy (desenvolvimento e testes).
MODEL_URI = os.getenv("MODEL_URI", "")