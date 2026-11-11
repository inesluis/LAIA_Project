# Configuração do serviço, lida de variáveis de ambiente 
import os

# URI do modelo no MLflow, formato "models:/<nome-do-modelo>@champion".
# O alias é "champion" (acordado com o Pedro); o <nome-do-modelo> é definido
# pelo Cura no script de treino -> ainda por confirmar.
# Vazio -> usa o modelo dummy (desenvolvimento e testes).
MODEL_URI = os.getenv("MODEL_URI", "")