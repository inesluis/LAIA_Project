"""Configuração comum a todos os testes.

Os testes nunca usam o MLflow nem o modelo real: com MODEL_URI vazio,
a API carrega o DummyModel. Isto tem de acontecer antes de a API ser
importada, porque o config.py lê a variável no momento do import.
"""

import os

os.environ["MODEL_URI"] = ""
