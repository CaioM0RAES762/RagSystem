# rag/generation/__init__.py

from .vision_describe import gerar_descricao_imagem
from .answer_gold import gerar_solucao_texto_gold, citations_ok
from .report import gerar_relatorio_trechos

__all__ = [
    "gerar_descricao_imagem",
    "gerar_solucao_texto_gold",
    "citations_ok",
    "gerar_relatorio_trechos",
]
