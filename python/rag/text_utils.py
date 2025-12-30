# rag/text_utils.py
import re


def limpar_texto(texto: str) -> str:
    if not texto:
        return ""

    texto = re.sub(r"\n?---\s*Pagina\s*\d+\s*---\n?",
                   " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", texto)
    texto = texto.replace("\r", " ").replace("\n", " ")
    texto = re.sub(r"\.{4,}", "...", texto)
    texto = re.sub(r"\s{2,}", " ", texto)
    return texto.strip()
