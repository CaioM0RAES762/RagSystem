# rag/text_extract.py
from typing import List
import PyPDF2

from .logger import log_err


def extrair_texto_pdf(pdf_path: str) -> str:
    """LEGACY: concatenado"""
    texto = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                texto += f"\n--- Pagina {page_num + 1} ---\n"
                texto += (page.extract_text() or "")
    except Exception as e:
        log_err(f"Erro ao extrair texto do PDF: {e}")
    return texto


def extrair_texto_pdf_por_pagina(pdf_path: str) -> List[str]:
    """Novo: retorna lista com o texto de cada página."""
    paginas: List[str] = []
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                paginas.append(page.extract_text() or "")
    except Exception as e:
        log_err(f"Erro ao extrair texto por página do PDF: {e}")
    return paginas
