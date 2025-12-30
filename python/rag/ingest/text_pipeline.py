from typing import List, Dict, Any, Optional, Tuple

from rag.logger import log_step, log_ok, log_warn
from rag.text_extract import extrair_texto_pdf_por_pagina, extrair_texto_pdf
from rag.text_utils import limpar_texto
from rag.chunking import paginas_para_chunks_texto
from rag.config import GPT_MODEL_MINI, OpenAI, OPENAI_API_KEY


def extract_text_pages(pdf_path: str) -> List[str]:
    pages = extrair_texto_pdf_por_pagina(pdf_path) or []
    if pages:
        return pages

    # fallback legacy
    txt = limpar_texto(extrair_texto_pdf(pdf_path) or "")
    if txt.strip():
        return [txt]

    return []


def chunk_text_pages(
    pages_text: List[str],
    *,
    max_words_per_page: int,
    subchunk_size: int,
    subchunk_overlap: int,
) -> List[Dict[str, Any]]:
    return paginas_para_chunks_texto(
        pages_text,
        max_words_per_page=max_words_per_page,
        subchunk_size=subchunk_size,
        subchunk_overlap=subchunk_overlap,
    )


def detect_language_light(text_sample: str) -> str:
    """
    detector leve via OpenAI (se disponível).
    retorna código: pt/en/es etc. ou 'desconhecido'
    """
    if not (OpenAI and OPENAI_API_KEY):
        return "desconhecido"

    if not text_sample or not text_sample.strip():
        return "vazio"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=GPT_MODEL_MINI,
            messages=[
                {"role": "system",
                    "content": "Você é um detector de idiomas. Responda APENAS com o código do idioma (pt, en, es, etc)."},
                {"role": "user",
                    "content": f"Qual o idioma deste texto? Responda apenas com o código:\n\n{text_sample[:500]}"},
            ],
            temperature=0,
            max_tokens=10,
        )
        idioma = (resp.choices[0].message.content or "").strip().lower()
        return idioma or "desconhecido"
    except Exception:
        return "desconhecido"
