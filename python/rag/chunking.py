# rag/chunking.py
from typing import List, Dict, Any

from .text_utils import limpar_texto


def dividir_em_chunks(texto: str, tamanho: int = 500, overlap: int = 50) -> List[str]:
    palavras = texto.split()
    if not palavras:
        return []
    chunks = []
    step = max(1, tamanho - overlap)
    for i in range(0, len(palavras), step):
        chunk = " ".join(palavras[i: i + tamanho]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def paginas_para_chunks_texto(
    paginas_texto: List[str],
    *,
    max_words_per_page: int,
    subchunk_size: int,
    subchunk_overlap: int,
) -> List[Dict[str, Any]]:
    """
    Converte textos por página em chunks com metadados:
    - se página <= max_words_per_page -> 1 chunk
    - se maior -> subchunks mantendo page + subchunk_index
    Retorna: {"page": int, "subchunk": int, "text": str}
    """
    out: List[Dict[str, Any]] = []
    for idx0, raw in enumerate(paginas_texto):
        page_num = idx0 + 1
        cleaned = limpar_texto(raw or "")
        if not cleaned.strip():
            continue

        words = cleaned.split()
        if len(words) <= max_words_per_page:
            out.append({"page": page_num, "subchunk": 0, "text": cleaned})
            continue

        subchunks = dividir_em_chunks(
            cleaned, tamanho=subchunk_size, overlap=subchunk_overlap)
        for si, sc in enumerate(subchunks):
            out.append({"page": page_num, "subchunk": si, "text": sc})

    return out
