# rag/page_selection.py
import re
from typing import Optional, List


def normalize_page_num(p: int, total_pages: int) -> int:
    """Converte página humana (1-based) para índice PyMuPDF (0-based)."""
    if p < 1:
        raise ValueError("Página deve ser >= 1")
    if p > total_pages:
        raise ValueError(f"Página {p} fora do range (1..{total_pages})")
    return p - 1


def parse_page_selection(selection: Optional[str], total_pages: int) -> List[int]:
    """
    Entrada exemplo:
      "20-60, 80,98,120"
    Retorna lista de índices 0-based, ordenada e sem duplicatas.
    Se selection vazio/None, retorna todas as páginas.
    """
    if not selection or not str(selection).strip():
        return list(range(total_pages))

    sel = str(selection).strip().replace(";", ",")
    sel = re.sub(r"\s+", "", sel)
    parts = [p for p in sel.split(",") if p]

    chosen = set()
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            if not a or not b:
                raise ValueError(f"Range inválido: '{part}'")
            start = int(a)
            end = int(b)
            if end < start:
                start, end = end, start
            for p in range(start, end + 1):
                chosen.add(normalize_page_num(p, total_pages))
        else:
            p = int(part)
            chosen.add(normalize_page_num(p, total_pages))

    return sorted(chosen)
