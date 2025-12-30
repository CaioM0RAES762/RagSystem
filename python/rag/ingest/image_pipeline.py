import os
from typing import List, Dict, Any


def build_image_items_from_rendered_pages(
    rendered_pages: List[Dict[str, Any]],
    *,
    maquina_id: str,
    pdf_path: str,
) -> List[Dict[str, Any]]:
    """
    Apenas normaliza o output do render_pages_to_png para o formato
    que o seu salvamento (RAGSystem) espera.
    """
    items: List[Dict[str, Any]] = []
    for it in rendered_pages:
        p = it.get("path")
        if not p or not os.path.exists(p):
            continue
        items.append({
            "path": p,
            "page": int(it.get("page", 0) or 0),
            "indice": int(it.get("indice", 0) or 0),
            "hash": it.get("hash", "") or "",
            "pdf_path": pdf_path,
            "maquina_id": str(maquina_id),
            "imagem_fonte": "pagina_inteira",
        })
    return items
