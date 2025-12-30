import os
import shutil
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF

from rag.logger import log_step, log_info, log_ok, log_warn
from rag.config import (
    RENDER_DPI_DEFAULT,
    PHASH_MAX_DISTANCE_DEFAULT,
    PAGE_MEAN_MIN_DEFAULT,
    PAGE_MEAN_MAX_DEFAULT,
    PAGE_STD_MIN_DEFAULT,
)


def render_page_to_png(
    pdf_path: str,
    page_index: int,
    output_dir: str,
    *,
    file_prefix: str,
    dpi: int = RENDER_DPI_DEFAULT,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc[page_index]

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=mat, alpha=False)
    filename = f"{file_prefix}_page_{page_index+1}.png"
    out_path = os.path.join(output_dir, filename)

    pix.save(out_path)
    doc.close()

    return out_path


def is_rendered_page_useful(
    png_path: str,
    *,
    mean_min: float = PAGE_MEAN_MIN_DEFAULT,
    mean_max: float = PAGE_MEAN_MAX_DEFAULT,
    std_min: float = PAGE_STD_MIN_DEFAULT,
) -> Tuple[bool, str]:
    if not png_path or not os.path.exists(png_path):
        return False, "arquivo_inexistente"

    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return True, "sem_pillow_numpy"

    try:
        im = Image.open(png_path).convert("L")
        arr = np.array(im)
        mean = float(arr.mean())
        std = float(arr.std())

        if mean < mean_min:
            return False, f"muito_escura(mean={mean:.1f})"
        if mean > mean_max:
            return False, f"muito_clara(mean={mean:.1f})"
        if std < std_min:
            return False, f"pouca_variacao(std={std:.1f})"

        return True, f"ok(mean={mean:.1f},std={std:.1f})"
    except Exception as e:
        return True, f"erro_analise({e})"


def phash_dedupe_pngs(
    png_paths: List[str],
    *,
    max_distance: int = PHASH_MAX_DISTANCE_DEFAULT
) -> Tuple[List[str], List[str]]:
    try:
        from PIL import Image
        import imagehash
    except Exception as e:
        raise RuntimeError(
            "Instale pillow e imagehash: pip install pillow imagehash") from e

    kept: List[str] = []
    dropped: List[str] = []
    hashes: List[Tuple[str, Any]] = []

    for p in png_paths:
        if not os.path.exists(p):
            continue

        try:
            with Image.open(p) as im:
                h = imagehash.phash(im)
        except Exception:
            kept.append(p)
            continue

        is_dup = False
        for _, prev_h in hashes:
            if (h - prev_h) <= max_distance:
                is_dup = True
                break

        if is_dup:
            dropped.append(p)
        else:
            hashes.append((p, h))
            kept.append(p)

    return kept, dropped


def render_pages_to_png(
    pdf_path: str,
    output_dir: str,
    *,
    file_prefix: str,
    page_indices: Optional[List[int]] = None,
    dpi: int = RENDER_DPI_DEFAULT,
    filtrar_paginas_vazias: bool = True,
    dedupe_phash: bool = False,
    phash_max_distance: int = PHASH_MAX_DISTANCE_DEFAULT,
) -> List[Dict[str, Any]]:

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    if page_indices is None:
        page_indices = list(range(total_pages))

    page_indices = [i for i in page_indices if 0 <= i < total_pages]
    if not page_indices:
        return []

    log_step("Renderização de páginas (PNG)")
    log_info(
        f"Páginas totais={total_pages} | selecionadas={len(page_indices)} | dpi={dpi}")

    rendered_paths: List[str] = []
    meta_by_path: Dict[str, Dict[str, Any]] = {}

    for idx0 in page_indices:
        try:
            png_path = render_page_to_png(
                pdf_path=pdf_path,
                page_index=idx0,
                output_dir=output_dir,
                file_prefix=file_prefix,
                dpi=dpi,
            )

            if filtrar_paginas_vazias:
                ok, motivo = is_rendered_page_useful(png_path)
                if not ok:
                    log_warn(
                        f"Página {idx0+1} não útil: {motivo} | {png_path} (mantendo mesmo assim)")
                else:
                    log_info(f"Página {idx0+1} OK: {motivo}")

            try:
                with open(png_path, "rb") as f:
                    b = f.read()
                md5 = hashlib.md5(b).hexdigest()
            except Exception:
                md5 = ""

            rendered_paths.append(png_path)
            meta_by_path[png_path] = {
                "path": png_path,
                "page": idx0 + 1,
                "indice": 0,
                "hash": md5,
            }

        except Exception as e:
            log_warn(f"Falha ao renderizar página {idx0+1}: {e}")

    if dedupe_phash and len(rendered_paths) > 1:
        try:
            kept, dropped = phash_dedupe_pngs(
                rendered_paths, max_distance=phash_max_distance)
            for p in dropped:
                try:
                    os.remove(p)
                except Exception:
                    pass
                meta_by_path.pop(p, None)
            rendered_paths = kept
            log_info(f"pHash dedupe: removidas {len(dropped)} duplicadas")
        except Exception as e:
            log_warn(f"Falha pHash dedupe (ignorando): {e}")

    out = [meta_by_path[p] for p in rendered_paths if p in meta_by_path]
    log_ok(f"Páginas renderizadas válidas: {len(out)}")
    return out
