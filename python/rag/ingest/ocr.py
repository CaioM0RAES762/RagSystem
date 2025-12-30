import os
import shutil
import subprocess
from typing import Tuple, Optional

import fitz  # PyMuPDF

from rag.logger import log_info, log_ok, log_warn, log_step
from rag.config import (
    OCR_LANGUAGE_DEFAULT,
    OCR_JOBS_DEFAULT,
    OCR_MAX_PAGES_DEFAULT,
)


def _pdf_page_count(pdf_path: str) -> Optional[int]:
    try:
        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return None


def run_ocr_if_needed(
    pdf_path: str,
    output_pdf_path: str,
    *,
    language: str = OCR_LANGUAGE_DEFAULT,
    deskew: bool = True,
    rotate_pages: bool = True,
    skip_text: bool = True,
    force_ocr: bool = False,
    jobs: int = OCR_JOBS_DEFAULT,
    max_pages: int = OCR_MAX_PAGES_DEFAULT,
) -> Tuple[bool, str]:

    if not os.path.exists(pdf_path):
        return False, f"PDF não existe: {pdf_path}"

    n_pages = _pdf_page_count(pdf_path)
    if n_pages and n_pages > max_pages:
        return False, f"PDF tem {n_pages} páginas (> {max_pages}). Abortando OCR."

    if shutil.which("ocrmypdf") is None:
        return False, "ocrmypdf não encontrado no PATH. Instale com: pip install ocrmypdf"

    cmd = ["ocrmypdf"]
    if skip_text and not force_ocr:
        cmd.append("--skip-text")
    if force_ocr:
        cmd.append("--force-ocr")
    if deskew:
        cmd.append("--deskew")
    if rotate_pages:
        cmd.append("--rotate-pages")

    cmd += ["-l", language, "--jobs", str(jobs), pdf_path, output_pdf_path]

    log_step("OCR offline (OCRmyPDF)")
    log_info("Executando: " + " ".join(cmd))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            return False, f"OCRmyPDF falhou (code={proc.returncode}): {err[:800]}"
        log_ok(f"OCR concluído: {output_pdf_path}")
        return True, f"OCR concluído: {output_pdf_path}"
    except Exception as e:
        log_warn(f"Falha ao executar OCRmyPDF: {e}")
        return False, f"Falha ao executar OCRmyPDF: {e}"
