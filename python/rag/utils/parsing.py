# rag/utils/parsing.py
import os
import re
from typing import Optional, Tuple, List


def parse_manual_filename(pdf_path: str) -> Tuple[Optional[int], str]:
    """
    Extrai manual_id e nome_maquina do nome do arquivo.
    Ex:
      123_FORNO_DICU.pdf -> (123, "FORNO DICU")
    Se não achar ID no começo, retorna (None, nome do arquivo sem extensão).
    """
    base = os.path.basename(pdf_path)
    name = os.path.splitext(base)[0].strip()

    m = re.match(r"^\s*(\d+)[_\-\s]+(.+)$", name)
    if m:
        try:
            mid = int(m.group(1))
        except Exception:
            mid = None
        nm = m.group(2).replace("_", " ").strip()
        return mid, nm

    return None, name.replace("_", " ").strip()


def ask_bool(prompt: str, default: bool = True) -> bool:
    suf = " [S/n] " if default else " [s/N] "
    val = input(prompt + suf).strip().lower()
    if not val:
        return default
    return val in ("s", "sim", "y", "yes", "1", "true")


def ask_int(prompt: str, default: int) -> int:
    val = input(f"{prompt} (default={default}): ").strip()
    if not val:
        return default
    try:
        return int(val)
    except Exception:
        return default


def ask_float(prompt: str, default: float) -> float:
    val = input(f"{prompt} (default={default}): ").strip().replace(",", ".")
    if not val:
        return default
    try:
        return float(val)
    except Exception:
        return default


def list_pdfs_in_dir(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    out = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(".pdf"):
            out.append(os.path.join(folder, fn))
    return out
