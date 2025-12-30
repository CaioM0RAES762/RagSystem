# rag/utils/paths.py
import os


def abs_path(p: str) -> str:
    """Retorna path absoluto com fallback seguro."""
    try:
        return os.path.abspath(p)
    except Exception:
        return p


def ensure_dir(p: str) -> str:
    """Garante diretório existente e retorna o path absoluto."""
    ap = abs_path(p)
    os.makedirs(ap, exist_ok=True)
    return ap


def safe_len(x) -> int:
    """len(x) com fallback."""
    try:
        return len(x)
    except Exception:
        return 0
