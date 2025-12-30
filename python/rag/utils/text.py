# rag/utils/text.py
import re


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def safe_lower(text: str) -> str:
    return (text or "").strip().lower()
