# rag/utils/scoring.py
import re
from typing import Optional, Dict, Any, List, Tuple


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _split_words(text: str) -> List[str]:
    return (text or "").strip().split()


def _non_alnum_ratio(text: str) -> float:
    if not text:
        return 1.0
    allowed = set(" \n\t.,;:()[]{}-_/+%°")
    non_alnum = sum(1 for c in text if (
        not c.isalnum()) and (c not in allowed))
    return non_alnum / max(1, len(text))


# ---------------------------------------------------------
# Boost de área / densidade técnica
# ---------------------------------------------------------

def area_boost(text: str, area_keywords: Dict[str, float]) -> float:
    low = (text or "").lower()
    boost = 0.0
    for k, w in (area_keywords or {}).items():
        if k in low:
            boost += float(w)
    return boost


def tech_density(text: str, area_keywords: Dict[str, float]) -> float:
    low = (text or "").lower()
    if not low.strip():
        return 0.0

    hits = 0
    for k in (area_keywords or {}).keys():
        if k in low:
            hits += 1

    codes = re.findall(r"\b[A-Z]{2,}[0-9A-Z\-\._/]{1,}\b", text or "")
    units = re.findall(
        r"\b\d+(?:[\.,]\d+)?\s?(?:v|a|ma|hz|ohm|k\s?ohm|%|bar|mm|ºc|c)\b",
        low
    )

    total = max(1, len(_split_words(text)))
    return (hits + 0.5 * len(codes) + 0.3 * len(units)) / total


# ---------------------------------------------------------
# Chunk quality gates
# ---------------------------------------------------------

def chunk_is_bad(
    text: str,
    *,
    min_chunk_words: int = 26,
    bad_patterns: Optional[List[str]] = None
) -> bool:
    txt = (text or "").strip()
    if not txt:
        return True

    w = _split_words(txt)
    if len(w) < int(min_chunk_words):
        return True
    if len(w) > 2500:
        return True

    low = txt.lower()
    for p in (bad_patterns or []):
        try:
            if re.search(p, low):
                return True
        except Exception:
            # pattern inválido -> ignora
            pass

    # muito "lixo" / símbolos
    if _non_alnum_ratio(txt) > 0.20:
        return True

    return False


# ---------------------------------------------------------
# Utilidade do trecho
# ---------------------------------------------------------

def usefulness_score(
    text: str,
    *,
    usefulness_patterns: Dict[str, str],
    area_keywords: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Score de utilidade baseado em:
    - números/unidades
    - alarmes/códigos
    - passos/procedimentos
    - limites/tolerâncias
    + bônus por densidade técnica e area_boost
    """
    if not text:
        return 0.0

    low = text.lower()
    score = 0.0

    weights = weights or {
        "has_numbers": 0.7,
        "has_units": 1.2,
        "has_alarm": 1.0,
        "has_step": 1.1,
        "has_limits": 1.0,
        "has_code": 0.9,
    }

    for key, pat in (usefulness_patterns or {}).items():
        try:
            if re.search(pat, low, flags=re.I):
                score += float(weights.get(key, 0.5))
        except Exception:
            pass

    score += 2.8 * tech_density(text, area_keywords)
    score += 0.04 * area_boost(text, area_keywords)
    return float(score)


# ---------------------------------------------------------
# Recorte do melhor span
# ---------------------------------------------------------

def extract_best_span(text: str, query: str, max_len: int = 650) -> str:
    """
    Recorta o trecho para mostrar a parte mais útil (núcleo).
    Mantém contexto ao redor de hits.
    """
    if not text:
        return ""

    txt = re.sub(r"[ \t]+", " ", text).strip()
    if len(txt) <= max_len:
        return txt

    q = (query or "").strip().lower()
    q_terms = [t for t in re.findall(
        r"[a-zA-Z0-9_/%\.-]{3,}", q) if len(t) >= 3]
    if not q_terms:
        return txt[:max_len].rstrip() + "..."

    low = txt.lower()
    best_i = 0
    best_hits = -1
    window = int(max_len)

    step = max(120, window // 4)
    for start in range(0, max(1, len(txt) - window), step):
        end = start + window
        chunk = low[start:end]
        hits = sum(1 for t in q_terms if t in chunk)
        if hits > best_hits:
            best_hits = hits
            best_i = start

    span = txt[best_i:best_i + window].strip()
    span = span.strip(" .,:;-\n\t")
    if best_i > 0:
        span = "..." + span
    if (best_i + window) < len(txt):
        span = span + "..."
    return span


# ---------------------------------------------------------
# Coverage do conjunto de trechos
# ---------------------------------------------------------

def coverage_score(docs: List[str]) -> Dict[str, int]:
    low = "\n".join(docs or []).lower()
    return {
        "procedimento": int(any(k in low for k in ["procedimento", "passo", "step", "sequência", "sequencia"])),
        "alarme": int(any(k in low for k in ["alarme", "fault", "warning", "trip", "erro"])),
        "limite": int(any(k in low for k in ["limite", "range", "min", "max", "tolerância", "tolerancia"])),
        "parâmetro": int(any(k in low for k in ["parâmetro", "parametro", "setpoint", "sp", "pv", "ajuste"])),
    }


# ---------------------------------------------------------
# Score composto (GOLD)
# ---------------------------------------------------------

def composite_score(
    doc: str,
    meta: dict,
    rerank_score: Optional[float],
    query: str,
    *,
    area_keywords: Dict[str, float],
    usefulness_patterns: Dict[str, str],
    level_bonus: float = 1.0
) -> float:
    if not doc:
        return -999.0

    meta = meta or {}
    low = doc.lower()

    rr = float(rerank_score) if rerank_score is not None else 0.0

    useful = usefulness_score(
        doc,
        usefulness_patterns=usefulness_patterns,
        area_keywords=area_keywords
    )
    dens = tech_density(doc, area_keywords)
    boost = area_boost(doc, area_keywords) + \
        float(meta.get("area_boost", 0.0) or 0.0)

    q = (query or "").strip().lower()
    q_terms = [t for t in re.findall(
        r"[a-zA-Z0-9_/%\.-]{3,}", q) if len(t) >= 3]
    hits = sum(1 for t in q_terms if t in low)
    hits = min(hits, 10)

    # tipo
    tipo = meta.get("tipo", "texto")
    tipo_bonus = 0.05 if tipo == "texto" else -0.02

    # chunking bonus
    chunking = meta.get("chunking", "")
    chunking_bonus = 0.06 if chunking == "pagina" else 0.0

    # penaliza excesso de tamanho
    wlen = len(_split_words(doc))
    size_pen = 0.0
    if wlen > 700:
        size_pen = -0.12
    elif wlen > 520:
        size_pen = -0.06

    score = (
        1.75 * rr +
        1.55 * useful +
        2.25 * dens +
        0.05 * boost +
        0.24 * hits +
        tipo_bonus +
        chunking_bonus +
        size_pen
    )

    score = score * float(level_bonus or 1.0)
    return float(score)


# ---------------------------------------------------------
# Gates finais para qualidade do conjunto
# ---------------------------------------------------------

def quality_gates(
    fontes: List[Tuple[str, dict, Optional[float]]],
    *,
    gate_min_unique_pages: int = 5,
    gate_min_numbers: int = 8,
    gate_min_units: int = 4
) -> Dict[str, Any]:
    pages = set()
    tipos = set()
    text_all = ""

    for t, m, _ in fontes or []:
        meta = m or {}
        p = meta.get("page")
        if p is not None:
            pages.add(p)
        tipos.add(meta.get("tipo", "texto"))
        text_all += "\n" + (t or "")

    nums = re.findall(r"\b\d+(?:[\.,]\d+)?\b", text_all)
    units = re.findall(
        r"\b\d+(?:[\.,]\d+)?\s?(?:v|a|ma|hz|ohm|k\s?ohm|%|bar|mm|ºc|c)\b",
        text_all.lower()
    )

    return {
        "unique_pages": len(pages),
        "unique_types": len(tipos),
        "numbers": len(nums),
        "units": len(units),
        "ok_pages": len(pages) >= int(gate_min_unique_pages),
        "ok_numbers": len(nums) >= int(gate_min_numbers),
        "ok_units": len(units) >= int(gate_min_units),
    }
