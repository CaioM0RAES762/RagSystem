import re
from typing import Optional, Dict, Any


class ChunkFilters:
    """
    Reúne filtros de qualidade e heurísticas:
    - chunk_is_bad
    - tech_density
    - area_boost
    - usefulness_score
    """

    def __init__(
        self,
        *,
        area_keywords: Optional[Dict[str, float]] = None,
        usefulness_patterns: Optional[Dict[str, str]] = None,
        bad_patterns=None,
        min_chunk_words: int = 26,
        min_tech_density: float = 0.008,
    ):
        self.area_keywords = area_keywords or {}
        self.usefulness_patterns = usefulness_patterns or {}

        self.bad_patterns = bad_patterns or [
            r"\bconfidential\b",
            r"\ball rights reserved\b",
            r"\bwww\.",
            r"\bthis document\b",
            r"\bdisclaimer\b",
        ]

        self.min_chunk_words = min_chunk_words
        self.min_tech_density = min_tech_density

    # ---------------------------
    # boost por keywords de área
    # ---------------------------
    def area_boost(self, text: str) -> float:
        low = (text or "").lower()
        boost = 0.0
        for k, w in self.area_keywords.items():
            if k in low:
                boost += float(w)
        return boost

    # ---------------------------
    # densidade técnica
    # ---------------------------
    def tech_density(self, text: str) -> float:
        low = (text or "").lower()
        if not low.strip():
            return 0.0

        hits = 0
        for k in self.area_keywords.keys():
            if k in low:
                hits += 1

        codes = re.findall(r"\b[A-Z]{2,}[0-9A-Z\-\._/]{1,}\b", text or "")
        units = re.findall(
            r"\b\d+(?:[\.,]\d+)?\s?(?:v|a|ma|hz|ohm|k\s?ohm|%|bar|mm|ºc|c)\b",
            low
        )

        total = max(1, len((text or "").split()))
        return (hits + 0.5 * len(codes) + 0.3 * len(units)) / total

    # ---------------------------
    # chunk ruim
    # ---------------------------
    def chunk_is_bad(self, text: str) -> bool:
        txt = (text or "").strip()
        if not txt:
            return True
        w = txt.split()
        if len(w) < self.min_chunk_words:
            return True
        if len(w) > 2500:
            return True

        low = txt.lower()
        for p in self.bad_patterns:
            if re.search(p, low):
                return True

        non_alnum = sum(1 for c in txt if not c.isalnum()
                        and c not in " \n\t.,;:()[]{}-_/+%°")
        if non_alnum > 0.20 * len(txt):
            return True

        return False

    # ---------------------------
    # utilidade do trecho
    # ---------------------------
    def usefulness_score(self, text: str) -> float:
        if not text:
            return 0.0

        low = text.lower()
        score = 0.0

        weights = {
            "has_numbers": 0.7,
            "has_units": 1.2,
            "has_alarm": 1.0,
            "has_step": 1.1,
            "has_limits": 1.0,
            "has_code": 0.9,
        }

        for key, pat in self.usefulness_patterns.items():
            try:
                if re.search(pat, low, flags=re.I):
                    score += weights.get(key, 0.5)
            except Exception:
                pass

        score += 2.8 * self.tech_density(text)
        score += 0.04 * self.area_boost(text)
        return score

    # ---------------------------
    # composite score
    # ---------------------------
    def composite_score(
        self,
        doc: str,
        meta: Dict[str, Any],
        rerank_score: Optional[float],
        query: str,
        *,
        level_bonus: float = 1.0,
    ) -> float:
        if not doc:
            return -999.0
        meta = meta or {}
        low = doc.lower()

        rr = float(rerank_score) if (rerank_score is not None) else 0.0

        useful = self.usefulness_score(doc)
        dens = self.tech_density(doc)
        boost = self.area_boost(
            doc) + float(meta.get("area_boost", 0.0) or 0.0)

        q = (query or "").strip().lower()
        q_terms = [t for t in re.findall(
            r"[a-zA-Z0-9_/%\.-]{3,}", q) if len(t) >= 3]
        hits = sum(1 for t in q_terms if t in low)
        hits = min(hits, 10)

        tipo = meta.get("tipo", "texto")
        tipo_bonus = 0.05 if tipo == "texto" else -0.02

        chunking = meta.get("chunking", "")
        chunking_bonus = 0.06 if chunking == "pagina" else 0.0

        wlen = len((doc or "").split())
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

        return float(score) * float(level_bonus or 1.0)
