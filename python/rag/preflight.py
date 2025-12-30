# rag/preflight.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import fitz  # PyMuPDF
import os


@dataclass
class PagePreflight:
    page_index: int
    text_len: int
    word_count: int
    images_count: int
    is_likely_scanned: bool
    scan_score: float
    notes: str = ""


def preflight_pdf_pages(
    pdf_path: str,
    min_text_len: int = 80,
    min_word_count: int = 15,
    scanned_score_threshold: float = 0.65,
    max_pages: Optional[int] = None,
) -> List[PagePreflight]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    doc = fitz.open(pdf_path)
    total = len(doc)
    if max_pages:
        total = min(total, max_pages)

    results: List[PagePreflight] = []

    for i in range(total):
        page = doc[i]
        text = (page.get_text("text") or "").strip()
        text_len = len(text)
        word_count = len(text.split()) if text else 0
        imgs = page.get_images(full=True) or []
        images_count = len(imgs)

        text_bad = 1.0 if (
            text_len < min_text_len or word_count < min_word_count) else 0.0
        img_present = 1.0 if images_count > 0 else 0.0
        score = 0.65 * text_bad + 0.35 * img_present
        is_scanned = score >= scanned_score_threshold

        notes = []
        if text_len < min_text_len:
            notes.append(f"text_len<{min_text_len}")
        if word_count < min_word_count:
            notes.append(f"words<{min_word_count}")
        if images_count > 0:
            notes.append("has_images")
        if not notes:
            notes.append("ok_text")

        results.append(
            PagePreflight(
                page_index=i,
                text_len=text_len,
                word_count=word_count,
                images_count=images_count,
                is_likely_scanned=is_scanned,
                scan_score=round(score, 3),
                notes=",".join(notes),
            )
        )

    doc.close()
    return results


def preflight_summary(preflight: List[PagePreflight]) -> Dict[str, Any]:
    total = len(preflight)
    scanned = sum(1 for p in preflight if p.is_likely_scanned)
    with_text = sum(1 for p in preflight if p.text_len > 0)
    return {
        "total_pages_analyzed": total,
        "likely_scanned_pages": scanned,
        "likely_scanned_ratio": round(scanned / max(1, total), 3),
        "pages_with_any_text": with_text,
    }
