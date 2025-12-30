# rag/utils/__init__.py

from .paths import abs_path, ensure_dir, safe_len
from .parsing import (
    parse_manual_filename,
    ask_bool,
    ask_int,
    ask_float,
    list_pdfs_in_dir,
)
from .scoring import (
    area_boost,
    tech_density,
    chunk_is_bad,
    usefulness_score,
    extract_best_span,
    coverage_score,
    composite_score,
    quality_gates,
)
