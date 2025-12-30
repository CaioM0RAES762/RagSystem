# rag/config.py (GOLD 2025-12-26 v4 - FIX CHROMA ROOT AUTO)
import os
from pathlib import Path
from dotenv import load_dotenv

# ==========================================================
# ROOT / ENV (robusto: detecta a raiz do projeto e procura .env)
# ==========================================================


def _find_project_root(start: Path, max_up: int = 8) -> Path:
    """
    Sobe pastas até encontrar algo que indique a raiz do projeto.
    Prioridade:
      - package.json (Node backend)
      - server.js
      - pyproject.toml / requirements.txt
      - .env
    """
    p = start
    for _ in range(max_up):
        if (p / "package.json").exists():
            return p
        if (p / "server.js").exists():
            return p
        if (p / "pyproject.toml").exists():
            return p
        if (p / "requirements.txt").exists():
            return p
        if (p / ".env").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start


def _find_env(start: Path, filename: str = ".env", max_up: int = 8) -> Path:
    """
    Procura .env subindo pastas.
    """
    p = start
    for _ in range(max_up):
        candidate = p / filename
        if candidate.exists():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    return start / filename  # fallback


THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
ROOT_DIR = _find_project_root(THIS_DIR)

ENV_PATH = _find_env(ROOT_DIR)
load_dotenv(dotenv_path=str(ENV_PATH), override=False)

# ==========================================================
# OPENAI
# ==========================================================
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# ==========================================================
# MODELOS (recomendado para >= 9.5)
# ==========================================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5-mini")
GPT_MODEL_MINI = os.getenv("GPT_MODEL_MINI", "gpt-5-mini")

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5.1")
VISION_MODEL_DB = os.getenv("MANUALS_VISION_MODEL", VISION_MODEL)

# ==========================================================
# CHROMA (FIX DEFINITIVO)
# - Garante ABSOLUTO e aponta pro chroma_db do backend
# ==========================================================


def _resolve_chroma_path() -> str:
    """
    Resolve o path final do Chroma de maneira segura:
    - se existir CHROMA_DB_PATH no env, usa
    - se for relativo, transforma em absoluto baseado no ROOT_DIR
    - se não existir env, usa ROOT_DIR/chroma_db
    """
    env_path = (os.getenv("CHROMA_DB_PATH") or "").strip()

    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = (ROOT_DIR / p).resolve()
        return str(p)

    return str((ROOT_DIR / "chroma_db").resolve())


CHROMA_DB_PATH = _resolve_chroma_path()
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# 🔥 garante que tudo que importar rag/config verá o mesmo valor
os.environ["CHROMA_DB_PATH"] = CHROMA_DB_PATH


print("\n" + "=" * 96)
print("✅ [RAG CONFIG] ROOT_DIR:", str(ROOT_DIR))
print("✅ [RAG CONFIG] ENV_PATH:", str(ENV_PATH))
print("✅ [RAG CONFIG] CHROMA_DB_PATH (FINAL ABS):", CHROMA_DB_PATH)
print("✅ [RAG CONFIG] CWD:", os.getcwd())
print("=" * 96 + "\n")

# ==========================================================
# POSTGRES
# ==========================================================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "metalsider_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "admin123")
DB_PORT = os.getenv("DB_PORT", "5432")

# ==========================================================
# OCR defaults
# ==========================================================
OCR_MAX_PAGES_DEFAULT = int(os.getenv("OCR_MAX_PAGES", "400"))
OCR_LANGUAGE_DEFAULT = os.getenv("OCR_LANGUAGE", "por+eng")
OCR_JOBS_DEFAULT = int(os.getenv("OCR_JOBS", "2"))
OCR_SCANNED_RATIO_THRESHOLD_DEFAULT = float(
    os.getenv("OCR_SCANNED_RATIO_THRESHOLD", "0.5"))

# ==========================================================
# RENDER defaults
# ==========================================================
RENDER_DPI_DEFAULT = int(os.getenv("RENDER_DPI", "160"))
PHASH_MAX_DISTANCE_DEFAULT = int(os.getenv("PHASH_MAX_DISTANCE", "6"))

# ==========================================================
# FILTRO de página útil
# ==========================================================
PAGE_MEAN_MIN_DEFAULT = float(os.getenv("PAGE_MEAN_MIN", "10"))
PAGE_MEAN_MAX_DEFAULT = float(os.getenv("PAGE_MEAN_MAX", "245"))
PAGE_STD_MIN_DEFAULT = float(os.getenv("PAGE_STD_MIN", "5"))

# ==========================================================
# TEXTO por página / chunking
# ==========================================================
TEXT_PAGE_MAX_WORDS_DEFAULT = int(os.getenv("TEXT_PAGE_MAX_WORDS", "999"))
TEXT_SUBCHUNK_SIZE_DEFAULT = int(os.getenv("TEXT_SUBCHUNK_SIZE", "450"))
TEXT_SUBCHUNK_OVERLAP_DEFAULT = int(os.getenv("TEXT_SUBCHUNK_OVERLAP", "60"))

# ==========================================================
# PATHS (uploads)
# ==========================================================
UPLOADS_DIR = Path(
    os.getenv("UPLOADS_DIR", str(ROOT_DIR / "uploads"))).resolve()
MANUALS_IMAGES_DIR = str((UPLOADS_DIR / "manuals_images").resolve())
MANUALS_PAGES_DIR = str((UPLOADS_DIR / "manuals_pages").resolve())

os.makedirs(MANUALS_IMAGES_DIR, exist_ok=True)
os.makedirs(MANUALS_PAGES_DIR, exist_ok=True)

# ==========================================================
# ENV VARS (OCR) - só seta se existir
# ==========================================================


def _set_env_if_exists(var: str, value: str):
    if value and Path(value).exists():
        os.environ[var] = value


_set_env_if_exists("GS", os.getenv(
    "GS", r"C:\Program Files\gs\gs10.06.0\bin\gswin64c.exe"))
_set_env_if_exists("TESSDATA_PREFIX", os.getenv(
    "TESSDATA_PREFIX", r"C:\Program Files\Tesseract-OCR\tessdata"))

# ==========================================================
# PROMPT (IMAGENS) — JSON schema real e válido
# ==========================================================
SYSTEM_PROMPT_IMAGENS = r"""
Você é um EXTRATOR TÉCNICO INDUSTRIAL (nível comissionamento/PLC) a partir de IMAGENS de manuais (fotos de painel, prints de HMI/SCADA, diagramas elétricos, tabelas e páginas escaneadas).
Seu objetivo é transformar CADA imagem em DADOS ESTRUTURADOS, acionáveis e auditáveis para RAG/KB, SEM misturar contexto de outras páginas.

REGRA ZERO (anti-alucinação / anti-contaminação)
- Extraia SOMENTE o que está VISÍVEL nesta imagem.
- Se algo estiver ilegível: escreva "ilegível" e liste o que precisa de zoom/corte em open_questions.

PRIORIDADE OBRIGATÓRIA (ordem)
1) NÚMEROS E UNIDADES visíveis (display, tabelas, limites, frequências, tensões, correntes, resistências).
2) I/O e CONECTORES (X1..Xn), pinos/terminais, sinais (+24V, GND, IN1..IN4, 0–10V, 4–20mA etc.).
3) LEDs/ALARMES/BOTÕES (nome exato impresso).
4) PROCEDIMENTOS (passos presentes na página, condições e resultados esperados).
5) COMPONENTES e TAGS de diagrama (L1, C1, F1, M1, part numbers, códigos).
6) Interpretação funcional curta (somente se explícita).

SAÍDA OBRIGATÓRIA
Responda SOMENTE em JSON válido (sem markdown e sem texto fora do JSON),
seguindo exatamente este schema:

{
  "page_type": "hmi|diagram|table|procedure|wiring|photo|unknown",
  "main_topic": "string",
  "visible_procedures": [
    {
      "step": "string",
      "expected_result": "string",
      "notes": "string"
    }
  ],
  "alarms_or_faults": [
    {
      "code": "string",
      "description": "string",
      "condition": "string"
    }
  ],
  "parameters_limits": [
    {
      "name": "string",
      "value": "string",
      "unit": "string",
      "range": "string"
    }
  ],
  "io_connectors": [
    {
      "connector": "string",
      "pin_or_terminal": "string",
      "signal": "string",
      "value_or_unit": "string"
    }
  ],
  "components_tags": [
    {
      "tag": "string",
      "description": "string"
    }
  ],
  "numbers_units": [
    {
      "raw": "string"
    }
  ],
  "short_summary": "string",
  "confidence": "low|medium|high",
  "legibility": "ok|partial|poor",
  "open_questions": [
    "string"
  ]
}
"""
