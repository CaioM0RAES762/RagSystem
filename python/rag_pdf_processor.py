import os
import re
import time
import json
import base64
import hashlib
import shutil
import subprocess
from typing import Optional, Tuple, Dict, Any, List

import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import psycopg2
from psycopg2.extras import RealDictCursor

# -----------------------------------------------------
# IMPORTS DA SUA ARQUITETURA (pasta rag/)
# -----------------------------------------------------
from rag.config import (
    OpenAI, OPENAI_API_KEY,
    CHROMA_DB_PATH, EMBEDDING_MODEL,
    GPT_MODEL, GPT_MODEL_MINI, VISION_MODEL,
    DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT,
    VISION_MODEL_DB,
    OCR_MAX_PAGES_DEFAULT, OCR_LANGUAGE_DEFAULT, OCR_JOBS_DEFAULT,
    OCR_SCANNED_RATIO_THRESHOLD_DEFAULT,
    RENDER_DPI_DEFAULT, PHASH_MAX_DISTANCE_DEFAULT,
    PAGE_MEAN_MIN_DEFAULT, PAGE_MEAN_MAX_DEFAULT, PAGE_STD_MIN_DEFAULT,
    TEXT_PAGE_MAX_WORDS_DEFAULT, TEXT_SUBCHUNK_SIZE_DEFAULT, TEXT_SUBCHUNK_OVERLAP_DEFAULT,
    SYSTEM_PROMPT_IMAGENS,
    MANUALS_IMAGES_DIR, MANUALS_PAGES_DIR
)

from rag.logger import (
    fmt_secs as _fmt_secs,
    log_step as _log_step,
    log_info as _log_info,
    log_ok as _log_ok,
    log_warn as _log_warn,
    log_err as _log_err,
)

from rag.page_selection import parse_page_selection
from rag.preflight import preflight_pdf_pages, preflight_summary
from rag.text_utils import limpar_texto
from rag.text_extract import extrair_texto_pdf, extrair_texto_pdf_por_pagina
from rag.chunking import paginas_para_chunks_texto


# ==========================================================
# Helpers de PATH / ambiente (IMPORTANTE para não duplicar DB)
# ==========================================================

def _abs_path(p: str) -> str:
    try:
        return os.path.abspath(p)
    except Exception:
        return p


def _ensure_dir(p: str) -> str:
    ap = _abs_path(p)
    os.makedirs(ap, exist_ok=True)
    return ap


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


# ==========================================================
# RAG GOLD SYSTEM (APrimorado)
# ==========================================================

class RAGSystem:
    """
    RAG GOLD (APrimorado)

    Esta versão foi revisada para GARANTIR retrieval, mesmo quando:
    - where com $and quebra,
    - query() falha por dim_index None,
    - collection.count() é bugado,
    - CHROMA_DB_PATH relativo aponta para outro banco.

    Estratégia:
    1) Normaliza CHROMA_DB_PATH para absoluto e loga CWD/ABS.
    2) Usa count via get() como fonte de verdade.
    3) Retrieval híbrido multi-estratégia com relaxamento:
        - embedding query com where strict -> grupo_only -> none
        - keyword query_texts com os mesmos wheres
        - substring scan via get() (fallback seguro)
    4) Re-rank opcional com CrossEncoder.
    5) Seleção final melhorada: score composto + MMR + diversificação + cobertura.
    6) Novo modo RELATÓRIO: retorna os trechos encontrados + justificativa.
    """

    # -----------------------------------------------------
    # INIT
    # -----------------------------------------------------
    def __init__(self, traduzir_automaticamente: bool = True, grupo: str = "geral"):
        t0 = time.time()

        self.traduzir_automaticamente = bool(traduzir_automaticamente)
        self.grupo = (grupo or "geral").strip() or "geral"

        # ---------- PATH ABSOLUTO (crítico) ----------
        self.chroma_path = _ensure_dir(CHROMA_DB_PATH)

        _log_info(
            f"Inicializando RAG GOLD... (Tradução: {self.traduzir_automaticamente}, Grupo: {self.grupo})"
        )
        _log_info(
            f"[PATH] CHROMA_DB_PATH={CHROMA_DB_PATH} | ABS={self.chroma_path} | CWD={os.getcwd()}"
        )

        self._dbg_env_loaded = self._debug_check_dotenv_loaded()

        # ---------- CHROMA ----------
        self.chroma_client = None
        self.collection = None
        self.collection_name = None

        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection_name = self._collection_name_for_group(self.grupo)

            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "grupo": self.grupo},
            )
            _log_ok(f"Usando coleção Chroma: {self.collection_name}")
        except Exception as e:
            _log_warn(f"Erro ao inicializar ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None

        # ---------- EMBEDDINGS ----------
        self.embedder = None
        self.embedding_dim = None

        try:
            _log_info(f"Carregando modelo de embeddings: {EMBEDDING_MODEL}")
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)

            try:
                v = self.embedder.encode(["teste_dim"]).tolist()
                if v and isinstance(v, list):
                    self.embedding_dim = len(v[0]) if isinstance(
                        v[0], list) else len(v)
                _log_ok(f"Embedder carregado. dim={self.embedding_dim}")
            except Exception:
                _log_ok("Embedder carregado.")
        except Exception as e:
            _log_warn(f"Erro ao carregar modelo de embeddings: {e}")
            self.embedder = None

        # ---------- RERANKER ----------
        self.reranker = None
        try:
            from sentence_transformers import CrossEncoder
            _log_info(
                "Carregando reranker CrossEncoder (ms-marco-MiniLM-L-6-v2)...")
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2")
            _log_ok("Reranker carregado.")
        except Exception as e:
            _log_warn(f"Reranker indisponível (vai rodar sem rerank): {e}")
            self.reranker = None

        # ---------- OPENAI ----------
        self.openai_client = None
        if OpenAI and OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                _log_ok("OpenAI client inicializado.")
            except Exception as e:
                _log_warn(f"Erro ao inicializar OpenAI client: {e}")
                self.openai_client = None
        else:
            _log_warn("OpenAI client não disponível (biblioteca/OPENAI_API_KEY).")

        # ---------- PASTAS ----------
        self.image_output_dir = _ensure_dir(
            os.path.join(MANUALS_IMAGES_DIR, self.grupo))
        self.pages_output_dir = _ensure_dir(
            os.path.join(MANUALS_PAGES_DIR, self.grupo))
        _log_ok(f"Pasta imagens: {self.image_output_dir}")
        _log_ok(f"Pasta páginas: {self.pages_output_dir}")

        self._pg_schema_cache = None

        # -------------------------------------------------
        # PARÂMETROS GOLD (refinados)
        # -------------------------------------------------
        self.retrieval_topk = 180
        self.retrieval_prefilter_max = 110
        self.final_k = 16

        self.rerank_score_min = 0.16
        self.min_evidence = 5

        # imagens
        self.imagem_score_min = 0.22
        self.max_imagens_para_resposta = 2
        self.max_fontes_para_resposta = 10

        # filtros
        self.min_chunk_words = 26
        self.max_chunk_words = 520
        self.min_tech_density = 0.008

        # keyword retrieval
        self.keyword_topk = 40

        # substring scan
        self.substring_scan_max_docs = 2200
        self.substring_scan_max_return = 40
        self.substring_min_hits = 1

        # MMR
        self.mmr_lambda = 0.76
        self.mmr_max_total = 32

        # 2-pass refine
        self.enable_two_pass = True
        self.two_pass_temperature = 0.14
        self.two_pass_max_tokens = 1250

        # cobertura mínima esperada (não obrigatória)
        self.coverage_required = ["procedimento", "parâmetro"]

        # Trechos ruins
        self.bad_patterns = [
            r"\bconfidential\b",
            r"\ball rights reserved\b",
            r"\bwww\.",
            r"\bthis document\b",
            r"\bdisclaimer\b",
        ]

        # Vocabulário de área/boost
        self.area_keywords = {
            "forno": 4.0, "fornos": 4.0, "dicu": 4.2,
            "fusao": 3.8, "fusão": 3.8,
            "prodapt": 3.0, "prodapt md": 3.8,
            "fundicao": 2.6, "fundição": 2.6,
            "disa": 2.6, "moldagem": 2.2, "areia": 2.0,
            "queimador": 2.6, "termopar": 2.6,
            "refrat": 2.2, "refratário": 2.2,
            "chama": 2.0, "temperatura": 2.0,
            "metal liquido": 2.6, "metal líquido": 2.6,
            "cadinho": 2.2, "vazamento": 2.1,
            "oxigenio": 1.8, "oxigênio": 1.8,
            "pressao": 1.6, "pressão": 1.6,
            "ignicao": 1.7, "ignição": 1.7,
            "intertravamento": 2.2, "permissiva": 2.2,
            "alarme": 2.0, "fault": 2.0, "trip": 2.0,
            "reset": 1.2, "shutdown": 2.0,
            "pid": 1.6, "setpoint": 1.8, "sp": 1.2, "pv": 1.2,
            "válvula": 1.6, "valvula": 1.6, "bomba": 1.6,
        }

        # Heurísticas de utilidade (quanto mais disso no trecho, melhor)
        self.usefulness_patterns = {
            "has_numbers": r"\b\d+(?:[\.,]\d+)?\b",
            "has_units": r"\b\d+(?:[\.,]\d+)?\s?(?:v|a|ma|hz|ohm|k\s?ohm|%|bar|mm|ºc|c)\b",
            "has_alarm": r"\b(alarme|fault|warning|trip|erro|error)\b",
            "has_step": r"\b(passo|procedimento|step|sequência|sequencia|verifique|ajuste|medir|medição|teste)\b",
            "has_limits": r"\b(limite|min\.|máx\.|max\.|range|tolerância|tolerancia)\b",
            "has_code": r"\b[A-Z]{2,}[0-9A-Z\-\._/]{1,}\b",
        }
        

        # -------------------------------------------------
        # GOLD PRO: retrieval ladder + bônus + compressão + gates
        # -------------------------------------------------
        self.enable_retrieval_ladder = True

        # níveis do ladder (mais rígido -> mais relaxado)
        # cada nível aplica: filtros + wheres + thresholds e tem um bônus
        self.retrieval_ladder_levels = [
            {"name": "L0_ultra",  "where_mode": "maquina", "topk": 160,
                "min_density": 0.010, "min_words": 28, "score_min": 0.18, "bonus": 1.35},
            {"name": "L1_strict", "where_mode": "maquina", "topk": 180,
                "min_density": 0.008, "min_words": 24, "score_min": 0.16, "bonus": 1.25},
            {"name": "L2_mid",    "where_mode": "grupo",   "topk": 200,
                "min_density": 0.006, "min_words": 20, "score_min": 0.14, "bonus": 1.15},
            {"name": "L3_loose",  "where_mode": "grupo",   "topk": 240,
                "min_density": 0.004, "min_words": 16, "score_min": 0.12, "bonus": 1.05},
            {"name": "L4_any",    "where_mode": "none",    "topk": 260,
                "min_density": 0.000, "min_words": 12, "score_min": 0.10, "bonus": 1.00},
            ]

        # MMR v2 com penalidades
        self.enable_mmr_v2 = True
        self.mmr_page_penalty = 0.08      
        self.mmr_type_penalty = 0.05      

        # compressão de evidências antes do LLM
        self.enable_evidence_compression = True
        self.compression_target_chars = 4200
        self.compression_per_source_chars = 420
        self.compression_keep_numbers = True

        # gates
        self.gate_min_unique_pages = 5
        self.gate_min_unique_sources = 7
        self.gate_min_numbers = 8
        self.gate_min_units = 4

        # prompt/citações
        self.min_citations_required = 9


        self._debug_log_chroma_overview()
        _log_ok(f"RAG GOLD inicializado em {_fmt_secs(time.time() - t0)}")

    # ==========================================================
    # DEBUG
    # ==========================================================

    def _debug_check_dotenv_loaded(self) -> bool:
        try:
            in_env = "CHROMA_DB_PATH" in os.environ
            if not in_env:
                _log_warn(
                    "[DEBUG] CHROMA_DB_PATH não está no os.environ. .env pode não estar carregando aqui.")
            else:
                _log_info(
                    "[DEBUG] CHROMA_DB_PATH encontrado no os.environ (ok).")
            return in_env
        except Exception:
            return False

    def _debug_log_chroma_overview(self):
        if not self.chroma_client:
            return
        try:
            cols = self.chroma_client.list_collections()
            names = [c.name for c in cols] if cols else []
            _log_info(
                f"[DEBUG] Coleções disponíveis ({len(names)}): {names[:50]}")
            if self.collection_name not in names:
                _log_warn(f"[DEBUG] Coleção alvo '{self.collection_name}' NÃO está na lista! "
                          f"Pode indicar path errado ou grupo errado. "
                          f"get_or_create_collection pode ter criado uma vazia.")
        except Exception as e:
            _log_warn(f"[DEBUG] Falha ao listar coleções: {e}")

    # ==========================================================
    # GRUPO / COLEÇÃO
    # ==========================================================
    def _collection_name_for_group(self, grupo: str) -> str:
        grupo = (grupo or "geral").strip() or "geral"
        return "manuais_maquinas" if grupo in ("geral", "", None) else f"manuais_maquinas_{grupo}"

    def set_grupo(self, novo_grupo: str) -> bool:
        try:
            novo_grupo = (novo_grupo or "geral").strip() or "geral"

            if self.grupo == novo_grupo and self.collection is not None:
                return True

            if not self.chroma_client:
                _log_warn("[set_grupo] chroma_client não inicializado.")
                self.grupo = novo_grupo
                return False

            self.grupo = novo_grupo
            self.collection_name = self._collection_name_for_group(novo_grupo)

            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "grupo": self.grupo},
            )

            self.image_output_dir = _ensure_dir(
                os.path.join(MANUALS_IMAGES_DIR, self.grupo))
            self.pages_output_dir = _ensure_dir(
                os.path.join(MANUALS_PAGES_DIR, self.grupo))

            _log_ok(
                f"[set_grupo] Grupo alterado: {self.grupo} | coleção={self.collection_name}")
            return True
        except Exception as e:
            _log_warn(f"[set_grupo] Falha ao trocar grupo: {e}")
            return False

    # ==========================================================
    # UTILIDADES
    # ==========================================================
    def _is_refusal(self, text: str) -> bool:
        if not text:
            return True
        t = text.strip().lower()
        padroes = [
            "i'm sorry, i can't assist with that",
            "i cannot assist with that",
            "não posso ajudar com isso",
            "não posso fornecer assistência",
            "nao posso ajudar com isso",
        ]
        return any(p in t for p in padroes)
    
    def _build_where_strict(self, maquina_id: Optional[Any] = None) -> Optional[dict]:
        if maquina_id is None:
            return {"grupo": self.grupo}
        return {"maquina_id": str(maquina_id), "grupo": self.grupo}


    def _post_filter_group(self, docs: List[str], metas: List[dict]) -> Tuple[List[str], List[dict]]:
        if not docs:
            return docs, metas
        out_d, out_m = [], []
        for d, m in zip(docs, metas):
            if not isinstance(m, dict):
                continue
            if m.get("grupo") == self.grupo:
                out_d.append(d)
                out_m.append(m)
        return out_d, out_m

    def _area_boost(self, text: str) -> float:
        low = (text or "").lower()
        boost = 0.0
        for k, w in self.area_keywords.items():
            if k in low:
                boost += w
        return boost

    def _tech_density(self, text: str) -> float:
        low = (text or "").lower()
        if not low.strip():
            return 0.0
        hits = 0
        for k in self.area_keywords.keys():
            if k in low:
                hits += 1
        codes = re.findall(r"\b[A-Z]{2,}[0-9A-Z\-\._/]{1,}\b", text or "")
        units = re.findall(
            r"\b\d+(?:[\.,]\d+)?\s?(?:v|a|ma|hz|ohm|k\s?ohm|%|bar|mm|ºc|c)\b", low)
        total = max(1, len((text or "").split()))
        return (hits + 0.5 * len(codes) + 0.3 * len(units)) / total

    def _chunk_is_bad(self, text: str) -> bool:
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

    # ==========================================================
    # UTILIDADE DO TRECHO (NOVO)
    # ==========================================================
    def _usefulness_score(self, text: str) -> float:
        """
        Retorna um score de utilidade baseado em presença de:
        - números/unidades
        - alarmes/códigos
        - passos/procedimentos
        - limites/tolerâncias
        """
        if not text:
            return 0.0
        low = text.lower()
        score = 0.0

        # pesos calibrados
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

        # bônus por densidade técnica
        score += 2.8 * self._tech_density(text)
        score += 0.04 * self._area_boost(text)
        return score

    def _extract_best_span(self, text: str, query: str, max_len: int = 650) -> str:
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
        # encontra melhor janela (com mais termos)
        best_i = 0
        best_hits = -1
        window = max_len

        for start in range(0, max(1, len(txt) - window), max(120, window // 4)):
            end = start + window
            chunk = low[start:end]
            hits = sum(1 for t in q_terms if t in chunk)
            if hits > best_hits:
                best_hits = hits
                best_i = start

        span = txt[best_i:best_i + window].strip()
        # limpa bordas
        span = span.strip(" .,:;-\n\t")
        if best_i > 0:
            span = "..." + span
        if (best_i + window) < len(txt):
            span = span + "..."
        return span

    # ==========================================================
    # CHROMA HELPERS (SAFE QUERY)
    # ==========================================================
    def _safe_query_chroma(
        self,
        *,
        query_emb: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        n_results: int = 30,
        where: Optional[dict] = None
    ) -> Dict[str, Any]:
        if not self.collection:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

        def _empty():
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

        try:
            if query_emb is not None:
                if where:
                    return self.collection.query(query_embeddings=[query_emb], n_results=n_results, where=where)
                return self.collection.query(query_embeddings=[query_emb], n_results=n_results)

            if query_text is not None:
                if where:
                    return self.collection.query(query_texts=[query_text], n_results=n_results, where=where)
                return self.collection.query(query_texts=[query_text], n_results=n_results)

            return _empty()
        except Exception as e:
            _log_warn(f"[Chroma] query falhou com where={where}. Erro: {e}")
            try:
                if query_emb is not None:
                    return self.collection.query(query_embeddings=[query_emb], n_results=n_results)
                if query_text is not None:
                    return self.collection.query(query_texts=[query_text], n_results=n_results)
                return _empty()
            except Exception as e2:
                _log_warn(f"[Chroma] query fallback sem where falhou: {e2}")
                return _empty()

    def _get_collection_count(self) -> int:
        if not self.collection:
            return 0

        try:
            c = self.collection.count()
            if isinstance(c, int):
                return c
        except Exception as e:
            _log_warn(f"[count] collection.count() falhou: {e}")

        try:
            data = self.collection.get()
            ids = data.get("ids", None)
            if ids is None:
                return 0
            if isinstance(ids, list):
                if len(ids) == 0:
                    return 0
                if isinstance(ids[0], list):
                    return len(ids[0])
                return len(ids)
            return int(len(ids))
        except Exception as e:
            _log_warn(f"[count] Falha ao contar via get(): {e}")
            return 0

    def _collection_has_embeddings(self) -> bool:
        if not self.collection:
            return False
        try:
            got = self.collection.get(include=["embeddings"], limit=1)
            embs = got.get("embeddings", None)
            if embs is None:
                return False
            try:
                return len(embs) > 0 and len(embs[0]) > 0
            except Exception:
                return False
        except Exception:
            return False

    # ==========================================================
    # KEYWORD SUBSTRING SCAN (SAFE)
    # ==========================================================
    def _get_for_scan(self, *, where: Optional[dict], max_docs: int, debug: bool = False) -> Tuple[List[str], List[dict]]:
        if not self.collection:
            return [], []

        tries = [
            ("where+limit", lambda: self.collection.get(where=where,
             include=["documents", "metadatas"], limit=max_docs) if where else None),
            ("limit", lambda: self.collection.get(
                include=["documents", "metadatas"], limit=max_docs)),
            ("all", lambda: self.collection.get(
                include=["documents", "metadatas"])),
        ]

        for tag, fn in tries:
            try:
                if tag == "where+limit" and where is None:
                    continue
                got = fn()
                if not got:
                    continue
                docs = got.get("documents", None) or []
                metas = got.get("metadatas", None) or []
                if isinstance(docs, list) and isinstance(metas, list) and len(docs) > 0:
                    if debug:
                        _log_info(f"[scan_get] ok tag={tag} docs={len(docs)}")
                    return docs[:max_docs], metas[:max_docs]
                if debug:
                    _log_info(f"[scan_get] empty tag={tag}")
            except Exception as e:
                if debug:
                    _log_warn(f"[scan_get] fail tag={tag}: {e}")
                continue

        return [], []

    def _keyword_substring_scan(
        self,
        query: str,
        *,
        where: Optional[dict] = None,
        max_docs: int = 1500,
        topk: int = 25,
        debug: bool = False,
        post_filter_group: bool = False,
    ) -> List[Tuple[float, str, dict]]:
        if not self.collection:
            return []

        q = (query or "").strip().lower()
        if not q:
            return []

        docs, metas = self._get_for_scan(
            where=where, max_docs=max_docs, debug=debug)
        if not docs:
            return []

        if post_filter_group:
            docs, metas = self._post_filter_group(docs, metas)

        q_terms = [t for t in re.findall(
            r"[a-zA-Z0-9_/%\.-]{2,}", q) if len(t) >= 3]
        q_set = set(q_terms)

        q_codes = re.findall(r"\b[A-Z]{1,5}[-_ ]?\d{2,6}\b", query or "")
        q_codes = [c.strip().lower().replace(" ", "").replace("_", "-")
                   for c in q_codes]

        ranked: List[Tuple[float, str, dict]] = []

        for d, m in zip(docs, metas):
            if not isinstance(d, str):
                continue
            txt = (d or "").strip()
            low = txt.lower()
            if not low:
                continue

            hits = 0.0
            if len(q) >= 4 and q in low:
                hits += 6.5

            token_hits = 0
            for t in q_set:
                if t in low:
                    token_hits += 1
            hits += float(token_hits)

            low_norm = low.replace(" ", "").replace("_", "-")
            for c in q_codes:
                if c and c in low_norm:
                    hits += 5.0

            if hits < self.substring_min_hits:
                continue

            boost = self._area_boost(
                txt) + float((m or {}).get("area_boost", 0.0) or 0.0)
            dens = self._tech_density(txt)
            useful = self._usefulness_score(txt)

            score = hits + 0.06 * boost + 2.4 * dens + 1.7 * useful
            if len(txt.split()) < 10:
                continue

            ranked.append((score, txt, m or {}))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[:topk]

    # ==========================================================
    # OCR OFFLINE (OCRmyPDF)
    # ==========================================================
    def run_ocr_if_needed(
        self,
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

        try:
            doc = fitz.open(pdf_path)
            n_pages = len(doc)
            doc.close()
        except Exception:
            n_pages = None

        if n_pages and n_pages > max_pages:
            return False, f"PDF tem {n_pages} páginas (> {max_pages}). Abortando OCR por segurança."

        if shutil.which("ocrmypdf") is None:
            return False, "ocrmypdf não encontrado no PATH. Instale: pip install ocrmypdf (e dependências do sistema)."

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

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                return False, f"OCRmyPDF falhou (code={proc.returncode}): {stderr[:800]}"
            return True, f"OCR concluído: {output_pdf_path}"
        except Exception as e:
            return False, f"Falha ao executar OCRmyPDF: {e}"

    # ==========================================================
    # RENDER PAGE -> PNG
    # ==========================================================
    def render_page_to_png(self, pdf_path: str, page_index: int, maquina_id: Any, dpi: int = RENDER_DPI_DEFAULT) -> str:
        os.makedirs(self.pages_output_dir, exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)

        doc = fitz.open(pdf_path)
        if page_index < 0 or page_index >= len(doc):
            doc.close()
            raise IndexError(f"page_index fora do range: {page_index}")

        page = doc[page_index]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        filename = f"{maquina_id}_page_{page_index+1}.png"
        out_png_pages = os.path.join(self.pages_output_dir, filename)
        pix.save(out_png_pages)

        out_png_images = os.path.join(self.image_output_dir, filename)
        try:
            shutil.copy2(out_png_pages, out_png_images)
        except Exception as e:
            _log_warn(f"Falha ao copiar para manuals_images: {e}")

        doc.close()
        return out_png_pages

    def is_rendered_page_useful(
        self,
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

    def phash_dedupe_pngs(self, png_paths: List[str], max_distance: int = PHASH_MAX_DISTANCE_DEFAULT) -> Tuple[List[str], List[str]]:
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

    def _renderizar_paginas_pdf(
        self,
        pdf_path: str,
        maquina_id: Any,
        *,
        page_indices: Optional[List[int]] = None,
        dpi: int = RENDER_DPI_DEFAULT,
        filtrar_paginas_vazias: bool = True,
        dedupe_phash: bool = False,
        phash_max_distance: int = PHASH_MAX_DISTANCE_DEFAULT,
    ) -> List[Dict[str, Any]]:

        imagens_info: List[Dict[str, Any]] = []

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        if page_indices is None:
            page_indices = list(range(total_pages))

        page_indices = [i for i in page_indices if 0 <= i < total_pages]
        if not page_indices:
            return []

        _log_step("Renderizando páginas inteiras do PDF (PNG)")
        _log_info(
            f"Páginas totais: {total_pages} | selecionadas: {len(page_indices)} | dpi={dpi}")

        rendered_paths: List[str] = []
        meta_by_path: Dict[str, Dict[str, Any]] = {}

        for idx0 in page_indices:
            try:
                png_path = self.render_page_to_png(
                    pdf_path=pdf_path, page_index=idx0, maquina_id=maquina_id, dpi=dpi)

                if filtrar_paginas_vazias:
                    ok, motivo = self.is_rendered_page_useful(png_path)
                    if not ok:
                        _log_warn(
                            f"Página {idx0+1} marcada como 'não útil': {motivo} | {png_path} (vou manter mesmo assim)")
                    else:
                        _log_info(f"Página {idx0+1} OK: {motivo}")

                try:
                    with open(png_path, "rb") as f:
                        b = f.read()
                    md5 = hashlib.md5(b).hexdigest()
                except Exception:
                    md5 = ""

                rendered_paths.append(png_path)
                meta_by_path[png_path] = {
                    "path": png_path, "page": idx0 + 1, "indice": 0, "hash": md5}

            except Exception as e:
                _log_warn(f"Falha ao renderizar página {idx0+1}: {e}")

        if not rendered_paths:
            _log_warn("Nenhuma página renderizada válida.")
            return []

        if dedupe_phash and len(rendered_paths) > 1:
            try:
                kept, dropped = self.phash_dedupe_pngs(
                    rendered_paths, max_distance=phash_max_distance)
                for p in dropped:
                    _log_info(f"pHash dedupe: removendo duplicada {p}")
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                    meta_by_path.pop(p, None)
                rendered_paths = kept
            except Exception as e:
                _log_warn(f"Falha no pHash dedupe (ignorando): {e}")

        for p in rendered_paths:
            if p in meta_by_path:
                imagens_info.append(meta_by_path[p])

        _log_ok(f"Páginas renderizadas válidas: {len(imagens_info)}")
        return imagens_info

    # ==========================================================
    # POSTGRES (lazy) + schema cache
    # ==========================================================
    def _pg_conn(self):
        if hasattr(self, "_pg_conn_cache"):
            return self._pg_conn_cache

        if not all([DB_HOST, DB_NAME, DB_USER, DB_PASS]):
            _log_info(
                "Variáveis de banco não configuradas; não vou salvar manual_imagens no Postgres.")
            self._pg_conn_cache = None
            return None

        try:
            conn = psycopg2.connect(
                host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT
            )
            conn.autocommit = True
            self._pg_conn_cache = conn
            _log_ok("Conectado ao PostgreSQL (rag_pdf_processor).")
            return conn
        except Exception as e:
            _log_warn(f"Não foi possível conectar ao Postgres: {e}")
            self._pg_conn_cache = None
            return None

    def _pg_schema(self, conn) -> Dict[str, Any]:
        if self._pg_schema_cache is not None:
            return self._pg_schema_cache

        schema = {"manual_imagens_cols": set(), "has_pgvector": False}

        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT column_name, udt_name
                FROM information_schema.columns
                WHERE table_name='manual_imagens';
            """)
            rows = cur.fetchall()
            cur.close()

            cols = set()
            for col, udt in rows:
                cols.add(col)
                if udt == "vector":
                    schema["has_pgvector"] = True

            schema["manual_imagens_cols"] = cols
        except Exception as e:
            _log_warn(f"[PG] Falha ao detectar schema de manual_imagens: {e}")

        self._pg_schema_cache = schema
        return schema

    def _upsert_manual_pg(self, conn, manual_id: int, nome_maquina: Optional[str], pdf_path: Optional[str]):
        if manual_id is None:
            return

        cur = None
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT id FROM manuais WHERE id = %s;", (manual_id,))
            row = cur.fetchone()
            if row:
                return

            nome_maquina_final = nome_maquina or f"Máquina {manual_id}"
            nome_manual_final = nome_maquina or f"Manual ID {manual_id}"
            caminho_pdf_final = pdf_path or ""

            cur.execute("""
                INSERT INTO manuais (
                    id, maquina_id, nome_maquina, nome_manual, caminho_pdf, status
                )
                VALUES (%s, %s, %s, %s, %s, 'processado')
                ON CONFLICT (id) DO NOTHING;
            """, (manual_id, manual_id, nome_maquina_final, nome_manual_final, caminho_pdf_final))

            _log_ok(
                f"[PG] Registro básico em 'manuais' criado para id={manual_id}")
        except Exception as e:
            _log_warn(
                f"[PG] _upsert_manual_pg falhou para manual_id={manual_id}: {e}")
        finally:
            if cur:
                cur.close()

    # ==========================================================
    # DESCRIÇÃO CURTA (retrieval card)
    # ==========================================================
    def _gerar_descricao_curta(self, descricao_completa: str, nome_maquina: str, maquina_id: Any, page: int) -> str:
        desc = (descricao_completa or "").strip()
        desc_low = desc.lower()

        tipo = "Página/Imagem técnica"
        if "alarme" in desc_low or "fault" in desc_low or "warning" in desc_low:
            tipo = "Alarmes / Diagnóstico"
        elif "procedimento" in desc_low or "passo" in desc_low:
            tipo = "Procedimento / Passo a passo"
        elif "tabela" in desc_low:
            tipo = "Tabela técnica / Parâmetros"
        elif "diagrama" in desc_low:
            tipo = "Diagrama técnico"

        kws = re.findall(
            r"\b[A-Z]{2,}[0-9A-Z\-\._/]{1,}\b", descricao_completa or "")
        kws = kws[:14]
        kws_txt = ", ".join(kws)

        vals = re.findall(
            r"\b\d+(?:[\.,]\d+)?\s?(?:v|a|ma|hz|ohm|k\s?ohm|%|bar|mm|ºc|c)\b", desc_low)
        vals = list(dict.fromkeys(vals))[:12]
        vals_txt = ", ".join(vals)

        essencia = re.sub(r"\s+", " ", desc).strip()
        if len(essencia) > 260:
            essencia = essencia[:260].rstrip() + "..."

        card = (
            f"Tipo: {tipo}\n"
            f"Máquina: {nome_maquina} (ID: {maquina_id}) | Grupo: {self.grupo} | Página: {page}\n"
            f"Essência: {essencia}\n"
        )

        if kws_txt:
            card += f"Tags/Códigos: {kws_txt}\n"
        if vals_txt:
            card += f"Valores/Unidades: {vals_txt}\n"

        return card.strip()[:900]

    # ==========================================================
    # SALVAR EMBEDDING CURTO NO PG
    # ==========================================================
    def _salvar_embedding_curto_pg(self, conn, row_id: int, embedding: List[float]):
        if not conn or not row_id:
            return

        schema = self._pg_schema(conn)
        cols = schema.get("manual_imagens_cols", set())
        has_vector_col = "embedding_curto" in cols
        has_json_col = "embedding_curto_json" in cols

        cur = None
        try:
            cur = conn.cursor()

            if has_vector_col:
                vec_str = "[" + ",".join([f"{x:.8f}" for x in embedding]) + "]"
                cur.execute(
                    "UPDATE manual_imagens SET embedding_curto = %s WHERE id = %s;", (vec_str, row_id))
                return

            if has_json_col:
                cur.execute(
                    "UPDATE manual_imagens SET embedding_curto_json = %s WHERE id = %s;",
                    (json.dumps(embedding), row_id),
                )
                return

        except Exception as e:
            _log_warn(
                f"[PG] Falha ao salvar embedding_curto para id={row_id}: {e}")
        finally:
            if cur:
                cur.close()

    # ==========================================================
    # SALVAR IMAGEM + DESCRIÇÕES NO PG
    # ==========================================================
    def _salvar_imagem_pg(
        self,
        conn,
        manual_id: int,
        pagina: int,
        indice: int,
        caminho: str,
        hash_md5: str,
        descricao_completa: str,
        descricao_curta: str,
        modelo_vision: str,
    ) -> Optional[int]:
        cur = None
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT id
                FROM manual_imagens
                WHERE manual_id = %s AND pagina = %s AND indice_imagem = %s;
            """, (manual_id, pagina, indice))
            row = cur.fetchone()

            if row:
                cur.execute("""
                    UPDATE manual_imagens
                    SET arquivo_imagem=%s, hash_md5=%s,
                        descricao_completa=%s, descricao_curta=%s,
                        modelo_vision=%s,
                        foi_reprocessada=TRUE,
                        atualizado_em=NOW()
                    WHERE id=%s
                    RETURNING id;
                """, (caminho, hash_md5, descricao_completa, descricao_curta, modelo_vision, row["id"]))
                rid = cur.fetchone()["id"]
                _log_ok(f"[PG] manual_imagens atualizado (id={rid})")
                return rid
            else:
                cur.execute("""
                    INSERT INTO manual_imagens (
                        manual_id, pagina, indice_imagem,
                        arquivo_imagem, hash_md5,
                        descricao_completa, descricao_curta,
                        modelo_vision, foi_reprocessada
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,TRUE)
                    RETURNING id;
                """, (manual_id, pagina, indice, caminho, hash_md5, descricao_completa, descricao_curta, modelo_vision))
                novo_id = cur.fetchone()["id"]
                _log_ok(f"[PG] manual_imagens criado (id={novo_id})")
                return novo_id
        except Exception as e:
            _log_warn(
                f"[PG] Falha ao salvar manual_imagens (manual={manual_id}, pag={pagina}, idx={indice}): {e}")
            return None
        finally:
            if cur:
                cur.close()

    # ==========================================================
    # DETECTAR IDIOMA (leve)
    # ==========================================================
    def _detectar_idioma(self, texto: str) -> str:
        if not self.traduzir_automaticamente or not self.openai_client:
            return "desconhecido"
        amostra = (texto[:500] or "").strip()
        if not amostra:
            return "vazio"
        try:
            _log_step("Detectando idioma do texto")
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL_MINI,
                messages=[
                    {"role": "system",
                        "content": "Você é um detector de idiomas. Responda APENAS com o código do idioma (pt, en, es, etc)."},
                    {"role": "user", "content": f"Qual o idioma deste texto? Responda apenas com o código:\n\n{amostra}"},
                ],
                temperature=0,
                max_tokens=10,
            )
            idioma = (
                response.choices[0].message.content or "").strip().lower()
            _log_ok(f"Idioma detectado: {idioma}")
            return idioma
        except Exception as e:
            _log_warn(f"Erro ao detectar idioma: {e}")
            return "erro"

    # ==========================================================
    # PROCESSAR PDF HÍBRIDO (OCR + PROCESSAR)
    # ==========================================================
    def processar_pdf_hibrido(
        self,
        pdf_path: str,
        maquina_id: Any,
        nome_maquina: str,
        *,
        scanned_ratio_threshold: float = OCR_SCANNED_RATIO_THRESHOLD_DEFAULT,
        ocr_language: str = OCR_LANGUAGE_DEFAULT,
        ocr_jobs: int = OCR_JOBS_DEFAULT,
        force_ocr: bool = False,
        keep_ocr_pdf: bool = True,
        processar_imagens: bool = False,
        imagens_paginas: Optional[str] = None,
        imagens_modo_pagina_inteira: bool = True,
        render_dpi: int = RENDER_DPI_DEFAULT,
        filtrar_paginas_vazias: bool = True,
        dedupe_paginas_phash: bool = False,
        phash_max_distance: int = PHASH_MAX_DISTANCE_DEFAULT,
        max_words_per_page: int = TEXT_PAGE_MAX_WORDS_DEFAULT,
        subchunk_size: int = TEXT_SUBCHUNK_SIZE_DEFAULT,
        subchunk_overlap: int = TEXT_SUBCHUNK_OVERLAP_DEFAULT,
    ) -> bool:

        if not self.collection or not self.embedder:
            _log_err(
                "Sistema RAG não está completamente inicializado (Chroma/Embedder).")
            return False

        if not os.path.exists(pdf_path):
            _log_err(f"PDF não encontrado: {pdf_path}")
            return False

        _log_step("Preflight do PDF (texto vs imagem)")
        try:
            pf = preflight_pdf_pages(pdf_path)
            summ = preflight_summary(pf)
            _log_info(json.dumps(summ, ensure_ascii=False))
        except Exception as e:
            _log_warn(f"Falha no preflight (vou seguir sem OCR): {e}")
            summ = {"likely_scanned_ratio": 0.0}

        use_pdf = pdf_path
        temp_ocr_pdf = None

        if float(summ.get("likely_scanned_ratio", 0.0)) >= scanned_ratio_threshold:
            _log_step("PDF parece escaneado: tentando OCR offline (OCRmyPDF)")
            ocr_out = os.path.splitext(pdf_path)[0] + "_OCR.pdf"
            ok, msg = self.run_ocr_if_needed(
                pdf_path=pdf_path,
                output_pdf_path=ocr_out,
                language=ocr_language,
                jobs=ocr_jobs,
                force_ocr=force_ocr,
            )
            if ok:
                _log_ok(msg)
                use_pdf = ocr_out
                temp_ocr_pdf = ocr_out
            else:
                _log_warn(msg)
                _log_warn(
                    "Vou seguir com o PDF original (texto pode ser pobre).")

        ok = self.processar_pdf(
            use_pdf,
            maquina_id,
            nome_maquina,
            processar_imagens=processar_imagens,
            imagens_paginas=imagens_paginas,
            imagens_modo_pagina_inteira=imagens_modo_pagina_inteira,
            render_dpi=render_dpi,
            filtrar_paginas_vazias=filtrar_paginas_vazias,
            dedupe_paginas_phash=dedupe_paginas_phash,
            phash_max_distance=phash_max_distance,
            max_words_per_page=max_words_per_page,
            subchunk_size=subchunk_size,
            subchunk_overlap=subchunk_overlap,
        )

        if temp_ocr_pdf and (not keep_ocr_pdf):
            try:
                os.remove(temp_ocr_pdf)
            except Exception:
                pass

        return ok

    # ==========================================================
    # PROCESSAR PDF (TEXTO + IMAGENS)
    # ==========================================================
    def processar_pdf(
        self,
        pdf_path: str,
        maquina_id: Any,
        nome_maquina: str,
        processar_imagens: bool = True,
        *,
        imagens_paginas: Optional[str] = None,
        imagens_modo_pagina_inteira: bool = True,
        render_dpi: int = RENDER_DPI_DEFAULT,
        filtrar_paginas_vazias: bool = True,
        dedupe_paginas_phash: bool = False,
        phash_max_distance: int = PHASH_MAX_DISTANCE_DEFAULT,
        max_words_per_page: int = TEXT_PAGE_MAX_WORDS_DEFAULT,
        subchunk_size: int = TEXT_SUBCHUNK_SIZE_DEFAULT,
        subchunk_overlap: int = TEXT_SUBCHUNK_OVERLAP_DEFAULT,
    ) -> bool:

        if not self.collection or not self.embedder:
            _log_err(
                "Sistema RAG não está completamente inicializado (Chroma/Embedder).")
            return False

        if not os.path.exists(pdf_path):
            _log_err(f"PDF não encontrado: {pdf_path}")
            return False

        t0 = time.time()
        _log_info(f"\n📄 Processando PDF: {pdf_path} (grupo: {self.grupo})")

        # -------- TEXTO (POR PÁGINA) --------
        _log_step("Extraindo texto do PDF (por página)")
        t_text = time.time()

        paginas_texto = extrair_texto_pdf_por_pagina(pdf_path)
        if not paginas_texto:
            _log_err(f"Não foi possível extrair texto do PDF {pdf_path}")
            return False

        amostra_concat = " ".join(
            [limpar_texto(p or "") for p in paginas_texto[:10] if (p or "").strip()])[:4000]
        if not amostra_concat.strip():
            _log_warn(
                "Amostra por páginas vazia; tentando extração concatenada (legacy).")
            texto_leg = limpar_texto(extrair_texto_pdf(pdf_path))
            if not texto_leg.strip():
                _log_err(
                    f"Não foi possível extrair texto do PDF {pdf_path} (nem por página nem legacy).")
                return False
            amostra_concat = texto_leg[:4000]

        idioma = self._detectar_idioma(amostra_concat)

        _log_ok(
            f"Texto por página extraído em {_fmt_secs(time.time() - t_text)} | páginas={len(paginas_texto)} | idioma={idioma}")

        _log_step("Chunking do texto (por página)")
        t_chunk = time.time()

        chunks_info = paginas_para_chunks_texto(
            paginas_texto,
            max_words_per_page=max_words_per_page,
            subchunk_size=subchunk_size,
            subchunk_overlap=subchunk_overlap,
        )

        _log_ok(
            f"Gerados {len(chunks_info)} chunks (por página) em {_fmt_secs(time.time() - t_chunk)}")
        if not chunks_info:
            _log_err("Nenhum chunk de texto gerado (todas as páginas sem texto?).")
            return False

        # -------- SALVAR TEXTO NO CHROMA --------
        _log_step(
            "Salvando chunks de TEXTO no ChromaDB (filtros + densidade + boost)")
        t_save_text = time.time()

        total = len(chunks_info)
        adicionados, pulados, ruins = 0, 0, 0

        for i, info in enumerate(chunks_info):
            page = int(info["page"])
            subchunk = int(info["subchunk"])
            chunk = limpar_texto(info["text"] or "")

            if self._chunk_is_bad(chunk):
                ruins += 1
                continue

            dens = self._tech_density(chunk)
            if dens < self.min_tech_density:
                if len(chunk.split()) < 60:
                    ruins += 1
                    continue

            chunk_for_embedding = (
                f"Manual: {nome_maquina}\n"
                f"Grupo: {self.grupo}\n"
                f"Prioridade: forno, dicu, fusão, prodapt md, disa, moldagem, fundição\n"
                f"Página: {page}\n\n{chunk}"
            )

            chunk_id = f"{self.grupo}_{maquina_id}_texto_p{page}_s{subchunk}"

            try:
                existing = self.collection.get(ids=[chunk_id], include=["ids"])
                if existing and existing.get("ids"):
                    pulados += 1
                    continue
            except Exception:
                pass

            emb = self.embedder.encode([chunk_for_embedding]).tolist()

            self.collection.add(
                documents=[chunk],
                embeddings=emb,
                metadatas=[{
                    "tipo": "texto",
                    "grupo": self.grupo,
                    "maquina_id": str(maquina_id),
                    "nome_maquina": nome_maquina,
                    "page": page,
                    "subchunk": subchunk,
                    "pdf_path": pdf_path,
                    "idioma_original": idioma or "pt",
                    "foi_traduzido": "False",
                    "chunking": "pagina" if subchunk == 0 else "pagina_subchunk",
                }],
                ids=[chunk_id],
            )

            adicionados += 1
            if (i + 1) % 45 == 0 or (i + 1) == total:
                _log_info(
                    f"Chunks: {i+1}/{total} | adicionados={adicionados} | pulados={pulados} | ruins={ruins}")

        _log_ok(
            f"Texto no Chroma: adicionados={adicionados} | pulados={pulados} | ruins={ruins} | tempo={_fmt_secs(time.time() - t_save_text)}")

        # -------- IMAGENS (opcional) --------
        if processar_imagens:
            page_indices = None
            try:
                dtmp = fitz.open(pdf_path)
                total_pages = len(dtmp)
                dtmp.close()

                if imagens_paginas and str(imagens_paginas).strip():
                    page_indices = parse_page_selection(
                        imagens_paginas, total_pages)
                    _log_info(
                        f"Vai processar imagens apenas das páginas: {imagens_paginas}")
            except Exception as e:
                _log_warn(
                    f"Falha ao preparar filtro de páginas '{imagens_paginas}'. Vou processar todas. Erro: {e}")
                page_indices = None

            if imagens_modo_pagina_inteira:
                imagens_extraidas = self._renderizar_paginas_pdf(
                    pdf_path=pdf_path,
                    maquina_id=maquina_id,
                    page_indices=page_indices,
                    dpi=render_dpi,
                    filtrar_paginas_vazias=filtrar_paginas_vazias,
                    dedupe_phash=dedupe_paginas_phash,
                    phash_max_distance=phash_max_distance,
                )
            else:
                _log_warn(
                    "Modo legacy (objetos do PDF) desativado nesta versão por segurança/auditabilidade.")
                imagens_extraidas = []

            _log_step(
                "Processando imagens (descrição + embedding_curta + Chroma + PG)")
            _log_info(f"Imagens/Páginas a processar: {len(imagens_extraidas)}")

            t_imgs = time.time()
            ok_imgs, fail_imgs = 0, 0

            for n, img_info in enumerate(imagens_extraidas, 1):
                _log_info(
                    f"Imagem {n}/{len(imagens_extraidas)} | pag={img_info.get('page')} idx={img_info.get('indice')}")
                r = self._processar_imagem_para_chroma(
                    img_info, maquina_id, nome_maquina, pdf_path)
                if r:
                    ok_imgs += 1
                else:
                    fail_imgs += 1

            _log_ok(
                f"Imagens processadas: ok={ok_imgs} fail={fail_imgs} | tempo={_fmt_secs(time.time() - t_imgs)}")
        else:
            _log_info(
                "Processamento de imagens desativado (processar_imagens=False).")

        _log_ok(
            f"Processamento concluído para {nome_maquina} (grupo: {self.grupo}) em {_fmt_secs(time.time() - t0)}")
        return True

    # ==========================================================
    # PROCESSAR IMAGEM -> descrição -> embedding -> Chroma -> PG
    # ==========================================================
    def _processar_imagem_para_chroma(self, img_info: Dict[str, Any], maquina_id: Any, nome_maquina: str, pdf_path: str) -> bool:
        if not self.collection or not self.embedder:
            _log_err("Chroma ou embedder não inicializados ao processar imagem.")
            return False

        image_path = img_info.get("path")
        page = int(img_info.get("page", 0) or 0)
        indice_img = int(img_info.get("indice", 0) or 0)
        hash_md5 = img_info.get("hash") or ""

        if not image_path or not os.path.exists(image_path):
            _log_warn(f"Caminho de imagem inválido/não existe: {image_path}")
            return False

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            _log_warn(f"Não foi possível ler imagem {image_path}: {e}")
            return False

        if not hash_md5:
            try:
                hash_md5 = hashlib.md5(image_bytes).hexdigest()
            except Exception:
                hash_md5 = ""

        if not self.openai_client:
            descricao_completa = f"Imagem da pagina {page} (OpenAI nao disponivel)"
        else:
            try:
                descricao_completa = self._gerar_descricao_imagem(
                    image_bytes=image_bytes, nome_maquina=nome_maquina, maquina_id=str(
                        maquina_id)
                )
                if self._is_refusal(descricao_completa):
                    _log_warn(
                        "Modelo retornou recusa para a imagem. Não vou salvar no Chroma/PG.")
                    return False
            except Exception as e:
                _log_warn(
                    f"Falha ao gerar descrição técnica para imagem {image_path}: {e}")
                descricao_completa = f"Imagem da pagina {page} (falha ao gerar descricao: {e})"

        descricao_curta = self._gerar_descricao_curta(
            descricao_completa, nome_maquina, maquina_id, page)
        image_id = f"{self.grupo}_{maquina_id}_img_p{page}_i{indice_img}"

        try:
            emb_img = self.embedder.encode(
                [f"{descricao_curta}\n\nContexto: {nome_maquina}"]).tolist()
        except Exception as e:
            _log_warn(
                f"Falha ao gerar embedding da descrição curta de imagem: {e}")
            return False

        try:
            self.collection.add(
                documents=[descricao_curta],
                embeddings=emb_img,
                metadatas=[{
                    "tipo": "imagem",
                    "grupo": self.grupo,
                    "maquina_id": str(maquina_id),
                    "nome_maquina": nome_maquina,
                    "pdf_path": pdf_path,
                    "page": page,
                    "indice_imagem": indice_img,
                    "imagem_path": image_path,
                    "imagem_hash": hash_md5,
                    "imagem_fonte": "pagina_inteira",
                    "descricao_completa": descricao_completa,
                    "area_boost": self._area_boost(descricao_completa),
                }],
                ids=[image_id],
            )
            _log_ok(f"Imagem salva no Chroma (id={image_id})")
        except Exception as e:
            _log_warn(f"Falha ao adicionar imagem ao Chroma: {e}")
            return False

        conn = self._pg_conn()
        if not conn:
            return True

        try:
            manual_id_int = int(maquina_id)
        except Exception:
            _log_warn(
                f"maquina_id='{maquina_id}' não é inteiro. Não vou salvar manual_imagens no Postgres.")
            return True

        try:
            self._upsert_manual_pg(conn, manual_id_int, nome_maquina, pdf_path)
            row_id = self._salvar_imagem_pg(
                conn=conn,
                manual_id=manual_id_int,
                pagina=page,
                indice=indice_img,
                caminho=image_path,
                hash_md5=hash_md5 or "",
                descricao_completa=descricao_completa,
                descricao_curta=descricao_curta,
                modelo_vision=VISION_MODEL_DB,
            )
            if row_id:
                self._salvar_embedding_curto_pg(conn, row_id, emb_img[0])
        except Exception as e:
            _log_warn(f"Falha na sincronização da imagem para Postgres: {e}")

        return True

    # ==========================================================
    # GERAR DESCRIÇÃO DE IMAGEM (visão)
    # ==========================================================
    def _gerar_descricao_imagem(self, image_bytes: bytes, nome_maquina: str, maquina_id: str) -> str:
        if not self.openai_client:
            raise RuntimeError("OpenAI client nao inicializado.")

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        contexto_texto = (
            "Analise a imagem anexada do manual tecnico. "
            "Extraia procedimentos, parâmetros, alarmes, limites, sequência operacional e componentes. "
            "Priorize fornos, DICU, fusão, Prodapt MD, DISA e moldagem se houver."
        )
        if nome_maquina:
            contexto_texto += f" A imagem pertence à maquina: {nome_maquina}."
        if maquina_id:
            contexto_texto += f" ID interno da maquina: {maquina_id}."

        mensagens = [
            {"role": "system", "content": SYSTEM_PROMPT_IMAGENS},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": contexto_texto},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ]

        resp = self.openai_client.chat.completions.create(
            model=VISION_MODEL,
            messages=mensagens,
            temperature=0.2,
            max_tokens=650,
        )

        content = resp.choices[0].message.content
        descricao = content.strip() if isinstance(content, str) else (
            str(content).strip() if content else "")
        return descricao or "Falha ao gerar descricao tecnica da imagem."

    # ==========================================================
    # DIVERSIDADE (por page/tipo)
    # ==========================================================
    def _dedupe_and_diversify(self, docs: List[str], metas: List[dict], max_per_page: int = 1, max_total: int = 12):
        seen = {}
        out_docs, out_metas = [], []
        for doc, meta in zip(docs, metas):
            if not isinstance(meta, dict):
                continue
            page = meta.get("page")
            tipo = meta.get("tipo")
            key = (tipo, page)
            if seen.get(key, 0) >= max_per_page:
                continue
            seen[key] = seen.get(key, 0) + 1
            out_docs.append(doc)
            out_metas.append(meta)
            if len(out_docs) >= max_total:
                break
        return out_docs, out_metas

    # ==========================================================
    # KEYWORD RETRIEVAL LOCAL
    # ==========================================================
    def _keyword_retrieval_local(self, query: str, docs: List[str], metas: List[dict], topk: int = 18):
        q = (query or "").lower().strip()
        if not q or not docs:
            return []

        q_terms = [t for t in re.findall(
            r"[a-zA-Z0-9_/%\.-]{2,}", q) if len(t) > 2]
        q_set = set(q_terms)

        ranked = []
        for d, m in zip(docs, metas):
            txt = (d or "").lower()
            if not txt:
                continue

            hits = 0
            for term in q_set:
                if term in txt:
                    hits += 1

            codes_q = re.findall(r"\b[A-Z]{1,5}[-_ ]?\d{2,6}\b", query or "")
            for c in codes_q:
                if c.lower().replace(" ", "").replace("_", "-") in txt.replace(" ", "").replace("_", "-"):
                    hits += 4

            boost = self._area_boost(
                d) + float((m or {}).get("area_boost", 0.0) or 0.0)
            dens = self._tech_density(d)
            useful = self._usefulness_score(d)

            score = hits + 0.05 * boost + 2.1 * dens + 1.2 * useful

            if hits > 0:
                ranked.append((score, d, m))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[:topk]

    # ==========================================================
    # MMR
    # ==========================================================
    def _cosine(self, a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) + 1e-9
        nb = math.sqrt(sum(x * x for x in b)) + 1e-9
        return dot / (na * nb)

    def _mmr_select(self, query_emb: List[float], docs: List[str], metas: List[dict], doc_embs: List[List[float]], max_total: int = 20):
        if not docs:
            return [], [], []

        selected_idx = []
        remaining = list(range(len(docs)))

        if not remaining:
            return [], [], []
        selected_idx.append(remaining.pop(0))


        while remaining and len(selected_idx) < max_total:
            best = None
            best_score = -1e9

            for i in remaining:
                sim_q = self._cosine(query_emb, doc_embs[i])
                sim_sel = max(self._cosine(
                    doc_embs[i], doc_embs[j]) for j in selected_idx)
                mmr = self.mmr_lambda * sim_q - (1 - self.mmr_lambda) * sim_sel

                if mmr > best_score:
                    best_score = mmr
                    best = i

            selected_idx.append(best)
            remaining.remove(best)

        out_docs = [docs[i] for i in selected_idx]
        out_metas = [metas[i] for i in selected_idx]
        out_embs = [doc_embs[i] for i in selected_idx]
        return out_docs, out_metas, out_embs

    # ==========================================================
    # COVERAGE CHECK
    # ==========================================================
    def _coverage_score(self, docs: List[str]) -> Dict[str, int]:
        low = "\n".join(docs).lower()
        score = {
            "procedimento": int(any(k in low for k in ["procedimento", "passo", "step", "sequência", "sequencia"])),
            "alarme": int(any(k in low for k in ["alarme", "fault", "warning", "trip", "erro"])),
            "limite": int(any(k in low for k in ["limite", "range", "min", "max", "tolerância", "tolerancia"])),
            "parâmetro": int(any(k in low for k in ["parâmetro", "parametro", "setpoint", "sp", "pv", "ajuste"])),
        }
        return score

    # ==========================================================
    # SCORE COMPOSTO 
    # ==========================================================
    def _composite_score(self, doc: str, meta: dict, rerank_score: Optional[float], query: str, level_bonus: float = 1.0) -> float:
        if not doc:
            return -999.0
        meta = meta or {}
        low = doc.lower()

        rr = float(rerank_score) if (rerank_score is not None) else 0.0

        useful = self._usefulness_score(doc)
        dens = self._tech_density(doc)
        boost = self._area_boost(doc) + float(meta.get("area_boost", 0.0) or 0.0)

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

        # bônus por nível do ladder
        score = score * float(level_bonus or 1.0)
        return float(score)


    def _run_ladder_retrieval(self, descricao_problema: str, maquina_id: Optional[Any], debug: bool = False):
        """
        Retrieval regressivo:
        - tenta níveis rígidos primeiro
        - se não atingir evidência mínima, relaxa
        - aplica bônus por nível
        """
        where_maquina = self._build_where_strict(
            maquina_id) if maquina_id is not None else None
        where_grupo = {"grupo": self.grupo}

        query_enriched = (
            f"{descricao_problema}\n"
            f"Prioridade: forno, DICU, fusão, Prodapt MD, DISA, moldagem, fundição.\n"
            f"Busque: parâmetros, limites, alarmes, diagnóstico, procedimentos, checklist.\n"
            f"Considere intertravamentos, permissivas, sequência, sinais e valores."
        )
        query_emb = self.embedder.encode([query_enriched]).tolist()[0]

        has_embs = self._collection_has_embeddings()
        best_pack = None

        for level in self.retrieval_ladder_levels:
            name = level["name"]
            where_mode = level["where_mode"]
            topk = int(level["topk"])
            min_density = float(level["min_density"])
            min_words = int(level["min_words"])
            score_min = float(level["score_min"])
            bonus = float(level["bonus"])

            if where_mode == "maquina" and where_maquina:
                wf = where_maquina
                post_group = True
            elif where_mode == "grupo":
                wf = where_grupo
                post_group = False
            else:
                wf = None
                post_group = False

            docs, metas = [], []

            # embedding query
            if has_embs:
                res = self._safe_query_chroma(
                    query_emb=query_emb, n_results=topk, where=wf)
                docs = res["documents"][0] if res.get("documents") else []
                metas = res["metadatas"][0] if res.get("metadatas") else []
                if post_group and docs:
                    docs, metas = self._post_filter_group(docs, metas)

            # keyword query_texts
            if not docs:
                resk = self._safe_query_chroma(
                    query_text=descricao_problema, n_results=max(20, self.keyword_topk), where=wf)
                docs = resk["documents"][0] if resk.get("documents") else []
                metas = resk["metadatas"][0] if resk.get("metadatas") else []
                if post_group and docs:
                    docs, metas = self._post_filter_group(docs, metas)

            # substring scan (sempre complementa)
            substring_ranked = self._keyword_substring_scan(
                descricao_problema,
                where=wf,
                max_docs=self.substring_scan_max_docs,
                topk=self.substring_scan_max_return,
                debug=debug,
                post_filter_group=post_group,
            )

            if debug:
                _log_info(
                    f"[LADDER] {name} where={where_mode} docs={len(docs)} substr={len(substring_ranked)} bonus={bonus}")

            # aplica filtros
            filtered = []
            for d, m in zip(docs, metas):
                if not d:
                    continue
                txt = (d or "").strip()
                if len(txt.split()) < min_words:
                    continue
                if self._chunk_is_bad(txt):
                    continue
                dens = self._tech_density(txt)
                if dens < min_density:
                    continue
                useful = self._usefulness_score(txt)
                boost = self._area_boost(
                    txt) + float((m or {}).get("area_boost", 0.0) or 0.0)
                filtered.append((dens + 0.03 * boost + 0.9 * useful, txt, m))
            filtered.sort(key=lambda x: x[0], reverse=True)

            docs_f = [d for _, d, _ in filtered]
            metas_f = [m for _, _, m in filtered]

            # merge com substring
            merged_docs, merged_metas = [], []
            seen = set()

            def _key(d, m):
                return (m.get("tipo"), m.get("page"), m.get("subchunk"), (d[:110] if d else ""))

            for d, m in zip(docs_f, metas_f):
                k = _key(d, m or {})
                if k in seen:
                    continue
                seen.add(k)
                merged_docs.append(d)
                merged_metas.append(m or {})

            for _, d, m in substring_ranked:
                k = _key(d, m or {})
                if k in seen:
                    continue
                seen.add(k)
                merged_docs.append(d)
                merged_metas.append(m or {})

            if not merged_docs:
                continue

            # rerank
            rerank_scores = None
            if self.reranker:
                try:
                    pairs = [(descricao_problema, d) for d in merged_docs]
                    rerank_scores = self.reranker.predict(pairs).tolist()
                except Exception:
                    rerank_scores = None

            composite = []
            for i, (d, m) in enumerate(zip(merged_docs, merged_metas)):
                rr = rerank_scores[i] if (
                    rerank_scores and i < len(rerank_scores)) else None
                comp = self._composite_score(
                    d, m, rr, descricao_problema, level_bonus=bonus)
                composite.append((comp, d, m, rr))
            composite.sort(key=lambda x: x[0], reverse=True)

            # gate de score mínimo do nível
            composite = [x for x in composite if x[0] >= score_min]
            if len(composite) < self.min_evidence:
                continue

            best_pack = {
                "level": name,
                "bonus": bonus,
                "query_emb": query_emb,
                "items": composite[:max(self.mmr_max_total, self.final_k)],
            }
            break

        return best_pack



    # ==========================================================
    # BUSCAR NO MANUAL (melhorado)
    # ==========================================================

    def buscar_no_manual(self, descricao_problema: str, maquina_id: Optional[Any] = None, debug: bool = False) -> Dict[str, Any]:
        """
        GOLD PRO buscar_no_manual:
        - Usa retrieval ladder regressivo (L0..L4) se disponível
        - Aplica bônus por rigidez (level_bonus)
        - MMR v2 (se _mmr_select já estiver atualizado)
        - Fallback para pipeline antigo se ladder não existir ou falhar
        - Retorna spans recortados + metadados + scores + coverage + relatorio
        """

        # -------------------------
        # SANITY CHECKS
        # -------------------------
        if not self.collection or not self.embedder:
            return {"trechos": [], "metadados": [], "scores": [], "coverage": {}, "relatorio": {}}

        col_count = self._get_collection_count()
        if debug:
            _log_info(
                f"[DEBUG] coleção={self.collection_name} grupo={self.grupo} count(get)={col_count}")

        if col_count == 0:
            _log_warn(
                f"[RAG] Coleção vazia: {self.collection_name}. Manual provavelmente não indexado neste grupo.")
            return {"trechos": [], "metadados": [], "scores": [], "coverage": {}, "relatorio": {}}

        # -------------------------
        # QUERY EMBEDDING (base)
        # -------------------------
        query_enriched = (
            f"{descricao_problema}\n"
            f"Prioridade: forno, DICU, fusão, Prodapt MD, DISA, moldagem, fundição.\n"
            f"Busque: parâmetros, limites, alarmes, diagnóstico, procedimentos, checklist.\n"
            f"Considere intertravamentos, permissivas, sequência, sinais e valores."
        )
        query_emb = self.embedder.encode([query_enriched]).tolist()[0]

        has_embs = self._collection_has_embeddings()
        if debug:
            _log_info(f"[DEBUG] collection_has_embeddings={has_embs}")

        # =========================================================
        # 1) TENTAR LADDER (se disponível)
        # =========================================================
        used_try = None
        ladder_level = None
        ladder_bonus = 1.0
        merged_docs, merged_metas, ranked_comp = [], [], []

        pack = None
        if getattr(self, "enable_retrieval_ladder", False) and hasattr(self, "_run_ladder_retrieval"):
            try:
                pack = self._run_ladder_retrieval(
                    descricao_problema, maquina_id, debug=debug)
            except Exception as e:
                _log_warn(
                    f"[LADDER] Falha no retrieval ladder (vou cair no fallback): {e}")
                pack = None

        if pack:
            # pack = {"level": ..., "bonus": ..., "query_emb": ..., "items": [(comp,d,m,rr), ...]}
            ladder_level = pack.get("level")
            ladder_bonus = float(pack.get("bonus", 1.0) or 1.0)
            used_try = f"ladder:{ladder_level}"
            query_emb = pack.get("query_emb") or query_emb

            items = pack.get("items") or []
            merged_docs = [d for _, d, _, _ in items]
            merged_metas = [m for _, _, m, _ in items]
            ranked_comp = [c for c, _, _, _ in items]

            if debug:
                _log_info(
                    f"[LADDER] usando nível={ladder_level} bonus={ladder_bonus} candidates={len(merged_docs)}")

        # =========================================================
        # 2) FALLBACK: SE LADDER NÃO DEU NADA, USA SEU PIPELINE ANTIGO
        # =========================================================
        if not merged_docs:
            where_maquina = self._build_where_strict(maquina_id)
            where_grupo = {"grupo": self.grupo}

            tries_where = []
            if maquina_id is not None:
                tries_where.append(("maquina_id", where_maquina, True))
            tries_where.append(("grupo_only", where_grupo, False))
            tries_where.append(("none", None, False))

            docs, metas = [], []
            used_try = None

            # 1) Embedding query
            if has_embs:
                for tag, wf, post_group in tries_where:
                    res = self._safe_query_chroma(
                        query_emb=query_emb, n_results=self.retrieval_topk, where=wf)
                    d = res["documents"][0] if res.get("documents") else []
                    m = res["metadatas"][0] if res.get("metadatas") else []

                    if post_group and d:
                        d, m = self._post_filter_group(d, m)

                    if debug:
                        _log_info(
                            f"[DEBUG] embedding try={tag} where={wf} -> docs={len(d)}")

                    if d:
                        docs, metas = d, m
                        used_try = f"embedding:{tag}"
                        break

            # 2) keyword query_texts
            if not docs:
                for tag, wf, post_group in tries_where:
                    resk = self._safe_query_chroma(
                        query_text=descricao_problema, n_results=max(self.keyword_topk, 18), where=wf)
                    kd = resk["documents"][0] if resk.get("documents") else []
                    km = resk["metadatas"][0] if resk.get("metadatas") else []

                    if post_group and kd:
                        kd, km = self._post_filter_group(kd, km)

                    if debug:
                        _log_info(
                            f"[DEBUG] keyword(query_texts) try={tag} where={wf} -> docs={len(kd)}")

                    if kd:
                        docs, metas = kd, km
                        used_try = f"keyword_query:{tag}"
                        break

            # 3) substring scan fallback
            substring_ranked = []
            for tag, wf, post_group in tries_where:
                substring_ranked = self._keyword_substring_scan(
                    descricao_problema,
                    where=wf,
                    max_docs=self.substring_scan_max_docs,
                    topk=self.substring_scan_max_return,
                    debug=debug,
                    post_filter_group=post_group,
                )
                if debug:
                    _log_info(
                        f"[DEBUG] substring_scan try={tag} where={wf} -> hits={len(substring_ranked)}")
                if substring_ranked:
                    if used_try is None:
                        used_try = f"substring_scan:{tag}"
                    break

            if not docs and not substring_ranked:
                substring_ranked = self._keyword_substring_scan(
                    descricao_problema,
                    where=None,
                    max_docs=max(self.substring_scan_max_docs, 2500),
                    topk=self.substring_scan_max_return,
                    debug=debug,
                    post_filter_group=False,
                )
                if substring_ranked and used_try is None:
                    used_try = "substring_scan:total"

            if not docs and not substring_ranked:
                return {"trechos": [], "metadados": [], "scores": [], "coverage": {}, "relatorio": {}}

            # -------------------------
            # FILTROS (com relax)
            # -------------------------
            def apply_filters(docs_in, metas_in, min_density, min_words):
                filtered_local = []
                for d, m in zip(docs_in, metas_in):
                    if not d:
                        continue
                    txt = (d or "").strip()
                    wlen = len(txt.split())
                    if wlen < min_words:
                        continue
                    if self._chunk_is_bad(txt):
                        continue
                    dens = self._tech_density(txt)
                    if dens < min_density:
                        continue
                    useful = self._usefulness_score(txt)
                    boost = self._area_boost(
                        txt) + float((m or {}).get("area_boost", 0.0) or 0.0)
                    filtered_local.append(
                        (dens + 0.03 * boost + 0.8 * useful, txt, m))
                filtered_local.sort(key=lambda x: x[0], reverse=True)
                return filtered_local

            filtered = apply_filters(
                docs, metas, self.min_tech_density, self.min_chunk_words) if docs else []

            if docs and not filtered:
                relax_steps = [
                    (max(0.003, self.min_tech_density * 0.6),
                     max(16, int(self.min_chunk_words * 0.7))),
                    (max(0.002, self.min_tech_density * 0.4), 12),
                    (0.0, 10),
                ]
                for md, mw in relax_steps:
                    filtered = apply_filters(docs, metas, md, mw)
                    if filtered:
                        break

            filtered = filtered[: self.retrieval_prefilter_max] if filtered else []
            docs_f = [d for _, d, _ in filtered] if filtered else []
            metas_f = [m for _, _, m in filtered] if filtered else []

            kw_ranked_local = self._keyword_retrieval_local(
                descricao_problema, docs_f, metas_f, topk=self.keyword_topk) if docs_f else []

            merged_docs, merged_metas = [], []
            seen = set()

            def _key(d, m):
                if not isinstance(m, dict):
                    m = {}
                return (m.get("tipo"), m.get("page"), m.get("subchunk"), (d[:110] if d else ""))

            for d, m in zip(docs_f, metas_f):
                k = _key(d, m)
                if k in seen:
                    continue
                seen.add(k)
                merged_docs.append(d)
                merged_metas.append(m)

            for _, d, m in kw_ranked_local:
                k = _key(d, m)
                if k in seen:
                    continue
                seen.add(k)
                merged_docs.append(d)
                merged_metas.append(m)

            for _, d, m in substring_ranked:
                k = _key(d, m)
                if k in seen:
                    continue
                seen.add(k)
                merged_docs.append(d)
                merged_metas.append(m)

            if not merged_docs:
                return {"trechos": [], "metadados": [], "scores": [], "coverage": {}, "relatorio": {}}

            # -------------------------
            # RERANK + SCORE COMPOSTO
            # -------------------------
            rerank_scores = None
            if self.reranker:
                try:
                    pairs = [(descricao_problema, d) for d in merged_docs]
                    rerank_scores = self.reranker.predict(pairs).tolist()
                except Exception as e:
                    _log_warn(f"Falha no rerank (ignorando): {e}")
                    rerank_scores = None

            composite = []
            for i, (d, m) in enumerate(zip(merged_docs, merged_metas)):
                rr = rerank_scores[i] if (
                    rerank_scores and i < len(rerank_scores)) else None

                # level_bonus = 1.0 no fallback (sem ladder)
                comp = self._composite_score(d, m, rr, descricao_problema, level_bonus=1.0) \
                    if "level_bonus" in self._composite_score.__code__.co_varnames \
                    else self._composite_score(d, m, rr, descricao_problema)

                composite.append((comp, d, m, rr))

            composite.sort(key=lambda x: x[0], reverse=True)

            merged_docs = [d for _, d, _, _ in composite]
            merged_metas = [m for _, _, m, _ in composite]
            ranked_comp = [c for c, _, _, _ in composite]

        # =========================================================
        # 3) DIVERSIFICAÇÃO + MMR (GOLD PRO)
        # =========================================================
        ranked_docs = merged_docs[:]
        ranked_metas = merged_metas[:]
        ranked_comp = ranked_comp[:len(ranked_docs)]

        # Diversificação inicial (máx 1 por page/tipo) antes do MMR
        ranked_docs, ranked_metas = self._dedupe_and_diversify(
            ranked_docs, ranked_metas, max_per_page=1, max_total=self.mmr_max_total
        )
        ranked_comp = ranked_comp[:len(ranked_docs)]

        # MMR para diversidade semântica (MMR v2 se você substituiu _mmr_select)
        try:
            doc_embs = self.embedder.encode(ranked_docs).tolist()
            ranked_docs, ranked_metas, _ = self._mmr_select(
                query_emb, ranked_docs, ranked_metas, doc_embs, max_total=self.mmr_max_total
            )
        except Exception as e:
            if debug:
                _log_warn(f"[MMR] Falha (ignorando): {e}")

        # Diversificação final + corte final_k
        ranked_docs, ranked_metas = self._dedupe_and_diversify(
            ranked_docs, ranked_metas, max_per_page=1, max_total=self.final_k
        )

        # coverage
        coverage = self._coverage_score(ranked_docs)

        # spans (melhor parte do trecho)
        spans = []
        for d in ranked_docs:
            spans.append(self._extract_best_span(
                d, descricao_problema, max_len=650))

        # ajusta scores para tamanho final
        # (se ranked_comp foi truncado por diversificação, alinha por tamanho)
        final_scores = []
        if ranked_comp:
            # tenta mapear score pelo início do doc (método robusto)
            score_map = {}
            for d, sc in zip(merged_docs, ranked_comp):
                if isinstance(d, str):
                    score_map[d[:160]] = sc
            for d in ranked_docs:
                final_scores.append(score_map.get(d[:160], None))
        else:
            final_scores = [None] * len(spans)

        # relatório detalhado
        relatorio = {
            "used_try": used_try,
            "ladder_level": ladder_level,
            "ladder_bonus": ladder_bonus,
            "col_count": col_count,
            "has_embeddings": has_embs,
            "pre_total_candidates": len(merged_docs),
            "final_k": len(ranked_docs),
            "coverage": coverage,
        }

        if debug:
            _log_info(f"[DEBUG] used_try={used_try} ladder_level={ladder_level} "
                    f"pre={len(merged_docs)} final_docs={len(ranked_docs)} coverage={coverage}")

        return {
            "trechos": spans,
            "metadados": ranked_metas,
            "scores": final_scores[:len(spans)],
            "coverage": coverage,
            "relatorio": relatorio
        }

        
    def _quality_gates(self, fontes: List[Tuple[str, dict, Optional[float]]]) -> Dict[str, Any]:
        pages = set()
        tipos = set()
        text_all = ""

        for t, m, _ in fontes:
            meta = m or {}
            p = meta.get("page")
            if p is not None:
               pages.add(p)
            tipos.add(meta.get("tipo", "texto"))
            text_all += "\n" + (t or "")

        nums = re.findall(r"\b\d+(?:[\.,]\d+)?\b", text_all)
        units = re.findall(
            r"\b\d+(?:[\.,]\d+)?\s?(?:v|a|ma|hz|ohm|k\s?ohm|%|bar|mm|ºc|c)\b", text_all.lower())

        return {
            "unique_pages": len(pages),
            "unique_types": len(tipos),
            "numbers": len(nums),
            "units": len(units),
            "ok_pages": len(pages) >= self.gate_min_unique_pages,
            "ok_numbers": len(nums) >= self.gate_min_numbers,
            "ok_units": len(units) >= self.gate_min_units,
        }


    # ==========================================================
    # RELATÓRIO: mostrar trechos encontrados (NÃO GERA SOLUÇÃO)
    # ==========================================================
    def gerar_relatorio_trechos(self, descricao_problema: str, maquina_id: Optional[Any] = None, debug: bool = False) -> Dict[str, Any]:
        """
        Retorna um relatório dos trechos encontrados:
        - lista dos melhores trechos já recortados
        - metadados (tipo, página, subchunk, pdf_path)
        - score composto
        - justificativa de seleção
        """
        r = self.buscar_no_manual(
            descricao_problema, maquina_id=maquina_id, debug=debug)
        trechos = r.get("trechos", []) or []
        metas = r.get("metadados", []) or []
        scores = r.get("scores", []) or []
        rel = r.get("relatorio", {}) or {}

        itens = []
        for i in range(len(trechos)):
            trecho = trechos[i]
            meta = metas[i] if i < len(metas) else {}
            sc = scores[i] if i < len(scores) else None

            dens = self._tech_density(trecho)
            useful = self._usefulness_score(trecho)
            boost = self._area_boost(trecho)

            itens.append({
                "rank": i + 1,
                "score_composto": sc,
                "tipo": (meta or {}).get("tipo", "texto"),
                "page": (meta or {}).get("page", None),
                "subchunk": (meta or {}).get("subchunk", None),
                "maquina_id": (meta or {}).get("maquina_id", None),
                "nome_maquina": (meta or {}).get("nome_maquina", None),
                "pdf_path": (meta or {}).get("pdf_path", None),
                "densidade_tecnica": round(float(dens), 5),
                "utilidade": round(float(useful), 5),
                "area_boost": round(float(boost), 3),
                "trecho": trecho,
            })

        return {
            "sucesso": bool(itens),
            "consulta": {
                "descricao_problema": descricao_problema,
                "maquina_id": maquina_id,
                "grupo": self.grupo,
            },
            "relatorio": rel,
            "itens": itens
        }

    # ==========================================================
    # GERAR SOLUÇÃO (GOLD)
    # ==========================================================
    def _fonte_card(self, idx: int, trecho: str, meta: dict, score: Optional[float]) -> str:
        meta = meta if isinstance(meta, dict) else {}

        tipo = meta.get("tipo", "texto")
        page = meta.get("page", "?")
        maquina_id = meta.get("maquina_id", "?")
        nome = meta.get("nome_maquina", "")
        subchunk = meta.get("subchunk", None)

        txt = re.sub(r"\s+", " ", (trecho or "")).strip()

        # limite p/ card curto e legível
        if len(txt) > 820:
            txt = txt[:820].rstrip() + "..."

        head = f"[Fonte {idx}] (tipo={tipo}, page={page}, maquina_id={maquina_id}, score={score})"
        if nome:
            head += f" | maquina={nome}"
        if subchunk is not None:
            head += f" | subchunk={subchunk}"

        return f"{head}\n{txt}\n"

    def _citations_ok(self, text: str) -> bool:
        if not text or "[Fonte" not in text:
            return False

        # mínimo de citações distribuídas
        n = len(re.findall(r"\[Fonte\s+\d+\]", text))
        if n < getattr(self, "min_citations_required", 7):
            return False

        required_sections = [
            "Diagnóstico provável",
            "Causas prováveis",
            "Procedimento passo a passo",
            "Parâmetros/limites",
            "Checklist final",
            "Plano B",
            "Itens não encontrados",
        ]

        low = text.lower()
        for s in required_sections:
            if s.lower() not in low:
                return False

        return True
    
    def gerar_solucao(self,descricao_problema: str,casos_similares: Optional[List[Dict[str, Any]]] = None,maquina_id: Optional[Any] = None) -> Dict[str, Any]:

        if not self.openai_client:
            return {"sucesso": False, "erro": "OpenAI client nao disponivel. Configure OPENAI_API_KEY."}

        _log_info(
            f"🧠 Gerando solução GOLD | grupo={self.grupo} | maquina_id={maquina_id}")

        resultado_manual = self.buscar_no_manual(
            descricao_problema, maquina_id, debug=False)

        docs = resultado_manual.get("trechos", []) or []
        metas = resultado_manual.get("metadados", []) or []
        scores = resultado_manual.get("scores", []) or []
        coverage = resultado_manual.get("coverage", {}) or {}

        # ---- gates
        if len(docs) < self.min_evidence:
            return {
                "sucesso": False,
                "erro": "pouca_evidencia",
                "trechos_manual": docs,
                "metadados_manual": metas,
                "scores": scores,
                "coverage": coverage,
                "grupo": self.grupo,
            }

        if scores and scores[0] is not None and float(scores[0]) < self.rerank_score_min:
            return {
                "sucesso": False,
                "erro": f"score_fraco({scores[0]:.3f})",
                "trechos_manual": docs,
                "metadados_manual": metas,
                "scores": scores,
                "coverage": coverage,
                "grupo": self.grupo,
            }

        # ---- separa texto / imagem
        trechos_texto = []
        trechos_imagem = []

        for i in range(len(docs)):
            doc = docs[i]
            meta = metas[i] if i < len(metas) else {}
            sc = scores[i] if i < len(scores) else None
            tipo = (meta or {}).get("tipo", "texto")

            if tipo == "imagem":
                trechos_imagem.append({
                    "descricao_curta": doc,
                    "descricao_completa": (meta or {}).get("descricao_completa", "") or "",
                    "page": (meta or {}).get("page", 0),
                    "score": sc,
                })
            else:
                trechos_texto.append((doc, meta, sc))

        # ---- escolhe imagens fortes (se existirem)
        imagens_fortes = []
        for img in trechos_imagem:
            s = img.get("score")
            if s is None:
                continue
            if float(s) >= self.imagem_score_min:
                imagens_fortes.append(img)

        imagens_fortes = imagens_fortes[: self.max_imagens_para_resposta]

        # ---- monta fontes finais (texto + imagens fortes)
        fontes = [(doc, meta, sc) for doc, meta, sc in trechos_texto]

        for img in imagens_fortes:
            fontes.append((
                f"[DESCRIÇÃO DE IMAGEM] Página {img.get('page')}\n"
                f"Resumo: {img.get('descricao_curta')}\n"
                f"Detalhes: {(img.get('descricao_completa') or '')[:1100]}",
                {"tipo": "imagem", "page": img.get(
                    "page"), "maquina_id": maquina_id},
                img.get("score"),
            ))

        fontes = fontes[: self.max_fontes_para_resposta]

        out = self._gerar_solucao_texto_gold(
            descricao_problema=descricao_problema,
            fontes=fontes,
            casos_similares=casos_similares,
            coverage=coverage,
        )

        out["trechos_manual"] = docs
        out["metadados_manual"] = metas
        out["scores"] = scores
        out["coverage"] = coverage
        out["grupo"] = self.grupo

        out["sucesso"] = bool(out.get("solucao"))
        return out

    def _gerar_solucao_texto_gold(self,descricao_problema: str,fontes: List[Tuple[str, dict, Optional[float]]],casos_similares: Optional[List[Dict[str, Any]]] = None,coverage: Optional[Dict[str, int]] = None,) -> Dict[str, Any]:

        # ---- monta contexto com cards
        contexto = ""
        for i, (trecho, meta, sc) in enumerate(fontes, 1):
            contexto += self._fonte_card(i, trecho, meta, sc)

        # ---- casos similares
        if casos_similares:
            contexto += "\n[CASOS SIMILARES]\n"
            for i, caso in enumerate(casos_similares, 1):
                problema = (caso.get("descricao") or "N/A").strip()
                sol = (caso.get("solucao") or "N/A").strip()
                if len(sol) > 900:
                    sol = sol[:900].rstrip() + "..."
                contexto += f"[Caso {i}] Problema: {problema}\n[Caso {i}] Solução: {sol}\n\n"

        cov_line = f"Coverage do manual: {coverage}" if coverage else ""

        prompt_answer = f"""
Você é um especialista em manutenção industrial da Metalsider.
Prioridade: Fornos, DICU, Fusão, Prodapt MD. Também DISA quando relacionado.

REGRAS INEGOCIÁVEIS:
1) Use SOMENTE as fontes abaixo.
2) Toda afirmação técnica deve conter [Fonte X].
3) Se não houver evidência, escreva: "não encontrado no manual".
4) Não invente tags, alarmes, telas, valores.
5) Sempre inclua: como validar + plano B.
6) Todo passo do procedimento deve ter OK/NOK e o que fazer se NOK.
7) A resposta deve ter números/unidades reais extraídos das fontes.

{cov_line}

FONTES:
{contexto}

PROBLEMA:
{descricao_problema}

FORMATO OBRIGATÓRIO:
1) Diagnóstico provável (com citações)
2) Causas prováveis (ordem de probabilidade) (com citações)
3) Procedimento passo a passo (com citações, cada passo com OK/NOK)
4) Parâmetros/limites relevantes (com citações)
5) Checklist final + validação (com citações)
6) Plano B (se não resolver)
7) Itens não encontrados no manual
"""

        try:
            # ------------- PASS 1 -------------
            resp1 = self.openai_client.chat.completions.create(
                model=GPT_MODEL_MINI,
                messages=[
                    {"role": "system",
                        "content": "Você é um especialista em manutenção. Você DEVE citar fontes [Fonte X]."},
                    {"role": "user", "content": prompt_answer},
                ],
                temperature=0.20,
                max_tokens=self.two_pass_max_tokens if self.enable_two_pass else 980,
            )
            answer1 = (resp1.choices[0].message.content or "").strip()

            if not self.enable_two_pass:
                if not self._citations_ok(answer1):
                    answer1 = "⚠️ Resposta com citações insuficientes. Aumente top-k ou informe alarme/código.\n\n" + answer1
                return {"solucao": answer1, "modelo_usado": GPT_MODEL_MINI}

            # ------------- PASS 2 (AUDITOR AGRESSIVO) -------------
            prompt_critic = f"""
Você é um AUDITOR técnico agressivo e reprovador.

CHECKLIST OBRIGATÓRIO:
A) Cada seção deve ter [Fonte X] em toda afirmação.
B) Cada passo deve ter: Ação + Como medir + Critério OK/NOK + O que fazer se NOK.
C) Deve existir pelo menos:
   - 8 números OU
   - 4 unidades (°C, bar, V, A, Hz, %, mm, Ohm, etc)
D) Deve listar explicitamente "Itens não encontrados no manual".
E) Se houver qualquer frase genérica: reescreva como instrução executável.
F) Se houver afirmação sem fonte: reescreva ou substitua por "não encontrado no manual".

PROBLEMA:
{descricao_problema}

FONTES:
{contexto}

RESPOSTA ATUAL:
{answer1}

SAÍDA:
Entregue SOMENTE a versão FINAL revisada.
"""

            resp2 = self.openai_client.chat.completions.create(
                model=GPT_MODEL_MINI,
                messages=[
                    {"role": "system",
                        "content": "Você é um auditor técnico. Reescreva para padrão ouro e mantenha citações [Fonte X]."},
                    {"role": "user", "content": prompt_critic},
                ],
                temperature=self.two_pass_temperature,
                max_tokens=self.two_pass_max_tokens,
            )
            final = (resp2.choices[0].message.content or "").strip()

            if not self._citations_ok(final):
                final = "⚠️ Resultado final com densidade de citações abaixo do padrão. Forneça código/alarme ou aumente top-k.\n\n" + final

            return {"solucao": final, "modelo_usado": GPT_MODEL_MINI, "refino": True}

        except Exception as e:
            _log_err(f"Erro ao gerar solução GOLD: {e}")
            return {"erro": str(e)}


    # ==========================================================
    # LISTAR MÁQUINAS PROCESSADAS (por grupo)
    # ==========================================================
    def listar_maquinas_processadas(self) -> Dict[str, str]:
        if not self.collection:
            return {}

        try:
            all_data = self.collection.get(include=["metadatas"])
        except Exception:
            try:
                all_data = self.collection.get()
            except Exception:
                return {}

        maquinas = {}
        metas = all_data.get("metadatas", None)
        metas = metas if metas is not None else []

        for metadata in metas:
            if not isinstance(metadata, dict):
                continue
            if metadata.get("grupo", self.grupo) != self.grupo:
                continue

            maquina_id = metadata.get("maquina_id")
            nome = metadata.get("nome_maquina")
            if maquina_id and maquina_id not in maquinas:
                maquinas[maquina_id] = nome

        return maquinas

# ==========================================================
# CLEANUP / FINAL HELPERS / CLI
# ==========================================================


def _parse_manual_filename(pdf_path: str) -> Tuple[Optional[int], str]:
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

    # fallback
    return None, name.replace("_", " ").strip()


def _ask_bool(prompt: str, default: bool = True) -> bool:
    suf = " [S/n] " if default else " [s/N] "
    val = input(prompt + suf).strip().lower()
    if not val:
        return default
    return val in ("s", "sim", "y", "yes", "1", "true")


def _ask_int(prompt: str, default: int) -> int:
    val = input(f"{prompt} (default={default}): ").strip()
    if not val:
        return default
    try:
        return int(val)
    except Exception:
        return default


def _ask_float(prompt: str, default: float) -> float:
    val = input(f"{prompt} (default={default}): ").strip().replace(",", ".")
    if not val:
        return default
    try:
        return float(val)
    except Exception:
        return default


def _list_pdfs_in_dir(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    out = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(".pdf"):
            out.append(os.path.join(folder, fn))
    return out


def _print_header():
    print("\n" + "=" * 72)
    print("RAG GOLD - Processador de Manuais (Texto + Imagens + OCR opcional)")
    print("=" * 72 + "\n")


def _close_safely(rag: "RAGSystem"):
    try:
        conn = getattr(rag, "_pg_conn_cache", None)
        if conn:
            conn.close()
            _log_ok("[PG] Conexão encerrada.")
    except Exception:
        pass


def processar_pdf_interativo(rag: "RAGSystem"):
    pdf_path = input("Caminho do PDF: ").strip().strip('"').strip("'")
    if not os.path.exists(pdf_path):
        print("Arquivo não encontrado.")
        return

    mid, nome = _parse_manual_filename(pdf_path)

    print(f"\nDetectado pelo nome do arquivo:")
    print(f"  maquina_id/manual_id: {mid}")
    print(f"  nome_maquina: {nome}\n")

    manual_id = input(
        f"Digite maquina_id/manual_id (ENTER para usar {mid}): ").strip()
    if manual_id:
        maquina_id = manual_id
    else:
        maquina_id = mid if mid is not None else input(
            "Digite maquina_id/manual_id: ").strip()

    nome_maquina = input(
        f"Digite nome_maquina (ENTER para usar '{nome}'): ").strip()
    if not nome_maquina:
        nome_maquina = nome

    use_ocr = _ask_bool("Rodar OCR se PDF for escaneado?", default=True)
    processar_imgs = _ask_bool(
        "Processar imagens/páginas inteiras?", default=True)

    imgs_paginas = None
    if processar_imgs:
        imgs_paginas = input(
            "Filtrar páginas das imagens? (ex: 1-5,8,10) ENTER=TODAS: ").strip()
        imgs_paginas = imgs_paginas if imgs_paginas else None

    render_dpi = _ask_int("DPI do render (imagens)",
                          default=RENDER_DPI_DEFAULT)
    dedupe_phash = _ask_bool(
        "Remover páginas duplicadas por pHash?", default=False)

    ok = False
    if use_ocr:
        ok = rag.processar_pdf_hibrido(
            pdf_path=pdf_path,
            maquina_id=maquina_id,
            nome_maquina=nome_maquina,
            processar_imagens=processar_imgs,
            imagens_paginas=imgs_paginas,
            imagens_modo_pagina_inteira=True,
            render_dpi=render_dpi,
            filtrar_paginas_vazias=True,
            dedupe_paginas_phash=dedupe_phash,
        )
    else:
        ok = rag.processar_pdf(
            pdf_path=pdf_path,
            maquina_id=maquina_id,
            nome_maquina=nome_maquina,
            processar_imagens=processar_imgs,
            imagens_paginas=imgs_paginas,
            imagens_modo_pagina_inteira=True,
            render_dpi=render_dpi,
            filtrar_paginas_vazias=True,
            dedupe_paginas_phash=dedupe_phash,
        )

    print("\n✅ OK!" if ok else "\n❌ Falhou!")
    return


def processar_pasta_interativo(rag: "RAGSystem"):
    folder = input("Pasta com PDFs (default=./manuais): ").strip()
    if not folder:
        folder = "./manuais"
    folder = _abs_path(folder)

    pdfs = _list_pdfs_in_dir(folder)
    if not pdfs:
        print(f"Nenhum PDF encontrado em {folder}")
        return

    print(f"\nEncontrados {len(pdfs)} PDFs:\n")
    for i, p in enumerate(pdfs, 1):
        print(f"{i:>3}. {os.path.basename(p)}")

    sel = input("\nEscolha índices (ex: 1,3,5) ou ENTER=TODOS: ").strip()
    if sel:
        idxs = []
        for part in sel.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                idxs.append(int(part))
            except Exception:
                pass
        pdfs_sel = []
        for i in idxs:
            if 1 <= i <= len(pdfs):
                pdfs_sel.append(pdfs[i-1])
        pdfs = pdfs_sel

    if not pdfs:
        print("Nenhum PDF selecionado.")
        return

    use_ocr = _ask_bool("Rodar OCR se PDF for escaneado?", default=True)
    processar_imgs = _ask_bool(
        "Processar imagens/páginas inteiras?", default=True)
    imgs_paginas = None
    if processar_imgs:
        imgs_paginas = input(
            "Filtrar páginas das imagens? (ex: 1-5,8,10) ENTER=TODAS: ").strip()
        imgs_paginas = imgs_paginas if imgs_paginas else None

    render_dpi = _ask_int("DPI do render (imagens)",
                          default=RENDER_DPI_DEFAULT)
    dedupe_phash = _ask_bool(
        "Remover páginas duplicadas por pHash?", default=False)

    print("\n--- Iniciando processamento ---\n")
    ok_total, fail_total = 0, 0

    for n, pdf_path in enumerate(pdfs, 1):
        mid, nome = _parse_manual_filename(pdf_path)
        if mid is None:
            print(
                f"\n[{n}/{len(pdfs)}] ⚠️ Não detectei ID no nome: {os.path.basename(pdf_path)}")
            maquina_id = input("Digite maquina_id/manual_id: ").strip()
        else:
            maquina_id = mid

        nome_maquina = nome

        print(
            f"\n[{n}/{len(pdfs)}] 📄 {os.path.basename(pdf_path)} | ID={maquina_id} | nome={nome_maquina}")

        try:
            if use_ocr:
                ok = rag.processar_pdf_hibrido(
                    pdf_path=pdf_path,
                    maquina_id=maquina_id,
                    nome_maquina=nome_maquina,
                    processar_imagens=processar_imgs,
                    imagens_paginas=imgs_paginas,
                    imagens_modo_pagina_inteira=True,
                    render_dpi=render_dpi,
                    filtrar_paginas_vazias=True,
                    dedupe_paginas_phash=dedupe_phash,
                )
            else:
                ok = rag.processar_pdf(
                    pdf_path=pdf_path,
                    maquina_id=maquina_id,
                    nome_maquina=nome_maquina,
                    processar_imagens=processar_imgs,
                    imagens_paginas=imgs_paginas,
                    imagens_modo_pagina_inteira=True,
                    render_dpi=render_dpi,
                    filtrar_paginas_vazias=True,
                    dedupe_paginas_phash=dedupe_phash,
                )

            if ok:
                ok_total += 1
            else:
                fail_total += 1
        except Exception as e:
            fail_total += 1
            _log_err(f"Erro inesperado: {e}")

    print("\n--- FIM ---")
    print(f"✅ OK:   {ok_total}")
    print(f"❌ FAIL: {fail_total}\n")


def consultar_interativo(rag: "RAGSystem"):
    problema = input("Descreva o problema (consulta ao manual): ").strip()
    if not problema:
        print("Sem descrição.")
        return

    maquina_id = input("Filtrar por maquina_id? (ENTER=sem filtro): ").strip()
    maquina_id = maquina_id if maquina_id else None

    modo = input(
        "Modo: (1) Solução (2) Relatório de trechos  [default=1]: ").strip()
    if modo == "2":
        r = rag.gerar_relatorio_trechos(
            problema, maquina_id=maquina_id, debug=False)
        print("\nRELATÓRIO:")
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    r = rag.gerar_solucao(problema, casos_similares=None,
                          maquina_id=maquina_id)
    if not r.get("sucesso"):
        print("\n❌ Não foi possível gerar solução.")
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    print("\n✅ SOLUÇÃO:")
    print(r.get("solucao", "").strip())
    return


def main():
    _print_header()

    grupo = input("Grupo inicial (ENTER=geral): ").strip() or "geral"
    traduzir = _ask_bool(
        "Traduzir/Detectar idioma via OpenAI (se disponível)?", default=True)

    rag = None
    try:
        rag = RAGSystem(traduzir_automaticamente=traduzir, grupo=grupo)

        while True:
            print("\nMENU:")
            print("  1) Processar PDF (interativo)")
            print("  2) Processar pasta de PDFs (./manuais)")
            print("  3) Consultar manual (gerar solução/relatório)")
            print("  4) Listar máquinas processadas (grupo atual)")
            print("  5) Trocar grupo")
            print("  0) Sair")

            op = input("\nEscolha: ").strip()

            if op == "1":
                processar_pdf_interativo(rag)

            elif op == "2":
                processar_pasta_interativo(rag)

            elif op == "3":
                consultar_interativo(rag)

            elif op == "4":
                m = rag.listar_maquinas_processadas()
                if not m:
                    print("\nNenhuma máquina encontrada no grupo atual.")
                else:
                    print("\nMÁQUINAS PROCESSADAS:")
                    for k, v in sorted(m.items(), key=lambda x: str(x[0])):
                        print(f"  - ID={k} | {v}")

            elif op == "5":
                novo = input("Novo grupo: ").strip() or "geral"
                rag.set_grupo(novo)

            elif op == "0":
                break

            else:
                print("Opção inválida.")

    finally:
        if rag:
            _close_safely(rag)


# ==========================================================
# ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    main()
