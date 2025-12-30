# rag/rag_system.py
"""
Facade principal do pacote rag/

Objetivo:
- Expor uma API única e estável:
    - processar_pdf / processar_pdf_hibrido
    - buscar_no_manual
    - gerar_solucao
    - gerar_relatorio_trechos
    - set_grupo / listar_maquinas_processadas
- Conectar ingest + retrieval + generation + storage
- Garantir que o Chroma use SEMPRE path absoluto e consistente (evita "ainda não indexado")
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple

# ---------------------------
# Config / Logging
# ---------------------------
from rag.config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    GPT_MODEL_MINI,
    GPT_MODEL,
    VISION_MODEL,
    OPENAI_API_KEY,
    OpenAI,
    DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT,
    MANUALS_IMAGES_DIR,
    MANUALS_PAGES_DIR,
)

from rag.logger import (
    fmt_secs,
    log_step,
    log_info,
    log_ok,
    log_warn,
    log_err,
)

# ---------------------------
# Utils / Paths
# ---------------------------
from rag.utils.paths import (
    ensure_dir,
    abs_project_path,
    resolve_chroma_path,
)

# ---------------------------
# Storage
# ---------------------------
from rag.storage.chroma_store import ChromaStore
from rag.storage.pg_store import PGStore

# ---------------------------
# Ingest
# ---------------------------
from rag.ingest.text_pipeline import TextPipeline
from rag.ingest.image_pipeline import ImagePipeline
from rag.ingest.ocr import OCRPipeline
from rag.ingest.render_pages import RenderPages

# ---------------------------
# Retrieval
# ---------------------------
from rag.retrieval.query import QueryBuilder
from rag.retrieval.ladder import RetrievalLadder
from rag.retrieval.rerank import Reranker
from rag.retrieval.mmr import MMRSelector

# ---------------------------
# Generation
# ---------------------------
from rag.generation.answer_gold import GoldAnswerGenerator
from rag.generation.report import ReportGenerator


# ==========================================================
# Dataclass de configs internas
# ==========================================================
@dataclass
class RAGSystemOptions:
    grupo: str = "geral"
    traduzir_automaticamente: bool = True

    # retrieval
    final_k: int = 16
    topk: int = 180
    enable_ladder: bool = True
    enable_rerank: bool = True
    enable_mmr: bool = True

    # ingest
    processar_imagens: bool = True
    imagens_modo_pagina_inteira: bool = True

    # guards
    min_evidence: int = 5


# ==========================================================
# Facade principal
# ==========================================================
class RAGSystem:
    """
    Classe pública / Facade do pacote rag/.

    Use como:
        rag = RAGSystem(grupo="geral")
        rag.processar_pdf_hibrido("manual.pdf", 123, "Forno DICU")
        r = rag.gerar_solucao("forno sem ignição", maquina_id=123)
    """

    def __init__(self, traduzir_automaticamente: bool = True, grupo: str = "geral", **kwargs):
        t0 = time.time()

        self.options = RAGSystemOptions(
            grupo=(grupo or "geral").strip() or "geral",
            traduzir_automaticamente=bool(traduzir_automaticamente),
        )

        # permitir overrides via kwargs
        for k, v in kwargs.items():
            if hasattr(self.options, k):
                setattr(self.options, k, v)

        # -----------------------------------
        # Fix de path absoluto / consistente
        # -----------------------------------
        chroma_abs = resolve_chroma_path(CHROMA_DB_PATH)
        ensure_dir(chroma_abs)

        self.chroma_path = chroma_abs
        self.grupo = self.options.grupo
        self.collection_name = self._collection_name_for_group(self.grupo)

        log_info("Inicializando RAGSystem (Facade)")
        log_info(f"[PATH] CHROMA_DB_PATH={CHROMA_DB_PATH}")
        log_info(f"[PATH] CHROMA_ABS={self.chroma_path}")
        log_info(f"[PATH] CWD={os.getcwd()}")
        log_info(
            f"[GROUP] grupo={self.grupo} | collection={self.collection_name}")

        # -----------------------------------
        # OpenAI client
        # -----------------------------------
        self.openai_client = None
        if OpenAI and OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                log_ok("OpenAI client inicializado.")
            except Exception as e:
                log_warn(f"OpenAI client falhou: {e}")
                self.openai_client = None
        else:
            log_warn("OpenAI não disponível (lib/OPENAI_API_KEY).")

        # -----------------------------------
        # Storage: Chroma
        # -----------------------------------
        self.store = ChromaStore(
            chroma_path=self.chroma_path,
            collection_name=self.collection_name,
            embedding_model=EMBEDDING_MODEL,
            grupo=self.grupo,
        )

        # -----------------------------------
        # Storage: Postgres (opcional)
        # -----------------------------------
        self.pg = PGStore(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT,
        )

        # -----------------------------------
        # Pipelines de ingest
        # -----------------------------------
        self.text_pipeline = TextPipeline(store=self.store, grupo=self.grupo)
        self.image_pipeline = ImagePipeline(
            store=self.store,
            pg=self.pg,
            openai_client=self.openai_client,
            grupo=self.grupo,
            pages_dir=ensure_dir(os.path.join(MANUALS_PAGES_DIR, self.grupo)),
            images_dir=ensure_dir(os.path.join(
                MANUALS_IMAGES_DIR, self.grupo)),
        )
        self.ocr_pipeline = OCRPipeline()
        self.render_pages = RenderPages(
            pages_dir=ensure_dir(os.path.join(MANUALS_PAGES_DIR, self.grupo)),
            images_dir=ensure_dir(os.path.join(
                MANUALS_IMAGES_DIR, self.grupo)),
        )

        # -----------------------------------
        # Retrieval
        # -----------------------------------
        self.query_builder = QueryBuilder(grupo=self.grupo)
        self.reranker = Reranker(enabled=self.options.enable_rerank)
        self.mmr = MMRSelector(enabled=self.options.enable_mmr)
        self.ladder = RetrievalLadder(enabled=self.options.enable_ladder)

        # -----------------------------------
        # Generation
        # -----------------------------------
        self.answer_gen = GoldAnswerGenerator(
            openai_client=self.openai_client,
            model=GPT_MODEL_MINI,
        )
        self.report_gen = ReportGenerator()

        log_ok(f"RAGSystem pronto em {fmt_secs(time.time() - t0)}")

    # ==========================================================
    # Grupo / coleção
    # ==========================================================
    def _collection_name_for_group(self, grupo: str) -> str:
        grupo = (grupo or "geral").strip() or "geral"
        return "manuais_maquinas" if grupo == "geral" else f"manuais_maquinas_{grupo}"

    def set_grupo(self, novo_grupo: str) -> bool:
        novo_grupo = (novo_grupo or "geral").strip() or "geral"
        if novo_grupo == self.grupo:
            return True

        self.grupo = novo_grupo
        self.options.grupo = novo_grupo
        self.collection_name = self._collection_name_for_group(novo_grupo)

        # atualiza store
        ok = self.store.set_collection(self.collection_name, grupo=self.grupo)

        # atualiza pipelines
        self.text_pipeline.set_grupo(self.grupo)
        self.image_pipeline.set_grupo(self.grupo)

        log_ok(
            f"[set_grupo] grupo={self.grupo} | collection={self.collection_name}")
        return ok

    # ==========================================================
    # Ingest (PDF -> texto + imagens)
    # ==========================================================
    def processar_pdf_hibrido(
        self,
        pdf_path: str,
        maquina_id: Any,
        nome_maquina: str,
        *,
        scanned_ratio_threshold: float = 0.65,
        ocr_language: str = "por",
        ocr_jobs: int = 2,
        force_ocr: bool = False,
        keep_ocr_pdf: bool = True,
        processar_imagens: bool = True,
        imagens_paginas: Optional[str] = None,
        render_dpi: int = 170,
        dedupe_paginas_phash: bool = False,
    ) -> bool:

        log_step("Preflight + OCR (se necessário)")
        use_pdf = pdf_path

        # OCR decision
        try:
            need_ocr = self.ocr_pipeline.should_ocr(
                pdf_path, scanned_ratio_threshold)
        except Exception:
            need_ocr = False

        temp_ocr_pdf = None
        if need_ocr:
            log_info("PDF parece escaneado → tentando OCRmyPDF")
            out_pdf = os.path.splitext(pdf_path)[0] + "_OCR.pdf"
            ok, msg = self.ocr_pipeline.run_ocr(
                pdf_path=pdf_path,
                output_pdf_path=out_pdf,
                language=ocr_language,
                jobs=ocr_jobs,
                force_ocr=force_ocr,
            )
            if ok:
                log_ok(msg)
                use_pdf = out_pdf
                temp_ocr_pdf = out_pdf
            else:
                log_warn(msg)
                log_warn("Seguindo com o PDF original")

        ok = self.processar_pdf(
            use_pdf,
            maquina_id,
            nome_maquina,
            processar_imagens=processar_imagens,
            imagens_paginas=imagens_paginas,
            render_dpi=render_dpi,
            dedupe_paginas_phash=dedupe_paginas_phash,
        )

        if temp_ocr_pdf and (not keep_ocr_pdf):
            try:
                os.remove(temp_ocr_pdf)
            except Exception:
                pass

        return ok

    def processar_pdf(
        self,
        pdf_path: str,
        maquina_id: Any,
        nome_maquina: str,
        processar_imagens: bool = True,
        *,
        imagens_paginas: Optional[str] = None,
        render_dpi: int = 170,
        dedupe_paginas_phash: bool = False,
    ) -> bool:

        if not os.path.exists(pdf_path):
            log_err(f"PDF não existe: {pdf_path}")
            return False

        t0 = time.time()
        log_step(
            f"Processando PDF: {os.path.basename(pdf_path)} | grupo={self.grupo}")

        # --- texto
        ok_texto = self.text_pipeline.ingest_pdf_text(
            pdf_path=pdf_path,
            maquina_id=maquina_id,
            nome_maquina=nome_maquina,
        )
        if not ok_texto:
            log_warn("Texto falhou ou não gerou chunks suficientes")

        # --- imagens
        ok_img = True
        if processar_imagens:
            pages = self.render_pages.render_pdf_pages(
                pdf_path=pdf_path,
                maquina_id=maquina_id,
                imagens_paginas=imagens_paginas,
                dpi=render_dpi,
                dedupe_phash=dedupe_paginas_phash,
            )

            ok_img = self.image_pipeline.ingest_rendered_pages(
                pages,
                maquina_id=maquina_id,
                nome_maquina=nome_maquina,
                pdf_path=pdf_path,
            )

        log_ok(f"Processamento finalizado em {fmt_secs(time.time() - t0)}")
        return bool(ok_texto or ok_img)

    # ==========================================================
    # Retrieval
    # ==========================================================
    def buscar_no_manual(self, descricao_problema: str, maquina_id: Optional[Any] = None, debug: bool = False) -> Dict[str, Any]:

        col_count = self.store.count()
        if col_count <= 0:
            log_warn(
                f"Coleção vazia: {self.collection_name} | grupo={self.grupo}")
            return {
                "trechos": [],
                "metadados": [],
                "scores": [],
                "coverage": {},
                "relatorio": {"col_count": col_count, "grupo": self.grupo}
            }

        query = self.query_builder.build(descricao_problema)
        candidates = self.store.query(
            query, topk=self.options.topk, maquina_id=maquina_id)

        # ladder
        if self.options.enable_ladder:
            candidates = self.ladder.run(
                query, candidates, maquina_id=maquina_id)

        # rerank
        if self.options.enable_rerank:
            candidates = self.reranker.run(descricao_problema, candidates)

        # mmr
        if self.options.enable_mmr:
            candidates = self.mmr.run(
                query, candidates, final_k=self.options.final_k)

        trechos = [c["span"] for c in candidates]
        metas = [c["meta"] for c in candidates]
        scores = [c.get("score") for c in candidates]

        relatorio = {
            "grupo": self.grupo,
            "collection": self.collection_name,
            "col_count": col_count,
            "final_k": len(trechos),
        }

        return {
            "trechos": trechos,
            "metadados": metas,
            "scores": scores,
            "coverage": self.report_gen.coverage(trechos),
            "relatorio": relatorio,
        }

    # ==========================================================
    # Report
    # ==========================================================
    def gerar_relatorio_trechos(self, descricao_problema: str, maquina_id: Optional[Any] = None, debug: bool = False) -> Dict[str, Any]:
        r = self.buscar_no_manual(
            descricao_problema, maquina_id=maquina_id, debug=debug)
        return self.report_gen.generate(descricao_problema, r)

    # ==========================================================
    # Answer gold (solução)
    # ==========================================================
    def gerar_solucao(
        self,
        descricao_problema: str,
        casos_similares: Optional[List[Dict[str, Any]]] = None,
        maquina_id: Optional[Any] = None,
    ) -> Dict[str, Any]:

        if not self.openai_client:
            return {"sucesso": False, "erro": "OpenAI não disponível. Configure OPENAI_API_KEY."}

        r = self.buscar_no_manual(
            descricao_problema, maquina_id=maquina_id, debug=False)
        trechos = r.get("trechos") or []
        metas = r.get("metadados") or []
        scores = r.get("scores") or []

        if len(trechos) < self.options.min_evidence:
            return {
                "sucesso": False,
                "erro": "pouca_evidencia",
                "trechos_manual": trechos,
                "metadados_manual": metas,
                "scores": scores,
                "relatorio": r.get("relatorio", {}),
            }

        sol = self.answer_gen.generate(
            descricao_problema=descricao_problema,
            fontes=list(zip(trechos, metas, scores)),
            casos_similares=casos_similares,
            coverage=r.get("coverage", {}),
        )

        sol["trechos_manual"] = trechos
        sol["metadados_manual"] = metas
        sol["scores"] = scores
        sol["relatorio"] = r.get("relatorio", {})
        sol["sucesso"] = bool(sol.get("solucao"))

        return sol

    # ==========================================================
    # Lista máquinas
    # ==========================================================
    def listar_maquinas_processadas(self) -> Dict[str, str]:
        return self.store.list_maquinas(grupo=self.grupo)
