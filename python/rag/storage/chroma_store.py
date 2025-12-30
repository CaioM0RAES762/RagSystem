# rag/storage/chroma_store.py

import os
from typing import Optional, Dict, Any, List, Tuple

import chromadb
from chromadb.config import Settings

from rag.logger import log_info as _log_info, log_ok as _log_ok, log_warn as _log_warn


def _abs_path(p: str) -> str:
    try:
        return os.path.abspath(p)
    except Exception:
        return p


def _ensure_dir(p: str) -> str:
    ap = _abs_path(p)
    os.makedirs(ap, exist_ok=True)
    return ap


class ChromaStore:
    """
    Camada de acesso ChromaDB SAFE.
    - Normaliza path absoluto
    - Query com fallback sem where
    - count() seguro via get()
    - Helpers para add/get/query
    """

    def __init__(self, chroma_path: str, grupo: str = "geral"):
        self.grupo = (grupo or "geral").strip() or "geral"
        self.chroma_path = _ensure_dir(chroma_path)

        self.client = None
        self.collection = None
        self.collection_name = None

        _log_info(
            f"[ChromaStore] Path={chroma_path} | ABS={self.chroma_path} | CWD={os.getcwd()}")

        try:
            self.client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False),
            )
            _log_ok("[ChromaStore] PersistentClient inicializado.")
        except Exception as e:
            _log_warn(
                f"[ChromaStore] Falha ao inicializar PersistentClient: {e}")
            self.client = None

    # --------------------------
    # Coleção por grupo
    # --------------------------
    def collection_name_for_group(self, grupo: str) -> str:
        g = (grupo or "geral").strip() or "geral"
        return "manuais_maquinas" if g in ("geral", "", None) else f"manuais_maquinas_{g}"

    def set_group(self, grupo: str) -> bool:
        self.grupo = (grupo or "geral").strip() or "geral"
        return self.open_collection(self.grupo)

    def open_collection(self, grupo: Optional[str] = None) -> bool:
        if not self.client:
            return False

        grupo = (grupo or self.grupo or "geral").strip() or "geral"
        self.grupo = grupo

        try:
            self.collection_name = self.collection_name_for_group(grupo)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "grupo": self.grupo},
            )
            _log_ok(f"[ChromaStore] Usando coleção: {self.collection_name}")
            return True
        except Exception as e:
            _log_warn(f"[ChromaStore] Falha open_collection: {e}")
            self.collection = None
            return False

    # --------------------------
    # Helpers SAFE
    # --------------------------
    def safe_query(
        self,
        *,
        query_emb: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        n_results: int = 30,
        where: Optional[dict] = None,
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
            _log_warn(f"[ChromaStore] query falhou com where={where}: {e}")

            # fallback sem where
            try:
                if query_emb is not None:
                    return self.collection.query(query_embeddings=[query_emb], n_results=n_results)
                if query_text is not None:
                    return self.collection.query(query_texts=[query_text], n_results=n_results)
                return _empty()
            except Exception as e2:
                _log_warn(
                    f"[ChromaStore] fallback sem where também falhou: {e2}")
                return _empty()

    def count_safe(self) -> int:
        if not self.collection:
            return 0

        try:
            c = self.collection.count()
            if isinstance(c, int):
                return c
        except Exception as e:
            _log_warn(f"[ChromaStore] collection.count() falhou: {e}")

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
            _log_warn(f"[ChromaStore] count via get() falhou: {e}")
            return 0

    def has_embeddings(self) -> bool:
        if not self.collection:
            return False
        try:
            got = self.collection.get(include=["embeddings"], limit=1)
            embs = got.get("embeddings", None)
            if not embs:
                return False
            return len(embs) > 0 and len(embs[0]) > 0
        except Exception:
            return False

    def get_by_ids(self, ids: List[str], include: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self.collection:
            return {}
        try:
            return self.collection.get(ids=ids, include=include or ["ids"])
        except Exception:
            return {}

    def scan_get(
        self,
        *,
        where: Optional[dict],
        max_docs: int,
        include: Optional[List[str]] = None,
        debug: bool = False,
    ) -> Tuple[List[str], List[dict]]:
        """
        Busca documentos/metadados para substring scan
        """
        if not self.collection:
            return [], []

        include = include or ["documents", "metadatas"]

        tries = [
            ("where+limit", lambda: self.collection.get(where=where,
             include=include, limit=max_docs) if where else None),
            ("limit", lambda: self.collection.get(
                include=include, limit=max_docs)),
            ("all", lambda: self.collection.get(include=include)),
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
                        _log_info(
                            f"[ChromaStore.scan_get] ok tag={tag} docs={len(docs)}")
                    return docs[:max_docs], metas[:max_docs]
            except Exception as e:
                if debug:
                    _log_warn(f"[ChromaStore.scan_get] fail tag={tag}: {e}")
                continue

        return [], []

    # --------------------------
    # ADD HELPERS
    # --------------------------
    def add_text_chunk(
        self,
        *,
        chunk_id: str,
        text: str,
        embedding: List[float],
        metadata: dict,
    ) -> bool:
        if not self.collection:
            return False

        try:
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[chunk_id],
            )
            return True
        except Exception as e:
            _log_warn(
                f"[ChromaStore] add_text_chunk falhou id={chunk_id}: {e}")
            return False

    def add_image_doc(
        self,
        *,
        image_id: str,
        description_short: str,
        embedding: List[float],
        metadata: dict,
    ) -> bool:
        if not self.collection:
            return False

        try:
            self.collection.add(
                documents=[description_short],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[image_id],
            )
            return True
        except Exception as e:
            _log_warn(f"[ChromaStore] add_image_doc falhou id={image_id}: {e}")
            return False
