from typing import Optional, Any, Dict, List, Tuple
import re


class RetrievalEngine:
    """
    Retrieval Engine: encapsula o pipeline completo:
    - ladder (opcional)
    - fallback embedding/keyword/substring
    - filtros, rerank, score composto
    - diversificação e MMR
    """

    def __init__(
        self,
        *,
        embedder,
        chroma_collection,
        filters,
        ladder=None,
        reranker=None,
        mmr_select_fn=None,
        logger=None,
    ):
        self.embedder = embedder
        self.collection = chroma_collection
        self.filters = filters
        self.ladder = ladder
        self.reranker = reranker
        self.mmr_select_fn = mmr_select_fn
        self.logger = logger

    # ------------------------------------
    # safe query chroma
    # ------------------------------------
    def safe_query(self, *, query_emb=None, query_text=None, n_results=30, where=None):
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
        except Exception:
            try:
                if query_emb is not None:
                    return self.collection.query(query_embeddings=[query_emb], n_results=n_results)
                if query_text is not None:
                    return self.collection.query(query_texts=[query_text], n_results=n_results)
                return _empty()
            except Exception:
                return _empty()

    def collection_has_embeddings(self) -> bool:
        if not self.collection:
            return False
        try:
            got = self.collection.get(include=["embeddings"], limit=1)
            embs = got.get("embeddings", None)
            if embs is None:
                return False
            return len(embs) > 0 and len(embs[0]) > 0
        except Exception:
            return False

    def build_where(self, grupo: str, maquina_id: Optional[Any] = None) -> Optional[dict]:
        if maquina_id is None:
            return {"grupo": grupo}
        return {"maquina_id": str(maquina_id), "grupo": grupo}

    def post_filter_group(self, docs: List[str], metas: List[dict], grupo: str):
        out_d, out_m = [], []
        for d, m in zip(docs, metas):
            if isinstance(m, dict) and m.get("grupo") == grupo:
                out_d.append(d)
                out_m.append(m)
        return out_d, out_m

    # ------------------------------------
    # keyword substring scan
    # ------------------------------------
    def keyword_substring_scan(
        self,
        query: str,
        *,
        where: Optional[dict] = None,
        max_docs: int = 2200,
        topk: int = 40,
        substring_min_hits: int = 1,
        post_filter_group: bool = False,
        grupo: Optional[str] = None,
    ) -> List[Tuple[float, str, dict]]:
        if not self.collection:
            return []

        q = (query or "").strip().lower()
        if not q:
            return []

        # get docs for scan
        try:
            got = self.collection.get(where=where, include=["documents", "metadatas"], limit=max_docs) if where else \
                self.collection.get(
                    include=["documents", "metadatas"], limit=max_docs)
        except Exception:
            try:
                got = self.collection.get(
                    include=["documents", "metadatas"], limit=max_docs)
            except Exception:
                return []

        docs = got.get("documents", []) or []
        metas = got.get("metadatas", []) or []

        if post_filter_group and grupo:
            docs, metas = self.post_filter_group(docs, metas, grupo)

        q_terms = [t for t in re.findall(
            r"[a-zA-Z0-9_/%\.-]{2,}", q) if len(t) >= 3]
        q_set = set(q_terms)

        ranked = []
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

            token_hits = sum(1 for t in q_set if t in low)
            hits += float(token_hits)

            if hits < substring_min_hits:
                continue

            boost = self.filters.area_boost(
                txt) + float((m or {}).get("area_boost", 0.0) or 0.0)
            dens = self.filters.tech_density(txt)
            useful = self.filters.usefulness_score(txt)

            score = hits + 0.06 * boost + 2.4 * dens + 1.7 * useful

            if len(txt.split()) < 10:
                continue

            ranked.append((score, txt, m or {}))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[:topk]

    # ------------------------------------
    # Dedupe / diversify
    # ------------------------------------
    def dedupe_and_diversify(self, docs, metas, max_per_page=1, max_total=12):
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

    # ------------------------------------
    # MAIN
    # ------------------------------------
    def retrieve(
        self,
        *,
        descricao_problema: str,
        grupo: str,
        maquina_id: Optional[Any] = None,
        retrieval_topk: int = 180,
        keyword_topk: int = 40,
        final_k: int = 16,
        mmr_lambda: float = 0.76,
        mmr_max_total: int = 32,
        min_evidence: int = 5,
        enable_ladder: bool = True,
        debug: bool = False,
    ) -> Dict[str, Any]:

        if not self.collection or not self.embedder:
            return {"docs": [], "metas": [], "scores": [], "relatorio": {}}

        query_enriched = (
            f"{descricao_problema}\n"
            f"Prioridade: forno, DICU, fusão, Prodapt MD, DISA, moldagem, fundição.\n"
            f"Busque: parâmetros, limites, alarmes, diagnóstico, procedimentos, checklist.\n"
            f"Considere intertravamentos, permissivas, sequência, sinais e valores."
        )

        query_emb = self.embedder.encode([query_enriched]).tolist()[0]
        has_embs = self.collection_has_embeddings()

        where_maquina = self.build_where(
            grupo, maquina_id) if maquina_id is not None else None
        where_grupo = {"grupo": grupo}

        used_try = None
        ladder_level = None
        ladder_bonus = 1.0

        # ------------------------------------
        # 1) LADDER
        # ------------------------------------
        merged_docs, merged_metas = [], []
        ranked_comp = []

        if enable_ladder and self.ladder:
            pack = self.ladder.run(
                query=descricao_problema,
                query_emb=query_emb,
                maquina_id=maquina_id,
                grupo=grupo,
                where_maquina=where_maquina,
                where_grupo=where_grupo,
                keyword_topk=keyword_topk,
                substring_scan_cfg={
                    "max_docs": 2200,
                    "topk": 40,
                    "substring_min_hits": 1,
                    "grupo": grupo,
                },
                reranker=self.reranker,
                min_evidence=min_evidence,
                debug=debug,
            )
            if pack:
                ladder_level = pack.get("level")
                ladder_bonus = float(pack.get("bonus", 1.0) or 1.0)
                used_try = f"ladder:{ladder_level}"

                items = pack.get("items") or []
                merged_docs = [d for _, d, _, _ in items]
                merged_metas = [m for _, _, m, _ in items]
                ranked_comp = [c for c, _, _, _ in items]

        # ------------------------------------
        # 2) FALLBACK
        # ------------------------------------
        if not merged_docs:
            tries_where = []
            if maquina_id is not None:
                tries_where.append(("maquina_id", where_maquina, True))
            tries_where.append(("grupo_only", where_grupo, False))
            tries_where.append(("none", None, False))

            docs, metas = [], []

            if has_embs:
                for tag, wf, post_group in tries_where:
                    res = self.safe_query(
                        query_emb=query_emb, n_results=retrieval_topk, where=wf)
                    d = res["documents"][0] if res.get("documents") else []
                    m = res["metadatas"][0] if res.get("metadatas") else []
                    if post_group and d:
                        d, m = self.post_filter_group(d, m, grupo)
                    if d:
                        docs, metas = d, m
                        used_try = f"embedding:{tag}"
                        break

            if not docs:
                for tag, wf, post_group in tries_where:
                    resk = self.safe_query(
                        query_text=descricao_problema, n_results=max(keyword_topk, 18), where=wf)
                    kd = resk["documents"][0] if resk.get("documents") else []
                    km = resk["metadatas"][0] if resk.get("metadatas") else []
                    if post_group and kd:
                        kd, km = self.post_filter_group(kd, km, grupo)
                    if kd:
                        docs, metas = kd, km
                        used_try = f"keyword_query:{tag}"
                        break

            substring_ranked = []
            for tag, wf, post_group in tries_where:
                substring_ranked = self.keyword_substring_scan(
                    descricao_problema,
                    where=wf,
                    max_docs=2200,
                    topk=40,
                    substring_min_hits=1,
                    post_filter_group=post_group,
                    grupo=grupo,
                )
                if substring_ranked:
                    if used_try is None:
                        used_try = f"substring_scan:{tag}"
                    break

            if not docs and not substring_ranked:
                return {"docs": [], "metas": [], "scores": [], "relatorio": {}}

            # filtros básicos
            filtered = []
            for d, m in zip(docs, metas):
                if not d:
                    continue
                txt = (d or "").strip()
                if self.filters.chunk_is_bad(txt):
                    continue
                dens = self.filters.tech_density(txt)
                useful = self.filters.usefulness_score(txt)
                boost = self.filters.area_boost(
                    txt) + float((m or {}).get("area_boost", 0.0) or 0.0)
                filtered.append((dens + 0.03 * boost + 0.8 * useful, txt, m))
            filtered.sort(key=lambda x: x[0], reverse=True)

            docs_f = [d for _, d, _ in filtered][:110]
            metas_f = [m for _, _, m in filtered][:110]

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

            # rerank + score composto
            rerank_scores = None
            if self.reranker and self.reranker.available():
                rerank_scores = self.reranker.score(
                    descricao_problema, merged_docs)

            composite = []
            for i, (d, m) in enumerate(zip(merged_docs, merged_metas)):
                rr = rerank_scores[i] if (
                    rerank_scores and i < len(rerank_scores)) else None
                comp = self.filters.composite_score(
                    d, m, rr, descricao_problema, level_bonus=1.0)
                composite.append((comp, d, m, rr))
            composite.sort(key=lambda x: x[0], reverse=True)

            merged_docs = [d for _, d, _, _ in composite]
            merged_metas = [m for _, _, m, _ in composite]
            ranked_comp = [c for c, _, _, _ in composite]

        # ------------------------------------
        # 3) DIVERSIFICA + MMR
        # ------------------------------------
        ranked_docs = merged_docs[:]
        ranked_metas = merged_metas[:]
        ranked_comp = ranked_comp[: len(ranked_docs)]

        ranked_docs, ranked_metas = self.dedupe_and_diversify(
            ranked_docs, ranked_metas, max_per_page=1, max_total=mmr_max_total
        )

        # MMR
        if self.mmr_select_fn:
            try:
                doc_embs = self.embedder.encode(ranked_docs).tolist()
                ranked_docs, ranked_metas, _ = self.mmr_select_fn(
                    query_emb, ranked_docs, ranked_metas, doc_embs,
                    lambda_=mmr_lambda,
                    max_total=mmr_max_total,
                )
            except Exception:
                pass

        ranked_docs, ranked_metas = self.dedupe_and_diversify(
            ranked_docs, ranked_metas, max_per_page=1, max_total=final_k
        )

        # scores finais alinhados por prefixo
        final_scores = []
        if ranked_comp:
            score_map = {}
            for d, sc in zip(merged_docs, ranked_comp):
                if isinstance(d, str):
                    score_map[d[:160]] = sc
            for d in ranked_docs:
                final_scores.append(score_map.get(d[:160], None))
        else:
            final_scores = [None] * len(ranked_docs)

        relatorio = {
            "used_try": used_try,
            "ladder_level": ladder_level,
            "ladder_bonus": ladder_bonus,
            "has_embeddings": has_embs,
            "pre_total_candidates": len(merged_docs),
            "final_k": len(ranked_docs),
        }

        return {
            "docs": ranked_docs,
            "metas": ranked_metas,
            "scores": final_scores,
            "relatorio": relatorio,
        }
