from typing import Optional, Any, Dict, List, Tuple
import re


class RetrievalLadder:
    """
    Implementa o ladder regressivo:
    - níveis rígidos -> relaxados
    - tenta embedding query / keyword query / substring scan
    - aplica filtros e bônus
    """

    def __init__(
        self,
        *,
        levels: List[Dict[str, Any]],
        filters,
        keyword_scan_fn,
        safe_query_fn,
        has_embeddings_fn,
    ):
        self.levels = levels
        self.filters = filters
        self.keyword_scan_fn = keyword_scan_fn
        self.safe_query_fn = safe_query_fn
        self.has_embeddings_fn = has_embeddings_fn

    def _post_filter_group(self, docs, metas, grupo: str):
        out_d, out_m = [], []
        for d, m in zip(docs, metas):
            if isinstance(m, dict) and m.get("grupo") == grupo:
                out_d.append(d)
                out_m.append(m)
        return out_d, out_m

    def run(
        self,
        *,
        query: str,
        query_emb,
        maquina_id: Optional[Any],
        grupo: str,
        where_maquina: Optional[dict],
        where_grupo: dict,
        keyword_topk: int,
        substring_scan_cfg: dict,
        reranker=None,
        min_evidence: int = 5,
        debug: bool = False,
    ) -> Optional[Dict[str, Any]]:

        has_embs = bool(self.has_embeddings_fn())

        for level in self.levels:
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

            # embedding retrieval
            if has_embs:
                res = self.safe_query_fn(
                    query_emb=query_emb, n_results=topk, where=wf)
                docs = res["documents"][0] if res.get("documents") else []
                metas = res["metadatas"][0] if res.get("metadatas") else []
                if post_group and docs:
                    docs, metas = self._post_filter_group(docs, metas, grupo)

            # keyword query_texts
            if not docs:
                resk = self.safe_query_fn(
                    query_text=query, n_results=max(20, keyword_topk), where=wf)
                docs = resk["documents"][0] if resk.get("documents") else []
                metas = resk["metadatas"][0] if resk.get("metadatas") else []
                if post_group and docs:
                    docs, metas = self._post_filter_group(docs, metas, grupo)

            # substring scan (sempre complementa)
            substring_ranked = self.keyword_scan_fn(
                query,
                where=wf,
                **substring_scan_cfg,
                post_filter_group=post_group,
            )

            # filtros
            filtered = []
            for d, m in zip(docs, metas):
                if not d:
                    continue
                txt = (d or "").strip()
                if len(txt.split()) < min_words:
                    continue
                if self.filters.chunk_is_bad(txt):
                    continue
                dens = self.filters.tech_density(txt)
                if dens < min_density:
                    continue
                useful = self.filters.usefulness_score(txt)
                boost = self.filters.area_boost(
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
            if reranker and reranker.available():
                rerank_scores = reranker.score(query, merged_docs)

            composite = []
            for i, (d, m) in enumerate(zip(merged_docs, merged_metas)):
                rr = rerank_scores[i] if (
                    rerank_scores and i < len(rerank_scores)) else None
                comp = self.filters.composite_score(
                    d, m, rr, query, level_bonus=bonus)
                composite.append((comp, d, m, rr))

            composite.sort(key=lambda x: x[0], reverse=True)

            # gate score mínimo
            composite = [x for x in composite if x[0] >= score_min]
            if len(composite) < min_evidence:
                continue

            return {
                "level": name,
                "bonus": bonus,
                "items": composite,
                "where_mode": where_mode,
            }

        return None
