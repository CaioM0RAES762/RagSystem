from typing import List, Tuple
import math


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) + 1e-9
    nb = math.sqrt(sum(x * x for x in b)) + 1e-9
    return dot / (na * nb)


def mmr_select(
    query_emb: List[float],
    docs: List[str],
    metas: List[dict],
    doc_embs: List[List[float]],
    *,
    lambda_: float = 0.76,
    max_total: int = 20,
) -> Tuple[List[str], List[dict], List[List[float]]]:
    """
    Seleção MMR clássica.
    """
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
            sim_q = cosine(query_emb, doc_embs[i])
            sim_sel = max(cosine(doc_embs[i], doc_embs[j])
                          for j in selected_idx)
            mmr = lambda_ * sim_q - (1 - lambda_) * sim_sel

            if mmr > best_score:
                best_score = mmr
                best = i

        selected_idx.append(best)
        remaining.remove(best)

    out_docs = [docs[i] for i in selected_idx]
    out_metas = [metas[i] for i in selected_idx]
    out_embs = [doc_embs[i] for i in selected_idx]
    return out_docs, out_metas, out_embs
