# rag/generation/report.py

from typing import Dict, Any, List


def gerar_relatorio_trechos(
    *,
    descricao_problema: str,
    maquina_id: Any,
    grupo: str,
    trechos: List[str],
    metas: List[dict],
    scores: List[Any],
    relatorio: Dict[str, Any],
    tech_density_fn,
    usefulness_score_fn,
    area_boost_fn,
) -> Dict[str, Any]:
    """
    Gera relatório JSON detalhado dos trechos selecionados.
    Você passa as funções do RAGSystem para calcular densidade/utilidade/boost.
    """

    itens = []
    for i in range(len(trechos)):
        trecho = trechos[i]
        meta = metas[i] if i < len(metas) else {}
        sc = scores[i] if i < len(scores) else None

        dens = tech_density_fn(trecho)
        useful = usefulness_score_fn(trecho)
        boost = area_boost_fn(trecho)

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
            "grupo": grupo,
        },
        "relatorio": relatorio,
        "itens": itens
    }
