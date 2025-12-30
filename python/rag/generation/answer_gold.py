# rag/generation/answer_gold.py

import re
from typing import List, Tuple, Optional, Dict, Any


def citations_ok(text: str, min_citations_required: int = 7) -> bool:
    if not text or "[Fonte" not in text:
        return False

    n = len(re.findall(r"\[Fonte\s+\d+\]", text))
    if n < min_citations_required:
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


def _fonte_card(idx: int, trecho: str, meta: dict, score: Optional[float]) -> str:
    meta = meta if isinstance(meta, dict) else {}

    tipo = meta.get("tipo", "texto")
    page = meta.get("page", "?")
    maquina_id = meta.get("maquina_id", "?")
    nome = meta.get("nome_maquina", "")
    subchunk = meta.get("subchunk", None)

    txt = re.sub(r"\s+", " ", (trecho or "")).strip()
    if len(txt) > 820:
        txt = txt[:820].rstrip() + "..."

    head = f"[Fonte {idx}] (tipo={tipo}, page={page}, maquina_id={maquina_id}, score={score})"
    if nome:
        head += f" | maquina={nome}"
    if subchunk is not None:
        head += f" | subchunk={subchunk}"

    return f"{head}\n{txt}\n"


def gerar_solucao_texto_gold(
    *,
    openai_client,
    descricao_problema: str,
    fontes: List[Tuple[str, dict, Optional[float]]],
    casos_similares: Optional[List[Dict[str, Any]]] = None,
    coverage: Optional[Dict[str, int]] = None,
    gpt_model_mini: str,
    enable_two_pass: bool = True,
    two_pass_temperature: float = 0.14,
    two_pass_max_tokens: int = 1250,
    min_citations_required: int = 7,
) -> Dict[str, Any]:
    """
    Gera resposta GOLD com:
    - PASS 1: solução completa citando [Fonte X]
    - PASS 2: auditor agressivo revisa e garante padrão ouro

    Retorna dict com "solucao" e "modelo_usado"
    """

    if not openai_client:
        return {"erro": "OpenAI client nao disponivel."}

    contexto = ""
    for i, (trecho, meta, sc) in enumerate(fontes, 1):
        contexto += _fonte_card(i, trecho, meta, sc)

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
        # -------- PASS 1 --------
        resp1 = openai_client.chat.completions.create(
            model=gpt_model_mini,
            messages=[
                {"role": "system",
                    "content": "Você é um especialista em manutenção. Você DEVE citar fontes [Fonte X]."},
                {"role": "user", "content": prompt_answer},
            ],
            temperature=0.20,
            max_tokens=two_pass_max_tokens if enable_two_pass else 980,
        )
        answer1 = (resp1.choices[0].message.content or "").strip()

        if not enable_two_pass:
            if not citations_ok(answer1, min_citations_required=min_citations_required):
                answer1 = "⚠️ Resposta com citações insuficientes. Aumente top-k ou informe alarme/código.\n\n" + answer1
            return {"solucao": answer1, "modelo_usado": gpt_model_mini}

        # -------- PASS 2 (AUDITOR) --------
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

        resp2 = openai_client.chat.completions.create(
            model=gpt_model_mini,
            messages=[
                {"role": "system",
                    "content": "Você é um auditor técnico. Reescreva para padrão ouro e mantenha citações [Fonte X]."},
                {"role": "user", "content": prompt_critic},
            ],
            temperature=two_pass_temperature,
            max_tokens=two_pass_max_tokens,
        )
        final = (resp2.choices[0].message.content or "").strip()

        if not citations_ok(final, min_citations_required=min_citations_required):
            final = "⚠️ Resultado final com densidade de citações abaixo do padrão. Forneça código/alarme ou aumente top-k.\n\n" + final

        return {"solucao": final, "modelo_usado": gpt_model_mini, "refino": True}

    except Exception as e:
        return {"erro": str(e)}
