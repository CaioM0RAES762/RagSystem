# rag/generation/vision_describe.py

import base64
from typing import Optional


def gerar_descricao_imagem(
    openai_client,
    *,
    image_bytes: bytes,
    system_prompt: str,
    vision_model: str,
    nome_maquina: str = "",
    maquina_id: str = "",
    temperature: float = 0.2,
    max_tokens: int = 650,
) -> str:
    """
    Gera descrição técnica para uma imagem (página renderizada do manual).
    Retorna texto técnico (procedimentos, parâmetros, alarmes, limites, componentes etc).
    """

    if not openai_client:
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
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": contexto_texto},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        },
    ]

    resp = openai_client.chat.completions.create(
        model=vision_model,
        messages=mensagens,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = resp.choices[0].message.content
    descricao = content.strip() if isinstance(content, str) else (
        str(content).strip() if content else "")
    return descricao or "Falha ao gerar descricao tecnica da imagem."
