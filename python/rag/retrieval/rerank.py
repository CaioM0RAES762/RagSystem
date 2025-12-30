from typing import List, Optional


class CrossEncoderReranker:
    """
    Wrapper seguro para CrossEncoder.
    Se não tiver o pacote/modelo, roda sem rerank.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None

        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None

    def available(self) -> bool:
        return self.model is not None

    def score(self, query: str, docs: List[str]) -> Optional[List[float]]:
        if not self.model or not docs:
            return None

        try:
            pairs = [(query, d) for d in docs]
            out = self.model.predict(pairs)
            return out.tolist() if hasattr(out, "tolist") else list(out)
        except Exception:
            return None
