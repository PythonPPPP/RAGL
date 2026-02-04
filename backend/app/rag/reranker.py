from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=4)
def _get_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(query: str, passages: List[str], model_name: str) -> List[float]:
    m = _get_reranker(model_name)
    pairs = [(query, p) for p in passages]
    scores = m.predict(pairs)
    return [float(s) for s in scores]
