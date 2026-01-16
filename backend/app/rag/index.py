from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None
import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.chunking import Chunk


def _tokenize(text: str) -> List[str]:
    """Very simple tokenization for BM25."""
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    tokens = [t for t in cleaned.split() if len(t) > 1]
    return tokens


def build_indexes(
    index_dir: Path,
    chunks: List[Chunk],
    embeddings: np.ndarray,
    metadata: Dict,
) -> None:
    """Persist FAISS dense index + BM25 index + chunk metadata."""
    index_dir.mkdir(parents=True, exist_ok=True)

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    # Always persist raw embeddings for a portable fallback.
    np.save(str(index_dir / "embeddings.npy"), embeddings)

    # Optional FAISS index (faster), if available.
    if faiss is not None:
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, str(index_dir / "dense.index"))

    chunk_dicts = [asdict(c) for c in chunks]
    (index_dir / "chunks.json").write_text(json.dumps(chunk_dicts, ensure_ascii=False), encoding="utf-8")

    tokenized = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with (index_dir / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)

    (index_dir / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_indexes(index_dir: Path):
    """Load FAISS + BM25 + chunk metadata."""
    # Always load raw embeddings (portable + required for MMR).
    emb_path = index_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError("No embeddings found. Rebuild the index.")
    embeddings = np.load(str(emb_path))

    # Prefer FAISS if installed and index file exists.
    faiss_index = None
    dense_path = index_dir / "dense.index"
    if faiss is not None and dense_path.exists():
        faiss_index = faiss.read_index(str(dense_path))
    else:
        # Fallback: use embeddings matrix for dense search.
        faiss_index = embeddings
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    with (index_dir / "bm25.pkl").open("rb") as f:
        bm25_obj = pickle.load(f)
    return faiss_index, embeddings, bm25_obj["bm25"], bm25_obj["tokenized"], chunks, meta


def dense_search(
    faiss_index,
    query_vec: np.ndarray,
    top_k: int,
) -> Tuple[List[int], List[float]]:
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    # FAISS path
    if hasattr(faiss_index, "search"):
        scores, idxs = faiss_index.search(query_vec.astype(np.float32), top_k)
        return idxs[0].tolist(), scores[0].tolist()

    # Numpy fallback: embeddings @ query
    emb = np.asarray(faiss_index, dtype=np.float32)  # (N, D)
    q = query_vec.astype(np.float32)[0]
    scores = emb @ q
    if emb.shape[0] == 0:
        return [], []
    idxs = np.argsort(-scores)[:top_k]
    return idxs.tolist(), [float(scores[i]) for i in idxs]


def bm25_search(
    bm25: BM25Okapi,
    query_text: str,
    top_k: int,
) -> Tuple[List[int], List[float]]:
    q = _tokenize(query_text)
    scores = bm25.get_scores(q)
    idxs = np.argsort(-scores)[:top_k]
    return idxs.tolist(), [float(scores[i]) for i in idxs]


def hybrid_merge(
    dense: Tuple[List[int], List[float]],
    sparse: Tuple[List[int], List[float]],
    alpha: float = 0.6,
) -> Tuple[List[int], List[float]]:
    """Merge dense and sparse ranked lists with min-max normalization."""
    d_i, d_s = dense
    s_i, s_s = sparse

    def norm(vals: List[float]) -> List[float]:
        if not vals:
            return vals
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-9:
            return [1.0 for _ in vals]
        return [(v - vmin) / (vmax - vmin) for v in vals]

    d_sn = norm(d_s)
    s_sn = norm(s_s)

    merged: Dict[int, float] = {}
    for idx, sc in zip(d_i, d_sn):
        merged[idx] = merged.get(idx, 0.0) + alpha * sc
    for idx, sc in zip(s_i, s_sn):
        merged[idx] = merged.get(idx, 0.0) + (1.0 - alpha) * sc

    items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    idxs = [i for i, _ in items]
    scores = [float(s) for _, s in items]
    return idxs, scores
