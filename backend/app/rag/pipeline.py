from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from pydantic import ConfigDict


class QueryConfig(BaseModel):
    """Controls query rewriting/expansion before retrieval."""

    mode: str = Field(default="none", description="none|keywords|multi")
    max_queries: int = 3


class ContextConfig(BaseModel):
    """Controls how context is assembled."""

    max_chars: int = 8000


class GuardrailsConfig(BaseModel):
    """Lightweight post-processing / safety constraints for local runs."""

    require_citations: bool = True
    min_citations: int = 1


from app.rag.chunking import Chunk, make_chunks
from app.rag.embeddings import embed_texts
from app.rag.generator import build_prompt, generate_text_with_stats, load_llm
from app.rag.index import (
    build_indexes,
    dense_search,
    bm25_search,
    hybrid_merge,
    load_indexes,
)
from app.rag.reranker import rerank


class RetrieverConfig(BaseModel):
    mode: str = Field(default="hybrid", description="dense|bm25|hybrid")
    top_k: int = 16
    hybrid_alpha: float = 0.6
    use_reranker: bool = True
    rerank_top_n: int = 6
    min_score: float | None = None
    use_mmr: bool = False
    mmr_lambda: float = 0.6
    mmr_k: int = 30


class ChunkerConfig(BaseModel):
    chunk_size: int = 900
    overlap: int = 120


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    temperature: float = 0.1
    max_new_tokens: int = 256
    top_p: float = 0.9
    system_prompt: str = (
        "Ты ассистент, отвечающий на вопросы по документам пользователя. "
        "Если ответа нет в контексте — честно скажи, что данных в документах недостаточно."
    )


class PipelineConfig(BaseModel):
    name: str = "Pipeline"
    description: str = ""
    chunker: ChunkerConfig = ChunkerConfig()
    # Defaults are intentionally multilingual-friendly.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    query: QueryConfig = QueryConfig()
    context: ContextConfig = ContextConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    generator: GeneratorConfig

    # UI-only metadata that the backend ignores in logic.
    ui: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class RetrievedChunk:
    idx: int
    score: float
    chunk: Dict[str, Any]


def build_index_for_docs(
    index_dir: Path,
    docs: List[Tuple[int, str, str]],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """Build FAISS and BM25 indexes from docs.

    docs: list of (doc_id, filename, text)
    """
    chunks: List[Chunk] = []
    for doc_id, filename, text in docs:
        chunks.extend(make_chunks(doc_id, filename, text, cfg.chunker.chunk_size, cfg.chunker.overlap))

    texts = [c.text for c in chunks]
    t0 = time.time()
    embs = embed_texts(texts, cfg.embedding_model)
    build_time = time.time() - t0

    meta = {
        "pipeline": cfg.model_dump(mode="json"),
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "build_time_sec": build_time,
        "created_at": time.time(),
    }
    build_indexes(index_dir, chunks, embs, meta)
    return meta


def retrieve(
    index_dir: Path,
    question: str,
    cfg: PipelineConfig,
) -> Tuple[List[RetrievedChunk], Dict[str, float]]:
    """Retrieve chunks and return timings."""
    t = {}
    t0 = time.time()
    faiss_index, embeddings, bm25, tokenized, chunks, meta = load_indexes(index_dir)
    t["load_index"] = time.time() - t0

    # --- Query enhancement ---
    def _stopwords() -> set[str]:
        # lightweight bilingual list (en/ru) for keyword mode
        return {
            "a",
            "an",
            "the",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "is",
            "are",
            "was",
            "were",
            "be",
            "как",
            "что",
            "это",
            "я",
            "мы",
            "вы",
            "он",
            "она",
            "они",
            "и",
            "или",
            "в",
            "на",
            "по",
            "для",
            "с",
            "о",
            "об",
            "у",
            "к",
            "из",
            "не",
        }

    def _keywords(q: str) -> str:
        toks = [t.lower() for t in "".join(ch if ch.isalnum() else " " for ch in q).split()]
        toks = [t for t in toks if len(t) > 2 and t not in _stopwords()]
        # keep order, unique
        seen = set()
        out = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return " ".join(out[:12])

    queries: List[str] = [question]
    if cfg.query.mode == "keywords":
        kq = _keywords(question)
        if kq:
            queries = [kq]
    elif cfg.query.mode == "multi":
        kq = _keywords(question)
        compact = " ".join([t for t in question.split() if len(t) > 2])
        candidates = [question, kq, compact]
        queries = [q for q in candidates if q and q.strip()]
        queries = queries[: max(1, cfg.query.max_queries)]

    # --- Embed (dense) queries ---
    t1 = time.time()
    qvs = embed_texts(queries, cfg.embedding_model)
    t["embed_query"] = time.time() - t1

    top_k = max(cfg.retriever.top_k, cfg.retriever.rerank_top_n, cfg.retriever.mmr_k)

    # --- Retrieval for each query; merge by max normalized score ---
    t2 = time.time()

    def _minmax_norm(vals: List[float]) -> List[float]:
        if not vals:
            return vals
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-9:
            return [1.0 for _ in vals]
        return [(v - vmin) / (vmax - vmin) for v in vals]

    merged_scores: Dict[int, float] = {}

    for qv, q in zip(qvs, queries):
        dense = dense_search(faiss_index, qv, top_k=top_k)
        sparse = bm25_search(bm25, q, top_k=top_k)

        if cfg.retriever.mode == "dense":
            idxs, scores = dense
        elif cfg.retriever.mode == "bm25":
            idxs, scores = sparse
        else:
            idxs, scores = hybrid_merge(dense, sparse, alpha=cfg.retriever.hybrid_alpha)

        scores_n = _minmax_norm(scores)
        for idx, sc in zip(idxs, scores_n):
            merged_scores[idx] = max(merged_scores.get(idx, 0.0), float(sc))

    items = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    idxs = [i for i, _ in items]
    scores = [float(s) for _, s in items]

    # Optional score filtering
    if cfg.retriever.min_score is not None:
        keep = [(i, s) for i, s in zip(idxs, scores) if s >= float(cfg.retriever.min_score)]
        idxs = [i for i, _ in keep]
        scores = [float(s) for _, s in keep]

    t["retrieve"] = time.time() - t2

    # Build candidate chunks
    cand_idxs = idxs[: cfg.retriever.top_k]

    # Optional MMR diversification
    if cfg.retriever.use_mmr and len(cand_idxs) > 0:
        try:
            q = np.asarray(qvs[0], dtype=np.float32)
            # Normalize
            q = q / (np.linalg.norm(q) + 1e-9)
            X = np.asarray([embeddings[i] for i in idxs[: cfg.retriever.mmr_k]], dtype=np.float32)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            base_idxs = idxs[: cfg.retriever.mmr_k]
            pos = {idx: j for j, idx in enumerate(base_idxs)}
            sim_q = (X @ q).tolist()

            selected: List[int] = []
            selected_set = set()
            lam = float(np.clip(cfg.retriever.mmr_lambda, 0.0, 1.0))

            # greedily pick items maximizing MMR objective
            while len(selected) < min(cfg.retriever.top_k, len(base_idxs)):
                best_i = None
                best_val = -1e9
                for j, idx in enumerate(base_idxs):
                    if idx in selected_set:
                        continue
                    if not selected:
                        val = sim_q[j]
                    else:
                        # max similarity to already selected
                        smax = max((float(X[j] @ X[pos[si]]) for si in selected if si in pos), default=0.0)
                        val = lam * sim_q[j] - (1.0 - lam) * smax
                    if val > best_val:
                        best_val = val
                        best_i = idx
                if best_i is None:
                    break
                selected.append(best_i)
                selected_set.add(best_i)
            cand_idxs = selected
        except Exception:
            # fall back silently
            pass

    candidates = [chunks[i] for i in cand_idxs]
    cand_texts = [c["text"] for c in candidates]
    cand_scores = [merged_scores.get(i, 0.0) for i in cand_idxs]

    if cfg.retriever.use_reranker:
        t3 = time.time()
        rr_scores = rerank(question, cand_texts, cfg.reranker_model)
        order = sorted(range(len(rr_scores)), key=lambda i: rr_scores[i], reverse=True)
        order = order[: cfg.retriever.rerank_top_n]
        out = [RetrievedChunk(idx=i, score=float(rr_scores[i]), chunk=candidates[i]) for i in order]
        t["rerank"] = time.time() - t3
    else:
        out = [RetrievedChunk(idx=i, score=float(sc), chunk=ch) for i, (ch, sc) in enumerate(zip(candidates, cand_scores))]

    return out, t


def format_context(retrieved: List[RetrievedChunk]) -> str:
    blocks = []
    for i, rc in enumerate(retrieved, start=1):
        filename = rc.chunk.get("filename", "")
        text = rc.chunk.get("text", "").strip()
        blocks.append(f"[{i}] ({filename})\n{text}")
    return "\n\n".join(blocks)


def answer(
    index_dir: Path,
    question: str,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """Run full RAG pipeline."""
    t0 = time.time()
    retrieved, timings = retrieve(index_dir, question, cfg)

    # --- Prompt/context budget ---
    # Many local chat models have a small context window (e.g. 2048). If we exceed it,
    # generation quality degrades badly. We therefore cap the context by *tokens*.
    try:
        model, tokenizer, _device = load_llm(cfg.generator.model_id)
        window = (
            int(getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0)
            or int(getattr(tokenizer, "model_max_length", 0) or 0)
            or 2048
        )
        # Some tokenizers report a huge sentinel value; clamp to something sane.
        if window > 32768:
            window = 4096
        reserve = int(cfg.generator.max_new_tokens) + 96
        max_input_tokens = max(256, window - reserve)
    except Exception:
        tokenizer = None
        max_input_tokens = None
    # Context cap (chars + tokens)
    blocks: List[str] = []
    for i, rc in enumerate(retrieved, start=1):
        filename = rc.chunk.get("filename", "")
        text = (rc.chunk.get("text", "") or "").strip()
        block = f"[{i}] ({filename})\n{text}"

        candidate_context = "\n\n".join(blocks + [block])
        if len(candidate_context) > int(cfg.context.max_chars):
            break

        # Token budget check: make sure the *full* prompt still fits.
        if tokenizer is not None and max_input_tokens is not None:
            candidate_prompt = build_prompt(
                question=question,
                context=candidate_context,
                system_prompt=cfg.generator.system_prompt,
            )
            try:
                n_tokens = int(tokenizer(candidate_prompt, return_tensors="pt", truncation=False)["input_ids"].shape[-1])
                if n_tokens > int(max_input_tokens):
                    break
            except Exception:
                # If token counting fails, fall back to char-based capping.
                pass

        blocks.append(block)

    context = "\n\n".join(blocks)

    t1 = time.time()
    prompt = build_prompt(question=question, context=context, system_prompt=cfg.generator.system_prompt)
    timings["prompt"] = time.time() - t1

    t2 = time.time()
    gen = generate_text_with_stats(
        model_id=cfg.generator.model_id,
        prompt=prompt,
        temperature=cfg.generator.temperature,
        max_new_tokens=cfg.generator.max_new_tokens,
        top_p=cfg.generator.top_p,
    )
    text = str(gen.get("text", ""))
    tokens_in = int(gen.get("input_tokens", 0) or 0)
    tokens_out = int(gen.get("output_tokens", 0) or 0)
    timings["generate"] = time.time() - t2

    # Guardrails: ensure citations exist if requested
    if cfg.guardrails.require_citations:
        import re

        if not re.search(r"\[\d+\]", text):
            n = max(1, int(cfg.guardrails.min_citations))
            n = min(n, len(retrieved))
            if n > 0:
                cites = " ".join([f"[{i}]" for i in range(1, n + 1)])
                text = text.rstrip() + f"\n\nSources: {cites}"

    total = time.time() - t0

    sources = [
        {
            "rank": i + 1,
            "score": rc.score,
            "doc_id": rc.chunk.get("doc_id"),
            "filename": rc.chunk.get("filename"),
            "chunk_id": rc.chunk.get("chunk_id"),
            "text": rc.chunk.get("text"),
        }
        for i, rc in enumerate(retrieved)
    ]

    return {
        "answer": text,
        "sources": sources,
        "timings": {**timings, "total": total},
        "tokens": {"input": tokens_in, "output": tokens_out, "total": tokens_in + tokens_out},
        "cost_usd": 0.0,
        "pipeline": cfg.name,
        "prompt_preview": prompt[:2000],
        "context_chars": len(context),
    }
