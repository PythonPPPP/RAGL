from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from app.rag.embeddings import embed_texts
from app.rag.pipeline import PipelineConfig, answer


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings are normalized
    return float(np.dot(a, b))


def _split_sentences(text: str) -> List[str]:
    # lightweight sentence splitter
    raw = text.replace("\n", " ").strip()
    if not raw:
        return []
    seps = [".", "!", "?", ";"]
    out: List[str] = []
    buf = ""
    for ch in raw:
        buf += ch
        if ch in seps and len(buf) > 10:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out[:12]


def score_one(
    index_dir: str | Path,
    question: str,
    cfg: PipelineConfig,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    idx_dir = Path(index_dir)
    # Run pipeline
    t0 = time.time()
    out = answer(index_dir=idx_dir, question=question, cfg=cfg)
    latency = float(out["timings"]["total"])

    # Compute metrics
    q_emb = embed_texts([question], cfg.embedding_model)[0]
    a_text = out["answer"]
    a_emb = embed_texts([a_text], cfg.embedding_model)[0]

    rel = max(0.0, min(1.0, _cos(q_emb, a_emb)))

    ctx_texts = [s["text"] for s in out["sources"]]
    if ctx_texts:
        c_embs = embed_texts(ctx_texts, cfg.embedding_model)
        # Context precision: similarity of each retrieved chunk to the question
        sims = [max(0.0, min(1.0, float(np.dot(q_emb, c_embs[i])))) for i in range(len(ctx_texts))]
        ctx_precision = float(np.mean(sims))

        # Context recall (proxy): similarity of answer to best chunk
        ans_to_ctx = [max(0.0, min(1.0, float(np.dot(a_emb, c_embs[i])))) for i in range(len(ctx_texts))]
        ctx_recall = float(np.max(ans_to_ctx))

        # Faithfulness (proxy): sentence-wise support by any chunk
        sents = _split_sentences(a_text)
        if sents:
            s_embs = embed_texts(sents, cfg.embedding_model)
            best = []
            for i in range(len(sents)):
                best.append(float(np.max(np.dot(c_embs, s_embs[i]))))
            faith = float(np.mean([max(0.0, min(1.0, b)) for b in best]))
        else:
            faith = 0.0
    else:
        ctx_precision = 0.0
        ctx_recall = 0.0
        faith = 0.0

    # Latency inverse (normalize with soft cap)
    latency_inv = 1.0 / (1.0 + latency)

    # Composite score
    ragas_score = 0.35 * faith + 0.25 * rel + 0.2 * ctx_precision + 0.1 * ctx_recall + 0.1 * latency_inv

    def _finite01(x: float) -> float:
        if not np.isfinite(x):
            return 0.0
        return float(max(0.0, min(1.0, x)))

    metrics = {
        "faithfulness": _finite01(faith),
        "answer_relevancy": _finite01(rel),
        "context_precision": _finite01(ctx_precision),
        "context_recall": _finite01(ctx_recall),
        "latency_sec": latency,
        "latency_inv": float(0.0 if not np.isfinite(latency_inv) else latency_inv),
        "ragas_score": float(0.0 if not np.isfinite(ragas_score) else ragas_score),
    }

    sample = {
        "question": question,
        "answer": a_text,
        "sources": out["sources"],
        "timings": out["timings"],
    }

    return metrics, sample


def evaluate_dataset(
    index_dir: str | Path,
    dataset: List[Dict[str, Any]],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    idx_dir = Path(index_dir)
    all_metrics: List[Dict[str, float]] = []
    samples: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for row in dataset:
        q = (row.get("question") or "").strip()
        if not q:
            continue
        try:
            m, s = score_one(index_dir=idx_dir, question=q, cfg=cfg)
            all_metrics.append(m)
            samples.append(s)
        except Exception as e:
            errors.append({"question": q, "error": str(e)})
            continue

    if not all_metrics:
        return {
            "count": 0,
            "metrics": {},
            "samples": [],
            "errors": errors,
            "errors_count": len(errors),
            "error_rate": 1.0 if errors else 0.0,
        }

    keys = all_metrics[0].keys()
    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}

    total = len(all_metrics) + len(errors)
    return {
        "count": len(all_metrics),
        "metrics": avg,
        "samples": samples[:50],
        "errors": errors[:50],
        "errors_count": len(errors),
        "error_rate": (len(errors) / total) if total else 0.0,
    }
