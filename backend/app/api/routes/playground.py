from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from pydantic import ConfigDict
from sqlmodel import Session

from app.api.deps import db_session
from app.db.models import Pipeline
from app.rag.pipeline import PipelineConfig, RetrievedChunk, retrieve
from app.rag.generator import generate_text_with_stats
from app.services.storage import indexes_dir

router = APIRouter()


class RetrieveRequest(BaseModel):
    project_id: int
    pipeline_id: int
    question: str
    override_model_id: Optional[str] = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    prompt: str
    temperature: float = 0.1
    max_new_tokens: int = 256
    top_p: float = 0.9


def _derive_queries(question: str, cfg: PipelineConfig) -> List[str]:
    # Keep logic aligned with app.rag.pipeline.retrieve
    def _stopwords() -> set[str]:
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
        seen = set()
        out: List[str] = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return " ".join(out[:12])

    queries: List[str] = [question]
    mode = cfg.query.mode if cfg.query else "none"
    if mode == "keywords":
        kq = _keywords(question)
        if kq:
            queries = [kq]
    elif mode == "multi":
        kq = _keywords(question)
        compact = " ".join([t for t in question.split() if len(t) > 2])
        candidates = [question, kq, compact]
        queries = [q for q in candidates if q and q.strip()]
        queries = queries[: max(1, int(cfg.query.max_queries or 3))]
    return queries


@router.post("/retrieve")
def playground_retrieve(payload: RetrieveRequest, session: Session = Depends(db_session)):
    p = session.get(Pipeline, payload.pipeline_id)
    if not p or p.project_id != payload.project_id:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    cfg_dict = json.loads(p.config_json or "{}")
    if payload.override_model_id:
        cfg_dict.setdefault("generator", {})
        cfg_dict["generator"]["model_id"] = payload.override_model_id
    cfg = PipelineConfig(**cfg_dict)

    idx_dir = indexes_dir(payload.project_id) / f"pipeline_{payload.pipeline_id}"
    if not idx_dir.exists():
        raise HTTPException(status_code=400, detail="Index not built for this pipeline")

    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    retrieved, timings = retrieve(idx_dir, q, cfg)

    # Build context preview (same logic as answer)
    blocks: List[str] = []
    for i, rc in enumerate(retrieved, start=1):
        filename = rc.chunk.get("filename", "")
        text = (rc.chunk.get("text", "") or "").strip()
        block = f"[{i}] ({filename})\n{text}"
        candidate = "\n\n".join(blocks + [block])
        if len(candidate) > int(cfg.context.max_chars):
            break
        blocks.append(block)
    context = "\n\n".join(blocks)

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
        "ok": True,
        "queries": _derive_queries(q, cfg),
        "context_preview": context[:4000],
        "context_chars": len(context),
        "sources": sources,
        "timings": timings,
    }


@router.post("/generate")
def playground_generate(payload: GenerateRequest):
    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")

    out = generate_text_with_stats(
        model_id=payload.model_id,
        prompt=prompt,
        temperature=payload.temperature,
        max_new_tokens=payload.max_new_tokens,
        top_p=payload.top_p,
    )
    return {
        "ok": True,
        "text": out.get("text", ""),
        "tokens": {
            "input": int(out.get("input_tokens", 0) or 0),
            "output": int(out.get("output_tokens", 0) or 0),
        },
        "cost_usd": 0.0,
    }
