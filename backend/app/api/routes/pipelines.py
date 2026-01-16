from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import db_session
from app.core.config import settings
from app.db.models import Document, Pipeline
from app.rag.loaders import load_text_from_file, normalize_text
from app.rag.pipeline import PipelineConfig, build_index_for_docs
from app.services import background
from app.services.storage import indexes_dir

router = APIRouter()


class PipelineCreate(BaseModel):
    project_id: int
    name: str
    description: str = ""
    config: Dict[str, Any]


class PipelineOut(BaseModel):
    id: int
    project_id: int
    name: str
    description: str


class PipelineDuplicateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@router.get("/templates")
def templates():
    base = {
        "embedding_model": settings.default_embed_model,
        "reranker_model": settings.default_rerank_model,
        "query": {"mode": "none", "max_queries": 3},
        "context": {"max_chars": 8000},
        "guardrails": {"require_citations": True, "min_citations": 1},
        "generator": {
            "model_id": settings.default_llm_model,
            "temperature": 0.1,
            "max_new_tokens": 256,
            "top_p": 0.9,
        },
    }

    naive_fast = {
        **base,
        "name": "Naive (Fast)",
        "description": "Dense retrieval + small top-k. No rerank.",
        "chunker": {"chunk_size": 1024, "overlap": 80},
        "retriever": {
            "mode": "dense",
            "top_k": 6,
            "use_reranker": False,
            "rerank_top_n": 0,
            "min_score": None,
            "use_mmr": False,
        },
    }

    hybrid_rerank = {
        **base,
        "name": "Hybrid + Rerank",
        "description": "Hybrid dense+BM25 retrieval + reranking for higher faithfulness.",
        "chunker": {"chunk_size": 900, "overlap": 120},
        "retriever": {
            "mode": "hybrid",
            "top_k": 16,
            "hybrid_alpha": 0.6,
            "use_reranker": True,
            "rerank_top_n": 6,
            "min_score": 0.15,
            "use_mmr": False,
        },
    }

    long_context = {
        **base,
        "name": "Long-context",
        "description": "Bigger chunks and higher context cap for long answers.",
        "chunker": {"chunk_size": 1400, "overlap": 150},
        "context": {"max_chars": 14000},
        "retriever": {
            "mode": "hybrid",
            "top_k": 18,
            "hybrid_alpha": 0.55,
            "use_reranker": True,
            "rerank_top_n": 8,
            "min_score": 0.10,
            "use_mmr": False,
        },
    }

    diverse_mmr = {
        **base,
        "name": "Hybrid + MMR (Diverse)",
        "description": "Diverse retrieval via MMR to reduce redundancy.",
        "chunker": {"chunk_size": 900, "overlap": 120},
        "query": {"mode": "multi", "max_queries": 3},
        "retriever": {
            "mode": "hybrid",
            "top_k": 16,
            "hybrid_alpha": 0.6,
            "use_reranker": True,
            "rerank_top_n": 6,
            "min_score": 0.10,
            "use_mmr": True,
            "mmr_lambda": 0.65,
            "mmr_k": 30,
        },
    }

    return {
        "naive": naive_fast,
        "advanced": hybrid_rerank,
        "long_context": long_context,
        "diverse": diverse_mmr,
    }


@router.get("", response_model=List[PipelineOut])
def list_pipelines(project_id: int, session: Session = Depends(db_session)):
    items = session.exec(select(Pipeline).where(Pipeline.project_id == project_id).order_by(Pipeline.id.desc())).all()
    return [PipelineOut(id=p.id, project_id=p.project_id, name=p.name, description=p.description) for p in items]


@router.post("", response_model=PipelineOut)
def create_pipeline(payload: PipelineCreate, session: Session = Depends(db_session)):
    # validate config
    cfg = PipelineConfig(**payload.config)
    p = Pipeline(
        project_id=payload.project_id,
        name=payload.name,
        description=payload.description,
        config_json=json.dumps(cfg.model_dump(mode="json"), ensure_ascii=False),
    )
    session.add(p)
    session.commit()
    session.refresh(p)
    return PipelineOut(id=p.id, project_id=p.project_id, name=p.name, description=p.description)


@router.get("/{pipeline_id}")
def get_pipeline(pipeline_id: int, session: Session = Depends(db_session)):
    p = session.get(Pipeline, pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return {
        "id": p.id,
        "project_id": p.project_id,
        "name": p.name,
        "description": p.description,
        "config": json.loads(p.config_json or "{}"),
        "index_ready": (indexes_dir(p.project_id) / f"pipeline_{p.id}").exists(),
    }


@router.put("/{pipeline_id}")
def update_pipeline(pipeline_id: int, payload: PipelineCreate, session: Session = Depends(db_session)):
    p = session.get(Pipeline, pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    cfg = PipelineConfig(**payload.config)
    p.name = payload.name
    p.description = payload.description
    p.config_json = json.dumps(cfg.model_dump(mode="json"), ensure_ascii=False)
    session.add(p)
    session.commit()
    return {"ok": True}


@router.delete("/{pipeline_id}")
def delete_pipeline(
    pipeline_id: int,
    delete_index: bool = Query(default=False, description="Also delete the built index for this pipeline"),
    session: Session = Depends(db_session),
):
    p = session.get(Pipeline, pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    proj = p.project_id
    session.delete(p)
    session.commit()

    if delete_index:
        idx_dir = indexes_dir(proj) / f"pipeline_{pipeline_id}"
        if idx_dir.exists():
            # best-effort removal
            import shutil

            shutil.rmtree(idx_dir, ignore_errors=True)

    return {"ok": True}


@router.post("/{pipeline_id}/duplicate", response_model=PipelineOut)
def duplicate_pipeline(pipeline_id: int, payload: PipelineDuplicateRequest, session: Session = Depends(db_session)):
    p = session.get(Pipeline, pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    name = payload.name or f"{p.name} (copy)"
    desc = payload.description if payload.description is not None else (p.description or "")
    dup = Pipeline(project_id=p.project_id, name=name, description=desc, config_json=p.config_json)
    session.add(dup)
    session.commit()
    session.refresh(dup)
    return PipelineOut(id=dup.id, project_id=dup.project_id, name=dup.name, description=dup.description)


@router.post("/{pipeline_id}/build_index")
def build_index(pipeline_id: int, session: Session = Depends(db_session)):
    p = session.get(Pipeline, pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    cfg = PipelineConfig(**json.loads(p.config_json))

    docs = session.exec(select(Document).where(Document.project_id == p.project_id)).all()
    if not docs:
        raise HTTPException(status_code=400, detail="No documents uploaded")

    loaded = []
    for d in docs:
        path = Path(d.original_path)
        if not path.exists():
            continue
        text = normalize_text(load_text_from_file(path))
        loaded.append((d.id, d.filename, text))

    if not loaded:
        raise HTTPException(status_code=400, detail="No readable documents")

    idx_dir = indexes_dir(p.project_id) / f"pipeline_{p.id}"
    meta = build_index_for_docs(idx_dir, loaded, cfg)

    # mark documents processed
    for d in docs:
        d.status = "processed"
        session.add(d)
    session.commit()

    return {"ok": True, "index_dir": str(idx_dir), "meta": meta}


@router.post("/{pipeline_id}/build_index_async")
def build_index_async(pipeline_id: int, session: Session = Depends(db_session)):
    p = session.get(Pipeline, pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    job_id = f"build_index:{p.project_id}:{p.id}"

    # Background service keeps a global pool; we re-open DB session in the job for thread safety
    from app.db.session import engine
    from sqlmodel import Session as SqlSession

    def _threadsafe_job():
        with SqlSession(engine) as s:
            # re-fetch inside the thread
            pp = s.get(Pipeline, pipeline_id)
            if not pp:
                raise RuntimeError("Pipeline not found")
            cfg = PipelineConfig(**json.loads(pp.config_json or "{}"))
            docs = s.exec(select(Document).where(Document.project_id == pp.project_id)).all()
            if not docs:
                raise RuntimeError("No documents uploaded")
            loaded = []
            for d in docs:
                path = Path(d.original_path)
                if not path.exists():
                    continue
                text = normalize_text(load_text_from_file(path))
                loaded.append((d.id, d.filename, text))
            if not loaded:
                raise RuntimeError("No readable documents")
            idx_dir = indexes_dir(pp.project_id) / f"pipeline_{pp.id}"
            meta = build_index_for_docs(idx_dir, loaded, cfg)
            for d in docs:
                d.status = "processed"
                s.add(d)
            s.commit()
            return {"ok": True, "index_dir": str(idx_dir), "meta": meta}

    background.submit(job_id, _threadsafe_job)
    return {"ok": True, "job_id": job_id, "status": background.status(job_id)}


@router.get("/{pipeline_id}/build_index_status")
def build_index_status(pipeline_id: int, session: Session = Depends(db_session)):
    p = session.get(Pipeline, pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    job_id = f"build_index:{p.project_id}:{p.id}"
    st = background.status(job_id)
    res = background.get_result(job_id)
    err = background.get_error(job_id)
    return {"job_id": job_id, "status": st, "result": res, "error": err}
