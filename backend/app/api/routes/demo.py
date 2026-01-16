from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, select, delete

from app.api.deps import db_session
from app.db.models import Dataset, Document, Pipeline, Project, Run
from app.rag.pipeline import PipelineConfig, build_index_for_docs
from app.services.storage import documents_dir, indexes_dir, project_dir
from app.core.config import settings


router = APIRouter()


class DemoSeedRequest(BaseModel):
    force: bool = False
    build_index: bool = True


def _write_demo_docs(p_id: int) -> List[Path]:
    ddir = documents_dir(p_id)
    files: List[Path] = []

    docs = {
        "company_handbook.md": (
            "# Company Handbook (Demo)\n\n"
            "## Refund policy\n"
            "We offer a **14-day refund window** from the purchase date.\n"
            "Refunds are processed within **5 business days** after approval.\n\n"
            "## Password reset\n"
            "To reset your password: open the login page → click **Forgot password** → "
            "enter your email → follow the link.\n\n"
            "## Support hours\n"
            "Support is available **Mon–Fri, 09:00–18:00 (local time)**.\n"
        ),
        "product_faq.txt": (
            "Product FAQ (Demo)\n"
            "- The starter plan includes 3 projects.\n"
            "- Embeddings are computed locally with SentenceTransformers.\n"
            "- Indices are stored on disk in the project folder.\n"
        ),
        "security_notes.md": (
            "# Security Notes (Demo)\n\n"
            "- Data never leaves your machine in Local mode.\n"
            "- Models are downloaded from HuggingFace into the local cache.\n"
            "- You can switch to a smaller LLM for faster CPU inference.\n"
        ),
    }

    for name, content in docs.items():
        path = ddir / name
        path.write_text(content, encoding="utf-8")
        files.append(path)

    return files


@router.post("/seed")
def seed_demo(payload: DemoSeedRequest, session: Session = Depends(db_session)):
    """Create a full demo pack: project + docs + datasets + pipelines.

    This endpoint is intentionally idempotent unless `force=true`.
    """

    existing = session.exec(select(Project).where(Project.name == "Demo Project")).first()
    if existing and payload.force:
        pid = existing.id
        # Delete rows
        session.exec(delete(Document).where(Document.project_id == pid))
        session.exec(delete(Dataset).where(Dataset.project_id == pid))
        session.exec(delete(Pipeline).where(Pipeline.project_id == pid))
        session.exec(delete(Run).where(Run.project_id == pid))
        session.exec(delete(Project).where(Project.id == pid))
        session.commit()
        # Delete files
        try:
            shutil.rmtree(project_dir(pid), ignore_errors=True)
        except Exception:
            pass
        existing = None

    if existing:
        return {"ok": True, "project_id": existing.id, "message": "Demo already exists"}

    # Create project
    p = Project(name="Demo Project", description="A ready-to-try demo pack (docs, datasets, pipelines)")
    session.add(p)
    session.commit()
    session.refresh(p)

    # Create demo docs
    files = _write_demo_docs(p.id)
    for fp in files:
        d = Document(project_id=p.id, filename=fp.name, original_path=str(fp), kind="file", status="uploaded")
        session.add(d)
    session.commit()

    # Create datasets
    ds1 = Dataset(
        project_id=p.id,
        name="Support QA v1",
        description="Short operational questions (demo)",
        data_json=json.dumps(
            [
                {"question": "What is the refund policy?", "reference": "14-day refund window; 5 business days processing"},
                {"question": "How do I reset my password?", "reference": "Forgot password → email link"},
                {"question": "What are your support hours?", "reference": "Mon–Fri 09:00–18:00"},
            ],
            ensure_ascii=False,
        ),
    )
    ds2 = Dataset(
        project_id=p.id,
        name="Stress / Adversarial v1",
        description="Trick questions to test hallucinations (demo)",
        data_json=json.dumps(
            [
                {"question": "Do you have 30-day refunds?", "reference": "No — 14 days"},
                {"question": "Is data sent to the cloud by default?", "reference": "No — local mode"},
            ],
            ensure_ascii=False,
        ),
    )
    session.add(ds1)
    session.add(ds2)
    session.commit()

    # Pipelines from templates
    templates = {
        "naive": {
            "name": "Naive (Fast)",
            "description": "Dense retrieval + small top-k. No rerank.",
            "chunker": {"chunk_size": 1024, "overlap": 80},
            "embedding_model": settings.default_embed_model,
            "reranker_model": settings.default_rerank_model,
            "query": {"mode": "none", "max_queries": 3},
            "context": {"max_chars": 8000},
            "guardrails": {"require_citations": True, "min_citations": 1},
            "retriever": {"mode": "dense", "top_k": 6, "use_reranker": False, "rerank_top_n": 0},
            "generator": {"model_id": settings.default_llm_model, "temperature": 0.1, "max_new_tokens": 256, "top_p": 0.9},
        },
        "advanced": {
            "name": "Hybrid + Rerank",
            "description": "Hybrid dense+BM25 retrieval + reranking for higher faithfulness.",
            "chunker": {"chunk_size": 900, "overlap": 120},
            "embedding_model": settings.default_embed_model,
            "reranker_model": settings.default_rerank_model,
            "query": {"mode": "none", "max_queries": 3},
            "context": {"max_chars": 8000},
            "guardrails": {"require_citations": True, "min_citations": 1},
            "retriever": {"mode": "hybrid", "top_k": 16, "hybrid_alpha": 0.6, "use_reranker": True, "rerank_top_n": 6, "min_score": 0.15},
            "generator": {"model_id": settings.default_llm_model, "temperature": 0.1, "max_new_tokens": 256, "top_p": 0.9},
        },
        "diverse": {
            "name": "Hybrid + MMR (Diverse)",
            "description": "Diverse retrieval via multi-query + MMR.",
            "chunker": {"chunk_size": 900, "overlap": 120},
            "embedding_model": settings.default_embed_model,
            "reranker_model": settings.default_rerank_model,
            "query": {"mode": "multi", "max_queries": 3},
            "context": {"max_chars": 9000},
            "guardrails": {"require_citations": True, "min_citations": 1},
            "retriever": {"mode": "hybrid", "top_k": 16, "hybrid_alpha": 0.6, "use_reranker": True, "rerank_top_n": 6, "use_mmr": True, "mmr_lambda": 0.65, "mmr_k": 30},
            "generator": {"model_id": settings.default_llm_model, "temperature": 0.1, "max_new_tokens": 256, "top_p": 0.9},
        },
    }

    created: Dict[str, int] = {}
    for key, conf in templates.items():
        cfg = PipelineConfig(**conf)
        pl = Pipeline(project_id=p.id, name=cfg.name, description=cfg.description, config_json=json.dumps(cfg.model_dump(mode="json"), ensure_ascii=False))
        session.add(pl)
        session.commit()
        session.refresh(pl)
        created[key] = pl.id

    # Optionally build index for the advanced pipeline
    built_for: List[int] = []
    if payload.build_index and "advanced" in created:
        pipeline_id = created["advanced"]
        pl = session.get(Pipeline, pipeline_id)
        docs = session.exec(select(Document).where(Document.project_id == p.id)).all()
        loaded = []
        for d in docs:
            fp = Path(d.original_path)
            if fp.exists():
                loaded.append((d.id, d.filename, fp.read_text(encoding="utf-8", errors="ignore")))
        idx_dir = indexes_dir(p.id) / f"pipeline_{pipeline_id}"
        build_index_for_docs(idx_dir, loaded, PipelineConfig(**json.loads(pl.config_json)))
        for d in docs:
            d.status = "processed"
            session.add(d)
        session.commit()
        built_for.append(pipeline_id)

    return {
        "ok": True,
        "project_id": p.id,
        "pipelines": created,
        "index_built_for": built_for,
        "message": "Demo pack created",
    }
