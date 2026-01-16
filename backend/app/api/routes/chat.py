from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from app.api.deps import db_session
from app.db.models import Pipeline
from app.rag.pipeline import PipelineConfig, answer
from app.services.storage import indexes_dir

router = APIRouter()


class ChatRequest(BaseModel):
    project_id: int
    pipeline_id: int
    question: str
    override_model_id: Optional[str] = None


@router.post("/ask")
def ask(payload: ChatRequest, session: Session = Depends(db_session)):
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

    out = answer(index_dir=idx_dir, question=q, cfg=cfg)
    return out
