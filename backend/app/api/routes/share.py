from __future__ import annotations

import json
import secrets
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from app.api.deps import db_session
from app.db.models import SharedPipeline
from app.rag.pipeline import PipelineConfig

router = APIRouter()


class ShareCreate(BaseModel):
    name: str = ""
    description: str = ""
    config: Dict[str, Any]


@router.post("")
def create_share(payload: ShareCreate, session: Session = Depends(db_session)):
    # validate config
    cfg = PipelineConfig(**payload.config)

    # generate a short-ish code
    for _ in range(8):
        code = secrets.token_urlsafe(8).replace("-", "").replace("_", "")[:10]
        if not code:
            continue
        if session.get(SharedPipeline, code) is None:
            sp = SharedPipeline(
                code=code,
                name=payload.name or cfg.name or "Shared pipeline",
                description=payload.description or cfg.description or "",
                config_json=json.dumps(cfg.model_dump(mode="json"), ensure_ascii=False),
                created_at=datetime.utcnow(),
            )
            session.add(sp)
            session.commit()
            return {"ok": True, "code": code}

    raise HTTPException(status_code=500, detail="Failed to allocate share code")


@router.get("/{code}")
def get_share(code: str, session: Session = Depends(db_session)):
    sp = session.get(SharedPipeline, code)
    if not sp:
        raise HTTPException(status_code=404, detail="Share not found")

    try:
        cfg = json.loads(sp.config_json or "{}")
    except Exception:
        cfg = {}

    return {
        "ok": True,
        "code": sp.code,
        "name": sp.name,
        "description": sp.description,
        "config": cfg,
        "created_at": sp.created_at.isoformat() if isinstance(sp.created_at, datetime) else str(sp.created_at),
    }
