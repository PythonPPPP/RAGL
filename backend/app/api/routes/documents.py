from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import db_session
from app.db.models import Document
from app.services.storage import documents_dir

router = APIRouter()


class DocumentOut(BaseModel):
    id: int
    project_id: int
    filename: str
    status: str
    meta: Dict[str, Any] = {}
    created_at: str


@router.get("", response_model=List[DocumentOut])
def list_documents(project_id: int, session: Session = Depends(db_session)):
    docs = session.exec(select(Document).where(Document.project_id == project_id).order_by(Document.id.desc())).all()
    out: List[DocumentOut] = []
    for d in docs:
        try:
            meta = json.loads(d.meta_json or "{}")
        except Exception:
            meta = {}
        out.append(
            DocumentOut(
                id=d.id,
                project_id=d.project_id,
                filename=d.filename,
                status=d.status,
                meta=meta,
                created_at=d.created_at.isoformat() if isinstance(d.created_at, datetime) else str(d.created_at),
            )
        )
    return out


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    project_id: int = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(db_session),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    dest_dir = documents_dir(project_id)
    dest = dest_dir / file.filename
    # avoid overwrite
    i = 1
    while dest.exists():
        stem = Path(file.filename).stem
        suf = Path(file.filename).suffix
        dest = dest_dir / f"{stem}_{i}{suf}"
        i += 1

    data = await file.read()
    dest.write_bytes(data)

    doc = Document(
        project_id=project_id,
        filename=dest.name,
        original_path=str(dest),
        status="uploaded",
        meta_json=json.dumps({"size": len(data)}),
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    return DocumentOut(
        id=doc.id,
        project_id=doc.project_id,
        filename=doc.filename,
        status=doc.status,
        meta=json.loads(doc.meta_json or "{}"),
        created_at=doc.created_at.isoformat(),
    )


@router.delete("/{doc_id}")
def delete_document(doc_id: int, session: Session = Depends(db_session)):
    d = session.get(Document, doc_id)
    if not d:
        raise HTTPException(status_code=404, detail="Document not found")
    # best-effort: remove file from disk
    path = Path(d.original_path)
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        pass
    session.delete(d)
    session.commit()
    return {"ok": True}
