from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import db_session
from app.db.models import Dataset

router = APIRouter()


class DatasetCreate(BaseModel):
    project_id: int
    name: str
    description: str = ""
    data: List[Dict[str, Any]]


class DatasetOut(BaseModel):
    id: int
    project_id: int
    name: str
    description: str
    count: int


@router.get("", response_model=List[DatasetOut])
def list_datasets(project_id: int, session: Session = Depends(db_session)):
    items = session.exec(select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.id.desc())).all()
    out = []
    for d in items:
        data = json.loads(d.data_json or "[]")
        out.append(DatasetOut(id=d.id, project_id=d.project_id, name=d.name, description=d.description, count=len(data)))
    return out


@router.post("", response_model=DatasetOut)
def create_dataset(payload: DatasetCreate, session: Session = Depends(db_session)):
    if not payload.data:
        raise HTTPException(status_code=400, detail="Dataset data is empty")
    ds = Dataset(project_id=payload.project_id, name=payload.name, description=payload.description, data_json=json.dumps(payload.data, ensure_ascii=False))
    session.add(ds)
    session.commit()
    session.refresh(ds)
    return DatasetOut(id=ds.id, project_id=ds.project_id, name=ds.name, description=ds.description, count=len(payload.data))


@router.get("/{dataset_id}")
def get_dataset(dataset_id: int, session: Session = Depends(db_session)):
    ds = session.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "id": ds.id,
        "project_id": ds.project_id,
        "name": ds.name,
        "description": ds.description,
        "data": json.loads(ds.data_json or "[]"),
    }


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, session: Session = Depends(db_session)):
    ds = session.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    session.delete(ds)
    session.commit()
    return {"ok": True}
