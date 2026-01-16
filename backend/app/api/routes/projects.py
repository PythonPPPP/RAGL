from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import db_session
from app.db.models import Project

router = APIRouter()


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectOut(BaseModel):
    id: int
    name: str
    description: str


def _ensure_default(session: Session) -> None:
    existing = session.exec(select(Project)).first()
    if existing is None:
        p = Project(name="Demo Project", description="Starter project")
        session.add(p)
        session.commit()


@router.get("", response_model=List[ProjectOut])
def list_projects(session: Session = Depends(db_session)):
    _ensure_default(session)
    items = session.exec(select(Project).order_by(Project.id)).all()
    return [ProjectOut(id=p.id, name=p.name, description=p.description) for p in items]


@router.post("", response_model=ProjectOut)
def create_project(payload: ProjectCreate, session: Session = Depends(db_session)):
    p = Project(name=payload.name, description=payload.description)
    session.add(p)
    session.commit()
    session.refresh(p)
    return ProjectOut(id=p.id, name=p.name, description=p.description)


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: int, session: Session = Depends(db_session)):
    p = session.get(Project, project_id)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectOut(id=p.id, name=p.name, description=p.description)
