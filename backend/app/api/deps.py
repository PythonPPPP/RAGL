from __future__ import annotations

from fastapi import Depends
from sqlmodel import Session

from app.db.session import get_session


def db_session(session: Session = Depends(get_session)) -> Session:
    return session
