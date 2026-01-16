from __future__ import annotations

from sqlmodel import SQLModel, Session, create_engine

from app.core.config import settings


def get_engine():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db_path = settings.data_dir / "rag_architect.db"
    return create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )


engine = get_engine()


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
