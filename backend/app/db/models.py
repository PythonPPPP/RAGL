from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    filename: str
    original_path: str
    kind: str = "file"
    status: str = "uploaded"  # uploaded|processed
    meta_json: str = Field(default="{}", sa_column=Column(Text))
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Dataset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    name: str
    description: str = ""
    data_json: str = Field(default="[]", sa_column=Column(Text))  # list of {question, reference?}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Pipeline(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    name: str
    description: str = ""
    config_json: str = Field(default="{}", sa_column=Column(Text))
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Run(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    pipeline_id: int = Field(index=True)
    dataset_id: Optional[int] = Field(default=None, index=True)
    status: str = "queued"  # queued|running|done|error
    metrics_json: str = Field(default="{}", sa_column=Column(Text))
    notes: str = ""
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SharedPipeline(SQLModel, table=True):
    """Stores shareable pipeline configurations.

    The config is project-agnostic; the frontend can import it into any project.
    """

    code: str = Field(primary_key=True)
    name: str = ""
    description: str = ""
    config_json: str = Field(default="{}", sa_column=Column(Text))
    created_at: datetime = Field(default_factory=datetime.utcnow)
