from datetime import datetime
from typing import Literal

from pydantic import BaseModel


ProjectStatus = Literal["queued", "processing", "completed", "failed"]
InputType = Literal["upload", "recording"]


class Language(BaseModel):
    code: str
    name: str
    voice: str


class Project(BaseModel):
    id: str
    original_filename: str
    input_type: InputType
    source_language: str
    target_language: str
    voice_gender: Literal["female", "male"]
    status: ProjectStatus
    stage: str
    progress: int
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    download_url: str | None = None
    preview_url: str | None = None
    input_download_url: str | None = None
    input_preview_url: str | None = None
    artifact_urls: dict[str, str] | None = None


class ProjectList(BaseModel):
    items: list[Project]


class LiveSessionCreate(BaseModel):
    source_language: str
    target_language: str
    voice_gender: Literal["female", "male"] = "female"


class LiveSession(BaseModel):
    id: str
    status: str
    chunk_count: int
    project_id: str | None = None
