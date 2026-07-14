from contextlib import asynccontextmanager
from pathlib import Path
import mimetypes
import re
import shutil
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import settings
from .database import initialize_database
from .languages import LANGUAGES, LANGUAGE_CODES
from .repository import (
    create_live_session, create_project, delete_project, finish_live_session, get_live_session,
    get_project, list_projects, register_live_chunk,
)
from .schemas import Language, LiveSession, LiveSessionCreate, Project, ProjectList


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(title="Video Translation API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/languages", response_model=list[Language])
def languages() -> list[dict]:
    return LANGUAGES


@app.post("/api/v1/live-sessions", response_model=LiveSession, status_code=status.HTTP_201_CREATED)
def start_live_session(payload: LiveSessionCreate) -> dict:
    if payload.source_language not in LANGUAGE_CODES or payload.target_language not in LANGUAGE_CODES:
        raise HTTPException(422, "Unsupported source or target language")
    if payload.source_language == payload.target_language:
        raise HTTPException(422, "Source and target languages must be different")
    session = create_live_session(payload.source_language, payload.target_language, payload.voice_gender)
    (settings.media_dir / "live" / session["id"] / "chunks").mkdir(parents=True, exist_ok=True)
    return session


@app.post("/api/v1/live-sessions/{session_id}/chunks/{chunk_index}", response_model=LiveSession)
async def upload_live_chunk(session_id: str, chunk_index: int, chunk: UploadFile = File(...)) -> dict:
    if chunk_index < 0:
        raise HTTPException(422, "Invalid chunk index")
    session = get_live_session(session_id)
    if not session:
        raise HTTPException(404, "Live session not found")
    if session["status"] != "recording":
        raise HTTPException(409, "Live session is no longer recording")
    directory = settings.media_dir / "live" / session_id / "chunks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{chunk_index:08d}.part"
    size = 0
    try:
        with path.open("wb") as output:
            while data := await chunk.read(1024 * 1024):
                size += len(data)
                output.write(data)
    finally:
        await chunk.close()
    if not size:
        path.unlink(missing_ok=True)
        raise HTTPException(422, "Recording chunk is empty")
    try:
        count = register_live_chunk(session_id, chunk_index)
    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(409, str(exc)) from exc
    return {"id": session_id, "status": "recording", "chunk_count": count, "project_id": None}


@app.post("/api/v1/live-sessions/{session_id}/finish", response_model=Project, status_code=status.HTTP_202_ACCEPTED)
def finish_recording(session_id: str) -> dict:
    session = get_live_session(session_id)
    if not session:
        raise HTTPException(404, "Live session not found")
    if session["status"] != "recording":
        raise HTTPException(409, "Live session has already finished")
    chunk_paths = sorted((settings.media_dir / "live" / session_id / "chunks").glob("*.part"))
    if not chunk_paths:
        raise HTTPException(422, "No recording chunks were uploaded")
    inputs = settings.media_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    recording = inputs / f"{session_id}_recording.webm"
    with recording.open("wb") as output:
        for path in chunk_paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output)
    project = create_project(
        filename=f"recording-{session_id}.webm", input_path=recording, input_type="recording",
        source=session["source_language"], target=session["target_language"], voice_gender=session["voice_gender"],
    )
    finish_live_session(session_id, project["id"])
    return project


@app.post("/api/v1/projects", response_model=Project, status_code=status.HTTP_202_ACCEPTED)
async def submit_project(
    video: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...),
    input_type: str = Form("upload"),
    voice_gender: str = Form("female"),
) -> dict:
    if source_language not in LANGUAGE_CODES or target_language not in LANGUAGE_CODES:
        raise HTTPException(422, "Unsupported source or target language")
    if source_language == target_language:
        raise HTTPException(422, "Source and target languages must be different")
    if input_type not in {"upload", "recording"}:
        raise HTTPException(422, "input_type must be upload or recording")
    if voice_gender not in {"female", "male"}:
        raise HTTPException(422, "voice_gender must be female or male")
    if not (video.content_type or "").startswith("video/"):
        raise HTTPException(415, "A video file is required")

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(video.filename or "video.webm").name)
    upload_dir = settings.media_dir / "inputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid4()}_{safe_name}"
    size = 0
    try:
        with path.open("wb") as output:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"Video exceeds {settings.max_upload_mb} MB")
                output.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await video.close()

    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(422, "The uploaded video is empty")
    return create_project(
        filename=safe_name, input_path=path, input_type=input_type,
        source=source_language, target=target_language, voice_gender=voice_gender,
    )


@app.get("/api/v1/projects", response_model=ProjectList)
def projects() -> dict:
    return {"items": list_projects()}


@app.get("/api/v1/projects/{project_id}", response_model=Project)
def project(project_id: str) -> dict:
    item = get_project(project_id)
    if not item:
        raise HTTPException(404, "Project not found")
    return item


def project_input(project_id: str) -> tuple[dict, Path, str]:
    item = get_project(project_id, include_paths=True)
    if not item:
        raise HTTPException(404, "Project not found")
    path = Path(item["input_path"])
    if not path.is_file():
        raise HTTPException(404, "Input video not found")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return item, path, media_type


@app.get("/api/v1/projects/{project_id}/input/preview")
def input_preview(project_id: str) -> FileResponse:
    _, path, media_type = project_input(project_id)
    return FileResponse(path, media_type=media_type, content_disposition_type="inline")


@app.get("/api/v1/projects/{project_id}/input/download")
def input_download(project_id: str) -> FileResponse:
    item, path, media_type = project_input(project_id)
    return FileResponse(path, media_type=media_type, filename=item["original_filename"])


@app.delete("/api/v1/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_project(project_id: str) -> None:
    try:
        item = delete_project(project_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if not item:
        raise HTTPException(404, "Project not found")
    input_path = Path(item["input_path"])
    input_path.unlink(missing_ok=True)
    output_path = Path(item["output_path"]) if item.get("output_path") else None
    output_directory = output_path.parent if output_path else settings.media_dir / "outputs" / project_id
    if output_directory.is_dir() and settings.media_dir in output_directory.parents:
        shutil.rmtree(output_directory, ignore_errors=True)
    for session_id in item["live_session_ids"]:
        live_directory = settings.media_dir / "live" / session_id
        if live_directory.is_dir():
            shutil.rmtree(live_directory, ignore_errors=True)


@app.get("/api/v1/projects/{project_id}/download")
def download(project_id: str) -> FileResponse:
    item = get_project(project_id, include_paths=True)
    if not item:
        raise HTTPException(404, "Project not found")
    if item["status"] != "completed" or not item["output_path"]:
        raise HTTPException(409, "Project is not complete")
    path = Path(item["output_path"])
    if not path.is_file():
        raise HTTPException(404, "Output file not found")
    return FileResponse(path, media_type="video/mp4", filename=f"translated-{project_id}.mp4")


@app.get("/api/v1/projects/{project_id}/preview")
def preview(project_id: str) -> FileResponse:
    item = get_project(project_id, include_paths=True)
    if not item:
        raise HTTPException(404, "Project not found")
    if item["status"] != "completed" or not item["output_path"]:
        raise HTTPException(409, "Project is not complete")
    path = Path(item["output_path"])
    if not path.is_file():
        raise HTTPException(404, "Output file not found")
    return FileResponse(path, media_type="video/mp4", content_disposition_type="inline")


@app.get("/api/v1/projects/{project_id}/artifacts/{artifact_name}")
def artifact(project_id: str, artifact_name: str) -> FileResponse:
    if artifact_name not in {"deepgram.json", "segmentation.json"}:
        raise HTTPException(404, "Artifact not found")
    item = get_project(project_id, include_paths=True)
    if not item:
        raise HTTPException(404, "Project not found")
    if item["status"] != "completed" or not item["output_path"]:
        raise HTTPException(409, "Project is not complete")
    path = Path(item["output_path"]).parent / artifact_name
    if not path.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path, media_type="application/json", filename=artifact_name)
