from pathlib import Path
import sqlite3
from uuid import uuid4

from .database import connect, transaction, utc_now


def _serialize(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    item = dict(row)
    item.pop("input_path", None)
    item.pop("output_path", None)
    item["download_url"] = f"/api/v1/projects/{item['id']}/download" if item["status"] == "completed" else None
    item["preview_url"] = f"/api/v1/projects/{item['id']}/preview" if item["status"] == "completed" else None
    item["input_download_url"] = f"/api/v1/projects/{item['id']}/input/download"
    item["input_preview_url"] = f"/api/v1/projects/{item['id']}/input/preview"
    item["artifact_urls"] = (
        {
            "deepgram": f"/api/v1/projects/{item['id']}/artifacts/deepgram.json",
            "segmentation": f"/api/v1/projects/{item['id']}/artifacts/segmentation.json",
        }
        if item["status"] == "completed" else None
    )
    return item


def create_project(*, filename: str, input_path: Path, input_type: str, source: str, target: str, voice_gender: str = "female") -> dict:
    project_id = str(uuid4())
    now = utc_now()
    with transaction() as db:
        db.execute(
            """INSERT INTO projects
            (id, original_filename, input_path, input_type, source_language, target_language, voice_gender,
             status, stage, progress, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 'Waiting for worker', 0, ?, ?)""",
            (project_id, filename, str(input_path), input_type, source, target, voice_gender, now, now),
        )
        row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return _serialize(row)


def get_project(project_id: str, include_paths: bool = False) -> dict | None:
    with connect() as db:
        row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return dict(row) if include_paths and row else _serialize(row)


def list_projects(limit: int = 50) -> list[dict]:
    with connect() as db:
        rows = db.execute("SELECT * FROM projects ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_serialize(row) for row in rows]


def claim_next_project() -> dict | None:
    with transaction() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM projects WHERE status = 'queued' ORDER BY created_at LIMIT 1").fetchone()
        if not row:
            return None
        now = utc_now()
        changed = db.execute(
            """UPDATE projects SET status='processing', stage='Starting', progress=1,
            started_at=?, updated_at=? WHERE id=? AND status='queued'""",
            (now, now, row["id"]),
        ).rowcount
        if not changed:
            return None
        project = dict(row)
        project.update(status="processing", stage="Starting", progress=1, started_at=now, updated_at=now)
        return project


def update_project(project_id: str, *, stage: str, progress: int) -> None:
    with transaction() as db:
        db.execute(
            "UPDATE projects SET stage=?, progress=?, updated_at=? WHERE id=?",
            (stage, max(0, min(progress, 100)), utc_now(), project_id),
        )


def complete_project(project_id: str, output_path: Path) -> None:
    now = utc_now()
    with transaction() as db:
        db.execute(
            """UPDATE projects SET status='completed', stage='Completed', progress=100,
            output_path=?, error=NULL, completed_at=?, updated_at=? WHERE id=?""",
            (str(output_path), now, now, project_id),
        )


def fail_project(project_id: str, error: str) -> None:
    with transaction() as db:
        db.execute(
            "UPDATE projects SET status='failed', stage='Failed', error=?, updated_at=? WHERE id=?",
            (error[:2000], utc_now(), project_id),
        )


def create_live_session(source: str, target: str, voice_gender: str = "female") -> dict:
    session_id = str(uuid4())
    now = utc_now()
    with transaction() as db:
        db.execute(
            """INSERT INTO live_sessions
            (id, source_language, target_language, voice_gender, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'recording', ?, ?)""",
            (session_id, source, target, voice_gender, now, now),
        )
    return {"id": session_id, "status": "recording", "chunk_count": 0, "project_id": None}


def get_live_session(session_id: str) -> dict | None:
    with connect() as db:
        row = db.execute("SELECT * FROM live_sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def register_live_chunk(session_id: str, chunk_index: int) -> int:
    now = utc_now()
    with transaction() as db:
        row = db.execute("SELECT status, chunk_count FROM live_sessions WHERE id=?", (session_id,)).fetchone()
        if not row or row["status"] != "recording":
            raise ValueError("Live session is not recording")
        count = max(row["chunk_count"], chunk_index + 1)
        db.execute("UPDATE live_sessions SET chunk_count=?, updated_at=? WHERE id=?", (count, now, session_id))
    return count


def finish_live_session(session_id: str, project_id: str) -> None:
    with transaction() as db:
        db.execute(
            "UPDATE live_sessions SET status='processing', project_id=?, updated_at=? WHERE id=?",
            (project_id, utc_now(), session_id),
        )


def delete_project(project_id: str) -> dict | None:
    with transaction() as db:
        row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if not row:
            return None
        if row["status"] == "processing":
            raise ValueError("A project cannot be deleted while its worker is processing")
        project = dict(row)
        sessions = db.execute("SELECT id FROM live_sessions WHERE project_id=?", (project_id,)).fetchall()
        project["live_session_ids"] = [item["id"] for item in sessions]
        db.execute("DELETE FROM live_sessions WHERE project_id=?", (project_id,))
        db.execute("DELETE FROM projects WHERE id=?", (project_id,))
    return project
