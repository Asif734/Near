from contextlib import contextmanager
from datetime import datetime, timezone
import sqlite3
from typing import Iterator

from .config import settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(settings.database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    connection = connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    with transaction() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                input_path TEXT NOT NULL,
                output_path TEXT,
                input_type TEXT NOT NULL,
                source_language TEXT NOT NULL,
                target_language TEXT NOT NULL,
                voice_gender TEXT NOT NULL DEFAULT 'female',
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            )
            """
        )
        db.execute("CREATE INDEX IF NOT EXISTS projects_status_idx ON projects(status, created_at)")
        project_columns = {row["name"] for row in db.execute("PRAGMA table_info(projects)").fetchall()}
        if "voice_gender" not in project_columns:
            db.execute("ALTER TABLE projects ADD COLUMN voice_gender TEXT NOT NULL DEFAULT 'female'")
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS live_sessions (
                id TEXT PRIMARY KEY,
                source_language TEXT NOT NULL,
                target_language TEXT NOT NULL,
                voice_gender TEXT NOT NULL DEFAULT 'female',
                status TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                project_id TEXT
            )
            """
        )
        live_columns = {row["name"] for row in db.execute("PRAGMA table_info(live_sessions)").fetchall()}
        if "voice_gender" not in live_columns:
            db.execute("ALTER TABLE live_sessions ADD COLUMN voice_gender TEXT NOT NULL DEFAULT 'female'")
