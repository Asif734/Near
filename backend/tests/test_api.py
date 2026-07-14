from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_health_and_languages():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert len(client.get("/api/v1/languages").json()) >= 2


def test_rejects_non_video():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/projects",
            data={"source_language": "en", "target_language": "bn", "input_type": "upload"},
            files={"video": ("note.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 415


def test_live_recording_chunks_create_background_project():
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/live-sessions",
            json={"source_language": "bn", "target_language": "zh-CN", "voice_gender": "male"},
        )
        assert session.status_code == 201
        session_id = session.json()["id"]
        uploaded = client.post(
            f"/api/v1/live-sessions/{session_id}/chunks/0",
            files={"chunk": ("chunk.webm", b"recording-chunk", "video/webm")},
        )
        assert uploaded.status_code == 200
        assert uploaded.json()["chunk_count"] == 1
        finished = client.post(f"/api/v1/live-sessions/{session_id}/finish")
        assert finished.status_code == 202
        assert finished.json()["input_type"] == "recording"
        assert finished.json()["status"] == "queued"
        assert finished.json()["voice_gender"] == "male"
        assert client.get(finished.json()["input_preview_url"]).content == b"recording-chunk"
        assert client.get(finished.json()["input_download_url"]).status_code == 200
        deleted = client.delete(f"/api/v1/projects/{finished.json()['id']}")
        assert deleted.status_code == 204
        assert client.get(f"/api/v1/projects/{finished.json()['id']}").status_code == 404
