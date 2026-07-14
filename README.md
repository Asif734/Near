# Dubflow — FastAPI + React video translation

This is a self-contained FastAPI and React video translation application. It includes:

- Video upload from disk
- In-page camera and microphone recording with `MediaRecorder`
- A shared multipart upload endpoint for both input types
- A separate durable worker process
- SQLite-backed project state and progress
- React project list with automatic status polling
- FFmpeg normalization to MP4, including browser-generated WebM recordings
- Downloadable processed output

## Processing pipeline

The background worker now runs the complete pipeline:

1. Extract 16 kHz mono audio with FFmpeg.
2. Transcribe through Deepgram Nova-3 with word timestamps.
3. Build natural, timestamped speech segments.
4. Translate every segment with Google Translator.
5. Generate target-language speech with Edge TTS.
6. Fit each TTS clip into its available timeline window.
7. Mix fitted clips onto a non-overlapping audio timeline.
8. Generate translated SRT subtitles.
9. Replace the source audio and burn subtitles into the final MP4.

Segment metadata and intermediate artifacts are retained inside each project output folder for debugging.

The complete engine is owned by this repository under `backend/app/video_engine`. It preserves Deepgram gap/tail retries, silence-aware segmentation, OpenAI transcript and translation QA, timing rewrites, Edge TTS retry/fallback behavior, segment scheduling, SRT/ASS generation, and subtitle rendering. It does not import, mount, or call another project at runtime.

Every completed project folder includes the stable artifacts:

- `deepgram.json`
- `segmentation.json`
- `artifacts.json`
- `output.mp4`

## Run with Docker

```bash
docker compose up --build
```

Open <http://localhost:8080>. FastAPI documentation is at <http://localhost:8000/docs>.

## Run locally

Backend, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload
```

Worker, in another terminal with the same environment:

```bash
cd backend
python -m app.worker
```

Frontend, in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

Camera recording requires a secure browser context. `localhost` is accepted during development; production must use HTTPS.

Create `.env` from `.env.example` and provide `DEEPGRAM_API_KEY`. Export the variables before starting the API and worker, or pass them through Docker Compose.

## API

- `GET /health`
- `GET /api/v1/languages`
- `POST /api/v1/projects`
- `GET /api/v1/projects`
- `GET /api/v1/projects/{id}`
- `GET /api/v1/projects/{id}/download`

The project creation request is `multipart/form-data` with `video`, `source_language`, `target_language`, and `input_type` (`upload` or `recording`).
