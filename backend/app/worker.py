import logging
import os
import signal
import time
from datetime import datetime, timedelta

from .database import initialize_database
from .pipeline import process_video
from .repository import claim_next_project, complete_project, fail_project, update_project

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("video-worker")
running = True


def print_process_time(project_id: str, started_at: datetime, elapsed_seconds: float, outcome: str) -> None:
    ended_at = datetime.now().astimezone()
    total = timedelta(seconds=round(elapsed_seconds))
    message = (
        "\nVideo processing time\n"
        f"Project    : {project_id}\n"
        f"Status     : {outcome}\n"
        f"Start time : {started_at:%Y-%m-%d %H:%M:%S %Z}\n"
        f"End time   : {ended_at:%Y-%m-%d %H:%M:%S %Z}\n"
        f"Total time : {total} ({elapsed_seconds:.2f} seconds)"
    )
    print(message, flush=True)
    logger.info(
        "Project %s processing finished with status=%s in %.2f seconds",
        project_id, outcome, elapsed_seconds,
    )


def stop(*_) -> None:
    global running
    running = False


def run() -> None:
    initialize_database()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logger.info("Worker started")
    while running:
        project = claim_next_project()
        if not project:
            time.sleep(1)
            continue
        logger.info("Processing project %s", project["id"])
        started_at = datetime.now().astimezone()
        timer_started = time.perf_counter()
        outcome = "failed"
        try:
            output = process_video(
                project,
                lambda stage, progress: update_project(project["id"], stage=stage, progress=progress),
            )
            complete_project(project["id"], output)
            outcome = "completed"
        except Exception as exc:
            logger.exception("Project %s failed", project["id"])
            fail_project(project["id"], str(exc))
        finally:
            print_process_time(
                project["id"], started_at, time.perf_counter() - timer_started, outcome,
            )


if __name__ == "__main__":
    run()
