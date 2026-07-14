from datetime import datetime, timezone

from app.worker import print_process_time


def test_process_time_log_contains_project_status_and_duration(capsys):
    print_process_time("project-123", datetime.now(timezone.utc), 125.4, "completed")
    output = capsys.readouterr().out
    assert "Project    : project-123" in output
    assert "Status     : completed" in output
    assert "Total time : 0:02:05 (125.40 seconds)" in output

