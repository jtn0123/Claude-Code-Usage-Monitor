import importlib.util
from pathlib import Path
from unittest.mock import patch
from argparse import Namespace
from datetime import datetime

spec = importlib.util.spec_from_file_location(
    "ccusage_monitor", Path(__file__).resolve().parents[1] / "ccusage_monitor.py"
)
assert spec and spec.loader
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_run_plain_once_basic(capsys):
    data = {
        "blocks": [
            {
                "isActive": True,
                "startTime": "2024-01-01T00:00:00Z",
                "model": "claude-opus-4",
                "totalTokens": 1000,
                "sessionId": "abc123",
            }
        ]
    }
    session_info = {
        "sessions": [
            {
                "sessionId": "abc123",
                "modelBreakdowns": [{"model": "claude-opus-4", "totalTokens": 1000}],
            }
        ]
    }
    args = Namespace(plan="pro", reset_hour=None, timezone="UTC", plain=True)
    monitor.run_plain_once(args, 7000, data, session_info)

    captured = capsys.readouterr().out
    assert "CLAUDE TOKEN MONITOR" in captured
    assert "Token Usage" in captured


def test_run_plain_once_no_start_time(capsys):
    """Monitor should handle missing startTime."""
    data = {
        "blocks": [
            {
                "isActive": True,
                "model": "claude-opus-4",
                "totalTokens": 1000,
                "sessionId": "abc123",
            }
        ]
    }
    session_info = {
        "sessions": [
            {
                "sessionId": "abc123",
                "modelBreakdowns": [{"model": "claude-opus-4", "totalTokens": 1000}],
            }
        ]
    }
    args = Namespace(plan="pro", reset_hour=None, timezone="UTC", plain=True)
    with patch.object(monitor, "get_next_reset_time", return_value=datetime.now()):
        monitor.run_plain_once(args, 7000, data, session_info)

    captured = capsys.readouterr().out
    assert "CLAUDE TOKEN MONITOR" in captured
    assert "Token Usage" in captured
