import importlib.util
from pathlib import Path
from unittest.mock import patch
import argparse

spec = importlib.util.spec_from_file_location(
    "ccusage_monitor", Path(__file__).resolve().parents[1] / "ccusage_monitor.py"
)
assert spec and spec.loader
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


@patch.object(monitor, "run_ccusage")
@patch.object(monitor, "run_ccusage_session")
def test_process_snapshot_plain(mock_session, mock_usage, capsys):
    mock_usage.return_value = {
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
    mock_session.return_value = {
        "sessions": [
            {
                "sessionId": "abc123",
                "modelBreakdowns": [{"model": "claude-opus-4", "totalTokens": 1000}],
            }
        ]
    }
    args = argparse.Namespace(plan="pro", reset_hour=None, timezone="Europe/Warsaw", plain=True)
    token_limit = monitor.get_token_limit(args.plan)
    output, token_limit, _, _ = monitor.process_snapshot(
        args, token_limit, False, False, use_rich=False
    )
    assert "CLAUDE TOKEN MONITOR" in output
    assert "Token Usage" in output
