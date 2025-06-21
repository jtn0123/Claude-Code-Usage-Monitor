import importlib.util
from pathlib import Path
from unittest.mock import patch
import sys

spec = importlib.util.spec_from_file_location(
    "ccusage_monitor", Path(__file__).resolve().parents[1] / "ccusage_monitor.py"
)
assert spec and spec.loader
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


@patch("os.system")
@patch("time.sleep", side_effect=KeyboardInterrupt)
@patch.object(monitor, "run_ccusage")
@patch.object(monitor, "run_ccusage_session")
def test_main_plain_mode_once(mock_session, mock_usage, mock_sleep, mock_os, capsys):
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
    with patch.object(sys, "argv", ["prog", "--plain"]):
        try:
            monitor.main()
        except SystemExit:
            pass

    captured = capsys.readouterr().out
    assert "CLAUDE TOKEN MONITOR" in captured
    assert "Token Usage" in captured
