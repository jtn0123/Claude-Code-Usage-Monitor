import importlib.util
from pathlib import Path
import json
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "ccusage_monitor", Path(__file__).resolve().parents[1] / "ccusage_monitor.py"
)
assert spec and spec.loader
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_run_ccusage_session_parses_json():
    mock_output = json.dumps({"sessions": []})
    mock_completed = Mock(stdout=mock_output)
    with patch("subprocess.run", return_value=mock_completed) as run_mock:
        result = monitor.run_ccusage_session()
        run_mock.assert_called_once()
        assert result == {"sessions": []}


def test_get_session_model_usage_by_id():
    active_block = {
        "sessionId": "abc123",
        "startTime": "2024-01-01T00:00:00Z",
        "lastActivity": "2024-01-01T00:10:00Z",
    }
    session_info = {
        "sessions": [
            {
                "sessionId": "abc123",
                "modelBreakdowns": [
                    {"model": "claude-sonnet-4", "totalTokens": 1000},
                    {"model": "claude-opus-4", "total": 500},
                ],
            }
        ]
    }
    mapping = monitor.get_session_model_usage(active_block, session_info)
    assert mapping == {"claude-sonnet-4": 1000, "claude-opus-4": 500}


def test_get_session_model_usage_no_match():
    active_block = {
        "startTime": "2024-01-02T00:00:00Z",
        "lastActivity": "2024-01-02T00:10:00Z",
    }
    session_info = {
        "sessions": [
            {
                "sessionId": "s1",
                "lastActivity": "2024-01-01T00:10:00Z",
                "modelBreakdowns": [{"model": "claude-sonnet-4", "totalTokens": 50}],
            },
            {
                "sessionId": "s2",
                "lastActivity": "2024-01-02T00:20:00Z",
                "modelBreakdowns": [{"model": "claude-opus-4", "totalTokens": 75}],
            },
        ]
    }
    mapping = monitor.get_session_model_usage(active_block, session_info)
    assert mapping == {"claude-opus-4": 75}
