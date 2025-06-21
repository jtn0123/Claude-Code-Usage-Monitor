import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import json
from unittest.mock import patch, Mock


from ccusage_monitor import run_ccusage_session, get_session_model_usage


def test_run_ccusage_session_parses_json():
    mock_output = json.dumps({"sessions": []})
    mock_completed = Mock(stdout=mock_output)
    with patch('subprocess.run', return_value=mock_completed) as run_mock:
        result = run_ccusage_session()
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
    mapping = get_session_model_usage(active_block, session_info)
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
                "modelBreakdowns": [
                    {"model": "claude-sonnet-4", "totalTokens": 50}
                ],
            },
            {
                "sessionId": "s2",
                "lastActivity": "2024-01-02T00:20:00Z",
                "modelBreakdowns": [
                    {"model": "claude-opus-4", "totalTokens": 75}
                ],
            },
        ]
    }
    mapping = get_session_model_usage(active_block, session_info)
    assert mapping == {"claude-opus-4": 75}
