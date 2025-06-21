import importlib.util
from pathlib import Path
import json
from unittest.mock import Mock, patch
from argparse import Namespace
from rich.console import Console

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
                    {
                        "model": "claude-sonnet-4",
                        "totalTokens": 1000,
                        "inputTokens": 600,
                        "outputTokens": 400,
                    },
                    {
                        "model": "claude-opus-4",
                        "total": 500,
                        "inputTokens": 200,
                        "outputTokens": 300,
                    },
                ],
            }
        ]
    }
    mapping = monitor.get_session_model_usage(active_block, session_info)
    assert mapping == {
        "claude-sonnet-4": {
            "total": 1000,
            "input_tokens": 600,
            "output_tokens": 400,
        },
        "claude-opus-4": {
            "total": 500,
            "input_tokens": 200,
            "output_tokens": 300,
        },
    }


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
                    {
                        "model": "claude-sonnet-4",
                        "totalTokens": 50,
                    }
                ],
            },
            {
                "sessionId": "s2",
                "lastActivity": "2024-01-02T00:20:00Z",
                "modelBreakdowns": [
                    {
                        "model": "claude-opus-4",
                        "totalTokens": 75,
                    }
                ],
            },
        ]
    }
    mapping = monitor.get_session_model_usage(active_block, session_info)
    assert mapping == {
        "claude-opus-4": {
            "total": 75,
            "input_tokens": None,
            "output_tokens": None,
        }
    }


def test_format_model_usage_summary():
    tokens = 5000
    total_tokens = 10000
    input_tokens = 2000
    output_tokens = 3000
    pricing = {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
    }
    with patch.object(monitor, "get_model_pricing", return_value=pricing):
        summary = monitor.format_model_usage(
            "claude-opus-4",
            tokens,
            total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    expected_cost = (
        input_tokens * pricing["input_cost_per_token"]
        + output_tokens * pricing["output_cost_per_token"]
    )
    expected_cost_str = f"${expected_cost:.2f}"
    assert "50.0%" in summary
    assert "5,000" in summary
    assert expected_cost_str in summary


def test_format_model_usage_unknown_model():
    with patch.object(monitor, "get_model_pricing", return_value=None):
        summary = monitor.format_model_usage("unknown-model", 100, 1000)
    assert "10.0%" in summary
    assert "$0.00" in summary


def test_format_model_usage_average_fallback():
    tokens = 4000
    total_tokens = 8000
    pricing = {
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.00001,
    }
    with patch.object(monitor, "get_model_pricing", return_value=pricing):
        summary = monitor.format_model_usage("claude-sonnet-4", tokens, total_tokens)
    expected_cost = (
        tokens * (pricing["input_cost_per_token"] + pricing["output_cost_per_token"]) / 2
    )
    expected_cost_str = f"${expected_cost:.2f}"
    assert expected_cost_str in summary


def test_get_model_pricing_fallback():
    monitor._PRICING_CACHE = None
    with patch("urllib.request.urlopen", side_effect=Exception):
        pricing = monitor.get_model_pricing("claude-opus-4")
    assert pricing == monitor.DEFAULT_MODEL_PRICING["claude-opus-4"]


def test_create_model_ratio_bar_plain():
    usage = {
        "claude-opus-4": {"total": 70},
        "claude-sonnet-4": {"total": 30},
    }
    bar = monitor.create_model_ratio_bar(usage, width=20, plain=True)
    assert bar.count("█") == 20
    assert "Opus" in bar and "Sonnet" in bar


def test_run_rich_once_combined_bar():
    data = {
        "blocks": [
            {
                "isActive": True,
                "startTime": "2024-01-01T00:00:00Z",
                "model": "claude-opus-4",
                "totalTokens": 1000,
                "sessionId": "abc",
            }
        ]
    }
    session_info = {
        "sessions": [
            {
                "sessionId": "abc",
                "modelBreakdowns": [
                    {"model": "claude-opus-4", "totalTokens": 700},
                    {"model": "claude-sonnet-4", "totalTokens": 300},
                ],
            }
        ]
    }
    console = Console(record=True)
    args = Namespace(plan="pro", reset_hour=None, timezone="UTC", plain=False)
    state = monitor.MonitorState(token_limit=7000)
    monitor.run_rich_once(args, data, session_info, console=console, state=state)
    output = console.export_text()
    lines = [line for line in output.splitlines() if "Opus" in line and "Sonnet" in line]
    assert lines
