import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
import sys
import subprocess
from unittest.mock import Mock, patch
import pytest

spec = importlib.util.spec_from_file_location(
    "ccusage_monitor",
    Path(__file__).resolve().parents[1] / "ccusage_monitor.py",
)
assert spec and spec.loader
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_run_ccusage_success():
    mock_completed = Mock(stdout=json.dumps({"blocks": []}))
    with patch("subprocess.run", return_value=mock_completed) as run_mock:
        result = monitor.run_ccusage()
        run_mock.assert_called_once()
        assert result == {"blocks": []}


def test_run_ccusage_failure():
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ccusage")):
        assert monitor.run_ccusage() is None


def test_run_ccusage_invalid_json():
    mock_completed = Mock(stdout="{invalid")
    with patch("subprocess.run", return_value=mock_completed):
        assert monitor.run_ccusage() is None


def test_run_ccusage_session_success():
    mock_completed = Mock(stdout=json.dumps({"sessions": []}))
    with patch("subprocess.run", return_value=mock_completed):
        assert monitor.run_ccusage_session() == {"sessions": []}


def test_run_ccusage_session_bad_json():
    mock_completed = Mock(stdout="{invalid")
    with patch("subprocess.run", return_value=mock_completed):
        assert monitor.run_ccusage_session() is None


def test_format_time():
    assert monitor.format_time(45) == "45m"
    assert monitor.format_time(60) == "1h"
    assert monitor.format_time(125) == "2h 5m"


def test_get_velocity_indicator():
    assert monitor.get_velocity_indicator(10) == "🐌"
    assert monitor.get_velocity_indicator(100) == "➡️"
    assert monitor.get_velocity_indicator(200) == "🚀"
    assert monitor.get_velocity_indicator(400) == "⚡"


def test_calculate_hourly_burn_rate():
    now = datetime(2024, 1, 1, 12, 0, 0)
    blocks = [
        {
            "startTime": (now - timedelta(minutes=70)).isoformat(),
            "actualEndTime": (now - timedelta(minutes=50)).isoformat(),
            "isActive": False,
            "totalTokens": 60,
        },
        {
            "startTime": (now - timedelta(minutes=30)).isoformat(),
            "isActive": True,
            "totalTokens": 120,
        },
        {
            "startTime": (now - timedelta(hours=2)).isoformat(),
            "actualEndTime": (now - timedelta(minutes=90)).isoformat(),
            "isActive": False,
            "totalTokens": 100,
        },
    ]
    rate = monitor.calculate_hourly_burn_rate(blocks, now)
    assert pytest.approx(rate, rel=1e-6) == 2.5


def test_get_next_reset_time_default():
    current = datetime(2024, 1, 1, 3, 30)
    expected = datetime(2024, 1, 1, 4, 0, tzinfo=ZoneInfo("Europe/Warsaw"))
    assert monitor.get_next_reset_time(current) == expected


def test_get_next_reset_time_custom_hour_next_day():
    current = datetime(2024, 1, 1, 11, 0, tzinfo=ZoneInfo("UTC"))
    expected = datetime(2024, 1, 2, 5, 0, tzinfo=ZoneInfo("UTC"))
    assert monitor.get_next_reset_time(current, custom_reset_hour=5, timezone_str="UTC") == expected


def test_get_next_reset_time_timezone_conversion():
    current = datetime(2024, 1, 1, 2, 0, tzinfo=ZoneInfo("UTC"))
    expected = datetime(2024, 1, 1, 4, 0, tzinfo=ZoneInfo("UTC"))
    assert monitor.get_next_reset_time(current, timezone_str="America/New_York") == expected


def test_parse_args(monkeypatch):
    argv = [
        "prog",
        "--plan",
        "max5",
        "--reset-hour",
        "10",
        "--timezone",
        "UTC",
        "--plain",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = monitor.parse_args()
    assert args.plan == "max5"
    assert args.reset_hour == 10
    assert args.timezone == "UTC"
    assert args.plain is True


def test_get_token_limit_plans():
    assert monitor.get_token_limit("pro") == 7000
    assert monitor.get_token_limit("max5") == 35000
    assert monitor.get_token_limit("max20") == 140000


def test_get_token_limit_custom_max():
    blocks = [
        {"isGap": False, "isActive": False, "totalTokens": 10000},
        {"isGap": True, "totalTokens": 20000},
        {"isGap": False, "isActive": True, "totalTokens": 30000},
        {"isGap": False, "isActive": False, "totalTokens": 25000},
    ]
    assert monitor.get_token_limit("custom_max", blocks) == 25000


def test_get_token_limit_custom_max_default():
    blocks = [
        {"isGap": True, "totalTokens": 5000},
        {"isActive": True, "totalTokens": 8000},
    ]
    assert monitor.get_token_limit("custom_max", blocks) == 7000
