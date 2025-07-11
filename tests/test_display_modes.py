import importlib.util
from pathlib import Path
from unittest.mock import patch
from rich.progress import Progress

spec = importlib.util.spec_from_file_location(
    "ccusage_monitor", Path(__file__).resolve().parents[1] / "ccusage_monitor.py"
)
assert spec and spec.loader
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_token_progress_bar_plain_and_rich():
    with patch.object(monitor, "RICH_AVAILABLE", False):
        result = monitor.create_token_progress_bar(50)
        assert isinstance(result, str)
        result_plain = monitor.create_token_progress_bar(50, plain=True)
        assert isinstance(result_plain, str)

    with patch.object(monitor, "RICH_AVAILABLE", True):
        result_rich = monitor.create_token_progress_bar(50)
        assert isinstance(result_rich, Progress)
        result_plain = monitor.create_token_progress_bar(50, plain=True)
        assert isinstance(result_plain, str)


def test_time_progress_bar_plain_and_rich():
    with patch.object(monitor, "RICH_AVAILABLE", False):
        result = monitor.create_time_progress_bar(5, 10)
        assert isinstance(result, str)
        result_plain = monitor.create_time_progress_bar(5, 10, plain=True)
        assert isinstance(result_plain, str)

    with patch.object(monitor, "RICH_AVAILABLE", True):
        result_rich = monitor.create_time_progress_bar(5, 10)
        assert isinstance(result_rich, Progress)
        result_plain = monitor.create_time_progress_bar(5, 10, plain=True)
        assert isinstance(result_plain, str)


def test_model_progress_bar_plain_and_rich():
    with patch.object(monitor, "RICH_AVAILABLE", False):
        usage = monitor.ModelUsageInfo(used=50, total=100)
        result = monitor.create_model_progress_bar("claude-opus-4", usage)
        assert isinstance(result, str)
        result_plain = monitor.create_model_progress_bar("claude-opus-4", usage, plain=True)
        assert isinstance(result_plain, str)

    with patch.object(monitor, "RICH_AVAILABLE", True):
        usage = monitor.ModelUsageInfo(used=50, total=100)
        result_rich = monitor.create_model_progress_bar("claude-opus-4", usage)
        assert isinstance(result_rich, Progress)
        result_plain = monitor.create_model_progress_bar("claude-opus-4", usage, plain=True)
        assert isinstance(result_plain, str)
