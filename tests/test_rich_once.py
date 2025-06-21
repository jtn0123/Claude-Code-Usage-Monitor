import importlib.util
from pathlib import Path
from argparse import Namespace
from rich.console import Console

spec = importlib.util.spec_from_file_location(
    "ccusage_monitor", Path(__file__).resolve().parents[1] / "ccusage_monitor.py"
)
assert spec and spec.loader
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def run_once_helper(data, session_info):
    console = Console(record=True)
    args = Namespace(plan="pro", reset_hour=None, timezone="UTC", plain=False)
    state = monitor.MonitorState(token_limit=7000)
    monitor.run_rich_once(args, data, session_info, console=console, state=state)
    return console.export_text()


def test_rich_once_progress_bars():
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
                "modelBreakdowns": [{"model": "claude-opus-4", "totalTokens": 1000}],
            }
        ]
    }
    output = run_once_helper(data, session_info)
    progress_lines = [line for line in output.splitlines() if "━" in line]
    assert len(progress_lines) >= 2
