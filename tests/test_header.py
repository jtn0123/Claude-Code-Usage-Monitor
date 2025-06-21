import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ccusage_monitor", Path(__file__).resolve().parents[1] / "ccusage_monitor.py"
)
assert spec and spec.loader
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_print_header_shows_model(capsys):
    monitor.print_header("claude-opus-4")
    captured = capsys.readouterr().out
    assert "Active Model: Opus" in captured

